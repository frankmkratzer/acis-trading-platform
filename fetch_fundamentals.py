#!/usr/bin/env python3
# File: fetch_fundamentals.py
# Purpose: Fetch annual & quarterly fundamentals from Alpha Vantage with smooth
#          global rate limiting, adaptive backpressure, and fast batched upserts.

import os
import io
import re
import math
import time
import uuid
import random
import logging
import threading
from queue import Queue, Empty
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from reliability_manager import (
    retry_with_backoff, log_errors, validate_fundamentals_data,
    with_circuit_breaker, log_script_health, get_memory_usage
)

# ────────────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
POSTGRES_URL = os.getenv("POSTGRES_URL")
engine = create_engine(POSTGRES_URL)

AV_URL = "https://www.alphavantage.co/query"

# ULTRA-PREMIUM concurrency & rate limiting for 300 calls/min API key
MAX_WORKERS      = int(os.getenv("AV_FUND_THREADS", "12"))        # Increased for premium
MAX_CALLS_PER_MIN= int(os.getenv("AV_MAX_CALLS_PER_MIN", "290"))  # Premium 300/min minus buffer
HEADROOM_PCT     = float(os.getenv("AV_HEADROOM_PCT", "0.95"))    # Use 95% of premium capacity
EFFECTIVE_LIMIT  = max(1, int(math.floor(MAX_CALLS_PER_MIN * HEADROOM_PCT)))
TOKENS_PER_SEC   = EFFECTIVE_LIMIT / 60.0
BUCKET_CAPACITY  = int(os.getenv("AV_BUCKET_CAPACITY", "5"))      # Small burst allowance
RETRY_LIMIT      = int(os.getenv("AV_RETRY_LIMIT", "3"))
FRESHNESS_DAYS   = int(os.getenv("FUND_FRESHNESS_DAYS", "30"))
VERBOSE_RATE     = os.getenv("AV_VERBOSE_RATE", "0").lower() in ("1", "true", "yes")

# Ultra-premium: Allow more parallel requests for 300 calls/min API key
MAX_PARALLEL     = int(os.getenv("AV_MAX_PARALLEL", "8"))         # Increased for premium
REQUEST_SEM      = threading.Semaphore(MAX_PARALLEL)

# Adaptive global backpressure knobs
SOFTLIM_BUMP_SEC   = float(os.getenv("AV_SOFTLIM_BUMP_SEC", "2.0"))   # add per hit
SOFTLIM_MAX_SEC    = float(os.getenv("AV_SOFTLIM_MAX_SEC", "20.0"))   # cap added delay
SOFTLIM_DECAY_SEC  = float(os.getenv("AV_SOFTLIM_DECAY_SEC", "0.05")) # decay after clean call

# Limp mode: if we see many soft-limit hits in a short window, slow down harder
SOFTLIM_WINDOW_SEC   = float(os.getenv("AV_SOFTLIM_WINDOW_SEC", "10"))
SOFTLIM_HIT_THRESH   = int(os.getenv("AV_SOFTLIM_HIT_THRESH", "3"))
AV_LIMP_DURATION_SEC = float(os.getenv("AV_LIMP_DURATION_SEC", "45"))

RUN_ID  = datetime.now(timezone.utc).strftime("av_fund_%Y%m%dT%H%M%SZ")
SOURCE  = "AlphaVantage"

# ────────────────────────────────────────────────────────────────────────────
# Logging
# ────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename="fetch_fundamentals.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("fetch_fundamentals")

# ────────────────────────────────────────────────────────────────────────────
# HTTP session with retries
# ────────────────────────────────────────────────────────────────────────────
session = requests.Session()
retry = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
    raise_on_status=False,
)
adapter = HTTPAdapter(max_retries=retry, pool_maxsize=MAX_WORKERS * 2)
session.mount("https://", adapter)
session.mount("http://", adapter)

# ────────────────────────────────────────────────────────────────────────────
# Token Bucket Limiter
# ────────────────────────────────────────────────────────────────────────────
class TokenBucket:
    def __init__(self, rate_per_sec: float, capacity: int):
        self.rate = rate_per_sec
        self.capacity = capacity
        self.tokens = float(capacity)
        self.last = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self, tokens: float = 1.0):
        while True:
            with self.lock:
                now = time.monotonic()
                elapsed = now - self.last
                if elapsed > 0:
                    self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                    self.last = now
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    if VERBOSE_RATE:
                        logger.info("Token granted. tokens=%.2f", self.tokens)
                    break
                need = tokens - self.tokens
                wait = need / self.rate if self.rate > 0 else 0.2
            time.sleep(wait + random.uniform(0.005, 0.02))

rate_limiter = TokenBucket(TOKENS_PER_SEC, BUCKET_CAPACITY)

# ────────────────────────────────────────────────────────────────────────────
# Global adaptive backpressure state
# ────────────────────────────────────────────────────────────────────────────
_softlim_lock = threading.Lock()
_softlim_extra = 0.0     # extra delay (seconds) applied to every request
_softlim_hits  = []      # timestamps of recent soft-limit hits
_limp_until    = 0.0     # monotonic deadline for limp mode

def _decay_softlim():
    global _softlim_extra
    if _softlim_extra <= 0:
        return
    _softlim_extra = max(0.0, _softlim_extra - SOFTLIM_DECAY_SEC)

def _register_softlimit():
    """Record a soft-limit hit, increase global delay, possibly enter limp mode."""
    global _softlim_extra, _limp_until
    now = time.monotonic()
    with _softlim_lock:
        _softlim_extra = min(SOFTLIM_MAX_SEC, _softlim_extra + SOFTLIM_BUMP_SEC)
        _softlim_hits.append(now)
        # prune window
        cutoff = now - SOFTLIM_WINDOW_SEC
        while _softlim_hits and _softlim_hits[0] < cutoff:
            _softlim_hits.pop(0)
        if len(_softlim_hits) >= SOFTLIM_HIT_THRESH:
            _limp_until = max(_limp_until, now + AV_LIMP_DURATION_SEC)
            logger.warning("Entering limp mode for %.1fs (hits=%d, extra=%.2fs)",
                           AV_LIMP_DURATION_SEC, len(_softlim_hits), _softlim_extra)

def _current_extra_delay() -> float:
    """Extra delay to apply before a request."""
    now = time.monotonic()
    base = _softlim_extra
    if now < _limp_until:
        # Add a small fixed limp penalty per call
        base += 0.4
    return base

# ────────────────────────────────────────────────────────────────────────────
# Request helper with token bucket, semaphore, and backpressure
# ────────────────────────────────────────────────────────────────────────────
def rate_limited_get(url, params, timeout=20):
    # global extra delay
    extra = _current_extra_delay()
    if extra > 0:
        time.sleep(extra)

    with REQUEST_SEM:
        rate_limiter.acquire(1.0)  # one token per AV call
        time.sleep(random.uniform(0.0, 0.01))  # jitter
        resp = session.get(url, params=params, timeout=timeout)

    # decay a hair on clean HTTP 200; bump on 429
    if resp.status_code == 200:
        _decay_softlim()
    elif resp.status_code == 429:
        _register_softlimit()
    return resp

# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────
def to_num(x, as_int=False):
    """Safe numeric parse for AV strings; returns None if not parseable."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return int(x) if as_int and pd.notna(x) else float(x)
    try:
        val = pd.to_numeric(str(x).replace(",", ""), errors="coerce")
        if pd.isna(val):
            return None
        return int(val) if as_int else float(val)
    except Exception:
        return None

@retry_with_backoff(max_retries=2)
@with_circuit_breaker('alpha_vantage')
@log_errors('fetch_fundamentals')
def av_fetch(symbol, function, attempt):
    params = {"function": function, "symbol": symbol, "apikey": API_KEY}
    resp = rate_limited_get(AV_URL, params)
    if resp.status_code != 200:
        logger.warning("HTTP %s for %s/%s (attempt %d)",
                       resp.status_code, symbol, function, attempt)
        return None
    data = resp.json()
    if isinstance(data, dict) and any(k in data for k in ("Note", "Information", "Error Message")):
        msg = data.get("Note") or data.get("Information") or data.get("Error Message")
        # Treat this as soft-limit noise
        logger.warning("AV soft-limit/info for %s/%s: %s (attempt %d)",
                       symbol, function, msg, attempt)
        _register_softlimit()
        return None
    _decay_softlim()
    return data

def fetch_endpoint(symbol, function):
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            data = av_fetch(symbol, function, attempt)
            if data is None:
                # gentle backoff; global extra delay already applies
                time.sleep(min(2.0 * attempt, 6.0))
                continue
            return {
                "annual": data.get("annualReports", []) or [],
                "quarterly": data.get("quarterlyReports", []) or [],
            }
        except Exception as e:
            logger.exception("Error %s/%s attempt %d: %s", symbol, function, attempt, e)
            time.sleep(min(2 ** (attempt - 1), 8))
    return {"annual": [], "quarterly": []}

# ────────────────────────────────────────────────────────────────────────────
# Parsing
# ────────────────────────────────────────────────────────────────────────────
def parse_reports(symbol, income, balance, cash):
    """Merge AV income/balance/cash reports by fiscalDateEnding, lower-casing columns."""
    dates = {r.get("fiscalDateEnding") for r in income if r.get("fiscalDateEnding")}
    if not dates:
        dates = {r.get("fiscalDateEnding") for r in (balance or []) if r.get("fiscalDateEnding")}
        dates |= {r.get("fiscalDateEnding") for r in (cash or []) if r.get("fiscalDateEnding")}

    rows = []
    for d in sorted(dates):
        i = next((r for r in (income or []) if r.get("fiscalDateEnding") == d), {}) or {}
        b = next((r for r in (balance or []) if r.get("fiscalDateEnding") == d), {}) or {}
        c = next((r for r in (cash or []) if r.get("fiscalDateEnding") == d), {}) or {}

        eps_raw = i.get("eps", i.get("reportedEPS"))

        row = {
            "symbol": symbol,
            "fiscal_date": pd.to_datetime(d, errors="coerce"),
            "source": SOURCE,
            "run_id": RUN_ID,
            # Income
            "totalRevenue": to_num(i.get("totalRevenue"), as_int=True),
            "grossProfit": to_num(i.get("grossProfit"), as_int=True),
            "netIncome": to_num(i.get("netIncome"), as_int=True),
            "eps": to_num(eps_raw, as_int=False),
            # Balance Sheet
            "totalAssets": to_num(b.get("totalAssets"), as_int=True),
            "totalLiabilities": to_num(b.get("totalLiabilities"), as_int=True),
            "totalShareholderEquity": to_num(b.get("totalShareholderEquity"), as_int=True),
            
            # Enhanced Funnel Analysis Fields
            "cashAndCashEquivalentsAtCarryingValue": to_num(b.get("cashAndCashEquivalentsAtCarryingValue"), as_int=True),
            "cashAndShortTermInvestments": to_num(b.get("cashAndShortTermInvestments"), as_int=True),
            "shortTermInvestments": to_num(b.get("shortTermInvestments"), as_int=True),
            "commonStockSharesOutstanding": to_num(b.get("commonStockSharesOutstanding"), as_int=True),
            "currentNetReceivables": to_num(b.get("currentNetReceivables"), as_int=True),
            "inventory": to_num(b.get("inventory"), as_int=True),
            "currentAccountsPayable": to_num(b.get("currentAccountsPayable"), as_int=True),
            "longTermDebt": to_num(b.get("longTermDebt"), as_int=True),
            "shortTermDebt": to_num(b.get("shortTermDebt"), as_int=True),
            "goodwill": to_num(b.get("goodwill"), as_int=True),
            "intangibleAssets": to_num(b.get("intangibleAssets"), as_int=True),
            
            # Cash Flow
            "operatingCashflow": to_num(c.get("operatingCashflow"), as_int=True),
            "capitalExpenditures": to_num(c.get("capitalExpenditures"), as_int=True),
            "dividendPayout": to_num(c.get("dividendPayout"), as_int=True),
        }

        # Derived calculations
        ocf, capex = row["operatingCashflow"], row["capitalExpenditures"]
        shares = row["commonStockSharesOutstanding"]
        
        # Free Cash Flow
        row["free_cf"] = (ocf + capex) if (ocf is not None and capex is not None) else None
        
        # Cash Flow Per Share
        if ocf is not None and shares is not None and shares > 0:
            row["cash_flow_per_share"] = float(ocf) / float(shares)
        else:
            row["cash_flow_per_share"] = None
        row["fetched_at"] = datetime.now(timezone.utc)
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=[
            "symbol","fiscal_date","source","run_id",
            "totalrevenue","grossprofit","netincome","eps",
            "totalassets","totalliabilities","totalshareholderequity",
            "cashandcashequivalentsatcarryingvalue","cashandshorttermInvestments","shorttermInvestments",
            "commonstocksharesoutstanding","currentnetreceivables","inventory","currentaccountspayable",
            "longtermdebt","shorttermdebt","goodwill","intangibleassets",
            "operatingcashflow","capitalexpenditures","dividendpayout",
            "free_cf","cash_flow_per_share","fetched_at",
            # New columns added for compatibility with removed fundamentals table
            "pe_ratio","pb_ratio","roe","debt_to_equity","dividend_yield",
            "revenue_growth","eps_growth","current_ratio","quick_ratio",
            "working_capital","working_capital_efficiency","shares_outstanding","book_value_per_share"
        ])

    df = pd.DataFrame(rows)
    df.columns = [c.lower() for c in df.columns]
    df["fiscal_date"] = pd.to_datetime(df["fiscal_date"], errors="coerce")
    df = df.dropna(subset=["fiscal_date"]).drop_duplicates(subset=["symbol", "fiscal_date"])
    
    # Calculate derived financial ratios and metrics (new columns)
    df = calculate_derived_financials(df)
    
    return df

def calculate_derived_financials(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate derived financial ratios and metrics from raw fundamental data."""
    
    # Convert numeric columns to float for calculations
    numeric_cols = ['totalrevenue', 'netincome', 'eps', 'totalassets', 'totalliabilities', 
                   'totalshareholderequity', 'operatingcashflow', 'capitalexpenditures',
                   'dividendpayout', 'commonstocksharesoutstanding', 'currentnetreceivables',
                   'inventory', 'currentaccountspayable', 'longtermdebt', 'shorttermdebt']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calculate new financial ratios and metrics
    
    # 1. PE Ratio (Price-to-Earnings) - Need stock price, approximated from market data if available
    # For now, leave as NULL - would need current stock price
    df['pe_ratio'] = None
    
    # 2. PB Ratio (Price-to-Book) - Need stock price and book value per share
    # Calculate book value per share first
    df['book_value_per_share'] = df['totalshareholderequity'] / df['commonstocksharesoutstanding'].replace(0, None)
    df['pb_ratio'] = None  # Need stock price
    
    # 3. ROE (Return on Equity) - Net Income / Shareholders' Equity  
    df['roe'] = df['netincome'] / df['totalshareholderequity'].replace(0, None)
    
    # 4. Debt-to-Equity Ratio - Total Debt / Shareholders' Equity
    total_debt = (df['longtermdebt'].fillna(0) + df['shorttermdebt'].fillna(0))
    df['debt_to_equity'] = total_debt / df['totalshareholderequity'].replace(0, None)
    
    # 5. Dividend Yield - Need stock price, leave as NULL for now
    df['dividend_yield'] = None
    
    # 6. Revenue Growth - Need historical data for comparison
    df['revenue_growth'] = None  # Would need previous period data
    
    # 7. EPS Growth - Need historical data for comparison  
    df['eps_growth'] = None  # Would need previous period data
    
    # 8. Current Ratio - Current Assets / Current Liabilities
    # Approximate current assets and current liabilities
    current_assets = (df['cashandcashequivalentsatcarryingvalue'].fillna(0) + 
                     df['currentnetreceivables'].fillna(0) + 
                     df['inventory'].fillna(0))
    current_liabilities = df['currentaccountspayable'].fillna(0)
    df['current_ratio'] = current_assets / current_liabilities.replace(0, None)
    
    # 9. Quick Ratio - (Current Assets - Inventory) / Current Liabilities
    quick_assets = current_assets - df['inventory'].fillna(0)
    df['quick_ratio'] = quick_assets / current_liabilities.replace(0, None)
    
    # 10. Working Capital - Current Assets - Current Liabilities
    df['working_capital'] = current_assets - current_liabilities
    
    # 11. Working Capital Efficiency - Revenue / Working Capital
    df['working_capital_efficiency'] = df['totalrevenue'] / df['working_capital'].replace(0, None)
    
    # 12. Shares Outstanding - use existing commonstocksharesoutstanding
    df['shares_outstanding'] = df['commonstocksharesoutstanding']
    
    # Clean up infinite and extremely large values
    ratio_cols = ['roe', 'debt_to_equity', 'current_ratio', 'quick_ratio', 'working_capital_efficiency']
    for col in ratio_cols:
        if col in df.columns:
            df[col] = df[col].replace([float('inf'), float('-inf')], None)
            # Cap extreme values
            df[col] = df[col].clip(-1000, 1000)
    
    return df

# BIGINT columns in Postgres (must be integer-looking text in CSV)
INT_COLS = [
    "totalrevenue","grossprofit","netincome",
    "totalassets","totalliabilities","totalshareholderequity",
    "cashandcashequivalentsatcarryingvalue","cashandshorttermInvestments","shorttermInvestments",
    "commonstocksharesoutstanding","currentnetreceivables","inventory","currentaccountspayable",
    "longtermdebt","shorttermdebt","goodwill","intangibleassets",
    "operatingcashflow","capitalexpenditures","dividendpayout","free_cf",
    # New BIGINT columns
    "working_capital","shares_outstanding",
]
FLOAT_COLS = ["eps", "cash_flow_per_share", 
              # New numeric ratio columns
              "pe_ratio", "pb_ratio", "roe", "debt_to_equity", "dividend_yield",
              "revenue_growth", "eps_growth", "current_ratio", "quick_ratio", 
              "working_capital_efficiency", "book_value_per_share"]

# Columns, normalized lower-case
LOWER_COLS = [
    "symbol","fiscal_date","source","run_id",
    "totalrevenue","grossprofit","netincome","eps",
    "totalassets","totalliabilities","totalshareholderequity",
    "operatingcashflow","capitalexpenditures","dividendpayout",
    "free_cf","cash_flow_per_share","fetched_at",
    # New derived columns
    "pe_ratio","pb_ratio","roe","debt_to_equity","dividend_yield",
    "revenue_growth","eps_growth","current_ratio","quick_ratio",
    "working_capital","working_capital_efficiency","shares_outstanding","book_value_per_share"
]

# ────────────────────────────────────────────────────────────────────────────
# Batched upsert via COPY into temp → upsert into target
# ────────────────────────────────────────────────────────────────────────────
class BatchedWriter:
    def __init__(self, table: str, flush_rows: int = 80_000, flush_secs: float = 20.0):
        self.table = table
        self.flush_rows = flush_rows
        self.flush_secs = flush_secs
        self.q = Queue()
        self.last_flush = time.time()
        self._stop = False

    def offer(self, df: pd.DataFrame):
        if df is None or df.empty:
            return
        d = df.copy()
        d.columns = [c.lower() for c in d.columns]
        for c in LOWER_COLS:
            if c not in d.columns:
                d[c] = None
        d = d[LOWER_COLS]

        # Format date/timestamp to CSV-friendly strings
        d["fiscal_date"] = pd.to_datetime(d["fiscal_date"], errors="coerce").dt.date.astype(str)
        d["fetched_at"] = pd.to_datetime(d["fetched_at"], errors="coerce").astype(str)

        # BIGINTs: ensure clean integer text; NaN → ""
        for c in INT_COLS:
            if c in d.columns:
                s = pd.to_numeric(d[c], errors="coerce")
                d[c] = s.apply(lambda v: "" if pd.isna(v) else str(int(v)))

        self.q.put(d)

    def stop(self):
        self._stop = True

    def _flush_once(self, frames):
        frames = [f for f in frames if f is not None and not f.empty]
        if not frames:
            return
        df = pd.concat(frames, ignore_index=True)
        if df.empty:
            return

        tmp = f"tmp_{self.table}_{uuid.uuid4().hex[:8]}"
        cols = ",".join(LOWER_COLS)

        raw = engine.raw_connection()  # psycopg2 connection
        try:
            cur = raw.cursor()
            cur.execute(f"CREATE TEMP TABLE {tmp} (LIKE {self.table} INCLUDING DEFAULTS) ON COMMIT DROP;")
            buf = io.StringIO()
            df.to_csv(buf, index=False, header=False)
            buf.seek(0)
            cur.copy_expert(f"COPY {tmp} ({cols}) FROM STDIN WITH (FORMAT CSV, NULL '')", buf)

            cur.execute(f"""
                INSERT INTO {self.table} ({cols})
                SELECT {cols} FROM {tmp}
                ON CONFLICT (symbol, fiscal_date) DO UPDATE SET
                    source                 = EXCLUDED.source,
                    run_id                 = EXCLUDED.run_id,
                    totalrevenue           = EXCLUDED.totalrevenue,
                    grossprofit            = EXCLUDED.grossprofit,
                    netincome              = EXCLUDED.netincome,
                    eps                    = EXCLUDED.eps,
                    totalassets            = EXCLUDED.totalassets,
                    totalliabilities       = EXCLUDED.totalliabilities,
                    totalshareholderequity = EXCLUDED.totalshareholderequity,
                    operatingcashflow      = EXCLUDED.operatingcashflow,
                    capitalexpenditures    = EXCLUDED.capitalexpenditures,
                    dividendpayout         = EXCLUDED.dividendpayout,
                    free_cf                = EXCLUDED.free_cf,
                    cash_flow_per_share    = EXCLUDED.cash_flow_per_share,
                    fetched_at             = EXCLUDED.fetched_at,
                    -- New derived columns
                    pe_ratio               = EXCLUDED.pe_ratio,
                    pb_ratio               = EXCLUDED.pb_ratio,
                    roe                    = EXCLUDED.roe,
                    debt_to_equity         = EXCLUDED.debt_to_equity,
                    dividend_yield         = EXCLUDED.dividend_yield,
                    revenue_growth         = EXCLUDED.revenue_growth,
                    eps_growth             = EXCLUDED.eps_growth,
                    current_ratio          = EXCLUDED.current_ratio,
                    quick_ratio            = EXCLUDED.quick_ratio,
                    working_capital        = EXCLUDED.working_capital,
                    working_capital_efficiency = EXCLUDED.working_capital_efficiency,
                    shares_outstanding     = EXCLUDED.shares_outstanding,
                    book_value_per_share   = EXCLUDED.book_value_per_share;
            """)
            raw.commit()
            logger.info("📦 Flushed %d row(s) into %s", len(df), self.table)
        finally:
            try:
                raw.close()
            except Exception:
                pass

    def run(self):
        frames = []
        while not self._stop or not self.q.empty():
            try:
                d = self.q.get(timeout=0.5)
                frames.append(d)
            except Empty:
                pass

            need_time_flush = (time.time() - self.last_flush) >= self.flush_secs
            need_size_flush = sum(len(f) for f in frames) >= self.flush_rows
            if frames and (need_time_flush or need_size_flush):
                self._flush_once(frames)
                frames.clear()
                self.last_flush = time.time()
        if frames:
            self._flush_once(frames)

# ────────────────────────────────────────────────────────────────────────────
# Symbol selection (only US commons/ADR/REIT, skip ETFs, regex guard)
# ────────────────────────────────────────────────────────────────────────────
# Ticker regex (allow BRK.B style; block preferreds/warrants/units/rights/trailing digits)
TICKER_ALLOW = re.compile(r"^[A-Z]{1,5}(?:\.[A-Z])?$")
TICKER_BLOCK = re.compile(r"(?:-P-[A-Z]+$|[./-](?:WS|W|WT|Warrant|U|UN|R|RT)$|\d$)")

def symbols_to_update():
    """
    Prioritize symbols that plausibly have new fundamentals:
      - quarterly fiscal_date older than ~70d (or never) and last fetch > 7d
      - or annual fiscal_date older than last year-end and last fetch > 30d
      - filter: exclude ETFs; include Common/ADR/REIT only; regex guard on tickers
    """
    sql = text("""
        WITH last_q AS (
            SELECT symbol, MAX(fiscal_date) AS q_fiscal, MAX(fetched_at) AS q_fetch
            FROM fundamentals_quarterly GROUP BY symbol
        ),
        last_a AS (
            SELECT symbol, MAX(fiscal_date) AS a_fiscal, MAX(fetched_at) AS a_fetch
            FROM fundamentals_annual GROUP BY symbol
        )
        SELECT u.symbol, q.q_fiscal, q.q_fetch, a.a_fiscal, a.a_fetch
        FROM symbol_universe u
        LEFT JOIN last_q q ON u.symbol = q.symbol
        LEFT JOIN last_a a ON u.symbol = a.symbol
        WHERE COALESCE(u.is_etf, false) = false
          AND (
                u.security_type ILIKE 'Common%%' OR
                u.security_type ILIKE 'ADR%%'    OR
                u.security_type ILIKE 'REIT%%'
              )
    """)
    df = pd.read_sql(sql, engine)

    # Regex guard (skip odd tickers)
    sym = df["symbol"].astype(str).str.upper().str.strip()
    keep = sym.str.match(TICKER_ALLOW, na=False) & ~sym.str.contains(TICKER_BLOCK, na=False)
    df = df.loc[keep].copy()

    # Normalize EVERYTHING to tz-aware UTC
    q_fiscal_ts = pd.to_datetime(df["q_fiscal"], errors="coerce", utc=True)
    q_fetch_ts  = pd.to_datetime(df["q_fetch"],  errors="coerce", utc=True)
    a_fiscal_ts = pd.to_datetime(df["a_fiscal"], errors="coerce", utc=True)
    a_fetch_ts  = pd.to_datetime(df["a_fetch"],  errors="coerce", utc=True)

    # UTC anchors at midnight
    today_utc = datetime.now(timezone.utc).date()
    today_ts = pd.Timestamp(today_utc, tz="UTC")
    last_year_end_ts = pd.Timestamp(year=today_ts.year - 1, month=12, day=31, tz="UTC")

    # Age in days (NaT → huge)
    q_age_days  = (today_ts - q_fiscal_ts).dt.days.fillna(10**6)
    q_fetch_age = (today_ts - q_fetch_ts).dt.days.fillna(10**6)
    a_fetch_age = (today_ts - a_fetch_ts).dt.days.fillna(10**6)

    # Update logic
    need_q = q_fiscal_ts.isna() | ((q_age_days > 70) & (q_fetch_age > 7))
    need_a = a_fiscal_ts.isna() | ((a_fiscal_ts < last_year_end_ts) & (a_fetch_age > 30))

    syms = df.loc[need_q | need_a, "symbol"].dropna().astype(str).str.upper().str.strip().tolist()

    # Optional: prefer bigger caps first
    try:
        mkt = pd.read_sql("SELECT symbol, market_cap FROM mv_symbol_with_metadata", engine)
        order = {s: i for i, s in enumerate(mkt.sort_values("market_cap", ascending=False)["symbol"])}
        syms.sort(key=lambda s: order.get(s, 10**9))
    except Exception:
        pass
    return syms

# ────────────────────────────────────────────────────────────────────────────
# Worker
# ────────────────────────────────────────────────────────────────────────────
def process_symbol(symbol: str, writers: dict[str, BatchedWriter]):
    logger.info("🔍 %s", symbol)

    inc = fetch_endpoint(symbol, "INCOME_STATEMENT")
    bal = fetch_endpoint(symbol, "BALANCE_SHEET")
    cas = fetch_endpoint(symbol, "CASH_FLOW")

    df_a = parse_reports(symbol, inc["annual"],    bal["annual"],    cas["annual"])
    df_q = parse_reports(symbol, inc["quarterly"], bal["quarterly"], cas["quarterly"])

    if not df_a.empty:
        writers["fundamentals_annual"].offer(df_a)
    if not df_q.empty:
        writers["fundamentals_quarterly"].offer(df_q)
    if df_a.empty and df_q.empty:
        logger.info("⚠️ %s: no fundamentals returned", symbol)

# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────
def main():
    if not API_KEY:
        raise RuntimeError("ALPHA_VANTAGE_API_KEY is not set")
    syms = symbols_to_update()
    random.shuffle(syms)
    logger.info("📥 %d symbols queued for fundamentals update (run_id=%s)", len(syms), RUN_ID)

    if not syms:
        print("✅ Fundamentals are fresh; nothing to do.")
        return

    # Start writers
    writer_a = BatchedWriter(
        "fundamentals_annual",
        flush_rows=int(os.getenv("FUND_FLUSH_ROWS", "80000")),
        flush_secs=float(os.getenv("FUND_FLUSH_SECS", "20")),
    )
    writer_q = BatchedWriter(
        "fundamentals_quarterly",
        flush_rows=int(os.getenv("FUND_FLUSH_ROWS", "80000")),
        flush_secs=float(os.getenv("FUND_FLUSH_SECS", "20")),
    )

    wt_a = threading.Thread(target=writer_a.run, daemon=True)
    wt_q = threading.Thread(target=writer_q.run, daemon=True)
    wt_a.start(); wt_q.start()

    writers = {
        "fundamentals_annual": writer_a,
        "fundamentals_quarterly": writer_q,
    }

    # Progress bar + logging
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex, \
         tqdm(total=len(syms), desc="📊 Fetching fundamentals", unit="symbol") as pbar:

        futures = {ex.submit(process_symbol, s, writers): s for s in syms}
        for fut in as_completed(futures):
            s = futures[fut]
            try:
                fut.result()
            except Exception as e:
                logger.exception("❌ Uncaught error for %s: %s", s, e)
            finally:
                pbar.update(1)
                logger.info("✅ Finished %s", s)

    # Stop writers and join
    writer_a.stop(); writer_q.stop()
    wt_a.join(); wt_q.join()

    logger.info("🎉 Fundamentals fetch complete for %d symbols.", len(syms))
    print("🎉 Fundamentals fetch complete.")

if __name__ == "__main__":
    main()





