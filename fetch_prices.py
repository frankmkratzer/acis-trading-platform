#!/usr/bin/env python3
# File: fetch_prices.py
# Purpose: Fetch EOD daily adjusted prices (20y+ + incremental) with smooth global rate limiting and robust retries

import os
import time
import math
import random
import logging
import threading
import requests
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, timezone
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ─── Config ─────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
POSTGRES_URL = os.getenv("POSTGRES_URL")
engine = create_engine(POSTGRES_URL)

AV_URL = "https://www.alphavantage.co/query"

# Rate limits (use headroom below hard cap to avoid drift)
MAX_CALLS_PER_MIN = int(os.getenv("AV_MAX_CALLS_PER_MIN", "600"))
HEADROOM_PCT = float(os.getenv("AV_HEADROOM_PCT", "0.98"))      # 98% of cap by default
EFFECTIVE_LIMIT = max(1, int(math.floor(MAX_CALLS_PER_MIN * HEADROOM_PCT)))  # e.g., 588/min
TOKENS_PER_SEC = EFFECTIVE_LIMIT / 60.0
BUCKET_CAPACITY = int(os.getenv("AV_BUCKET_CAPACITY", str(max(5, EFFECTIVE_LIMIT // 6))))  # small burst allowance
MAX_WORKERS = int(os.getenv("AV_MAX_WORKERS", "4"))
RETRY_LIMIT = int(os.getenv("AV_RETRY_LIMIT", "3"))
VERBOSE_RATE = os.getenv("AV_VERBOSE_RATE", "0").lower() in ("1", "true", "yes")

# ─── Logging ────────────────────────────────────────────────────
logging.basicConfig(
    filename="fetch_prices.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("fetch_prices")

# ─── HTTP session with retries ──────────────────────────────────
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

# ─── Token Bucket Limiter ───────────────────────────────────────
class TokenBucket:
    def __init__(self, rate_per_sec: float, capacity: int):
        self.rate = rate_per_sec
        self.capacity = capacity
        self.tokens = float(capacity)
        self.last = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self, tokens: float = 1.0):
        """Block until 'tokens' are available (global & thread-safe)."""
        while True:
            with self.lock:
                now = time.monotonic()
                # refill
                elapsed = now - self.last
                if elapsed > 0:
                    self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                    self.last = now

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    if VERBOSE_RATE:
                        log.info("Token granted. tokens=%.2f", self.tokens)
                    break

                # compute wait time needed, release lock while sleeping
                need = tokens - self.tokens
                wait = need / self.rate if self.rate > 0 else 0.2
            # jitter to prevent thread herd
            jitter = random.uniform(0.005, 0.02)
            t = max(0.0, wait) + jitter
            if VERBOSE_RATE:
                log.info("Rate bucket empty, sleeping %.3fs", t)
            time.sleep(t)

rate_limiter = TokenBucket(TOKENS_PER_SEC, BUCKET_CAPACITY)

def rate_limited_get(url, params, timeout=20):
    # Acquire one token per Alpha Vantage call
    rate_limiter.acquire(1.0)
    # tiny jitter so parallel threads don't align
    time.sleep(random.uniform(0.0, 0.01))
    return session.get(url, params=params, timeout=timeout)

# ─── DB helpers ─────────────────────────────────────────────────
def get_latest_trade_date(symbol):
    query = text("SELECT MAX(trade_date) FROM stock_eod_daily WHERE symbol = :symbol")
    with engine.connect() as conn:
        return conn.execute(query, {"symbol": symbol}).scalar()

def upsert_prices(df: pd.DataFrame):
    if df.empty:
        return
    # Ensure correct dtypes early
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    with engine.begin() as conn:
        temp_table = "temp_eod_prices"
        df.to_sql(temp_table, conn, if_exists="replace", index=False, method="multi", chunksize=1000)
        conn.execute(text(f"""
            INSERT INTO stock_eod_daily (
                symbol, trade_date, open, high, low, close, adjusted_close,
                volume, dividend_amount, split_coefficient, fetched_at
            )
            SELECT
                symbol,
                trade_date::DATE,
                open,
                high,
                low,
                close,
                adjusted_close,
                volume,
                dividend_amount,
                split_coefficient,
                fetched_at
            FROM  {temp_table}
            ON CONFLICT (symbol, trade_date) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                adjusted_close = EXCLUDED.adjusted_close,
                volume = EXCLUDED.volume,
                dividend_amount = EXCLUDED.dividend_amount,
                split_coefficient = EXCLUDED.split_coefficient,
                fetched_at = EXCLUDED.fetched_at;
        """))
        conn.execute(text(f"DROP TABLE {temp_table}"))

# ─── Fetch Logic ────────────────────────────────────────────────
def fetch_price_data(symbol, outputsize="full"):
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            params = {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": symbol,
                "apikey": API_KEY,
                "outputsize": outputsize,
                "datatype": "json",
            }
            resp = rate_limited_get(AV_URL, params)
            if resp.status_code != 200:
                log.warning("HTTP %s for %s (attempt %d)", resp.status_code, symbol, attempt)
                continue

            raw = resp.json()
            # Alpha Vantage soft rate-limit or quota messages
            if isinstance(raw, dict) and any(k in raw for k in ("Note", "Information", "Error Message")):
                msg = raw.get("Note") or raw.get("Information") or raw.get("Error Message")
                log.warning("AV message for %s: %s (attempt %d)", symbol, msg, attempt)
                # backoff a bit more aggressively when server says slow down
                time.sleep(min(2.0 * attempt, 6.0))
                continue

            ts = raw.get("Time Series (Daily)")
            if not ts:
                log.info("No price data for %s (attempt %d)", symbol, attempt)
                return None

            records = []
            append = records.append
            for date_str, values in ts.items():
                append({
                    "symbol": symbol,
                    "trade_date": date_str,
                    "open": float(values.get("1. open", 0) or 0),
                    "high": float(values.get("2. high", 0) or 0),
                    "low": float(values.get("3. low", 0) or 0),
                    "close": float(values.get("4. close", 0) or 0),
                    "adjusted_close": float(values.get("5. adjusted close", 0) or 0),
                    "volume": int(values.get("6. volume", 0) or 0),
                    "dividend_amount": float(values.get("7. dividend amount", 0) or 0),
                    "split_coefficient": float(values.get("8. split coefficient", 1) or 1),
                    "fetched_at": datetime.now(timezone.utc),
                })
            return records

        except Exception as e:
            log.exception("Error fetching %s on attempt %d: %s", symbol, attempt, e)
            time.sleep(min(2 ** (attempt - 1), 8))  # exponential backoff

    return None

def process_symbol(symbol):
    latest_date = get_latest_trade_date(symbol)
    outputsize = "compact" if latest_date else "full"
    print(f"📈 Fetching {symbol} ({outputsize})")
    records = fetch_price_data(symbol, outputsize=outputsize)
    if not records:
        return
    df = pd.DataFrame(records)

    # If we already have data, trim to strictly newer dates to reduce DB IO
    if latest_date:
        df = df[pd.to_datetime(df["trade_date"]).dt.date > latest_date]

    if not df.empty:
        upsert_prices(df)
        print(f"✅ Saved {len(df)} rows for {symbol}")

def main():
    symbols = pd.read_sql("SELECT symbol FROM symbol_universe", engine)["symbol"].tolist()
    # Small random shuffle to avoid the same symbols always hitting at minute boundaries
    random.shuffle(symbols)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_symbol, s): s for s in symbols}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                fut.result()
            except Exception as e:
                log.exception("Uncaught error processing %s: %s", sym, e)
                print(f"❌ Error processing {sym}: {e}")

if __name__ == "__main__":
    main()
