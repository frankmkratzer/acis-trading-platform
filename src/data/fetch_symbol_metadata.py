#!/usr/bin/env python3
# File: fetch_symbol_metadata.py
# Purpose: Build a clean US investable symbol_universe (common/ordinary/REIT only)

import os
import re
import time
import logging
from io import StringIO
from datetime import date, datetime, timezone, timedelta
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import requests
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from tqdm import tqdm

# ─── Config ─────────────────────────────────────────────────────────────
load_dotenv()
POSTGRES_URL = os.getenv("POSTGRES_URL")
if not POSTGRES_URL:
    raise RuntimeError("POSTGRES_URL is not set")
engine = create_engine(POSTGRES_URL)

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

MAX_THREADS = int(os.getenv("LISTING_THREADS", "8"))

# Global request budget (shared across all threads)
REQUESTS_PER_MIN = int(os.getenv("AV_MAX_CALLS_PER_MIN", "600"))
REQUEST_DELAY = 60.0 / max(1, REQUESTS_PER_MIN)  # spacing between calls globally

logging.basicConfig(
    filename="fetch_symbol_metadata.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("fetch_symbol_metadata")

# ─── Shared Rate Limit Lock (global scheduler) ──────────────────────────
rate_limit_lock = Lock()
last_request_time = [0.0]  # next-available time marker (epoch seconds)

# ─── CLI ────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, help="Limit number of symbols to fetch/enrich")
    return p.parse_args()

# ─── Helper: Pre-ingest US-only, equity-only filters ────────────────────
EXCH_WHITELIST = {"NASDAQ", "NYSE", "AMEX", "NYSE AMERICAN", "NYSE American"}

# Ticker allowlist & blocklist patterns
TICKER_ALLOW = re.compile(r"^[A-Z]{1,5}(?:\.[A-Z])?$")  # allow BRK.B-style if desired
TICKER_BLOCK = re.compile(r"(?:-P-[A-Z]+$|[./-](?:WS|W|WT|Warrant|U|UN|R|RT)$|\d$)")
NAME_BLOCK   = re.compile(
    r"(?:ETF|ETN|ETP|Fund|Trust(?!.*REIT)|Closed[- ]End|Notes|Unit|Right|Warrant|SPAC|Depositary|ADR)",
    re.I,
)

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]
    return df

def _pre_filter_listing(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    df = df[df["assettype"].fillna("").str.lower() == "stock"]  # null-safe
    df = df[df["exchange"].isin(EXCH_WHITELIST)]
    # Basic symbol sanity
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df = df[df["symbol"].str.match(TICKER_ALLOW, na=False)]
    df = df[~df["symbol"].str.contains(TICKER_BLOCK, na=False)]
    # Name-based junk
    df["name"] = df["name"].astype(str).str.strip()
    df = df[~df["name"].str.contains(NAME_BLOCK, na=False)]
    n1 = len(df)
    logger.info("Pre-filter: kept %d / %d symbols after US/equity/junk screens", n1, n0)
    return df

# ─── Download symbol list ───────────────────────────────────────────────
def download_symbol_list(max_retries=5, backoff_factor=1.5) -> pd.DataFrame:
    print("📥 Downloading symbols from Alpha Vantage...")
    if not ALPHA_VANTAGE_API_KEY:
        raise RuntimeError("ALPHA_VANTAGE_API_KEY is not set")
    url = f"https://www.alphavantage.co/query?function=LISTING_STATUS&apikey={ALPHA_VANTAGE_API_KEY}"

    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            raw_text = r.text
            df = pd.read_csv(StringIO(raw_text))
            df = _normalize_columns(df)
            if "assettype" not in df.columns:
                logger.error("❌ Missing 'assettype' in LISTING_STATUS. Columns: %s", df.columns.tolist())
                logger.error("Head:\n%s", raw_text[:500])
                raise ValueError("AlphaVantage LISTING_STATUS missing 'assettype'")
            break
        except Exception as e:
            wait = backoff_factor * (2 ** attempt)
            logger.warning("Retry %d/%d LISTING_STATUS failed: %s (sleep %.1fs)", attempt + 1, max_retries, e, wait)
            time.sleep(wait)
    else:
        raise RuntimeError("Failed to fetch LISTING_STATUS after retries.")

    # Required cols
    for col in ["symbol", "name", "exchange"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # IPO date sanity (drop future listings)
    if "ipodate" in df.columns:
        df["ipodate"] = pd.to_datetime(df["ipodate"], errors="coerce")
        df = df[df["ipodate"].isna() | (df["ipodate"] <= pd.Timestamp.now())]

    # Pre-ingest US/equity filter
    df = _pre_filter_listing(df)

    # Map to our schema columns we can fill now
    df = df.rename(columns={
        "exchange": "exchange",
        "assettype": "security_type",
        "ipodate": "listed_date",
    })

    # 🔧 Normalize AV 'Stock' to satisfy DB constraint (Common% | Ordinary% | REIT%)
    df["security_type"] = df["security_type"].astype(str).str.strip()
    df.loc[df["security_type"].str.lower() == "stock", "security_type"] = "Common Stock"

    # Values we’ll fill via OVERVIEW later (currency/country/sector/industry/market_cap/…)
    add = {
        "is_etf": False,
        "sector": None,
        "industry": None,
        "market_cap": None,
        "currency": None,
        "country": None,
        "dividend_yield": None,
        "pe_ratio": None,
        "peg_ratio": None,
        "week_52_high": None,
        "week_52_low": None,
        "fetched_at": datetime.now(timezone.utc),
    }
    for k, v in add.items():
        if k not in df.columns:
            df[k] = v

    cols = [
        "symbol", "name", "exchange", "security_type",
        "is_etf", "listed_date", "sector", "industry", "market_cap", "currency",
        "country", "dividend_yield", "pe_ratio", "peg_ratio",
        "week_52_high", "week_52_low", "fetched_at"
    ]
    df = df[cols].dropna(subset=["symbol", "name", "exchange"]).drop_duplicates(subset=["symbol"])
    return df


# ─── Enrichment (Alpha Vantage OVERVIEW) ────────────────────────────────
def enrich_metadata(symbol: str, max_retries=3) -> dict:
    url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
    for attempt in range(max_retries):
        try:
            # Global scheduler: compute a unique next_time under the lock,
            # then sleep outside the lock so other threads can schedule too.
            with rate_limit_lock:
                now = time.time()
                next_time = max(last_request_time[0] + REQUEST_DELAY, now)
                last_request_time[0] = next_time
            delay = max(0.0, next_time - time.time())
            if delay > 0:
                time.sleep(delay)

            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                raise ValueError(f"HTTP {resp.status_code}")
            data = resp.json()
            if not data or "Symbol" not in data:
                raise ValueError("Empty or invalid JSON")
            return {
                "symbol": symbol,
                "sector": data.get("Sector"),
                "industry": data.get("Industry"),
                "market_cap": pd.to_numeric(data.get("MarketCapitalization"), errors="coerce"),
                "currency": (data.get("Currency") or "").upper() or None,
                "country": data.get("Country"),
                "dividend_yield": pd.to_numeric(data.get("DividendYield"), errors="coerce"),
                "pe_ratio": pd.to_numeric(data.get("PERatio"), errors="coerce"),
                "peg_ratio": pd.to_numeric(data.get("PEGRatio"), errors="coerce"),
                "week_52_high": pd.to_numeric(data.get("52WeekHigh"), errors="coerce"),
                "week_52_low": pd.to_numeric(data.get("52WeekLow"), errors="coerce"),
            }
        except Exception as e:
            logger.warning("[Retry %d] OVERVIEW %s failed: %s", attempt + 1, symbol, e)
            time.sleep(min(2 ** attempt, 8))
    logger.error("❌ Giving up on OVERVIEW for %s", symbol)
    return {}

# ─── Pruning helpers ────────────────────────────────────────────────────
def prune_symbol_universe(engine, min_ipo_days: int = 120, aggressive: bool = False) -> int:
    """
    Remove symbols that are unlikely to be usable for fundamentals-based strategies.

    min_ipo_days: Don’t prune symbols listed more recently than this many days.
    aggressive:
        - False: “light-touch” — drop symbols missing OVERVIEW (sector/industry/market_cap).
        - True : “authoritative” — drop symbols with NO rows in both fundamentals_annual & fundamentals_quarterly.
    """
    cutoff = date.today() - timedelta(days=min_ipo_days)

    if aggressive:
        # Faster & index-friendly: NOT EXISTS instead of LEFT JOIN + GROUP BY
        sql = text("""
            WITH del AS (
                SELECT u.symbol
                FROM symbol_universe u
                WHERE (u.listed_date IS NULL OR u.listed_date <= :cutoff)
                  AND NOT EXISTS (SELECT 1 FROM fundamentals_annual    a WHERE a.symbol = u.symbol)
                  AND NOT EXISTS (SELECT 1 FROM fundamentals_quarterly q WHERE q.symbol = u.symbol)
            )
            DELETE FROM symbol_universe u
            USING del d
            WHERE u.symbol = d.symbol
            RETURNING u.symbol;
        """)
        mode = "authoritative (no annual+quarterly fundamentals)"
    else:
        # Light-touch: no OVERVIEW enrichment proxy and not a fresh IPO
        sql = text("""
            WITH del AS (
                SELECT u.symbol
                FROM symbol_universe u
                WHERE (u.listed_date IS NULL OR u.listed_date <= :cutoff)
                  AND COALESCE(NULLIF(TRIM(u.sector), ''),   NULL) IS NULL
                  AND COALESCE(NULLIF(TRIM(u.industry), ''), NULL) IS NULL
                  AND u.market_cap IS NULL
            )
            DELETE FROM symbol_universe u
            USING del d
            WHERE u.symbol = d.symbol
            RETURNING u.symbol;
        """)
        mode = "light-touch (no OVERVIEW fields)"

    with engine.begin() as conn:
        rows = conn.execute(sql, {"cutoff": cutoff}).fetchall()
    deleted = len(rows)
    logger.info("🧹 Pruned %d symbols from symbol_universe (%s, IPO cutoff=%s).",
                deleted, mode, cutoff.isoformat())
    if deleted:
        print(f"🧹 Pruned {deleted} symbols ({mode}).")
    return deleted

def has_annual_fundamentals(engine) -> bool:
    """Return True if fundamentals_annual contains at least one row."""
    sql = text("SELECT EXISTS (SELECT 1 FROM fundamentals_annual LIMIT 1)")
    try:
        with engine.begin() as conn:
            return bool(conn.execute(sql).scalar())
    except Exception as e:
        logger.warning("Could not check fundamentals_annual presence; defaulting to light-touch. Error: %s", e)
        return False

# ─── Main ───────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    base_df = download_symbol_list()
    if args.limit:
        base_df = base_df.head(args.limit)
    print(f"📊 Candidates after pre-filter: {len(base_df):,}")

    # Enrich
    enriched_rows = []
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as ex:
        futs = {ex.submit(enrich_metadata, row.symbol): row.symbol for _, row in base_df.iterrows()}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Enriching symbols"):
            sym = futs[fut]
            try:
                d = fut.result()
                if d:
                    enriched_rows.append(d)
            except Exception as e:
                logger.exception("Unhandled enrich error for %s: %s", sym, e)

    enriched_df = pd.DataFrame(enriched_rows)
    if not enriched_df.empty:
        enriched_df["symbol"] = enriched_df["symbol"].astype(str).str.upper().str.strip()

    # Merge: prefer enriched fields where available
    base_df = base_df.copy()
    base_df["symbol"] = base_df["symbol"].astype(str).str.upper().str.strip()
    base_df = base_df.set_index("symbol")
    merged = base_df

    if not enriched_df.empty:
        enriched_df = enriched_df.set_index("symbol")
        # update() fills non-NA values from enriched into base_df aligned on index
        merged.update(enriched_df)

    merged = merged.reset_index()

    # ── Post-enrichment strict US filters ────────────────────────────────
    n_before_post = len(merged)

    # Keep only USD + US
    merged["currency"] = merged["currency"].astype(str).str.upper()
    merged["country"] = merged["country"].astype(str).str.strip()

    is_us_country = merged["country"].str.contains(
        r"^(United States|USA|United States of America)$",
        case=False, regex=True, na=False
    )
    is_usd = merged["currency"].eq("USD")

    # Keep common/ordinary/REIT if we have a type string (fallback to keeping if unknown)
    sec = merged["security_type"].astype(str)
    keep_sectype = (
        sec.str.contains(r"^(Stock|Common|Ordinary|REIT)", case=False, regex=True, na=False) |
        sec.isna() | sec.eq("")
    )

    # Aggressive name/ticker blocks after enrichment (to squash ADRs & odd classes)
    allow_mask = merged["symbol"].str.fullmatch(TICKER_ALLOW, na=False)
    block_mask = merged["symbol"].str.contains(TICKER_BLOCK, na=False)
    keep_symbol = allow_mask & ~block_mask
    keep_name = ~merged["name"].str.contains(NAME_BLOCK, na=False)

    final_mask = is_us_country & is_usd & keep_sectype & keep_symbol & keep_name
    merged = merged.loc[final_mask].drop_duplicates(subset=["symbol"]).reset_index(drop=True)

    n_after_post = len(merged)
    logger.info(
        "Post-filter: kept %d / %d after USD/US + type + name/ticker guards",
        n_after_post, n_before_post
    )
    print(f"✅ Final investable universe: {n_after_post:,} symbols")

    # ── Upsert into symbol_universe ──────────────────────────────────────
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS tmp_symbol_universe"))
        conn.execute(text("CREATE TEMP TABLE tmp_symbol_universe AS SELECT * FROM symbol_universe WITH NO DATA;"))
        merged.to_sql("tmp_symbol_universe", conn, if_exists="append", index=False, method="multi")

        conn.execute(text("""
            INSERT INTO symbol_universe AS t (
                symbol, name, exchange, security_type, is_etf, listed_date,
                sector, industry, market_cap, currency, country,
                dividend_yield, pe_ratio, peg_ratio, week_52_high, week_52_low, fetched_at
            )
            SELECT s.symbol, s.name, s.exchange, s.security_type, s.is_etf, s.listed_date,
                   s.sector, s.industry, s.market_cap, s.currency, s.country,
                   s.dividend_yield, s.pe_ratio, s.peg_ratio, s.week_52_high, s.week_52_low, s.fetched_at
            FROM tmp_symbol_universe s
            ON CONFLICT (symbol) DO UPDATE SET
                name = EXCLUDED.name,
                exchange = EXCLUDED.exchange,
                security_type = EXCLUDED.security_type,
                is_etf = EXCLUDED.is_etf,
                listed_date = EXCLUDED.listed_date,
                sector = EXCLUDED.sector,
                industry = EXCLUDED.industry,
                market_cap = EXCLUDED.market_cap,
                currency = EXCLUDED.currency,
                country = EXCLUDED.country,
                dividend_yield = EXCLUDED.dividend_yield,
                pe_ratio = EXCLUDED.pe_ratio,
                peg_ratio = EXCLUDED.peg_ratio,
                week_52_high = EXCLUDED.week_52_high,
                week_52_low = EXCLUDED.week_52_low,
                fetched_at = EXCLUDED.fetched_at;
        """))

    logger.info("✅ Upserted %d symbols into symbol_universe", len(merged))
    print("🏁 symbol_universe updated.")

    # ── Prune once, auto-mode based on fundamentals_annual presence ──────
    aggressive = has_annual_fundamentals(engine)
    mode_txt = "aggressive" if aggressive else "light-touch"
    print(f"🧹 Pruning universe with {mode_txt} mode (IPO cutoff=120d)…")
    prune_symbol_universe(engine, min_ipo_days=120, aggressive=aggressive)

if __name__ == "__main__":
    main()
