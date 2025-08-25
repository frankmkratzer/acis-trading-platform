#!/usr/bin/env python3
# File: fetch_sp500_history.py
# Purpose: Fetch full adjusted SPY price history into sp500_price_history table (safe upsert + schema guard)

import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from collections import deque
import logging

# ─── ENV SETUP ─────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
POSTGRES_URL = os.getenv("POSTGRES_URL")
engine = create_engine(POSTGRES_URL)

# ─── Constants ─────────────────────────────────────────────
AV_URL = "https://www.alphavantage.co/query"
SPY_SYMBOL = "SPY"
MAX_CALLS_PER_MIN = 600
RATE_WINDOW_SEC = 60
RETRY_LIMIT = 3

# ─── Logger ────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fetch_sp500_history")

# ─── Rate Limiter ──────────────────────────────────────────
request_times = deque(maxlen=MAX_CALLS_PER_MIN)

def rate_limited_request(url, params):
    now = time.time()
    while len(request_times) >= MAX_CALLS_PER_MIN and (now - request_times[0]) < RATE_WINDOW_SEC:
        sleep_time = RATE_WINDOW_SEC - (now - request_times[0]) + 0.1
        logger.info(f"⏳ Rate limit reached. Sleeping {sleep_time:.2f}s...")
        time.sleep(sleep_time)
        now = time.time()
    request_times.append(now)
    return requests.get(url, params=params, timeout=30)


# ─── Freshness Check ───────────────────────────────────────
def is_data_fresh():
    query = text("SELECT MAX(trade_date)::date FROM sp500_price_history")
    with engine.connect() as conn:
        latest = conn.execute(query).scalar()
    if latest is None:
        return False
    return latest >= datetime.now(timezone.utc).date()

# ─── Fetch SPY Data ────────────────────────────────────────
def fetch_spy_history():
    for attempt in range(RETRY_LIMIT):
        try:
            params = {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": SPY_SYMBOL,
                "apikey": API_KEY,
                "outputsize": "full",
                "datatype": "json"
            }
            resp = rate_limited_request(AV_URL, params)
            if resp.status_code != 200:
                raise Exception(f"HTTP {resp.status_code}")

            raw = resp.json()
            if "Time Series (Daily)" not in raw:
                raise ValueError("Missing 'Time Series (Daily)' in response")

            ts = raw["Time Series (Daily)"]
            records = []
            for date_str, values in ts.items():
                records.append({
                    "trade_date": pd.to_datetime(date_str).date(),  # ensure DATE, not TIMESTAMP
                    "open": float(values.get("1. open", 0)),
                    "high": float(values.get("2. high", 0)),
                    "low": float(values.get("3. low", 0)),
                    "close": float(values.get("4. close", 0)),
                    "adjusted_close": float(values.get("5. adjusted close", 0)),
                    "volume": int(values.get("6. volume", 0)),
                    "dividend_amount": float(values.get("7. dividend amount", 0)),
                    "split_coefficient": float(values.get("8. split coefficient", 1)),
                    "fetched_at": datetime.now(timezone.utc)
                })
            df = pd.DataFrame(records).sort_values("trade_date")
            return df
        except Exception as e:
            logger.warning(f"⚠️ Attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError("❌ Failed to fetch SPY data after retries.")

# ─── Upsert into typed table ───────────────────────────────
def upsert_spy(df: pd.DataFrame):
    if df.empty:
        return
    with engine.begin() as conn:
        temp_table = "temp_sp500_price_history"
        df.to_sql(temp_table, conn, if_exists="replace", index=False, method="multi")

        # Rely on UNIQUE index ux_sp500_trade_date for conflict target
        conn.execute(text(f"""
            INSERT INTO sp500_price_history (
                trade_date, open, high, low, close, adjusted_close,
                volume, dividend_amount, split_coefficient, fetched_at
            )
            SELECT
                trade_date::date,
                open::numeric,
                high::numeric,
                low::numeric,
                close::numeric,
                adjusted_close::numeric,
                volume::bigint,
                dividend_amount::numeric,
                split_coefficient::numeric,
                fetched_at
            FROM {temp_table}
            ON CONFLICT (trade_date) DO UPDATE SET
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
        conn.execute(text(f"DROP TABLE IF EXISTS {temp_table}"))

# ─── Main ──────────────────────────────────────────────────
def main():


    if is_data_fresh():
        logger.info("✅ SPY data already up-to-date.")
        return

    logger.info("📥 Fetching SPY daily adjusted history...")
    df = fetch_spy_history()
    upsert_spy(df)
    logger.info(f"✅ Upserted {len(df)} SPY rows into sp500_price_history")

if __name__ == "__main__":
    main()
