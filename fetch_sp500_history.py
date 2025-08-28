#!/usr/bin/env python3
"""
S&P 500 History Fetcher
Fetches SPY ETF price history as proxy for S&P 500 index
Optimized for 600 calls/min Premium API
"""

import os
import sys
import time
import logging
import threading
import requests
import pandas as pd
from datetime import datetime, timezone
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Configuration
load_dotenv()
API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
POSTGRES_URL = os.getenv("POSTGRES_URL")

if not API_KEY:
    print("[ERROR] ALPHA_VANTAGE_API_KEY not set")
    sys.exit(1)
if not POSTGRES_URL:
    print("[ERROR] POSTGRES_URL not set")
    sys.exit(1)

engine = create_engine(POSTGRES_URL)
AV_URL = "https://www.alphavantage.co/query"
SPY_SYMBOL = "SPY"

# Rate limiting - 600 calls/min = 10 calls/sec
CALLS_PER_MIN = 580  # Slightly below 600 for safety
MIN_INTERVAL = 60.0 / CALLS_PER_MIN

# Logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/fetch_sp500_history.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("fetch_sp500_history")

# Simple rate limiter
last_call_time = 0
rate_limit_lock = threading.Lock()

def rate_limited_get(url, params, timeout=30):
    """Rate-limited GET request"""
    global last_call_time
    
    with rate_limit_lock:
        current_time = time.time()
        time_since_last = current_time - last_call_time
        if time_since_last < MIN_INTERVAL:
            sleep_time = MIN_INTERVAL - time_since_last
            time.sleep(sleep_time)
        last_call_time = time.time()
    
    return requests.get(url, params=params, timeout=timeout)

def is_data_fresh():
    """Check if we already have today's data"""
    query = text("SELECT MAX(trade_date)::date FROM sp500_price_history")
    with engine.connect() as conn:
        latest = conn.execute(query).scalar()
    if latest is None:
        return False
    # Consider data fresh if we have data from today or yesterday (markets closed)
    from datetime import timedelta
    return latest >= (datetime.now(timezone.utc).date() - timedelta(days=1))

def fetch_spy_history():
    """Fetch SPY ETF history as proxy for S&P 500"""
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": SPY_SYMBOL,
        "apikey": API_KEY,
        "outputsize": "full",
        "datatype": "json"
    }
    
    try:
        resp = rate_limited_get(AV_URL, params)
        
        if resp.status_code != 200:
            logger.warning(f"HTTP {resp.status_code} for SPY")
            return None
        
        raw = resp.json()
        
        # Check for rate limit or error messages
        if "Note" in raw:
            logger.warning(f"API rate limit note: {raw['Note']}")
            time.sleep(60)  # Wait a minute if rate limited
            return None
        
        if "Information" in raw:
            logger.warning(f"API info: {raw['Information']}")
            time.sleep(10)
            return None
            
        if "Error Message" in raw:
            logger.warning(f"API error: {raw['Error Message']}")
            return None
        
        if "Time Series (Daily)" not in raw:
            logger.error("Missing 'Time Series (Daily)' in response")
            return None
        
        ts = raw["Time Series (Daily)"]
        records = []
        for date_str, values in ts.items():
            # Use standard close if adjusted close is missing or zero
            raw_close = float(values.get("4. close", 0) or 0)
            raw_adjusted = float(values.get("5. adjusted close", 0) or 0)
            
            # Use close price if adjusted seems wrong
            if raw_adjusted == 0 or (raw_close > 0 and (raw_adjusted / raw_close < 0.5 or raw_adjusted / raw_close > 2.0)):
                adjusted_close = raw_close
            else:
                adjusted_close = raw_adjusted
            
            records.append({
                "trade_date": pd.to_datetime(date_str).date(),
                "open": float(values.get("1. open", 0) or 0),
                "high": float(values.get("2. high", 0) or 0),
                "low": float(values.get("3. low", 0) or 0),
                "close": raw_close,
                "adjusted_close": adjusted_close,
                "volume": int(values.get("6. volume", 0) or 0),
                "dividend_amount": float(values.get("7. dividend amount", 0) or 0),
                "split_coefficient": float(values.get("8. split coefficient", 1) or 1),
                "fetched_at": datetime.now(timezone.utc)
            })
        
        df = pd.DataFrame(records).sort_values("trade_date")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching SPY data: {e}")
        return None

def upsert_spy_data(df: pd.DataFrame):
    """Upsert SPY data to sp500_price_history table"""
    if df.empty:
        return 0
    
    try:
        with engine.begin() as conn:
            # Create temp table
            temp_table = f"temp_sp500_{int(time.time())}"
            df.to_sql(temp_table, conn, if_exists="replace", index=False, method="multi")
            
            # Upsert from temp table
            conn.execute(text(f"""
                INSERT INTO sp500_price_history (
                    trade_date, open, high, low, close, adjusted_close,
                    volume, dividend_amount, split_coefficient, fetched_at
                )
                SELECT
                    trade_date::date,
                    open, high, low, close, adjusted_close,
                    volume, dividend_amount, split_coefficient, fetched_at
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
                    fetched_at = EXCLUDED.fetched_at
            """))
            
            # Drop temp table
            conn.execute(text(f"DROP TABLE {temp_table}"))
            
            return len(df)
            
    except Exception as e:
        logger.error(f"Error upserting SPY data: {e}")
        return 0

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch S&P 500 (SPY) price history")
    parser.add_argument("--force", action="store_true", help="Force refresh even if data is fresh")
    args = parser.parse_args()
    
    start_time = time.time()
    print("\n" + "=" * 60)
    print("S&P 500 HISTORY FETCHER")
    print("=" * 60)
    
    # Check if data is already fresh
    if not args.force and is_data_fresh():
        print("[INFO] SPY data is already up-to-date")
        logger.info("SPY data already fresh, skipping fetch")
        return
    
    print("[INFO] Fetching SPY daily adjusted history...")
    
    # Fetch SPY data
    df = fetch_spy_history()
    
    if df is None:
        print("[ERROR] Failed to fetch SPY data")
        sys.exit(1)
    
    print(f"[INFO] Retrieved {len(df)} historical records")
    
    # Get date range
    min_date = df['trade_date'].min()
    max_date = df['trade_date'].max()
    print(f"[INFO] Date range: {min_date} to {max_date}")
    
    # Upsert to database
    records = upsert_spy_data(df)
    
    if records > 0:
        duration = time.time() - start_time
        print(f"\n[SUCCESS] Upserted {records} SPY records to sp500_price_history")
        print(f"[PERFORMANCE] Duration: {duration:.1f}s")
        print(f"[PERFORMANCE] Rate: {records/duration:.0f} records/sec")
        logger.info(f"Successfully upserted {records} SPY records in {duration:.1f}s")
    else:
        print("[ERROR] Failed to upsert SPY data")
        logger.error("Failed to upsert SPY data to database")
        sys.exit(1)

if __name__ == "__main__":
    main()