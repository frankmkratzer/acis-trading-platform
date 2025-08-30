#!/usr/bin/env python3
"""
S&P 500 History Fetcher
Fetches SPY ETF price history as proxy for S&P 500 index
Optimized for 600 calls/min Premium API
"""

import os
import sys
import time
import requests
import pandas as pd
from datetime import datetime, timezone
from sqlalchemy import text
from dotenv import load_dotenv
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.logging_config import setup_logger
from database.db_connection_manager import DatabaseConnectionManager
from data_fetch.base.rate_limiter import AlphaVantageRateLimiter

# Configuration
load_dotenv()

# Initialize centralized utilities
logger = setup_logger("fetch_sp500_history")
db_manager = DatabaseConnectionManager()
engine = db_manager.get_engine()
rate_limiter = AlphaVantageRateLimiter.get_instance()

API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
if not API_KEY:
    logger.error("ALPHA_VANTAGE_API_KEY not set")
    sys.exit(1)

AV_URL = "https://www.alphavantage.co/query"
SPY_SYMBOL = "SPY"

def rate_limited_get(url, params, timeout=30):
    """Rate-limited GET request using centralized rate limiter"""
    rate_limiter.wait_if_needed()
    return requests.get(url, params=params, timeout=timeout)

def is_data_fresh():
    """Check if we already have today's data"""
    query = text("SELECT MAX(trade_date)::date FROM sp500_history")
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

def calculate_returns():
    """Calculate forward returns for S&P 500 data"""
    try:
        with engine.begin() as conn:
            # Simpler approach - find future prices based on approximate trading days
            conn.execute(text("""
                WITH future_prices AS (
                    SELECT 
                        h1.trade_date as base_date,
                        h1.adjusted_close as base_close,
                        -- Find price approximately 21 trading days later (1 month)
                        (SELECT h2.adjusted_close 
                         FROM sp500_history h2 
                         WHERE h2.trade_date > h1.trade_date 
                           AND h2.trade_date <= h1.trade_date + INTERVAL '35 days'
                         ORDER BY h2.trade_date 
                         OFFSET 20 LIMIT 1) as close_1m,
                        -- Find price approximately 63 trading days later (3 months)
                        (SELECT h3.adjusted_close 
                         FROM sp500_history h3 
                         WHERE h3.trade_date > h1.trade_date 
                           AND h3.trade_date <= h1.trade_date + INTERVAL '100 days'
                         ORDER BY h3.trade_date 
                         OFFSET 62 LIMIT 1) as close_3m,
                        -- Find price approximately 126 trading days later (6 months)
                        (SELECT h6.adjusted_close 
                         FROM sp500_history h6 
                         WHERE h6.trade_date > h1.trade_date 
                           AND h6.trade_date <= h1.trade_date + INTERVAL '200 days'
                         ORDER BY h6.trade_date 
                         OFFSET 125 LIMIT 1) as close_6m,
                        -- Find price approximately 252 trading days later (1 year)
                        (SELECT h12.adjusted_close 
                         FROM sp500_history h12 
                         WHERE h12.trade_date > h1.trade_date 
                           AND h12.trade_date <= h1.trade_date + INTERVAL '380 days'
                         ORDER BY h12.trade_date 
                         OFFSET 251 LIMIT 1) as close_1y
                    FROM sp500_history h1
                )
                UPDATE sp500_history
                SET 
                    return_1m = CASE 
                        WHEN fp.close_1m IS NOT NULL AND fp.base_close > 0
                        THEN (fp.close_1m - fp.base_close) / fp.base_close * 100 
                        ELSE NULL 
                    END,
                    return_3m = CASE 
                        WHEN fp.close_3m IS NOT NULL AND fp.base_close > 0
                        THEN (fp.close_3m - fp.base_close) / fp.base_close * 100 
                        ELSE NULL 
                    END,
                    return_6m = CASE 
                        WHEN fp.close_6m IS NOT NULL AND fp.base_close > 0
                        THEN (fp.close_6m - fp.base_close) / fp.base_close * 100 
                        ELSE NULL 
                    END,
                    return_1y = CASE 
                        WHEN fp.close_1y IS NOT NULL AND fp.base_close > 0
                        THEN (fp.close_1y - fp.base_close) / fp.base_close * 100 
                        ELSE NULL 
                    END
                FROM future_prices fp
                WHERE sp500_history.trade_date = fp.base_date
            """))
            
            # Get count of updated records
            result = conn.execute(text("""
                SELECT COUNT(*) FROM sp500_history 
                WHERE return_1m IS NOT NULL 
                   OR return_3m IS NOT NULL 
                   OR return_6m IS NOT NULL 
                   OR return_1y IS NOT NULL
            """))
            
            updated_count = result.scalar()
            logger.info(f"Calculated returns for {updated_count} records")
            return updated_count
            
    except Exception as e:
        logger.error(f"Error calculating returns: {e}")
        return 0

def upsert_spy_data(df: pd.DataFrame):
    """Upsert SPY data to sp500_history table"""
    if df.empty:
        return 0
    
    try:
        with engine.begin() as conn:
            # Create temp table
            temp_table = f"temp_sp500_{int(time.time())}"
            
            # The sp500_history table already uses trade_date, open, close etc.
            # No need to rename most columns, they match the table schema
            df = df.rename(columns={
                # Keep trade_date as is - it matches the table
                # Keep open, high, low, close as is - they match the table
                # adjusted_close is already in the dataframe and matches the table
            })
            
            # Keep only columns that exist in sp500_history table
            columns_to_keep = ['trade_date', 'open', 'high', 'low', 'close', 'adjusted_close', 'volume', 'fetched_at']
            df = df[[col for col in columns_to_keep if col in df.columns]]
            
            df.to_sql(temp_table, conn, if_exists="replace", index=False, method="multi")
            
            # Upsert from temp table
            conn.execute(text(f"""
                INSERT INTO sp500_history (
                    trade_date, open, high, low, close, adjusted_close, volume, fetched_at
                )
                SELECT
                    trade_date::date,
                    open, high, low, close, adjusted_close, volume, fetched_at
                FROM {temp_table}
                ON CONFLICT (trade_date) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    adjusted_close = EXCLUDED.adjusted_close,
                    volume = EXCLUDED.volume,
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
        print(f"\n[SUCCESS] Upserted {records} SPY records to sp500_history")
        
        # Calculate forward returns
        print("[INFO] Calculating forward returns (1m, 3m, 6m, 1y)...")
        returns_updated = calculate_returns()
        
        if returns_updated > 0:
            print(f"[SUCCESS] Calculated returns for {returns_updated} records")
        else:
            print("[WARNING] No returns calculated - may need more historical data")
        
        duration = time.time() - start_time
        print(f"\n[PERFORMANCE] Total duration: {duration:.1f}s")
        print(f"[PERFORMANCE] Rate: {records/duration:.0f} records/sec")
        logger.info(f"Successfully upserted {records} SPY records and calculated {returns_updated} returns in {duration:.1f}s")
    else:
        print("[ERROR] Failed to upsert SPY data")
        logger.error("Failed to upsert SPY data to database")
        sys.exit(1)

if __name__ == "__main__":
    main()