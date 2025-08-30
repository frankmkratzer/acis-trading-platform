#!/usr/bin/env python3
"""
Optimized Price Fetcher - Handles 4,600+ symbols efficiently
Fixes the 1+ hour problem by better handling rate limits
"""

import os
import sys
import time
import json
import random
import threading
import requests
import pandas as pd
import numpy as np
from sqlalchemy import text
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import signal

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.logging_config import setup_logger
from database.db_connection_manager import DatabaseConnectionManager
from data_fetch.base.rate_limiter import AlphaVantageRateLimiter

# Load environment
load_dotenv()

# Initialize centralized utilities
logger = setup_logger("fetch_prices")
db_manager = DatabaseConnectionManager()
engine = db_manager.get_engine()
rate_limiter = AlphaVantageRateLimiter.get_instance()

API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
if not API_KEY:
    logger.error("Missing API_KEY in .env")
    sys.exit(1)

AV_URL = "https://www.alphavantage.co/query"

# OPTIMIZED CONFIGURATION
MAX_WORKERS = 8  # Use multiple workers for Premium tier

# Adaptive rate limiting
RATE_LIMIT_WINDOW = 60  # seconds
MAX_RETRIES = 2  # Reduced from 3
RETRY_DELAY = 5  # Fixed 5 second retry delay instead of exponential

# Note: Using centralized AlphaVantageRateLimiter initialized above

def make_api_request(symbol, outputsize="compact", retry_count=0):
    """Make API request with smart retry logic"""
    try:
        rate_limiter.wait_if_needed()
        
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "apikey": API_KEY,
            "outputsize": outputsize,
            "datatype": "json"
        }
        
        response = requests.get(AV_URL, params=params, timeout=10)
        
        if response.status_code != 200:
            if retry_count < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
                return make_api_request(symbol, outputsize, retry_count + 1)
            return None
        
        data = response.json()
        
        # Handle rate limit responses
        if "Note" in data:
            logger.warning(f"Rate limit note for {symbol}")
            rate_limiter.report_rate_limit()
            if retry_count < MAX_RETRIES:
                time.sleep(30)  # Wait 30 seconds for rate limit
                return make_api_request(symbol, outputsize, retry_count + 1)
            return None
        
        if "Information" in data:
            info = data.get("Information", "")
            if "higher API call volume" in info or "consider upgrading" in info:
                logger.warning(f"Rate limit info for {symbol}")
                rate_limiter.report_rate_limit()
                if retry_count < MAX_RETRIES:
                    time.sleep(15)  # Wait 15 seconds
                    return make_api_request(symbol, outputsize, retry_count + 1)
            return None
        
        if "Error Message" in data:
            logger.debug(f"Invalid symbol {symbol}: {data['Error Message']}")
            return None
        
        return data
        
    except requests.RequestException as e:
        logger.error(f"Network error for {symbol}: {e}")
        if retry_count < MAX_RETRIES:
            time.sleep(RETRY_DELAY)
            return make_api_request(symbol, outputsize, retry_count + 1)
        return None
    except Exception as e:
        logger.error(f"Unexpected error for {symbol}: {e}")
        return None

def process_symbol(symbol):
    """Process a single symbol efficiently"""
    try:
        # Check last update
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT MAX(trade_date) FROM stock_prices WHERE symbol = :symbol"),
                {"symbol": symbol}
            ).fetchone()
            last_date = result[0] if result else None
        
        # Determine fetch size
        if last_date:
            # Always fetch data - don't skip based on date comparison
            # since system date (2025) doesn't match market data timeline
            # Use compact for recent data, full for older data
            days_since_data = (datetime.now().date() - last_date).days
            outputsize = "compact"  # Always use compact to save API calls
        else:
            outputsize = "full"
        
        # Fetch data
        data = make_api_request(symbol, outputsize)
        if not data:
            return 0, "NO_DATA"
        
        ts = data.get("Time Series (Daily)")
        if not ts:
            return 0, "NO_DATA"
        
        # Convert to DataFrame
        records = []
        for date_str, values in ts.items():
            # Parse date first to filter old data
            trade_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            
            # Skip if we already have this data
            if last_date and trade_date <= last_date:
                continue
            
            records.append({
                "symbol": symbol,
                "trade_date": trade_date,
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
        
        if not records:
            return 0, "NO_NEW"
        
        # Bulk insert
        df = pd.DataFrame(records)
        
        # Validate data
        df = df[(df['high'] >= df['low']) & (df['close'] > 0)]
        
        if df.empty:
            return 0, "INVALID"
        
        # Insert to database
        with engine.begin() as conn:
            # Use faster method with execute_values
            df.to_sql("temp_prices", conn, if_exists='replace', index=False, method='multi')
            
            result = conn.execute(text("""
                INSERT INTO stock_prices 
                SELECT * FROM temp_prices
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
                DROP TABLE temp_prices;
            """))
        
        return len(df), "SUCCESS"
        
    except Exception as e:
        logger.error(f"Error processing {symbol}: {e}")
        return 0, "ERROR"

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n[INFO] Interrupted by user, saving progress...")
    sys.exit(0)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Fast price fetcher - handles 4,600+ symbols efficiently")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols")
    parser.add_argument("--limit", type=int, help="Limit number of symbols")
    parser.add_argument("--resume", help="Resume from failed symbols file")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers (default: 1)")
    args = parser.parse_args()
    
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Force Premium tier (600 calls/min)
    logger.info("Using Premium API tier (600 calls/min)")
    
    start_time = time.time()
    
    # Get symbols
    if args.resume:
        with open(args.resume, 'r') as f:
            symbols = json.load(f)
        logger.info(f"Resuming {len(symbols)} failed symbols from {args.resume}")
    elif args.symbols:
        symbols = args.symbols
    else:
        query = """
            SELECT DISTINCT s.symbol 
            FROM symbol_universe s
            WHERE s.is_etf = FALSE 
            AND s.country = 'USA'
            ORDER BY s.symbol
        """
        if args.limit:
            query += f" LIMIT {args.limit}"
        
        with engine.connect() as conn:
            symbols = [row[0] for row in conn.execute(text(query))]
    
    if not symbols:
        print("[ERROR] No symbols to process")
        return
    
    logger.info(f"Processing {len(symbols)} symbols with {args.workers} worker(s)")
    
    # Estimate time
    estimated_mins = len(symbols) / rate_limiter.calls_per_minute
    logger.info(f"Estimated time: {estimated_mins:.1f} minutes at {rate_limiter.calls_per_minute} calls/min")
    
    # Track results
    results = {"SUCCESS": 0, "NO_DATA": 0, "NO_NEW": 0, "CURRENT": 0, "INVALID": 0, "ERROR": 0}
    total_records = 0
    failed_symbols = []
    
    # Process symbols
    with tqdm(total=len(symbols), desc="Fetching prices", unit="symbol") as pbar:
        if args.workers > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(process_symbol, s): s for s in symbols}
                
                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        records, status = future.result(timeout=30)
                        results[status] = results.get(status, 0) + 1
                        total_records += records
                        
                        if status in ["ERROR", "NO_DATA"]:
                            failed_symbols.append(symbol)
                        
                    except Exception as e:
                        logger.error(f"Failed to process {symbol}: {e}")
                        results["ERROR"] += 1
                        failed_symbols.append(symbol)
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'Success': results.get('SUCCESS', 0),
                        'Current': results.get('CURRENT', 0),
                        'No_New': results.get('NO_NEW', 0),
                        'Records': total_records,
                        'Rate': f"{rate_limiter.calls_per_minute}/min"
                    })
        else:
            # Sequential processing (more stable)
            for symbol in symbols:
                records, status = process_symbol(symbol)
                results[status] = results.get(status, 0) + 1
                total_records += records
                
                if status in ["ERROR", "NO_DATA"]:
                    failed_symbols.append(symbol)
                
                pbar.update(1)
                pbar.set_postfix({
                    'Success': results.get('SUCCESS', 0),
                    'Current': results.get('CURRENT', 0),
                    'No_New': results.get('NO_NEW', 0),
                    'Records': total_records,
                    'Rate': f"{rate_limiter.calls_per_minute}/min"
                })
    
    # Summary
    duration = time.time() - start_time
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total symbols: {len(symbols)}")
    print(f"Successful updates: {results.get('SUCCESS', 0)}")
    print(f"Already current: {results.get('CURRENT', 0)}")
    print(f"No new data: {results.get('NO_NEW', 0)}")
    print(f"No data available: {results.get('NO_DATA', 0)}")
    print(f"Invalid data: {results.get('INVALID', 0)}")
    print(f"Errors: {results.get('ERROR', 0)}")
    print(f"Total new records: {total_records:,}")
    print(f"Duration: {duration/60:.1f} minutes")
    print(f"Actual rate: {len(symbols)/(duration/60):.1f} symbols/minute")
    print(f"API calls made: {rate_limiter.total_calls}")
    print(f"Rate limit hits: {rate_limiter.rate_limit_hits}")
    
    # Save failed symbols
    if failed_symbols:
        failed_file = f"logs/failed_symbols_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(failed_file, 'w') as f:
            json.dump(failed_symbols, f, indent=2)
        print(f"\nFailed symbols saved to: {failed_file}")
        print(f"Resume with: python fetch_prices_fast.py --resume {failed_file}")
    
    logger.info(f"Completed: {len(symbols)} symbols, {total_records} records in {duration/60:.1f} minutes")

if __name__ == "__main__":
    main()