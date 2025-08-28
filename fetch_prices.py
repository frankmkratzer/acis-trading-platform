#!/usr/bin/env python3
# File: fetch_prices.py
# Purpose: Fetch EOD daily adjusted prices with smart rate limiting
# Optimized for 600 calls/min Premium API

import os
import time
import random
import logging
import threading
import requests
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, timezone
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ─── Config ─────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
POSTGRES_URL = os.getenv("POSTGRES_URL")

if not API_KEY:
    print("[ERROR] ALPHA_VANTAGE_API_KEY not set in .env file")
    exit(1)

if not POSTGRES_URL:
    print("[ERROR] POSTGRES_URL not set in .env file") 
    exit(1)

engine = create_engine(POSTGRES_URL)
AV_URL = "https://www.alphavantage.co/query"

# Rate limiting - 600 calls/min = 10 calls/sec
MAX_WORKERS = 8  # Number of parallel threads
CALLS_PER_MIN = 580  # Slightly below 600 for safety
MIN_INTERVAL = 60.0 / CALLS_PER_MIN  # Minimum seconds between calls

# ─── Logging ────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/fetch_prices.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("fetch_prices")

# Simple rate limiter
last_call_time = 0
rate_limit_lock = threading.Lock()

def rate_limited_get(url, params, timeout=20):
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

# ─── DB helpers ─────────────────────────────────────────────────
def get_latest_trade_date(symbol):
    query = text("SELECT MAX(trade_date) FROM stock_prices WHERE symbol = :symbol")
    with engine.connect() as conn:
        return conn.execute(query, {"symbol": symbol}).scalar()

def upsert_prices(df: pd.DataFrame):
    if df.empty:
        return

    
    # Use pandas to_sql with temp table for bulk upsert
    with engine.begin() as conn:
        # Create temp table
        temp_table = f"temp_prices_{int(time.time())}"
        df.to_sql(temp_table, conn, if_exists='replace', index=False, method='multi')
        
        # Upsert from temp table with proper date casting
        conn.execute(text(f"""
            INSERT INTO stock_prices (symbol, trade_date, open, high, low, 
                                    close, adjusted_close, volume, dividend_amount, 
                                    split_coefficient, fetched_at)
            SELECT symbol, trade_date::date, open, high, low, 
                   close, adjusted_close, volume, dividend_amount,
                   split_coefficient, fetched_at
            FROM {temp_table}
            ON CONFLICT (symbol, trade_date) DO UPDATE SET
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

def fetch_price_data(symbol, outputsize="compact"):
    """Fetch price data for a symbol"""
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
            log.warning(f"HTTP {resp.status_code} for {symbol}")
            return None

        raw = resp.json()
        
        # Check for rate limit or error messages
        if "Note" in raw:
            log.warning(f"API rate limit note for {symbol}: {raw['Note']}")
            time.sleep(60)  # Wait a minute if rate limited
            return None
        
        if "Information" in raw:
            log.warning(f"API info for {symbol}: {raw['Information']}")
            time.sleep(10)
            return None
            
        if "Error Message" in raw:
            log.warning(f"API error for {symbol}: {raw['Error Message']}")
            return None

        ts = raw.get("Time Series (Daily)")
        if not ts:
            log.info(f"No price data for {symbol}")
            return None

        records = []
        for date_str, values in ts.items():
            raw_close = float(values.get("4. close", 0) or 0)
            raw_adjusted = float(values.get("5. adjusted close", 0) or 0)
            
            # Use close price if adjusted seems wrong
            if raw_adjusted == 0 or (raw_close > 0 and (raw_adjusted / raw_close < 0.5 or raw_adjusted / raw_close > 2.0)):
                adjusted_close = raw_close
            else:
                adjusted_close = raw_adjusted
            
            records.append({
                "symbol": symbol,
                "trade_date": date_str,
                "open": float(values.get("1. open", 0) or 0),
                "high": float(values.get("2. high", 0) or 0),
                "low": float(values.get("3. low", 0) or 0),
                "close": float(values.get("close", 0) or 0),
                "adjusted_close": adjusted_close,
                "volume": int(values.get("6. volume", 0) or 0),
                "dividend_amount": float(values.get("7. dividend amount", 0) or 0),
                "split_coefficient": float(values.get("8. split coefficient", 1) or 1),
                "fetched_at": datetime.now(timezone.utc),
            })
        
        return records

    except Exception as e:
        log.exception(f"Error fetching {symbol}: {e}")
        return None

def process_symbol(symbol, pbar=None):
    """Process a single symbol"""
    latest_date = get_latest_trade_date(symbol)
    
    # Use compact if we have data, full for new symbols
    outputsize = "compact" if latest_date else "full"
    
    if pbar:
        pbar.set_postfix({'symbol': symbol, 'mode': outputsize})
    
    try:
        records = fetch_price_data(symbol, outputsize=outputsize)
        if not records:
            return 0, 'NO_DATA'
        
        df = pd.DataFrame(records)
        
        # Filter to only new data if we have existing data
        if latest_date:
            df = df[pd.to_datetime(df["trade_date"]).dt.date > latest_date]
        
        if not df.empty:
            upsert_prices(df)
            return len(df), 'SUCCESS'
        else:
            return 0, 'NO_NEW'
            
    except Exception as e:
        log.error(f"Error processing {symbol}: {e}")
        return 0, 'FAILED'

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Price fetcher optimized for 600 calls/min Premium API")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to fetch")
    parser.add_argument("--limit", type=int, help="Limit number of symbols to fetch")
    parser.add_argument("--parallel", action="store_true", help="Use parallel processing (default: sequential)")
    args = parser.parse_args()
    
    start_time = time.time()
    print("[START] Price Fetcher Starting...")
    
    # Get symbols
    if args.symbols:
        symbols = args.symbols
        print(f"[INFO] Fetching specific symbols: {symbols}")
    else:
        query = "SELECT symbol FROM symbol_universe WHERE is_etf = FALSE AND country = 'USA'"
        if args.limit:
            query += f" LIMIT {args.limit}"
        symbols = pd.read_sql(query, engine)["symbol"].tolist()
        print(f"[INFO] Found {len(symbols)} symbols to fetch")
    
    if not symbols:
        print("[ERROR] No symbols found")
        return
    
    # Small random shuffle to avoid the same symbols always hitting at minute boundaries
    random.shuffle(symbols)
    
    total_records = 0
    status_counts = {'SUCCESS': 0, 'NO_DATA': 0, 'NO_NEW': 0, 'FAILED': 0}
    
    print("\n[PROGRESS] Processing symbols:")
    
    if args.parallel:
        # Parallel processing with ThreadPoolExecutor
        with tqdm(total=len(symbols), desc="Fetching prices", unit="symbol", 
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {executor.submit(process_symbol, s, pbar): s for s in symbols}
                
                for fut in as_completed(futures):
                    sym = futures[fut]
                    try:
                        records, status = fut.result()
                        status_counts[status] += 1
                        total_records += records
                        
                        pbar.update(1)
                        pbar.set_postfix({
                            'Success': status_counts['SUCCESS'],
                            'No_Data': status_counts['NO_DATA'],
                            'No_New': status_counts['NO_NEW'],
                            'Failed': status_counts['FAILED'],
                            'Records': total_records
                        })
                        
                    except Exception as e:
                        log.exception(f"Uncaught error processing {sym}: {e}")
                        status_counts['FAILED'] += 1
                        pbar.update(1)
    else:
        # Sequential processing (default, more stable)
        with tqdm(total=len(symbols), desc="Fetching prices", unit="symbol") as pbar:
            for symbol in symbols:
                records, status = process_symbol(symbol, pbar)
                status_counts[status] += 1
                total_records += records
                
                pbar.update(1)
                pbar.set_postfix({
                    'Success': status_counts['SUCCESS'],
                    'No_Data': status_counts['NO_DATA'],
                    'No_New': status_counts['NO_NEW'],
                    'Failed': status_counts['FAILED'],
                    'Records': total_records
                })
    
    duration = time.time() - start_time
    
    print(f"\n[SUMMARY] PRICE FETCHING SUMMARY:")
    print(f"   Total symbols: {len(symbols)}")
    print(f"   Successful: {status_counts['SUCCESS']}")
    print(f"   No data: {status_counts['NO_DATA']}")
    print(f"   No new data: {status_counts['NO_NEW']}")
    print(f"   Failed: {status_counts['FAILED']}")
    print(f"   Total records: {total_records:,}")
    print(f"   Duration: {duration:.1f}s")
    print(f"   Rate: {len(symbols)/duration:.2f} symbols/sec")

if __name__ == "__main__":
    main()