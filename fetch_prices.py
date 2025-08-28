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
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import Optional, Dict, List, Tuple
from functools import wraps
import json
from dataclasses import dataclass
from enum import Enum

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

# Retry configuration
MAX_RETRIES = 3
BASE_RETRY_DELAY = 1.0  # seconds
MAX_RETRY_DELAY = 300  # 5 minutes max
CIRCUIT_BREAKER_THRESHOLD = 5  # consecutive failures
CIRCUIT_BREAKER_TIMEOUT = 300  # 5 minutes

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

class FetchStatus(Enum):
    SUCCESS = "SUCCESS"
    NO_DATA = "NO_DATA"
    NO_NEW = "NO_NEW"
    FAILED = "FAILED"
    RATE_LIMITED = "RATE_LIMITED"
    INVALID_DATA = "INVALID_DATA"
    RETRYING = "RETRYING"

@dataclass
class CircuitBreaker:
    """Circuit breaker for API calls"""
    failure_count: int = 0
    last_failure_time: Optional[float] = None
    is_open: bool = False
    
    def record_success(self):
        self.failure_count = 0
        self.is_open = False
        
    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= CIRCUIT_BREAKER_THRESHOLD:
            self.is_open = True
            log.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def can_proceed(self) -> bool:
        if not self.is_open:
            return True
        if self.last_failure_time and (time.time() - self.last_failure_time > CIRCUIT_BREAKER_TIMEOUT):
            self.is_open = False
            self.failure_count = 0
            log.info("Circuit breaker reset after timeout")
            return True
        return False

circuit_breaker = CircuitBreaker()

def exponential_backoff_retry(max_retries=MAX_RETRIES):
    """Decorator for exponential backoff retry logic"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    if not circuit_breaker.can_proceed():
                        raise Exception("Circuit breaker is open")
                    
                    result = func(*args, **kwargs)
                    circuit_breaker.record_success()
                    return result
                    
                except (requests.RequestException, requests.Timeout) as e:
                    last_exception = e
                    circuit_breaker.record_failure()
                    
                    if attempt < max_retries - 1:
                        # Calculate delay with exponential backoff and jitter
                        delay = min(
                            BASE_RETRY_DELAY * (2 ** attempt) + random.uniform(0, 1),
                            MAX_RETRY_DELAY
                        )
                        log.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                    else:
                        log.error(f"All {max_retries} attempts failed: {e}")
                        
            raise last_exception if last_exception else Exception(f"Failed after {max_retries} attempts")
        return wrapper
    return decorator

@exponential_backoff_retry()
def rate_limited_get(url, params, timeout=20):
    """Rate-limited GET request with retry logic"""
    global last_call_time
    
    with rate_limit_lock:
        current_time = time.time()
        time_since_last = current_time - last_call_time
        if time_since_last < MIN_INTERVAL:
            sleep_time = MIN_INTERVAL - time_since_last
            time.sleep(sleep_time)
        last_call_time = time.time()
    
    # Set encoding to handle potential Unicode issues
    response = requests.get(url, params=params, timeout=timeout)
    response.encoding = response.apparent_encoding or 'utf-8'
    return response

# ─── DB helpers ─────────────────────────────────────────────────
def get_latest_trade_date(symbol):
    query = text("SELECT MAX(trade_date) FROM stock_prices WHERE symbol = :symbol")
    with engine.connect() as conn:
        return conn.execute(query, {"symbol": symbol}).scalar()

def validate_price_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Validate price data for consistency and quality"""
    issues = []
    
    # Remove rows with invalid OHLC relationships
    invalid_ohlc = (
        (df['high'] < df['low']) | 
        (df['high'] < df['open']) | 
        (df['high'] < df['close']) |
        (df['low'] > df['open']) | 
        (df['low'] > df['close'])
    )
    
    if invalid_ohlc.any():
        issues.append(f"Removed {invalid_ohlc.sum()} rows with invalid OHLC relationships")
        df = df[~invalid_ohlc]
    
    # Check for zero or negative prices
    zero_prices = (df[['open', 'high', 'low', 'close']] <= 0).any(axis=1)
    if zero_prices.any():
        issues.append(f"Removed {zero_prices.sum()} rows with zero/negative prices")
        df = df[~zero_prices]
    
    # Check for extreme price movements (>50% in a day)
    df['daily_change'] = abs(df['close'] - df['open']) / df['open']
    extreme_moves = df['daily_change'] > 0.5
    if extreme_moves.any():
        issues.append(f"Flagged {extreme_moves.sum()} rows with >50% daily moves")
        # Don't remove these, just flag them
    
    # Check for volume anomalies
    if 'volume' in df.columns:
        # Negative volumes
        negative_vol = df['volume'] < 0
        if negative_vol.any():
            issues.append(f"Fixed {negative_vol.sum()} rows with negative volume")
            df.loc[negative_vol, 'volume'] = 0
    
    # Remove temporary calculation columns
    df = df.drop(columns=['daily_change'], errors='ignore')
    
    return df, issues

def upsert_prices(df: pd.DataFrame, symbol: str = None):
    """Upsert prices with validation and error handling"""
    if df.empty:
        return
    
    # Validate data first
    df, validation_issues = validate_price_data(df)
    
    if validation_issues and symbol:
        for issue in validation_issues:
            log.warning(f"{symbol}: {issue}")
    
    if df.empty:
        log.error(f"All data failed validation for {symbol}")
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

        try:
            raw = resp.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            log.warning(f"Failed to decode JSON response for {symbol}: {e}")
            return None
        
        # Check for rate limit or error messages
        if "Note" in raw:
            log.warning(f"API rate limit note for {symbol}: {raw['Note']}")
            # Exponential backoff for rate limiting
            wait_time = min(60 * (2 ** circuit_breaker.failure_count), MAX_RETRY_DELAY)
            time.sleep(wait_time)
            return None
        
        if "Information" in raw:
            log.warning(f"API info for {symbol}: {raw['Information']}")
            # Check if this is a rate limit message
            if "higher API call volume" in raw.get('Information', ''):
                wait_time = min(30 * (2 ** circuit_breaker.failure_count), MAX_RETRY_DELAY)
                time.sleep(wait_time)
            else:
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
            # Fix: Remove 'or 0' which causes the bug
            raw_close = float(values.get("4. close", 0))
            raw_adjusted = float(values.get("5. adjusted close", 0))
            
            # Validate and use close price if adjusted seems wrong
            if raw_adjusted == 0 or (raw_close > 0 and (raw_adjusted / raw_close < 0.3 or raw_adjusted / raw_close > 3.0)):
                adjusted_close = raw_close
                log.debug(f"{symbol}: Using close price instead of adjusted for {date_str}")
            else:
                adjusted_close = raw_adjusted
            
            records.append({
                "symbol": symbol,
                "trade_date": date_str,
                "open": float(values.get("1. open", 0)),
                "high": float(values.get("2. high", 0)),
                "low": float(values.get("3. low", 0)),
                "close": raw_close,
                "adjusted_close": adjusted_close,
                "volume": int(values.get("6. volume", 0)),
                "dividend_amount": float(values.get("7. dividend amount", 0)),
                "split_coefficient": float(values.get("8. split coefficient", 1)),
                "fetched_at": datetime.now(timezone.utc),
            })
        
        return records

    except Exception as e:
        log.exception(f"Error fetching {symbol}: {e}")
        return None

def process_symbol(symbol, pbar=None, retry_count=0) -> Tuple[int, str]:
    """Process a single symbol with comprehensive error handling"""
    max_symbol_retries = 3
    
    try:
        latest_date = get_latest_trade_date(symbol)
        
        # Use compact if we have recent data (within 30 days), full otherwise
        days_since_update = (datetime.now().date() - latest_date).days if latest_date else float('inf')
        outputsize = "compact" if latest_date and days_since_update < 30 else "full"
        
        if pbar:
            pbar.set_postfix({'symbol': symbol, 'mode': outputsize, 'retry': retry_count})
        
        # Check circuit breaker
        if not circuit_breaker.can_proceed():
            log.warning(f"Circuit breaker open, skipping {symbol}")
            return 0, FetchStatus.RATE_LIMITED.value
        
        records = fetch_price_data(symbol, outputsize=outputsize)
        if not records:
            # Retry with exponential backoff for temporary failures
            if retry_count < max_symbol_retries:
                time.sleep(BASE_RETRY_DELAY * (2 ** retry_count))
                return process_symbol(symbol, pbar, retry_count + 1)
            return 0, FetchStatus.NO_DATA.value
        
        df = pd.DataFrame(records)
        
        # Filter to only new data if we have existing data
        if latest_date:
            df = df[pd.to_datetime(df["trade_date"]).dt.date > latest_date]
        
        if not df.empty:
            # Validate before inserting
            original_len = len(df)
            df, issues = validate_price_data(df)
            
            if df.empty:
                log.error(f"{symbol}: All {original_len} records failed validation")
                return 0, FetchStatus.INVALID_DATA.value
            
            if len(df) < original_len:
                log.warning(f"{symbol}: {original_len - len(df)} records failed validation")
            
            upsert_prices(df, symbol)
            return len(df), FetchStatus.SUCCESS.value
        else:
            return 0, FetchStatus.NO_NEW.value
            
    except requests.RequestException as e:
        log.error(f"Network error processing {symbol}: {e}")
        if retry_count < max_symbol_retries:
            time.sleep(BASE_RETRY_DELAY * (2 ** retry_count))
            return process_symbol(symbol, pbar, retry_count + 1)
        return 0, FetchStatus.FAILED.value
        
    except Exception as e:
        log.exception(f"Unexpected error processing {symbol}: {e}")
        return 0, FetchStatus.FAILED.value

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
    status_counts = {status.value: 0 for status in FetchStatus}
    failed_symbols = []
    
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
                        
                        if status == FetchStatus.FAILED.value:
                            failed_symbols.append(sym)
                        
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
                
                if status == FetchStatus.FAILED.value:
                    failed_symbols.append(symbol)
                
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
    print(f"   Successful: {status_counts.get(FetchStatus.SUCCESS.value, 0)}")
    print(f"   No data: {status_counts.get(FetchStatus.NO_DATA.value, 0)}")
    print(f"   No new data: {status_counts.get(FetchStatus.NO_NEW.value, 0)}")
    print(f"   Invalid data: {status_counts.get(FetchStatus.INVALID_DATA.value, 0)}")
    print(f"   Rate limited: {status_counts.get(FetchStatus.RATE_LIMITED.value, 0)}")
    print(f"   Failed: {status_counts.get(FetchStatus.FAILED.value, 0)}")
    print(f"   Total records: {total_records:,}")
    print(f"   Duration: {duration:.1f}s")
    print(f"   Rate: {len(symbols)/duration:.2f} symbols/sec")
    
    # Save failed symbols for retry
    if failed_symbols:
        failed_file = f"logs/failed_symbols_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(failed_file, 'w') as f:
            json.dump(failed_symbols, f, indent=2)
        print(f"\n[INFO] Failed symbols saved to {failed_file} for retry")
    
    # Log summary to file
    log.info(f"Fetch complete: {len(symbols)} symbols, {total_records} records, "
             f"{status_counts.get(FetchStatus.FAILED.value, 0)} failures")

if __name__ == "__main__":
    main()