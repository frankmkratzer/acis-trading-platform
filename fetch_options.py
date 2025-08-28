#!/usr/bin/env python3
"""
Options Data Fetcher for Alpha Vantage
Fetches real-time and historical options chains
Optimized for 600 calls/min Premium API
"""

import os
import sys
import time
import logging
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from tqdm import tqdm

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

# Rate limiting - 600 calls/min = 10 calls/sec
MAX_WORKERS = int(os.getenv("OPTIONS_THREADS", "8"))
CALLS_PER_MIN = 580  # Slightly below 600 for safety
MIN_INTERVAL = 60.0 / CALLS_PER_MIN  # Minimum seconds between calls

# Options-specific configuration
FETCH_HISTORICAL = os.getenv("OPTIONS_FETCH_HISTORICAL", "0").lower() in ("1", "true", "yes")
HISTORICAL_DAYS_BACK = int(os.getenv("OPTIONS_HISTORICAL_DAYS", "7"))
FETCH_GREEKS = os.getenv("OPTIONS_FETCH_GREEKS", "1").lower() in ("1", "true", "yes")
MIN_VOLUME_FILTER = int(os.getenv("OPTIONS_MIN_VOLUME", "10"))
MAX_DAYS_TO_EXPIRY = int(os.getenv("OPTIONS_MAX_DAYS_TO_EXPIRY", "180"))

# Logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/fetch_options.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("fetch_options")

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

def get_active_symbols():
    """Get symbols that have options (typically liquid stocks and ETFs)"""
    query = text("""
        SELECT s.symbol, s.market_cap
        FROM symbol_universe s
        WHERE s.is_etf = TRUE  -- ETFs typically have options
           OR s.market_cap > 10000000000  -- Large cap stocks
           OR s.symbol IN (
               -- Always include major indices and popular options symbols
               'SPY', 'QQQ', 'IWM', 'DIA', 'VIX', 'GLD', 'SLV', 
               'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'AMD'
           )
        ORDER BY s.market_cap DESC NULLS LAST
        LIMIT :limit
    """)
    
    limit = int(os.getenv("OPTIONS_SYMBOLS_LIMIT", "50"))
    
    with engine.connect() as conn:
        result = conn.execute(query, {"limit": limit})
        symbols = [row[0] for row in result]  # row[0] is symbol, row[1] is market_cap
    
    logger.info(f"Found {len(symbols)} symbols with likely options activity")
    return symbols

def fetch_realtime_options(symbol):
    """Fetch real-time options chain for a symbol"""
    params = {
        "function": "REALTIME_OPTIONS",
        "symbol": symbol,
        "apikey": API_KEY,
        "datatype": "json"
    }
    
    if FETCH_GREEKS:
        params["require_greeks"] = "true"
    
    try:
        resp = rate_limited_get(AV_URL, params)
        
        if resp.status_code != 200:
            logger.warning(f"HTTP {resp.status_code} for realtime options {symbol}")
            return None
        
        data = resp.json()
        
        # Check for rate limit or error messages
        if "Note" in data:
            logger.warning(f"API rate limit note for {symbol}: {data['Note']}")
            time.sleep(60)  # Wait a minute if rate limited
            return None
        
        if "Information" in data:
            logger.warning(f"API info for {symbol}: {data['Information']}")
            time.sleep(10)
            return None
            
        if "Error Message" in data:
            logger.warning(f"API error for {symbol}: {data['Error Message']}")
            return None
        
        # Check if we have options data
        if "data" not in data:
            logger.info(f"No options data available for {symbol}")
            return None
        
        return data.get("data", [])
        
    except Exception as e:
        logger.error(f"Error fetching realtime options for {symbol}: {e}")
        return None

def fetch_historical_options(symbol, date=None):
    """Fetch historical options chain for a specific date"""
    params = {
        "function": "HISTORICAL_OPTIONS",
        "symbol": symbol,
        "apikey": API_KEY,
        "datatype": "json"
    }
    
    if date:
        params["date"] = date.strftime("%Y-%m-%d")
    
    try:
        resp = rate_limited_get(AV_URL, params)
        
        if resp.status_code != 200:
            logger.warning(f"HTTP {resp.status_code} for historical options {symbol} on {date}")
            return None
        
        data = resp.json()
        
        # Check for rate limit or error messages
        if "Note" in data:
            logger.warning(f"API rate limit note: {data['Note']}")
            time.sleep(60)
            return None
        
        if "Information" in data:
            logger.warning(f"API info: {data['Information']}")
            time.sleep(10)
            return None
            
        if "Error Message" in data:
            logger.warning(f"API error: {data['Error Message']}")
            return None
        
        # Check if we have options data
        if "data" not in data:
            logger.info(f"No historical options data for {symbol} on {date}")
            return None
        
        return data.get("data", [])
        
    except Exception as e:
        logger.error(f"Error fetching historical options for {symbol}: {e}")
        return None

def parse_options_data(raw_data, symbol, fetch_type="realtime"):
    """Parse raw options data into DataFrame"""
    if not raw_data:
        return pd.DataFrame()
    
    records = []
    for option in raw_data:
        # Basic fields
        record = {
            "symbol": symbol,
            "contract_id": option.get("contractID") or option.get("contract_id"),
            "option_type": option.get("type", "").lower(),  # call or put
            "strike_price": float(option.get("strike", 0) or 0),
            "expiration_date": pd.to_datetime(option.get("expiration"), errors="coerce"),
            "quote_date": datetime.now(timezone.utc).date(),
            "bid": float(option.get("bid", 0) or 0),
            "ask": float(option.get("ask", 0) or 0),
            "last_price": float(option.get("last", 0) or 0),
            "volume": int(option.get("volume", 0) or 0),
            "open_interest": int(option.get("open_interest", 0) or 0),
            "implied_volatility": float(option.get("implied_volatility", 0) or 0),
            "in_the_money": option.get("in_the_money", "").lower() == "true",
            "fetched_at": datetime.now(timezone.utc)
        }
        
        # Greeks (if available)
        if FETCH_GREEKS:
            record["delta"] = float(option.get("delta", 0) or 0)
            record["gamma"] = float(option.get("gamma", 0) or 0)
            record["theta"] = float(option.get("theta", 0) or 0)
            record["vega"] = float(option.get("vega", 0) or 0)
            record["rho"] = float(option.get("rho", 0) or 0)
        
        # Calculate derived fields
        if record["bid"] > 0 and record["ask"] > 0:
            record["time_value"] = record["last_price"] if not record["in_the_money"] else None
            record["intrinsic_value"] = max(0, record["last_price"] - record["time_value"]) if record["time_value"] else None
        
        records.append(record)
    
    df = pd.DataFrame(records)
    
    # Filter based on configuration
    if not df.empty:
        # Filter by volume
        if MIN_VOLUME_FILTER > 0:
            df = df[(df["volume"] >= MIN_VOLUME_FILTER) | (df["open_interest"] >= MIN_VOLUME_FILTER * 10)]
        
        # Filter by days to expiration
        if MAX_DAYS_TO_EXPIRY > 0 and 'expiration_date' in df.columns:
            df['days_to_expiry'] = (pd.to_datetime(df['expiration_date']) - datetime.now(timezone.utc)).dt.days
            df = df[df["days_to_expiry"] <= MAX_DAYS_TO_EXPIRY]
            df = df.drop('days_to_expiry', axis=1)  # Remove temporary column
    
    return df

def upsert_options_data(df):
    """Upsert options data to database"""
    if df.empty:
        return 0
    
    try:
        # Use pandas to_sql with temp table for bulk upsert
        with engine.begin() as conn:
            # Create temp table
            temp_table = f"temp_options_{int(time.time())}"
            df.to_sql(temp_table, conn, if_exists='replace', index=False, method='multi')
            
            # Upsert from temp table
            conn.execute(text(f"""
                INSERT INTO options_data (
                    symbol, contract_id, option_type, strike_price, expiration_date,
                    quote_date, bid, ask, last_price, volume, open_interest,
                    implied_volatility, delta, gamma, theta, vega, rho,
                    in_the_money, time_value, intrinsic_value, fetched_at
                )
                SELECT 
                    symbol, contract_id, option_type, strike_price, expiration_date::date,
                    quote_date::date, bid, ask, last_price, volume, open_interest,
                    implied_volatility, delta, gamma, theta, vega, rho,
                    in_the_money, time_value, intrinsic_value, fetched_at
                FROM {temp_table}
                ON CONFLICT (contract_id, quote_date) DO UPDATE SET
                    bid = EXCLUDED.bid,
                    ask = EXCLUDED.ask,
                    last_price = EXCLUDED.last_price,
                    volume = EXCLUDED.volume,
                    open_interest = EXCLUDED.open_interest,
                    implied_volatility = EXCLUDED.implied_volatility,
                    delta = EXCLUDED.delta,
                    gamma = EXCLUDED.gamma,
                    theta = EXCLUDED.theta,
                    vega = EXCLUDED.vega,
                    rho = EXCLUDED.rho,
                    fetched_at = EXCLUDED.fetched_at
            """))
            
            # Drop temp table
            conn.execute(text(f"DROP TABLE {temp_table}"))
            
            return len(df)
            
    except Exception as e:
        logger.error(f"Error upserting options data: {e}")
        return 0

def process_symbol_options(symbol):
    """Process options data for a single symbol"""
    total_records = 0
    
    try:
        # Fetch real-time options
        realtime_data = fetch_realtime_options(symbol)
        if realtime_data:
            df_realtime = parse_options_data(realtime_data, symbol, "realtime")
            if not df_realtime.empty:
                records = upsert_options_data(df_realtime)
                total_records += records
                logger.info(f"Saved {records} realtime options for {symbol}")
        
        # Fetch historical options if configured
        if FETCH_HISTORICAL:
            # Fetch last N days of historical data
            end_date = datetime.now().date()
            for days_back in range(1, min(HISTORICAL_DAYS_BACK + 1, 7)):  # Max 7 days
                hist_date = end_date - timedelta(days=days_back)
                
                # Skip weekends
                if hist_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                    continue
                
                historical_data = fetch_historical_options(symbol, hist_date)
                if historical_data:
                    df_historical = parse_options_data(historical_data, symbol, f"historical_{hist_date}")
                    if not df_historical.empty:
                        records = upsert_options_data(df_historical)
                        total_records += records
                        logger.info(f"Saved {records} historical options for {symbol} on {hist_date}")
        
        return total_records
        
    except Exception as e:
        logger.error(f"Error processing options for {symbol}: {e}")
        return 0

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Options fetcher optimized for 600 calls/min Premium API")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to fetch")
    parser.add_argument("--limit", type=int, help="Limit number of symbols")
    parser.add_argument("--parallel", action="store_true", help="Use parallel processing")
    args = parser.parse_args()
    
    start_time = time.time()
    print("\n" + "=" * 60)
    print("OPTIONS DATA FETCHER")
    print("=" * 60)
    
    # Get symbols with options
    if args.symbols:
        symbols = args.symbols
        print(f"[INFO] Fetching specific symbols: {symbols}")
    else:
        symbols = get_active_symbols()
        if args.limit:
            symbols = symbols[:args.limit]
        print(f"[INFO] Found {len(symbols)} symbols with likely options")
    
    if not symbols:
        print("[ERROR] No symbols found")
        return
    
    total_records = 0
    status_counts = {'SUCCESS': 0, 'NO_DATA': 0, 'FAILED': 0}
    
    print("\n[PROGRESS] Processing symbols:")
    
    if args.parallel:
        # Parallel processing
        with tqdm(total=len(symbols), desc="Fetching options", unit="symbol") as pbar:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {executor.submit(process_symbol_options, s): s for s in symbols}
                
                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        records = future.result()
                        if records > 0:
                            status_counts['SUCCESS'] += 1
                            total_records += records
                        else:
                            status_counts['NO_DATA'] += 1
                        
                        pbar.update(1)
                        pbar.set_postfix({
                            'Success': status_counts['SUCCESS'],
                            'No_Data': status_counts['NO_DATA'],
                            'Failed': status_counts['FAILED'],
                            'Records': total_records
                        })
                        
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
                        status_counts['FAILED'] += 1
                        pbar.update(1)
    else:
        # Sequential processing
        with tqdm(total=len(symbols), desc="Fetching options", unit="symbol") as pbar:
            for symbol in symbols:
                try:
                    records = process_symbol_options(symbol)
                    if records > 0:
                        status_counts['SUCCESS'] += 1
                        total_records += records
                    else:
                        status_counts['NO_DATA'] += 1
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'Success': status_counts['SUCCESS'],
                        'No_Data': status_counts['NO_DATA'],
                        'Failed': status_counts['FAILED'],
                        'Records': total_records
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    status_counts['FAILED'] += 1
                    pbar.update(1)
    
    duration = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total symbols: {len(symbols)}")
    print(f"Successful: {status_counts['SUCCESS']}")
    print(f"No data: {status_counts['NO_DATA']}")
    print(f"Failed: {status_counts['FAILED']}")
    print(f"Total records: {total_records:,}")
    print(f"Duration: {duration:.1f}s")
    print(f"Rate: {total_records/duration:.2f} records/sec" if duration > 0 else "N/A")
    
    logger.info(f"Options fetch complete: {status_counts}")

if __name__ == "__main__":
    main()