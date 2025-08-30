#!/usr/bin/env python3
"""
Insider Transactions Fetcher
Fetches insider buying/selling data from Alpha Vantage
Tracks CEO/CFO buying patterns for alpha generation
"""

import os
import sys
import time
import json
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from sqlalchemy import text
from dotenv import load_dotenv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.logging_config import setup_logger, log_script_start, log_script_end
from database.db_connection_manager import DatabaseConnectionManager
from data_fetch.base.rate_limiter import AlphaVantageRateLimiter

# Load environment
load_dotenv()

# Initialize centralized utilities
logger = setup_logger("fetch_insider_transactions")
db_manager = DatabaseConnectionManager()
engine = db_manager.get_engine()
rate_limiter = AlphaVantageRateLimiter.get_instance()

API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
if not API_KEY:
    logger.error("ALPHA_VANTAGE_API_KEY not set")
    sys.exit(1)

AV_URL = "https://www.alphavantage.co/query"

# Configuration
MAX_WORKERS = 8  # Can use more workers with centralized rate limiter
CALLS_PER_MIN = 600  # Premium API tier rate limit
LOOKBACK_DAYS = None  # Fetch all available historical transactions

def rate_limited_get(url, params, timeout=20):
    """Rate-limited GET request using centralized rate limiter"""
    rate_limiter.wait_if_needed()
    return requests.get(url, params=params, timeout=timeout)

def fetch_insider_transactions(symbol):
    """Fetch insider transaction data for a symbol"""
    try:
        params = {
            "function": "INSIDER_TRANSACTIONS",
            "symbol": symbol,
            "apikey": API_KEY
        }
        
        response = rate_limited_get(AV_URL, params)
        
        if response.status_code != 200:
            logger.warning(f"HTTP {response.status_code} for {symbol}")
            return None
            
        data = response.json()
        
        if "Error Message" in data or "Note" in data:
            logger.warning(f"API error for {symbol}: {data}")
            return None
        
        if "data" not in data:
            logger.debug(f"No insider data for {symbol}")
            return None
            
        return data
        
    except Exception as e:
        logger.error(f"Error fetching insider transactions for {symbol}: {e}")
        return None

def parse_insider_data(symbol, data):
    """Parse insider transaction data into records"""
    if not data or "data" not in data:
        return []
    
    records = []
    
    for transaction in data["data"]:
        try:
            # Parse transaction date
            trans_date = pd.to_datetime(transaction.get("transaction_date"))
            if pd.isna(trans_date):
                continue
            
            # Skip old transactions (make timezone-naive for comparison)
            if trans_date.replace(tzinfo=None) < datetime.now() - timedelta(days=LOOKBACK_DAYS):
                continue
            
            # Parse shares (using correct field name from API)
            shares = transaction.get("shares", 0)
            if isinstance(shares, str):
                shares = float(shares.replace(",", "").replace("", "0"))
            
            # Parse price per share
            price_per_share = transaction.get("share_price", 0)
            if isinstance(price_per_share, str):
                price_per_share = float(price_per_share.replace(",", "").replace("$", "").replace("", "0"))
            
            # Calculate value if not provided
            value = abs(shares * price_per_share) if price_per_share > 0 else 0
            
            # Determine transaction type (A = Acquisition/Buy, D = Disposition/Sell)
            acquisition = transaction.get("acquisition_or_disposal", "").upper()
            is_purchase = acquisition in ["A", "ACQUISITION", "BUY", "P"]
            
            # Get owner details (using correct field names)
            owner_name = transaction.get("executive", "")
            owner_title = transaction.get("executive_title", "").upper()
            
            # Identify key insiders
            is_ceo = "CEO" in owner_title or "CHIEF EXECUTIVE" in owner_title
            is_cfo = "CFO" in owner_title or "CHIEF FINANCIAL" in owner_title
            is_director = "DIRECTOR" in owner_title
            is_officer = "OFFICER" in owner_title or is_ceo or is_cfo
            
            # Check for 10b5-1 plan (scheduled sales) - not in this API response
            is_10b51 = False
            
            record = {
                'symbol': symbol,
                'transaction_date': trans_date.date(),
                'reporting_date': trans_date.date(),  # API doesn't provide separate filing date
                'owner_name': owner_name,
                'owner_title': owner_title,
                'owner_type': transaction.get("security_type", "Common Stock"),  # Using security_type as proxy
                'transaction_type': "BUY" if is_purchase else "SELL",
                'shares': abs(shares) if shares else 0,
                'price_per_share': price_per_share if price_per_share > 0 else None,
                'total_value': abs(value),
                'shares_owned_after': None,  # Not provided by this API
                'is_ceo': is_ceo,
                'is_cfo': is_cfo,
                'is_director': is_director,
                'is_officer': is_officer,
                'is_10b51_plan': is_10b51,
                'fetched_at': datetime.now(timezone.utc)
            }
            
            records.append(record)
            
        except Exception as e:
            logger.error(f"Error parsing transaction for {symbol}: {e}")
            continue
    
    return records

def calculate_insider_signals(symbol_transactions):
    """Calculate insider buying/selling signals from transactions"""
    if not symbol_transactions:
        return None
    
    df = pd.DataFrame(symbol_transactions)
    
    # Calculate signals for last 30, 90, 180 days
    now = datetime.now().date()
    
    signals = {
        'symbol': df['symbol'].iloc[0],
        'calculation_date': now,
        
        # 30-day metrics
        'buys_30d': len(df[(df['transaction_type'] == 'BUY') & (df['transaction_date'] > now - timedelta(days=30))]),
        'sells_30d': len(df[(df['transaction_type'] == 'SELL') & (df['transaction_date'] > now - timedelta(days=30))]),
        'buy_value_30d': df[(df['transaction_type'] == 'BUY') & (df['transaction_date'] > now - timedelta(days=30))]['total_value'].sum(),
        'sell_value_30d': df[(df['transaction_type'] == 'SELL') & (df['transaction_date'] > now - timedelta(days=30))]['total_value'].sum(),
        
        # 90-day metrics
        'buys_90d': len(df[(df['transaction_type'] == 'BUY') & (df['transaction_date'] > now - timedelta(days=90))]),
        'sells_90d': len(df[(df['transaction_type'] == 'SELL') & (df['transaction_date'] > now - timedelta(days=90))]),
        
        # Key insider activity
        'ceo_buying': any(df[(df['is_ceo'] == True) & (df['transaction_type'] == 'BUY') & (df['transaction_date'] > now - timedelta(days=90))].index),
        'cfo_buying': any(df[(df['is_cfo'] == True) & (df['transaction_type'] == 'BUY') & (df['transaction_date'] > now - timedelta(days=90))].index),
        'officer_cluster_buying': len(df[(df['is_officer'] == True) & (df['transaction_type'] == 'BUY') & (df['transaction_date'] > now - timedelta(days=30))]) >= 3,
        
        # Net insider sentiment
        'net_insider_buying_30d': df[(df['transaction_date'] > now - timedelta(days=30))].apply(
            lambda x: x['total_value'] if x['transaction_type'] == 'BUY' else -x['total_value'], axis=1
        ).sum(),
        
        'updated_at': datetime.now(timezone.utc)
    }
    
    # Calculate insider score (0-100)
    score = 50  # Base score
    
    # CEO/CFO buying is strongest signal
    if signals['ceo_buying']:
        score += 20
    if signals['cfo_buying']:
        score += 15
    
    # Cluster buying is very bullish
    if signals['officer_cluster_buying']:
        score += 15
    
    # Net buying/selling ratio
    if signals['buys_30d'] > 0 or signals['sells_30d'] > 0:
        buy_sell_ratio = signals['buys_30d'] / (signals['buys_30d'] + signals['sells_30d'])
        score += (buy_sell_ratio - 0.5) * 20  # Add -10 to +10 based on ratio
    
    # Net value sentiment
    if signals['net_insider_buying_30d'] > 1000000:  # Net buying > $1M
        score += 10
    elif signals['net_insider_buying_30d'] < -1000000:  # Net selling > $1M
        score -= 10
    
    signals['insider_score'] = max(0, min(100, score))
    
    return signals

def upsert_transactions(transactions):
    """Upsert insider transactions to database"""
    if not transactions:
        return 0
    
    df = pd.DataFrame(transactions)
    
    try:
        with engine.begin() as conn:
            # Create temp table
            temp_table = f"temp_insider_trans_{int(time.time() * 1000)}"
            df.to_sql(temp_table, conn, if_exists='replace', index=False, method='multi', chunksize=1000)
            
            # Upsert from temp table
            conn.execute(text(f"""
                INSERT INTO insider_transactions (
                    symbol, transaction_date, reporting_date, owner_name, owner_title,
                    owner_type, transaction_type, shares, price_per_share, total_value,
                    shares_owned_after, is_ceo, is_cfo, is_director, is_officer,
                    is_10b51_plan, fetched_at
                )
                SELECT 
                    symbol, transaction_date::date, reporting_date::date, owner_name, owner_title,
                    owner_type, transaction_type, shares::numeric, price_per_share::numeric, total_value::numeric,
                    shares_owned_after::numeric, is_ceo::boolean, is_cfo::boolean, is_director::boolean, is_officer::boolean,
                    is_10b51_plan::boolean, fetched_at
                FROM {temp_table}
                ON CONFLICT (symbol, transaction_date, owner_name, transaction_type, shares) 
                DO UPDATE SET
                    reporting_date = EXCLUDED.reporting_date,
                    price_per_share = EXCLUDED.price_per_share,
                    total_value = EXCLUDED.total_value,
                    shares_owned_after = EXCLUDED.shares_owned_after,
                    fetched_at = EXCLUDED.fetched_at
            """))
            
            # Drop temp table
            conn.execute(text(f"DROP TABLE {temp_table}"))
            
            return len(df)
            
    except Exception as e:
        logger.error(f"Error upserting transactions: {e}")
        return 0

def upsert_signals(signals):
    """Upsert insider signals to database"""
    if not signals:
        return 0
    
    df = pd.DataFrame(signals)
    
    try:
        with engine.begin() as conn:
            # Create temp table
            temp_table = f"temp_insider_signals_{int(time.time() * 1000)}"
            df.to_sql(temp_table, conn, if_exists='replace', index=False, method='multi')
            
            # Upsert from temp table
            conn.execute(text(f"""
                INSERT INTO insider_signals (
                    symbol, calculation_date, buys_30d, sells_30d, buy_value_30d, sell_value_30d,
                    buys_90d, sells_90d, ceo_buying, cfo_buying, officer_cluster_buying,
                    net_insider_buying_30d, insider_score, updated_at
                )
                SELECT 
                    symbol, calculation_date::date, buys_30d::int, sells_30d::int, 
                    buy_value_30d::numeric, sell_value_30d::numeric,
                    buys_90d::int, sells_90d::int, ceo_buying::boolean, cfo_buying::boolean, 
                    officer_cluster_buying::boolean, net_insider_buying_30d::numeric, 
                    insider_score::numeric, updated_at
                FROM {temp_table}
                ON CONFLICT (symbol) DO UPDATE SET
                    calculation_date = EXCLUDED.calculation_date,
                    buys_30d = EXCLUDED.buys_30d,
                    sells_30d = EXCLUDED.sells_30d,
                    buy_value_30d = EXCLUDED.buy_value_30d,
                    sell_value_30d = EXCLUDED.sell_value_30d,
                    buys_90d = EXCLUDED.buys_90d,
                    sells_90d = EXCLUDED.sells_90d,
                    ceo_buying = EXCLUDED.ceo_buying,
                    cfo_buying = EXCLUDED.cfo_buying,
                    officer_cluster_buying = EXCLUDED.officer_cluster_buying,
                    net_insider_buying_30d = EXCLUDED.net_insider_buying_30d,
                    insider_score = EXCLUDED.insider_score,
                    updated_at = EXCLUDED.updated_at
            """))
            
            # Drop temp table
            conn.execute(text(f"DROP TABLE {temp_table}"))
            
            return len(df)
            
    except Exception as e:
        logger.error(f"Error upserting signals: {e}")
        return 0

def process_symbol(symbol):
    """Process insider transactions for a single symbol"""
    try:
        # Fetch data
        data = fetch_insider_transactions(symbol)
        if not data:
            return symbol, 0, 0, 'NO_DATA'
        
        # Parse transactions
        transactions = parse_insider_data(symbol, data)
        if not transactions:
            return symbol, 0, 0, 'NO_TRANSACTIONS'
        
        # Calculate signals
        signals = calculate_insider_signals(transactions)
        
        # Upsert to database
        trans_count = upsert_transactions(transactions)
        signal_count = upsert_signals([signals]) if signals else 0
        
        if trans_count > 0:
            return symbol, trans_count, signal_count, 'SUCCESS'
        else:
            return symbol, 0, 0, 'FAILED'
            
    except Exception as e:
        logger.error(f"Error processing {symbol}: {e}")
        return symbol, 0, 0, 'ERROR'

def get_symbols_to_update():
    """Get symbols that need insider transaction updates"""
    query = text("""
        WITH last_update AS (
            SELECT symbol, MAX(fetched_at) as last_fetched
            FROM insider_transactions
            GROUP BY symbol
        )
        SELECT s.symbol
        FROM symbol_universe s
        LEFT JOIN last_update lu ON s.symbol = lu.symbol
        WHERE s.is_etf = FALSE 
        AND s.country = 'USA'
        AND s.market_cap >= 2000000000  -- Mid-cap and above
        AND (
            lu.last_fetched IS NULL  -- Never fetched
            OR lu.last_fetched < CURRENT_TIMESTAMP - INTERVAL '7 days'  -- Update weekly
        )
        AND s.symbol IN (
            SELECT symbol FROM stock_prices 
            GROUP BY symbol 
            HAVING MAX(trade_date) >= CURRENT_DATE - INTERVAL '30 days'  -- Active stocks only
        )
        ORDER BY 
            lu.last_fetched ASC NULLS FIRST,  -- Prioritize never fetched
            s.market_cap DESC  -- Then by market cap
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query)
        return [row[0] for row in result]

def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch insider transaction data from Alpha Vantage")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to process")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of symbols")
    parser.add_argument("--parallel", action="store_true", help="Use parallel processing")
    args = parser.parse_args()
    
    start_time = time.time()
    log_script_start(logger, "fetch_insider_transactions", "Fetching insider transaction data")
    
    print("\n" + "=" * 60)
    print("INSIDER TRANSACTIONS FETCHER")
    print("=" * 60)
    print(f"[CONFIG] Rate limit: {CALLS_PER_MIN} calls/minute")
    print(f"[CONFIG] Workers: {MAX_WORKERS if args.parallel else 1}")
    print(f"[CONFIG] Lookback: {LOOKBACK_DAYS} days")
    
    # Get symbols
    if args.symbols:
        symbols = args.symbols
        print(f"[INFO] Processing specific symbols: {symbols}")
    else:
        symbols = get_symbols_to_update()
        if args.limit:
            symbols = symbols[:args.limit]
        print(f"[INFO] Found {len(symbols)} symbols to process")
    
    if not symbols:
        print("[INFO] No symbols need updating")
        return
    
    # Process symbols
    results = {'SUCCESS': 0, 'NO_DATA': 0, 'NO_TRANSACTIONS': 0, 'FAILED': 0, 'ERROR': 0}
    total_transactions = 0
    total_signals = 0
    high_score_symbols = []
    
    # Estimate time
    estimated_time = len(symbols) / CALLS_PER_MIN * 60
    print(f"[INFO] Estimated time: {estimated_time/60:.1f} minutes")
    
    with tqdm(total=len(symbols), desc="Fetching insider data", unit="symbol") as pbar:
        if args.parallel:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {executor.submit(process_symbol, s): s for s in symbols}
                
                for future in as_completed(futures):
                    symbol, trans_count, signal_count, status = future.result()
                    results[status] += 1
                    total_transactions += trans_count
                    total_signals += signal_count
                    
                    # Track high insider score symbols
                    if signal_count > 0:
                        with engine.connect() as conn:
                            score_result = conn.execute(text(
                                "SELECT insider_score FROM insider_signals WHERE symbol = :symbol"
                            ), {"symbol": symbol}).fetchone()
                            if score_result and score_result[0] >= 70:
                                high_score_symbols.append((symbol, score_result[0]))
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'Success': results['SUCCESS'],
                        'Transactions': total_transactions
                    })
        else:
            for symbol in symbols:
                symbol, trans_count, signal_count, status = process_symbol(symbol)
                results[status] += 1
                total_transactions += trans_count
                total_signals += signal_count
                
                # Track high insider score symbols
                if signal_count > 0:
                    with engine.connect() as conn:
                        score_result = conn.execute(text(
                            "SELECT insider_score FROM insider_signals WHERE symbol = :symbol"
                        ), {"symbol": symbol}).fetchone()
                        if score_result and score_result[0] >= 70:
                            high_score_symbols.append((symbol, score_result[0]))
                
                pbar.update(1)
                pbar.set_postfix({
                    'Success': results['SUCCESS'],
                    'Transactions': total_transactions
                })
    
    # Summary
    duration = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total symbols: {len(symbols)}")
    print(f"Successful: {results['SUCCESS']}")
    print(f"No data: {results['NO_DATA']}")
    print(f"No transactions: {results['NO_TRANSACTIONS']}")
    print(f"Failed: {results['FAILED']}")
    print(f"Errors: {results['ERROR']}")
    print(f"Total transactions: {total_transactions:,}")
    print(f"Total signals calculated: {total_signals}")
    print(f"Duration: {duration/60:.1f} minutes")
    
    # Show high score symbols
    if high_score_symbols:
        print("\n" + "=" * 60)
        print("HIGH INSIDER SCORE SYMBOLS (>=70)")
        print("=" * 60)
        high_score_symbols.sort(key=lambda x: x[1], reverse=True)
        for symbol, score in high_score_symbols[:20]:
            print(f"  {symbol}: {score:.0f}")
    
    log_script_end(logger, "fetch_insider_transactions", True, duration, {
        "symbols_processed": len(symbols),
        "transactions": total_transactions,
        "high_score_count": len(high_score_symbols)
    })
    
    # Exit code based on success rate
    success_rate = results['SUCCESS'] / len(symbols) if symbols else 0
    if success_rate < 0.3:
        print(f"[WARNING] Low success rate: {success_rate:.1%}")
        sys.exit(1)
    else:
        print(f"[SUCCESS] Success rate: {success_rate:.1%}")

if __name__ == "__main__":
    main()