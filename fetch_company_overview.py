#!/usr/bin/env python3
"""
Company Overview Fetcher
Fetches sector, industry, market cap, and other fundamental data
Optimized for API efficiency with caching and incremental updates
"""

import os
import sys
import time
import json
import threading
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging
from colorama import init, Fore, Style

init(autoreset=True)

load_dotenv()
engine = create_engine(os.getenv("POSTGRES_URL"))
API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

if not API_KEY:
    print(f"{Fore.RED}[ERROR] ALPHA_VANTAGE_API_KEY not set{Style.RESET_ALL}")
    sys.exit(1)

# Configuration
AV_URL = "https://www.alphavantage.co/query"
MAX_WORKERS = 20  # Increase workers for premium API
CALLS_PER_MIN = 595  # Premium tier (slightly under 600 for safety)
MIN_INTERVAL = 60.0 / CALLS_PER_MIN
CACHE_DAYS = 30  # Cache overview data for 30 days

# Logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/fetch_company_overview.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Rate limiter
last_call_time = 0
rate_limit_lock = threading.Lock()

def rate_limited_request(symbol):
    """Make rate-limited request for company overview"""
    global last_call_time
    
    with rate_limit_lock:
        current_time = time.time()
        time_since_last = current_time - last_call_time
        if time_since_last < MIN_INTERVAL:
            time.sleep(MIN_INTERVAL - time_since_last)
        last_call_time = time.time()
    
    params = {
        'function': 'OVERVIEW',
        'symbol': symbol,
        'apikey': API_KEY
    }
    
    try:
        response = requests.get(AV_URL, params=params, timeout=30)
        if response.status_code != 200:
            logger.warning(f"HTTP {response.status_code} for {symbol}")
            return None
            
        data = response.json()
        
        # Check for rate limit
        if "Note" in data or "Information" in data:
            logger.warning(f"Rate limit hit for {symbol}")
            time.sleep(60)  # Wait a minute
            return None
        
        # Check if valid data returned
        if not data or "Symbol" not in data:
            logger.debug(f"No overview data for {symbol}")
            return None
            
        return data
        
    except Exception as e:
        logger.error(f"Error fetching {symbol}: {e}")
        return None

def get_symbols_needing_update():
    """Get symbols that need overview data or haven't been updated recently"""
    query = text("""
        WITH overview_status AS (
            SELECT 
                su.symbol,
                su.sector,
                su.industry,
                su.market_cap,
                su.overview_fetched_at,
                sp.latest_price_date,
                sp.price_count
            FROM symbol_universe su
            LEFT JOIN (
                SELECT 
                    symbol,
                    MAX(trade_date) as latest_price_date,
                    COUNT(*) as price_count
                FROM stock_prices
                GROUP BY symbol
            ) sp ON su.symbol = sp.symbol
            WHERE su.is_etf = FALSE 
              AND su.country = 'USA'
              AND su.security_type = 'Common Stock'
              AND su.delisted_date IS NULL
        )
        SELECT symbol
        FROM overview_status
        WHERE 
            -- Never fetched
            (overview_fetched_at IS NULL)
            -- Or core fields missing
            OR (sector IS NULL OR industry IS NULL OR market_cap IS NULL)
            -- Or data is stale (30+ days old) for actively traded stocks
            OR (overview_fetched_at < CURRENT_TIMESTAMP - INTERVAL '30 days' 
                AND latest_price_date > CURRENT_DATE - INTERVAL '7 days')
        ORDER BY 
            -- Prioritize stocks with recent price data
            CASE WHEN latest_price_date > CURRENT_DATE - INTERVAL '7 days' THEN 0 ELSE 1 END,
            -- Then by price record count (more active stocks first)
            price_count DESC NULLS LAST,
            symbol
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query)
        return [row[0] for row in result.fetchall()]

def parse_overview_data(data, symbol):
    """Parse company overview data into database format"""
    if not data:
        return None
    
    try:
        # Parse numeric fields safely
        def safe_float(value, default=None):
            try:
                if value and value not in ['None', 'N/A', '-']:
                    # Remove commas and convert
                    return float(str(value).replace(',', ''))
            except:
                pass
            return default
        
        def safe_int(value, default=None):
            try:
                if value and value not in ['None', 'N/A', '-']:
                    return int(float(str(value).replace(',', '')))
            except:
                pass
            return default
        
        # Extract key fields
        overview = {
            'symbol': symbol,
            'sector': data.get('Sector') or None,
            'industry': data.get('Industry') or None,
            'market_cap': safe_float(data.get('MarketCapitalization')),
            'description': data.get('Description', '')[:1000],  # Truncate long descriptions
            
            # Valuation metrics
            'pe_ratio': safe_float(data.get('PERatio')),
            'peg_ratio': safe_float(data.get('PEGRatio')),
            'book_value': safe_float(data.get('BookValue')),
            'price_to_book': safe_float(data.get('PriceToBookRatio')),
            'price_to_sales': safe_float(data.get('PriceToSalesRatioTTM')),
            'ev_to_revenue': safe_float(data.get('EVToRevenue')),
            'ev_to_ebitda': safe_float(data.get('EVToEBITDA')),
            
            # Profitability
            'profit_margin': safe_float(data.get('ProfitMargin')),
            'operating_margin': safe_float(data.get('OperatingMarginTTM')),
            'return_on_assets': safe_float(data.get('ReturnOnAssetsTTM')),
            'return_on_equity': safe_float(data.get('ReturnOnEquityTTM')),
            
            # Financials
            'revenue_ttm': safe_float(data.get('RevenueTTM')),
            'revenue_per_share': safe_float(data.get('RevenuePerShareTTM')),
            'quarterly_revenue_growth': safe_float(data.get('QuarterlyRevenueGrowthYOY')),
            'gross_profit_ttm': safe_float(data.get('GrossProfitTTM')),
            'ebitda': safe_float(data.get('EBITDA')),
            'eps': safe_float(data.get('EPS')),
            'quarterly_earnings_growth': safe_float(data.get('QuarterlyEarningsGrowthYOY')),
            
            # Dividends
            'dividend_per_share': safe_float(data.get('DividendPerShare')),
            'dividend_yield': safe_float(data.get('DividendYield')),
            'dividend_date': data.get('DividendDate') if data.get('DividendDate') not in ['None', 'N/A', '-'] else None,
            'ex_dividend_date': data.get('ExDividendDate') if data.get('ExDividendDate') not in ['None', 'N/A', '-'] else None,
            
            # Other metrics
            'beta': safe_float(data.get('Beta')),
            'shares_outstanding': safe_int(data.get('SharesOutstanding')),
            'shares_float': safe_int(data.get('SharesFloat')),
            'shares_short': safe_int(data.get('SharesShort')),
            'short_ratio': safe_float(data.get('ShortRatio')),
            'forward_pe': safe_float(data.get('ForwardPE')),
            'trailing_pe': safe_float(data.get('TrailingPE')),
            
            # Analyst data
            'analyst_target_price': safe_float(data.get('AnalystTargetPrice')),
            'analyst_rating_strong_buy': safe_int(data.get('AnalystRatingStrongBuy')),
            'analyst_rating_buy': safe_int(data.get('AnalystRatingBuy')),
            'analyst_rating_hold': safe_int(data.get('AnalystRatingHold')),
            'analyst_rating_sell': safe_int(data.get('AnalystRatingSell')),
            'analyst_rating_strong_sell': safe_int(data.get('AnalystRatingStrongSell')),
            
            # Dates
            'fiscal_year_end': data.get('FiscalYearEnd'),
            'latest_quarter': data.get('LatestQuarter'),
            
            'overview_fetched_at': datetime.now(timezone.utc)
        }
        
        return overview
        
    except Exception as e:
        logger.error(f"Error parsing overview for {symbol}: {e}")
        return None

def update_company_overview(overviews):
    """Bulk update company overview data"""
    if not overviews:
        return 0
    
    df = pd.DataFrame(overviews)
    
    try:
        with engine.begin() as conn:
            # Create temp table
            temp_table = f"temp_overview_{int(time.time())}"
            df.to_sql(temp_table, conn, if_exists='replace', index=False, method='multi')
            
            # Update symbol_universe with core fields
            result = conn.execute(text(f"""
                UPDATE symbol_universe su
                SET 
                    sector = t.sector,
                    industry = t.industry,
                    market_cap = t.market_cap,
                    overview_fetched_at = t.overview_fetched_at
                FROM {temp_table} t
                WHERE su.symbol = t.symbol
                RETURNING su.symbol
            """))
            
            updated_count = len(result.fetchall())
            
            # Store detailed metrics in a separate table (if it exists)
            # You may want to create a company_fundamentals table for this data
            
            # Drop temp table
            conn.execute(text(f"DROP TABLE {temp_table}"))
            
            return updated_count
            
    except Exception as e:
        logger.error(f"Error updating overview data: {e}")
        return 0

def fetch_batch(symbols, pbar):
    """Fetch overview data for a batch of symbols"""
    results = []
    
    for symbol in symbols:
        overview_data = rate_limited_request(symbol)
        
        if overview_data:
            parsed = parse_overview_data(overview_data, symbol)
            if parsed:
                results.append(parsed)
                pbar.set_postfix({'Last': symbol, 'Fetched': len(results)})
        
        pbar.update(1)
    
    return results

def main():
    """Main execution"""
    start_time = time.time()
    
    print("\n" + "="*60)
    print(f"{Fore.CYAN}COMPANY OVERVIEW FETCHER{Style.RESET_ALL}")
    print("="*60)
    
    # Get symbols needing update
    print(f"{Fore.YELLOW}[INFO] Checking for symbols needing overview data...{Style.RESET_ALL}")
    symbols = get_symbols_needing_update()
    
    if not symbols:
        print(f"{Fore.GREEN}[INFO] All symbols have recent overview data!{Style.RESET_ALL}")
        return 0
    
    print(f"{Fore.GREEN}[INFO] Found {len(symbols)} symbols needing updates{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}[INFO] Using premium API: {CALLS_PER_MIN} calls/min with {MAX_WORKERS} workers{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}[INFO] Estimated time: {len(symbols) / CALLS_PER_MIN:.1f} minutes{Style.RESET_ALL}")
    
    # Process with progress bar - optimized for high throughput
    all_overviews = []
    batch_size = 100  # Larger batches for faster processing
    
    with tqdm(total=len(symbols), desc="Fetching overviews", unit="symbols",
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        
        # Track success/failure
        successful = 0
        failed = 0
        
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            results = fetch_batch(batch, pbar)
            all_overviews.extend(results)
            successful += len(results)
            failed += len(batch) - len(results)
            
            # Update progress bar stats
            pbar.set_postfix({
                'Success': successful,
                'Failed': failed,
                'Rate': f'{successful/(time.time()-start_time)*60:.0f}/min'
            })
            
            # Periodic database update - larger batches for efficiency
            if len(all_overviews) >= 500:
                updated = update_company_overview(all_overviews)
                logger.info(f"Updated {updated} companies")
                all_overviews = []
    
    # Final update
    if all_overviews:
        updated = update_company_overview(all_overviews)
        logger.info(f"Final update of {updated} companies")
    
    # Summary
    duration = time.time() - start_time
    print("\n" + "="*60)
    print(f"{Fore.GREEN}SUMMARY{Style.RESET_ALL}")
    print("="*60)
    print(f"Duration: {duration/60:.1f} minutes ({duration:.0f} seconds)")
    print(f"Symbols requested: {len(symbols)}")
    print(f"Successfully fetched: {Fore.GREEN}{successful}{Style.RESET_ALL}")
    print(f"Failed/skipped: {Fore.RED}{failed}{Style.RESET_ALL}")
    print(f"Actual rate: {Fore.YELLOW}{successful/duration*60:.0f} symbols/min{Style.RESET_ALL}")
    print(f"API efficiency: {Fore.CYAN}{successful/len(symbols)*100:.1f}%{Style.RESET_ALL}")
    
    # Show sample of updated data
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT sector, COUNT(*) as count
            FROM symbol_universe
            WHERE sector IS NOT NULL
            GROUP BY sector
            ORDER BY count DESC
            LIMIT 10
        """))
        
        print(f"\n{Fore.CYAN}Top Sectors:{Style.RESET_ALL}")
        for row in result:
            print(f"  {row[0]}: {row[1]} companies")
    
    logger.info(f"Completed: {len(symbols)} symbols in {duration:.1f}s")
    
    return 0

if __name__ == "__main__":
    import threading
    sys.exit(main())