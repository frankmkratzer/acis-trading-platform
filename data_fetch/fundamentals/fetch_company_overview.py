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
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from sqlalchemy import text
from dotenv import load_dotenv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from colorama import init, Fore, Style

init(autoreset=True)

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.logging_config import setup_logger
from database.db_connection_manager import DatabaseConnectionManager
from data_fetch.base.rate_limiter import AlphaVantageRateLimiter

load_dotenv()

# Initialize centralized utilities
logger = setup_logger("fetch_company_overview")
db_manager = DatabaseConnectionManager()
engine = db_manager.get_engine()
rate_limiter = AlphaVantageRateLimiter.get_instance()

API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
if not API_KEY:
    logger.error("ALPHA_VANTAGE_API_KEY not set")
    sys.exit(1)

# Configuration
AV_URL = "https://www.alphavantage.co/query"
MAX_WORKERS = 20  # Increase workers for premium API
CACHE_DAYS = 30  # Cache overview data for 30 days
CALLS_PER_MIN = 580  # Conservative limit for 600 calls/min API

def rate_limited_request(symbol):
    """Make rate-limited request for company overview"""
    # Use centralized rate limiter
    rate_limiter.wait_if_needed()
    
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
    """Get symbols that need overview data - only fetch if a quarter has passed since latest_quarter"""
    query = text("""
        WITH overview_status AS (
            SELECT 
                su.symbol,
                su.sector,
                su.industry,
                su.market_cap,
                su.overview_fetched_at,
                cfo.fetched_at as detailed_fetched_at,
                cfo.latest_quarter,
                sp.latest_price_date,
                sp.price_count
            FROM symbol_universe su
            LEFT JOIN company_fundamentals_overview cfo ON su.symbol = cfo.symbol
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
              AND su.market_cap >= 2000000000  -- Mid-cap and above only
        )
        SELECT symbol
        FROM overview_status
        WHERE 
            -- Never fetched detailed data
            (detailed_fetched_at IS NULL)
            -- Or core fields missing
            OR (sector IS NULL OR industry IS NULL OR market_cap IS NULL)
            -- INTELLIGENT FETCHING: Only update if more than a quarter has passed since latest_quarter
            OR (latest_quarter IS NOT NULL 
                AND latest_quarter < CURRENT_DATE - INTERVAL '92 days'  -- More than 1 quarter old
                AND latest_price_date > CURRENT_DATE - INTERVAL '7 days')  -- Still actively traded
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
            'forward_pe': safe_float(data.get('ForwardPE')),
            'trailing_pe': safe_float(data.get('TrailingPE')),
            
            # Calculate enterprise value if possible
            # Note: Alpha Vantage doesn't provide EV directly, would need debt data
            'enterprise_value': None,  # Will need to calculate from fundamentals
            
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
    
    # Ensure numeric columns are proper types
    numeric_cols = ['market_cap', 'shares_outstanding', 'pe_ratio', 'peg_ratio', 'book_value',
                   'price_to_book', 'price_to_sales', 'ev_to_revenue', 'ev_to_ebitda',
                   'forward_pe', 'trailing_pe', 'profit_margin', 'operating_margin',
                   'return_on_assets', 'return_on_equity', 'revenue_ttm', 'revenue_per_share',
                   'quarterly_revenue_growth', 'gross_profit_ttm', 'ebitda', 'eps',
                   'quarterly_earnings_growth', 'dividend_per_share', 'dividend_yield',
                   'shares_float', 'beta',
                   'analyst_target_price', 'analyst_rating_strong_buy', 'analyst_rating_buy',
                   'analyst_rating_hold', 'analyst_rating_sell', 'analyst_rating_strong_sell',
                   'enterprise_value']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
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
                    shares_outstanding = t.shares_outstanding,
                    overview_fetched_at = t.overview_fetched_at
                FROM {temp_table} t
                WHERE su.symbol = t.symbol
                RETURNING su.symbol
            """))
            
            updated_count = len(result.fetchall())
            
            # Store detailed metrics in company_fundamentals_overview table
            conn.execute(text(f"""
                INSERT INTO company_fundamentals_overview (
                    symbol, pe_ratio, peg_ratio, book_value, price_to_book, price_to_sales,
                    ev_to_revenue, ev_to_ebitda, forward_pe, trailing_pe,
                    profit_margin, operating_margin, return_on_assets, return_on_equity,
                    revenue_ttm, revenue_per_share, quarterly_revenue_growth, gross_profit_ttm,
                    ebitda, eps, quarterly_earnings_growth,
                    dividend_per_share, dividend_yield, dividend_date, ex_dividend_date,
                    shares_float, beta,
                    analyst_target_price, analyst_rating_strong_buy, analyst_rating_buy,
                    analyst_rating_hold, analyst_rating_sell, analyst_rating_strong_sell,
                    fiscal_year_end, latest_quarter, fetched_at
                )
                SELECT 
                    symbol, pe_ratio, peg_ratio, book_value, price_to_book, price_to_sales,
                    ev_to_revenue, ev_to_ebitda, forward_pe, trailing_pe,
                    profit_margin, operating_margin, return_on_assets, return_on_equity,
                    revenue_ttm, revenue_per_share, quarterly_revenue_growth, gross_profit_ttm,
                    ebitda, eps, quarterly_earnings_growth,
                    dividend_per_share, dividend_yield, 
                    CASE WHEN dividend_date ~ '^\d{{4}}-\d{{2}}-\d{{2}}$' THEN dividend_date::DATE ELSE NULL END,
                    CASE WHEN ex_dividend_date ~ '^\d{{4}}-\d{{2}}-\d{{2}}$' THEN ex_dividend_date::DATE ELSE NULL END,
                    shares_float, beta,
                    analyst_target_price, analyst_rating_strong_buy, analyst_rating_buy,
                    analyst_rating_hold, analyst_rating_sell, analyst_rating_strong_sell,
                    fiscal_year_end, 
                    CASE WHEN latest_quarter ~ '^\d{{4}}-\d{{2}}-\d{{2}}$' THEN latest_quarter::DATE ELSE NULL END,
                    overview_fetched_at
                FROM {temp_table}
                ON CONFLICT (symbol) DO UPDATE SET
                    pe_ratio = EXCLUDED.pe_ratio,
                    peg_ratio = EXCLUDED.peg_ratio,
                    book_value = EXCLUDED.book_value,
                    price_to_book = EXCLUDED.price_to_book,
                    price_to_sales = EXCLUDED.price_to_sales,
                    ev_to_revenue = EXCLUDED.ev_to_revenue,
                    ev_to_ebitda = EXCLUDED.ev_to_ebitda,
                    forward_pe = EXCLUDED.forward_pe,
                    trailing_pe = EXCLUDED.trailing_pe,
                    profit_margin = EXCLUDED.profit_margin,
                    operating_margin = EXCLUDED.operating_margin,
                    return_on_assets = EXCLUDED.return_on_assets,
                    return_on_equity = EXCLUDED.return_on_equity,
                    revenue_ttm = EXCLUDED.revenue_ttm,
                    revenue_per_share = EXCLUDED.revenue_per_share,
                    quarterly_revenue_growth = EXCLUDED.quarterly_revenue_growth,
                    gross_profit_ttm = EXCLUDED.gross_profit_ttm,
                    ebitda = EXCLUDED.ebitda,
                    eps = EXCLUDED.eps,
                    quarterly_earnings_growth = EXCLUDED.quarterly_earnings_growth,
                    dividend_per_share = EXCLUDED.dividend_per_share,
                    dividend_yield = EXCLUDED.dividend_yield,
                    dividend_date = EXCLUDED.dividend_date,
                    ex_dividend_date = EXCLUDED.ex_dividend_date,
                    shares_float = EXCLUDED.shares_float,
                    beta = EXCLUDED.beta,
                    analyst_target_price = EXCLUDED.analyst_target_price,
                    analyst_rating_strong_buy = EXCLUDED.analyst_rating_strong_buy,
                    analyst_rating_buy = EXCLUDED.analyst_rating_buy,
                    analyst_rating_hold = EXCLUDED.analyst_rating_hold,
                    analyst_rating_sell = EXCLUDED.analyst_rating_sell,
                    analyst_rating_strong_sell = EXCLUDED.analyst_rating_strong_sell,
                    fiscal_year_end = EXCLUDED.fiscal_year_end,
                    latest_quarter = EXCLUDED.latest_quarter,
                    fetched_at = EXCLUDED.fetched_at
            """))
            
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
    
    logger.info("\n" + "="*60)
    logger.info("="*60)
    logger.info("COMPANY OVERVIEW FETCHER")
    logger.info("="*60)
    
    # Get symbols needing update
    logger.info("Checking for symbols needing overview data...")
    logger.info("INTELLIGENT FETCHING: Only updating symbols with latest_quarter > 92 days old")
    symbols = get_symbols_needing_update()
    
    if not symbols:
        logger.info("All symbols have recent overview data (within last quarter)!")
        return 0
    
    logger.info(f"Found {len(symbols)} symbols needing updates (>92 days old or never fetched)")
    logger.info(f"Using premium API: {CALLS_PER_MIN} calls/min with {MAX_WORKERS} workers")
    logger.info(f"Estimated time: {len(symbols) / CALLS_PER_MIN:.1f} minutes")
    
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
    logger.info("\n" + "="*60)
    logger.info("="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"Duration: {duration/60:.1f} minutes ({duration:.0f} seconds)")
    logger.info(f"Symbols requested: {len(symbols)}")
    logger.info(f"Successfully fetched: {successful}")
    logger.info(f"Failed/skipped: {failed}")
    logger.info(f"Actual rate: {successful/duration*60:.0f} symbols/min")
    logger.info(f"API efficiency: {successful/len(symbols)*100:.1f}%")
    
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
        
        logger.info("\nTop Sectors:")
        for row in result:
            logger.info(f"  {row[0]}: {row[1]} companies")
    
    logger.info(f"Completed: {len(symbols)} symbols in {duration:.1f}s")
    
    return 0

if __name__ == "__main__":
    import threading
    sys.exit(main())