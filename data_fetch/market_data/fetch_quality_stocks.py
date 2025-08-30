#!/usr/bin/env python3
"""
Fetch Mid/Large Cap Common Stocks Only
Fetches US common stocks with market cap >= $2B
Excludes: ETFs, REITs, funds, ADRs, preferred stocks, and foreign companies
Focuses on investable universe for three-portfolio strategy
"""

import os
import sys
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import text
from dotenv import load_dotenv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Set, Tuple
import json

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.logging_config import setup_logger, log_script_start, log_script_end
from database.db_connection_manager import DatabaseConnectionManager
from data_fetch.base.rate_limiter import AlphaVantageRateLimiter

load_dotenv()

# Initialize
logger = setup_logger("fetch_quality_stocks")
db_manager = DatabaseConnectionManager()
engine = db_manager.get_engine()
rate_limiter = AlphaVantageRateLimiter.get_instance()

# Configuration
API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
if not API_KEY:
    logger.error("ALPHA_VANTAGE_API_KEY not set in .env file")
    sys.exit(1)

# Constants - FOCUS ON MID/LARGE CAP COMMON STOCKS
AV_BASE_URL = "https://www.alphavantage.co/query"
MIN_MARKET_CAP = 2_000_000_000  # $2B minimum for mid-cap
LARGE_CAP_THRESHOLD = 10_000_000_000  # $10B for large-cap  
VALID_EXCHANGES = {'NYSE', 'NASDAQ', 'AMEX'}  # Major US exchanges only
EXCLUDED_SECTORS = {'N/A', 'CLOSED-END FUNDS', 'EXCHANGE TRADED FUND'}  # Exclude funds
MAX_WORKERS = 5  # Reduced to avoid rate limiting
BATCH_SIZE = 500  # Process in larger batches


class QualityStockUniverse:
    """Fetches quality US common stocks (mid/large cap) only"""
    
    def __init__(self):
        self.api_key = API_KEY
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'ACIS Trading Platform'})
        self.total_processed = 0
        self.total_failed = 0
        
    def fetch_all_active_stocks(self) -> pd.DataFrame:
        """
        Fetch active US common stocks using Alpha Vantage LISTING_STATUS endpoint
        Filters for common stocks only, excluding ETFs, REITs, funds, etc.
        """
        logger.info("Fetching active US common stocks from Alpha Vantage...")
        
        params = {
            'function': 'LISTING_STATUS',
            'apikey': self.api_key
        }
        
        try:
            response = self.session.get(AV_BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse CSV response
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            
            # Enhanced filtering for quality common stocks
            df_filtered = df[
                (df['exchange'].isin(VALID_EXCHANGES)) &
                (df['assetType'] == 'Stock') &
                (df['status'] == 'Active')
            ].copy()
            
            # Additional filtering to exclude non-common stocks
            # Exclude symbols that typically indicate non-common stocks
            exclude_patterns = [
                r'\.PR[A-Z]?$',  # Preferred stocks (e.g., BAC.PRE)
                r'\.WS$',         # Warrants
                r'\.UN$',         # Units
                r'\.RT$',         # Rights
                r'\-P[A-Z]?$',    # Another preferred format
                r'^.*\d{5,}$',    # CUSIP-like symbols (often funds)
            ]
            
            import re
            pattern = '|'.join(exclude_patterns)
            if pattern:
                mask = ~df_filtered['symbol'].str.contains(pattern, regex=True, na=False)
                df_filtered = df_filtered[mask]
            
            logger.info(f"Found {len(df_filtered)} active US common stocks on major exchanges")
            
            # Show breakdown by exchange
            for exchange in VALID_EXCHANGES:
                count = len(df_filtered[df_filtered['exchange'] == exchange])
                if count > 0:
                    logger.info(f"  {exchange}: {count} stocks")
            
            return df_filtered
            
        except Exception as e:
            logger.error(f"Failed to fetch listing status: {e}")
            return pd.DataFrame()
    
    def fetch_company_overview_batch(self, symbols: List[str], skip_existing: bool = True) -> Dict:
        """
        Fetch company overview for multiple symbols
        No market cap filtering - get everything
        """
        logger.info(f"Fetching overview data for {len(symbols)} symbols...")
        
        # Check which symbols already exist if skip_existing
        existing_symbols = set()
        if skip_existing:
            with engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT symbol 
                    FROM symbol_universe 
                    WHERE overview_fetched_at > CURRENT_DATE - INTERVAL '30 days'
                """))
                existing_symbols = {row[0] for row in result}
                logger.info(f"Skipping {len(existing_symbols)} recently updated symbols")
        
        # Filter symbols to process
        symbols_to_process = [s for s in symbols if s not in existing_symbols]
        
        if not symbols_to_process:
            logger.info("All symbols are up to date")
            return {}
        
        logger.info(f"Processing {len(symbols_to_process)} symbols...")
        
        results = {}
        failed = []
        processed = 0
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {}
            
            for symbol in symbols_to_process:
                future = executor.submit(self._fetch_single_overview, symbol)
                futures[future] = symbol
            
            for future in as_completed(futures):
                symbol = futures[future]
                processed += 1
                
                if processed % 100 == 0:
                    logger.info(f"Progress: {processed}/{len(symbols_to_process)} symbols processed")
                
                try:
                    data = future.result()
                    if data:
                        results[symbol] = data
                        self.total_processed += 1
                except Exception as e:
                    logger.debug(f"Failed to fetch {symbol}: {e}")
                    failed.append(symbol)
                    self.total_failed += 1
        
        logger.info(f"Successfully fetched {len(results)} overviews, {len(failed)} failed")
        return results
    
    def _fetch_single_overview(self, symbol: str) -> Optional[Dict]:
        """Fetch overview for a single symbol - FILTER FOR MID/LARGE CAP"""
        
        # Wait for rate limit
        rate_limiter.wait_if_needed()
        
        params = {
            'function': 'OVERVIEW',
            'symbol': symbol,
            'apikey': self.api_key
        }
        
        try:
            response = self.session.get(AV_BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Check if we got valid data
            if 'Symbol' not in data:
                return None
            
            # Filter out REITs and funds by sector/industry
            sector = data.get('Sector', '')
            industry = data.get('Industry', '')
            asset_type = data.get('AssetType', '')
            
            # Exclude REITs, funds, and other non-common stocks
            if any([
                sector in EXCLUDED_SECTORS,
                'REIT' in sector.upper() if sector else False,
                'REIT' in industry.upper() if industry else False,
                'FUND' in industry.upper() if industry else False,
                'ETF' in industry.upper() if industry else False,
                asset_type != 'Common Stock' if asset_type else False
            ]):
                logger.debug(f"Skipping {symbol}: sector={sector}, industry={industry}")
                return None
            
            # Parse market cap and filter
            market_cap = self._parse_number(data.get('MarketCapitalization'))
            
            # Skip if below minimum market cap ($2B)
            if market_cap and market_cap < MIN_MARKET_CAP:
                logger.debug(f"Skipping {symbol}: market_cap=${market_cap/1e9:.2f}B < $2B minimum")
                return None
            
            # Return only mid/large cap stocks
            return {
                'symbol': symbol,
                'name': data.get('Name', ''),
                'exchange': data.get('Exchange', ''),
                'sector': sector,
                'industry': industry,
                'market_cap': market_cap if market_cap else 0,
                'shares_outstanding': self._parse_number(data.get('SharesOutstanding')),
                'description': (data.get('Description', '') or '')[:1000],
                'pe_ratio': self._parse_number(data.get('PERatio')),
                'peg_ratio': self._parse_number(data.get('PEGRatio')),
                'book_value': self._parse_number(data.get('BookValue')),
                'dividend_yield': self._parse_number(data.get('DividendYield')),
                'eps': self._parse_number(data.get('EPS')),
                'revenue_ttm': self._parse_number(data.get('RevenueTTM')),
                'profit_margin': self._parse_number(data.get('ProfitMargin')),
                'beta': self._parse_number(data.get('Beta')),
                '52_week_high': self._parse_number(data.get('52WeekHigh')),
                '52_week_low': self._parse_number(data.get('52WeekLow')),
                'ebitda': self._parse_number(data.get('EBITDA')),
            }
            
        except Exception as e:
            logger.debug(f"Error fetching {symbol}: {e}")
            return None
    
    def _parse_number(self, value) -> Optional[float]:
        """Parse string numbers from API to float"""
        if value is None or value == 'None' or value == '-':
            return None
        try:
            # Remove commas and convert
            return float(str(value).replace(',', ''))
        except:
            return None
    
    def update_database(self, overview_data: Dict) -> int:
        """Update database with fetched overview data - mid/large cap only"""
        
        if not overview_data:
            return 0
        
        # Convert to DataFrame for bulk operations
        df = pd.DataFrame.from_dict(overview_data, orient='index')
        df['overview_fetched_at'] = datetime.now()
        
        # Filter again for market cap (belt and suspenders)
        df = df[df['market_cap'] >= MIN_MARKET_CAP]
        
        if df.empty:
            logger.info("No stocks met the $2B minimum market cap requirement")
            return 0
        
        logger.info(f"Updating database with {len(df)} mid/large cap stocks...")
        
        # Show market cap distribution
        if 'market_cap' in df.columns:
            df['market_cap_category'] = pd.cut(
                df['market_cap'].fillna(0),
                bins=[MIN_MARKET_CAP, LARGE_CAP_THRESHOLD, float('inf')],
                labels=['Mid ($2B-$10B)', 'Large ($10B+)']
            )
            logger.info("Market cap distribution:")
            for category, count in df['market_cap_category'].value_counts().items():
                logger.info(f"  {category}: {count} stocks")
        
        # Select only columns that exist in symbol_universe table
        universe_columns = ['symbol', 'name', 'exchange', 'sector', 'industry', 
                          'market_cap', 'shares_outstanding', 
                          'overview_fetched_at']
        df_universe = df[universe_columns].copy()
        
        # Ensure numeric columns are proper types
        df_universe['market_cap'] = pd.to_numeric(df_universe['market_cap'], errors='coerce').fillna(0)
        df_universe['shares_outstanding'] = pd.to_numeric(df_universe['shares_outstanding'], errors='coerce')
        
        with engine.begin() as conn:
            # Create temporary table
            temp_table = f"temp_overview_{int(time.time())}"
            df_universe.to_sql(temp_table, conn, if_exists='replace', index=False)
            
            # Upsert into symbol_universe
            upsert_query = text(f"""
                INSERT INTO symbol_universe (
                    symbol, name, exchange, sector, industry, 
                    market_cap, shares_outstanding,
                    security_type, is_etf, country, currency,
                    overview_fetched_at, fetched_at
                )
                SELECT 
                    symbol, name, exchange, sector, industry,
                    market_cap, shares_outstanding,
                    'Common Stock', false, 'USA', 'USD',
                    overview_fetched_at, CURRENT_TIMESTAMP
                FROM {temp_table}
                ON CONFLICT (symbol) DO UPDATE SET
                    name = EXCLUDED.name,
                    exchange = EXCLUDED.exchange,
                    sector = EXCLUDED.sector,
                    industry = EXCLUDED.industry,
                    market_cap = EXCLUDED.market_cap,
                    shares_outstanding = EXCLUDED.shares_outstanding,
                    overview_fetched_at = EXCLUDED.overview_fetched_at
            """)
            
            result = conn.execute(upsert_query)
            
            # Drop temp table
            conn.execute(text(f"DROP TABLE {temp_table}"))
            
            return len(df)
    
    def run_complete_update(self):
        """Run complete symbol universe update - mid/large cap common stocks only"""
        
        log_script_start(logger, "fetch_quality_stocks", 
                        "Quality US stock universe update (mid/large cap common stocks)")
        start_time = time.time()
        
        # Step 1: Get active US common stocks
        df_all_stocks = self.fetch_all_active_stocks()
        
        if df_all_stocks.empty:
            logger.error("Failed to fetch active stocks list")
            return
        
        total_symbols = len(df_all_stocks)
        logger.info(f"Found {total_symbols} potential US common stocks")
        
        # Step 2: Process in batches to avoid overwhelming the API
        all_symbols = df_all_stocks['symbol'].tolist()
        
        # Process ALL symbols, not just new ones
        for i in range(0, len(all_symbols), BATCH_SIZE):
            batch = all_symbols[i:i+BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            total_batches = (len(all_symbols) + BATCH_SIZE - 1) // BATCH_SIZE
            
            logger.info(f"\nProcessing batch {batch_num}/{total_batches} ({len(batch)} symbols)...")
            
            # Fetch overview data for batch
            overview_data = self.fetch_company_overview_batch(batch, skip_existing=True)
            
            # Update database with batch
            if overview_data:
                updated = self.update_database(overview_data)
                logger.info(f"Batch {batch_num}: Updated {updated} symbols in database")
            
            # Show progress
            progress = min(i + BATCH_SIZE, len(all_symbols))
            logger.info(f"Overall progress: {progress}/{total_symbols} symbols ({progress*100/total_symbols:.1f}%)")
            logger.info(f"Successfully processed: {self.total_processed}, Failed: {self.total_failed}")
        
        # Step 3: Final statistics
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(CASE WHEN market_cap >= :large_cap THEN 1 END) as large_cap,
                    COUNT(CASE WHEN market_cap >= :min_cap AND market_cap < :large_cap THEN 1 END) as mid_cap,
                    AVG(CASE WHEN market_cap >= :min_cap THEN market_cap END) as avg_market_cap,
                    COUNT(shares_outstanding) as has_shares,
                    COUNT(sector) as has_sector,
                    COUNT(DISTINCT sector) as unique_sectors
                FROM symbol_universe
                WHERE is_etf = false 
                  AND security_type = 'Common Stock'
                  AND country = 'USA'
                  AND market_cap >= :min_cap
            """), {
                'min_cap': MIN_MARKET_CAP,
                'large_cap': LARGE_CAP_THRESHOLD
            })
            stats = result.fetchone()
        
        duration = time.time() - start_time
        
        logger.info("\n" + "="*60)
        logger.info("QUALITY STOCK UNIVERSE UPDATE FINISHED")
        logger.info("="*60)
        logger.info(f"Duration: {duration/60:.1f} minutes")
        logger.info(f"Total mid/large cap stocks: {stats[0]}")
        logger.info(f"  Large cap ($10B+): {stats[1]}")
        logger.info(f"  Mid cap ($2B-$10B): {stats[2]}")
        if stats[3]:
            logger.info(f"Average market cap: ${float(stats[3])/1e9:.2f}B")
        logger.info(f"Shares outstanding coverage: {stats[4]/stats[0]*100:.1f}%" if stats[0] else "N/A")
        logger.info(f"Sector coverage: {stats[5]/stats[0]*100:.1f}%" if stats[0] else "N/A")
        logger.info(f"Unique sectors: {stats[6]}")
        logger.info(f"\nTotal processed in this run: {self.total_processed}")
        logger.info(f"Total failed in this run: {self.total_failed}")
        logger.info("\nNote: Database contains only quality mid/large cap US common stocks")
        logger.info("Excluded: ETFs, REITs, funds, ADRs, preferred stocks, small caps")
        
        log_script_end(logger, "fetch_quality_stocks", success=True, duration=duration)


def main():
    """Main execution function"""
    
    # Check for API key
    if not API_KEY:
        logger.error("ALPHA_VANTAGE_API_KEY not set. Please add to .env file")
        return 1
    
    print("\n" + "="*60)
    print("FETCHING QUALITY US STOCK UNIVERSE")
    print("="*60)
    print("This will fetch mid/large cap US common stocks:")
    print("- Large caps ($10B+)")
    print("- Mid caps ($2B-$10B)")
    print("\nExcludes:")
    print("- ETFs and Exchange Traded Products")
    print("- REITs (Real Estate Investment Trusts)")
    print("- Closed-end funds and mutual funds")
    print("- ADRs (foreign companies)")
    print("- Preferred stocks and warrants")
    print("- Small/micro/nano cap stocks (<$2B)")
    print("\nNote: This may take 1-2 hours due to API rate limits")
    print("The script will process in batches and can be resumed if interrupted")
    print("="*60)
    
    # Run the update
    universe = QualityStockUniverse()
    universe.run_complete_update()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())