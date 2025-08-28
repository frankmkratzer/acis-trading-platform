#!/usr/bin/env python3
"""
Symbol Universe Fetcher - US Stocks Only
Fetches and maintains fresh symbol_universe table with US common stocks
Excludes ETFs, REITs, foreign stocks, funds, warrants, etc.
"""

import os
import sys
import time
import requests
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from logging_config import setup_logger, log_script_start, log_script_end

load_dotenv()
logger = setup_logger("fetch_symbol_universe")

# Configuration
POSTGRES_URL = os.getenv("POSTGRES_URL")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")

if not POSTGRES_URL:
    logger.error("POSTGRES_URL not set")
    sys.exit(1)

engine = create_engine(POSTGRES_URL)

# US exchanges we want
VALID_EXCHANGES = {'NYSE', 'NASDAQ', 'AMEX', 'NYSEMkt'}

# Asset types to exclude
EXCLUDED_ASSET_TYPES = {
    'ETF', 'REIT', 'Fund', 'Trust', 'Partnership', 'LP', 'Warrant', 'Unit',
    'Right', 'Depositary', 'ADR', 'GDR', 'Preferred', 'Note', 'Bond'
}

# Name patterns to exclude (case insensitive)
EXCLUDED_NAME_PATTERNS = [
    'etf', 'fund', 'trust', 'reit', 'warrant', 'depositary', 'receipt',
    'preferred', 'pref', 'note', 'bond', 'unit', 'right', 'lp', 'l.p.',
    'acquisition', 'spac', 'holdings', 'capital corp'
]

def fetch_from_alpha_vantage():
    """Fetch symbol list from Alpha Vantage API"""
    if not ALPHA_VANTAGE_API_KEY:
        logger.warning("ALPHA_VANTAGE_API_KEY not set")
        return None
    
    logger.info("Fetching symbol list from Alpha Vantage...")
    
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'LISTING_STATUS',
            'apikey': ALPHA_VANTAGE_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        
        # Parse CSV response
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
        
        logger.info(f"Alpha Vantage returned {len(df)} symbols")
        
        # Standardize column names
        df.columns = df.columns.str.lower()
        if 'symbol' not in df.columns:
            df.rename(columns={'ticker': 'symbol'}, inplace=True)
        
        return df
        
    except Exception as e:
        logger.error(f"Alpha Vantage fetch failed: {e}")
        return None

def fetch_from_fmp():
    """Fetch symbol list from FMP API"""
    if not FMP_API_KEY:
        logger.warning("FMP_API_KEY not set")
        return None
    
    logger.info("Fetching symbol list from FMP...")
    
    try:
        url = "https://financialmodelingprep.com/api/v3/stock/list"
        params = {'apikey': FMP_API_KEY}
        
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        
        data = response.json()
        df = pd.DataFrame(data)
        
        logger.info(f"FMP returned {len(df)} symbols")
        
        # Standardize column names
        df.columns = df.columns.str.lower()
        if 'name' not in df.columns and 'companyname' in df.columns:
            df.rename(columns={'companyname': 'name'}, inplace=True)
        
        return df
        
    except Exception as e:
        logger.error(f"FMP fetch failed: {e}")
        return None

def is_valid_us_stock(row):
    """Check if a symbol represents a valid US common stock"""
    symbol = str(row.get('symbol', '')).strip().upper()
    name = str(row.get('name', '')).strip().lower()
    exchange = str(row.get('exchange', '')).strip().upper()
    asset_type = str(row.get('assettype', '')).strip()
    
    # Basic validation - check for null, nan, or empty symbols
    if not symbol or symbol == 'NAN' or symbol == 'NONE' or len(symbol) > 5 or not symbol.isalpha():
        return False
    
    # Check exchange
    if exchange not in VALID_EXCHANGES:
        return False
    
    # Check asset type
    if asset_type in EXCLUDED_ASSET_TYPES:
        return False
    
    # Check name patterns
    for pattern in EXCLUDED_NAME_PATTERNS:
        if pattern in name:
            return False
    
    # Additional checks for common non-stock patterns
    if any(x in symbol for x in ['.', '-', '_', '/']):
        return False
    
    # Check for warrant/preferred suffixes
    if any(symbol.endswith(x) for x in ['W', 'WT', 'WS', 'PR', 'P']):
        return False
    
    return True

def clean_and_filter_symbols(df):
    """Clean and filter symbol dataframe"""
    if df is None or df.empty:
        return pd.DataFrame()
    
    logger.info(f"Cleaning and filtering {len(df)} raw symbols...")
    
    # Apply filters
    filtered_df = df[df.apply(is_valid_us_stock, axis=1)].copy()
    
    # Standardize data
    filtered_df['symbol'] = filtered_df['symbol'].str.upper().str.strip()
    filtered_df['exchange'] = filtered_df['exchange'].str.upper().str.strip()
    
    # Remove duplicates
    filtered_df = filtered_df.drop_duplicates(subset=['symbol']).reset_index(drop=True)
    
    logger.info(f"After filtering: {len(filtered_df)} valid US common stocks")
    
    return filtered_df

def update_symbol_universe(df):
    """Update symbol_universe table with fresh data"""
    if df.empty:
        logger.error("No symbols to update")
        return False
    
    logger.info(f"Updating symbol_universe with {len(df)} symbols...")
    
    try:
        with engine.begin() as conn:
            # Prepare data for upsert
            symbols_data = []
            for _, row in df.iterrows():
                symbol = str(row.get('symbol', '')).strip()
                # Skip rows without valid symbols
                if not symbol:
                    logger.warning(f"Skipping row with empty symbol: {row.get('name', 'Unknown')}")
                    continue
                
                # Parse dates
                delisted_date = None
                if pd.notna(row.get('delistingdate')):
                    try:
                        delisted_date = pd.to_datetime(row.get('delistingdate')).date()
                    except:
                        pass
                
                # Determine security type from assetType
                asset_type = str(row.get('assettype', 'Stock')).strip()
                security_type = 'Common Stock' if asset_type == 'Stock' else asset_type
                    
                symbols_data.append({
                    'symbol': symbol,
                    'name': str(row.get('name', '')).strip()[:200] or 'Unknown Company',
                    'exchange': row.get('exchange', 'UNKNOWN'),
                    'security_type': security_type,
                    'is_etf': 'ETF' in asset_type,
                    'country': 'USA',
                    'currency': 'USD',
                    'delisted_date': delisted_date
                })
            
            # Create temp table for bulk upsert
            temp_table = f"temp_symbol_universe_{int(time.time())}"
            
            df_temp = pd.DataFrame(symbols_data)
            # Ensure date columns are properly typed
            if 'delisted_date' in df_temp.columns:
                df_temp['delisted_date'] = pd.to_datetime(df_temp['delisted_date'], errors='coerce')
            
            df_temp.to_sql(temp_table, conn, if_exists='replace', index=False, method='multi')
            
            # Ultra-fast upsert with all fields
            result = conn.execute(text(f"""
                INSERT INTO symbol_universe (
                    symbol, name, exchange, security_type, is_etf, country, currency, 
                    delisted_date, fetched_at
                )
                SELECT 
                    symbol, name, exchange, security_type, is_etf, country, currency,
                    delisted_date::date, CURRENT_TIMESTAMP
                FROM {temp_table}
                ON CONFLICT (symbol) DO UPDATE SET
                    name = EXCLUDED.name,
                    exchange = EXCLUDED.exchange,
                    security_type = EXCLUDED.security_type,
                    is_etf = EXCLUDED.is_etf,
                    delisted_date = EXCLUDED.delisted_date,
                    fetched_at = CURRENT_TIMESTAMP
                RETURNING symbol
            """))
            
            upserted_count = len(result.fetchall())
            
            # Note: Skipping inactive marking since is_active column doesn't exist
            # This would normally mark symbols not in our fresh list as inactive
            
            # Drop temp table
            conn.execute(text(f"DROP TABLE {temp_table}"))
            
            # Get final counts
            result = conn.execute(text("""
                SELECT 
                    COUNT(*) as total_count
                FROM symbol_universe 
                WHERE is_etf = FALSE AND country = 'USA'
            """))
            
            counts = result.fetchone()
            
            logger.info(f"Symbol universe updated:")
            logger.info(f"  Upserted: {upserted_count} symbols")
            logger.info(f"  Total US stocks: {counts[0]}")
            
            return True
            
    except Exception as e:
        logger.error(f"Symbol universe update failed: {e}")
        return False

def main():
    """Main execution function"""
    start_time = time.time()
    log_script_start(logger, "fetch_symbol_universe", "Fetch and maintain US stock symbol universe")
    
    try:
        # Try to fetch from multiple sources
        df = None
        
        # Try Alpha Vantage first
        if ALPHA_VANTAGE_API_KEY:
            df = fetch_from_alpha_vantage()
        
        # Fallback to FMP if Alpha Vantage fails
        if df is None and FMP_API_KEY:
            df = fetch_from_fmp()
        
        if df is None:
            logger.error("Failed to fetch symbols from any source")
            log_script_end(logger, "fetch_symbol_universe", False, time.time() - start_time)
            sys.exit(1)
        
        # Clean and filter
        clean_df = clean_and_filter_symbols(df)
        
        if clean_df.empty:
            logger.error("No valid symbols after filtering")
            log_script_end(logger, "fetch_symbol_universe", False, time.time() - start_time)
            sys.exit(1)
        
        # Update database
        success = update_symbol_universe(clean_df)
        
        duration = time.time() - start_time
        if success:
            log_script_end(logger, "fetch_symbol_universe", True, duration, {
                "Symbols processed": len(clean_df),
                "Data source": "Alpha Vantage" if ALPHA_VANTAGE_API_KEY and df is not None else "FMP",
                "Status": "Symbol universe updated"
            })
        else:
            log_script_end(logger, "fetch_symbol_universe", False, duration)
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Symbol universe fetch failed: {e}")
        log_script_end(logger, "fetch_symbol_universe", False, time.time() - start_time)
        sys.exit(1)

if __name__ == "__main__":
    main()