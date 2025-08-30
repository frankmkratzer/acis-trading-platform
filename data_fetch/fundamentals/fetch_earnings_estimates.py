"""
Fetch earnings estimates from Alpha Vantage EARNINGS_ESTIMATES endpoint.

This script fetches forward-looking earnings estimates including:
- Quarterly and annual EPS estimates
- Revenue estimates
- Number of analysts
- Surprise history
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
import requests
import time
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
from tqdm import tqdm
import json

# Setup logging first
from utils.logging_config import setup_logger, log_script_start, log_script_end
logger = setup_logger("fetch_earnings_estimates")

# Database connection
from database.db_connection_manager import DatabaseConnectionManager

# Rate limiting
from data_fetch.base.rate_limiter import AlphaVantageRateLimiter

# Fetch management
from data_fetch.base.fetch_manager import IncrementalFetchManager

# Load environment variables
load_dotenv()
API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')

def get_symbols_to_fetch(engine):
    """Get symbols that need earnings estimates fetched."""
    
    query = """
    WITH last_fetch AS (
        SELECT 
            symbol,
            MAX(fetched_at) as last_fetched
        FROM earnings_estimates
        GROUP BY symbol
    )
    SELECT 
        su.symbol
    FROM symbol_universe su
    LEFT JOIN last_fetch lf ON su.symbol = lf.symbol
    WHERE su.market_cap >= 2e9  -- Mid-cap and above
      AND su.country = 'USA'
      AND su.security_type = 'Common Stock'
      AND (
          lf.last_fetched IS NULL  -- Never fetched
          OR lf.last_fetched < CURRENT_DATE - INTERVAL '7 days'  -- Older than a week
      )
    ORDER BY su.market_cap DESC
    """
    
    df = pd.read_sql(query, engine)
    return df['symbol'].tolist()

def fetch_earnings_estimates(symbol, rate_limiter):
    """Fetch earnings estimates for a single symbol."""
    
    try:
        # Wait for rate limit
        rate_limiter.wait_if_needed()
        
        url = 'https://www.alphavantage.co/query'
        params = {
            'function': 'EARNINGS_ESTIMATES',
            'symbol': symbol,
            'apikey': API_KEY
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Check for API errors
        if 'Error Message' in data:
            logger.warning(f"API error for {symbol}: {data['Error Message']}")
            return None
        
        if 'Note' in data:
            logger.warning(f"API limit message: {data['Note']}")
            rate_limiter.hit_limit()
            return None
            
        # Extract earnings estimates
        if 'quarterlyEarningsEstimates' not in data and 'annualEarningsEstimates' not in data:
            logger.debug(f"No earnings estimates for {symbol}")
            return None
            
        return data
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error for {symbol}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error for {symbol}: {e}")
        return None

def parse_earnings_estimates(symbol, data):
    """Parse earnings estimates response into database records."""
    
    records = []
    
    # Parse quarterly estimates
    if 'quarterlyEarningsEstimates' in data:
        for estimate in data['quarterlyEarningsEstimates']:
            try:
                record = {
                    'symbol': symbol,
                    'fiscal_period': estimate.get('fiscalDateEnding'),
                    'period_type': 'quarterly',
                    
                    # EPS estimates
                    'eps_consensus': float(estimate.get('consensusEPS')) if estimate.get('consensusEPS') else None,
                    'eps_high': float(estimate.get('highEstimate')) if estimate.get('highEstimate') else None,
                    'eps_low': float(estimate.get('lowEstimate')) if estimate.get('lowEstimate') else None,
                    'eps_num_estimates': int(estimate.get('numberOfEstimates')) if estimate.get('numberOfEstimates') else None,
                    
                    # Revenue estimates (if available)
                    'revenue_consensus': float(estimate.get('consensusRevenue')) if estimate.get('consensusRevenue') else None,
                    'revenue_high': float(estimate.get('revenueHighEstimate')) if estimate.get('revenueHighEstimate') else None,
                    'revenue_low': float(estimate.get('revenueLowEstimate')) if estimate.get('revenueLowEstimate') else None,
                    
                    # Reported values (for surprise calculation)
                    'reported_eps': float(estimate.get('reportedEPS')) if estimate.get('reportedEPS') else None,
                    'reported_date': estimate.get('reportedDate'),
                    'surprise_amount': float(estimate.get('surpriseAmount')) if estimate.get('surpriseAmount') else None,
                    'surprise_percentage': float(estimate.get('surprisePercentage').replace('%', '')) if estimate.get('surprisePercentage') else None,
                    
                    'fetched_at': datetime.now()
                }
                
                # Only add if we have a valid fiscal period
                if record['fiscal_period']:
                    records.append(record)
                    
            except Exception as e:
                logger.error(f"Error parsing quarterly estimate for {symbol}: {e}")
    
    # Parse annual estimates
    if 'annualEarningsEstimates' in data:
        for estimate in data['annualEarningsEstimates']:
            try:
                record = {
                    'symbol': symbol,
                    'fiscal_period': estimate.get('fiscalDateEnding'),
                    'period_type': 'annual',
                    
                    # EPS estimates
                    'eps_consensus': float(estimate.get('consensusEPS')) if estimate.get('consensusEPS') else None,
                    'eps_high': float(estimate.get('highEstimate')) if estimate.get('highEstimate') else None,
                    'eps_low': float(estimate.get('lowEstimate')) if estimate.get('lowEstimate') else None,
                    'eps_num_estimates': int(estimate.get('numberOfEstimates')) if estimate.get('numberOfEstimates') else None,
                    
                    # Revenue estimates
                    'revenue_consensus': float(estimate.get('consensusRevenue')) if estimate.get('consensusRevenue') else None,
                    'revenue_high': float(estimate.get('revenueHighEstimate')) if estimate.get('revenueHighEstimate') else None,
                    'revenue_low': float(estimate.get('revenueLowEstimate')) if estimate.get('revenueLowEstimate') else None,
                    
                    # Reported values
                    'reported_eps': float(estimate.get('reportedEPS')) if estimate.get('reportedEPS') else None,
                    'reported_date': estimate.get('reportedDate'),
                    'surprise_amount': float(estimate.get('surpriseAmount')) if estimate.get('surpriseAmount') else None,
                    'surprise_percentage': float(estimate.get('surprisePercentage').replace('%', '')) if estimate.get('surprisePercentage') else None,
                    
                    'fetched_at': datetime.now()
                }
                
                if record['fiscal_period']:
                    records.append(record)
                    
            except Exception as e:
                logger.error(f"Error parsing annual estimate for {symbol}: {e}")
    
    return records

def save_estimates_batch(engine, records):
    """Save batch of earnings estimates to database."""
    
    if not records:
        return
    
    df = pd.DataFrame(records)
    
    # Create temporary table
    temp_table = f"temp_earnings_{int(time.time() * 1000)}"
    
    try:
        df.to_sql(temp_table, engine, if_exists='replace', index=False)
        
        # Upsert from temp table
        upsert_query = f"""
            INSERT INTO earnings_estimates (
                symbol, fiscal_period, period_type,
                eps_consensus, eps_high, eps_low, eps_num_estimates,
                revenue_consensus, revenue_high, revenue_low,
                reported_eps, reported_date, surprise_amount, surprise_percentage,
                fetched_at
            )
            SELECT 
                symbol, fiscal_period::date, period_type,
                eps_consensus::numeric, eps_high::numeric, eps_low::numeric, eps_num_estimates::int,
                revenue_consensus::numeric, revenue_high::numeric, revenue_low::numeric,
                reported_eps::numeric, 
                CASE WHEN reported_date IS NOT NULL THEN reported_date::date ELSE NULL END,
                surprise_amount::numeric, surprise_percentage::numeric,
                fetched_at
            FROM {temp_table}
            ON CONFLICT (symbol, fiscal_period, period_type) DO UPDATE SET
                eps_consensus = EXCLUDED.eps_consensus,
                eps_high = EXCLUDED.eps_high,
                eps_low = EXCLUDED.eps_low,
                eps_num_estimates = EXCLUDED.eps_num_estimates,
                revenue_consensus = EXCLUDED.revenue_consensus,
                revenue_high = EXCLUDED.revenue_high,
                revenue_low = EXCLUDED.revenue_low,
                reported_eps = EXCLUDED.reported_eps,
                reported_date = EXCLUDED.reported_date,
                surprise_amount = EXCLUDED.surprise_amount,
                surprise_percentage = EXCLUDED.surprise_percentage,
                fetched_at = EXCLUDED.fetched_at
        """
        
        with engine.connect() as conn:
            conn.execute(text(upsert_query))
            conn.execute(text(f"DROP TABLE IF EXISTS {temp_table}"))
            conn.commit()
            
    except Exception as e:
        logger.error(f"Error saving estimates batch: {e}")
        with engine.connect() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {temp_table}"))
            conn.rollback()

def main():
    """Main execution function."""
    start_time = time.time()
    log_script_start(logger, "fetch_earnings_estimates", "Fetching earnings estimates from Alpha Vantage")
    
    print("\n" + "=" * 60)
    print("EARNINGS ESTIMATES FETCHER")
    print("=" * 60)
    print("Fetching forward-looking analyst estimates")
    print("=" * 60)
    
    try:
        # Setup
        db_manager = DatabaseConnectionManager()
        engine = db_manager.get_engine()
        rate_limiter = AlphaVantageRateLimiter.get_instance()
        fetch_manager = IncrementalFetchManager()
        
        # Get symbols to fetch
        symbols = get_symbols_to_fetch(engine)
        logger.info(f"Found {len(symbols)} symbols to fetch earnings estimates")
        
        if not symbols:
            logger.info("No symbols need earnings estimates fetching")
            return
        
        # Fetch estimates
        total_fetched = 0
        total_records = 0
        batch_records = []
        batch_size = 100
        
        with tqdm(total=len(symbols), desc="Fetching earnings estimates") as pbar:
            for symbol in symbols:
                # Check if recently fetched
                if fetch_manager.should_skip_fetch(symbol, 'earnings_estimates', days=7):
                    pbar.update(1)
                    continue
                
                # Fetch data
                data = fetch_earnings_estimates(symbol, rate_limiter)
                
                if data:
                    # Parse estimates
                    records = parse_earnings_estimates(symbol, data)
                    
                    if records:
                        batch_records.extend(records)
                        total_records += len(records)
                        total_fetched += 1
                        
                        # Save batch
                        if len(batch_records) >= batch_size:
                            save_estimates_batch(engine, batch_records)
                            batch_records = []
                    
                    # Update fetch status
                    fetch_manager.update_fetch_status(symbol, 'earnings_estimates', success=True)
                
                pbar.update(1)
                
                # Small delay between requests
                time.sleep(0.1)
        
        # Save remaining records
        if batch_records:
            save_estimates_batch(engine, batch_records)
        
        # Summary
        print(f"\n[INFO] Fetched estimates for {total_fetched} symbols")
        print(f"[INFO] Saved {total_records} estimate records")
        
        # Show sample of latest estimates
        query = """
        SELECT 
            ee.symbol,
            su.name,
            ee.fiscal_period,
            ee.eps_consensus,
            ee.eps_num_estimates,
            ee.surprise_percentage
        FROM earnings_estimates ee
        JOIN symbol_universe su ON ee.symbol = su.symbol
        WHERE ee.period_type = 'quarterly'
          AND ee.fiscal_period >= CURRENT_DATE
          AND ee.eps_num_estimates >= 5
        ORDER BY ee.eps_num_estimates DESC, ee.fiscal_period
        LIMIT 10
        """
        
        df = pd.read_sql(query, engine)
        if not df.empty:
            print("\nTop upcoming earnings with most analyst coverage:")
            print(df.to_string(index=False))
        
        # Log completion
        duration = time.time() - start_time
        log_script_end(logger, "fetch_earnings_estimates", success=True, duration=duration)
        print(f"\n[SUCCESS] Earnings estimates fetch completed in {duration:.1f} seconds")
        
    except Exception as e:
        logger.error(f"Script failed: {e}")
        log_script_end(logger, "fetch_earnings_estimates", success=False, duration=time.time()-start_time)
        raise

if __name__ == "__main__":
    main()