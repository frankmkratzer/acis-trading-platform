#!/usr/bin/env python3
"""
Ultra-Premium Dividend History Fetcher
Optimized for maximum performance with stock_prices table integration
"""

import os
import sys
import time
from datetime import datetime
from sqlalchemy import text
from dotenv import load_dotenv
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.logging_config import setup_logger, log_script_start, log_script_end
from database.db_connection_manager import DatabaseConnectionManager

load_dotenv()

# Initialize centralized utilities
logger = setup_logger("fetch_dividend_history")
db_manager = DatabaseConnectionManager()
engine = db_manager.get_engine()

SQL_UPSERT_FROM_PRICES = """
WITH since AS (
    SELECT MAX(ex_date) AS last_ex 
    FROM dividend_history
),
src AS (
    SELECT 
        p.symbol, 
        p.trade_date AS ex_date,
        -- Estimate payment date as 2-3 weeks after ex-date
        p.trade_date + INTERVAL '14 days' AS payment_date,
        -- Record date is typically 2 business days after ex-date
        p.trade_date + INTERVAL '2 days' AS record_date,
        -- Declaration date is typically 2-4 weeks before ex-date
        p.trade_date - INTERVAL '21 days' AS declaration_date,
        p.dividend_amount AS dividend,
        -- Adjusted dividend accounts for splits (using split coefficient if available)
        p.dividend_amount * COALESCE(p.split_coefficient, 1.0) AS adjusted_dividend,
        COALESCE(u.currency, 'USD') AS currency,
        -- Determine frequency based on dividend patterns (simplified)
        CASE 
            WHEN p.dividend_amount < 0.50 THEN 'QUARTERLY'
            WHEN p.dividend_amount < 1.00 THEN 'SEMI_ANNUAL'
            ELSE 'ANNUAL'
        END AS frequency,
        CURRENT_TIMESTAMP AS fetched_at 
    FROM stock_prices p 
    LEFT JOIN symbol_universe u ON u.symbol = p.symbol 
    WHERE p.dividend_amount IS NOT NULL 
      AND p.dividend_amount > 0 
      AND (
          (SELECT last_ex FROM since) IS NULL 
          OR p.trade_date > (SELECT last_ex FROM since)
      )
), 
up AS (
    INSERT INTO dividend_history (
        symbol, ex_date, payment_date, record_date, declaration_date,
        dividend, adjusted_dividend, currency, frequency, fetched_at
    )
    SELECT 
        symbol, ex_date, payment_date, record_date, declaration_date,
        dividend, adjusted_dividend, currency, frequency, fetched_at
    FROM src
    ON CONFLICT (symbol, ex_date) DO UPDATE SET
        payment_date = EXCLUDED.payment_date,
        record_date = EXCLUDED.record_date,
        declaration_date = EXCLUDED.declaration_date,
        dividend = EXCLUDED.dividend,
        adjusted_dividend = EXCLUDED.adjusted_dividend,
        currency = EXCLUDED.currency,
        frequency = EXCLUDED.frequency,
        fetched_at = EXCLUDED.fetched_at
    RETURNING 1
)
SELECT COUNT(*)::int AS upserted FROM up;
"""

def main():
    start_time = time.time()
    log_script_start(logger, "fetch_dividend_history", "Ultra-premium dividend history fetcher from stock_prices")
    
    try:
        # Table and indexes should already exist from setup_schema.py
        logger.info("Using existing dividend_history table from schema...")
        
        with engine.begin() as conn:
            logger.info("Starting ultra-fast dividend data upsert from stock_prices...")
            
            # Execute the ultra-fast upsert
            query_start = time.time()
            result = conn.execute(text(SQL_UPSERT_FROM_PRICES)).scalar()
            query_elapsed = time.time() - query_start
            
            upserted = result or 0
            logger.info(f"Dividend upsert completed: {upserted:,} records in {query_elapsed:.2f}s")
        
        duration = time.time() - start_time
        log_script_end(logger, "fetch_dividend_history", True, duration, {
            "Records upserted": f"{upserted:,}",
            "Query time": f"{query_elapsed:.2f}s",
            "Rate": f"{upserted/query_elapsed:.0f} records/second" if query_elapsed > 0 else "N/A"
        })
            
    except Exception as e:
        logger.error(f"Dividend history fetch failed: {e}")
        log_script_end(logger, "fetch_dividend_history", False, time.time() - start_time)
        sys.exit(1)

if __name__ == "__main__":
    main()