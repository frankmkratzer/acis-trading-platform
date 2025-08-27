#!/usr/bin/env python3
"""
Ultra-Premium Dividend History Fetcher
Optimized for maximum performance with stock_prices table integration
"""

import os
import sys
import time
from datetime import datetime
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
POSTGRES_URL = os.getenv("POSTGRES_URL")
if not POSTGRES_URL:
    print("[ERROR] POSTGRES_URL not set")
    sys.exit(1)

engine = create_engine(POSTGRES_URL)

from logging_config import setup_logger, log_script_start, log_script_end

logger = setup_logger("fetch_dividend_history")

SQL_UPSERT_FROM_PRICES = """
WITH since AS (
    SELECT MAX(ex_date) AS last_ex 
    FROM dividend_history
),
src AS (
    SELECT 
        p.symbol, 
        p.trade_date AS ex_date, 
        p.dividend_amount AS dividend, 
        COALESCE(u.currency, 'USD') AS currency, 
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
    INSERT INTO dividend_history (symbol, ex_date, dividend, currency, fetched_at)
    SELECT symbol, ex_date, dividend, currency, fetched_at
    FROM src
    ON CONFLICT (symbol, ex_date) DO UPDATE SET
        dividend = EXCLUDED.dividend, 
        currency = EXCLUDED.currency, 
        fetched_at = EXCLUDED.fetched_at
    RETURNING 1
)
SELECT COUNT(*)::int AS upserted FROM up;
"""

def main():
    start_time = time.time()
    log_script_start(logger, "fetch_dividend_history", "Ultra-premium dividend history fetcher from stock_prices")
    
    try:
        logger.info("Ensuring dividend_history table and indexes exist...")
        
        with engine.begin() as conn:
            # Ensure dividend_history table exists (should already exist from schema)
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS dividend_history (
                    symbol TEXT NOT NULL,
                    ex_date DATE NOT NULL,
                    dividend NUMERIC,
                    currency TEXT,
                    fetched_at TIMESTAMP,
                    PRIMARY KEY (symbol, ex_date)
                );
            """))
            
            # Ensure indexes exist
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_dividend_history_symbol ON dividend_history(symbol);
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_dividend_history_ex_date ON dividend_history(ex_date);
            """))
            
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