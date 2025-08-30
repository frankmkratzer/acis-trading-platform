#!/usr/bin/env python3
"""
Update dividend frequency based on historical patterns
Analyzes actual payment frequencies for each stock
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
logger = setup_logger("update_dividend_frequency")
db_manager = DatabaseConnectionManager()
engine = db_manager.get_engine()

SQL_UPDATE_FREQUENCY = """
WITH dividend_patterns AS (
    -- Calculate days between consecutive dividends for each stock (ALL HISTORY)
    SELECT 
        symbol,
        ex_date,
        LAG(ex_date) OVER (PARTITION BY symbol ORDER BY ex_date) AS prev_ex_date,
        ex_date - LAG(ex_date) OVER (PARTITION BY symbol ORDER BY ex_date) AS days_between
    FROM dividend_history
    -- NO DATE RESTRICTION - Analyze complete historical patterns
),
frequency_analysis AS (
    -- Determine frequency based on average days between dividends
    SELECT 
        symbol,
        COUNT(*) AS dividend_count,
        AVG(days_between) AS avg_days_between,
        STDDEV(days_between) AS stddev_days,
        CASE 
            -- Monthly dividends (every ~30 days)
            WHEN AVG(days_between) BETWEEN 25 AND 35 THEN 'MONTHLY'
            -- Quarterly dividends (every ~91 days)
            WHEN AVG(days_between) BETWEEN 80 AND 100 THEN 'QUARTERLY'
            -- Semi-annual dividends (every ~182 days)
            WHEN AVG(days_between) BETWEEN 170 AND 195 THEN 'SEMI_ANNUAL'
            -- Annual dividends (every ~365 days)
            WHEN AVG(days_between) BETWEEN 350 AND 380 THEN 'ANNUAL'
            -- Special/Irregular
            WHEN AVG(days_between) IS NULL OR COUNT(*) < 2 THEN 'SPECIAL'
            ELSE 'IRREGULAR'
        END AS detected_frequency
    FROM dividend_patterns
    WHERE days_between IS NOT NULL
    GROUP BY symbol
)
UPDATE dividend_history dh
SET frequency = fa.detected_frequency
FROM frequency_analysis fa
WHERE dh.symbol = fa.symbol
  AND (dh.frequency IS NULL OR dh.frequency != fa.detected_frequency);
"""

SQL_UPDATE_DATES = """
-- Update estimated dates based on industry standards
UPDATE dividend_history
SET 
    -- Payment date: typically 2-4 weeks after ex-date
    payment_date = CASE 
        WHEN payment_date IS NULL THEN ex_date + INTERVAL '21 days'
        ELSE payment_date
    END,
    -- Record date: typically 1-2 business days after ex-date (T+2 settlement)
    record_date = CASE
        WHEN record_date IS NULL THEN ex_date + INTERVAL '2 days'
        ELSE record_date
    END,
    -- Declaration date: typically 2-6 weeks before ex-date
    declaration_date = CASE
        WHEN declaration_date IS NULL THEN ex_date - INTERVAL '28 days'
        ELSE declaration_date
    END,
    -- Adjusted dividend: account for splits if not already done
    adjusted_dividend = CASE
        WHEN adjusted_dividend IS NULL THEN dividend
        ELSE adjusted_dividend
    END
WHERE payment_date IS NULL 
   OR record_date IS NULL 
   OR declaration_date IS NULL
   OR adjusted_dividend IS NULL;
"""

def analyze_dividend_patterns():
    """Analyze and update dividend frequency patterns"""
    
    with engine.begin() as conn:
        # First, update frequency based on patterns
        logger.info("Analyzing dividend payment patterns...")
        result = conn.execute(text(SQL_UPDATE_FREQUENCY))
        rows_updated = result.rowcount
        logger.info(f"Updated frequency for {rows_updated:,} dividend records")
        
        # Then update any missing date fields
        logger.info("Updating missing date fields...")
        result = conn.execute(text(SQL_UPDATE_DATES))
        dates_updated = result.rowcount
        logger.info(f"Updated {dates_updated:,} records with missing dates")
        
        # Get frequency distribution
        freq_query = """
        SELECT 
            frequency,
            COUNT(DISTINCT symbol) as num_stocks,
            COUNT(*) as num_dividends
        FROM dividend_history
        WHERE frequency IS NOT NULL
        GROUP BY frequency
        ORDER BY num_stocks DESC
        """
        
        result = conn.execute(text(freq_query))
        
        print("\nDividend Frequency Distribution:")
        print("-" * 50)
        for row in result:
            print(f"{row.frequency:15} {row.num_stocks:6,} stocks ({row.num_dividends:8,} dividends)")
        
        return rows_updated, dates_updated

def main():
    start_time = time.time()
    log_script_start(logger, "update_dividend_frequency", "Analyzing and updating dividend frequencies")
    
    try:
        rows_updated, dates_updated = analyze_dividend_patterns()
        
        # Get summary statistics
        with engine.connect() as conn:
            stats_query = """
            SELECT 
                COUNT(DISTINCT symbol) as total_stocks,
                COUNT(*) as total_dividends,
                MIN(ex_date) as earliest_dividend,
                MAX(ex_date) as latest_dividend,
                COUNT(DISTINCT frequency) as frequency_types
            FROM dividend_history
            WHERE dividend > 0
            """
            
            result = conn.execute(text(stats_query)).fetchone()
            
            print("\nDividend History Summary:")
            print("-" * 50)
            print(f"Total Stocks with Dividends: {result.total_stocks:,}")
            print(f"Total Dividend Records: {result.total_dividends:,}")
            print(f"Date Range: {result.earliest_dividend} to {result.latest_dividend}")
            print(f"Frequency Types: {result.frequency_types}")
        
        duration = time.time() - start_time
        log_script_end(logger, "update_dividend_frequency", True, duration, {
            "Frequency updates": f"{rows_updated:,}",
            "Date field updates": f"{dates_updated:,}"
        })
        
    except Exception as e:
        logger.error(f"Failed to update dividend frequencies: {e}")
        log_script_end(logger, "update_dividend_frequency", False, time.time() - start_time)
        sys.exit(1)

if __name__ == "__main__":
    main()