# =====================================
# Updated fetch_dividend_history.py
# =====================================

# !/usr/bin/env python3
# File: fetch_dividend_history.py
# Purpose: Populate dividend_history with database optimizations

import os
import sys
import time
from datetime import datetime
from sqlalchemy import text
from dotenv import load_dotenv

# Add core folder to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))
from optimized_database_utils import DatabaseOptimizer

load_dotenv()

# Use the optimized database connection
db_optimizer = DatabaseOptimizer(os.getenv("POSTGRES_URL"))

SQL_UPSERT_FROM_PRICES = """
                         WITH since AS (SELECT MAX(ex_date) AS last_ex \
                                        FROM dividend_history),
                              src AS (SELECT e.symbol, \
                                             e.trade_date::date AS ex_date, e.dividend_amount::numeric AS dividend, COALESCE(u.currency, 'USD') AS currency, \
                                             CURRENT_TIMESTAMP AS fetched_at \
                                      FROM stock_eod_daily e \
                                               LEFT JOIN symbol_universe u ON u.symbol = e.symbol \
                                      WHERE e.dividend_amount IS NOT NULL \
                                        AND e.dividend_amount > 0 \
                                        AND ( \
                                              (SELECT last_ex FROM since) IS NULL \
                                              OR e.trade_date:: date > (SELECT last_ex FROM since)
                             )
                             ), up AS (
                         INSERT \
                         INTO dividend_history (symbol, ex_date, dividend, currency, fetched_at)
                         SELECT symbol, ex_date, dividend, currency, fetched_at
                         FROM src
                         ON CONFLICT (symbol, ex_date) DO UPDATE SET
                             dividend = EXCLUDED.dividend, currency = EXCLUDED.currency, fetched_at = EXCLUDED.fetched_at
                             RETURNING 1
                             )
                         SELECT COUNT(*) ::int AS upserted \
                         FROM up; \
                         """


def main():
    print("📈 Starting optimized dividend history fetch...")

    # Use the optimized database context
    with db_optimizer.bulk_operation_context("dividend_history") as conn:
        # Ensure table exists
        conn.execute(text("""
                          CREATE TABLE IF NOT EXISTS dividend_history
                          (
                              symbol
                              TEXT
                              NOT
                              NULL,
                              ex_date
                              DATE
                              NOT
                              NULL,
                              dividend
                              NUMERIC,
                              currency
                              TEXT,
                              fetched_at
                              TIMESTAMP,
                              PRIMARY
                              KEY
                          (
                              symbol,
                              ex_date
                          )
                              );
                          """))

        # Create index if not exists
        conn.execute(text("""
                          CREATE INDEX IF NOT EXISTS idx_dividend_history_ex_date
                              ON dividend_history(ex_date);
                          """))

        # Do the incremental upsert with timing
        start_time = time.time()
        res = conn.execute(text(SQL_UPSERT_FROM_PRICES)).scalar()
        elapsed = time.time() - start_time

        print(f"✅ Upserted {res or 0} dividend rows in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
