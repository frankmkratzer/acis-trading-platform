#!/usr/bin/env python3
# File: fetch_dividend_history.py
# Purpose: Populate dividend_history directly from stock_eod_daily (no external API)

import os
from datetime import datetime
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
engine = create_engine(os.getenv("POSTGRES_URL"))

SQL_UPSERT_FROM_PRICES = """
WITH since AS (
  SELECT MAX(ex_date) AS last_ex FROM dividend_history
),
src AS (
  SELECT
      e.symbol,
      e.trade_date::date AS ex_date,
      e.dividend_amount::numeric AS dividend,
      COALESCE(u.currency, 'USD') AS currency,
      CURRENT_TIMESTAMP AS fetched_at
  FROM stock_eod_daily e
  LEFT JOIN symbol_universe u ON u.symbol = e.symbol
  WHERE e.dividend_amount IS NOT NULL
    AND e.dividend_amount > 0
    AND (
      (SELECT last_ex FROM since) IS NULL
      OR e.trade_date::date > (SELECT last_ex FROM since)
    )
),
up AS (
  INSERT INTO dividend_history (symbol, ex_date, dividend, currency, fetched_at)
  SELECT symbol, ex_date, dividend, currency, fetched_at
  FROM src
  ON CONFLICT (symbol, ex_date) DO UPDATE SET
    dividend   = EXCLUDED.dividend,
    currency   = EXCLUDED.currency,
    fetched_at = EXCLUDED.fetched_at
  RETURNING 1
)
SELECT COUNT(*)::int AS upserted FROM up;
"""

def main():
    with engine.begin() as conn:
        # Ensure table exists (you already create this in setup_schema; harmless if it exists)
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
        # Do the incremental upsert
        res = conn.execute(text(SQL_UPSERT_FROM_PRICES)).scalar()
        print(f"✅ Upserted {res or 0} dividend rows from stock_eod_daily.")

if __name__ == "__main__":
    main()
