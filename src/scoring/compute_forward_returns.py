#!/usr/bin/env python3
# File: compute_forward_returns.py
# Purpose: Compute 1m, 3m, 6m, 12m forward returns entirely in SQL (window functions)
#          Incremental + idempotent: recompute last 252d per symbol, upsert on (symbol, as_of_date)

import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
engine = create_engine(os.getenv("POSTGRES_URL"))

SQL = """
WITH latest AS (
  -- Latest forward-return date we have per symbol (may be NULL for new symbols)
  SELECT symbol, MAX(as_of_date) AS last_dt
  FROM forward_returns
  GROUP BY symbol
),
ordered AS (
  -- Precompute future adjusted_close using LEAD over trading days
  SELECT
      e.symbol,
      e.trade_date::date AS as_of_date,
      e.adjusted_close,
      LEAD(e.adjusted_close,  21) OVER (PARTITION BY e.symbol ORDER BY e.trade_date) AS adj_21,
      LEAD(e.adjusted_close,  63) OVER (PARTITION BY e.symbol ORDER BY e.trade_date) AS adj_63,
      LEAD(e.adjusted_close, 126) OVER (PARTITION BY e.symbol ORDER BY e.trade_date) AS adj_126,
      LEAD(e.adjusted_close, 252) OVER (PARTITION BY e.symbol ORDER BY e.trade_date) AS adj_252
  FROM stock_eod_daily e
),
calc AS (
  -- Recompute only the last 252 days per symbol (or everything if symbol is new)
  SELECT
      o.symbol,
      o.as_of_date,
      CASE WHEN o.adjusted_close > 0 AND o.adj_21  IS NOT NULL THEN o.adj_21  / o.adjusted_close - 1 END AS return_1m,
      CASE WHEN o.adjusted_close > 0 AND o.adj_63  IS NOT NULL THEN o.adj_63  / o.adjusted_close - 1 END AS return_3m,
      CASE WHEN o.adjusted_close > 0 AND o.adj_126 IS NOT NULL THEN o.adj_126 / o.adjusted_close - 1 END AS return_6m,
      CASE WHEN o.adjusted_close > 0 AND o.adj_252 IS NOT NULL THEN o.adj_252 / o.adjusted_close - 1 END AS return_12m
  FROM ordered o
  LEFT JOIN latest l ON l.symbol = o.symbol
  WHERE l.last_dt IS NULL                           -- brand new symbol: compute all rows
     OR o.as_of_date >= (l.last_dt - INTERVAL '252 days')  -- otherwise recompute rolling tail
),
up AS (
  INSERT INTO forward_returns (
      symbol, as_of_date, return_1m, return_3m, return_6m, return_12m
  )
  SELECT
      symbol, as_of_date, return_1m, return_3m, return_6m, return_12m
  FROM calc
  WHERE return_1m IS NOT NULL
     OR return_3m IS NOT NULL
     OR return_6m IS NOT NULL
     OR return_12m IS NOT NULL
  ON CONFLICT (symbol, as_of_date) DO UPDATE SET
      return_1m  = EXCLUDED.return_1m,
      return_3m  = EXCLUDED.return_3m,
      return_6m  = EXCLUDED.return_6m,
      return_12m = EXCLUDED.return_12m
  RETURNING 1
)
SELECT COUNT(*)::int AS affected FROM up;
"""

def main():
    with engine.begin() as conn:
        affected = conn.execute(text(SQL)).scalar() or 0
        print(f"✅ forward_returns upserted/updated rows: {affected}")

if __name__ == "__main__":
    main()

