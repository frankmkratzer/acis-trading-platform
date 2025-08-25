#!/usr/bin/env python3
# File: drop_acis_schema.py
# Purpose: Drop all ACIS tables and materialized views created by setup_schema.py

import os
import sys
import argparse
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

MATERIALIZED_VIEWS = [
    "mv_current_ai_portfolios",
    "mv_symbol_with_metadata",
    "mv_latest_annual_fundamentals",
    "mv_latest_forward_returns",
]

TABLES = [
    # Core reference
    "symbol_universe",
    "stock_metadata",

    # Prices & dividends
    "stock_eod_daily",
    "dividend_history",
    "sp500_price_history",

    # Fundamentals
    "fundamentals_annual",
    "fundamentals_quarterly",

    # Scores & snapshots
    "dividend_growth_scores",
    "ai_model_run_log",
    "ai_feature_snapshot",
    "ai_value_scores",
    "ai_growth_scores",
    "ai_dividend_scores",
    "ai_momentum_scores",

    # Portfolios / holdings / logs
    "ai_value_portfolio",
    "ai_growth_portfolio",
    "ai_dividend_portfolio",
    "ai_momentum_portfolio",
    "ai_portfolio_holdings",
    "portfolio_rebalance_log",

    # Strategy NAV & forward returns
    "strategy_nav",
    "forward_returns",

    # S&P 500 outperformance summary
    "sp500_outperformance_scores",
]

def parse_args():
    p = argparse.ArgumentParser(description="Drop ACIS schema objects")
    p.add_argument("-y", "--yes", action="store_true", help="Do not ask for confirmation")
    return p.parse_args()

def main():
    args = parse_args()
    if not args.yes:
        print("⚠️  This will DROP all ACIS tables and materialized views listed in setup_schema.py.")
        resp = input("Type 'drop it' to proceed: ").strip().lower()
        if resp != "drop it":
            print("Aborted.")
            sys.exit(1)

    load_dotenv()
    pg_url = os.getenv("POSTGRES_URL")
    if not pg_url:
        print("❌ POSTGRES_URL is not set.")
        sys.exit(1)

    engine = create_engine(pg_url)

    # Use a single connection; commit/rollback per statement
    with engine.connect() as conn:
        # Drop MVs first
        for mv in MATERIALIZED_VIEWS:
            stmt = f"DROP MATERIALIZED VIEW IF EXISTS {mv} CASCADE;"
            try:
                conn.execute(text(stmt))
                conn.commit()
                print(f"🧹 Dropped MV: {mv}")
            except Exception as e:
                conn.rollback()
                print(f"⚠️ Could not drop MV {mv}: {e}")

        # Then drop tables
        for tbl in TABLES:
            stmt = f"DROP TABLE IF EXISTS {tbl} CASCADE;"
            try:
                conn.execute(text(stmt))
                conn.commit()
                print(f"🧹 Dropped table: {tbl}")
            except Exception as e:
                conn.rollback()
                print(f"⚠️ Could not drop table {tbl}: {e}")

    print("✅ Drop complete.")

if __name__ == "__main__":
    main()