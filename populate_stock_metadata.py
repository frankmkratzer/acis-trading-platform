#!/usr/bin/env python3
# File: populate_stock_metadata.py
# Purpose: Sync or backfill stock_metadata from symbol_universe

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import logging

# ─── Config ───────────────────────────────────────────────
load_dotenv()
engine = create_engine(os.getenv("POSTGRES_URL"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Main ─────────────────────────────────────────────────
def main():
    logger.info("🔄 Populating stock_metadata from symbol_universe...")

    MARKET_CAP_THRESHOLDS = {
        "large": 10e9,
        "mid": 2e9,
        "small": 3e8,
    }

    with engine.begin() as conn:
        # Optional: Count source rows
        result = conn.execute(text("SELECT COUNT(*) FROM symbol_universe"))
        count = result.scalar()
        logger.info(f"📊 Found {count:,} rows in symbol_universe")

        if count == 0:
            logger.warning("⚠️ No data to process. Aborting.")
            return

        conn.execute(text("""
            INSERT INTO stock_metadata AS sm (
                symbol, name, exchange, currency, country,
                sector, industry, market_cap, market_cap_class, is_etf, fetched_at
            )
            SELECT
                su.symbol,
                su.name,
                su.exchange,
                su.currency,
                su.country,
                su.sector,
                su.industry,
                su.market_cap,
                CASE
                    WHEN su.market_cap >= :large THEN 'Large Cap'
                    WHEN su.market_cap >= :mid THEN 'Mid Cap'
                    WHEN su.market_cap >= :small THEN 'Small Cap'
                    ELSE 'Micro Cap'
                END AS market_cap_class,
                su.is_etf,
                su.fetched_at
            FROM symbol_universe su
            WHERE su.symbol IS NOT NULL
            ON CONFLICT (symbol) DO UPDATE SET
                name = EXCLUDED.name,
                exchange = EXCLUDED.exchange,
                currency = EXCLUDED.currency,
                country = EXCLUDED.country,
                sector = EXCLUDED.sector,
                industry = EXCLUDED.industry,
                market_cap = EXCLUDED.market_cap,
                market_cap_class = EXCLUDED.market_cap_class,
                is_etf = EXCLUDED.is_etf,
                fetched_at = EXCLUDED.fetched_at
        """), MARKET_CAP_THRESHOLDS)

    logger.info("✅ Done updating stock_metadata")

if __name__ == "__main__":
    main()
