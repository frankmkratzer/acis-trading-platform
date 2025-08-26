#!/usr/bin/env python3
"""
Migration script to update existing tables to match new schema requirements
"""

import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

postgres_url = os.getenv("POSTGRES_URL")
if not postgres_url:
    print("[ERROR] POSTGRES_URL not found")
    sys.exit(1)

engine = create_engine(postgres_url)

MIGRATION_SQL = """
-- Migrate trading_orders table to new schema
DO $$
BEGIN
    -- Add missing columns to trading_orders if they don't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'trading_orders' AND column_name = 'strategy') THEN
        ALTER TABLE trading_orders ADD COLUMN strategy TEXT;
        RAISE NOTICE 'Added strategy column to trading_orders';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'trading_orders' AND column_name = 'portfolio_id') THEN
        ALTER TABLE trading_orders ADD COLUMN portfolio_id TEXT;
        RAISE NOTICE 'Added portfolio_id column to trading_orders';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'trading_orders' AND column_name = 'parent_order_id') THEN
        ALTER TABLE trading_orders ADD COLUMN parent_order_id TEXT;
        RAISE NOTICE 'Added parent_order_id column to trading_orders';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'trading_orders' AND column_name = 'time_in_force') THEN
        ALTER TABLE trading_orders ADD COLUMN time_in_force TEXT DEFAULT 'DAY';
        RAISE NOTICE 'Added time_in_force column to trading_orders';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'trading_orders' AND column_name = 'broker_order_id') THEN
        ALTER TABLE trading_orders ADD COLUMN broker_order_id TEXT;
        RAISE NOTICE 'Added broker_order_id column to trading_orders';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'trading_orders' AND column_name = 'last_updated') THEN
        ALTER TABLE trading_orders ADD COLUMN last_updated TIMESTAMPTZ DEFAULT NOW();
        RAISE NOTICE 'Added last_updated column to trading_orders';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'trading_orders' AND column_name = 'error_message') THEN
        ALTER TABLE trading_orders ADD COLUMN error_message TEXT;
        RAISE NOTICE 'Added error_message column to trading_orders';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'trading_orders' AND column_name = 'tags') THEN
        ALTER TABLE trading_orders ADD COLUMN tags JSONB;
        RAISE NOTICE 'Added tags column to trading_orders';
    END IF;
    
    -- Rename metadata to tags if it exists and tags doesn't
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'trading_orders' AND column_name = 'metadata') 
       AND NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'trading_orders' AND column_name = 'tags') THEN
        ALTER TABLE trading_orders RENAME COLUMN metadata TO tags;
        RAISE NOTICE 'Renamed metadata column to tags in trading_orders';
    END IF;
END$$;

-- Create any missing indexes for trading_orders
CREATE INDEX IF NOT EXISTS idx_orders_strategy ON trading_orders(strategy);
CREATE INDEX IF NOT EXISTS idx_orders_portfolio ON trading_orders(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_orders_status_submitted ON trading_orders(status, submitted_at DESC);
CREATE INDEX IF NOT EXISTS idx_orders_symbol_status ON trading_orders(symbol, status, submitted_at DESC);
"""

def main():
    print("[START] Migrating existing schema...")
    
    try:
        with engine.begin() as conn:
            conn.execute(text(MIGRATION_SQL))
        
        print("[OK] Schema migration completed successfully")
        
        # Verify the changes
        with engine.begin() as conn:
            result = conn.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'trading_orders' 
                AND column_name IN ('strategy', 'portfolio_id', 'tags')
                ORDER BY column_name
            """))
            new_columns = [row[0] for row in result.fetchall()]
            print(f"[VERIFY] Added columns: {', '.join(new_columns)}")
            
    except SQLAlchemyError as e:
        print(f"[ERROR] Migration failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        sys.exit(1)
    finally:
        engine.dispose()

if __name__ == "__main__":
    main()