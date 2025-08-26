#!/usr/bin/env python3
"""
Check existing table structures to identify conflicts
"""

import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from sqlalchemy import create_engine, text

postgres_url = os.getenv("POSTGRES_URL")
if not postgres_url:
    print("No POSTGRES_URL found")
    sys.exit(1)

engine = create_engine(postgres_url)

try:
    with engine.begin() as conn:
        # Check all tables
        result = conn.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """))
        tables = [row[0] for row in result.fetchall()]
        
        print(f"Found {len(tables)} tables:")
        for table in tables:
            print(f"  - {table}")
        
        # Check problematic table structure (likely one that references 'strategy')
        for table in tables:
            try:
                result = conn.execute(text(f"""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = '{table}' 
                    AND table_schema = 'public'
                    ORDER BY ordinal_position
                """))
                columns = result.fetchall()
                
                # Look for strategy column or tables that might cause issues
                strategy_cols = [col for col in columns if 'strategy' in col[0].lower()]
                if strategy_cols or table in ['trading_orders', 'portfolio_holdings']:
                    print(f"\n{table} structure:")
                    for col_name, col_type in columns:
                        print(f"  {col_name}: {col_type}")
                        
            except Exception as e:
                print(f"Error checking {table}: {e}")
                
except Exception as e:
    print(f"Database error: {e}")
finally:
    engine.dispose()