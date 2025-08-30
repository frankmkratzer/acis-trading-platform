#!/usr/bin/env python3
"""
Verify Database Tables
Shows what tables exist after cleanup
"""

import os
import sys
from sqlalchemy import create_engine, text
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

load_dotenv()

def get_postgres_url():
    """Get PostgreSQL connection URL from environment"""
    postgres_url = os.getenv("POSTGRES_URL")
    if not postgres_url:
        raise ValueError("POSTGRES_URL not set in .env file")
    return postgres_url

def verify_tables():
    """Show current database tables"""
    
    try:
        engine = create_engine(get_postgres_url())
        
        print("\n" + "=" * 70)
        print("DATABASE TABLE VERIFICATION")
        print("=" * 70)
        
        with engine.connect() as conn:
            # Get all tables
            result = conn.execute(text("""
                SELECT 
                    t.table_name,
                    pg_size_pretty(pg_total_relation_size(t.table_schema||'.'||t.table_name)) AS size,
                    obj_description(c.oid) as comment
                FROM information_schema.tables t
                JOIN pg_class c ON c.relname = t.table_name
                JOIN pg_namespace n ON n.oid = c.relnamespace AND n.nspname = t.table_schema
                WHERE t.table_schema = 'public' 
                  AND t.table_type = 'BASE TABLE'
                ORDER BY t.table_name
            """))
            
            all_tables = result.fetchall()
            
            # Define categories
            categories = {
                'Core Data Tables': [
                    'symbol_universe',
                    'stock_prices', 
                    'sp500_history',
                    'forward_returns'
                ],
                'Fundamental Data Tables': [
                    'fundamentals',
                    'company_fundamentals_overview',
                    'dividend_history'
                ],
                'Strategy Analysis Tables': [
                    'excess_cash_flow_metrics',
                    'dividend_sustainability_metrics',
                    'breakout_signals',
                    'sp500_outperformance_detail'
                ],
                'Portfolio Management Tables': [
                    'portfolio_scores',
                    'portfolio_holdings',
                    'portfolio_rebalances',
                    'portfolio_performance'
                ],
                'Tables to Remove (if found)': [
                    'ml_models', 'ml_predictions', 'ml_features',
                    'model_training_queue', 'ml_forward_returns',
                    'strategy_signals', 'ranking_transitions',
                    'backtest_results', 'sp500_price_history',
                    'sp500_constituents', 'technical_indicators',
                    'stock_quality_rankings'
                ]
            }
            
            # Display by category
            for category, expected_tables in categories.items():
                print(f"\n{category}:")
                print("-" * 50)
                found_count = 0
                for table_name, size, comment in all_tables:
                    if table_name in expected_tables:
                        if category == 'Tables to Remove (if found)':
                            print(f"  X {table_name:35} {size:>10} [SHOULD BE REMOVED]")
                        else:
                            print(f"  + {table_name:35} {size:>10}")
                        found_count += 1
                
                # Show missing tables
                for expected_table in expected_tables:
                    if not any(table[0] == expected_table for table in all_tables):
                        if category != 'Tables to Remove (if found)':
                            print(f"  - {expected_table:35} {'MISSING':>10}")
                
                if category != 'Tables to Remove (if found)':
                    print(f"  Found: {found_count}/{len(expected_tables)}")
            
            # Show unexpected tables
            all_expected = []
            for tables in categories.values():
                all_expected.extend(tables)
            
            unexpected = [t for t in all_tables if t[0] not in all_expected]
            if unexpected:
                print(f"\nUnexpected Tables (not in our categories):")
                print("-" * 50)
                for table_name, size, comment in unexpected:
                    print(f"  ? {table_name:35} {size:>10}")
            
            # Summary
            print("\n" + "=" * 70)
            print("SUMMARY")
            print("=" * 70)
            print(f"Total tables in database: {len(all_tables)}")
            
            essential_count = 0
            for category, tables in categories.items():
                if category != 'Tables to Remove (if found)':
                    for table in tables:
                        if any(t[0] == table for t in all_tables):
                            essential_count += 1
            
            print(f"Essential tables present: {essential_count}/15")
            
            removed_count = 0
            for table in categories['Tables to Remove (if found)']:
                if not any(t[0] == table for t in all_tables):
                    removed_count += 1
            
            print(f"Unnecessary tables removed: {removed_count}/13")
            
            if essential_count == 15 and len(all_tables) == 15:
                print("\n[SUCCESS] Database is perfectly clean for Three-Portfolio Strategy!")
            elif essential_count == 15:
                print(f"\n[OK] All essential tables present, but {len(all_tables) - 15} extra tables found")
            else:
                print(f"\n[WARNING] Missing {15 - essential_count} essential tables")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        return 1
    
    return 0

def main():
    """Main execution"""
    return verify_tables()

if __name__ == "__main__":
    sys.exit(main())