#!/usr/bin/env python3
"""
Verify Database Tables - Enhanced Version
Shows all tables including TOP 1% strategy enhancements
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
    """Show current database tables including all enhancements"""
    
    try:
        engine = create_engine(get_postgres_url())
        
        print("\n" + "=" * 80)
        print("DATABASE TABLE VERIFICATION - TOP 1% ENHANCED")
        print("=" * 80)
        
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
            
            # Enhanced categories with TOP 1% tables
            categories = {
                'Core Data Tables': [
                    'symbol_universe',
                    'stock_prices', 
                    'sp500_history',
                    'forward_returns',
                    'ml_forward_returns'
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
                'TOP 1% Fundamental Scores': [
                    'piotroski_scores',
                    'altman_zscores',
                    'beneish_mscores'
                ],
                'TOP 1% Smart Money Tracking': [
                    'insider_transactions',
                    'insider_signals',
                    'institutional_holdings',
                    'institutional_signals',
                    'earnings_estimates'
                ],
                'TOP 1% Risk & Technical': [
                    'risk_metrics',
                    'technical_breakouts',
                    'sector_rotation_matrix',
                    'sector_correlations'
                ],
                'TOP 1% Portfolio Optimization': [
                    'kelly_criterion',
                    'portfolio_allocations',
                    'walk_forward_results',
                    'parameter_stability',
                    'performance_attribution',
                    'sector_attribution'
                ],
                'TOP 1% Backtesting & ML': [
                    'backtest_results',
                    'backtest_trades',
                    'ml_predictions'
                ],
                'Master Scoring System': [
                    'master_scores',
                    'portfolio_scores',
                    'portfolio_holdings',
                    'portfolio_rebalances',
                    'portfolio_performance'
                ]
            }
            
            # Display by category
            total_expected = 0
            total_found = 0
            
            for category, expected_tables in categories.items():
                print(f"\n{category}:")
                print("-" * 60)
                found_count = 0
                for table_name, size, comment in all_tables:
                    if table_name in expected_tables:
                        print(f"  ✓ {table_name:35} {size:>10}")
                        found_count += 1
                
                # Show missing tables
                for expected_table in expected_tables:
                    if not any(table[0] == expected_table for table in all_tables):
                        print(f"  ✗ {expected_table:35} {'MISSING':>10}")
                
                print(f"  Status: {found_count}/{len(expected_tables)} tables present")
                total_expected += len(expected_tables)
                total_found += found_count
            
            # Show unexpected tables
            all_expected = []
            for tables in categories.values():
                all_expected.extend(tables)
            
            unexpected = [t for t in all_tables if t[0] not in all_expected]
            if unexpected:
                print(f"\nOther Tables (not categorized):")
                print("-" * 60)
                for table_name, size, comment in unexpected:
                    print(f"  ? {table_name:35} {size:>10}")
            
            # Summary with color coding
            print("\n" + "=" * 80)
            print("SUMMARY")
            print("=" * 80)
            print(f"Total tables in database: {len(all_tables)}")
            print(f"Expected TOP 1% tables: {total_expected}")
            print(f"Found TOP 1% tables: {total_found}")
            
            completion_pct = (total_found / total_expected * 100) if total_expected > 0 else 0
            
            print(f"\nCompletion: {completion_pct:.1f}%")
            
            if completion_pct == 100:
                print("\n🎉 [PERFECT] All TOP 1% strategy tables are present!")
            elif completion_pct >= 80:
                print(f"\n✅ [GOOD] Most TOP 1% features implemented ({total_found}/{total_expected})")
            elif completion_pct >= 60:
                print(f"\n⚠️  [PARTIAL] Some TOP 1% features missing ({total_expected - total_found} tables)")
            else:
                print(f"\n❌ [INCOMPLETE] Many TOP 1% features missing ({total_expected - total_found} tables)")
            
            # Show what's missing for quick reference
            if total_found < total_expected:
                print("\nMissing TOP 1% Tables:")
                for category, tables in categories.items():
                    missing = [t for t in tables if not any(table[0] == t for table in all_tables)]
                    if missing:
                        print(f"  {category}:")
                        for table in missing:
                            print(f"    - {table}")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        return 1
    
    return 0

def main():
    """Main execution"""
    return verify_tables()

if __name__ == "__main__":
    sys.exit(main())