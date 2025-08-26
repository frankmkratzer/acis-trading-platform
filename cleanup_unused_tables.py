#!/usr/bin/env python3
"""
Clean up unused database tables
"""

import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

def cleanup_unused_tables():
    """Drop tables with 0 rows that aren't needed"""
    
    print("Cleaning up unused database tables...")
    
    load_dotenv()
    engine = create_engine(os.getenv('POSTGRES_URL'))
    
    # Tables that are empty and can be safely removed
    tables_to_drop = [
        'ai_dividend_portfolio',     # Empty, covered by main portfolio tables
        'ai_dividend_scores',        # Empty, dividend scoring not used
        'ai_feature_snapshot',       # Empty, features computed on-demand
        'ai_model_run_log',         # Empty, logging not actively used
        'ai_portfolio_holdings',     # Empty, covered by portfolio tables
        'backtest_nav',             # Empty, backtesting uses separate system
        'backtest_performance',     # Empty, backtesting uses separate system
        'backtest_positions',       # Empty, backtesting uses separate system  
        'backtest_runs',           # Empty, backtesting uses separate system
        'backtest_transactions',   # Empty, backtesting uses separate system
        'data_quality_checks',     # Empty, quality checks done in pipeline
        'economic_indicators',     # Empty, not using macro data yet
        'market_data_status',      # Empty, status tracked in pipeline
        'options_eod',            # Empty, not trading options yet
        'portfolio_positions',     # Empty, live trading not active
        'portfolio_rebalance_log', # Empty, rebalancing not active
        'risk_limits',            # Empty, risk managed in code
        'risk_metrics',           # Empty, risk calculated on-demand
        'stock_intraday',         # Empty, using EOD data only
        'strategy_nav',           # Empty, NAV calculated on-demand
        'system_alerts',          # Empty, alerts handled in code
        'trade_executions',       # Empty, live trading not active
        'trading_accounts',       # Empty, account info in broker APIs
        'trading_orders',         # Empty, orders handled by broker APIs
        'trading_test'            # Test table, can be removed
    ]
    
    # Tables to keep (have data or are essential)
    essential_tables = [
        'symbol_universe',          # 4,031 companies
        'stock_eod_daily',         # 13.4M price records - CORE DATA
        'fundamentals_annual',     # 56K fundamental records
        'fundamentals_quarterly',  # 216K quarterly data
        'dividend_history',        # 100K dividend records
        'sp500_price_history',     # 6.5K benchmark data
        'forward_returns',         # 13.4M AI predictions - CORE AI
        'ai_value_scores',         # 3.8K value scores - ACTIVE
        'ai_growth_scores',        # 3.4K growth scores - ACTIVE  
        'ai_momentum_scores',      # 3.8K momentum scores - ACTIVE
        'ai_value_portfolio',      # 30 value picks - ACTIVE
        'ai_growth_portfolio',     # 30 growth picks - ACTIVE
        'ai_momentum_portfolio',   # 30 momentum picks - ACTIVE
        'dividend_growth_scores',  # 1.9K dividend scores - USEFUL
        'sp500_outperformance_scores' # 4.1K outperformance scores - USEFUL
    ]
    
    with engine.connect() as conn:
        print(f"\nTables to DROP ({len(tables_to_drop)}):")
        for table in tables_to_drop:
            print(f"  - {table}")
            
        print(f"\nTables to KEEP ({len(essential_tables)}):")  
        for table in essential_tables:
            try:
                result = conn.execute(text(f'SELECT COUNT(*) FROM {table}'))
                count = result.fetchone()[0]
                print(f"  - {table:<25} {count:>10,} rows")
            except:
                print(f"  - {table:<25} ERROR/MISSING")
        
        print(f"\nProceeding to drop {len(tables_to_drop)} unused tables...")
        
        dropped = 0
        for table in tables_to_drop:
            try:
                conn.execute(text(f'DROP TABLE IF EXISTS {table} CASCADE'))
                print(f"  [OK] Dropped {table}")
                dropped += 1
            except Exception as e:
                print(f"  [FAIL] Failed to drop {table}: {e}")
        
        conn.commit()
        
        print(f"\nDatabase cleanup complete!")
        print(f"  Tables dropped: {dropped}")
        print(f"  Tables remaining: {len(essential_tables)}")
        print(f"  Database now optimized for production use")

if __name__ == "__main__":
    cleanup_unused_tables()