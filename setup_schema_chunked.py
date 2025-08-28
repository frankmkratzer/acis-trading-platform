#!/usr/bin/env python3
"""
Create ACIS Database Schema in chunks to avoid timeout
"""

import os
import sys
import time
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

def get_postgres_url():
    """Get PostgreSQL URL from environment."""
    postgres_url = os.getenv("POSTGRES_URL")
    if not postgres_url:
        raise RuntimeError("POSTGRES_URL not found in environment variables.")
    return postgres_url

def create_core_tables():
    """Create core trading tables"""
    SQL = """
    -- 1. Symbol Universe
    CREATE TABLE IF NOT EXISTS symbol_universe (
        symbol TEXT PRIMARY KEY,
        name TEXT,
        exchange TEXT,
        sector TEXT,
        industry TEXT,
        market_cap BIGINT,
        country TEXT DEFAULT 'USA',
        currency TEXT DEFAULT 'USD',
        security_type TEXT DEFAULT 'Common Stock',
        is_etf BOOLEAN DEFAULT FALSE,
        delisted_date DATE,
        fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        overview_fetched_at TIMESTAMP
    );
    
    -- 2. Stock Prices
    CREATE TABLE IF NOT EXISTS stock_prices (
        symbol TEXT NOT NULL,
        trade_date DATE NOT NULL,
        open NUMERIC,
        high NUMERIC,
        low NUMERIC,
        close NUMERIC NOT NULL,
        adjusted_close NUMERIC,
        volume BIGINT,
        dividend_amount NUMERIC DEFAULT 0,
        split_coefficient NUMERIC DEFAULT 1,
        fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, trade_date)
    );
    
    -- 3. SP500 History
    CREATE TABLE IF NOT EXISTS sp500_price_history (
        trade_date DATE PRIMARY KEY,
        open NUMERIC,
        high NUMERIC,
        low NUMERIC,
        close NUMERIC NOT NULL,
        adjusted_close NUMERIC,
        volume BIGINT,
        dividend_amount NUMERIC DEFAULT 0,
        split_coefficient NUMERIC DEFAULT 1,
        fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- 4. Forward Returns
    CREATE TABLE IF NOT EXISTS forward_returns (
        symbol TEXT NOT NULL,
        as_of_date DATE NOT NULL,
        return_1m NUMERIC,
        return_3m NUMERIC,
        return_6m NUMERIC,
        return_12m NUMERIC,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, as_of_date)
    );
    
    -- 5. ML Forward Returns (for ML with risk metrics)
    CREATE TABLE IF NOT EXISTS ml_forward_returns (
        symbol TEXT NOT NULL,
        ranking_date DATE NOT NULL,
        horizon_weeks INTEGER NOT NULL,
        forward_return NUMERIC,
        forward_excess_return NUMERIC,
        forward_volatility NUMERIC,
        forward_max_drawdown NUMERIC,
        forward_sharpe_ratio NUMERIC,
        forward_win_rate NUMERIC,
        forward_skewness NUMERIC,
        forward_kurtosis NUMERIC,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, ranking_date, horizon_weeks)
    );
    
    -- 6. Ranking Transitions (for ML training)
    CREATE TABLE IF NOT EXISTS ranking_transitions (
        symbol TEXT NOT NULL,
        from_date DATE NOT NULL,
        to_date DATE NOT NULL,
        horizon_weeks INTEGER NOT NULL,
        from_rank INTEGER,
        to_rank INTEGER,
        rank_change INTEGER,
        from_quality_score NUMERIC,
        to_quality_score NUMERIC,
        quality_score_change NUMERIC,
        actual_return NUMERIC,
        excess_return NUMERIC,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, from_date, horizon_weeks)
    );
    """
    return SQL

def create_fundamentals_tables():
    """Create fundamentals and technical tables"""
    SQL = """
    -- Technical Indicators
    CREATE TABLE IF NOT EXISTS technical_indicators (
        symbol TEXT NOT NULL,
        date DATE NOT NULL,
        rsi_14 NUMERIC,
        sma_20 NUMERIC,
        sma_50 NUMERIC,
        sma_200 NUMERIC,
        ema_12 NUMERIC,
        ema_26 NUMERIC,
        macd NUMERIC,
        macd_signal NUMERIC,
        macd_histogram NUMERIC,
        bb_upper NUMERIC,
        bb_middle NUMERIC,
        bb_lower NUMERIC,
        stoch_k NUMERIC,
        stoch_d NUMERIC,
        williams_r NUMERIC,
        adx NUMERIC,
        obv BIGINT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, date)
    );
    
    -- Fundamentals
    CREATE TABLE IF NOT EXISTS fundamentals (
        symbol TEXT NOT NULL,
        fiscal_date_ending DATE NOT NULL,
        period_type TEXT NOT NULL CHECK (period_type IN ('annual', 'quarterly')),
        reported_date DATE,
        revenue BIGINT,
        gross_profit BIGINT,
        net_income BIGINT,
        earnings_per_share NUMERIC,
        total_assets BIGINT,
        total_liabilities BIGINT,
        total_shareholder_equity BIGINT,
        cash_and_equivalents BIGINT,
        free_cash_flow BIGINT,
        operating_cash_flow BIGINT,
        shares_outstanding BIGINT,
        pe_ratio NUMERIC,
        return_on_equity NUMERIC,
        debt_to_equity NUMERIC,
        fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, fiscal_date_ending, period_type)
    );
    
    -- Dividend History
    CREATE TABLE IF NOT EXISTS dividend_history (
        symbol TEXT NOT NULL,
        ex_date DATE NOT NULL,
        dividend NUMERIC,
        currency TEXT DEFAULT 'USD',
        fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, ex_date)
    );
    """
    return SQL

def create_indexes():
    """Create performance indexes"""
    SQL = """
    -- Symbol universe indexes
    CREATE INDEX IF NOT EXISTS idx_symbol_universe_exchange ON symbol_universe(exchange);
    CREATE INDEX IF NOT EXISTS idx_symbol_universe_sector ON symbol_universe(sector);
    
    -- Stock prices indexes
    CREATE INDEX IF NOT EXISTS idx_stock_prices_symbol ON stock_prices(symbol);
    CREATE INDEX IF NOT EXISTS idx_stock_prices_trade_date ON stock_prices(trade_date DESC);
    CREATE INDEX IF NOT EXISTS idx_stock_prices_symbol_date ON stock_prices(symbol, trade_date DESC);
    
    -- Forward returns indexes
    CREATE INDEX IF NOT EXISTS idx_forward_returns_symbol_date ON forward_returns(symbol, as_of_date DESC);
    
    -- ML forward returns indexes
    CREATE INDEX IF NOT EXISTS idx_ml_forward_returns_symbol ON ml_forward_returns(symbol);
    CREATE INDEX IF NOT EXISTS idx_ml_forward_returns_date ON ml_forward_returns(ranking_date DESC);
    CREATE INDEX IF NOT EXISTS idx_ml_forward_returns_symbol_date ON ml_forward_returns(symbol, ranking_date DESC);
    
    -- Ranking transitions indexes
    CREATE INDEX IF NOT EXISTS idx_ranking_transitions_symbol ON ranking_transitions(symbol);
    CREATE INDEX IF NOT EXISTS idx_ranking_transitions_from_date ON ranking_transitions(from_date DESC);
    
    -- SP500 indexes
    CREATE INDEX IF NOT EXISTS idx_sp500_date ON sp500_price_history(trade_date DESC);
    """
    return SQL

def main():
    """Main execution function."""
    start_time = time.time()
    
    print("\n" + "=" * 60)
    print("ACIS DATABASE SCHEMA SETUP (SIMPLIFIED)")
    print("=" * 60)
    
    try:
        postgres_url = get_postgres_url()
        engine = create_engine(postgres_url)
        
        # Create tables in chunks
        chunks = [
            ("Core Tables", create_core_tables()),
            ("Fundamentals Tables", create_fundamentals_tables()),
            ("Indexes", create_indexes())
        ]
        
        for name, sql in chunks:
            print(f"\n[INFO] Creating {name}...")
            try:
                with engine.begin() as conn:
                    conn.execute(text(sql))
                print(f"[SUCCESS] {name} created")
            except Exception as e:
                print(f"[ERROR] Failed to create {name}: {e}")
                # Continue with other chunks even if one fails
        
        # Verify tables were created
        print("\n[INFO] Verifying tables...")
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                  AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """))
            
            tables = [row[0] for row in result]
            print(f"\n[SUCCESS] Created {len(tables)} tables:")
            for table in tables:
                print(f"  - {table}")
        
        duration = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"[SUCCESS] Schema setup completed in {duration:.1f} seconds")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[ERROR] Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()