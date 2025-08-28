#!/usr/bin/env python3
"""
Create ML Forward Returns table for calculate_forward_returns.py
This separates ML-focused forward returns from simple forward returns
"""

import os
import sys
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

def create_ml_forward_returns_table():
    """Create the ML-focused forward returns table"""
    
    postgres_url = os.getenv("POSTGRES_URL")
    if not postgres_url:
        print("[ERROR] POSTGRES_URL not found in environment variables.")
        return False
    
    try:
        engine = create_engine(postgres_url)
        
        with engine.begin() as conn:
            # Create the new ML forward returns table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS ml_forward_returns (
                    symbol TEXT NOT NULL,
                    ranking_date DATE NOT NULL,
                    horizon_weeks INTEGER NOT NULL,
                    
                    -- Return metrics
                    forward_return NUMERIC,
                    forward_excess_return NUMERIC,  -- vs SP500
                    
                    -- Risk metrics
                    forward_volatility NUMERIC,
                    forward_max_drawdown NUMERIC,
                    
                    -- Additional ML features (optional, for future expansion)
                    forward_sharpe_ratio NUMERIC,
                    forward_win_rate NUMERIC,  -- % of positive days
                    forward_skewness NUMERIC,
                    forward_kurtosis NUMERIC,
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    PRIMARY KEY (symbol, ranking_date, horizon_weeks)
                );
                
                -- Create indexes for performance
                CREATE INDEX IF NOT EXISTS idx_ml_forward_returns_symbol 
                    ON ml_forward_returns(symbol);
                CREATE INDEX IF NOT EXISTS idx_ml_forward_returns_date 
                    ON ml_forward_returns(ranking_date DESC);
                CREATE INDEX IF NOT EXISTS idx_ml_forward_returns_horizon 
                    ON ml_forward_returns(horizon_weeks);
                CREATE INDEX IF NOT EXISTS idx_ml_forward_returns_symbol_date 
                    ON ml_forward_returns(symbol, ranking_date DESC);
            """))
            
            print("[SUCCESS] ml_forward_returns table created successfully")
            
            # Also create a separate table for ranking transitions if needed
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS ranking_transitions (
                    symbol TEXT NOT NULL,
                    from_date DATE NOT NULL,
                    to_date DATE NOT NULL,
                    horizon_weeks INTEGER NOT NULL,
                    
                    -- Ranking changes
                    from_rank INTEGER,
                    to_rank INTEGER,
                    rank_change INTEGER,
                    
                    -- Quality score changes
                    from_quality_score NUMERIC,
                    to_quality_score NUMERIC,
                    quality_score_change NUMERIC,
                    
                    -- Actual return during period
                    actual_return NUMERIC,
                    excess_return NUMERIC,
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    PRIMARY KEY (symbol, from_date, horizon_weeks)
                );
                
                CREATE INDEX IF NOT EXISTS idx_ranking_transitions_symbol 
                    ON ranking_transitions(symbol);
                CREATE INDEX IF NOT EXISTS idx_ranking_transitions_from_date 
                    ON ranking_transitions(from_date DESC);
            """))
            
            print("[SUCCESS] ranking_transitions table created successfully")
            
        # Verify the tables were created
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT column_name, data_type
                FROM information_schema.columns 
                WHERE table_name = 'ml_forward_returns'
                ORDER BY ordinal_position
            """))
            
            columns = list(result)
            print(f"\n[INFO] ml_forward_returns table columns:")
            for col, dtype in columns:
                print(f"  - {col}: {dtype}")
                
            # Check if old forward_returns table exists
            result = conn.execute(text("""
                SELECT COUNT(*) 
                FROM information_schema.tables 
                WHERE table_name = 'forward_returns'
            """))
            
            if result.scalar() > 0:
                print("\n[INFO] Original forward_returns table still exists (good)")
                print("[INFO] Both tables can coexist:")
                print("  - forward_returns: For compute_forward_returns.py (simple daily returns)")
                print("  - ml_forward_returns: For calculate_forward_returns.py (ML risk metrics)")
            
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to create table: {e}")
        return False

if __name__ == "__main__":
    success = create_ml_forward_returns_table()
    sys.exit(0 if success else 1)