#!/usr/bin/env python3
"""
Create Portfolio Tables for Complete 12-Strategy System
Small Cap + Mid Cap + Large Cap versions of Value/Growth/Momentum/Dividend
"""

from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

def create_12_strategy_tables():
    """Create portfolio tables for all 12 strategies"""
    
    print("CREATING 12-STRATEGY PORTFOLIO TABLES")
    print("=" * 50)
    
    load_dotenv()
    engine = create_engine(os.getenv('POSTGRES_URL'))
    
    # Define all 12 strategies
    strategies = [
        # Small Cap Strategies
        ('value_small_cap', 'Small Cap Value'),
        ('growth_small_cap', 'Small Cap Growth'),
        ('momentum_small_cap', 'Small Cap Momentum'),
        ('dividend_small_cap', 'Small Cap Dividend'),
        
        # Mid Cap Strategies
        ('value_mid_cap', 'Mid Cap Value'),
        ('growth_mid_cap', 'Mid Cap Growth'),
        ('momentum_mid_cap', 'Mid Cap Momentum'),
        ('dividend_mid_cap', 'Mid Cap Dividend'),
        
        # Large Cap Strategies
        ('value_large_cap', 'Large Cap Value'),
        ('growth_large_cap', 'Large Cap Growth'),
        ('momentum_large_cap', 'Large Cap Momentum'),
        ('dividend_large_cap', 'Large Cap Dividend')
    ]
    
    with engine.connect() as conn:
        created_tables = 0
        
        for strategy_key, strategy_name in strategies:
            table_name = f"ai_{strategy_key}_portfolio"
            
            try:
                # Create portfolio table
                conn.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        symbol VARCHAR(10),
                        as_of_date DATE,
                        score NUMERIC,
                        percentile NUMERIC,
                        score_label VARCHAR(50),
                        rank INTEGER,
                        model_version VARCHAR(20),
                        score_type VARCHAR(20),
                        fetched_at TIMESTAMP,
                        PRIMARY KEY (symbol, as_of_date, score_type)
                    )
                """))
                
                print(f"  Created: {table_name}")
                created_tables += 1
                
            except Exception as e:
                print(f"  Error creating {table_name}: {e}")
        
        conn.commit()
        
        print(f"\nSuccessfully created {created_tables}/12 portfolio tables")
        print("\n12-Strategy Portfolio Tables:")
        print("Small Cap:")
        print("  ai_value_small_cap_portfolio")
        print("  ai_growth_small_cap_portfolio")
        print("  ai_momentum_small_cap_portfolio")
        print("  ai_dividend_small_cap_portfolio")
        print("\nMid Cap:")
        print("  ai_value_mid_cap_portfolio")
        print("  ai_growth_mid_cap_portfolio")
        print("  ai_momentum_mid_cap_portfolio")
        print("  ai_dividend_mid_cap_portfolio")
        print("\nLarge Cap:")
        print("  ai_value_large_cap_portfolio")
        print("  ai_growth_large_cap_portfolio")
        print("  ai_momentum_large_cap_portfolio")
        print("  ai_dividend_large_cap_portfolio")
        
        return created_tables == 12

if __name__ == "__main__":
    success = create_12_strategy_tables()
    if success:
        print("\nAll 12 strategy portfolio tables created successfully!")
        print("Complete market cap coverage system ready!")
    else:
        print("\nSome tables failed to create.")