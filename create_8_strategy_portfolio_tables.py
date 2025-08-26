#!/usr/bin/env python3
"""
Create Portfolio Tables for 8 Strategies
Large Cap + Small Cap versions of Value/Growth/Momentum/Dividend
"""

from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

def create_8_strategy_tables():
    """Create portfolio tables for all 8 strategies"""
    
    print("CREATING 8 STRATEGY PORTFOLIO TABLES")
    print("=" * 50)
    
    load_dotenv()
    engine = create_engine(os.getenv('POSTGRES_URL'))
    
    # Define the 8 strategies
    strategies = [
        ('value_large_cap', 'Large Cap Value'),
        ('value_small_cap', 'Small Cap Value'),
        ('growth_large_cap', 'Large Cap Growth'),
        ('growth_small_cap', 'Small Cap Growth'),
        ('momentum_large_cap', 'Large Cap Momentum'),
        ('momentum_small_cap', 'Small Cap Momentum'),
        ('dividend_large_cap', 'Large Cap Dividend'),
        ('dividend_small_cap', 'Small Cap Dividend')
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
        
        print(f"\nSuccessfully created {created_tables}/8 portfolio tables")
        print("\nStrategy Portfolio Tables:")
        for strategy_key, strategy_name in strategies:
            print(f"  ai_{strategy_key}_portfolio - {strategy_name}")
        
        return created_tables == 8

if __name__ == "__main__":
    success = create_8_strategy_tables()
    if success:
        print("\nAll 8 strategy portfolio tables created successfully!")
    else:
        print("\nSome tables failed to create.")