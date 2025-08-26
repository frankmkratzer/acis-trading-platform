#!/usr/bin/env python3
"""
Create ai_dividend_portfolio table to match other portfolio tables
"""

from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

def create_dividend_portfolio_table():
    """Create the ai_dividend_portfolio table"""
    
    load_dotenv()
    engine = create_engine(os.getenv('POSTGRES_URL'))
    
    with engine.connect() as conn:
        # Create ai_dividend_portfolio table to match other portfolio tables
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS ai_dividend_portfolio (
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
        
        conn.commit()
        
        print("Created ai_dividend_portfolio table successfully!")
        return True

if __name__ == "__main__":
    create_dividend_portfolio_table()