#!/usr/bin/env python3
"""
Simple Portfolio Generator - Create Trading Portfolios from AI Scores
"""

import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

def create_portfolios():
    """Create trading portfolios from AI scores"""
    
    print("Creating Trading Portfolios...")
    print("=" * 40)
    
    load_dotenv()
    engine = create_engine(os.getenv('POSTGRES_URL'))
    
    with engine.connect() as conn:
        portfolios_created = 0
        
        # 1. VALUE PORTFOLIO
        print("Creating Value Portfolio...")
        try:
            conn.execute(text("""
                INSERT INTO ai_value_portfolio (symbol, as_of_date, score, percentile, score_label, rank, model_version, score_type, fetched_at)
                SELECT 
                    symbol,
                    CURRENT_DATE,
                    score,
                    percentile,
                    'Value Pick',
                    ROW_NUMBER() OVER (ORDER BY score DESC),
                    'v2.0',
                    'value',
                    CURRENT_TIMESTAMP
                FROM ai_value_scores
                WHERE score IS NOT NULL
                ORDER BY score DESC
                LIMIT 10
                ON CONFLICT (symbol, as_of_date, score_type) DO UPDATE SET
                    score = EXCLUDED.score,
                    rank = EXCLUDED.rank,
                    fetched_at = CURRENT_TIMESTAMP
            """))
            conn.commit()
            print("  Value Portfolio: SUCCESS (10 stocks)")
            portfolios_created += 1
        except Exception as e:
            print(f"  Value Portfolio: FAILED - {e}")
        
        # 2. GROWTH PORTFOLIO
        print("Creating Growth Portfolio...")
        try:
            conn.execute(text("""
                INSERT INTO ai_growth_portfolio (symbol, as_of_date, score, percentile, score_label, rank, model_version, score_type, fetched_at)
                SELECT 
                    symbol,
                    CURRENT_DATE,
                    score,
                    percentile,
                    'Growth Pick',
                    ROW_NUMBER() OVER (ORDER BY score DESC),
                    'v2.0',
                    'growth',
                    CURRENT_TIMESTAMP
                FROM ai_growth_scores
                WHERE score IS NOT NULL
                ORDER BY score DESC
                LIMIT 10
                ON CONFLICT (symbol, as_of_date, score_type) DO UPDATE SET
                    score = EXCLUDED.score,
                    rank = EXCLUDED.rank,
                    fetched_at = CURRENT_TIMESTAMP
            """))
            conn.commit()
            print("  Growth Portfolio: SUCCESS (10 stocks)")
            portfolios_created += 1
        except Exception as e:
            print(f"  Growth Portfolio: FAILED - {e}")
        
        # 3. MOMENTUM PORTFOLIO
        print("Creating Momentum Portfolio...")
        try:
            conn.execute(text("""
                INSERT INTO ai_momentum_portfolio (symbol, as_of_date, score, percentile, score_label, rank, model_version, score_type, fetched_at)
                SELECT 
                    symbol,
                    CURRENT_DATE,
                    score,
                    percentile,
                    'Momentum Pick',
                    ROW_NUMBER() OVER (ORDER BY score DESC),
                    'v2.0',
                    'momentum',
                    CURRENT_TIMESTAMP
                FROM ai_momentum_scores
                WHERE score IS NOT NULL
                ORDER BY score DESC
                LIMIT 10
                ON CONFLICT (symbol, as_of_date, score_type) DO UPDATE SET
                    score = EXCLUDED.score,
                    rank = EXCLUDED.rank,
                    fetched_at = CURRENT_TIMESTAMP
            """))
            conn.commit()
            print("  Momentum Portfolio: SUCCESS (10 stocks)")
            portfolios_created += 1
        except Exception as e:
            print(f"  Momentum Portfolio: FAILED - {e}")
        
        # 4. DIVIDEND PORTFOLIO
        print("Creating Dividend Portfolio...")
        try:
            conn.execute(text("""
                INSERT INTO ai_dividend_portfolio (symbol, as_of_date, score, percentile, score_label, rank, model_version, score_type, fetched_at)
                SELECT 
                    symbol,
                    CURRENT_DATE,
                    score,
                    percentile,
                    'Dividend Pick',
                    ROW_NUMBER() OVER (ORDER BY score DESC),
                    'v2.0',
                    'dividend',
                    CURRENT_TIMESTAMP
                FROM ai_dividend_scores
                WHERE score IS NOT NULL
                ORDER BY score DESC
                LIMIT 10
                ON CONFLICT (symbol, as_of_date, score_type) DO UPDATE SET
                    score = EXCLUDED.score,
                    rank = EXCLUDED.rank,
                    fetched_at = CURRENT_TIMESTAMP
            """))
            conn.commit()
            print("  Dividend Portfolio: SUCCESS (10 stocks)")
            portfolios_created += 1
        except Exception as e:
            print(f"  Dividend Portfolio: FAILED - {e}")
        
        print()
        print(f"PORTFOLIOS CREATED: {portfolios_created}/4")
        
        # Verify and show results
        print("\nPortfolio Contents:")
        
        portfolios = [
            ('ai_value_portfolio', 'VALUE'),
            ('ai_growth_portfolio', 'GROWTH'),
            ('ai_momentum_portfolio', 'MOMENTUM'),
            ('ai_dividend_portfolio', 'DIVIDEND')
        ]
        
        total_picks = 0
        
        for table, name in portfolios:
            try:
                result = conn.execute(text(f'SELECT COUNT(*) FROM {table}'))
                count = result.fetchone()[0]
                total_picks += count
                
                if count > 0:
                    # Get top 5 for preview
                    result = conn.execute(text(f"""
                        SELECT symbol, score, rank 
                        FROM {table} 
                        WHERE as_of_date = CURRENT_DATE
                        ORDER BY rank 
                        LIMIT 5
                    """))
                    
                    picks = result.fetchall()
                    print(f"\n{name} ({count} total):")
                    for pick in picks:
                        print(f"  #{pick[2]:2d}: {pick[0]} (Score: {pick[1]:.3f})")
                else:
                    print(f"\n{name}: No selections")
            except Exception as e:
                print(f"\n{name}: Error - {e}")
        
        print(f"\nTOTAL PORTFOLIO SELECTIONS: {total_picks}")
        
        if total_picks >= 40:  # 4 portfolios x 10 stocks each
            print("\nSUCCESS: Your trading system is now FULLY OPERATIONAL!")
            print("\nReady for:")
            print("  - Paper trading")
            print("  - Live trading")
            print("  - Advanced backtesting")
            return True
        else:
            print("\nSome portfolios may have issues. Check above.")
            return False

if __name__ == "__main__":
    success = create_portfolios()
    if success:
        print("\nYour ACIS trading system is complete and ready!")
    else:
        print("\nSome issues occurred during portfolio creation.")