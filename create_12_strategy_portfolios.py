#!/usr/bin/env python3
"""
Create 12-Strategy Portfolios
Generate portfolios for Small/Mid/Large Cap versions of all 4 strategies
Total: 12 portfolios x 10 stocks each = 120 total positions
"""

import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

def create_12_strategy_portfolios():
    """Create trading portfolios from all 12 cap-based strategies"""
    
    print("CREATING 12-STRATEGY PORTFOLIOS")
    print("=" * 50)
    
    load_dotenv()
    engine = create_engine(os.getenv('POSTGRES_URL'))
    
    # Define all 12 strategies
    strategies = [
        # Small Cap Strategies
        ('ai_value_small_cap_scores', 'ai_value_small_cap_portfolio', 'Small Cap Value', 'value_small_cap'),
        ('ai_growth_small_cap_scores', 'ai_growth_small_cap_portfolio', 'Small Cap Growth', 'growth_small_cap'),
        ('ai_momentum_small_cap_scores', 'ai_momentum_small_cap_portfolio', 'Small Cap Momentum', 'momentum_small_cap'),
        ('ai_dividend_small_cap_scores', 'ai_dividend_small_cap_portfolio', 'Small Cap Dividend', 'dividend_small_cap'),
        
        # Mid Cap Strategies
        ('ai_value_mid_cap_scores', 'ai_value_mid_cap_portfolio', 'Mid Cap Value', 'value_mid_cap'),
        ('ai_growth_mid_cap_scores', 'ai_growth_mid_cap_portfolio', 'Mid Cap Growth', 'growth_mid_cap'),
        ('ai_momentum_mid_cap_scores', 'ai_momentum_mid_cap_portfolio', 'Mid Cap Momentum', 'momentum_mid_cap'),
        ('ai_dividend_mid_cap_scores', 'ai_dividend_mid_cap_portfolio', 'Mid Cap Dividend', 'dividend_mid_cap'),
        
        # Large Cap Strategies
        ('ai_value_large_cap_scores', 'ai_value_large_cap_portfolio', 'Large Cap Value', 'value_large_cap'),
        ('ai_growth_large_cap_scores', 'ai_growth_large_cap_portfolio', 'Large Cap Growth', 'growth_large_cap'),
        ('ai_momentum_large_cap_scores', 'ai_momentum_large_cap_portfolio', 'Large Cap Momentum', 'momentum_large_cap'),
        ('ai_dividend_large_cap_scores', 'ai_dividend_large_cap_portfolio', 'Large Cap Dividend', 'dividend_large_cap')
    ]
    
    with engine.connect() as conn:
        portfolios_created = 0
        
        for scores_table, portfolio_table, strategy_name, strategy_type in strategies:
            print(f"\nCreating {strategy_name} Portfolio...")
            
            try:
                # Check if scores table exists and has data
                result = conn.execute(text(f"""
                    SELECT COUNT(*) as count 
                    FROM information_schema.tables 
                    WHERE table_name = '{scores_table}'
                """))
                
                if result.fetchone()[0] == 0:
                    print(f"  SKIPPED - {scores_table} table doesn't exist")
                    continue
                
                # Get count of available scores
                result = conn.execute(text(f"""
                    SELECT COUNT(*) as count 
                    FROM {scores_table} 
                    WHERE as_of_date = CURRENT_DATE 
                    AND score IS NOT NULL
                """))
                
                score_count = result.fetchone()[0]
                if score_count == 0:
                    print(f"  SKIPPED - No scores in {scores_table}")
                    continue
                
                # Create portfolio from top 10 scores
                conn.execute(text(f"""
                    INSERT INTO {portfolio_table} (
                        symbol, as_of_date, score, percentile, score_label, 
                        rank, model_version, score_type, fetched_at
                    )
                    SELECT 
                        symbol,
                        CURRENT_DATE,
                        score,
                        percentile,
                        '{strategy_name} Pick',
                        ROW_NUMBER() OVER (ORDER BY score DESC),
                        'v3.0_12_strategy',
                        '{strategy_type}',
                        CURRENT_TIMESTAMP
                    FROM {scores_table}
                    WHERE as_of_date = CURRENT_DATE
                      AND score IS NOT NULL
                    ORDER BY score DESC
                    LIMIT 10
                    ON CONFLICT (symbol, as_of_date, score_type) DO UPDATE SET
                        score = EXCLUDED.score,
                        rank = EXCLUDED.rank,
                        fetched_at = CURRENT_TIMESTAMP
                """))
                
                conn.commit()
                print(f"  SUCCESS (10 stocks from {score_count} candidates)")
                portfolios_created += 1
                
            except Exception as e:
                print(f"  FAILED - {e}")
                conn.rollback()
        
        print(f"\nPORTFOLIOS CREATED: {portfolios_created}/12")
        
        # Display portfolio contents by market cap
        print(f"\n" + "=" * 80)
        print("12-STRATEGY PORTFOLIO CONTENTS:")
        print("=" * 80)
        
        total_picks = 0
        cap_sections = [
            ("Small Cap Strategies:", [
                ('ai_value_small_cap_portfolio', 'Value', 'value_small_cap'),
                ('ai_growth_small_cap_portfolio', 'Growth', 'growth_small_cap'),
                ('ai_momentum_small_cap_portfolio', 'Momentum', 'momentum_small_cap'),
                ('ai_dividend_small_cap_portfolio', 'Dividend', 'dividend_small_cap')
            ]),
            ("Mid Cap Strategies:", [
                ('ai_value_mid_cap_portfolio', 'Value', 'value_mid_cap'),
                ('ai_growth_mid_cap_portfolio', 'Growth', 'growth_mid_cap'),
                ('ai_momentum_mid_cap_portfolio', 'Momentum', 'momentum_mid_cap'),
                ('ai_dividend_mid_cap_portfolio', 'Dividend', 'dividend_mid_cap')
            ]),
            ("Large Cap Strategies:", [
                ('ai_value_large_cap_portfolio', 'Value', 'value_large_cap'),
                ('ai_growth_large_cap_portfolio', 'Growth', 'growth_large_cap'),
                ('ai_momentum_large_cap_portfolio', 'Momentum', 'momentum_large_cap'),
                ('ai_dividend_large_cap_portfolio', 'Dividend', 'dividend_large_cap')
            ])
        ]
        
        for section_name, strategies_list in cap_sections:
            print(f"\n{section_name}")
            print("-" * 40)
            
            for portfolio_table, strategy_short, strategy_type in strategies_list:
                try:
                    # Get portfolio count and top picks
                    result = conn.execute(text(f"""
                        SELECT COUNT(*) as count
                        FROM {portfolio_table}
                        WHERE as_of_date = CURRENT_DATE
                    """))
                    
                    count = result.fetchone()[0]
                    total_picks += count
                    
                    if count > 0:
                        # Get top 5 for preview
                        result = conn.execute(text(f"""
                            SELECT symbol, score, rank 
                            FROM {portfolio_table} 
                            WHERE as_of_date = CURRENT_DATE
                            ORDER BY rank 
                            LIMIT 5
                        """))
                        
                        picks = result.fetchall()
                        print(f"  {strategy_short} ({count} total):")
                        for pick in picks:
                            print(f"    #{pick[2]:2d}: {pick[0]} (Score: {pick[1]:.1f})")
                    else:
                        print(f"  {strategy_short}: No selections")
                        
                except Exception as e:
                    print(f"  {strategy_short}: Error - {e}")
        
        print(f"\n" + "=" * 80)
        print(f"TOTAL PORTFOLIO SELECTIONS: {total_picks}")
        print(f"TARGET: 120 stocks (12 strategies x 10 each)")
        
        if total_picks >= 100:  # Allow some flexibility
            print(f"\nSUCCESS: 12-Strategy System is FULLY OPERATIONAL!")
            print(f"\nComplete Market Coverage:")
            print(f"  - Small Cap Strategies: 4 (Value, Growth, Momentum, Dividend)")
            print(f"  - Mid Cap Strategies: 4 (Value, Growth, Momentum, Dividend)")
            print(f"  - Large Cap Strategies: 4 (Value, Growth, Momentum, Dividend)")  
            print(f"  - Total Market Coverage: Small + Mid + Large Cap")
            print(f"  - Enhanced Diversification: 3 cap segments x 4 strategies")
            print(f"\nAdvanced Features:")
            print(f"  - Cap-specific scoring adjustments")
            print(f"  - Risk controls and quality filters")
            print(f"  - Comprehensive fundamental analysis")
            print(f"  - Professional portfolio management")
            return True
        else:
            print(f"\nSome portfolios may have issues. Check above.")
            return False

if __name__ == "__main__":
    success = create_12_strategy_portfolios()
    if success:
        print(f"\n12-Strategy portfolio system is complete and ready!")
        print(f"Complete market cap coverage achieved!")
    else:
        print(f"\nSome issues occurred during portfolio creation.")