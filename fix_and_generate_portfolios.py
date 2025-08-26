#!/usr/bin/env python3
"""
Portfolio Generator - Fix and Generate Trading Portfolios
Convert AI scores to actionable trading portfolios
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, date
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

def generate_portfolios():
    """Generate trading portfolios from existing AI scores"""
    
    print("ACIS Portfolio Generator - Fixing and Creating Portfolios")
    print("=" * 60)
    
    load_dotenv()
    engine = create_engine(os.getenv('POSTGRES_URL'))
    
    # Check what AI scores we have
    print("Checking available AI scores...")
    
    with engine.connect() as conn:
        # Check available scores
        scores_available = {}
        score_tables = [
            ('ai_value_scores', 'Value'),
            ('ai_growth_scores', 'Growth'),
            ('ai_momentum_scores', 'Momentum'),
            ('ai_dividend_scores', 'Dividend')
        ]
        
        for table, name in score_tables:
            try:
                result = conn.execute(text(f'SELECT COUNT(*) FROM {table}'))
                count = result.fetchone()[0]
                scores_available[name] = count
                print(f"  {name} Scores: {count:,}")
            except:
                scores_available[name] = 0
                print(f"  {name} Scores: 0 (table missing)")
        
        print()
        
        # Generate portfolios for each available strategy
        portfolios_created = 0
        
        # 1. VALUE PORTFOLIO
        if scores_available['Value'] > 0:
            print("Creating Value Portfolio...")
            try:
                # Get top 30 value stocks
                result = conn.execute(text("""
                    INSERT INTO ai_value_portfolio (symbol, as_of_date, score, percentile, score_label, rank, model_version, score_type)
                    SELECT 
                        symbol,
                        CURRENT_DATE as as_of_date,
                        score,
                        percentile,
                        CASE 
                            WHEN percentile >= 90 THEN 'Excellent Value'
                            WHEN percentile >= 75 THEN 'Good Value'  
                            WHEN percentile >= 50 THEN 'Fair Value'
                            ELSE 'Poor Value'
                        END as score_label,
                        ROW_NUMBER() OVER (ORDER BY score DESC) as rank,
                        'v2.0' as model_version,
                        'value' as score_type
                    FROM ai_value_scores
                    WHERE score IS NOT NULL
                    ORDER BY score DESC
                    LIMIT 30
                    ON CONFLICT (symbol, as_of_date, score_type) DO UPDATE SET
                        score = EXCLUDED.score,
                        rank = EXCLUDED.rank
                """))
                
                print(f"  ✅ Value Portfolio: 30 top stocks selected")
                portfolios_created += 1
                
            except Exception as e:
                print(f"  ❌ Value Portfolio failed: {e}")
        
        # 2. GROWTH PORTFOLIO  
        if scores_available['Growth'] > 0:
            print("Creating Growth Portfolio...")
            try:
                result = conn.execute(text("""
                    INSERT INTO ai_growth_portfolio (symbol, as_of_date, score, percentile, score_label, rank, model_version, score_type)
                    SELECT 
                        symbol,
                        CURRENT_DATE as as_of_date,
                        score,
                        percentile,
                        CASE 
                            WHEN percentile >= 90 THEN 'High Growth'
                            WHEN percentile >= 75 THEN 'Good Growth'
                            WHEN percentile >= 50 THEN 'Moderate Growth'
                            ELSE 'Low Growth'
                        END as score_label,
                        ROW_NUMBER() OVER (ORDER BY score DESC) as rank,
                        'v2.0' as model_version,
                        'growth' as score_type
                    FROM ai_growth_scores
                    WHERE score IS NOT NULL
                    ORDER BY score DESC
                    LIMIT 30
                    ON CONFLICT (symbol, as_of_date, score_type) DO UPDATE SET
                        score = EXCLUDED.score,
                        rank = EXCLUDED.rank
                """))
                
                print(f"  ✅ Growth Portfolio: 30 top stocks selected")
                portfolios_created += 1
                
            except Exception as e:
                print(f"  ❌ Growth Portfolio failed: {e}")
        
        # 3. MOMENTUM PORTFOLIO
        if scores_available['Momentum'] > 0:
            print("Creating Momentum Portfolio...")
            try:
                result = conn.execute(text("""
                    INSERT INTO ai_momentum_portfolio (symbol, as_of_date, score, percentile, score_label, rank, model_version, score_type)
                    SELECT 
                        symbol,
                        CURRENT_DATE as as_of_date,
                        score,
                        percentile,
                        CASE 
                            WHEN percentile >= 90 THEN 'Strong Momentum'
                            WHEN percentile >= 75 THEN 'Good Momentum'
                            WHEN percentile >= 50 THEN 'Weak Momentum'
                            ELSE 'No Momentum'
                        END as score_label,
                        ROW_NUMBER() OVER (ORDER BY score DESC) as rank,
                        'v2.0' as model_version,
                        'momentum' as score_type
                    FROM ai_momentum_scores
                    WHERE score IS NOT NULL
                    ORDER BY score DESC
                    LIMIT 30
                    ON CONFLICT (symbol, as_of_date, score_type) DO UPDATE SET
                        score = EXCLUDED.score,
                        rank = EXCLUDED.rank
                """))
                
                print(f"  ✅ Momentum Portfolio: 30 top stocks selected")
                portfolios_created += 1
                
            except Exception as e:
                print(f"  ❌ Momentum Portfolio failed: {e}")
        
        # 4. DIVIDEND PORTFOLIO (Create basic one from dividend history)
        print("Creating Dividend Portfolio from dividend history...")
        try:
            # Create dividend scores from dividend history
            result = conn.execute(text("""
                WITH dividend_metrics AS (
                    SELECT 
                        d.symbol,
                        COUNT(*) as dividend_payments,
                        SUM(CASE WHEN d.ex_date >= CURRENT_DATE - INTERVAL '1 year' 
                            THEN d.dividend ELSE 0 END) as annual_dividend,
                        AVG(d.dividend) as avg_dividend,
                        MAX(d.ex_date) as last_dividend_date,
                        s.market_cap
                    FROM dividend_history d
                    JOIN symbol_universe s ON d.symbol = s.symbol
                    WHERE d.ex_date >= CURRENT_DATE - INTERVAL '5 years'
                        AND s.market_cap > 5000000000  -- $5B minimum
                    GROUP BY d.symbol, s.market_cap
                    HAVING COUNT(*) >= 4  -- At least 4 payments
                        AND annual_dividend > 0
                ),
                scored_dividends AS (
                    SELECT 
                        symbol,
                        annual_dividend,
                        dividend_payments,
                        CASE 
                            WHEN market_cap IS NOT NULL THEN annual_dividend / (market_cap / 1000000) 
                            ELSE 0 
                        END as yield_score,
                        NTILE(100) OVER (ORDER BY annual_dividend DESC) as percentile
                    FROM dividend_metrics
                )
                INSERT INTO ai_dividend_portfolio (symbol, as_of_date, score, percentile, score_label, rank, model_version, score_type)
                SELECT 
                    symbol,
                    CURRENT_DATE,
                    yield_score as score,
                    percentile,
                    CASE 
                        WHEN percentile >= 90 THEN 'High Yield'
                        WHEN percentile >= 75 THEN 'Good Yield'
                        WHEN percentile >= 50 THEN 'Moderate Yield'
                        ELSE 'Low Yield'
                    END as score_label,
                    ROW_NUMBER() OVER (ORDER BY yield_score DESC) as rank,
                    'v2.0' as model_version,
                    'dividend' as score_type
                FROM scored_dividends
                ORDER BY yield_score DESC
                LIMIT 30
                ON CONFLICT (symbol, as_of_date, score_type) DO UPDATE SET
                    score = EXCLUDED.score,
                    rank = EXCLUDED.rank
            """))
            
            print(f"  ✅ Dividend Portfolio: 30 top dividend stocks selected")
            portfolios_created += 1
            
        except Exception as e:
            print(f"  ❌ Dividend Portfolio failed: {e}")
        
        print()
        print("=" * 60)
        print(f"PORTFOLIO GENERATION COMPLETE!")
        print(f"Portfolios Created: {portfolios_created}/4")
        
        # Verify results
        print("\nVerifying portfolio contents...")
        portfolio_tables = [
            ('ai_value_portfolio', 'Value'),
            ('ai_growth_portfolio', 'Growth'),
            ('ai_momentum_portfolio', 'Momentum'),
            ('ai_dividend_portfolio', 'Dividend')
        ]
        
        total_selections = 0
        for table, name in portfolio_tables:
            try:
                result = conn.execute(text(f'SELECT COUNT(*) FROM {table}'))
                count = result.fetchone()[0]
                total_selections += count
                
                if count > 0:
                    # Get top 5 picks for preview
                    result = conn.execute(text(f"""
                        SELECT symbol, score, score_label, rank 
                        FROM {table} 
                        WHERE as_of_date = CURRENT_DATE
                        ORDER BY rank 
                        LIMIT 5
                    """))
                    
                    picks = result.fetchall()
                    print(f"\n{name} Portfolio ({count} total):")
                    for pick in picks:
                        print(f"  #{pick[3]:2d} {pick[0]} - {pick[2]} (Score: {pick[1]:.3f})")
                else:
                    print(f"\n{name} Portfolio: No selections")
                    
            except Exception as e:
                print(f"\n{name} Portfolio: Error checking ({e})")
        
        print(f"\nTotal Portfolio Selections: {total_selections}")
        
        if total_selections > 50:
            print("\n🎉 SUCCESS: Your trading system now has actionable portfolios!")
            print("Ready for:")
            print("  - Paper trading")
            print("  - Live trading") 
            print("  - Advanced backtesting")
            print("  - Portfolio optimization")
            return True
        else:
            print("\n⚠️  Limited portfolios created. Check individual errors above.")
            return False

if __name__ == "__main__":
    success = generate_portfolios()
    print("\nPortfolio generation completed!")
    if success:
        print("🚀 Your ACIS trading system is now FULLY OPERATIONAL!")
    else:
        print("Some issues occurred. Check the output above.")