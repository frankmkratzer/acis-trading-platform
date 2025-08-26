#!/usr/bin/env python3
"""
Direct Dividend Scoring - Create dividend scores using fundamental analysis
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

def create_dividend_scores():
    """Create dividend scores directly using fundamental analysis"""
    print("CREATING DIVIDEND SCORES")
    print("=" * 50)
    
    load_dotenv()
    engine = create_engine(os.getenv('POSTGRES_URL'))
    
    with engine.connect() as conn:
        # Get dividend-paying stocks with fundamental data
        query = text("""
            WITH dividend_stocks AS (
                SELECT DISTINCT ON (f.symbol)
                    f.symbol,
                    f.fiscal_date,
                    
                    -- Raw data
                    f.dividendpayout,
                    f.netincome,
                    f.free_cf,
                    f.operatingcashflow,
                    f.totalrevenue,
                    f.grossprofit,
                    f.totalshareholderequity,
                    f.totalassets,
                    f.totalliabilities,
                    
                    -- Previous year for growth
                    LAG(f.dividendpayout) OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date) as prev_dividends,
                    LAG(f.totalrevenue) OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date) as prev_revenue,
                    LAG(f.netincome) OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date) as prev_income,
                    
                    -- Get latest stock price for yield calculation
                    (SELECT adjusted_close 
                     FROM stock_eod_daily s 
                     WHERE s.symbol = f.symbol 
                     ORDER BY s.trade_date DESC 
                     LIMIT 1) as current_price,
                    
                    -- Count dividend history
                    COUNT(*) OVER (PARTITION BY f.symbol) as dividend_years
                    
                FROM fundamentals_annual f
                WHERE f.dividendpayout IS NOT NULL
                  AND f.dividendpayout > 0
                  AND f.netincome IS NOT NULL
                  AND f.netincome > 0
                  AND f.totalrevenue > 0
                  AND f.totalshareholderequity > 0
                  AND f.fiscal_date >= '2020-01-01'  -- Recent data
                ORDER BY f.symbol, f.fiscal_date DESC
            )
            SELECT * FROM dividend_stocks
            WHERE current_price IS NOT NULL
              AND current_price > 5  -- Avoid penny stocks
              AND dividend_years >= 2  -- At least 2 years of data
            ORDER BY symbol
        """)
        
        df = pd.read_sql(query, conn)
        
        if len(df) == 0:
            print("No dividend-paying stocks found")
            return False
        
        print(f"Found {len(df)} dividend-paying stocks")
        
        # Calculate dividend metrics and scores
        scores = []
        
        for _, row in df.iterrows():
            symbol = row['symbol']
            
            # Basic metrics
            dividends = float(row['dividendpayout']) if row['dividendpayout'] else 0
            net_income = float(row['netincome']) if row['netincome'] else 1
            free_cf = float(row['free_cf']) if row['free_cf'] else 0
            price = float(row['current_price']) if row['current_price'] else 100
            equity = float(row['totalshareholderequity']) if row['totalshareholderequity'] else 1
            revenue = float(row['totalrevenue']) if row['totalrevenue'] else 1
            assets = float(row['totalassets']) if row['totalassets'] else 1
            liabilities = float(row['totalliabilities']) if row['totalliabilities'] else 0
            
            # Calculate key ratios
            payout_ratio = min(dividends / net_income, 1.5) if net_income > 0 else 0
            dividend_yield = dividends / price if price > 0 and dividends > 0 else 0  # Already annual
            fcf_coverage = free_cf / dividends if dividends > 0 and free_cf > 0 else 0
            roe = net_income / equity if equity > 0 else 0
            debt_to_equity = liabilities / equity if equity > 0 else 5
            net_margin = net_income / revenue if revenue > 0 else 0
            
            # Growth rates
            prev_dividends = float(row['prev_dividends']) if row['prev_dividends'] else 0
            dividend_growth = (dividends / prev_dividends - 1) if prev_dividends > 0 else 0
            
            prev_revenue = float(row['prev_revenue']) if row['prev_revenue'] else 0
            revenue_growth = (revenue / prev_revenue - 1) if prev_revenue > 0 else 0
            
            # Calculate dividend score (0-100 scale)
            score = 0
            
            # 1. Dividend Yield (25 points max) - Sweet spot 3-8%
            if 0.02 <= dividend_yield <= 0.08:
                score += min(dividend_yield * 300, 25)
            elif dividend_yield > 0.08:  # Penalty for unsustainable high yields
                score += max(25 - (dividend_yield - 0.08) * 500, 5)
            
            # 2. Payout Ratio Sustainability (20 points max) - Optimal 30-70%
            if 0.2 <= payout_ratio <= 0.8:
                optimal_distance = abs(payout_ratio - 0.5)
                score += 20 * (0.3 - optimal_distance) / 0.3
            
            # 3. FCF Coverage (15 points max) - Coverage > 1.5x is good
            if fcf_coverage > 1.5:
                score += min(fcf_coverage * 5, 15)
            elif fcf_coverage > 1.0:
                score += fcf_coverage * 5
            
            # 4. ROE Quality (15 points max) - At least 10% ROE
            if roe > 0.10:
                score += min(roe * 100, 15)
            elif roe > 0.05:
                score += roe * 50
            
            # 5. Financial Stability (10 points max) - Low debt is good
            if debt_to_equity < 0.5:
                score += 10
            elif debt_to_equity < 1.0:
                score += 10 * (1.0 - debt_to_equity) / 0.5
            else:
                score -= min((debt_to_equity - 1.0) * 5, 10)  # Penalty for high debt
            
            # 6. Dividend Growth (10 points max) - Positive growth is good
            if dividend_growth > 0.05:  # >5% growth
                score += min(dividend_growth * 100, 10)
            elif dividend_growth > 0:
                score += dividend_growth * 50
            
            # 7. Business Quality (5 points max) - Net margin > 5%
            if net_margin > 0.05:
                score += min(net_margin * 100, 5)
            
            # Ensure score is non-negative
            score = max(score, 0)
            
            scores.append({
                'symbol': symbol,
                'score': score,
                'dividend_yield': dividend_yield,
                'payout_ratio': payout_ratio,
                'fcf_coverage': fcf_coverage,
                'roe': roe,
                'debt_to_equity': debt_to_equity,
                'dividend_growth': dividend_growth,
                'net_margin': net_margin
            })
        
        # Convert to DataFrame and calculate percentiles
        scores_df = pd.DataFrame(scores)
        scores_df['percentile'] = scores_df['score'].rank(pct=True)
        
        # Sort by score
        scores_df = scores_df.sort_values('score', ascending=False)
        
        print(f"\nTop 20 Dividend Stocks:")
        print("-" * 80)
        print(f"{'Rank':<4} {'Symbol':<8} {'Score':<8} {'Yield':<8} {'Payout':<8} {'ROE':<8} {'D/E':<8}")
        print("-" * 80)
        
        for i, (_, row) in enumerate(scores_df.head(20).iterrows(), 1):
            print(f"{i:<4} {row['symbol']:<8} {row['score']:<8.1f} "
                  f"{row['dividend_yield']:<8.1%} {row['payout_ratio']:<8.1%} "
                  f"{row['roe']:<8.1%} {row['debt_to_equity']:<8.1f}")
        
        # Save to database
        print(f"\nSaving dividend scores to database...")
        
        # Create table if it doesn't exist
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS ai_dividend_scores (
                symbol VARCHAR(10),
                as_of_date DATE,
                score NUMERIC,
                percentile NUMERIC,
                predicted_return NUMERIC,
                model_version VARCHAR(20),
                created_at TIMESTAMP,
                PRIMARY KEY (symbol, as_of_date)
            )
        """))
        
        # Clear existing scores
        conn.execute(text("DELETE FROM ai_dividend_scores WHERE as_of_date = CURRENT_DATE"))
        
        # Insert new scores
        inserted_count = 0
        for _, row in scores_df.iterrows():
            try:
                conn.execute(text("""
                    INSERT INTO ai_dividend_scores (
                        symbol, as_of_date, score, percentile, predicted_return, 
                        model_version, created_at
                    ) VALUES (
                        :symbol, CURRENT_DATE, :score, :percentile, NULL,
                        'v1.0_direct', CURRENT_TIMESTAMP
                    )
                """), {
                    'symbol': row['symbol'],
                    'score': float(row['score']),
                    'percentile': float(row['percentile'])
                })
                inserted_count += 1
            except Exception as e:
                print(f"Error inserting {row['symbol']}: {e}")
        
        conn.commit()
        
        print(f"Successfully saved {inserted_count} dividend scores!")
        print(f"Average dividend score: {scores_df['score'].mean():.1f}")
        print(f"Top dividend stocks ready for portfolio generation.")
        
        return True

if __name__ == "__main__":
    success = create_dividend_scores()
    if success:
        print("\nDividend scoring completed successfully!")
        print("You can now run create_portfolios.py to generate dividend portfolio.")
    else:
        print("\nDividend scoring failed.")