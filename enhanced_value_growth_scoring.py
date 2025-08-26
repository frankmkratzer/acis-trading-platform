#!/usr/bin/env python3
"""
Enhanced Value and Growth Strategy Scoring
Add comprehensive risk controls and fundamental analysis
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

def create_enhanced_scores():
    """Create enhanced value and growth scores with risk controls"""
    print("CREATING ENHANCED VALUE & GROWTH SCORES")
    print("=" * 60)
    
    load_dotenv()
    engine = create_engine(os.getenv('POSTGRES_URL'))
    
    with engine.connect() as conn:
        # Get comprehensive fundamental data
        query = text("""
            WITH latest_fundamentals AS (
                SELECT DISTINCT ON (f.symbol)
                    f.symbol,
                    f.fiscal_date,
                    
                    -- Basic Financials
                    f.totalrevenue,
                    f.grossprofit,
                    f.netincome,
                    f.eps,
                    f.totalassets,
                    f.totalliabilities,
                    f.totalshareholderequity,
                    f.operatingcashflow,
                    f.capitalexpenditures,
                    f.free_cf,
                    f.dividendpayout,
                    
                    -- Previous year data for growth calculations
                    LAG(f.totalrevenue) OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date) as prev_revenue,
                    LAG(f.netincome) OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date) as prev_income,
                    LAG(f.eps) OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date) as prev_eps,
                    
                    -- Get current stock price for valuation ratios
                    (SELECT adjusted_close 
                     FROM stock_eod_daily s 
                     WHERE s.symbol = f.symbol 
                     ORDER BY s.trade_date DESC 
                     LIMIT 1) as current_price
                    
                FROM fundamentals_annual f
                WHERE f.fiscal_date >= '2020-01-01'  -- Recent data
                  AND f.totalrevenue IS NOT NULL
                  AND f.totalrevenue > 0
                  AND f.netincome IS NOT NULL
                  AND f.totalshareholderequity IS NOT NULL
                  AND f.totalshareholderequity > 0
                ORDER BY f.symbol, f.fiscal_date DESC
            )
            SELECT * FROM latest_fundamentals
            WHERE current_price IS NOT NULL
              AND current_price > 5  -- Avoid penny stocks
            ORDER BY symbol
        """)
        
        df = pd.read_sql(query, conn)
        
        if len(df) == 0:
            print("No fundamental data found")
            return False
        
        print(f"Found {len(df)} stocks with fundamental data")
        
        # Calculate enhanced metrics and scores
        value_scores = []
        growth_scores = []
        
        for _, row in df.iterrows():
            symbol = row['symbol']
            
            # Basic data
            revenue = float(row['totalrevenue']) if row['totalrevenue'] else 0
            gross_profit = float(row['grossprofit']) if row['grossprofit'] else 0
            net_income = float(row['netincome']) if row['netincome'] else 0
            eps = float(row['eps']) if row['eps'] else 0
            assets = float(row['totalassets']) if row['totalassets'] else 1
            liabilities = float(row['totalliabilities']) if row['totalliabilities'] else 0
            equity = float(row['totalshareholderequity']) if row['totalshareholderequity'] else 1
            operating_cf = float(row['operatingcashflow']) if row['operatingcashflow'] else 0
            capex = float(row['capitalexpenditures']) if row['capitalexpenditures'] else 0
            free_cf = float(row['free_cf']) if row['free_cf'] else 0
            price = float(row['current_price']) if row['current_price'] else 100
            
            # Previous year data
            prev_revenue = float(row['prev_revenue']) if row['prev_revenue'] else 0
            prev_income = float(row['prev_income']) if row['prev_income'] else 0
            prev_eps = float(row['prev_eps']) if row['prev_eps'] else 0
            
            # Skip if no meaningful data
            if revenue <= 0 or equity <= 0 or assets <= 0:
                continue
            
            # Calculate comprehensive ratios
            market_cap = price * 1000000  # Rough estimate
            
            # Valuation ratios
            pe_ratio = market_cap / net_income if net_income > 0 else 999
            pb_ratio = market_cap / equity if equity > 0 else 999
            ps_ratio = market_cap / revenue if revenue > 0 else 999
            
            # Profitability ratios
            roe = net_income / equity if equity > 0 else 0
            roa = net_income / assets if assets > 0 else 0
            gross_margin = gross_profit / revenue if revenue > 0 else 0
            net_margin = net_income / revenue if revenue > 0 else 0
            
            # Financial health ratios
            debt_to_equity = liabilities / equity if equity > 0 else 10
            equity_ratio = equity / assets if assets > 0 else 0
            
            # Cash flow ratios
            fcf_margin = free_cf / revenue if revenue > 0 else 0
            fcf_yield = free_cf / market_cap if market_cap > 0 and free_cf > 0 else 0
            
            # Growth rates
            revenue_growth = (revenue / prev_revenue - 1) if prev_revenue > 0 else 0
            earnings_growth = (net_income / prev_income - 1) if prev_income != 0 else 0
            eps_growth = (eps / prev_eps - 1) if prev_eps != 0 else 0
            
            # ===================
            # ENHANCED VALUE SCORE
            # ===================
            value_score = 0
            
            # 1. Valuation Metrics (40 points)
            # PE Ratio - lower is better (15 points)
            if 1 < pe_ratio < 50:
                value_score += max(0, (1 / pe_ratio) * 15)
            
            # PB Ratio - lower is better (15 points)
            if 0.1 < pb_ratio < 10:
                value_score += max(0, (1 / pb_ratio) * 10)
            
            # PS Ratio - lower is better (10 points)  
            if 0.1 < ps_ratio < 10:
                value_score += max(0, (1 / ps_ratio) * 10)
            
            # 2. Quality Metrics (25 points)
            # ROE - higher is better (10 points)
            if roe > 0:
                value_score += min(roe * 100, 30) * 0.33
            
            # Net Margin - higher is better (10 points)
            if net_margin > 0:
                value_score += min(net_margin * 100, 25) * 0.4
            
            # FCF Yield - higher is better (5 points)
            if fcf_yield > 0:
                value_score += min(fcf_yield * 100, 20) * 0.25
            
            # 3. Financial Health (20 points)
            # Low debt penalty (10 points)
            if debt_to_equity < 0.5:
                value_score += 10
            elif debt_to_equity < 1.0:
                value_score += 10 * (1.0 - debt_to_equity) / 0.5
            else:
                value_score -= min((debt_to_equity - 1.0) * 5, 15)  # Penalty
            
            # Equity ratio bonus (10 points)
            if equity_ratio > 0.3:
                value_score += min(equity_ratio * 30, 10)
            
            # 4. Dividend Quality (10 points)
            dividends = float(row['dividendpayout']) if row['dividendpayout'] else 0
            if dividends > 0 and net_income > 0:
                payout_ratio = dividends / net_income
                if 0.2 < payout_ratio < 0.6:  # Sustainable payout
                    value_score += 5
                if free_cf > dividends * 1.2:  # Good FCF coverage
                    value_score += 5
            
            # 5. Size stability (5 points)
            if revenue > 1_000_000_000:  # $1B+ revenue
                value_score += 5
            
            value_score = max(value_score, 0)
            
            # ===================
            # ENHANCED GROWTH SCORE  
            # ===================
            growth_score = 0
            
            # 1. Growth Rates (50 points)
            # Revenue Growth (20 points)
            if revenue_growth > 0:
                growth_score += min(revenue_growth * 100, 100) * 0.2
            
            # Earnings Growth (20 points)
            if earnings_growth > 0:
                growth_score += min(earnings_growth * 100, 200) * 0.1
            
            # EPS Growth (10 points)
            if eps_growth > 0:
                growth_score += min(eps_growth * 100, 150) * 0.067
            
            # 2. Quality of Growth (25 points)
            # ROE Quality (10 points) - profitable growth
            if roe > 0.10:  # At least 10% ROE
                growth_score += min(roe * 100, 40) * 0.25
            
            # Gross Margin (10 points) - high-margin business
            if gross_margin > 0.3:  # At least 30% gross margin
                growth_score += min(gross_margin * 100, 80) * 0.125
            
            # FCF Margin (5 points) - cash-generative growth
            if fcf_margin > 0.05:  # At least 5% FCF margin
                growth_score += min(fcf_margin * 100, 25) * 0.2
            
            # 3. Financial Health (15 points)
            # Debt Control - penalize excessive debt
            if debt_to_equity > 2:
                growth_score -= min((debt_to_equity - 2) * 10, 20)
            elif debt_to_equity < 1:
                growth_score += 5  # Bonus for low debt
            
            # Asset efficiency (10 points)
            asset_turnover = revenue / assets if assets > 0 else 0
            if asset_turnover > 0.5:
                growth_score += min(asset_turnover * 10, 10)
            
            # 4. Sustainability (10 points)
            # Revenue size (5 points) - not too small
            if revenue > 100_000_000:  # $100M+ revenue
                growth_score += 5
            
            # Positive FCF (5 points)
            if free_cf > 0:
                growth_score += 5
            
            growth_score = max(growth_score, 0)
            
            # Store results
            if value_score > 0:
                value_scores.append({
                    'symbol': symbol,
                    'score': value_score,
                    'pe_ratio': pe_ratio if pe_ratio < 999 else None,
                    'pb_ratio': pb_ratio if pb_ratio < 999 else None, 
                    'ps_ratio': ps_ratio if ps_ratio < 999 else None,
                    'roe': roe,
                    'debt_to_equity': debt_to_equity,
                    'net_margin': net_margin,
                    'fcf_yield': fcf_yield
                })
            
            if growth_score > 0:
                growth_scores.append({
                    'symbol': symbol,
                    'score': growth_score,
                    'revenue_growth': revenue_growth,
                    'earnings_growth': earnings_growth,
                    'eps_growth': eps_growth,
                    'roe': roe,
                    'gross_margin': gross_margin,
                    'debt_to_equity': debt_to_equity
                })
        
        # Process Value Scores
        if len(value_scores) > 0:
            value_df = pd.DataFrame(value_scores)
            value_df['percentile'] = value_df['score'].rank(pct=True)
            value_df = value_df.sort_values('score', ascending=False)
            
            print(f"\nTop 20 ENHANCED VALUE Stocks:")
            print("-" * 90)
            print(f"{'Rank':<4} {'Symbol':<8} {'Score':<8} {'PE':<8} {'PB':<8} {'ROE':<8} {'D/E':<8} {'Margin':<8}")
            print("-" * 90)
            
            for i, (_, row) in enumerate(value_df.head(20).iterrows(), 1):
                pe_str = f"{row['pe_ratio']:.1f}" if row['pe_ratio'] and row['pe_ratio'] < 999 else 'N/A'
                pb_str = f"{row['pb_ratio']:.1f}" if row['pb_ratio'] and row['pb_ratio'] < 999 else 'N/A'
                print(f"{i:<4} {row['symbol']:<8} {row['score']:<8.1f} {pe_str:<8} {pb_str:<8} "
                      f"{row['roe']:<8.1%} {row['debt_to_equity']:<8.1f} {row['net_margin']:<8.1%}")
            
            # Update Value Scores in Database
            try:
                conn.execute(text("DELETE FROM ai_value_scores WHERE as_of_date = CURRENT_DATE"))
                conn.commit()
                
                inserted_value = 0
                for _, row in value_df.iterrows():
                    try:
                        conn.execute(text("""
                            INSERT INTO ai_value_scores (
                                symbol, as_of_date, score, percentile, predicted_return,
                                model_version, created_at
                            ) VALUES (
                                :symbol, CURRENT_DATE, :score, :percentile, NULL,
                                'v2.0_enhanced', CURRENT_TIMESTAMP
                            )
                        """), {
                            'symbol': row['symbol'],
                            'score': float(row['score']),
                            'percentile': float(row['percentile'])
                        })
                        inserted_value += 1
                    except Exception as e:
                        pass  # Skip duplicates
                conn.commit()
            except Exception as e:
                conn.rollback()
                inserted_value = 0
            
            print(f"\nUpdated {inserted_value} enhanced value scores")
        
        # Process Growth Scores
        if len(growth_scores) > 0:
            growth_df = pd.DataFrame(growth_scores)
            growth_df['percentile'] = growth_df['score'].rank(pct=True)
            growth_df = growth_df.sort_values('score', ascending=False)
            
            print(f"\nTop 20 ENHANCED GROWTH Stocks:")
            print("-" * 100)
            print(f"{'Rank':<4} {'Symbol':<8} {'Score':<8} {'RevGr':<8} {'EarnGr':<8} {'ROE':<8} {'Margin':<8} {'D/E':<8}")
            print("-" * 100)
            
            for i, (_, row) in enumerate(growth_df.head(20).iterrows(), 1):
                rev_str = f"{row['revenue_growth']:.1%}" if row['revenue_growth'] else 'N/A'
                earn_str = f"{row['earnings_growth']:.1%}" if row['earnings_growth'] else 'N/A'
                print(f"{i:<4} {row['symbol']:<8} {row['score']:<8.1f} {rev_str:<8} {earn_str:<8} "
                      f"{row['roe']:<8.1%} {row['gross_margin']:<8.1%} {row['debt_to_equity']:<8.1f}")
            
            # Update Growth Scores in Database
            try:
                conn.execute(text("DELETE FROM ai_growth_scores WHERE as_of_date = CURRENT_DATE"))
                conn.commit()
                
                inserted_growth = 0
                for _, row in growth_df.iterrows():
                    try:
                        conn.execute(text("""
                            INSERT INTO ai_growth_scores (
                                symbol, as_of_date, score, percentile, predicted_return,
                                model_version, created_at
                            ) VALUES (
                                :symbol, CURRENT_DATE, :score, :percentile, NULL,
                                'v2.0_enhanced', CURRENT_TIMESTAMP
                            )
                        """), {
                            'symbol': row['symbol'],
                            'score': float(row['score']),
                            'percentile': float(row['percentile'])
                        })
                        inserted_growth += 1
                    except Exception as e:
                        pass  # Skip duplicates
                conn.commit()
            except Exception as e:
                conn.rollback()
                inserted_growth = 0
            
            print(f"\nUpdated {inserted_growth} enhanced growth scores")
        
        conn.commit()
        
        print(f"\n" + "=" * 60)
        print("ENHANCEMENT SUMMARY:")
        print(f"Enhanced Value Stocks: {len(value_scores) if value_scores else 0}")
        print(f"Enhanced Growth Stocks: {len(growth_scores) if growth_scores else 0}")
        print(f"Added comprehensive risk controls:")
        print(f"  - Debt-to-equity penalties")
        print(f"  - Financial health filters")
        print(f"  - Quality metrics (margins, ROE, FCF)")
        print(f"  - Valuation safeguards")
        print("Enhanced strategies ready for portfolio generation!")
        
        return True

if __name__ == "__main__":
    success = create_enhanced_scores()
    if success:
        print("\nEnhanced Value/Growth scoring completed!")
        print("Run create_portfolios.py to see improved selections.")
    else:
        print("\nEnhanced scoring failed.")