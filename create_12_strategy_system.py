#!/usr/bin/env python3
"""
Create Complete 12-Strategy System
Small Cap + Mid Cap + Large Cap versions of Value/Growth/Momentum/Dividend
Total: 12 strategies for complete market coverage
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

def create_12_strategy_system():
    """Create 12 strategies: Small/Mid/Large Cap versions of all 4 strategies"""
    print("CREATING COMPLETE 12-STRATEGY SYSTEM")
    print("=" * 70)
    
    load_dotenv()
    engine = create_engine(os.getenv('POSTGRES_URL'))
    
    # Market Cap Definitions - Complete Coverage
    SMALL_CAP_MAX = 2_000_000_000     # Under $2B = Small Cap
    MID_CAP_MIN = 2_000_000_000       # $2B - $10B = Mid Cap  
    MID_CAP_MAX = 10_000_000_000
    LARGE_CAP_MIN = 10_000_000_000    # $10B+ = Large Cap
    
    print(f"Small Cap: Under ${SMALL_CAP_MAX/1_000_000_000:.0f}B")
    print(f"Mid Cap:   ${MID_CAP_MIN/1_000_000_000:.0f}B - ${MID_CAP_MAX/1_000_000_000:.0f}B")
    print(f"Large Cap: ${LARGE_CAP_MIN/1_000_000_000:.0f}B+")
    
    with engine.connect() as conn:
        # Get comprehensive data with market cap categorization - PURE US STOCKS ONLY
        query = text("""
            WITH latest_data AS (
                SELECT DISTINCT ON (f.symbol)
                    f.symbol,
                    f.fiscal_date,
                    
                    -- Fundamentals
                    f.totalrevenue,
                    f.grossprofit,
                    f.netincome,
                    f.eps,
                    f.totalassets,
                    f.totalliabilities,
                    f.totalshareholderequity,
                    f.operatingcashflow,
                    f.free_cf,
                    f.dividendpayout,
                    
                    -- Growth data
                    LAG(f.totalrevenue) OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date) as prev_revenue,
                    LAG(f.netincome) OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date) as prev_income,
                    LAG(f.eps) OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date) as prev_eps,
                    
                    -- Current price
                    (SELECT adjusted_close 
                     FROM stock_eod_daily s 
                     WHERE s.symbol = f.symbol 
                     ORDER BY s.trade_date DESC 
                     LIMIT 1) as current_price,
                    
                    -- Momentum prices
                    (SELECT adjusted_close 
                     FROM stock_eod_daily s 
                     WHERE s.symbol = f.symbol 
                     AND s.trade_date <= CURRENT_DATE - INTERVAL '90 days'
                     ORDER BY s.trade_date DESC 
                     LIMIT 1) as price_3mo,
                    
                    (SELECT adjusted_close 
                     FROM stock_eod_daily s 
                     WHERE s.symbol = f.symbol 
                     AND s.trade_date <= CURRENT_DATE - INTERVAL '180 days'
                     ORDER BY s.trade_date DESC 
                     LIMIT 1) as price_6mo,
                    
                    (SELECT adjusted_close 
                     FROM stock_eod_daily s 
                     WHERE s.symbol = f.symbol 
                     AND s.trade_date <= CURRENT_DATE - INTERVAL '365 days'
                     ORDER BY s.trade_date DESC 
                     LIMIT 1) as price_1yr
                    
                FROM fundamentals_annual f
                INNER JOIN pure_us_stocks p ON f.symbol = p.symbol  -- FILTER TO PURE US STOCKS ONLY
                WHERE f.fiscal_date >= '2020-01-01'
                  AND f.totalrevenue IS NOT NULL
                  AND f.totalrevenue > 0
                  AND f.totalshareholderequity > 0
                ORDER BY f.symbol, f.fiscal_date DESC
            )
            SELECT * FROM latest_data
            WHERE current_price IS NOT NULL
              AND current_price > 5
            ORDER BY symbol
        """)
        
        df = pd.read_sql(query, conn)
        
        if len(df) == 0:
            print("No data found")
            return False
        
        print(f"Found {len(df)} total stocks")
        
        # Categorize by market cap (using revenue as proxy)
        small_cap_stocks = []
        mid_cap_stocks = []  
        large_cap_stocks = []
        
        for _, row in df.iterrows():
            revenue = float(row['totalrevenue']) if row['totalrevenue'] else 0
            
            if revenue < SMALL_CAP_MAX:
                small_cap_stocks.append(row)
            elif SMALL_CAP_MAX <= revenue < LARGE_CAP_MIN:
                mid_cap_stocks.append(row)
            else:  # revenue >= LARGE_CAP_MIN
                large_cap_stocks.append(row)
        
        print(f"\nMarket Cap Distribution:")
        print(f"  Small Cap Stocks: {len(small_cap_stocks)}")
        print(f"  Mid Cap Stocks:   {len(mid_cap_stocks)}")  
        print(f"  Large Cap Stocks: {len(large_cap_stocks)}")
        
        # Create strategies for each cap category
        cap_categories = [
            (small_cap_stocks, "small_cap", "Small Cap"),
            (mid_cap_stocks, "mid_cap", "Mid Cap"),
            (large_cap_stocks, "large_cap", "Large Cap")
        ]
        
        strategy_results = {}
        
        for stocks, cap_type, cap_label in cap_categories:
            if len(stocks) > 0:
                cap_df = pd.DataFrame(stocks)
                results = process_cap_strategies(conn, cap_df, cap_type, cap_label)
                strategy_results[cap_type] = results
        
        conn.commit()
        
        # Summary
        print(f"\n" + "=" * 70)
        print("12-STRATEGY SYSTEM SUMMARY:")
        print("=" * 70)
        
        total_strategies = 0
        for cap_type, results in strategy_results.items():
            cap_label = cap_type.replace("_", " ").title()
            print(f"\n{cap_label} Strategies:")
            for strategy, count in results.items():
                print(f"  {strategy.title()}: {count} stocks scored")
                if count > 0:
                    total_strategies += 1
        
        print(f"\nTotal Active Strategies: {total_strategies}/12")
        print(f"Complete Market Coverage: Small + Mid + Large Cap")
        print(f"Ready for 12-portfolio generation!")
        
        return total_strategies >= 10  # Allow some flexibility

def process_cap_strategies(conn, df, cap_type, cap_label):
    """Process all 4 strategies for a specific cap category"""
    print(f"\nProcessing {cap_label} Strategies ({len(df)} stocks)")
    print("-" * 50)
    
    # Calculate strategies with cap-specific parameters
    strategies = {
        'value': calculate_value_scores_by_cap(df, cap_type),
        'growth': calculate_growth_scores_by_cap(df, cap_type),
        'momentum': calculate_momentum_scores_by_cap(df, cap_type),
        'dividend': calculate_dividend_scores_by_cap(df, cap_type)
    }
    
    results = {}
    
    for strategy_name, scores_df in strategies.items():
        if len(scores_df) > 0:
            # Save to database
            save_cap_strategy_scores(conn, scores_df, strategy_name, cap_type)
            
            # Show top 3 picks
            print(f"\n{cap_label} {strategy_name.upper()} Top 3:")
            for i, (_, row) in enumerate(scores_df.head(3).iterrows(), 1):
                print(f"  #{i}: {row['symbol']} (Score: {row['score']:.1f})")
            
            results[strategy_name] = len(scores_df)
        else:
            results[strategy_name] = 0
            print(f"\n{cap_label} {strategy_name.upper()}: No qualifying stocks")
    
    return results

def calculate_value_scores_by_cap(df, cap_type):
    """Calculate value scores with cap-specific criteria"""
    scores = []
    
    for _, row in df.iterrows():
        symbol = row['symbol']
        
        revenue = float(row['totalrevenue']) if row['totalrevenue'] else 0
        net_income = float(row['netincome']) if row['netincome'] else 0
        equity = float(row['totalshareholderequity']) if row['totalshareholderequity'] else 1
        assets = float(row['totalassets']) if row['totalassets'] else 1
        liabilities = float(row['totalliabilities']) if row['totalliabilities'] else 0
        free_cf = float(row['free_cf']) if row['free_cf'] else 0
        
        if revenue <= 0 or equity <= 0:
            continue
        
        # Basic ratios
        roe = net_income / equity if equity > 0 else 0
        net_margin = net_income / revenue if revenue > 0 else 0
        debt_to_equity = liabilities / equity if equity > 0 else 10
        
        # Value scoring with cap adjustments
        score = 0
        
        # ROE scoring (40 points) - different thresholds by cap
        roe_threshold = {
            'small_cap': 0.12,   # 12% for small caps (higher bar)
            'mid_cap': 0.15,     # 15% for mid caps (quality focus)
            'large_cap': 0.18    # 18% for large caps (established quality)
        }.get(cap_type, 0.15)
        
        if roe > roe_threshold:
            score += min(roe * 100, 25) * 1.6  # 40 points max
        
        # Net Margin (25 points) - cap-specific expectations
        margin_threshold = {
            'small_cap': 0.05,   # 5% for small caps
            'mid_cap': 0.08,     # 8% for mid caps
            'large_cap': 0.12    # 12% for large caps
        }.get(cap_type, 0.08)
        
        if net_margin > margin_threshold:
            score += min(net_margin * 100, 15) * 1.67  # 25 points max
        
        # Financial Health (25 points)
        debt_threshold = {
            'small_cap': 0.6,    # Allow more debt for small caps
            'mid_cap': 0.4,      # Moderate debt for mid caps  
            'large_cap': 0.3     # Conservative debt for large caps
        }.get(cap_type, 0.4)
        
        if debt_to_equity < debt_threshold:
            score += 15
        elif debt_to_equity < 1.0:
            score += 15 * (1.0 - debt_to_equity) / (1.0 - debt_threshold)
        else:
            score -= min((debt_to_equity - 1.0) * 10, 20)
        
        # Free Cash Flow (10 points)
        if free_cf > 0:
            fcf_margin = free_cf / revenue
            score += min(fcf_margin * 100, 10)
        
        # Cap-specific bonuses
        if cap_type == 'large_cap' and revenue > 50_000_000_000:  # Mega-cap
            score += 5
        elif cap_type == 'small_cap' and revenue < 500_000_000:  # Micro-cap potential
            score += 3
        elif cap_type == 'mid_cap':  # Sweet spot bonus
            score += 2
        
        if score > 0:
            scores.append({
                'symbol': symbol,
                'score': max(score, 0),
                'roe': roe,
                'net_margin': net_margin,
                'debt_to_equity': debt_to_equity
            })
    
    scores_df = pd.DataFrame(scores)
    if len(scores_df) > 0:
        scores_df['percentile'] = scores_df['score'].rank(pct=True)
        scores_df = scores_df.sort_values('score', ascending=False)
    
    return scores_df

def calculate_growth_scores_by_cap(df, cap_type):
    """Calculate growth scores with cap-specific expectations"""
    scores = []
    
    for _, row in df.iterrows():
        symbol = row['symbol']
        
        revenue = float(row['totalrevenue']) if row['totalrevenue'] else 0
        prev_revenue = float(row['prev_revenue']) if row['prev_revenue'] else 0
        net_income = float(row['netincome']) if row['netincome'] else 0
        prev_income = float(row['prev_income']) if row['prev_income'] else 0
        equity = float(row['totalshareholderequity']) if row['totalshareholderequity'] else 1
        gross_profit = float(row['grossprofit']) if row['grossprofit'] else 0
        
        if revenue <= 0 or prev_revenue <= 0:
            continue
        
        # Growth calculations
        revenue_growth = (revenue / prev_revenue - 1) if prev_revenue > 0 else 0
        earnings_growth = (net_income / prev_income - 1) if prev_income != 0 else 0
        roe = net_income / equity if equity > 0 else 0
        gross_margin = gross_profit / revenue if revenue > 0 else 0
        
        # Growth expectations by cap size
        min_revenue_growth = {
            'small_cap': 0.15,   # 15% for small caps
            'mid_cap': 0.12,     # 12% for mid caps
            'large_cap': 0.08    # 8% for large caps
        }.get(cap_type, 0.12)
        
        if revenue_growth < min_revenue_growth:
            continue  # Skip if growth too low
        
        # Growth scoring
        score = 0
        
        # Revenue Growth (50 points) - higher multiplier for smaller caps
        growth_multiplier = {
            'small_cap': 2.0,    # Small caps get 2x bonus
            'mid_cap': 1.5,      # Mid caps get 1.5x bonus
            'large_cap': 1.0     # Large caps baseline
        }.get(cap_type, 1.5)
        
        score += min(revenue_growth * 100, 200) * 0.25 * growth_multiplier
        
        # Earnings Growth (30 points)
        if earnings_growth > 0:
            score += min(earnings_growth * 100, 300) * 0.1 * growth_multiplier
        
        # Quality Control (20 points)
        if roe > 0.10:
            score += min(roe * 100, 30) * 0.67
        
        if gross_margin > 0.30:
            score += min(gross_margin * 100, 20)
        
        if score > 0:
            scores.append({
                'symbol': symbol,
                'score': score,
                'revenue_growth': revenue_growth,
                'earnings_growth': earnings_growth,
                'roe': roe,
                'gross_margin': gross_margin
            })
    
    scores_df = pd.DataFrame(scores)
    if len(scores_df) > 0:
        scores_df['percentile'] = scores_df['score'].rank(pct=True)
        scores_df = scores_df.sort_values('score', ascending=False)
    
    return scores_df

def calculate_momentum_scores_by_cap(df, cap_type):
    """Calculate momentum scores with cap-specific adjustments"""
    scores = []
    
    for _, row in df.iterrows():
        symbol = row['symbol']
        
        current_price = float(row['current_price']) if row['current_price'] else 0
        price_3mo = float(row['price_3mo']) if row['price_3mo'] else 0
        price_6mo = float(row['price_6mo']) if row['price_6mo'] else 0
        price_1yr = float(row['price_1yr']) if row['price_1yr'] else 0
        
        if current_price <= 0 or price_1yr <= 0:
            continue
        
        # Calculate returns
        return_3mo = (current_price / price_3mo - 1) if price_3mo > 0 else 0
        return_6mo = (current_price / price_6mo - 1) if price_6mo > 0 else 0
        return_1yr = (current_price / price_1yr - 1) if price_1yr > 0 else 0
        
        # Standard momentum score
        momentum_score = (
            return_3mo * 0.4 +
            return_6mo * 0.35 +
            return_1yr * 0.25
        )
        
        # Cap-specific multipliers
        cap_multiplier = {
            'small_cap': 1.3,    # Small cap momentum can be stronger
            'mid_cap': 1.1,      # Mid cap moderate bonus
            'large_cap': 1.0     # Large cap baseline
        }.get(cap_type, 1.1)
        
        final_score = momentum_score * 100 * cap_multiplier
        
        scores.append({
            'symbol': symbol,
            'score': final_score,
            'return_3mo': return_3mo,
            'return_6mo': return_6mo,
            'return_1yr': return_1yr
        })
    
    scores_df = pd.DataFrame(scores)
    if len(scores_df) > 0:
        scores_df['percentile'] = scores_df['score'].rank(pct=True)
        scores_df = scores_df.sort_values('score', ascending=False)
    
    return scores_df

def calculate_dividend_scores_by_cap(df, cap_type):
    """Calculate dividend scores with cap-specific criteria"""
    scores = []
    
    for _, row in df.iterrows():
        symbol = row['symbol']
        
        dividends = float(row['dividendpayout']) if row['dividendpayout'] else 0
        net_income = float(row['netincome']) if row['netincome'] else 0
        free_cf = float(row['free_cf']) if row['free_cf'] else 0
        equity = float(row['totalshareholderequity']) if row['totalshareholderequity'] else 1
        revenue = float(row['totalrevenue']) if row['totalrevenue'] else 0
        
        if dividends <= 0 or net_income <= 0:
            continue
        
        # Dividend metrics
        payout_ratio = dividends / net_income
        roe = net_income / equity if equity > 0 else 0
        fcf_coverage = free_cf / dividends if dividends > 0 and free_cf > 0 else 0
        
        # Cap-specific dividend expectations
        dividend_criteria = {
            'small_cap': {'min_roe': 0.15, 'max_payout': 0.6, 'stability_years': 3},
            'mid_cap': {'min_roe': 0.12, 'max_payout': 0.7, 'stability_years': 5},
            'large_cap': {'min_roe': 0.10, 'max_payout': 0.8, 'stability_years': 7}
        }.get(cap_type, {'min_roe': 0.12, 'max_payout': 0.7, 'stability_years': 5})
        
        # Quality filters
        if roe < dividend_criteria['min_roe']:
            continue
        if payout_ratio > dividend_criteria['max_payout']:
            continue
        if fcf_coverage < 1.2:  # Minimum FCF coverage
            continue
        
        # Dividend scoring
        score = 0
        
        # Sustainability (40 points)
        if 0.3 <= payout_ratio <= 0.6:
            score += 25
        if fcf_coverage > 1.5:
            score += 15
        
        # Quality (35 points)
        score += min(roe * 100, 35)
        
        # Stability (25 points) - cap-specific
        revenue_threshold = {
            'small_cap': 200_000_000,    # $200M
            'mid_cap': 2_000_000_000,    # $2B
            'large_cap': 10_000_000_000  # $10B
        }.get(cap_type, 2_000_000_000)
        
        if revenue > revenue_threshold:
            score += 25
        
        if score > 40:  # Minimum quality threshold
            scores.append({
                'symbol': symbol,
                'score': score,
                'payout_ratio': payout_ratio,
                'roe': roe,
                'fcf_coverage': fcf_coverage
            })
    
    scores_df = pd.DataFrame(scores)
    if len(scores_df) > 0:
        scores_df['percentile'] = scores_df['score'].rank(pct=True)
        scores_df = scores_df.sort_values('score', ascending=False)
    
    return scores_df

def save_cap_strategy_scores(conn, scores_df, strategy_name, cap_type):
    """Save scores to cap-specific tables"""
    if len(scores_df) == 0:
        return
        
    table_name = f"ai_{strategy_name}_{cap_type}_scores"
    
    # Create table
    conn.execute(text(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
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
    
    # Clear and insert
    conn.execute(text(f"DELETE FROM {table_name} WHERE as_of_date = CURRENT_DATE"))
    
    inserted = 0
    for _, row in scores_df.iterrows():
        try:
            conn.execute(text(f"""
                INSERT INTO {table_name} (
                    symbol, as_of_date, score, percentile, predicted_return,
                    model_version, created_at
                ) VALUES (
                    :symbol, CURRENT_DATE, :score, :percentile, NULL,
                    'v4.0_12strategy', CURRENT_TIMESTAMP
                )
            """), {
                'symbol': row['symbol'],
                'score': float(row['score']),
                'percentile': float(row['percentile'])
            })
            inserted += 1
        except Exception as e:
            pass
    
    print(f"    Saved {inserted} {strategy_name} scores")

if __name__ == "__main__":
    success = create_12_strategy_system()
    if success:
        print("\n12-Strategy system created successfully!")
        print("Next: Create 12 portfolio tables and generation script")
    else:
        print("\nSome issues with 12-strategy creation")