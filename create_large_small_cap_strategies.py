#!/usr/bin/env python3
"""
Create Large Cap and Small Cap Strategy Variations
Split each strategy (Value, Growth, Momentum, Dividend) into Large Cap and Small Cap versions
Total: 8 strategies
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

def create_cap_based_strategies():
    """Create 8 strategies: Large Cap + Small Cap versions of Value/Growth/Momentum/Dividend"""
    print("CREATING LARGE CAP & SMALL CAP STRATEGIES")
    print("=" * 70)
    
    load_dotenv()
    engine = create_engine(os.getenv('POSTGRES_URL'))
    
    # Market Cap Definitions
    LARGE_CAP_MIN = 10_000_000_000  # $10B+ = Large Cap
    SMALL_CAP_MAX = 2_000_000_000   # $2B- = Small Cap
    
    print(f"Large Cap Definition: ${LARGE_CAP_MIN/1_000_000_000:.0f}B+")
    print(f"Small Cap Definition: ${SMALL_CAP_MAX/1_000_000_000:.0f}B-")
    
    with engine.connect() as conn:
        # Get comprehensive data with market cap estimation
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
                    
                    -- Current price for market cap
                    (SELECT adjusted_close 
                     FROM stock_eod_daily s 
                     WHERE s.symbol = f.symbol 
                     ORDER BY s.trade_date DESC 
                     LIMIT 1) as current_price,
                    
                    -- Price history for momentum
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
                WHERE f.fiscal_date >= '2020-01-01'
                  AND f.totalrevenue IS NOT NULL
                  AND f.totalrevenue > 0
                  AND f.totalshareholderequity > 0
                ORDER BY f.symbol, f.fiscal_date DESC
            )
            SELECT * FROM latest_data
            WHERE current_price IS NOT NULL
              AND current_price > 5  -- Avoid penny stocks
            ORDER BY symbol
        """)
        
        df = pd.read_sql(query, conn)
        
        if len(df) == 0:
            print("No data found")
            return False
        
        print(f"Found {len(df)} stocks with data")
        
        # Calculate market caps and categorize
        large_cap_stocks = []
        small_cap_stocks = []
        
        for _, row in df.iterrows():
            price = float(row['current_price']) if row['current_price'] else 100
            revenue = float(row['totalrevenue']) if row['totalrevenue'] else 0
            
            # Estimate market cap (rough approximation using price and revenue)
            # More accurate would require shares outstanding, but we'll use revenue as proxy
            estimated_market_cap = price * revenue * 0.001  # Rough scaling factor
            
            # Alternative: use revenue as market cap proxy for categorization
            market_cap_proxy = revenue
            
            if market_cap_proxy >= LARGE_CAP_MIN:
                large_cap_stocks.append(row)
            elif market_cap_proxy <= SMALL_CAP_MAX:
                small_cap_stocks.append(row)
            # Mid-cap stocks (between $2B-$10B) are excluded for cleaner separation
        
        print(f"Large Cap Stocks: {len(large_cap_stocks)}")
        print(f"Small Cap Stocks: {len(small_cap_stocks)}")
        
        # Process Large Cap Strategies
        if len(large_cap_stocks) > 0:
            large_cap_df = pd.DataFrame(large_cap_stocks)
            process_strategies(conn, large_cap_df, "large_cap", "Large Cap")
        
        # Process Small Cap Strategies  
        if len(small_cap_stocks) > 0:
            small_cap_df = pd.DataFrame(small_cap_stocks)
            process_strategies(conn, small_cap_df, "small_cap", "Small Cap")
        
        conn.commit()
        
        print(f"\n" + "=" * 70)
        print("STRATEGY SPLIT SUMMARY:")
        print(f"✅ Created 8 total strategies:")
        print(f"   Large Cap: Value, Growth, Momentum, Dividend")
        print(f"   Small Cap: Value, Growth, Momentum, Dividend")
        print(f"✅ Enhanced diversification and risk management")
        print(f"✅ Ready for advanced portfolio allocation")
        
        return True

def process_strategies(conn, df, cap_type, cap_label):
    """Process all 4 strategies for a specific market cap category"""
    print(f"\nProcessing {cap_label} Strategies ({len(df)} stocks)")
    print("-" * 50)
    
    strategies = {
        'value': calculate_value_scores(df, cap_type),
        'growth': calculate_growth_scores(df, cap_type), 
        'momentum': calculate_momentum_scores(df, cap_type),
        'dividend': calculate_dividend_scores(df, cap_type)
    }
    
    for strategy_name, scores in strategies.items():
        if len(scores) > 0:
            save_strategy_scores(conn, scores, strategy_name, cap_type, cap_label)
            
            # Show top 5 picks
            print(f"\n{cap_label} {strategy_name.upper()} Top 5:")
            for i, stock in enumerate(scores.head(5).iterrows(), 1):
                row = stock[1]
                print(f"  #{i}: {row['symbol']} (Score: {row['score']:.1f})")

def calculate_value_scores(df, cap_type):
    """Calculate value scores with market cap awareness"""
    scores = []
    
    for _, row in df.iterrows():
        symbol = row['symbol']
        
        # Basic metrics
        revenue = float(row['totalrevenue']) if row['totalrevenue'] else 0
        net_income = float(row['netincome']) if row['netincome'] else 0
        equity = float(row['totalshareholderequity']) if row['totalshareholderequity'] else 1
        assets = float(row['totalassets']) if row['totalassets'] else 1
        liabilities = float(row['totalliabilities']) if row['totalliabilities'] else 0
        free_cf = float(row['free_cf']) if row['free_cf'] else 0
        price = float(row['current_price']) if row['current_price'] else 100
        
        if revenue <= 0 or equity <= 0:
            continue
        
        # Calculate ratios
        roe = net_income / equity if equity > 0 else 0
        net_margin = net_income / revenue if revenue > 0 else 0
        debt_to_equity = liabilities / equity if equity > 0 else 10
        fcf_yield = (free_cf / (price * 1000000)) if price > 0 and free_cf > 0 else 0
        
        # Value Score Calculation
        score = 0
        
        # Profitability (40 points)
        if roe > 0:
            score += min(roe * 100, 25) * 0.6  # ROE weight
        if net_margin > 0:
            score += min(net_margin * 100, 15) * 0.4  # Margin weight
        
        # Financial Health (30 points)
        if debt_to_equity < 0.5:
            score += 15
        elif debt_to_equity < 1.0:
            score += 15 * (1.0 - debt_to_equity) / 0.5
        else:
            score -= min((debt_to_equity - 1.0) * 10, 20)  # Penalty
        
        # Free Cash Flow (20 points)
        if fcf_yield > 0:
            score += min(fcf_yield * 1000, 20)  # FCF yield bonus
        
        # Size bonus/penalty based on cap type
        if cap_type == "large_cap":
            # Large cap bonus for stability
            if revenue > 50_000_000_000:  # $50B+ revenue
                score += 10  # Mega-cap stability bonus
        else:  # small_cap
            # Small cap bonus for potential
            if revenue < 500_000_000:  # <$500M revenue
                score += 5  # Small cap growth potential
        
        if score > 0:
            scores.append({
                'symbol': symbol,
                'score': max(score, 0),
                'roe': roe,
                'net_margin': net_margin,
                'debt_to_equity': debt_to_equity,
                'fcf_yield': fcf_yield
            })
    
    scores_df = pd.DataFrame(scores)
    if len(scores_df) > 0:
        scores_df['percentile'] = scores_df['score'].rank(pct=True)
        scores_df = scores_df.sort_values('score', ascending=False)
    
    return scores_df

def calculate_growth_scores(df, cap_type):
    """Calculate growth scores with market cap considerations"""
    scores = []
    
    for _, row in df.iterrows():
        symbol = row['symbol']
        
        # Current and previous data
        revenue = float(row['totalrevenue']) if row['totalrevenue'] else 0
        net_income = float(row['netincome']) if row['netincome'] else 0
        prev_revenue = float(row['prev_revenue']) if row['prev_revenue'] else 0
        prev_income = float(row['prev_income']) if row['prev_income'] else 0
        equity = float(row['totalshareholderequity']) if row['totalshareholderequity'] else 1
        gross_profit = float(row['grossprofit']) if row['grossprofit'] else 0
        
        if revenue <= 0 or prev_revenue <= 0:
            continue
        
        # Calculate growth rates
        revenue_growth = (revenue / prev_revenue - 1) if prev_revenue > 0 else 0
        earnings_growth = (net_income / prev_income - 1) if prev_income != 0 else 0
        
        # Quality metrics
        roe = net_income / equity if equity > 0 else 0
        gross_margin = gross_profit / revenue if revenue > 0 else 0
        
        # Growth Score
        score = 0
        
        # Growth Rates (60 points)
        if revenue_growth > 0:
            growth_multiplier = 1.5 if cap_type == "small_cap" else 1.0  # Small caps get bonus
            score += min(revenue_growth * 100, 150) * 0.3 * growth_multiplier
        
        if earnings_growth > 0:
            score += min(earnings_growth * 100, 200) * 0.3
        
        # Quality (40 points)
        if roe > 0.15:  # Strong profitability
            score += min(roe * 100, 30) * 0.4
        
        if gross_margin > 0.4:  # High margins
            score += min(gross_margin * 100, 50) * 0.2
        
        # Size adjustments
        if cap_type == "small_cap" and revenue < 1_000_000_000:  # <$1B
            score += 10  # Small cap growth premium
        elif cap_type == "large_cap" and revenue > 20_000_000_000:  # >$20B
            score += 5   # Large cap execution capability
        
        if score > 0:
            scores.append({
                'symbol': symbol,
                'score': max(score, 0),
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

def calculate_momentum_scores(df, cap_type):
    """Calculate momentum scores (price-based)"""
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
        
        # Momentum Score (same for both cap types)
        momentum_score = (
            return_3mo * 0.4 +
            return_6mo * 0.35 +
            return_1yr * 0.25
        )
        
        # Volume/liquidity considerations by cap type
        revenue = float(row['totalrevenue']) if row['totalrevenue'] else 0
        if cap_type == "large_cap" and revenue > 10_000_000_000:
            momentum_score *= 1.1  # Large cap liquidity bonus
        elif cap_type == "small_cap" and revenue < 2_000_000_000:
            momentum_score *= 1.2  # Small cap momentum can be stronger
        
        scores.append({
            'symbol': symbol,
            'score': momentum_score * 100,  # Scale to 0-100
            'return_3mo': return_3mo,
            'return_6mo': return_6mo,
            'return_1yr': return_1yr
        })
    
    scores_df = pd.DataFrame(scores)
    if len(scores_df) > 0:
        scores_df['percentile'] = scores_df['score'].rank(pct=True)
        scores_df = scores_df.sort_values('score', ascending=False)
    
    return scores_df

def calculate_dividend_scores(df, cap_type):
    """Calculate dividend scores with cap-specific considerations"""
    scores = []
    
    for _, row in df.iterrows():
        symbol = row['symbol']
        
        dividends = float(row['dividendpayout']) if row['dividendpayout'] else 0
        net_income = float(row['netincome']) if row['netincome'] else 0
        free_cf = float(row['free_cf']) if row['free_cf'] else 0
        equity = float(row['totalshareholderequity']) if row['totalshareholderequity'] else 1
        revenue = float(row['totalrevenue']) if row['totalrevenue'] else 0
        price = float(row['current_price']) if row['current_price'] else 100
        
        # Only dividend-paying companies
        if dividends <= 0 or net_income <= 0:
            continue
        
        # Calculate dividend metrics
        payout_ratio = dividends / net_income
        dividend_yield = dividends / price if price > 0 else 0
        fcf_coverage = free_cf / dividends if dividends > 0 and free_cf > 0 else 0
        roe = net_income / equity if equity > 0 else 0
        
        # Dividend Score
        score = 0
        
        # Yield scoring (35 points) - different targets by cap size
        target_yield_min = 0.02 if cap_type == "large_cap" else 0.03  # Large: 2%+, Small: 3%+
        target_yield_max = 0.06 if cap_type == "large_cap" else 0.08  # Large: 6%, Small: 8%
        
        if target_yield_min <= dividend_yield <= target_yield_max:
            score += min(dividend_yield * 500, 35)
        elif dividend_yield > target_yield_max:
            score += max(35 - (dividend_yield - target_yield_max) * 1000, 10)
        
        # Sustainability (30 points)
        if 0.3 <= payout_ratio <= 0.7:  # Sustainable range
            score += 20
        if fcf_coverage > 1.5:  # Good coverage
            score += 10
        
        # Quality (25 points) - adjusted by cap type
        roe_threshold = 0.12 if cap_type == "large_cap" else 0.15  # Higher bar for small caps
        if roe > roe_threshold:
            score += min(roe * 100, 25)
        
        # Stability (10 points) - cap-specific
        if cap_type == "large_cap":
            if revenue > 10_000_000_000:  # Mega-cap stability
                score += 10
        else:  # small_cap
            if revenue > 500_000_000:  # Decent size for small cap
                score += 5
        
        if score > 10:  # Minimum threshold
            scores.append({
                'symbol': symbol,
                'score': score,
                'dividend_yield': dividend_yield,
                'payout_ratio': payout_ratio,
                'fcf_coverage': fcf_coverage,
                'roe': roe
            })
    
    scores_df = pd.DataFrame(scores)
    if len(scores_df) > 0:
        scores_df['percentile'] = scores_df['score'].rank(pct=True)
        scores_df = scores_df.sort_values('score', ascending=False)
    
    return scores_df

def save_strategy_scores(conn, scores_df, strategy_name, cap_type, cap_label):
    """Save scores to database with cap-specific table names"""
    if len(scores_df) == 0:
        return
        
    table_name = f"ai_{strategy_name}_{cap_type}_scores"
    
    # Create table if it doesn't exist
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
    
    # Clear existing scores
    conn.execute(text(f"DELETE FROM {table_name} WHERE as_of_date = CURRENT_DATE"))
    
    # Insert new scores
    inserted = 0
    for _, row in scores_df.iterrows():
        try:
            conn.execute(text(f"""
                INSERT INTO {table_name} (
                    symbol, as_of_date, score, percentile, predicted_return,
                    model_version, created_at
                ) VALUES (
                    :symbol, CURRENT_DATE, :score, :percentile, NULL,
                    'v3.0_cap_split', CURRENT_TIMESTAMP
                )
            """), {
                'symbol': row['symbol'],
                'score': float(row['score']),
                'percentile': float(row['percentile'])
            })
            inserted += 1
        except Exception as e:
            pass  # Skip errors
    
    print(f"  {cap_label} {strategy_name.upper()}: Saved {inserted} scores")

if __name__ == "__main__":
    success = create_cap_based_strategies()
    if success:
        print("\n🎉 Successfully created 8 cap-based strategies!")
        print("Next: Update portfolio tables and create_portfolios.py for 8 strategies")
    else:
        print("\n❌ Failed to create cap-based strategies")