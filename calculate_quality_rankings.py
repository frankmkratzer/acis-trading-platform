#!/usr/bin/env python3
"""
Stock Quality Rankings Calculator
Calculates three core rankings: SP500 outperformance, Excess Cash Flow, Fundamentals Trend
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from scipy import stats

# Configuration
load_dotenv()
POSTGRES_URL = os.getenv("POSTGRES_URL")
if not POSTGRES_URL:
    print("[ERROR] POSTGRES_URL not set")
    sys.exit(1)

engine = create_engine(POSTGRES_URL)

# Logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/calculate_quality_rankings.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("calculate_quality_rankings")

class QualityRankingCalculator:
    """Calculate quality rankings for all stocks"""
    
    def __init__(self):
        self.engine = engine
        self.ranking_date = datetime.now().date()
        
    def calculate_sp500_outperformance(self):
        """Calculate SP500 outperformance rankings"""
        print("[INFO] Calculating SP500 outperformance rankings...")
        
        # Get annual returns for all stocks and SP500
        query = text("""
            WITH sp500_returns AS (
                SELECT 
                    EXTRACT(YEAR FROM trade_date) as year,
                    (LAST_VALUE(adjusted_close) OVER (PARTITION BY EXTRACT(YEAR FROM trade_date) ORDER BY trade_date
                     ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) /
                     FIRST_VALUE(adjusted_close) OVER (PARTITION BY EXTRACT(YEAR FROM trade_date) ORDER BY trade_date) - 1) * 100 as sp500_return
                FROM sp500_price_history
                WHERE trade_date >= CURRENT_DATE - INTERVAL '21 years'
                GROUP BY EXTRACT(YEAR FROM trade_date), trade_date, adjusted_close
            ),
            stock_returns AS (
                SELECT 
                    symbol,
                    EXTRACT(YEAR FROM trade_date) as year,
                    (LAST_VALUE(adjusted_close) OVER (PARTITION BY symbol, EXTRACT(YEAR FROM trade_date) ORDER BY trade_date
                     ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) /
                     FIRST_VALUE(adjusted_close) OVER (PARTITION BY symbol, EXTRACT(YEAR FROM trade_date) ORDER BY trade_date) - 1) * 100 as stock_return
                FROM stock_prices
                WHERE trade_date >= CURRENT_DATE - INTERVAL '21 years'
                  AND adjusted_close > 0
                GROUP BY symbol, EXTRACT(YEAR FROM trade_date), trade_date, adjusted_close
            )
            SELECT 
                s.symbol,
                s.year,
                s.stock_return,
                sp.sp500_return,
                s.stock_return - sp.sp500_return as excess_return,
                CASE WHEN s.stock_return > sp.sp500_return THEN TRUE ELSE FALSE END as beat_sp500
            FROM stock_returns s
            JOIN sp500_returns sp ON s.year = sp.year
            WHERE s.year >= EXTRACT(YEAR FROM CURRENT_DATE) - 20
            ORDER BY s.symbol, s.year DESC
        """)
        
        with engine.connect() as conn:
            df = pd.read_sql_query(query, conn)
        
        if df.empty:
            logger.warning("No data for SP500 outperformance calculation")
            return pd.DataFrame()
        
        # Calculate weighted scores (recent years weighted higher)
        current_year = datetime.now().year
        decay_factor = 0.95  # Each year back gets 95% of the weight
        
        rankings = []
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol].sort_values('year', ascending=False)
            
            if len(symbol_df) < 3:  # Need at least 3 years of data
                continue
            
            # Calculate metrics
            years_beating = symbol_df['beat_sp500'].sum()
            total_years = len(symbol_df)
            
            # Weighted score - recent years matter more
            weighted_score = 0
            weight_sum = 0
            for i, row in symbol_df.iterrows():
                years_ago = current_year - row['year']
                weight = decay_factor ** years_ago
                weighted_score += row['excess_return'] * weight if row['beat_sp500'] else row['excess_return'] * weight * 0.5
                weight_sum += weight
            
            weighted_score = weighted_score / weight_sum if weight_sum > 0 else 0
            
            # Recent performance
            recent_5yr = symbol_df.head(5)
            recent_5yr_beat = recent_5yr['beat_sp500'].sum() if len(recent_5yr) >= 5 else 0
            recent_1yr_excess = symbol_df.iloc[0]['excess_return'] if len(symbol_df) > 0 else 0
            
            rankings.append({
                'symbol': symbol,
                'years_beating_sp500': int(years_beating),
                'total_years': total_years,
                'sp500_weighted_score': weighted_score,
                'avg_annual_excess': symbol_df['excess_return'].mean(),
                'recent_5yr_beat_count': int(recent_5yr_beat),
                'recent_1yr_excess': recent_1yr_excess
            })
        
        # Create DataFrame and rank
        rank_df = pd.DataFrame(rankings)
        
        # Rank based on weighted score
        rank_df['beat_sp500_ranking'] = rank_df['sp500_weighted_score'].rank(ascending=False, method='min').astype(int)
        
        return rank_df
    
    def calculate_excess_cash_flow(self):
        """Calculate excess cash flow rankings"""
        print("[INFO] Calculating excess cash flow rankings...")
        
        query = text("""
            WITH latest_fundamentals AS (
                SELECT DISTINCT ON (symbol)
                    symbol,
                    free_cash_flow,
                    operating_cash_flow,
                    net_income,
                    revenue,
                    fiscal_date_ending
                FROM fundamentals
                WHERE period_type = 'annual'
                  AND fiscal_date_ending >= CURRENT_DATE - INTERVAL '1 year'
                ORDER BY symbol, fiscal_date_ending DESC
            ),
            historical_fcf AS (
                SELECT 
                    symbol,
                    fiscal_date_ending,
                    free_cash_flow,
                    revenue,
                    EXTRACT(YEAR FROM fiscal_date_ending) as year
                FROM fundamentals
                WHERE period_type = 'annual'
                  AND fiscal_date_ending >= CURRENT_DATE - INTERVAL '6 years'
                  AND free_cash_flow IS NOT NULL
            ),
            market_caps AS (
                SELECT symbol, market_cap
                FROM symbol_universe
                WHERE market_cap > 0
            ),
            fcf_metrics AS (
                SELECT 
                    lf.symbol,
                    lf.free_cash_flow,
                    lf.free_cash_flow / NULLIF(mc.market_cap, 0) * 100 as fcf_yield,
                    lf.free_cash_flow / NULLIF(lf.revenue, 0) * 100 as fcf_margin,
                    lf.free_cash_flow / NULLIF(lf.net_income, 0) as fcf_to_net_income,
                    mc.market_cap
                FROM latest_fundamentals lf
                LEFT JOIN market_caps mc ON lf.symbol = mc.symbol
                WHERE lf.free_cash_flow > 0
            )
            SELECT * FROM fcf_metrics
        """)
        
        with engine.connect() as conn:
            current_df = pd.read_sql_query(query, conn)
        
        # Get historical FCF for growth calculation
        hist_query = text("""
            SELECT 
                symbol,
                fiscal_date_ending,
                free_cash_flow,
                revenue
            FROM fundamentals
            WHERE period_type = 'annual'
              AND fiscal_date_ending >= CURRENT_DATE - INTERVAL '6 years'
              AND free_cash_flow IS NOT NULL
              AND free_cash_flow > 0
            ORDER BY symbol, fiscal_date_ending DESC
        """)
        
        with engine.connect() as conn:
            hist_df = pd.read_sql_query(hist_query, conn)
        
        rankings = []
        for symbol in current_df['symbol'].unique():
            symbol_current = current_df[current_df['symbol'] == symbol].iloc[0]
            symbol_hist = hist_df[hist_df['symbol'] == symbol].sort_values('fiscal_date_ending')
            
            # Calculate FCF growth (3-year CAGR)
            fcf_growth_3yr = 0
            if len(symbol_hist) >= 4:
                recent_fcf = symbol_hist.iloc[-1]['free_cash_flow']
                three_yr_ago_fcf = symbol_hist.iloc[-4]['free_cash_flow']
                if three_yr_ago_fcf > 0 and recent_fcf > 0:
                    fcf_growth_3yr = (((recent_fcf / three_yr_ago_fcf) ** (1/3)) - 1) * 100
            
            # Calculate FCF consistency (lower standard deviation is better)
            fcf_consistency = 0
            if len(symbol_hist) >= 3:
                fcf_values = symbol_hist['free_cash_flow'].values
                if len(fcf_values) > 1:
                    cv = np.std(fcf_values) / np.mean(fcf_values)  # Coefficient of variation
                    fcf_consistency = 100 / (1 + cv)  # Higher score for lower variation
            
            rankings.append({
                'symbol': symbol,
                'fcf_yield': symbol_current['fcf_yield'] if pd.notna(symbol_current['fcf_yield']) else 0,
                'fcf_margin': symbol_current['fcf_margin'] if pd.notna(symbol_current['fcf_margin']) else 0,
                'fcf_growth_3yr': fcf_growth_3yr,
                'fcf_consistency_score': fcf_consistency,
                'fcf_to_net_income': symbol_current['fcf_to_net_income'] if pd.notna(symbol_current['fcf_to_net_income']) else 0
            })
        
        # Create DataFrame and rank
        rank_df = pd.DataFrame(rankings)
        
        # Create composite score for ranking
        # Normalize each metric to 0-100 scale
        for col in ['fcf_yield', 'fcf_margin', 'fcf_growth_3yr', 'fcf_consistency_score']:
            if col in rank_df.columns and len(rank_df) > 0:
                min_val = rank_df[col].min()
                max_val = rank_df[col].max()
                if max_val > min_val:
                    rank_df[f'{col}_normalized'] = ((rank_df[col] - min_val) / (max_val - min_val)) * 100
                else:
                    rank_df[f'{col}_normalized'] = 50
        
        # Composite score (weighted average)
        rank_df['excess_cash_flow_score'] = (
            rank_df.get('fcf_yield_normalized', 0) * 0.35 +  # 35% weight on yield
            rank_df.get('fcf_margin_normalized', 0) * 0.25 +  # 25% on margin
            rank_df.get('fcf_growth_3yr_normalized', 0) * 0.25 +  # 25% on growth
            rank_df.get('fcf_consistency_score_normalized', 0) * 0.15  # 15% on consistency
        )
        
        rank_df['excess_cash_flow_ranking'] = rank_df['excess_cash_flow_score'].rank(ascending=False, method='min').astype(int)
        
        return rank_df
    
    def calculate_fundamentals_trend(self):
        """Calculate fundamentals trend rankings"""
        print("[INFO] Calculating fundamentals trend rankings...")
        
        # Get price trends
        price_query = text("""
            WITH price_data AS (
                SELECT 
                    symbol,
                    trade_date,
                    adjusted_close
                FROM stock_prices
                WHERE trade_date >= CURRENT_DATE - INTERVAL '11 years'
                  AND adjusted_close > 0
            ),
            price_metrics AS (
                SELECT 
                    symbol,
                    -- 10-year return
                    (LAST_VALUE(adjusted_close) FILTER (WHERE trade_date >= CURRENT_DATE - INTERVAL '30 days') 
                        OVER (PARTITION BY symbol) /
                     NULLIF(FIRST_VALUE(adjusted_close) FILTER (WHERE trade_date <= CURRENT_DATE - INTERVAL '10 years' + INTERVAL '30 days')
                        OVER (PARTITION BY symbol), 0) - 1) * 100 as price_change_10yr,
                    -- 5-year return
                    (LAST_VALUE(adjusted_close) FILTER (WHERE trade_date >= CURRENT_DATE - INTERVAL '30 days')
                        OVER (PARTITION BY symbol) /
                     NULLIF(FIRST_VALUE(adjusted_close) FILTER (WHERE trade_date <= CURRENT_DATE - INTERVAL '5 years' + INTERVAL '30 days')
                        OVER (PARTITION BY symbol), 0) - 1) * 100 as price_change_5yr,
                    -- 1-year return
                    (LAST_VALUE(adjusted_close) FILTER (WHERE trade_date >= CURRENT_DATE - INTERVAL '7 days')
                        OVER (PARTITION BY symbol) /
                     NULLIF(FIRST_VALUE(adjusted_close) FILTER (WHERE trade_date <= CURRENT_DATE - INTERVAL '1 year' + INTERVAL '7 days')
                        OVER (PARTITION BY symbol), 0) - 1) * 100 as price_change_1yr
                FROM price_data
            )
            SELECT DISTINCT 
                symbol,
                price_change_10yr,
                price_change_5yr,
                price_change_1yr
            FROM price_metrics
            WHERE price_change_10yr IS NOT NULL
        """)
        
        # Get fundamental trends
        fundamental_query = text("""
            WITH fundamental_history AS (
                SELECT 
                    symbol,
                    fiscal_date_ending,
                    revenue,
                    net_income,
                    net_margin,
                    return_on_equity as roe,
                    operating_cash_flow,
                    free_cash_flow
                FROM fundamentals
                WHERE period_type = 'annual'
                  AND fiscal_date_ending >= CURRENT_DATE - INTERVAL '11 years'
            ),
            fundamental_trends AS (
                SELECT 
                    symbol,
                    -- Current metrics
                    LAST_VALUE(net_margin) OVER (PARTITION BY symbol ORDER BY fiscal_date_ending
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as current_net_margin,
                    LAST_VALUE(roe) OVER (PARTITION BY symbol ORDER BY fiscal_date_ending
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as roe_current,
                    -- 5-year ago metrics
                    FIRST_VALUE(net_margin) FILTER (WHERE fiscal_date_ending <= CURRENT_DATE - INTERVAL '5 years')
                        OVER (PARTITION BY symbol ORDER BY fiscal_date_ending DESC) as net_margin_5yr_ago,
                    FIRST_VALUE(roe) FILTER (WHERE fiscal_date_ending <= CURRENT_DATE - INTERVAL '5 years')
                        OVER (PARTITION BY symbol ORDER BY fiscal_date_ending DESC) as roe_5yr_ago,
                    -- Revenue metrics
                    LAST_VALUE(revenue) OVER (PARTITION BY symbol ORDER BY fiscal_date_ending
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as current_revenue,
                    FIRST_VALUE(revenue) FILTER (WHERE fiscal_date_ending <= CURRENT_DATE - INTERVAL '10 years')
                        OVER (PARTITION BY symbol ORDER BY fiscal_date_ending DESC) as revenue_10yr_ago,
                    FIRST_VALUE(revenue) FILTER (WHERE fiscal_date_ending <= CURRENT_DATE - INTERVAL '5 years')
                        OVER (PARTITION BY symbol ORDER BY fiscal_date_ending DESC) as revenue_5yr_ago,
                    FIRST_VALUE(revenue) FILTER (WHERE fiscal_date_ending <= CURRENT_DATE - INTERVAL '1 year')
                        OVER (PARTITION BY symbol ORDER BY fiscal_date_ending DESC) as revenue_1yr_ago,
                    -- Cash flow
                    AVG(operating_cash_flow) OVER (PARTITION BY symbol) as avg_operating_cf
                FROM fundamental_history
            )
            SELECT DISTINCT
                symbol,
                current_net_margin,
                roe_current,
                current_net_margin - net_margin_5yr_ago as margin_change_5yr,
                CASE 
                    WHEN current_net_margin > net_margin_5yr_ago * 1.1 THEN 'Expanding'
                    WHEN current_net_margin < net_margin_5yr_ago * 0.9 THEN 'Contracting'
                    ELSE 'Stable'
                END as margin_trend_5yr,
                CASE
                    WHEN roe_current > roe_5yr_ago * 1.1 THEN 'Improving'
                    WHEN roe_current < roe_5yr_ago * 0.9 THEN 'Declining'
                    ELSE 'Stable'
                END as roe_trend_5yr,
                CASE 
                    WHEN revenue_10yr_ago > 0 THEN 
                        (POW(current_revenue / revenue_10yr_ago, 1.0/10) - 1) * 100
                    ELSE NULL
                END as revenue_growth_10yr,
                CASE 
                    WHEN revenue_5yr_ago > 0 THEN 
                        (POW(current_revenue / revenue_5yr_ago, 1.0/5) - 1) * 100
                    ELSE NULL
                END as revenue_growth_5yr,
                CASE 
                    WHEN revenue_1yr_ago > 0 THEN 
                        ((current_revenue / revenue_1yr_ago) - 1) * 100
                    ELSE NULL
                END as revenue_growth_1yr
            FROM fundamental_trends
            WHERE current_revenue IS NOT NULL
        """)
        
        with engine.connect() as conn:
            price_df = pd.read_sql_query(price_query, conn)
            fundamental_df = pd.read_sql_query(fundamental_query, conn)
        
        # Merge dataframes
        df = pd.merge(price_df, fundamental_df, on='symbol', how='inner')
        
        # Calculate price momentum score (recent performance weighted higher)
        df['price_momentum_score'] = (
            df['price_change_10yr'] * 0.2 +
            df['price_change_5yr'] * 0.3 +
            df['price_change_1yr'] * 0.5
        )
        
        # Determine revenue trend
        def get_revenue_trend(row):
            if pd.isna(row['revenue_growth_5yr']) or pd.isna(row['revenue_growth_1yr']):
                return 'Unknown'
            if row['revenue_growth_1yr'] > row['revenue_growth_5yr'] * 1.2:
                return 'Accelerating'
            elif row['revenue_growth_1yr'] > 0 and row['revenue_growth_5yr'] > 0:
                return 'Stable Growth'
            elif row['revenue_growth_1yr'] < row['revenue_growth_5yr'] * 0.8:
                return 'Decelerating'
            else:
                return 'Declining'
        
        df['revenue_trend'] = df.apply(get_revenue_trend, axis=1)
        
        # Create composite score for fundamentals
        df['fundamentals_score'] = 0
        
        # Price trend component (40% weight)
        if 'price_momentum_score' in df.columns:
            price_norm = (df['price_momentum_score'] - df['price_momentum_score'].min()) / (df['price_momentum_score'].max() - df['price_momentum_score'].min() + 0.001)
            df['fundamentals_score'] += price_norm * 40
        
        # Revenue growth component (30% weight)
        if 'revenue_growth_5yr' in df.columns:
            revenue_norm = (df['revenue_growth_5yr'] - df['revenue_growth_5yr'].min()) / (df['revenue_growth_5yr'].max() - df['revenue_growth_5yr'].min() + 0.001)
            df['fundamentals_score'] += revenue_norm * 30
        
        # Profitability component (30% weight)
        if 'roe_current' in df.columns:
            roe_norm = (df['roe_current'] - df['roe_current'].min()) / (df['roe_current'].max() - df['roe_current'].min() + 0.001)
            df['fundamentals_score'] += roe_norm * 15
        if 'current_net_margin' in df.columns:
            margin_norm = (df['current_net_margin'] - df['current_net_margin'].min()) / (df['current_net_margin'].max() - df['current_net_margin'].min() + 0.001)
            df['fundamentals_score'] += margin_norm * 15
        
        # Calculate ranking
        df['fundamentals_ranking'] = df['fundamentals_score'].rank(ascending=False, method='min').astype(int)
        
        # Convert trend columns to text
        df['margin_trend_5yr'] = df.get('margin_trend_5yr', 'Unknown')
        df['roe_trend_5yr'] = df.get('roe_trend_5yr', 'Unknown')
        df['cf_trend'] = 'Unknown'  # Placeholder, could be calculated
        
        # Rename columns for output
        df['price_trend_10yr'] = df['price_change_10yr']
        df['price_trend_5yr'] = df['price_change_5yr']
        df['price_trend_1yr'] = df['price_change_1yr']
        
        return df
    
    def calculate_composite_rankings(self):
        """Combine all rankings into final quality rankings"""
        print("[INFO] Calculating composite quality rankings...")
        
        # Get all three ranking components
        sp500_df = self.calculate_sp500_outperformance()
        cash_flow_df = self.calculate_excess_cash_flow()
        fundamentals_df = self.calculate_fundamentals_trend()
        
        # Get sector and market cap data
        sector_query = text("""
            SELECT symbol, sector, industry, market_cap
            FROM symbol_universe
        """)
        
        with engine.connect() as conn:
            sector_df = pd.read_sql_query(sector_query, conn)
        
        # Merge all dataframes
        result_df = sp500_df.merge(cash_flow_df, on='symbol', how='outer')
        result_df = result_df.merge(fundamentals_df, on='symbol', how='outer')
        result_df = result_df.merge(sector_df, on='symbol', how='left')
        
        # Fill NaN rankings with worst rank + 1
        max_symbols = len(result_df)
        result_df['beat_sp500_ranking'] = result_df['beat_sp500_ranking'].fillna(max_symbols + 1)
        result_df['excess_cash_flow_ranking'] = result_df['excess_cash_flow_ranking'].fillna(max_symbols + 1)
        result_df['fundamentals_ranking'] = result_df['fundamentals_ranking'].fillna(max_symbols + 1)
        
        # Calculate composite score (equal weight for now, can be adjusted)
        result_df['composite_quality_score'] = (
            (1 - result_df['beat_sp500_ranking'] / max_symbols) * 100 * 0.4 +  # 40% weight
            (1 - result_df['excess_cash_flow_ranking'] / max_symbols) * 100 * 0.3 +  # 30% weight
            (1 - result_df['fundamentals_ranking'] / max_symbols) * 100 * 0.3  # 30% weight
        )
        
        # Determine quality tier
        result_df['quality_tier'] = pd.cut(
            result_df['composite_quality_score'],
            bins=[-np.inf, 40, 55, 70, 85, np.inf],
            labels=['Below', 'Standard', 'Quality', 'Premium', 'Elite']
        )
        
        # Set flags
        result_df['is_sp500_beater'] = result_df['years_beating_sp500'] >= 10  # Beat SP500 at least 10 of 20 years
        result_df['is_cash_generator'] = result_df['excess_cash_flow_ranking'] <= max_symbols * 0.25  # Top quartile
        result_df['is_fundamental_grower'] = result_df['fundamentals_ranking'] <= max_symbols * 0.25  # Top quartile
        
        # Add ranking date
        result_df['ranking_date'] = self.ranking_date
        
        return result_df
    
    def save_rankings(self, df):
        """Save rankings to database"""
        print(f"[INFO] Saving {len(df)} stock rankings to database...")
        
        # Select columns for database
        columns = [
            'symbol', 'ranking_date',
            'beat_sp500_ranking', 'years_beating_sp500', 'sp500_weighted_score',
            'avg_annual_excess', 'recent_5yr_beat_count', 'recent_1yr_excess',
            'excess_cash_flow_ranking', 'fcf_yield', 'fcf_margin', 'fcf_growth_3yr',
            'fcf_consistency_score', 'fcf_to_net_income',
            'fundamentals_ranking', 'price_trend_10yr', 'price_trend_5yr', 'price_trend_1yr',
            'price_momentum_score', 'revenue_growth_10yr', 'revenue_growth_5yr',
            'revenue_growth_1yr', 'revenue_trend', 'margin_trend_5yr',
            'current_net_margin', 'margin_change_5yr', 'roe_current', 'roe_trend_5yr',
            'composite_quality_score', 'quality_tier',
            'is_sp500_beater', 'is_cash_generator', 'is_fundamental_grower',
            'sector', 'industry', 'market_cap'
        ]
        
        # Only keep columns that exist
        save_columns = [col for col in columns if col in df.columns]
        save_df = df[save_columns].copy()
        
        # Replace 'Unknown' with None for text columns that might cause issues
        text_cols = ['revenue_trend', 'margin_trend_5yr', 'roe_trend_5yr', 'cf_trend']
        for col in text_cols:
            if col in save_df.columns:
                save_df[col] = save_df[col].replace('Unknown', None)
        
        # Save to database
        try:
            save_df.to_sql('stock_quality_rankings', engine, if_exists='append', index=False, method='multi')
            print(f"[SUCCESS] Saved {len(save_df)} rankings")
        except Exception as e:
            logger.error(f"Error saving rankings: {e}")
            print(f"[ERROR] Failed to save rankings: {e}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate stock quality rankings")
    parser.add_argument("--limit", type=int, help="Limit number of stocks to rank")
    parser.add_argument("--save", action="store_true", help="Save rankings to database")
    args = parser.parse_args()
    
    start_time = time.time()
    print("\n" + "=" * 60)
    print("STOCK QUALITY RANKINGS CALCULATOR")
    print("=" * 60)
    
    calculator = QualityRankingCalculator()
    
    # Calculate rankings
    rankings_df = calculator.calculate_composite_rankings()
    
    # Apply limit if specified
    if args.limit:
        rankings_df = rankings_df.head(args.limit)
    
    # Display top stocks
    print("\n[TOP ELITE STOCKS]")
    elite = rankings_df[rankings_df['quality_tier'] == 'Elite'].head(20)
    if not elite.empty:
        print(elite[['symbol', 'beat_sp500_ranking', 'excess_cash_flow_ranking', 
                    'fundamentals_ranking', 'composite_quality_score']].to_string())
    
    # Save if requested
    if args.save:
        calculator.save_rankings(rankings_df)
    
    duration = time.time() - start_time
    print(f"\n[COMPLETE] Ranked {len(rankings_df)} stocks in {duration:.1f}s")

if __name__ == "__main__":
    main()