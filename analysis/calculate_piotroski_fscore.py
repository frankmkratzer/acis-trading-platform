#!/usr/bin/env python3
"""
Piotroski F-Score Calculator
Calculates 9-point fundamental strength score for all stocks
Academic research shows F-Score 8-9 stocks outperform by 7.5% annually
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import text
from pathlib import Path
from tqdm import tqdm

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.logging_config import setup_logger, log_script_start, log_script_end
from database.db_connection_manager import DatabaseConnectionManager

# Initialize centralized utilities
logger = setup_logger("calculate_piotroski_fscore")
db_manager = DatabaseConnectionManager()
engine = db_manager.get_engine()

def get_fundamental_data():
    """Get fundamental data needed for F-Score calculation"""
    query = text("""
        WITH latest_quarters AS (
            -- Get last 8 quarters for each symbol for year-over-year comparisons
            SELECT 
                f.symbol,
                f.fiscal_date_ending,
                f.period_type,
                ROW_NUMBER() OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date_ending DESC) as quarter_rank
            FROM fundamentals f
            WHERE f.period_type = 'quarterly'
        ),
        quarterly_data AS (
            SELECT 
                f.symbol,
                f.fiscal_date_ending,
                lq.quarter_rank,
                
                -- Income Statement items
                f.net_income_ttm as net_income,
                f.total_revenue_ttm as revenue,
                
                -- Cash Flow items
                f.operating_cash_flow,
                f.free_cash_flow,
                
                -- Balance Sheet items
                f.total_assets,
                f.total_liabilities,
                f.total_shareholder_equity,
                f.total_debt,
                f.shares_outstanding,
                
                -- Get current stock price for market cap
                sp.latest_price,
                sp.latest_price * f.shares_outstanding as market_cap
                
            FROM fundamentals f
            INNER JOIN latest_quarters lq 
                ON f.symbol = lq.symbol 
                AND f.fiscal_date_ending = lq.fiscal_date_ending
            LEFT JOIN (
                SELECT 
                    symbol, 
                    close as latest_price
                FROM stock_prices
                WHERE (symbol, trade_date) IN (
                    SELECT symbol, MAX(trade_date)
                    FROM stock_prices
                    GROUP BY symbol
                )
            ) sp ON f.symbol = sp.symbol
            WHERE lq.quarter_rank <= 8  -- Get last 2 years of data
        ),
        current_vs_prior AS (
            SELECT 
                c.symbol,
                c.fiscal_date_ending as current_date,
                p4.fiscal_date_ending as year_ago_date,
                
                -- Current quarter data
                c.net_income as current_net_income,
                c.revenue as current_revenue,
                c.operating_cash_flow as current_ocf,
                c.total_assets as current_assets,
                c.total_liabilities as current_liabilities,
                c.total_shareholder_equity as current_equity,
                c.total_debt as current_debt,
                c.shares_outstanding as current_shares,
                c.market_cap as current_market_cap,
                
                -- Year-ago data (4 quarters back)
                p4.net_income as prior_net_income,
                p4.revenue as prior_revenue,
                p4.operating_cash_flow as prior_ocf,
                p4.total_assets as prior_assets,
                p4.total_liabilities as prior_liabilities,
                p4.total_shareholder_equity as prior_equity,
                p4.total_debt as prior_debt,
                p4.shares_outstanding as prior_shares,
                
                -- Two quarters ago for current ratio comparison
                p1.total_assets as prev_quarter_assets,
                p1.total_liabilities as prev_quarter_liabilities
                
            FROM quarterly_data c
            LEFT JOIN quarterly_data p4 
                ON c.symbol = p4.symbol 
                AND p4.quarter_rank = c.quarter_rank + 4  -- Year ago
            LEFT JOIN quarterly_data p1
                ON c.symbol = p1.symbol
                AND p1.quarter_rank = c.quarter_rank + 1  -- Previous quarter
            WHERE c.quarter_rank = 1  -- Most recent quarter only
        )
        SELECT 
            cvp.*,
            su.sector,
            su.industry,
            su.market_cap as universe_market_cap
        FROM current_vs_prior cvp
        INNER JOIN symbol_universe su ON cvp.symbol = su.symbol
        WHERE su.is_etf = FALSE
        AND su.country = 'USA'
        AND su.market_cap >= 2000000000  -- Mid-cap and above
        ORDER BY cvp.symbol
    """)
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    
    return df

def calculate_piotroski_score(row):
    """
    Calculate Piotroski F-Score (0-9 points)
    
    PROFITABILITY (4 points):
    1. Net Income > 0
    2. Operating Cash Flow > 0  
    3. ROA increasing (Net Income / Assets)
    4. Quality of Earnings (OCF > Net Income)
    
    LEVERAGE/LIQUIDITY (3 points):
    5. Lower Debt/Assets ratio
    6. Higher Current Ratio
    7. No new shares issued
    
    EFFICIENCY (2 points):
    8. Higher Gross Margin
    9. Higher Asset Turnover
    """
    
    score = 0
    details = {}
    
    # === PROFITABILITY (4 points) ===
    
    # 1. Positive Net Income
    if pd.notna(row['current_net_income']) and row['current_net_income'] > 0:
        score += 1
        details['positive_net_income'] = True
    else:
        details['positive_net_income'] = False
    
    # 2. Positive Operating Cash Flow
    if pd.notna(row['current_ocf']) and row['current_ocf'] > 0:
        score += 1
        details['positive_ocf'] = True
    else:
        details['positive_ocf'] = False
    
    # 3. Increasing ROA (Return on Assets)
    if (pd.notna(row['current_net_income']) and pd.notna(row['current_assets']) and 
        pd.notna(row['prior_net_income']) and pd.notna(row['prior_assets']) and
        row['current_assets'] > 0 and row['prior_assets'] > 0):
        
        current_roa = row['current_net_income'] / row['current_assets']
        prior_roa = row['prior_net_income'] / row['prior_assets']
        
        if current_roa > prior_roa:
            score += 1
            details['increasing_roa'] = True
        else:
            details['increasing_roa'] = False
        
        details['current_roa'] = current_roa * 100
        details['prior_roa'] = prior_roa * 100
    else:
        details['increasing_roa'] = False
    
    # 4. Quality of Earnings (OCF > Net Income)
    if (pd.notna(row['current_ocf']) and pd.notna(row['current_net_income']) and
        row['current_ocf'] > row['current_net_income']):
        score += 1
        details['quality_earnings'] = True
    else:
        details['quality_earnings'] = False
    
    # === LEVERAGE/LIQUIDITY (3 points) ===
    
    # 5. Lower Debt/Assets Ratio
    if (pd.notna(row['current_debt']) and pd.notna(row['current_assets']) and
        pd.notna(row['prior_debt']) and pd.notna(row['prior_assets']) and
        row['current_assets'] > 0 and row['prior_assets'] > 0):
        
        current_debt_ratio = row['current_debt'] / row['current_assets']
        prior_debt_ratio = row['prior_debt'] / row['prior_assets']
        
        if current_debt_ratio < prior_debt_ratio:
            score += 1
            details['decreasing_leverage'] = True
        else:
            details['decreasing_leverage'] = False
        
        details['current_debt_ratio'] = current_debt_ratio
        details['prior_debt_ratio'] = prior_debt_ratio
    else:
        details['decreasing_leverage'] = False
    
    # 6. Higher Current Ratio (simplified - using total assets/liabilities as proxy)
    if (pd.notna(row['current_assets']) and pd.notna(row['current_liabilities']) and
        pd.notna(row['prev_quarter_assets']) and pd.notna(row['prev_quarter_liabilities']) and
        row['current_liabilities'] > 0 and row['prev_quarter_liabilities'] > 0):
        
        current_ratio = row['current_assets'] / row['current_liabilities']
        prev_ratio = row['prev_quarter_assets'] / row['prev_quarter_liabilities']
        
        if current_ratio > prev_ratio:
            score += 1
            details['improving_liquidity'] = True
        else:
            details['improving_liquidity'] = False
        
        details['current_ratio'] = current_ratio
    else:
        details['improving_liquidity'] = False
    
    # 7. No New Shares Issued
    if (pd.notna(row['current_shares']) and pd.notna(row['prior_shares']) and
        row['current_shares'] <= row['prior_shares'] * 1.02):  # Allow 2% tolerance
        score += 1
        details['no_dilution'] = True
    else:
        details['no_dilution'] = False
    
    # === EFFICIENCY (2 points) ===
    
    # 8. Higher Gross Margin (using operating margin as proxy)
    if (pd.notna(row['current_net_income']) and pd.notna(row['current_revenue']) and
        pd.notna(row['prior_net_income']) and pd.notna(row['prior_revenue']) and
        row['current_revenue'] > 0 and row['prior_revenue'] > 0):
        
        current_margin = row['current_net_income'] / row['current_revenue']
        prior_margin = row['prior_net_income'] / row['prior_revenue']
        
        if current_margin > prior_margin:
            score += 1
            details['improving_margin'] = True
        else:
            details['improving_margin'] = False
        
        details['current_margin'] = current_margin * 100
        details['prior_margin'] = prior_margin * 100
    else:
        details['improving_margin'] = False
    
    # 9. Higher Asset Turnover
    if (pd.notna(row['current_revenue']) and pd.notna(row['current_assets']) and
        pd.notna(row['prior_revenue']) and pd.notna(row['prior_assets']) and
        row['current_assets'] > 0 and row['prior_assets'] > 0):
        
        current_turnover = row['current_revenue'] / row['current_assets']
        prior_turnover = row['prior_revenue'] / row['prior_assets']
        
        if current_turnover > prior_turnover:
            score += 1
            details['improving_efficiency'] = True
        else:
            details['improving_efficiency'] = False
        
        details['current_turnover'] = current_turnover
        details['prior_turnover'] = prior_turnover
    else:
        details['improving_efficiency'] = False
    
    return score, details

def calculate_all_fscores():
    """Calculate F-Scores for all eligible stocks"""
    logger.info("Fetching fundamental data...")
    
    df = get_fundamental_data()
    
    if df.empty:
        logger.warning("No fundamental data found")
        return pd.DataFrame()
    
    logger.info(f"Calculating F-Scores for {len(df)} stocks...")
    
    results = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Calculating F-Scores"):
        score, details = calculate_piotroski_score(row)
        
        result = {
            'symbol': row['symbol'],
            'calculation_date': datetime.now().date(),
            'fscore': score,
            'fiscal_date': row['current_date'],
            
            # Profitability components
            'positive_net_income': details.get('positive_net_income', False),
            'positive_ocf': details.get('positive_ocf', False),
            'increasing_roa': details.get('increasing_roa', False),
            'quality_earnings': details.get('quality_earnings', False),
            
            # Leverage/Liquidity components
            'decreasing_leverage': details.get('decreasing_leverage', False),
            'improving_liquidity': details.get('improving_liquidity', False),
            'no_dilution': details.get('no_dilution', False),
            
            # Efficiency components
            'improving_margin': details.get('improving_margin', False),
            'improving_efficiency': details.get('improving_efficiency', False),
            
            # Metrics
            'current_roa': details.get('current_roa'),
            'prior_roa': details.get('prior_roa'),
            'current_margin': details.get('current_margin'),
            'prior_margin': details.get('prior_margin'),
            'current_debt_ratio': details.get('current_debt_ratio'),
            'current_ratio': details.get('current_ratio'),
            'current_turnover': details.get('current_turnover'),
            
            # Classification
            'strength_category': 'STRONG' if score >= 7 else 'MODERATE' if score >= 4 else 'WEAK',
            
            'updated_at': datetime.now()
        }
        
        results.append(result)
    
    return pd.DataFrame(results)

def upsert_fscores(df):
    """Upsert F-Scores to database"""
    if df.empty:
        return 0
    
    try:
        with engine.begin() as conn:
            # Create temp table
            temp_table = f"temp_piotroski_{int(time.time() * 1000)}"
            df.to_sql(temp_table, conn, if_exists='replace', index=False, method='multi', chunksize=1000)
            
            # Upsert from temp table
            conn.execute(text(f"""
                INSERT INTO piotroski_scores (
                    symbol, calculation_date, fscore, fiscal_date,
                    positive_net_income, positive_ocf, increasing_roa, quality_earnings,
                    decreasing_leverage, improving_liquidity, no_dilution,
                    improving_margin, improving_efficiency,
                    current_roa, prior_roa, current_margin, prior_margin,
                    current_debt_ratio, current_ratio, current_turnover,
                    strength_category, updated_at
                )
                SELECT 
                    symbol, calculation_date::date, fscore::int, fiscal_date::date,
                    positive_net_income::boolean, positive_ocf::boolean, 
                    increasing_roa::boolean, quality_earnings::boolean,
                    decreasing_leverage::boolean, improving_liquidity::boolean, no_dilution::boolean,
                    improving_margin::boolean, improving_efficiency::boolean,
                    current_roa::numeric, prior_roa::numeric, 
                    current_margin::numeric, prior_margin::numeric,
                    current_debt_ratio::numeric, current_ratio::numeric, current_turnover::numeric,
                    strength_category, updated_at
                FROM {temp_table}
                ON CONFLICT (symbol, calculation_date) DO UPDATE SET
                    fscore = EXCLUDED.fscore,
                    fiscal_date = EXCLUDED.fiscal_date,
                    positive_net_income = EXCLUDED.positive_net_income,
                    positive_ocf = EXCLUDED.positive_ocf,
                    increasing_roa = EXCLUDED.increasing_roa,
                    quality_earnings = EXCLUDED.quality_earnings,
                    decreasing_leverage = EXCLUDED.decreasing_leverage,
                    improving_liquidity = EXCLUDED.improving_liquidity,
                    no_dilution = EXCLUDED.no_dilution,
                    improving_margin = EXCLUDED.improving_margin,
                    improving_efficiency = EXCLUDED.improving_efficiency,
                    current_roa = EXCLUDED.current_roa,
                    prior_roa = EXCLUDED.prior_roa,
                    current_margin = EXCLUDED.current_margin,
                    prior_margin = EXCLUDED.prior_margin,
                    current_debt_ratio = EXCLUDED.current_debt_ratio,
                    current_ratio = EXCLUDED.current_ratio,
                    current_turnover = EXCLUDED.current_turnover,
                    strength_category = EXCLUDED.strength_category,
                    updated_at = EXCLUDED.updated_at
            """))
            
            # Drop temp table
            conn.execute(text(f"DROP TABLE {temp_table}"))
            
            return len(df)
            
    except Exception as e:
        logger.error(f"Error upserting F-Scores: {e}")
        return 0

def analyze_results(df):
    """Analyze and display F-Score distribution"""
    if df.empty:
        return
    
    print("\n" + "=" * 60)
    print("PIOTROSKI F-SCORE ANALYSIS")
    print("=" * 60)
    
    # Score distribution
    score_dist = df['fscore'].value_counts().sort_index()
    print("\nScore Distribution:")
    for score, count in score_dist.items():
        pct = (count / len(df)) * 100
        bar = "█" * int(pct / 2)
        print(f"  Score {score}: {count:4d} ({pct:5.1f}%) {bar}")
    
    # Category breakdown
    print("\nStrength Categories:")
    categories = df['strength_category'].value_counts()
    for category, count in categories.items():
        pct = (count / len(df)) * 100
        print(f"  {category:10s}: {count:4d} ({pct:5.1f}%)")
    
    # Top F-Score stocks
    top_stocks = df[df['fscore'] >= 8].sort_values('fscore', ascending=False)
    if not top_stocks.empty:
        print(f"\nTop F-Score Stocks (8-9):")
        print(f"Found {len(top_stocks)} stocks with F-Score >= 8")
        for _, row in top_stocks.head(20).iterrows():
            print(f"  {row['symbol']:6s} - Score: {row['fscore']} ({row['strength_category']})")
    
    # Component analysis
    print("\nComponent Success Rates:")
    components = [
        'positive_net_income', 'positive_ocf', 'increasing_roa', 'quality_earnings',
        'decreasing_leverage', 'improving_liquidity', 'no_dilution',
        'improving_margin', 'improving_efficiency'
    ]
    
    for comp in components:
        if comp in df.columns:
            success_rate = (df[comp].sum() / len(df)) * 100
            print(f"  {comp:25s}: {success_rate:5.1f}%")
    
    # Sector analysis
    print("\nAverage F-Score by Sector:")
    with engine.connect() as conn:
        sector_scores = pd.read_sql(text("""
            SELECT 
                su.sector,
                COUNT(*) as count,
                AVG(ps.fscore) as avg_fscore
            FROM piotroski_scores ps
            JOIN symbol_universe su ON ps.symbol = su.symbol
            WHERE ps.calculation_date = (SELECT MAX(calculation_date) FROM piotroski_scores)
            AND su.sector IS NOT NULL
            GROUP BY su.sector
            ORDER BY avg_fscore DESC
        """), conn)
        
        if not sector_scores.empty:
            for _, row in sector_scores.iterrows():
                print(f"  {row['sector']:25s}: {row['avg_fscore']:.2f} (n={row['count']})")

def main():
    """Main execution"""
    start_time = time.time()
    log_script_start(logger, "calculate_piotroski_fscore", "Calculating Piotroski F-Scores")
    
    print("\n" + "=" * 60)
    print("PIOTROSKI F-SCORE CALCULATOR")
    print("=" * 60)
    print("Fundamental Strength Score (0-9 points)")
    print("8-9 = Strong Buy | 0-2 = Strong Avoid")
    print("=" * 60)
    
    # Calculate F-Scores
    df = calculate_all_fscores()
    
    if df.empty:
        logger.error("No F-Scores calculated")
        return
    
    # Filter out invalid scores
    df = df[df['fscore'].between(0, 9)]
    
    print(f"\n[INFO] Calculated F-Scores for {len(df)} stocks")
    
    # Upsert to database
    count = upsert_fscores(df)
    print(f"[INFO] Upserted {count} F-Scores to database")
    
    # Analyze results
    analyze_results(df)
    
    # Summary statistics
    duration = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("EXECUTION SUMMARY")
    print("=" * 60)
    print(f"Total stocks analyzed: {len(df)}")
    print(f"Average F-Score: {df['fscore'].mean():.2f}")
    print(f"Stocks with F-Score >= 8: {len(df[df['fscore'] >= 8])}")
    print(f"Stocks with F-Score <= 2: {len(df[df['fscore'] <= 2])}")
    print(f"Duration: {duration:.1f} seconds")
    
    log_script_end(logger, "calculate_piotroski_fscore", True, duration, {
        "stocks_analyzed": len(df),
        "avg_fscore": df['fscore'].mean(),
        "high_score_count": len(df[df['fscore'] >= 8])
    })

if __name__ == "__main__":
    main()