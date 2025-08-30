"""
Calculate Beneish M-Score for earnings manipulation detection.

The Beneish M-Score is an 8-variable model to detect earnings manipulation:
- M-Score < -2.22: Unlikely to be manipulator
- M-Score > -2.22: Possible earnings manipulator

Variables:
1. Days Sales in Receivables Index (DSRI)
2. Gross Margin Index (GMI)
3. Asset Quality Index (AQI)
4. Sales Growth Index (SGI)
5. Depreciation Index (DEPI)
6. Sales General & Admin Index (SGAI)
7. Leverage Index (LVGI)
8. Total Accruals to Total Assets (TATA)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
from tqdm import tqdm
import time

# Setup logging first
from utils.logging_config import setup_logger, log_script_start, log_script_end
logger = setup_logger("calculate_beneish_mscore")

# Database connection
from database.db_connection_manager import DatabaseConnectionManager

def fetch_financial_data(engine):
    """Fetch financial data needed for M-Score calculation."""
    
    query = """
    WITH financial_data AS (
        SELECT 
            f.symbol,
            f.fiscal_date_ending,
            f.period_type,
            
            -- Income Statement items
            f.total_revenue_ttm as revenue,
            f.gross_profit_ttm as gross_profit,
            f.operating_income_ttm as operating_income,
            f.net_income_ttm as net_income,
            
            -- Balance Sheet items  
            f.total_assets,
            f.total_liabilities,
            f.total_shareholder_equity,
            f.cash_and_cash_equivalents,
            
            -- Cash Flow items
            f.operating_cash_flow,
            f.capital_expenditures,
            
            -- Calculate additional metrics
            f.total_revenue_ttm - f.gross_profit_ttm as cogs,  -- Cost of goods sold
            f.total_assets - f.cash_and_cash_equivalents as non_cash_assets,
            
            -- Get prior year data for comparison
            LAG(f.total_revenue_ttm, 4) OVER (
                PARTITION BY f.symbol 
                ORDER BY f.fiscal_date_ending
            ) as prior_revenue,
            
            LAG(f.gross_profit_ttm, 4) OVER (
                PARTITION BY f.symbol 
                ORDER BY f.fiscal_date_ending
            ) as prior_gross_profit,
            
            LAG(f.total_assets, 4) OVER (
                PARTITION BY f.symbol 
                ORDER BY f.fiscal_date_ending
            ) as prior_assets,
            
            LAG(f.total_liabilities, 4) OVER (
                PARTITION BY f.symbol 
                ORDER BY f.fiscal_date_ending
            ) as prior_liabilities,
            
            LAG(f.net_income_ttm, 4) OVER (
                PARTITION BY f.symbol 
                ORDER BY f.fiscal_date_ending
            ) as prior_net_income,
            
            LAG(f.operating_cash_flow, 4) OVER (
                PARTITION BY f.symbol 
                ORDER BY f.fiscal_date_ending
            ) as prior_ocf,
            
            -- Rank to get latest
            ROW_NUMBER() OVER (
                PARTITION BY f.symbol 
                ORDER BY f.fiscal_date_ending DESC
            ) as rn
            
        FROM fundamentals f
        INNER JOIN symbol_universe su 
            ON f.symbol = su.symbol
        WHERE f.period_type = 'quarterly'
          AND f.total_assets > 0
          AND su.market_cap >= 2e9
          AND su.country = 'USA'
          AND su.security_type = 'Common Stock'
    )
    SELECT *
    FROM financial_data
    WHERE rn = 1
      AND prior_revenue IS NOT NULL  -- Need YoY data
      AND prior_assets IS NOT NULL
    """
    
    logger.info("Fetching financial data for M-Score calculation...")
    df = pd.read_sql(query, engine)
    logger.info(f"Retrieved data for {len(df)} companies with YoY comparisons")
    
    return df

def calculate_mscore_variables(row):
    """Calculate the 8 Beneish M-Score variables."""
    
    try:
        # Prevent division by zero
        if (row['prior_revenue'] <= 0 or row['revenue'] <= 0 or 
            row['prior_assets'] <= 0 or row['total_assets'] <= 0):
            return None
        
        # 1. Days Sales in Receivables Index (DSRI)
        # Approximation: Use revenue/assets ratio as proxy
        dsr_current = (row['revenue'] / row['total_assets']) * 365
        dsr_prior = (row['prior_revenue'] / row['prior_assets']) * 365
        dsri = dsr_current / dsr_prior if dsr_prior > 0 else 1
        
        # 2. Gross Margin Index (GMI)
        gm_prior = row['prior_gross_profit'] / row['prior_revenue'] if row['prior_revenue'] > 0 else 0
        gm_current = row['gross_profit'] / row['revenue'] if row['revenue'] > 0 else 0
        gmi = gm_prior / gm_current if gm_current > 0 else 1
        
        # 3. Asset Quality Index (AQI)
        # Non-current assets / Total assets (excluding cash)
        nca_current = (row['total_assets'] - row['cash_and_cash_equivalents']) / row['total_assets']
        nca_prior = (row['prior_assets'] - row.get('prior_cash', 0)) / row['prior_assets']
        aqi = nca_current / nca_prior if nca_prior > 0 else 1
        
        # 4. Sales Growth Index (SGI)
        sgi = row['revenue'] / row['prior_revenue'] if row['prior_revenue'] > 0 else 1
        
        # 5. Depreciation Index (DEPI)
        # Approximation using CapEx as proxy
        depr_rate_prior = abs(row.get('prior_capex', 0)) / row['prior_assets'] if row['prior_assets'] > 0 else 0.05
        depr_rate_current = abs(row.get('capital_expenditures', 0)) / row['total_assets'] if row['total_assets'] > 0 else 0.05
        depi = depr_rate_prior / depr_rate_current if depr_rate_current > 0 else 1
        
        # 6. Sales General & Administrative Index (SGAI)
        # Using operating income margin as proxy
        sga_rate_current = 1 - (row['operating_income'] / row['revenue']) if row['revenue'] > 0 else 0
        sga_rate_prior = 1 - (row.get('prior_operating_income', row['operating_income']) / row['prior_revenue']) if row['prior_revenue'] > 0 else 0
        sgai = sga_rate_current / sga_rate_prior if sga_rate_prior > 0 else 1
        
        # 7. Leverage Index (LVGI)
        leverage_current = row['total_liabilities'] / row['total_assets'] if row['total_assets'] > 0 else 0
        leverage_prior = row['prior_liabilities'] / row['prior_assets'] if row['prior_assets'] > 0 else 0
        lvgi = leverage_current / leverage_prior if leverage_prior > 0 else 1
        
        # 8. Total Accruals to Total Assets (TATA)
        # (Net Income - Cash from Operations) / Total Assets
        total_accruals = (row['net_income'] - row['operating_cash_flow']) if row['operating_cash_flow'] else row['net_income']
        tata = total_accruals / row['total_assets'] if row['total_assets'] > 0 else 0
        
        return {
            'symbol': row['symbol'],
            'calculation_date': datetime.now().date(),
            'fiscal_date': row['fiscal_date_ending'],
            
            # Individual variables
            'dsri': dsri,
            'gmi': gmi,
            'aqi': aqi,
            'sgi': sgi,
            'depi': depi,
            'sgai': sgai,
            'lvgi': lvgi,
            'tata': tata,
            
            # Raw values for reference
            'revenue': row['revenue'],
            'total_assets': row['total_assets'],
            'revenue_growth': (row['revenue'] - row['prior_revenue']) / row['prior_revenue'] if row['prior_revenue'] > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"Error calculating M-Score for {row.get('symbol', 'Unknown')}: {e}")
        return None

def calculate_mscore(variables):
    """Calculate the Beneish M-Score from the 8 variables."""
    
    # Beneish M-Score formula
    m_score = (
        -4.84 +
        0.92 * variables['dsri'] +
        0.528 * variables['gmi'] +
        0.404 * variables['aqi'] +
        0.892 * variables['sgi'] +
        0.115 * variables['depi'] -
        0.172 * variables['sgai'] +
        4.679 * variables['tata'] -
        0.327 * variables['lvgi']
    )
    
    return m_score

def calculate_all_mscores(df):
    """Calculate M-Scores for all companies."""
    
    results = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Calculating M-Scores"):
        variables = calculate_mscore_variables(row)
        
        if variables:
            # Calculate M-Score
            m_score = calculate_mscore(variables)
            variables['m_score'] = m_score
            
            # Classify manipulation risk
            if m_score < -2.22:
                variables['manipulation_risk'] = 'LOW'
                variables['risk_flag'] = False
            elif m_score < -1.78:
                variables['manipulation_risk'] = 'MODERATE'
                variables['risk_flag'] = False
            else:
                variables['manipulation_risk'] = 'HIGH'
                variables['risk_flag'] = True
            
            # Calculate percentile score (0-100, higher = better)
            # Invert M-Score so lower M-Score = higher quality score
            variables['quality_score'] = max(0, min(100, 50 + ((-2.22 - m_score) * 20)))
            
            results.append(variables)
    
    return pd.DataFrame(results)

def save_mscores(engine, df):
    """Save M-Scores to database."""
    
    # Create table if not exists
    create_table_query = """
    CREATE TABLE IF NOT EXISTS beneish_mscores (
        symbol VARCHAR(10) NOT NULL,
        calculation_date DATE NOT NULL,
        fiscal_date DATE,
        
        -- M-Score variables
        dsri NUMERIC(12, 4),  -- Days Sales in Receivables Index
        gmi NUMERIC(12, 4),   -- Gross Margin Index
        aqi NUMERIC(12, 4),   -- Asset Quality Index
        sgi NUMERIC(12, 4),   -- Sales Growth Index
        depi NUMERIC(12, 4),  -- Depreciation Index
        sgai NUMERIC(12, 4),  -- SG&A Index
        lvgi NUMERIC(12, 4),  -- Leverage Index
        tata NUMERIC(12, 4),  -- Total Accruals to Total Assets
        
        -- Final score and classification
        m_score NUMERIC(12, 4),
        manipulation_risk VARCHAR(20) CHECK (manipulation_risk IN ('LOW', 'MODERATE', 'HIGH')),
        risk_flag BOOLEAN,
        quality_score NUMERIC(6, 2),
        
        -- Reference data
        revenue NUMERIC(20, 2),
        total_assets NUMERIC(20, 2),
        revenue_growth NUMERIC(12, 4),
        
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, calculation_date)
    );
    
    CREATE INDEX IF NOT EXISTS idx_beneish_mscore 
        ON beneish_mscores(m_score);
    CREATE INDEX IF NOT EXISTS idx_beneish_risk 
        ON beneish_mscores(manipulation_risk);
    CREATE INDEX IF NOT EXISTS idx_beneish_flag 
        ON beneish_mscores(risk_flag) WHERE risk_flag = TRUE;
    CREATE INDEX IF NOT EXISTS idx_beneish_quality 
        ON beneish_mscores(quality_score DESC);
    """
    
    with engine.connect() as conn:
        for statement in create_table_query.split(';'):
            if statement.strip():
                try:
                    conn.execute(text(statement))
                except Exception as e:
                    if 'already exists' not in str(e):
                        logger.error(f"Error creating table: {e}")
        conn.commit()
    
    # Save M-Scores
    if df.empty:
        logger.warning("No M-Scores to save")
        return
    
    temp_table = f"temp_beneish_{int(time.time() * 1000)}"
    
    try:
        df.to_sql(temp_table, engine, if_exists='replace', index=False)
        
        # Build column list
        cols = df.columns.tolist()
        cols_str = ', '.join(cols)
        
        with engine.connect() as conn:
            conn.execute(text(f"""
                INSERT INTO beneish_mscores ({cols_str})
                SELECT {cols_str} FROM {temp_table}
                ON CONFLICT (symbol, calculation_date) DO UPDATE SET
                    m_score = EXCLUDED.m_score,
                    manipulation_risk = EXCLUDED.manipulation_risk,
                    risk_flag = EXCLUDED.risk_flag,
                    quality_score = EXCLUDED.quality_score,
                    updated_at = CURRENT_TIMESTAMP
            """))
            conn.execute(text(f"DROP TABLE IF EXISTS {temp_table}"))
            conn.commit()
            
        logger.info(f"Saved M-Scores for {len(df)} stocks")
        
    except Exception as e:
        logger.error(f"Error saving M-Scores: {e}")
        with engine.connect() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {temp_table}"))
            conn.rollback()

def analyze_results(df):
    """Analyze and display M-Score results."""
    
    print("\n" + "=" * 80)
    print("BENEISH M-SCORE ANALYSIS")
    print("Earnings Manipulation Detection")
    print("=" * 80)
    
    # Risk distribution
    print("\n📊 MANIPULATION RISK DISTRIBUTION:")
    risk_dist = df['manipulation_risk'].value_counts()
    for risk in ['LOW', 'MODERATE', 'HIGH']:
        count = risk_dist.get(risk, 0)
        pct = count * 100 / len(df) if len(df) > 0 else 0
        print(f"  {risk:8s}: {count:4d} ({pct:5.1f}%)")
    
    # High risk companies
    print("\n⚠️ HIGH MANIPULATION RISK (M-Score > -1.78):")
    high_risk = df[df['manipulation_risk'] == 'HIGH'].nsmallest(10, 'm_score')
    if not high_risk.empty:
        for _, row in high_risk.iterrows():
            print(f"  {row['symbol']:6s} | M-Score: {row['m_score']:7.2f} | "
                  f"Rev Growth: {row['revenue_growth']*100:6.1f}%")
    else:
        print("  No high-risk companies detected")
    
    # Cleanest companies
    print("\n✅ CLEANEST COMPANIES (Lowest M-Score):")
    cleanest = df.nsmallest(10, 'm_score')
    for _, row in cleanest.iterrows():
        print(f"  {row['symbol']:6s} | M-Score: {row['m_score']:7.2f} | "
              f"Quality: {row['quality_score']:5.1f} | "
              f"Risk: {row['manipulation_risk']:8s}")
    
    # Key variables analysis
    print("\n📈 KEY MANIPULATION INDICATORS:")
    print(f"  High DSRI (>1.5): {len(df[df['dsri'] > 1.5])} companies")
    print(f"  High GMI (>1.2): {len(df[df['gmi'] > 1.2])} companies")
    print(f"  High SGI (>1.5): {len(df[df['sgi'] > 1.5])} companies")
    print(f"  High TATA (>0.1): {len(df[df['tata'] > 0.1])} companies")
    
    # Statistics
    print("\n📊 STATISTICS:")
    print(f"  Mean M-Score: {df['m_score'].mean():.2f}")
    print(f"  Median M-Score: {df['m_score'].median():.2f}")
    print(f"  Stocks below -2.22 threshold: {len(df[df['m_score'] < -2.22])} ({len(df[df['m_score'] < -2.22])*100/len(df):.1f}%)")

def main():
    """Main execution function."""
    start_time = time.time()
    log_script_start(logger, "calculate_beneish_mscore", "Calculating Beneish M-Scores for manipulation detection")
    
    print("\n" + "=" * 80)
    print("BENEISH M-SCORE CALCULATOR")
    print("Detecting Potential Earnings Manipulation")
    print("=" * 80)
    print("M-Score < -2.22 = Unlikely manipulator")
    print("M-Score > -2.22 = Possible manipulation")
    print("=" * 80)
    
    try:
        # Setup database
        load_dotenv()
        db_manager = DatabaseConnectionManager()
        engine = db_manager.get_engine()
        
        # Fetch financial data
        df = fetch_financial_data(engine)
        
        if df.empty:
            logger.warning("No financial data found for M-Score calculation")
            return
        
        # Calculate M-Scores
        logger.info(f"Calculating M-Scores for {len(df)} companies...")
        results_df = calculate_all_mscores(df)
        
        print(f"\n[INFO] Calculated M-Scores for {len(results_df)} companies")
        
        # Save to database
        save_mscores(engine, results_df)
        
        # Analyze results
        analyze_results(results_df)
        
        # Investment implications
        print("\n" + "=" * 80)
        print("💡 INVESTMENT IMPLICATIONS")
        print("=" * 80)
        print("\n✅ USE M-SCORE TO:")
        print("  1. Exclude stocks with M-Score > -1.78 (high manipulation risk)")
        print("  2. Favor stocks with M-Score < -2.22 (clean accounting)")
        print("  3. Double-check high revenue growth stocks (often manipulated)")
        print("  4. Verify quality of earnings (TATA variable)")
        print("  5. Monitor deteriorating gross margins (GMI > 1)")
        
        clean_count = len(results_df[results_df['m_score'] < -2.22])
        print(f"\n📊 PORTFOLIO IMPACT:")
        print(f"  Clean companies available: {clean_count}")
        print(f"  Exclusion rate: {(len(df) - clean_count)*100/len(df):.1f}%")
        print("  Combine with F-Score & Z-Score for maximum safety")
        
        # Log completion
        duration = time.time() - start_time
        log_script_end(logger, "calculate_beneish_mscore", success=True, duration=duration)
        print(f"\n[SUCCESS] M-Score calculation completed in {duration:.1f} seconds")
        
    except Exception as e:
        logger.error(f"Script failed: {e}")
        log_script_end(logger, "calculate_beneish_mscore", success=False, duration=time.time()-start_time)
        raise

if __name__ == "__main__":
    main()