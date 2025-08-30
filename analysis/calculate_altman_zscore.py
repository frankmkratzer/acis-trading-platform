"""
Calculate Altman Z-Score for bankruptcy risk assessment.

The Altman Z-Score is a formula for predicting bankruptcy within 2 years:
- Z > 2.99: Safe Zone (low bankruptcy risk)
- 1.81 < Z < 2.99: Grey Zone (moderate risk)
- Z < 1.81: Distress Zone (high bankruptcy risk)

For non-manufacturing companies, we use the modified Z'-Score:
- Z' > 2.6: Safe Zone
- 1.1 < Z' < 2.6: Grey Zone
- Z' < 1.1: Distress Zone
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
logger = setup_logger("calculate_altman_zscore")

# Database connection
from database.db_connection_manager import DatabaseConnectionManager

def fetch_financial_data(engine):
    """Fetch latest financial data for Z-Score calculation."""
    
    query = """
    WITH latest_data AS (
        SELECT 
            f.symbol,
            f.fiscal_date_ending,
            f.period_type,
            
            -- Balance sheet items (using available columns)
            f.total_assets,
            f.total_liabilities,
            f.total_assets - f.total_liabilities as current_assets,  -- Approximation
            f.total_liabilities * 0.5 as current_liabilities,  -- Approximation
            f.total_shareholder_equity * 0.3 as retained_earnings,  -- Approximation
            f.total_shareholder_equity,
            
            -- Income statement items
            f.total_revenue_ttm as revenue,
            f.operating_income_ttm as ebit,
            f.net_income_ttm as net_income,
            
            -- Get market cap and shares
            su.market_cap,
            su.shares_outstanding,
            su.sector,
            
            -- Rank to get latest data
            ROW_NUMBER() OVER (
                PARTITION BY f.symbol 
                ORDER BY f.fiscal_date_ending DESC
            ) as rn
            
        FROM fundamentals f
        INNER JOIN symbol_universe su 
            ON f.symbol = su.symbol
        WHERE f.period_type = 'quarterly'  -- Use quarterly data (annualized)
          AND f.total_assets > 0  -- Must have valid assets
          AND su.market_cap >= 2e9  -- Mid-cap and above
          AND su.country = 'USA'
          AND su.security_type = 'Common Stock'
    )
    SELECT *
    FROM latest_data
    WHERE rn = 1
    """
    
    logger.info("Fetching financial data for Z-Score calculation...")
    df = pd.read_sql(query, engine)
    logger.info(f"Retrieved data for {len(df)} companies")
    
    return df

def calculate_altman_zscore(row):
    """
    Calculate Altman Z-Score and Z'-Score.
    
    Original Z-Score (manufacturing):
    Z = 1.2(X1) + 1.4(X2) + 3.3(X3) + 0.6(X4) + 1.0(X5)
    
    Modified Z'-Score (non-manufacturing):
    Z' = 6.56(X1) + 3.26(X2) + 6.72(X3) + 1.05(X4)
    
    Where:
    X1 = Working Capital / Total Assets
    X2 = Retained Earnings / Total Assets
    X3 = EBIT / Total Assets
    X4 = Market Value of Equity / Book Value of Total Liabilities
    X5 = Sales / Total Assets (manufacturing only)
    """
    
    try:
        # Handle None values
        if pd.isna(row['total_assets']) or row['total_assets'] <= 0:
            return None, None, None, None, None, None, None, None
            
        # Calculate components
        working_capital = (row['current_assets'] or 0) - (row['current_liabilities'] or 0)
        
        # X1: Working Capital / Total Assets
        x1 = working_capital / row['total_assets']
        
        # X2: Retained Earnings / Total Assets
        x2 = (row['retained_earnings'] or 0) / row['total_assets']
        
        # X3: EBIT / Total Assets
        x3 = (row['ebit'] or 0) / row['total_assets']
        
        # X4: Market Value of Equity / Book Value of Total Liabilities
        if row['total_liabilities'] and row['total_liabilities'] > 0:
            x4 = (row['market_cap'] or 0) / row['total_liabilities']
        else:
            x4 = 10.0  # Cap at 10 if no liabilities
            
        # X5: Sales / Total Assets (for manufacturing Z-Score)
        x5 = (row['revenue'] or 0) / row['total_assets']
        
        # Calculate Z-Score (manufacturing formula)
        z_score = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5
        
        # Calculate Z'-Score (non-manufacturing formula)
        z_prime_score = 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4
        
        # Determine if company is manufacturing
        is_manufacturing = row.get('sector', '').upper() in ['MANUFACTURING', 'INDUSTRIALS']
        
        # Use appropriate score
        final_score = z_score if is_manufacturing else z_prime_score
        
        # Categorize risk
        if is_manufacturing:
            if z_score > 2.99:
                risk_category = 'SAFE'
            elif z_score > 1.81:
                risk_category = 'GREY'
            else:
                risk_category = 'DISTRESS'
        else:
            if z_prime_score > 2.6:
                risk_category = 'SAFE'
            elif z_prime_score > 1.1:
                risk_category = 'GREY'
            else:
                risk_category = 'DISTRESS'
                
        return z_score, z_prime_score, x1, x2, x3, x4, x5, risk_category
        
    except Exception as e:
        logger.error(f"Error calculating Z-Score for {row.get('symbol', 'Unknown')}: {e}")
        return None, None, None, None, None, None, None, None

def calculate_all_zscores(df):
    """Calculate Z-Scores for all companies."""
    
    results = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Calculating Z-Scores"):
        z_score, z_prime_score, x1, x2, x3, x4, x5, risk_category = calculate_altman_zscore(row)
        
        if z_score is not None:
            results.append({
                'symbol': row['symbol'],
                'calculation_date': datetime.now().date(),
                'fiscal_date': row['fiscal_date_ending'],
                'z_score': round(z_score, 2) if z_score else None,
                'z_prime_score': round(z_prime_score, 2) if z_prime_score else None,
                
                # Components
                'working_capital_ratio': round(x1, 4) if x1 else None,
                'retained_earnings_ratio': round(x2, 4) if x2 else None,
                'ebit_ratio': round(x3, 4) if x3 else None,
                'market_to_book_liability': round(min(x4, 10), 4) if x4 else None,  # Cap at 10
                'sales_to_assets': round(x5, 4) if x5 else None,
                
                # Risk assessment
                'risk_category': risk_category,
                'is_manufacturing': row.get('sector', '').upper() in ['MANUFACTURING', 'INDUSTRIALS'],
                
                # Raw values for reference
                'total_assets': row['total_assets'],
                'total_liabilities': row['total_liabilities'],
                'market_cap': row['market_cap'],
                
                'updated_at': datetime.now()
            })
    
    return pd.DataFrame(results)

def upsert_zscores(engine, df):
    """Upsert Z-Scores to database."""
    
    if df.empty:
        logger.warning("No Z-Scores to upsert")
        return
    
    # Create temporary table
    temp_table = f"temp_zscore_{int(time.time() * 1000)}"
    
    try:
        df.to_sql(temp_table, engine, if_exists='replace', index=False)
        
        # Upsert from temp table
        upsert_query = f"""
            INSERT INTO altman_zscores (
                symbol, calculation_date, fiscal_date,
                z_score, z_prime_score,
                working_capital_ratio, retained_earnings_ratio, ebit_ratio,
                market_to_book_liability, sales_to_assets,
                risk_category, is_manufacturing,
                total_assets, total_liabilities, market_cap,
                updated_at
            )
            SELECT 
                symbol, calculation_date::date, fiscal_date::date,
                z_score::numeric, z_prime_score::numeric,
                working_capital_ratio::numeric, retained_earnings_ratio::numeric, ebit_ratio::numeric,
                market_to_book_liability::numeric, sales_to_assets::numeric,
                risk_category, is_manufacturing::boolean,
                total_assets::numeric, total_liabilities::numeric, market_cap::numeric,
                updated_at
            FROM {temp_table}
            ON CONFLICT (symbol, calculation_date) DO UPDATE SET
                fiscal_date = EXCLUDED.fiscal_date,
                z_score = EXCLUDED.z_score,
                z_prime_score = EXCLUDED.z_prime_score,
                working_capital_ratio = EXCLUDED.working_capital_ratio,
                retained_earnings_ratio = EXCLUDED.retained_earnings_ratio,
                ebit_ratio = EXCLUDED.ebit_ratio,
                market_to_book_liability = EXCLUDED.market_to_book_liability,
                sales_to_assets = EXCLUDED.sales_to_assets,
                risk_category = EXCLUDED.risk_category,
                is_manufacturing = EXCLUDED.is_manufacturing,
                total_assets = EXCLUDED.total_assets,
                total_liabilities = EXCLUDED.total_liabilities,
                market_cap = EXCLUDED.market_cap,
                updated_at = EXCLUDED.updated_at
        """
        
        with engine.connect() as conn:
            conn.execute(text(upsert_query))
            conn.execute(text(f"DROP TABLE IF EXISTS {temp_table}"))
            conn.commit()
            
        logger.info(f"Upserted {len(df)} Z-Scores to database")
        
    except Exception as e:
        logger.error(f"Error upserting Z-Scores: {e}")
        with engine.connect() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {temp_table}"))
            conn.rollback()
        raise

def analyze_results(df):
    """Analyze and display Z-Score results."""
    
    print("\n" + "=" * 60)
    print("ALTMAN Z-SCORE ANALYSIS")
    print("=" * 60)
    
    # Overall distribution
    print("\nRisk Distribution:")
    risk_dist = df['risk_category'].value_counts()
    for category in ['SAFE', 'GREY', 'DISTRESS']:
        count = risk_dist.get(category, 0)
        pct = count * 100 / len(df) if len(df) > 0 else 0
        print(f"  {category:8s}: {count:4d} ({pct:5.1f}%)")
    
    # Sector analysis
    print("\nBy Sector (Distressed Companies):")
    distressed = df[df['risk_category'] == 'DISTRESS']
    if not distressed.empty:
        sector_dist = distressed.groupby('sector').size().sort_values(ascending=False).head(5)
        for sector, count in sector_dist.items():
            print(f"  {sector}: {count}")
    else:
        print("  No distressed companies found")
    
    # Top safe companies
    print("\nTop Safe Companies (Highest Z-Score):")
    safe = df[df['risk_category'] == 'SAFE'].nlargest(10, 'z_score')
    if not safe.empty:
        for _, row in safe.iterrows():
            score = row['z_score'] if row['is_manufacturing'] else row['z_prime_score']
            print(f"  {row['symbol']:6s}: {score:6.2f} ({row.get('name', 'N/A')[:30]})")
    
    # Most distressed companies
    print("\nMost Distressed Companies (Lowest Z-Score):")
    distressed = df[df['risk_category'] == 'DISTRESS'].nsmallest(10, 'z_score')
    if not distressed.empty:
        for _, row in distressed.iterrows():
            score = row['z_score'] if row['is_manufacturing'] else row['z_prime_score']
            print(f"  {row['symbol']:6s}: {score:6.2f} ({row.get('name', 'N/A')[:30]})")
    
    # Statistics
    print("\nZ-Score Statistics:")
    print(f"  Mean Z-Score: {df['z_score'].mean():.2f}")
    print(f"  Median Z-Score: {df['z_score'].median():.2f}")
    print(f"  Std Dev: {df['z_score'].std():.2f}")
    
    # Investment implications
    print("\nInvestment Implications:")
    safe_count = len(df[df['risk_category'] == 'SAFE'])
    total = len(df)
    print(f"  {safe_count}/{total} ({safe_count*100/total:.1f}%) companies in SAFE zone")
    print("  These are candidates for long-term investment")
    
    distress_count = len(df[df['risk_category'] == 'DISTRESS'])
    print(f"  {distress_count}/{total} ({distress_count*100/total:.1f}%) companies in DISTRESS zone")
    print("  These should be avoided or closely monitored")

def main():
    """Main execution function."""
    start_time = time.time()
    log_script_start(logger, "calculate_altman_zscore", "Calculating Altman Z-Scores for bankruptcy risk")
    
    print("\n" + "=" * 60)
    print("ALTMAN Z-SCORE CALCULATOR")
    print("=" * 60)
    print("Bankruptcy Risk Assessment")
    print("Z > 2.99 = Safe | 1.81-2.99 = Grey | < 1.81 = Distress")
    print("=" * 60)
    
    try:
        # Setup database
        load_dotenv()
        db_manager = DatabaseConnectionManager()
        engine = db_manager.get_engine()
        
        # Fetch financial data
        df = fetch_financial_data(engine)
        
        if df.empty:
            logger.warning("No data found for Z-Score calculation")
            return
        
        # Calculate Z-Scores
        logger.info(f"Calculating Z-Scores for {len(df)} companies...")
        results_df = calculate_all_zscores(df)
        
        print(f"\n[INFO] Calculated Z-Scores for {len(results_df)} companies")
        
        # Add company info for analysis
        company_info = df[['symbol', 'sector']].drop_duplicates()
        results_df = results_df.merge(company_info, on='symbol', how='left')
        
        # Upsert to database
        upsert_zscores(engine, results_df)
        print(f"[INFO] Saved {len(results_df)} Z-Scores to database")
        
        # Analyze results
        analyze_results(results_df)
        
        # Log completion
        duration = time.time() - start_time
        log_script_end(logger, "calculate_altman_zscore", success=True, duration=duration)
        
        print(f"\n[SUCCESS] Z-Score calculation completed in {duration:.1f} seconds")
        
    except Exception as e:
        logger.error(f"Script failed: {e}")
        log_script_end(logger, "calculate_altman_zscore", success=False, duration=time.time()-start_time)
        raise

if __name__ == "__main__":
    main()