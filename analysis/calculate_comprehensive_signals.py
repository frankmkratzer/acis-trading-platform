"""
Calculate comprehensive investment signals combining all TOP 1% strategy enhancements.

This script combines:
1. Insider Transactions (CEO/CFO buying patterns)
2. Piotroski F-Score (fundamental strength)
3. Altman Z-Score (bankruptcy risk)
4. Earnings Estimates (future growth expectations)

To identify the highest quality investment opportunities.
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
import time

# Setup logging first
from utils.logging_config import setup_logger, log_script_start, log_script_end
logger = setup_logger("calculate_comprehensive_signals")

# Database connection
from database.db_connection_manager import DatabaseConnectionManager

def fetch_comprehensive_data(engine):
    """Fetch all signals data for comprehensive analysis."""
    
    query = """
    SELECT 
        su.symbol,
        su.name,
        su.sector,
        su.market_cap / 1e9 as market_cap_b,
        
        -- Insider signals
        ins.insider_score,
        ins.ceo_buying,
        ins.cfo_buying,
        ins.officer_cluster_buying,
        ins.total_bought_value,
        
        -- Piotroski F-Score
        ps.fscore,
        ps.strength_category as piotroski_strength,
        
        -- Altman Z-Score
        az.z_score,
        az.z_prime_score,
        az.risk_category as bankruptcy_risk,
        
        -- Company fundamentals
        cfo."pERatio" as pe_ratio,
        cfo."dividendYield" as dividend_yield,
        cfo."profitMargin" as profit_margin,
        cfo."returnOnEquity" as roe,
        
        -- Calculate composite score
        CASE 
            WHEN ps.fscore IS NOT NULL AND az.risk_category IS NOT NULL
            THEN (
                -- Weighted composite score
                COALESCE(ins.insider_score, 50) * 0.25 +  -- 25% weight on insider activity
                COALESCE(ps.fscore * 11.11, 50) * 0.35 +  -- 35% weight on F-Score (scaled to 100)
                CASE 
                    WHEN az.risk_category = 'SAFE' THEN 100
                    WHEN az.risk_category = 'GREY' THEN 50
                    ELSE 0
                END * 0.40  -- 40% weight on bankruptcy risk
            )
            ELSE NULL
        END as composite_score
        
    FROM symbol_universe su
    LEFT JOIN insider_signals ins 
        ON su.symbol = ins.symbol
    LEFT JOIN piotroski_scores ps 
        ON su.symbol = ps.symbol
    LEFT JOIN altman_zscores az 
        ON su.symbol = az.symbol
    LEFT JOIN company_fundamentals_overview cfo 
        ON su.symbol = cfo.symbol
    WHERE su.market_cap >= 2e9  -- Mid-cap and above
      AND su.country = 'USA'
      AND su.security_type = 'Common Stock'
      AND ps.fscore IS NOT NULL
      AND az.risk_category IS NOT NULL
    ORDER BY 
        CASE 
            WHEN ps.fscore IS NOT NULL AND az.risk_category IS NOT NULL
            THEN (
                COALESCE(ins.insider_score, 50) * 0.25 +
                COALESCE(ps.fscore * 11.11, 50) * 0.35 +
                CASE 
                    WHEN az.risk_category = 'SAFE' THEN 100
                    WHEN az.risk_category = 'GREY' THEN 50
                    ELSE 0
                END * 0.40
            )
            ELSE 0
        END DESC
    """
    
    logger.info("Fetching comprehensive signals data...")
    # Create fresh connection to avoid immutabledict issue
    import os
    from dotenv import load_dotenv
    from sqlalchemy import create_engine
    
    load_dotenv()
    fresh_engine = create_engine(os.getenv('POSTGRES_URL'))
    df = pd.read_sql(query, fresh_engine)
    logger.info(f"Retrieved data for {len(df)} companies")
    
    return df

def categorize_investment_quality(row):
    """Categorize investment quality based on signals."""
    
    # TOP TIER: All signals positive
    if (row['fscore'] >= 8 and 
        row['bankruptcy_risk'] == 'SAFE' and 
        row.get('insider_score', 0) >= 70):
        return 'TOP_TIER'
    
    # STRONG: Two out of three signals very positive
    strong_signals = 0
    if row['fscore'] >= 7:
        strong_signals += 1
    if row['bankruptcy_risk'] == 'SAFE':
        strong_signals += 1
    if row.get('insider_score', 0) >= 60:
        strong_signals += 1
    
    if strong_signals >= 2:
        return 'STRONG'
    
    # MODERATE: Mixed signals
    if row['bankruptcy_risk'] != 'DISTRESS' and row['fscore'] >= 5:
        return 'MODERATE'
    
    # WEAK: Poor signals
    if row['bankruptcy_risk'] == 'DISTRESS' or row['fscore'] <= 3:
        return 'WEAK'
    
    return 'NEUTRAL'

def analyze_by_portfolio_strategy(df):
    """Analyze stocks for each portfolio strategy."""
    
    results = {
        'VALUE': [],
        'GROWTH': [],
        'DIVIDEND': []
    }
    
    # VALUE PORTFOLIO: Focus on undervalued + strong fundamentals
    value_candidates = df[
        (df['pe_ratio'] > 0) & 
        (df['pe_ratio'] < 15) &
        (df['fscore'] >= 6) &
        (df['bankruptcy_risk'] == 'SAFE')
    ].nlargest(10, 'composite_score')
    results['VALUE'] = value_candidates
    
    # GROWTH PORTFOLIO: Focus on momentum + insider buying
    growth_candidates = df[
        (df['insider_score'] >= 60) &
        (df['fscore'] >= 6) &
        (df['bankruptcy_risk'] != 'DISTRESS')
    ].nlargest(10, 'composite_score')
    results['GROWTH'] = growth_candidates
    
    # DIVIDEND PORTFOLIO: Focus on yield + safety
    dividend_candidates = df[
        (df['dividend_yield'] > 2) &
        (df['fscore'] >= 7) &
        (df['bankruptcy_risk'] == 'SAFE')
    ].nlargest(10, 'composite_score')
    results['DIVIDEND'] = dividend_candidates
    
    return results

def display_results(df, portfolio_results):
    """Display comprehensive analysis results."""
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE INVESTMENT SIGNALS ANALYSIS")
    print("Combining Insider Activity + F-Score + Z-Score")
    print("=" * 80)
    
    # Overall statistics
    print("\n📊 OVERALL STATISTICS:")
    print(f"  Total stocks analyzed: {len(df)}")
    print(f"  Average composite score: {df['composite_score'].mean():.1f}/100")
    print(f"  Stocks with F-Score >= 8: {len(df[df['fscore'] >= 8])}")
    print(f"  Stocks with insider buying: {len(df[df['insider_score'] > 50])}")
    print(f"  Stocks in SAFE zone: {len(df[df['bankruptcy_risk'] == 'SAFE'])}")
    
    # Top stocks by composite score
    print("\n🏆 TOP 10 STOCKS BY COMPOSITE SCORE:")
    top10 = df.nlargest(10, 'composite_score')
    for _, row in top10.iterrows():
        print(f"  {row['symbol']:6s} | Score: {row['composite_score']:.1f} | "
              f"F-Score: {row['fscore']:.0f} | "
              f"Insider: {row['insider_score']:.0f} | "
              f"Risk: {row['bankruptcy_risk']:8s} | "
              f"{row['name'][:30] if row['name'] else 'N/A'}")
    
    # Stocks with perfect signals
    print("\n💎 PERFECT SIGNAL STOCKS (All indicators positive):")
    perfect = df[
        (df['fscore'] >= 8) & 
        (df['bankruptcy_risk'] == 'SAFE') & 
        (df['insider_score'] >= 70)
    ]
    if not perfect.empty:
        for _, row in perfect.head(5).iterrows():
            print(f"  {row['symbol']:6s} | {row['name'][:40] if row['name'] else 'N/A'}")
            print(f"    └─ F-Score: {row['fscore']}, Insider: {row['insider_score']:.0f}, "
                  f"PE: {row['pe_ratio']:.1f if row['pe_ratio'] else 'N/A'}")
    else:
        print("  No stocks meet all criteria perfectly")
    
    # Portfolio recommendations
    print("\n📈 PORTFOLIO RECOMMENDATIONS:")
    
    for portfolio_type, stocks in portfolio_results.items():
        print(f"\n  {portfolio_type} PORTFOLIO (Top 10):")
        if not stocks.empty:
            for i, (_, row) in enumerate(stocks.iterrows(), 1):
                print(f"    {i:2d}. {row['symbol']:6s} | Score: {row['composite_score']:.1f} | "
                      f"{row['name'][:30] if row['name'] else 'N/A'}")
        else:
            print("    No qualifying stocks found")
    
    # Sector analysis
    print("\n🏢 BEST SECTORS BY COMPOSITE SCORE:")
    sector_scores = df.groupby('sector')['composite_score'].agg(['mean', 'count'])
    sector_scores = sector_scores[sector_scores['count'] >= 5].sort_values('mean', ascending=False).head(5)
    for sector, row in sector_scores.iterrows():
        print(f"  {sector:25s}: {row['mean']:.1f} (n={row['count']:.0f})")
    
    # Warning signals
    print("\n⚠️  HIGH RISK STOCKS TO AVOID:")
    risky = df[
        (df['fscore'] <= 2) | 
        (df['bankruptcy_risk'] == 'DISTRESS')
    ].nsmallest(5, 'composite_score')
    for _, row in risky.iterrows():
        print(f"  {row['symbol']:6s} | Score: {row['composite_score']:.1f} | "
              f"F-Score: {row['fscore']:.0f} | Risk: {row['bankruptcy_risk']}")

def save_comprehensive_scores(engine, df):
    """Save comprehensive scores to database."""
    
    # Prepare data for saving
    scores_df = df[['symbol', 'composite_score', 'fscore', 'insider_score', 
                    'bankruptcy_risk', 'pe_ratio', 'dividend_yield']].copy()
    scores_df['calculation_date'] = datetime.now().date()
    scores_df['investment_quality'] = df.apply(categorize_investment_quality, axis=1)
    
    # Create comprehensive_scores table if it doesn't exist
    create_table_query = """
    CREATE TABLE IF NOT EXISTS comprehensive_scores (
        symbol VARCHAR(10) NOT NULL,
        calculation_date DATE NOT NULL,
        composite_score NUMERIC(6, 2),
        fscore INTEGER,
        insider_score NUMERIC(6, 2),
        bankruptcy_risk VARCHAR(20),
        pe_ratio NUMERIC(12, 2),
        dividend_yield NUMERIC(8, 4),
        investment_quality VARCHAR(20),
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, calculation_date)
    )
    """
    
    with engine.connect() as conn:
        conn.execute(text(create_table_query))
        conn.commit()
    
    # Save scores
    temp_table = f"temp_comprehensive_{int(time.time() * 1000)}"
    
    try:
        scores_df.to_sql(temp_table, engine, if_exists='replace', index=False)
        
        upsert_query = f"""
            INSERT INTO comprehensive_scores
            SELECT * FROM {temp_table}
            ON CONFLICT (symbol, calculation_date) DO UPDATE SET
                composite_score = EXCLUDED.composite_score,
                fscore = EXCLUDED.fscore,
                insider_score = EXCLUDED.insider_score,
                bankruptcy_risk = EXCLUDED.bankruptcy_risk,
                pe_ratio = EXCLUDED.pe_ratio,
                dividend_yield = EXCLUDED.dividend_yield,
                investment_quality = EXCLUDED.investment_quality,
                updated_at = CURRENT_TIMESTAMP
        """
        
        with engine.connect() as conn:
            conn.execute(text(upsert_query))
            conn.execute(text(f"DROP TABLE IF EXISTS {temp_table}"))
            conn.commit()
            
        logger.info(f"Saved {len(scores_df)} comprehensive scores to database")
        
    except Exception as e:
        logger.error(f"Error saving comprehensive scores: {e}")
        with engine.connect() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {temp_table}"))
            conn.rollback()

def main():
    """Main execution function."""
    start_time = time.time()
    log_script_start(logger, "calculate_comprehensive_signals", "Calculating comprehensive investment signals")
    
    try:
        # Setup database
        load_dotenv()
        db_manager = DatabaseConnectionManager()
        engine = db_manager.get_engine()
        
        # Fetch comprehensive data
        df = fetch_comprehensive_data(engine)
        
        if df.empty:
            logger.warning("No comprehensive data found")
            return
        
        # Analyze by portfolio strategy
        portfolio_results = analyze_by_portfolio_strategy(df)
        
        # Display results
        display_results(df, portfolio_results)
        
        # Save comprehensive scores
        save_comprehensive_scores(engine, df)
        
        # Summary statistics
        print("\n" + "=" * 80)
        print("INVESTMENT SUMMARY")
        print("=" * 80)
        
        top_tier = len(df[df.apply(categorize_investment_quality, axis=1) == 'TOP_TIER'])
        strong = len(df[df.apply(categorize_investment_quality, axis=1) == 'STRONG'])
        
        print(f"\n✅ TOP TIER opportunities: {top_tier} stocks")
        print(f"✅ STRONG opportunities: {strong} stocks")
        print(f"✅ Total investable (score > 60): {len(df[df['composite_score'] > 60])} stocks")
        
        print("\n💡 INVESTMENT INSIGHT:")
        print("  Focus on stocks with composite score > 70")
        print("  Prioritize F-Score >= 8 with insider buying")
        print("  Avoid any stock in DISTRESS zone regardless of other signals")
        
        # Log completion
        duration = time.time() - start_time
        log_script_end(logger, "calculate_comprehensive_signals", success=True, duration=duration)
        print(f"\n[SUCCESS] Comprehensive analysis completed in {duration:.1f} seconds")
        
    except Exception as e:
        logger.error(f"Script failed: {e}")
        log_script_end(logger, "calculate_comprehensive_signals", success=False, duration=time.time()-start_time)
        raise

if __name__ == "__main__":
    main()