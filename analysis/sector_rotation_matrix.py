"""
Sector Rotation Matrix for Dynamic Allocation.

This module implements sector rotation strategies based on:
- Economic cycle positioning
- Sector momentum and relative strength
- Risk-on/Risk-off regime detection
- Cross-sector correlation analysis
- Sector-specific quality metrics
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
import seaborn as sns
import matplotlib.pyplot as plt

# Setup logging first
from utils.logging_config import setup_logger, log_script_start, log_script_end
logger = setup_logger("sector_rotation_matrix")

# Database connection
from database.db_connection_manager import DatabaseConnectionManager

# Economic cycle phases and preferred sectors
CYCLE_SECTORS = {
    'EARLY_RECOVERY': ['Technology', 'Consumer Discretionary', 'Financials'],
    'MID_CYCLE': ['Technology', 'Industrials', 'Materials'],
    'LATE_CYCLE': ['Energy', 'Materials', 'Consumer Staples'],
    'RECESSION': ['Consumer Staples', 'Healthcare', 'Utilities']
}

# Risk regime preferences
RISK_REGIME_SECTORS = {
    'RISK_ON': ['Technology', 'Consumer Discretionary', 'Financials', 'Industrials'],
    'RISK_OFF': ['Consumer Staples', 'Healthcare', 'Utilities', 'Real Estate']
}

def fetch_sector_data(engine, lookback_days=252):
    """Fetch sector performance and metrics."""
    
    query = f"""
    WITH sector_prices AS (
        SELECT 
            su.sector,
            sp.trade_date,
            AVG(sp.close) as avg_close,
            SUM(sp.volume * sp.close) / SUM(sp.volume) as vwap,
            COUNT(DISTINCT sp.symbol) as num_stocks,
            SUM(su.market_cap) / 1e9 as sector_mcap_b
        FROM stock_prices sp
        JOIN symbol_universe su ON sp.symbol = su.symbol
        WHERE sp.trade_date >= CURRENT_DATE - INTERVAL '{lookback_days} days'
          AND su.market_cap >= 2e9
          AND su.country = 'USA'
          AND su.security_type = 'Common Stock'
          AND su.sector IS NOT NULL
        GROUP BY su.sector, sp.trade_date
    ),
    sector_returns AS (
        SELECT 
            sector,
            trade_date,
            vwap,
            (vwap / LAG(vwap) OVER (PARTITION BY sector ORDER BY trade_date) - 1) as daily_return
        FROM sector_prices
    ),
    sector_quality AS (
        SELECT 
            su.sector,
            AVG(ps.fscore) as avg_fscore,
            AVG(az.zscore) as avg_zscore,
            AVG(ms.composite_score) as avg_quality_score,
            AVG(rm.sharpe_ratio) as avg_sharpe,
            AVG(ih.institutional_ownership_pct) as avg_inst_ownership
        FROM symbol_universe su
        LEFT JOIN piotroski_scores ps ON su.symbol = ps.symbol
        LEFT JOIN altman_zscores az ON su.symbol = az.symbol
        LEFT JOIN master_scores ms ON su.symbol = ms.symbol
        LEFT JOIN risk_metrics rm ON su.symbol = rm.symbol
        LEFT JOIN institutional_holdings ih ON su.symbol = ih.symbol
        WHERE su.market_cap >= 2e9
          AND su.country = 'USA'
          AND su.sector IS NOT NULL
        GROUP BY su.sector
    )
    SELECT 
        sr.*,
        sq.avg_fscore,
        sq.avg_zscore,
        sq.avg_quality_score,
        sq.avg_sharpe,
        sq.avg_inst_ownership
    FROM sector_returns sr
    LEFT JOIN sector_quality sq ON sr.sector = sq.sector
    ORDER BY sr.trade_date, sr.sector
    """
    
    logger.info(f"Fetching {lookback_days} days of sector data...")
    df = pd.read_sql(query, engine)
    logger.info(f"Retrieved data for {df['sector'].nunique()} sectors")
    
    return df

def calculate_sector_momentum(df):
    """Calculate momentum scores for each sector."""
    
    momentum_scores = []
    sectors = df['sector'].unique()
    latest_date = df['trade_date'].max()
    
    for sector in sectors:
        sector_data = df[df['sector'] == sector].copy()
        sector_data = sector_data.sort_values('trade_date')
        
        if len(sector_data) < 20:
            continue
        
        # Calculate various momentum periods
        returns = sector_data['daily_return'].dropna()
        
        # Short-term momentum (1 month)
        momentum_1m = (1 + returns.tail(20)).prod() - 1 if len(returns) >= 20 else 0
        
        # Medium-term momentum (3 months)
        momentum_3m = (1 + returns.tail(60)).prod() - 1 if len(returns) >= 60 else 0
        
        # Long-term momentum (6 months)
        momentum_6m = (1 + returns.tail(120)).prod() - 1 if len(returns) >= 120 else 0
        
        # Relative strength vs market
        market_return = df.groupby('trade_date')['daily_return'].mean()
        sector_excess = returns - market_return.reindex(returns.index)
        relative_strength = sector_excess.tail(60).mean() * 252 if len(sector_excess) >= 60 else 0
        
        # Momentum consistency (% of positive days)
        consistency = (returns.tail(60) > 0).mean() if len(returns) >= 60 else 0.5
        
        # Volatility
        volatility = returns.tail(60).std() * np.sqrt(252) if len(returns) >= 60 else 0
        
        # Combined momentum score
        momentum_score = (
            momentum_1m * 0.2 +
            momentum_3m * 0.3 +
            momentum_6m * 0.2 +
            relative_strength * 0.2 +
            consistency * 0.1
        ) * 100
        
        momentum_scores.append({
            'sector': sector,
            'calculation_date': latest_date,
            'momentum_1m': momentum_1m,
            'momentum_3m': momentum_3m,
            'momentum_6m': momentum_6m,
            'relative_strength': relative_strength,
            'consistency': consistency,
            'volatility': volatility,
            'momentum_score': momentum_score,
            'avg_quality_score': sector_data['avg_quality_score'].iloc[-1] if 'avg_quality_score' in sector_data else 50
        })
    
    return pd.DataFrame(momentum_scores)

def detect_market_regime(df):
    """Detect current market regime (risk-on vs risk-off)."""
    
    # Calculate market-wide metrics
    latest_data = df.sort_values('trade_date').groupby('sector').last()
    
    # Risk-on indicators
    tech_momentum = latest_data.loc['Technology', 'daily_return'] if 'Technology' in latest_data.index else 0
    discretionary_momentum = latest_data.loc['Consumer Discretionary', 'daily_return'] if 'Consumer Discretionary' in latest_data.index else 0
    
    # Risk-off indicators
    staples_momentum = latest_data.loc['Consumer Staples', 'daily_return'] if 'Consumer Staples' in latest_data.index else 0
    utilities_momentum = latest_data.loc['Utilities', 'daily_return'] if 'Utilities' in latest_data.index else 0
    
    # Calculate risk appetite
    risk_on_score = (tech_momentum + discretionary_momentum) / 2
    risk_off_score = (staples_momentum + utilities_momentum) / 2
    
    # Volatility check
    market_vol = df.groupby('trade_date')['daily_return'].std().tail(20).mean()
    high_vol_threshold = df.groupby('trade_date')['daily_return'].std().quantile(0.75)
    
    # Determine regime
    if risk_on_score > risk_off_score and market_vol < high_vol_threshold:
        regime = 'RISK_ON'
        confidence = min(100, (risk_on_score - risk_off_score) * 1000)
    elif risk_off_score > risk_on_score or market_vol > high_vol_threshold:
        regime = 'RISK_OFF'
        confidence = min(100, (risk_off_score - risk_on_score) * 1000 + market_vol * 100)
    else:
        regime = 'NEUTRAL'
        confidence = 50
    
    return regime, confidence

def detect_economic_cycle(df):
    """Detect economic cycle phase based on sector performance."""
    
    momentum_df = calculate_sector_momentum(df)
    
    # Leading indicators
    tech_momentum = momentum_df[momentum_df['sector'] == 'Technology']['momentum_3m'].values[0] if 'Technology' in momentum_df['sector'].values else 0
    financials_momentum = momentum_df[momentum_df['sector'] == 'Financials']['momentum_3m'].values[0] if 'Financials' in momentum_df['sector'].values else 0
    
    # Defensive indicators  
    staples_momentum = momentum_df[momentum_df['sector'] == 'Consumer Staples']['momentum_3m'].values[0] if 'Consumer Staples' in momentum_df['sector'].values else 0
    utilities_momentum = momentum_df[momentum_df['sector'] == 'Utilities']['momentum_3m'].values[0] if 'Utilities' in momentum_df['sector'].values else 0
    
    # Cyclical indicators
    materials_momentum = momentum_df[momentum_df['sector'] == 'Materials']['momentum_3m'].values[0] if 'Materials' in momentum_df['sector'].values else 0
    energy_momentum = momentum_df[momentum_df['sector'] == 'Energy']['momentum_3m'].values[0] if 'Energy' in momentum_df['sector'].values else 0
    
    # Determine cycle phase
    if tech_momentum > 0 and financials_momentum > 0 and staples_momentum < 0:
        phase = 'EARLY_RECOVERY'
    elif materials_momentum > 0 and energy_momentum > 0:
        phase = 'LATE_CYCLE'
    elif staples_momentum > 0 and utilities_momentum > 0 and tech_momentum < 0:
        phase = 'RECESSION'
    else:
        phase = 'MID_CYCLE'
    
    return phase

def create_rotation_matrix(momentum_df, regime, cycle_phase):
    """Create sector rotation matrix with allocation recommendations."""
    
    rotation_matrix = []
    
    for _, row in momentum_df.iterrows():
        sector = row['sector']
        
        # Base score from momentum
        score = row['momentum_score']
        
        # Adjust for market regime
        if regime == 'RISK_ON' and sector in RISK_REGIME_SECTORS['RISK_ON']:
            score *= 1.2
        elif regime == 'RISK_OFF' and sector in RISK_REGIME_SECTORS['RISK_OFF']:
            score *= 1.2
        else:
            score *= 0.9
        
        # Adjust for economic cycle
        if sector in CYCLE_SECTORS.get(cycle_phase, []):
            score *= 1.15
        
        # Adjust for quality
        if row['avg_quality_score'] > 70:
            score *= 1.1
        elif row['avg_quality_score'] < 30:
            score *= 0.8
        
        # Normalize score
        score = max(0, min(100, score))
        
        # Determine allocation
        if score >= 80:
            allocation = 'OVERWEIGHT'
            weight_pct = 15
        elif score >= 60:
            allocation = 'NEUTRAL'
            weight_pct = 10
        elif score >= 40:
            allocation = 'UNDERWEIGHT'
            weight_pct = 5
        else:
            allocation = 'AVOID'
            weight_pct = 0
        
        rotation_matrix.append({
            'sector': sector,
            'rotation_score': score,
            'momentum_1m': row['momentum_1m'] * 100,
            'momentum_3m': row['momentum_3m'] * 100,
            'relative_strength': row['relative_strength'],
            'quality_score': row['avg_quality_score'],
            'allocation': allocation,
            'target_weight_pct': weight_pct,
            'regime': regime,
            'cycle_phase': cycle_phase
        })
    
    matrix_df = pd.DataFrame(rotation_matrix)
    
    # Normalize weights to sum to 100%
    total_weight = matrix_df['target_weight_pct'].sum()
    if total_weight > 0:
        matrix_df['normalized_weight_pct'] = (matrix_df['target_weight_pct'] / total_weight) * 100
    else:
        matrix_df['normalized_weight_pct'] = 0
    
    return matrix_df.sort_values('rotation_score', ascending=False)

def save_rotation_matrix(engine, matrix_df):
    """Save sector rotation matrix to database."""
    
    create_table_query = """
    CREATE TABLE IF NOT EXISTS sector_rotation_matrix (
        sector VARCHAR(50) NOT NULL,
        calculation_date DATE NOT NULL,
        
        -- Scores and metrics
        rotation_score NUMERIC(6, 2),
        momentum_1m NUMERIC(10, 4),
        momentum_3m NUMERIC(10, 4),
        momentum_6m NUMERIC(10, 4),
        relative_strength NUMERIC(10, 4),
        quality_score NUMERIC(6, 2),
        volatility NUMERIC(10, 4),
        
        -- Allocation
        allocation VARCHAR(20) CHECK (allocation IN 
            ('OVERWEIGHT', 'NEUTRAL', 'UNDERWEIGHT', 'AVOID')),
        target_weight_pct NUMERIC(5, 2),
        normalized_weight_pct NUMERIC(5, 2),
        
        -- Market context
        market_regime VARCHAR(20),
        economic_cycle VARCHAR(20),
        regime_confidence NUMERIC(6, 2),
        
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (sector, calculation_date)
    );
    
    CREATE TABLE IF NOT EXISTS sector_correlations (
        calculation_date DATE NOT NULL,
        sector1 VARCHAR(50) NOT NULL,
        sector2 VARCHAR(50) NOT NULL,
        correlation NUMERIC(6, 4),
        rolling_30d_corr NUMERIC(6, 4),
        rolling_90d_corr NUMERIC(6, 4),
        PRIMARY KEY (calculation_date, sector1, sector2)
    );
    
    CREATE INDEX IF NOT EXISTS idx_rotation_score 
        ON sector_rotation_matrix(rotation_score DESC);
    CREATE INDEX IF NOT EXISTS idx_rotation_allocation 
        ON sector_rotation_matrix(allocation);
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
    
    # Save rotation matrix
    if not matrix_df.empty:
        matrix_df['calculation_date'] = datetime.now().date()
        matrix_df['market_regime'] = matrix_df['regime'].iloc[0] if 'regime' in matrix_df else 'NEUTRAL'
        matrix_df['economic_cycle'] = matrix_df['cycle_phase'].iloc[0] if 'cycle_phase' in matrix_df else 'MID_CYCLE'
        
        temp_table = f"temp_rotation_{int(time.time() * 1000)}"
        
        save_cols = ['sector', 'calculation_date', 'rotation_score', 
                    'momentum_1m', 'momentum_3m', 'relative_strength',
                    'quality_score', 'allocation', 'target_weight_pct',
                    'normalized_weight_pct', 'market_regime', 'economic_cycle']
        
        matrix_df[save_cols].to_sql(temp_table, engine, if_exists='replace', index=False)
        
        with engine.connect() as conn:
            conn.execute(text(f"""
                INSERT INTO sector_rotation_matrix
                SELECT * FROM {temp_table}
                ON CONFLICT (sector, calculation_date) DO UPDATE SET
                    rotation_score = EXCLUDED.rotation_score,
                    momentum_1m = EXCLUDED.momentum_1m,
                    momentum_3m = EXCLUDED.momentum_3m,
                    relative_strength = EXCLUDED.relative_strength,
                    quality_score = EXCLUDED.quality_score,
                    allocation = EXCLUDED.allocation,
                    target_weight_pct = EXCLUDED.target_weight_pct,
                    normalized_weight_pct = EXCLUDED.normalized_weight_pct,
                    market_regime = EXCLUDED.market_regime,
                    economic_cycle = EXCLUDED.economic_cycle,
                    updated_at = CURRENT_TIMESTAMP
            """))
            conn.execute(text(f"DROP TABLE IF EXISTS {temp_table}"))
            conn.commit()
        
        logger.info(f"Saved rotation matrix for {len(matrix_df)} sectors")

def calculate_sector_correlations(df):
    """Calculate correlation matrix between sectors."""
    
    # Pivot data to get sectors as columns
    pivot_df = df.pivot_table(
        index='trade_date',
        columns='sector',
        values='daily_return',
        aggfunc='mean'
    )
    
    # Calculate correlation matrix
    corr_matrix = pivot_df.corr()
    
    # Calculate rolling correlations
    corr_30d = pivot_df.tail(30).corr()
    corr_90d = pivot_df.tail(90).corr()
    
    return corr_matrix, corr_30d, corr_90d

def analyze_rotation_results(matrix_df):
    """Analyze and display sector rotation results."""
    
    print("\n" + "=" * 80)
    print("SECTOR ROTATION MATRIX")
    print("=" * 80)
    
    print(f"\nMarket Regime: {matrix_df['regime'].iloc[0] if 'regime' in matrix_df else 'NEUTRAL'}")
    print(f"Economic Cycle: {matrix_df['cycle_phase'].iloc[0] if 'cycle_phase' in matrix_df else 'MID_CYCLE'}")
    
    print("\nSECTOR ALLOCATIONS:")
    for _, row in matrix_df.iterrows():
        allocation_emoji = {
            'OVERWEIGHT': '🟢', 'NEUTRAL': '🟡',
            'UNDERWEIGHT': '🟠', 'AVOID': '🔴'
        }.get(row['allocation'], '⚪')
        
        print(f"  {row['sector']:20s} | {allocation_emoji} {row['allocation']:11s} | "
              f"Weight: {row['normalized_weight_pct']:5.1f}% | "
              f"Score: {row['rotation_score']:5.1f} | "
              f"3M Mom: {row['momentum_3m']:+6.1f}%")
    
    print("\nTOP OPPORTUNITIES:")
    overweight = matrix_df[matrix_df['allocation'] == 'OVERWEIGHT']
    if not overweight.empty:
        for _, row in overweight.iterrows():
            print(f"  ✅ {row['sector']}: Strong momentum + cycle alignment")
    
    print("\nSECTORS TO AVOID:")
    avoid = matrix_df[matrix_df['allocation'] == 'AVOID']
    if not avoid.empty:
        for _, row in avoid.iterrows():
            print(f"  ❌ {row['sector']}: Weak momentum or cycle headwinds")

def main():
    """Main execution function."""
    start_time = time.time()
    log_script_start(logger, "sector_rotation_matrix", "Calculating sector rotation allocations")
    
    print("\n" + "=" * 80)
    print("SECTOR ROTATION STRATEGY")
    print("Dynamic Sector Allocation Based on Momentum and Cycles")
    print("=" * 80)
    
    try:
        # Setup database
        load_dotenv()
        db_manager = DatabaseConnectionManager()
        engine = db_manager.get_engine()
        
        # Fetch sector data
        df = fetch_sector_data(engine, lookback_days=252)
        
        if df.empty:
            logger.warning("No sector data found")
            return
        
        # Calculate sector momentum
        print("\n[INFO] Calculating sector momentum...")
        momentum_df = calculate_sector_momentum(df)
        
        # Detect market regime
        print("[INFO] Detecting market regime...")
        regime, confidence = detect_market_regime(df)
        print(f"  Market Regime: {regime} (Confidence: {confidence:.0f}%)")
        
        # Detect economic cycle
        print("[INFO] Detecting economic cycle...")
        cycle_phase = detect_economic_cycle(df)
        print(f"  Economic Cycle: {cycle_phase}")
        
        # Create rotation matrix
        print("\n[INFO] Creating rotation matrix...")
        matrix_df = create_rotation_matrix(momentum_df, regime, cycle_phase)
        
        # Calculate correlations
        print("[INFO] Calculating sector correlations...")
        corr_matrix, corr_30d, corr_90d = calculate_sector_correlations(df)
        
        # Save results
        save_rotation_matrix(engine, matrix_df)
        
        # Analyze results
        analyze_rotation_results(matrix_df)
        
        # Investment insights
        print("\n" + "=" * 80)
        print("SECTOR ROTATION INSIGHTS")
        print("=" * 80)
        print("\nKey Principles:")
        print("  1. Overweight sectors with strong momentum in favorable cycles")
        print("  2. Reduce exposure to sectors facing cycle headwinds")
        print("  3. Risk-on favors growth sectors (Tech, Discretionary)")
        print("  4. Risk-off favors defensive sectors (Staples, Utilities)")
        print("  5. Rebalance monthly to capture rotation opportunities")
        
        print("\nCycle Playbook:")
        print("  - Early Recovery: Tech, Discretionary, Financials")
        print("  - Mid Cycle: Tech, Industrials, Materials")
        print("  - Late Cycle: Energy, Materials, Staples")
        print("  - Recession: Staples, Healthcare, Utilities")
        
        # Log completion
        duration = time.time() - start_time
        log_script_end(logger, "sector_rotation_matrix", success=True, duration=duration)
        print(f"\n[SUCCESS] Sector rotation analysis completed in {duration:.1f} seconds")
        
    except Exception as e:
        logger.error(f"Script failed: {e}")
        log_script_end(logger, "sector_rotation_matrix", success=False, duration=time.time()-start_time)
        raise

if __name__ == "__main__":
    main()