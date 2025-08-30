"""
Calculate Kelly Criterion for optimal position sizing.

The Kelly Criterion determines the optimal fraction of capital to allocate
to each position based on expected returns and win probability.

Kelly Formula: f* = (p*b - q) / b
Where:
- f* = fraction of capital to wager
- p = probability of winning
- q = probability of losing (1-p)
- b = odds received on the wager (win/loss ratio)

Modified Kelly for stocks: f* = μ/σ²
Where:
- μ = expected excess return
- σ² = variance of returns
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
logger = setup_logger("calculate_kelly_criterion")

# Database connection
from database.db_connection_manager import DatabaseConnectionManager

def fetch_return_data(engine, lookback_days=252):
    """Fetch historical returns for Kelly calculation."""
    
    query = f"""
    WITH price_data AS (
        SELECT 
            sp.symbol,
            sp.trade_date,
            sp.close_price,
            LAG(sp.close_price) OVER (PARTITION BY sp.symbol ORDER BY sp.trade_date) as prev_close,
            su.market_cap / 1e9 as market_cap_b
        FROM stock_prices sp
        JOIN symbol_universe su ON sp.symbol = su.symbol
        WHERE sp.trade_date >= CURRENT_DATE - INTERVAL '{lookback_days} days'
          AND su.market_cap >= 2e9  -- Mid-cap and above
          AND su.country = 'USA'
          AND su.security_type = 'Common Stock'
    ),
    returns AS (
        SELECT 
            symbol,
            trade_date,
            (close_price / prev_close - 1) as daily_return
        FROM price_data
        WHERE prev_close IS NOT NULL
    ),
    win_loss AS (
        SELECT 
            symbol,
            COUNT(*) as total_trades,
            SUM(CASE WHEN daily_return > 0 THEN 1 ELSE 0 END) as winning_trades,
            AVG(CASE WHEN daily_return > 0 THEN daily_return ELSE NULL END) as avg_win,
            AVG(CASE WHEN daily_return <= 0 THEN ABS(daily_return) ELSE NULL END) as avg_loss,
            AVG(daily_return) as mean_return,
            STDDEV(daily_return) as std_dev,
            MAX(daily_return) as max_return,
            MIN(daily_return) as min_return
        FROM returns
        GROUP BY symbol
        HAVING COUNT(*) >= 60  -- Minimum 3 months of data
    )
    SELECT 
        wl.*,
        ms.value_score,
        ms.growth_score,
        ms.dividend_score,
        ms.composite_score
    FROM win_loss wl
    LEFT JOIN master_scores ms ON wl.symbol = ms.symbol
    """
    
    logger.info(f"Fetching {lookback_days} days of return data...")
    df = pd.read_sql(query, engine)
    logger.info(f"Retrieved data for {len(df)} stocks")
    
    return df

def calculate_kelly_position(row):
    """Calculate Kelly position size for a single stock."""
    
    # Basic Kelly inputs
    win_prob = row['winning_trades'] / row['total_trades'] if row['total_trades'] > 0 else 0
    lose_prob = 1 - win_prob
    
    # Win/loss ratio (b in Kelly formula)
    win_loss_ratio = row['avg_win'] / row['avg_loss'] if row['avg_loss'] > 0 else 0
    
    # Standard Kelly: f* = (p*b - q) / b
    if win_loss_ratio > 0:
        standard_kelly = (win_prob * win_loss_ratio - lose_prob) / win_loss_ratio
    else:
        standard_kelly = 0
    
    # Simplified Kelly for continuous outcomes: f* = μ/σ²
    # Using excess return over risk-free rate (3% annual = 0.012% daily)
    risk_free_daily = 0.03 / 252
    excess_return = row['mean_return'] - risk_free_daily
    
    if row['std_dev'] > 0:
        variance = row['std_dev'] ** 2
        continuous_kelly = excess_return / variance
    else:
        continuous_kelly = 0
    
    # Use the more conservative estimate
    raw_kelly = min(standard_kelly, continuous_kelly)
    
    # Apply Kelly fraction limits
    # Never bet more than 25% on a single position (quarter Kelly)
    # Never bet negative (no shorting in this implementation)
    kelly_fraction = max(0, min(0.25, raw_kelly))
    
    # Further adjustments based on quality scores
    quality_multiplier = 1.0
    
    if row['composite_score'] is not None:
        if row['composite_score'] >= 80:
            quality_multiplier = 1.2  # Increase position for high quality
        elif row['composite_score'] >= 60:
            quality_multiplier = 1.0  # Normal position
        elif row['composite_score'] >= 40:
            quality_multiplier = 0.8  # Reduce position for lower quality
        else:
            quality_multiplier = 0.5  # Half position for poor quality
    
    # Adjusted Kelly
    adjusted_kelly = kelly_fraction * quality_multiplier
    
    # Additional safety: cap at 10% for any single position
    final_kelly = min(0.10, adjusted_kelly)
    
    return {
        'symbol': row['symbol'],
        'calculation_date': datetime.now().date(),
        
        # Win/loss statistics
        'win_probability': round(win_prob, 4),
        'avg_win_pct': round(row['avg_win'] * 100, 4) if row['avg_win'] else 0,
        'avg_loss_pct': round(row['avg_loss'] * 100, 4) if row['avg_loss'] else 0,
        'win_loss_ratio': round(win_loss_ratio, 4),
        
        # Kelly calculations
        'standard_kelly': round(standard_kelly, 4),
        'continuous_kelly': round(continuous_kelly, 4),
        'raw_kelly_pct': round(raw_kelly * 100, 2),
        'adjusted_kelly_pct': round(adjusted_kelly * 100, 2),
        'final_position_pct': round(final_kelly * 100, 2),
        
        # Risk metrics
        'daily_volatility': round(row['std_dev'], 6),
        'annual_volatility': round(row['std_dev'] * np.sqrt(252), 4),
        'sharpe_ratio': round(excess_return / row['std_dev'] * np.sqrt(252), 4) if row['std_dev'] > 0 else 0,
        
        # Quality adjustment
        'quality_score': row['composite_score'] if row['composite_score'] else 50,
        'quality_multiplier': round(quality_multiplier, 2),
        
        # Position sizing category
        'position_category': categorize_position(final_kelly)
    }

def categorize_position(kelly_pct):
    """Categorize position size for easy interpretation."""
    if kelly_pct >= 0.08:
        return 'FULL'
    elif kelly_pct >= 0.05:
        return 'LARGE'
    elif kelly_pct >= 0.03:
        return 'MEDIUM'
    elif kelly_pct >= 0.01:
        return 'SMALL'
    elif kelly_pct > 0:
        return 'MINIMAL'
    else:
        return 'NONE'

def calculate_portfolio_allocation(df, portfolio_type='BALANCED', max_positions=20):
    """Calculate portfolio allocation using Kelly Criterion."""
    
    # Filter based on portfolio type
    if portfolio_type == 'VALUE':
        df_filtered = df[df['value_score'] >= 70] if 'value_score' in df.columns else df
    elif portfolio_type == 'GROWTH':
        df_filtered = df[df['growth_score'] >= 70] if 'growth_score' in df.columns else df
    elif portfolio_type == 'DIVIDEND':
        df_filtered = df[df['dividend_score'] >= 70] if 'dividend_score' in df.columns else df
    else:  # BALANCED
        df_filtered = df[df['composite_score'] >= 60] if 'composite_score' in df.columns else df
    
    # Sort by Kelly position size
    df_filtered = df_filtered.sort_values('final_position_pct', ascending=False)
    
    # Select top positions
    top_positions = df_filtered.head(max_positions).copy()
    
    # Normalize allocations to sum to 100%
    total_kelly = top_positions['final_position_pct'].sum()
    
    if total_kelly > 0:
        # If total Kelly > 100%, scale down proportionally
        if total_kelly > 1.0:
            top_positions['normalized_position_pct'] = (
                top_positions['final_position_pct'] / total_kelly * 100
            )
        else:
            # If total Kelly < 100%, use actual Kelly percentages
            top_positions['normalized_position_pct'] = top_positions['final_position_pct'] * 100
            
        # Calculate dollar allocations for a $100,000 portfolio
        portfolio_value = 100000
        top_positions['dollar_allocation'] = (
            top_positions['normalized_position_pct'] / 100 * portfolio_value
        )
        
        # Add portfolio metadata
        top_positions['portfolio_type'] = portfolio_type
        top_positions['total_positions'] = len(top_positions)
        top_positions['cash_reserve_pct'] = max(0, 100 - top_positions['normalized_position_pct'].sum())
        
    return top_positions

def save_kelly_allocations(engine, kelly_df, portfolio_df):
    """Save Kelly calculations and portfolio allocations to database."""
    
    # Create tables if not exist
    create_tables_query = """
    CREATE TABLE IF NOT EXISTS kelly_criterion (
        symbol VARCHAR(10) NOT NULL,
        calculation_date DATE NOT NULL,
        
        -- Win/loss statistics
        win_probability NUMERIC(6, 4),
        avg_win_pct NUMERIC(10, 4),
        avg_loss_pct NUMERIC(10, 4),
        win_loss_ratio NUMERIC(10, 4),
        
        -- Kelly calculations
        standard_kelly NUMERIC(10, 4),
        continuous_kelly NUMERIC(10, 4),
        raw_kelly_pct NUMERIC(6, 2),
        adjusted_kelly_pct NUMERIC(6, 2),
        final_position_pct NUMERIC(6, 2),
        
        -- Risk metrics
        daily_volatility NUMERIC(10, 6),
        annual_volatility NUMERIC(10, 4),
        sharpe_ratio NUMERIC(10, 4),
        
        -- Quality adjustment
        quality_score NUMERIC(6, 2),
        quality_multiplier NUMERIC(4, 2),
        
        -- Position category
        position_category VARCHAR(20),
        
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, calculation_date)
    );
    
    CREATE TABLE IF NOT EXISTS portfolio_allocations (
        symbol VARCHAR(10) NOT NULL,
        allocation_date DATE NOT NULL,
        portfolio_type VARCHAR(20) NOT NULL,
        
        -- Position sizing
        kelly_position_pct NUMERIC(6, 2),
        normalized_position_pct NUMERIC(6, 2),
        dollar_allocation NUMERIC(12, 2),
        
        -- Portfolio metadata
        total_positions INTEGER,
        cash_reserve_pct NUMERIC(6, 2),
        position_rank INTEGER,
        
        -- Risk/return metrics
        expected_return NUMERIC(10, 4),
        position_risk NUMERIC(10, 4),
        contribution_to_portfolio NUMERIC(10, 4),
        
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, allocation_date, portfolio_type)
    );
    
    CREATE INDEX IF NOT EXISTS idx_kelly_position 
        ON kelly_criterion(final_position_pct DESC);
    CREATE INDEX IF NOT EXISTS idx_kelly_quality 
        ON kelly_criterion(quality_score DESC);
    CREATE INDEX IF NOT EXISTS idx_portfolio_type 
        ON portfolio_allocations(portfolio_type, allocation_date);
    """
    
    with engine.connect() as conn:
        for statement in create_tables_query.split(';'):
            if statement.strip():
                try:
                    conn.execute(text(statement))
                except Exception as e:
                    if 'already exists' not in str(e):
                        logger.error(f"Error creating table: {e}")
        conn.commit()
    
    # Save Kelly calculations
    if not kelly_df.empty:
        temp_table = f"temp_kelly_{int(time.time() * 1000)}"
        kelly_df.to_sql(temp_table, engine, if_exists='replace', index=False)
        
        with engine.connect() as conn:
            cols = kelly_df.columns.tolist()
            cols_str = ', '.join(cols)
            
            # Upsert Kelly data
            upsert_query = f"""
                INSERT INTO kelly_criterion ({cols_str})
                SELECT {cols_str} FROM {temp_table}
                ON CONFLICT (symbol, calculation_date) DO UPDATE SET
                    win_probability = EXCLUDED.win_probability,
                    avg_win_pct = EXCLUDED.avg_win_pct,
                    avg_loss_pct = EXCLUDED.avg_loss_pct,
                    win_loss_ratio = EXCLUDED.win_loss_ratio,
                    standard_kelly = EXCLUDED.standard_kelly,
                    continuous_kelly = EXCLUDED.continuous_kelly,
                    raw_kelly_pct = EXCLUDED.raw_kelly_pct,
                    adjusted_kelly_pct = EXCLUDED.adjusted_kelly_pct,
                    final_position_pct = EXCLUDED.final_position_pct,
                    sharpe_ratio = EXCLUDED.sharpe_ratio,
                    quality_score = EXCLUDED.quality_score,
                    position_category = EXCLUDED.position_category,
                    updated_at = CURRENT_TIMESTAMP
            """
            
            conn.execute(text(upsert_query))
            conn.execute(text(f"DROP TABLE IF EXISTS {temp_table}"))
            conn.commit()
        
        logger.info(f"Saved Kelly calculations for {len(kelly_df)} stocks")
    
    # Save portfolio allocations
    if not portfolio_df.empty:
        # Prepare portfolio data
        portfolio_save = portfolio_df[['symbol', 'portfolio_type', 
                                       'final_position_pct', 'normalized_position_pct',
                                       'dollar_allocation', 'total_positions',
                                       'cash_reserve_pct']].copy()
        
        portfolio_save.columns = ['symbol', 'portfolio_type', 
                                  'kelly_position_pct', 'normalized_position_pct',
                                  'dollar_allocation', 'total_positions',
                                  'cash_reserve_pct']
        
        portfolio_save['allocation_date'] = datetime.now().date()
        portfolio_save['position_rank'] = range(1, len(portfolio_save) + 1)
        
        temp_table = f"temp_portfolio_{int(time.time() * 1000)}"
        portfolio_save.to_sql(temp_table, engine, if_exists='replace', index=False)
        
        with engine.connect() as conn:
            conn.execute(text(f"""
                INSERT INTO portfolio_allocations
                SELECT * FROM {temp_table}
                ON CONFLICT (symbol, allocation_date, portfolio_type) DO UPDATE SET
                    kelly_position_pct = EXCLUDED.kelly_position_pct,
                    normalized_position_pct = EXCLUDED.normalized_position_pct,
                    dollar_allocation = EXCLUDED.dollar_allocation,
                    total_positions = EXCLUDED.total_positions,
                    cash_reserve_pct = EXCLUDED.cash_reserve_pct,
                    position_rank = EXCLUDED.position_rank,
                    updated_at = CURRENT_TIMESTAMP
            """))
            conn.execute(text(f"DROP TABLE IF EXISTS {temp_table}"))
            conn.commit()
        
        logger.info(f"Saved portfolio allocation for {portfolio_df['portfolio_type'].iloc[0]}")

def analyze_kelly_results(engine):
    """Analyze and display Kelly Criterion results."""
    
    # Top Kelly positions
    query = """
    SELECT 
        kc.symbol,
        su.name,
        kc.final_position_pct,
        kc.win_probability * 100 as win_rate,
        kc.win_loss_ratio,
        kc.sharpe_ratio,
        kc.quality_score,
        kc.position_category
    FROM kelly_criterion kc
    JOIN symbol_universe su ON kc.symbol = su.symbol
    WHERE kc.final_position_pct > 0
    ORDER BY kc.final_position_pct DESC
    LIMIT 20
    """
    
    df = pd.read_sql(query, engine)
    
    print("\n" + "=" * 80)
    print("KELLY CRITERION POSITION SIZING")
    print("=" * 80)
    
    print("\nTOP KELLY ALLOCATIONS:")
    for _, row in df.head(10).iterrows():
        category_emoji = {
            'FULL': '🔥', 'LARGE': '📈', 'MEDIUM': '➡️',
            'SMALL': '📉', 'MINIMAL': '🔻', 'NONE': '❌'
        }.get(row['position_category'], '❓')
        
        print(f"  {row['symbol']:6s} | {category_emoji} {row['position_category']:8s} | "
              f"Size: {row['final_position_pct']:.1f}% | "
              f"Win: {row['win_rate']:.0f}% | "
              f"W/L: {row['win_loss_ratio']:.2f} | "
              f"Sharpe: {row['sharpe_ratio']:.2f}")
    
    # Portfolio allocations
    portfolio_query = """
    SELECT 
        portfolio_type,
        COUNT(*) as num_positions,
        SUM(normalized_position_pct) as total_allocated,
        AVG(normalized_position_pct) as avg_position,
        MAX(normalized_position_pct) as max_position
    FROM portfolio_allocations
    WHERE allocation_date = CURRENT_DATE
    GROUP BY portfolio_type
    """
    
    portfolio_df = pd.read_sql(portfolio_query, engine)
    
    if not portfolio_df.empty:
        print("\nPORTFOLIO ALLOCATIONS:")
        for _, row in portfolio_df.iterrows():
            print(f"  {row['portfolio_type']:10s}: "
                  f"{row['num_positions']:2d} positions | "
                  f"Total: {row['total_allocated']:.1f}% | "
                  f"Avg: {row['avg_position']:.1f}% | "
                  f"Max: {row['max_position']:.1f}%")
    
    # Risk distribution
    risk_query = """
    SELECT 
        position_category,
        COUNT(*) as count,
        AVG(final_position_pct) as avg_allocation
    FROM kelly_criterion
    WHERE calculation_date = CURRENT_DATE
    GROUP BY position_category
    ORDER BY 
        CASE position_category
            WHEN 'FULL' THEN 1
            WHEN 'LARGE' THEN 2
            WHEN 'MEDIUM' THEN 3
            WHEN 'SMALL' THEN 4
            WHEN 'MINIMAL' THEN 5
            WHEN 'NONE' THEN 6
        END
    """
    
    risk_df = pd.read_sql(risk_query, engine)
    
    print("\nPOSITION SIZE DISTRIBUTION:")
    for _, row in risk_df.iterrows():
        print(f"  {row['position_category']:8s}: {row['count']:4d} stocks "
              f"(avg {row['avg_allocation']:.2f}%)")

def main():
    """Main execution function."""
    start_time = time.time()
    log_script_start(logger, "calculate_kelly_criterion", "Calculating optimal position sizes using Kelly Criterion")
    
    print("\n" + "=" * 80)
    print("KELLY CRITERION CALCULATOR")
    print("Optimal Position Sizing for Risk-Adjusted Returns")
    print("=" * 80)
    
    try:
        # Setup database
        load_dotenv()
        db_manager = DatabaseConnectionManager()
        engine = db_manager.get_engine()
        
        # Fetch return data
        df = fetch_return_data(engine, lookback_days=252)
        
        if df.empty:
            logger.warning("No return data found")
            return
        
        # Calculate Kelly for all stocks
        logger.info("Calculating Kelly positions...")
        kelly_results = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Calculating Kelly"):
            kelly_metrics = calculate_kelly_position(row)
            kelly_results.append(kelly_metrics)
        
        kelly_df = pd.DataFrame(kelly_results)
        
        print(f"\n[INFO] Calculated Kelly positions for {len(kelly_df)} stocks")
        
        # Calculate portfolio allocations
        print("\n[INFO] Building optimal portfolios...")
        
        # Merge Kelly with original data for portfolio construction
        merged_df = pd.merge(
            kelly_df[['symbol', 'final_position_pct', 'quality_score', 'sharpe_ratio']],
            df[['symbol', 'value_score', 'growth_score', 'dividend_score', 'composite_score']],
            on='symbol',
            how='left'
        )
        
        portfolios = []
        for portfolio_type in ['BALANCED', 'VALUE', 'GROWTH', 'DIVIDEND']:
            portfolio = calculate_portfolio_allocation(merged_df, portfolio_type)
            if not portfolio.empty:
                portfolios.append(portfolio)
                save_kelly_allocations(engine, pd.DataFrame(), portfolio)
        
        # Save Kelly calculations
        save_kelly_allocations(engine, kelly_df, pd.DataFrame())
        
        # Analyze results
        analyze_kelly_results(engine)
        
        # Investment insights
        print("\n" + "=" * 80)
        print("KELLY CRITERION INSIGHTS")
        print("=" * 80)
        print("\nKey Principles:")
        print("  1. Never bet more than Kelly suggests (avoid ruin)")
        print("  2. Consider using 'Fractional Kelly' (25-50%) for safety")
        print("  3. Diversification reduces risk without sacrificing returns")
        print("  4. Rebalance when positions drift significantly")
        print("  5. Higher Sharpe ratios justify larger positions")
        
        print("\nPosition Size Guidelines:")
        print("  - FULL (8-10%): Highest conviction, best risk/reward")
        print("  - LARGE (5-8%): Strong positions with good metrics")
        print("  - MEDIUM (3-5%): Core holdings with moderate confidence")
        print("  - SMALL (1-3%): Speculative or higher risk positions")
        print("  - MINIMAL (<1%): Tracking positions or high uncertainty")
        
        # Log completion
        duration = time.time() - start_time
        log_script_end(logger, "calculate_kelly_criterion", success=True, duration=duration)
        print(f"\n[SUCCESS] Kelly Criterion calculation completed in {duration:.1f} seconds")
        
    except Exception as e:
        logger.error(f"Script failed: {e}")
        log_script_end(logger, "calculate_kelly_criterion", success=False, duration=time.time()-start_time)
        raise

if __name__ == "__main__":
    main()