"""
Calculate comprehensive risk metrics for portfolio analysis.

Metrics include:
- Sharpe Ratio (risk-adjusted returns)
- Sortino Ratio (downside risk-adjusted returns)  
- Beta (market correlation)
- Maximum Drawdown
- Value at Risk (VaR)
- Volatility metrics
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
logger = setup_logger("calculate_risk_metrics")

# Database connection
from database.db_connection_manager import DatabaseConnectionManager

def fetch_price_data(engine, lookback_days=252):
    """Fetch historical price data for risk calculations."""
    
    query = f"""
    WITH price_data AS (
        SELECT 
            sp.symbol,
            sp.trade_date,
            sp.close_price,
            sp.volume,
            su.market_cap / 1e9 as market_cap_b,
            su.sector
        FROM stock_prices sp
        JOIN symbol_universe su ON sp.symbol = su.symbol
        WHERE sp.trade_date >= CURRENT_DATE - INTERVAL '{lookback_days} days'
          AND su.market_cap >= 2e9
          AND su.country = 'USA'
          AND su.security_type = 'Common Stock'
    ),
    spy_data AS (
        SELECT 
            trade_date,
            close_price as spy_close
        FROM stock_prices
        WHERE symbol = 'SPY'
          AND trade_date >= CURRENT_DATE - INTERVAL '{lookback_days} days'
    )
    SELECT 
        pd.*,
        sd.spy_close
    FROM price_data pd
    LEFT JOIN spy_data sd ON pd.trade_date = sd.trade_date
    ORDER BY pd.symbol, pd.trade_date
    """
    
    logger.info(f"Fetching {lookback_days} days of price data...")
    df = pd.read_sql(query, engine)
    logger.info(f"Retrieved {len(df)} price records for {df['symbol'].nunique()} stocks")
    
    return df

def calculate_returns(df):
    """Calculate daily and cumulative returns."""
    
    # Calculate daily returns for each stock
    df['daily_return'] = df.groupby('symbol')['close_price'].pct_change()
    
    # Calculate SPY returns (market benchmark)
    df['market_return'] = df.groupby('symbol')['spy_close'].pct_change()
    
    # Calculate excess returns (stock - risk-free rate)
    # Using 3% annual risk-free rate
    risk_free_daily = 0.03 / 252
    df['excess_return'] = df['daily_return'] - risk_free_daily
    
    return df

def calculate_stock_metrics(symbol_data):
    """Calculate risk metrics for a single stock."""
    
    # Remove NaN values
    returns = symbol_data['daily_return'].dropna()
    excess_returns = symbol_data['excess_return'].dropna()
    market_returns = symbol_data['market_return'].dropna()
    
    if len(returns) < 20:  # Need minimum data points
        return None
    
    metrics = {
        'symbol': symbol_data['symbol'].iloc[0],
        'calculation_date': datetime.now().date(),
        
        # Basic statistics
        'annual_return': returns.mean() * 252,
        'annual_volatility': returns.std() * np.sqrt(252),
        'skewness': returns.skew(),
        'kurtosis': returns.kurtosis(),
        
        # Sharpe Ratio (excess return / volatility)
        'sharpe_ratio': (excess_returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0,
        
        # Sortino Ratio (excess return / downside volatility)
        'sortino_ratio': None,
        
        # Beta (correlation with market)
        'beta': None,
        
        # Maximum Drawdown
        'max_drawdown': None,
        
        # Value at Risk (95% confidence)
        'var_95': returns.quantile(0.05),
        
        # Conditional Value at Risk (Expected Shortfall)
        'cvar_95': returns[returns <= returns.quantile(0.05)].mean() if len(returns[returns <= returns.quantile(0.05)]) > 0 else None,
        
        # Tracking metrics
        'trading_days': len(returns),
        'positive_days': (returns > 0).sum(),
        'negative_days': (returns < 0).sum(),
        'win_rate': (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
    }
    
    # Calculate Sortino Ratio (using downside returns only)
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        downside_std = downside_returns.std() * np.sqrt(252)
        if downside_std > 0:
            metrics['sortino_ratio'] = (excess_returns.mean() * 252) / downside_std
    
    # Calculate Beta (if market data available)
    if len(market_returns) == len(returns) and market_returns.std() > 0:
        covariance = returns.cov(market_returns)
        market_variance = market_returns.var()
        if market_variance > 0:
            metrics['beta'] = covariance / market_variance
    
    # Calculate Maximum Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    metrics['max_drawdown'] = drawdown.min()
    
    # Risk-adjusted score (0-100 scale)
    risk_score = 50  # Base score
    
    # Adjust for Sharpe ratio
    if metrics['sharpe_ratio'] and metrics['sharpe_ratio'] > 0:
        risk_score += min(25, metrics['sharpe_ratio'] * 10)
    
    # Adjust for volatility
    if metrics['annual_volatility']:
        if metrics['annual_volatility'] < 0.2:  # Low volatility
            risk_score += 10
        elif metrics['annual_volatility'] > 0.4:  # High volatility
            risk_score -= 10
    
    # Adjust for max drawdown
    if metrics['max_drawdown']:
        if metrics['max_drawdown'] > -0.1:  # Small drawdown
            risk_score += 10
        elif metrics['max_drawdown'] < -0.3:  # Large drawdown
            risk_score -= 15
    
    # Adjust for win rate
    if metrics['win_rate'] > 0.55:
        risk_score += 5
    elif metrics['win_rate'] < 0.45:
        risk_score -= 5
    
    metrics['risk_adjusted_score'] = max(0, min(100, risk_score))
    
    # Categorize risk level
    if metrics['annual_volatility'] < 0.15:
        metrics['risk_category'] = 'LOW'
    elif metrics['annual_volatility'] < 0.25:
        metrics['risk_category'] = 'MODERATE'
    elif metrics['annual_volatility'] < 0.35:
        metrics['risk_category'] = 'HIGH'
    else:
        metrics['risk_category'] = 'VERY_HIGH'
    
    return metrics

def calculate_all_risk_metrics(df):
    """Calculate risk metrics for all stocks."""
    
    results = []
    symbols = df['symbol'].unique()
    
    for symbol in tqdm(symbols, desc="Calculating risk metrics"):
        symbol_data = df[df['symbol'] == symbol].copy()
        metrics = calculate_stock_metrics(symbol_data)
        
        if metrics:
            results.append(metrics)
    
    return pd.DataFrame(results)

def save_risk_metrics(engine, df):
    """Save risk metrics to database."""
    
    # Create table if not exists
    create_table_query = """
    CREATE TABLE IF NOT EXISTS risk_metrics (
        symbol VARCHAR(10) NOT NULL,
        calculation_date DATE NOT NULL,
        
        -- Return metrics
        annual_return NUMERIC(12, 4),
        annual_volatility NUMERIC(12, 4),
        skewness NUMERIC(12, 4),
        kurtosis NUMERIC(12, 4),
        
        -- Risk-adjusted metrics
        sharpe_ratio NUMERIC(12, 4),
        sortino_ratio NUMERIC(12, 4),
        beta NUMERIC(12, 4),
        
        -- Drawdown and VaR
        max_drawdown NUMERIC(12, 4),
        var_95 NUMERIC(12, 4),
        cvar_95 NUMERIC(12, 4),
        
        -- Trading statistics
        trading_days INTEGER,
        positive_days INTEGER,
        negative_days INTEGER,
        win_rate NUMERIC(6, 4),
        
        -- Composite scores
        risk_adjusted_score NUMERIC(6, 2),
        risk_category VARCHAR(20),
        
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, calculation_date)
    );
    
    CREATE INDEX IF NOT EXISTS idx_risk_metrics_sharpe 
        ON risk_metrics(sharpe_ratio DESC);
    CREATE INDEX IF NOT EXISTS idx_risk_metrics_volatility 
        ON risk_metrics(annual_volatility);
    CREATE INDEX IF NOT EXISTS idx_risk_metrics_category 
        ON risk_metrics(risk_category);
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
    
    # Save metrics
    temp_table = f"temp_risk_{int(time.time() * 1000)}"
    
    try:
        df.to_sql(temp_table, engine, if_exists='replace', index=False)
        
        with engine.connect() as conn:
            # Get column list
            cols = df.columns.tolist()
            cols_str = ', '.join(cols)
            
            # Build upsert query
            upsert_query = f"""
                INSERT INTO risk_metrics ({cols_str})
                SELECT {cols_str} FROM {temp_table}
                ON CONFLICT (symbol, calculation_date) DO UPDATE SET
                    annual_return = EXCLUDED.annual_return,
                    annual_volatility = EXCLUDED.annual_volatility,
                    sharpe_ratio = EXCLUDED.sharpe_ratio,
                    sortino_ratio = EXCLUDED.sortino_ratio,
                    beta = EXCLUDED.beta,
                    max_drawdown = EXCLUDED.max_drawdown,
                    risk_adjusted_score = EXCLUDED.risk_adjusted_score,
                    risk_category = EXCLUDED.risk_category,
                    updated_at = CURRENT_TIMESTAMP
            """
            
            conn.execute(text(upsert_query))
            conn.execute(text(f"DROP TABLE IF EXISTS {temp_table}"))
            conn.commit()
            
        logger.info(f"Saved risk metrics for {len(df)} stocks")
        
    except Exception as e:
        logger.error(f"Error saving risk metrics: {e}")
        with engine.connect() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {temp_table}"))
            conn.rollback()

def analyze_risk_results(engine):
    """Analyze and display risk metrics results."""
    
    query = """
    SELECT 
        rm.symbol,
        su.name,
        rm.annual_return * 100 as annual_return_pct,
        rm.annual_volatility * 100 as volatility_pct,
        rm.sharpe_ratio,
        rm.sortino_ratio,
        rm.beta,
        rm.max_drawdown * 100 as max_dd_pct,
        rm.risk_adjusted_score,
        rm.risk_category
    FROM risk_metrics rm
    JOIN symbol_universe su ON rm.symbol = su.symbol
    WHERE rm.sharpe_ratio IS NOT NULL
    ORDER BY rm.sharpe_ratio DESC
    """
    
    df = pd.read_sql(query, engine)
    
    print("\n" + "=" * 80)
    print("RISK METRICS ANALYSIS")
    print("=" * 80)
    
    # Top risk-adjusted returns (Sharpe Ratio)
    print("\nTOP 10 RISK-ADJUSTED RETURNS (Sharpe Ratio):")
    top_sharpe = df.nlargest(10, 'sharpe_ratio')
    for _, row in top_sharpe.iterrows():
        print(f"  {row['symbol']:6s} | Sharpe: {row['sharpe_ratio']:6.2f} | "
              f"Return: {row['annual_return_pct']:6.1f}% | "
              f"Vol: {row['volatility_pct']:5.1f}% | "
              f"Beta: {row['beta']:5.2f if row['beta'] else 0:5.2f}")
    
    # Low risk opportunities
    print("\nLOW RISK OPPORTUNITIES (Low Vol + Positive Returns):")
    low_risk = df[(df['risk_category'] == 'LOW') & (df['annual_return_pct'] > 0)].nlargest(5, 'risk_adjusted_score')
    for _, row in low_risk.iterrows():
        print(f"  {row['symbol']:6s} | Vol: {row['volatility_pct']:5.1f}% | "
              f"Return: {row['annual_return_pct']:6.1f}% | "
              f"Score: {row['risk_adjusted_score']:5.1f}")
    
    # Risk distribution
    print("\nRISK DISTRIBUTION:")
    risk_dist = df['risk_category'].value_counts()
    for category in ['LOW', 'MODERATE', 'HIGH', 'VERY_HIGH']:
        count = risk_dist.get(category, 0)
        pct = count * 100 / len(df) if len(df) > 0 else 0
        print(f"  {category:10s}: {count:4d} ({pct:5.1f}%)")
    
    # Summary statistics
    print("\nMARKET STATISTICS:")
    print(f"  Average Annual Return: {df['annual_return_pct'].mean():6.2f}%")
    print(f"  Average Volatility: {df['volatility_pct'].mean():6.2f}%")
    print(f"  Average Sharpe Ratio: {df['sharpe_ratio'].mean():6.2f}")
    print(f"  Average Beta: {df['beta'].mean():6.2f}")

def main():
    """Main execution function."""
    start_time = time.time()
    log_script_start(logger, "calculate_risk_metrics", "Calculating comprehensive risk metrics")
    
    print("\n" + "=" * 80)
    print("RISK METRICS CALCULATOR")
    print("Sharpe, Sortino, Beta, VaR, and more")
    print("=" * 80)
    
    try:
        # Setup database
        load_dotenv()
        db_manager = DatabaseConnectionManager()
        engine = db_manager.get_engine()
        
        # Fetch price data (1 year history)
        df = fetch_price_data(engine, lookback_days=252)
        
        if df.empty:
            logger.warning("No price data found")
            return
        
        # Calculate returns
        df = calculate_returns(df)
        
        # Calculate risk metrics for all stocks
        logger.info("Calculating risk metrics...")
        metrics_df = calculate_all_risk_metrics(df)
        
        print(f"\n[INFO] Calculated risk metrics for {len(metrics_df)} stocks")
        
        # Save to database
        save_risk_metrics(engine, metrics_df)
        
        # Analyze results
        analyze_risk_results(engine)
        
        # Investment insights
        print("\n" + "=" * 80)
        print("INVESTMENT INSIGHTS")
        print("=" * 80)
        print("\nOptimal Portfolio Characteristics:")
        print("  - Sharpe Ratio > 1.0 (good risk-adjusted returns)")
        print("  - Sortino Ratio > 1.5 (limited downside risk)")
        print("  - Beta 0.8-1.2 (moderate market correlation)")
        print("  - Max Drawdown > -20% (limited losses)")
        print("  - Annual Volatility < 25% (manageable risk)")
        
        # Log completion
        duration = time.time() - start_time
        log_script_end(logger, "calculate_risk_metrics", success=True, duration=duration)
        print(f"\n[SUCCESS] Risk metrics calculation completed in {duration:.1f} seconds")
        
    except Exception as e:
        logger.error(f"Script failed: {e}")
        log_script_end(logger, "calculate_risk_metrics", success=False, duration=time.time()-start_time)
        raise

if __name__ == "__main__":
    main()