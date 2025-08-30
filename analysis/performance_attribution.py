"""
Performance Attribution System for Portfolio Analysis.

This module decomposes portfolio returns to identify sources of alpha:
- Asset allocation effects
- Security selection effects  
- Sector allocation impacts
- Factor exposures (value, growth, momentum)
- Timing and rebalancing effects
- Risk-adjusted performance metrics
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
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging first
from utils.logging_config import setup_logger, log_script_start, log_script_end
logger = setup_logger("performance_attribution")

# Database connection
from database.db_connection_manager import DatabaseConnectionManager

class PerformanceAttributor:
    """Decompose portfolio performance into component sources."""
    
    def __init__(self, engine):
        self.engine = engine
        self.attribution_results = {}
        
    def fetch_portfolio_data(self, start_date, end_date, strategy='BALANCED'):
        """Fetch portfolio holdings and benchmark data."""
        
        query = f"""
        WITH portfolio_holdings AS (
            -- Get portfolio positions from backtest or live trading
            SELECT 
                bt.trade_date,
                bt.symbol,
                bt.shares,
                bt.trade_type,
                sp.close as price,
                su.sector,
                su.market_cap / 1e9 as market_cap_b
            FROM backtest_trades bt
            JOIN stock_prices sp ON bt.symbol = sp.symbol AND bt.trade_date = sp.trade_date
            JOIN symbol_universe su ON bt.symbol = su.symbol
            WHERE bt.trade_date BETWEEN '{start_date}' AND '{end_date}'
        ),
        benchmark_data AS (
            -- S&P 500 as benchmark
            SELECT 
                trade_date,
                close as spy_close,
                LAG(close) OVER (ORDER BY trade_date) as prev_close
            FROM stock_prices
            WHERE symbol = 'SPY'
              AND trade_date BETWEEN '{start_date}' AND '{end_date}'
        ),
        factor_data AS (
            -- Factor exposures
            SELECT 
                ms.symbol,
                ms.calculation_date,
                ms.value_score / 100 as value_factor,
                ms.growth_score / 100 as growth_factor,
                ms.composite_score / 100 as quality_factor,
                tb.momentum_score / 100 as momentum_factor
            FROM master_scores ms
            LEFT JOIN technical_breakouts tb ON ms.symbol = tb.symbol
        )
        SELECT 
            ph.*,
            bd.spy_close,
            bd.prev_close as spy_prev_close,
            fd.value_factor,
            fd.growth_factor,
            fd.quality_factor,
            fd.momentum_factor
        FROM portfolio_holdings ph
        LEFT JOIN benchmark_data bd ON ph.trade_date = bd.trade_date
        LEFT JOIN factor_data fd ON ph.symbol = fd.symbol
        ORDER BY ph.trade_date, ph.symbol
        """
        
        logger.info(f"Fetching portfolio data from {start_date} to {end_date}...")
        df = pd.read_sql(query, self.engine)
        
        # Also fetch daily prices for all holdings
        holdings_query = f"""
        SELECT 
            sp.symbol,
            sp.trade_date,
            sp.close,
            sp.open,
            LAG(sp.close) OVER (PARTITION BY sp.symbol ORDER BY sp.trade_date) as prev_close,
            su.sector
        FROM stock_prices sp
        JOIN symbol_universe su ON sp.symbol = su.symbol
        WHERE sp.trade_date BETWEEN '{start_date}' AND '{end_date}'
          AND sp.symbol IN (SELECT DISTINCT symbol FROM backtest_trades)
        """
        
        prices_df = pd.read_sql(holdings_query, self.engine)
        
        return df, prices_df
    
    def calculate_returns(self, prices_df):
        """Calculate daily returns for holdings and sectors."""
        
        # Stock returns
        prices_df['daily_return'] = (prices_df['close'] / prices_df['prev_close'] - 1).fillna(0)
        
        # Sector returns (equal-weighted within sector)
        sector_returns = prices_df.groupby(['trade_date', 'sector'])['daily_return'].mean().reset_index()
        sector_returns.columns = ['trade_date', 'sector', 'sector_return']
        
        # Market return (average of all stocks)
        market_returns = prices_df.groupby('trade_date')['daily_return'].mean().reset_index()
        market_returns.columns = ['trade_date', 'market_return']
        
        # Merge back
        prices_df = prices_df.merge(sector_returns, on=['trade_date', 'sector'], how='left')
        prices_df = prices_df.merge(market_returns, on='trade_date', how='left')
        
        return prices_df
    
    def brinson_attribution(self, portfolio_df, benchmark_df):
        """
        Brinson attribution decomposes returns into:
        1. Asset Allocation Effect
        2. Security Selection Effect
        3. Interaction Effect
        """
        
        attribution = {}
        
        # Calculate portfolio and benchmark weights by sector
        portfolio_weights = portfolio_df.groupby(['trade_date', 'sector']).agg({
            'market_value': 'sum'
        }).reset_index()
        
        total_value = portfolio_weights.groupby('trade_date')['market_value'].transform('sum')
        portfolio_weights['weight'] = portfolio_weights['market_value'] / total_value
        
        # Benchmark weights (assume equal sector weights for simplicity)
        sectors = portfolio_weights['sector'].unique()
        benchmark_weights = pd.DataFrame({
            'sector': sectors,
            'benchmark_weight': 1.0 / len(sectors)
        })
        
        # Calculate sector returns
        sector_returns = portfolio_df.groupby(['trade_date', 'sector'])['daily_return'].mean().reset_index()
        
        # Merge data
        analysis_df = portfolio_weights.merge(benchmark_weights, on='sector')
        analysis_df = analysis_df.merge(sector_returns, on=['trade_date', 'sector'])
        
        # Calculate attribution components
        analysis_df['allocation_effect'] = (
            (analysis_df['weight'] - analysis_df['benchmark_weight']) * 
            analysis_df['daily_return']
        )
        
        analysis_df['selection_effect'] = (
            analysis_df['benchmark_weight'] * 
            (analysis_df['daily_return'] - analysis_df['daily_return'].mean())
        )
        
        analysis_df['interaction_effect'] = (
            (analysis_df['weight'] - analysis_df['benchmark_weight']) * 
            (analysis_df['daily_return'] - analysis_df['daily_return'].mean())
        )
        
        # Aggregate attribution
        attribution['allocation_effect'] = analysis_df['allocation_effect'].sum()
        attribution['selection_effect'] = analysis_df['selection_effect'].sum()
        attribution['interaction_effect'] = analysis_df['interaction_effect'].sum()
        attribution['total_effect'] = (
            attribution['allocation_effect'] + 
            attribution['selection_effect'] + 
            attribution['interaction_effect']
        )
        
        # By sector
        sector_attribution = analysis_df.groupby('sector').agg({
            'allocation_effect': 'sum',
            'selection_effect': 'sum',
            'interaction_effect': 'sum'
        }).round(4)
        
        return attribution, sector_attribution
    
    def factor_attribution(self, portfolio_df):
        """
        Decompose returns by factor exposures:
        - Value factor
        - Growth factor
        - Momentum factor
        - Quality factor
        """
        
        factors = ['value_factor', 'growth_factor', 'momentum_factor', 'quality_factor']
        factor_returns = {}
        
        for factor in factors:
            if factor in portfolio_df.columns:
                # Calculate factor exposure
                exposure = portfolio_df[factor].mean()
                
                # Calculate return from factor
                # High factor stocks vs low factor stocks
                high_factor = portfolio_df[portfolio_df[factor] > portfolio_df[factor].median()]
                low_factor = portfolio_df[portfolio_df[factor] <= portfolio_df[factor].median()]
                
                high_return = high_factor['daily_return'].mean() if len(high_factor) > 0 else 0
                low_return = low_factor['daily_return'].mean() if len(low_factor) > 0 else 0
                
                factor_return = high_return - low_return
                contribution = exposure * factor_return
                
                factor_returns[factor] = {
                    'exposure': exposure,
                    'factor_return': factor_return,
                    'contribution': contribution
                }
        
        return factor_returns
    
    def calculate_risk_attribution(self, portfolio_df):
        """Calculate risk contribution by position and sector."""
        
        # Position-level risk
        position_vol = portfolio_df.groupby('symbol')['daily_return'].std() * np.sqrt(252)
        position_weights = portfolio_df.groupby('symbol')['weight'].mean()
        
        # Risk contribution
        position_risk = (position_vol * position_weights).fillna(0)
        
        # Sector-level risk
        sector_vol = portfolio_df.groupby('sector')['daily_return'].std() * np.sqrt(252)
        sector_weights = portfolio_df.groupby('sector')['weight'].sum()
        sector_risk = (sector_vol * sector_weights).fillna(0)
        
        # Concentration risk
        herfindahl = (position_weights ** 2).sum()
        effective_n = 1 / herfindahl if herfindahl > 0 else 1
        
        return {
            'position_risk': position_risk.to_dict(),
            'sector_risk': sector_risk.to_dict(),
            'concentration_hhi': herfindahl,
            'effective_positions': effective_n
        }
    
    def timing_attribution(self, portfolio_df, rebalance_dates):
        """Measure impact of rebalancing timing."""
        
        timing_effects = []
        
        for i in range(len(rebalance_dates) - 1):
            period_start = rebalance_dates[i]
            period_end = rebalance_dates[i + 1]
            
            # Get period data
            period_df = portfolio_df[
                (portfolio_df['trade_date'] >= period_start) & 
                (portfolio_df['trade_date'] < period_end)
            ]
            
            if period_df.empty:
                continue
            
            # Calculate buy-and-hold return
            buy_hold_return = (1 + period_df.groupby('symbol')['daily_return'].mean()).prod() - 1
            
            # Calculate actual return with rebalancing
            actual_return = (1 + period_df['daily_return']).prod() - 1
            
            # Timing effect
            timing_effect = actual_return - buy_hold_return
            
            timing_effects.append({
                'period_start': period_start,
                'period_end': period_end,
                'buy_hold_return': buy_hold_return,
                'actual_return': actual_return,
                'timing_effect': timing_effect
            })
        
        return pd.DataFrame(timing_effects)
    
    def calculate_alpha_beta(self, portfolio_returns, benchmark_returns):
        """Calculate portfolio alpha and beta vs benchmark."""
        
        # Ensure aligned data
        merged = pd.DataFrame({
            'portfolio': portfolio_returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        if len(merged) < 20:
            return {'alpha': 0, 'beta': 1, 'r_squared': 0}
        
        # Calculate beta (covariance / variance)
        covariance = merged['portfolio'].cov(merged['benchmark'])
        benchmark_variance = merged['benchmark'].var()
        
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 1
        
        # Calculate alpha (excess return not explained by beta)
        portfolio_mean = merged['portfolio'].mean() * 252
        benchmark_mean = merged['benchmark'].mean() * 252
        alpha = portfolio_mean - beta * benchmark_mean
        
        # R-squared
        correlation = merged['portfolio'].corr(merged['benchmark'])
        r_squared = correlation ** 2
        
        return {
            'alpha': alpha,
            'beta': beta,
            'r_squared': r_squared,
            'tracking_error': (merged['portfolio'] - merged['benchmark']).std() * np.sqrt(252),
            'information_ratio': alpha / ((merged['portfolio'] - merged['benchmark']).std() * np.sqrt(252))
        }
    
    def generate_attribution_report(self, portfolio_df, prices_df):
        """Generate comprehensive attribution report."""
        
        # Calculate returns
        prices_df = self.calculate_returns(prices_df)
        
        # Add market value to portfolio
        portfolio_df['market_value'] = portfolio_df['shares'] * portfolio_df['price']
        portfolio_df['weight'] = portfolio_df.groupby('trade_date')['market_value'].transform(
            lambda x: x / x.sum()
        )
        
        # Merge returns
        portfolio_df = portfolio_df.merge(
            prices_df[['symbol', 'trade_date', 'daily_return', 'sector_return', 'market_return']],
            on=['symbol', 'trade_date'],
            how='left'
        )
        
        # Portfolio return
        portfolio_df['weighted_return'] = portfolio_df['weight'] * portfolio_df['daily_return']
        daily_portfolio_returns = portfolio_df.groupby('trade_date')['weighted_return'].sum()
        
        # Benchmark return (market)
        daily_benchmark_returns = portfolio_df.groupby('trade_date')['market_return'].first()
        
        # 1. Brinson Attribution
        brinson_total, brinson_sector = self.brinson_attribution(portfolio_df, prices_df)
        
        # 2. Factor Attribution
        factor_attr = self.factor_attribution(portfolio_df)
        
        # 3. Risk Attribution
        risk_attr = self.calculate_risk_attribution(portfolio_df)
        
        # 4. Alpha/Beta
        alpha_beta = self.calculate_alpha_beta(daily_portfolio_returns, daily_benchmark_returns)
        
        # Compile report
        report = {
            'summary': {
                'total_return': (1 + daily_portfolio_returns).prod() - 1,
                'benchmark_return': (1 + daily_benchmark_returns).prod() - 1,
                'excess_return': ((1 + daily_portfolio_returns).prod() - 1) - 
                               ((1 + daily_benchmark_returns).prod() - 1),
                'alpha': alpha_beta['alpha'],
                'beta': alpha_beta['beta'],
                'r_squared': alpha_beta['r_squared'],
                'tracking_error': alpha_beta['tracking_error'],
                'information_ratio': alpha_beta['information_ratio']
            },
            'brinson_attribution': brinson_total,
            'sector_attribution': brinson_sector.to_dict(),
            'factor_attribution': factor_attr,
            'risk_attribution': risk_attr
        }
        
        return report

def save_attribution_results(engine, report, strategy, period_start, period_end):
    """Save attribution results to database."""
    
    create_table_query = """
    CREATE TABLE IF NOT EXISTS performance_attribution (
        attribution_id SERIAL PRIMARY KEY,
        strategy VARCHAR(20) NOT NULL,
        period_start DATE NOT NULL,
        period_end DATE NOT NULL,
        
        -- Summary metrics
        total_return NUMERIC(10, 6),
        benchmark_return NUMERIC(10, 6),
        excess_return NUMERIC(10, 6),
        alpha NUMERIC(10, 6),
        beta NUMERIC(10, 4),
        r_squared NUMERIC(10, 4),
        tracking_error NUMERIC(10, 6),
        information_ratio NUMERIC(10, 4),
        
        -- Brinson attribution
        allocation_effect NUMERIC(10, 6),
        selection_effect NUMERIC(10, 6),
        interaction_effect NUMERIC(10, 6),
        
        -- Factor contributions
        value_contribution NUMERIC(10, 6),
        growth_contribution NUMERIC(10, 6),
        momentum_contribution NUMERIC(10, 6),
        quality_contribution NUMERIC(10, 6),
        
        -- Risk metrics
        concentration_hhi NUMERIC(10, 6),
        effective_positions NUMERIC(10, 2),
        
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE TABLE IF NOT EXISTS sector_attribution (
        attribution_id INTEGER REFERENCES performance_attribution(attribution_id),
        sector VARCHAR(50) NOT NULL,
        allocation_effect NUMERIC(10, 6),
        selection_effect NUMERIC(10, 6),
        interaction_effect NUMERIC(10, 6),
        sector_weight NUMERIC(10, 4),
        benchmark_weight NUMERIC(10, 4),
        sector_return NUMERIC(10, 6)
    );
    
    CREATE INDEX IF NOT EXISTS idx_perf_attr_strategy 
        ON performance_attribution(strategy);
    CREATE INDEX IF NOT EXISTS idx_perf_attr_alpha 
        ON performance_attribution(alpha DESC);
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
    
    # Save main attribution
    with engine.connect() as conn:
        insert_query = """
        INSERT INTO performance_attribution (
            strategy, period_start, period_end,
            total_return, benchmark_return, excess_return,
            alpha, beta, r_squared, tracking_error, information_ratio,
            allocation_effect, selection_effect, interaction_effect,
            value_contribution, growth_contribution, momentum_contribution, quality_contribution,
            concentration_hhi, effective_positions
        ) VALUES (
            :strategy, :period_start, :period_end,
            :total_return, :benchmark_return, :excess_return,
            :alpha, :beta, :r_squared, :tracking_error, :information_ratio,
            :allocation_effect, :selection_effect, :interaction_effect,
            :value_contribution, :growth_contribution, :momentum_contribution, :quality_contribution,
            :concentration_hhi, :effective_positions
        ) RETURNING attribution_id
        """
        
        # Extract factor contributions
        factor_attr = report.get('factor_attribution', {})
        value_contrib = factor_attr.get('value_factor', {}).get('contribution', 0)
        growth_contrib = factor_attr.get('growth_factor', {}).get('contribution', 0)
        momentum_contrib = factor_attr.get('momentum_factor', {}).get('contribution', 0)
        quality_contrib = factor_attr.get('quality_factor', {}).get('contribution', 0)
        
        result = conn.execute(text(insert_query), {
            'strategy': strategy,
            'period_start': period_start,
            'period_end': period_end,
            'total_return': report['summary']['total_return'],
            'benchmark_return': report['summary']['benchmark_return'],
            'excess_return': report['summary']['excess_return'],
            'alpha': report['summary']['alpha'],
            'beta': report['summary']['beta'],
            'r_squared': report['summary']['r_squared'],
            'tracking_error': report['summary']['tracking_error'],
            'information_ratio': report['summary']['information_ratio'],
            'allocation_effect': report['brinson_attribution']['allocation_effect'],
            'selection_effect': report['brinson_attribution']['selection_effect'],
            'interaction_effect': report['brinson_attribution']['interaction_effect'],
            'value_contribution': value_contrib,
            'growth_contribution': growth_contrib,
            'momentum_contribution': momentum_contrib,
            'quality_contribution': quality_contrib,
            'concentration_hhi': report['risk_attribution']['concentration_hhi'],
            'effective_positions': report['risk_attribution']['effective_positions']
        })
        
        conn.commit()

def display_attribution_report(report):
    """Display attribution report in readable format."""
    
    print("\n" + "=" * 80)
    print("PERFORMANCE ATTRIBUTION REPORT")
    print("=" * 80)
    
    # Summary
    summary = report['summary']
    print("\nPERFORMANCE SUMMARY:")
    print(f"  Total Return:     {summary['total_return']*100:+.2f}%")
    print(f"  Benchmark Return: {summary['benchmark_return']*100:+.2f}%")
    print(f"  Excess Return:    {summary['excess_return']*100:+.2f}%")
    print(f"  Alpha (Annual):   {summary['alpha']*100:+.2f}%")
    print(f"  Beta:             {summary['beta']:.2f}")
    print(f"  R-Squared:        {summary['r_squared']:.2f}")
    print(f"  Tracking Error:   {summary['tracking_error']*100:.2f}%")
    print(f"  Info Ratio:       {summary['information_ratio']:.2f}")
    
    # Brinson Attribution
    brinson = report['brinson_attribution']
    print("\nBRINSON ATTRIBUTION:")
    print(f"  Allocation Effect:  {brinson['allocation_effect']*100:+.2f}%")
    print(f"  Selection Effect:   {brinson['selection_effect']*100:+.2f}%")
    print(f"  Interaction Effect: {brinson['interaction_effect']*100:+.2f}%")
    print(f"  Total Effect:       {brinson['total_effect']*100:+.2f}%")
    
    # Factor Attribution
    print("\nFACTOR ATTRIBUTION:")
    factor_attr = report.get('factor_attribution', {})
    for factor_name, metrics in factor_attr.items():
        clean_name = factor_name.replace('_factor', '').capitalize()
        print(f"  {clean_name:10s}: Exposure={metrics['exposure']:.2f}, "
              f"Return={metrics['factor_return']*100:+.2f}%, "
              f"Contribution={metrics['contribution']*100:+.2f}%")
    
    # Risk Attribution
    risk = report['risk_attribution']
    print("\nRISK METRICS:")
    print(f"  Concentration (HHI): {risk['concentration_hhi']:.3f}")
    print(f"  Effective Positions: {risk['effective_positions']:.1f}")
    
    # Top risk contributors
    if 'position_risk' in risk and risk['position_risk']:
        print("\n  Top Risk Contributors:")
        sorted_risks = sorted(risk['position_risk'].items(), key=lambda x: x[1], reverse=True)[:5]
        for symbol, risk_contrib in sorted_risks:
            print(f"    {symbol}: {risk_contrib*100:.1f}%")

def main():
    """Main execution function."""
    start_time = time.time()
    log_script_start(logger, "performance_attribution", "Analyzing portfolio performance attribution")
    
    print("\n" + "=" * 80)
    print("PERFORMANCE ATTRIBUTION SYSTEM")
    print("Decomposing Returns to Identify Alpha Sources")
    print("=" * 80)
    
    try:
        # Setup database
        load_dotenv()
        db_manager = DatabaseConnectionManager()
        engine = db_manager.get_engine()
        
        # Set analysis period
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=365)  # 1 year analysis
        
        print(f"\n[INFO] Attribution period: {start_date} to {end_date}")
        
        # Initialize attributor
        attributor = PerformanceAttributor(engine)
        
        # Analyze each strategy
        strategies = ['VALUE', 'GROWTH', 'DIVIDEND', 'BALANCED']
        
        for strategy in strategies:
            print(f"\n[INFO] Analyzing {strategy} strategy...")
            
            # Fetch portfolio data
            portfolio_df, prices_df = attributor.fetch_portfolio_data(
                start_date, end_date, strategy
            )
            
            if portfolio_df.empty or prices_df.empty:
                logger.warning(f"No data available for {strategy}")
                continue
            
            # Generate attribution report
            report = attributor.generate_attribution_report(portfolio_df, prices_df)
            
            # Save results
            save_attribution_results(engine, report, strategy, start_date, end_date)
            
            # Display report
            display_attribution_report(report)
        
        # Investment insights
        print("\n" + "=" * 80)
        print("ATTRIBUTION INSIGHTS")
        print("=" * 80)
        print("\nUnderstanding Your Alpha:")
        print("  • Allocation Effect: Value from sector/asset over/underweights")
        print("  • Selection Effect: Value from picking winners within sectors")
        print("  • Factor Exposure: Returns from value/growth/momentum tilts")
        print("  • Timing Effect: Value from rebalancing decisions")
        
        print("\nKey Metrics Explained:")
        print("  • Alpha: Excess return not explained by market beta")
        print("  • Information Ratio: Risk-adjusted active return")
        print("  • R-Squared: How much variance is explained by benchmark")
        print("  • Tracking Error: Volatility of excess returns")
        
        print("\nActionable Insights:")
        print("  1. Focus on areas with consistent positive attribution")
        print("  2. Reduce exposure to negative attribution sources")
        print("  3. Monitor factor exposures for style drift")
        print("  4. Optimize sector allocation based on attribution")
        print("  5. Improve selection within strong sectors")
        
        # Log completion
        duration = time.time() - start_time
        log_script_end(logger, "performance_attribution", success=True, duration=duration)
        print(f"\n[SUCCESS] Performance attribution completed in {duration:.1f} seconds")
        
    except Exception as e:
        logger.error(f"Script failed: {e}")
        log_script_end(logger, "performance_attribution", success=False, duration=time.time()-start_time)
        raise

if __name__ == "__main__":
    main()