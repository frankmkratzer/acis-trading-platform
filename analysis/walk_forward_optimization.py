"""
Walk-Forward Optimization Framework for Strategy Parameters.

This module implements rolling window optimization to:
- Avoid overfitting by using out-of-sample testing
- Dynamically adapt strategy parameters over time
- Test parameter stability across different market regimes
- Optimize multiple objectives (return, risk, Sharpe)
- Generate robust parameter sets for production trading
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
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# Setup logging first
from utils.logging_config import setup_logger, log_script_start, log_script_end
logger = setup_logger("walk_forward_optimization")

# Database connection
from database.db_connection_manager import DatabaseConnectionManager

class WalkForwardOptimizer:
    """Walk-forward optimization engine for strategy parameters."""
    
    def __init__(self, engine, strategy_type='VALUE'):
        self.engine = engine
        self.strategy_type = strategy_type
        self.optimization_results = []
        self.parameter_stability = {}
        
    def define_parameter_space(self):
        """Define parameter search space for each strategy."""
        
        if self.strategy_type == 'VALUE':
            return {
                'pe_weight': [0.2, 0.3, 0.4],
                'pb_weight': [0.1, 0.2, 0.3],
                'cash_flow_weight': [0.2, 0.3, 0.4],
                'fscore_threshold': [5, 6, 7],
                'num_stocks': [8, 10, 12, 15],
                'rebalance_freq': ['monthly', 'quarterly']
            }
        elif self.strategy_type == 'GROWTH':
            return {
                'revenue_growth_weight': [0.2, 0.3, 0.4],
                'earnings_growth_weight': [0.2, 0.3, 0.4],
                'momentum_weight': [0.1, 0.2, 0.3],
                'quality_threshold': [60, 70, 80],
                'num_stocks': [8, 10, 12, 15],
                'rebalance_freq': ['monthly', 'quarterly']
            }
        elif self.strategy_type == 'DIVIDEND':
            return {
                'yield_weight': [0.3, 0.4, 0.5],
                'growth_weight': [0.2, 0.3, 0.4],
                'payout_ratio_max': [0.5, 0.6, 0.7],
                'min_years': [5, 10, 15],
                'num_stocks': [10, 15, 20],
                'rebalance_freq': ['quarterly', 'annual']
            }
        else:  # BALANCED
            return {
                'value_weight': [0.25, 0.33, 0.4],
                'growth_weight': [0.25, 0.33, 0.4],
                'quality_weight': [0.2, 0.34, 0.4],
                'momentum_weight': [0.0, 0.1, 0.2],
                'num_stocks': [15, 20, 25],
                'rebalance_freq': ['monthly', 'quarterly']
            }
    
    def fetch_optimization_data(self, start_date, end_date):
        """Fetch data for optimization period."""
        
        query = f"""
        WITH combined_data AS (
            SELECT 
                sp.symbol,
                sp.trade_date,
                sp.open,
                sp.high,
                sp.low,
                sp.close,
                sp.volume,
                
                -- Fundamental metrics
                cfo.pe_ratio,
                cfo.price_to_book,
                cfo.dividend_yield,
                cfo.payout_ratio,
                ps.fscore,
                az.zscore,
                
                -- Growth metrics
                cfo.revenue_growth_yoy,
                cfo.quarterly_earnings_growth_yoy,
                
                -- Quality scores
                ms.value_score,
                ms.growth_score,
                ms.dividend_score,
                ms.composite_score,
                
                -- Technical
                tb.momentum_score,
                rm.sharpe_ratio,
                
                -- Market data
                su.market_cap / 1e9 as market_cap_b,
                su.sector
                
            FROM stock_prices sp
            JOIN symbol_universe su ON sp.symbol = su.symbol
            LEFT JOIN company_fundamentals_overview cfo ON sp.symbol = cfo.symbol
            LEFT JOIN piotroski_scores ps ON sp.symbol = ps.symbol
            LEFT JOIN altman_zscores az ON sp.symbol = az.symbol
            LEFT JOIN master_scores ms ON sp.symbol = ms.symbol
            LEFT JOIN technical_breakouts tb ON sp.symbol = tb.symbol
            LEFT JOIN risk_metrics rm ON sp.symbol = rm.symbol
            
            WHERE sp.trade_date BETWEEN '{start_date}' AND '{end_date}'
              AND su.market_cap >= 2e9
              AND su.country = 'USA'
              AND su.security_type = 'Common Stock'
        )
        SELECT * FROM combined_data
        ORDER BY trade_date, symbol
        """
        
        logger.info(f"Fetching data from {start_date} to {end_date}...")
        df = pd.read_sql(query, self.engine)
        logger.info(f"Retrieved {len(df)} records for optimization")
        
        return df
    
    def calculate_strategy_score(self, row, params):
        """Calculate strategy score based on parameters."""
        
        if self.strategy_type == 'VALUE':
            score = 0
            
            # PE component (lower is better, invert)
            if pd.notna(row['pe_ratio']) and row['pe_ratio'] > 0:
                pe_score = min(100, 1000 / row['pe_ratio'])  # Invert PE
                score += pe_score * params['pe_weight']
            
            # PB component (lower is better, invert)
            if pd.notna(row['price_to_book']) and row['price_to_book'] > 0:
                pb_score = min(100, 10 / row['price_to_book'])  # Invert PB
                score += pb_score * params['pb_weight']
            
            # Cash flow component (use value_score as proxy)
            if pd.notna(row['value_score']):
                score += row['value_score'] * params['cash_flow_weight']
            
            # F-Score filter
            if pd.notna(row['fscore']) and row['fscore'] < params['fscore_threshold']:
                score *= 0.5  # Penalize low F-Score
                
        elif self.strategy_type == 'GROWTH':
            score = 0
            
            # Revenue growth
            if pd.notna(row['revenue_growth_yoy']):
                rev_score = min(100, row['revenue_growth_yoy'] * 2)
                score += rev_score * params['revenue_growth_weight']
            
            # Earnings growth
            if pd.notna(row['quarterly_earnings_growth_yoy']):
                earn_score = min(100, row['quarterly_earnings_growth_yoy'] * 2)
                score += earn_score * params['earnings_growth_weight']
            
            # Momentum
            if pd.notna(row['momentum_score']):
                score += row['momentum_score'] * params['momentum_weight']
            
            # Quality filter
            if pd.notna(row['composite_score']) and row['composite_score'] < params['quality_threshold']:
                score *= 0.7
                
        elif self.strategy_type == 'DIVIDEND':
            score = 0
            
            # Yield component
            if pd.notna(row['dividend_yield']) and row['dividend_yield'] > 0:
                yield_score = min(100, row['dividend_yield'] * 20)
                score += yield_score * params['yield_weight']
            
            # Growth component (use dividend_score as proxy)
            if pd.notna(row['dividend_score']):
                score += row['dividend_score'] * params['growth_weight']
            
            # Payout ratio filter
            if pd.notna(row['payout_ratio']) and row['payout_ratio'] > params['payout_ratio_max']:
                score *= 0.6  # Penalize high payout
                
        else:  # BALANCED
            score = 0
            
            if pd.notna(row['value_score']):
                score += row['value_score'] * params.get('value_weight', 0.33)
            
            if pd.notna(row['growth_score']):
                score += row['growth_score'] * params.get('growth_weight', 0.33)
            
            if pd.notna(row['composite_score']):
                score += row['composite_score'] * params.get('quality_weight', 0.34)
            
            if pd.notna(row['momentum_score']):
                score += row['momentum_score'] * params.get('momentum_weight', 0)
        
        return score
    
    def simulate_strategy(self, df, params, start_date, end_date):
        """Simulate strategy with given parameters."""
        
        # Initialize portfolio
        initial_capital = 100000
        cash = initial_capital
        positions = {}
        portfolio_values = []
        
        # Get rebalance dates
        rebalance_freq = params.get('rebalance_freq', 'quarterly')
        rebalance_dates = self.get_rebalance_dates(start_date, end_date, rebalance_freq)
        
        # Group data by date
        daily_data = df.groupby('trade_date')
        
        for date in sorted(df['trade_date'].unique()):
            date_data = daily_data.get_group(date)
            
            # Rebalance if needed
            if pd.Timestamp(date) in rebalance_dates:
                # Calculate scores for all stocks
                scores = {}
                for _, row in date_data.iterrows():
                    score = self.calculate_strategy_score(row, params)
                    if score > 0:
                        scores[row['symbol']] = score
                
                # Select top stocks
                num_stocks = params.get('num_stocks', 10)
                top_stocks = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:num_stocks]
                selected_symbols = [s[0] for s in top_stocks]
                
                # Sell positions not in new selection
                for symbol in list(positions.keys()):
                    if symbol not in selected_symbols:
                        price = date_data[date_data['symbol'] == symbol]['close'].values
                        if len(price) > 0:
                            cash += positions[symbol] * price[0]
                            del positions[symbol]
                
                # Buy new positions (equal weight)
                if selected_symbols:
                    position_value = cash / len(selected_symbols)
                    new_cash = cash
                    
                    for symbol in selected_symbols:
                        price = date_data[date_data['symbol'] == symbol]['close'].values
                        if len(price) > 0 and price[0] > 0:
                            shares = int(position_value / price[0])
                            if symbol not in positions:
                                positions[symbol] = 0
                            positions[symbol] = shares
                            new_cash -= shares * price[0]
                    
                    cash = new_cash
            
            # Calculate portfolio value
            portfolio_value = cash
            for symbol, shares in positions.items():
                price = date_data[date_data['symbol'] == symbol]['close'].values
                if len(price) > 0:
                    portfolio_value += shares * price[0]
            
            portfolio_values.append(portfolio_value)
        
        # Calculate performance metrics
        if len(portfolio_values) < 2:
            return None
        
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        total_return = (portfolio_values[-1] - initial_capital) / initial_capital
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
        sharpe = (annual_return - 0.03) / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        cumulative = np.array(portfolio_values)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'final_value': portfolio_values[-1]
        }
    
    def get_rebalance_dates(self, start_date, end_date, frequency):
        """Generate rebalance dates."""
        
        dates = []
        current = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        
        while current <= end:
            dates.append(current)
            if frequency == 'monthly':
                current = current + pd.DateOffset(months=1)
            elif frequency == 'quarterly':
                current = current + pd.DateOffset(months=3)
            elif frequency == 'annual':
                current = current + pd.DateOffset(years=1)
            else:
                current = current + pd.DateOffset(weeks=1)
        
        return dates
    
    def optimize_window(self, train_start, train_end, test_start, test_end):
        """Optimize parameters on training window and test on out-of-sample."""
        
        logger.info(f"Optimizing from {train_start} to {train_end}")
        
        # Fetch training data
        train_df = self.fetch_optimization_data(train_start, train_end)
        
        if train_df.empty:
            logger.warning("No training data available")
            return None
        
        # Define parameter space
        param_space = self.define_parameter_space()
        
        # Generate all parameter combinations
        param_names = list(param_space.keys())
        param_values = list(param_space.values())
        param_combinations = list(product(*param_values))
        
        # Test each combination
        best_params = None
        best_sharpe = -999
        
        for combo in tqdm(param_combinations, desc="Testing parameters"):
            params = dict(zip(param_names, combo))
            
            # Simulate on training period
            performance = self.simulate_strategy(train_df, params, train_start, train_end)
            
            if performance and performance['sharpe_ratio'] > best_sharpe:
                best_sharpe = performance['sharpe_ratio']
                best_params = params
                best_train_performance = performance
        
        if best_params is None:
            logger.warning("No valid parameters found")
            return None
        
        # Test best parameters on out-of-sample period
        logger.info(f"Testing best params from {test_start} to {test_end}")
        test_df = self.fetch_optimization_data(test_start, test_end)
        
        if test_df.empty:
            logger.warning("No test data available")
            return None
        
        test_performance = self.simulate_strategy(test_df, best_params, test_start, test_end)
        
        return {
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'best_params': best_params,
            'train_performance': best_train_performance,
            'test_performance': test_performance,
            'in_sample_sharpe': best_train_performance['sharpe_ratio'],
            'out_sample_sharpe': test_performance['sharpe_ratio'] if test_performance else 0,
            'parameter_stability': abs(best_train_performance['sharpe_ratio'] - 
                                     (test_performance['sharpe_ratio'] if test_performance else 0))
        }
    
    def run_walk_forward(self, start_date, end_date, window_months=12, step_months=3):
        """Run complete walk-forward optimization."""
        
        results = []
        current_start = pd.Timestamp(start_date)
        final_end = pd.Timestamp(end_date)
        
        while current_start < final_end:
            # Define windows
            train_end = current_start + pd.DateOffset(months=window_months)
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=step_months)
            
            if test_end > final_end:
                test_end = final_end
            
            if test_start >= final_end:
                break
            
            # Optimize this window
            window_result = self.optimize_window(
                current_start, train_end,
                test_start, test_end
            )
            
            if window_result:
                results.append(window_result)
                self.optimization_results.append(window_result)
            
            # Step forward
            current_start = current_start + pd.DateOffset(months=step_months)
        
        return results
    
    def analyze_parameter_stability(self):
        """Analyze stability of optimal parameters across windows."""
        
        if not self.optimization_results:
            return {}
        
        # Extract parameters from each window
        param_history = {}
        
        for result in self.optimization_results:
            params = result['best_params']
            for param_name, param_value in params.items():
                if param_name not in param_history:
                    param_history[param_name] = []
                param_history[param_name].append(param_value)
        
        # Calculate stability metrics
        stability_metrics = {}
        
        for param_name, values in param_history.items():
            if len(values) > 1:
                # Check if numeric or categorical
                if isinstance(values[0], (int, float)):
                    stability_metrics[param_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else 0,
                        'min': min(values),
                        'max': max(values),
                        'stable': np.std(values) / np.mean(values) < 0.3 if np.mean(values) != 0 else False
                    }
                else:
                    # Categorical parameter
                    mode = max(set(values), key=values.count)
                    stability_metrics[param_name] = {
                        'mode': mode,
                        'consistency': values.count(mode) / len(values),
                        'stable': values.count(mode) / len(values) > 0.6
                    }
        
        self.parameter_stability = stability_metrics
        return stability_metrics

def save_optimization_results(engine, optimizer, strategy):
    """Save walk-forward optimization results to database."""
    
    create_table_query = """
    CREATE TABLE IF NOT EXISTS walk_forward_results (
        result_id SERIAL PRIMARY KEY,
        strategy VARCHAR(20) NOT NULL,
        
        -- Window dates
        train_start DATE,
        train_end DATE,
        test_start DATE,
        test_end DATE,
        
        -- Optimal parameters (stored as JSON)
        best_params JSONB,
        
        -- In-sample performance
        train_return NUMERIC(10, 6),
        train_sharpe NUMERIC(10, 4),
        train_volatility NUMERIC(10, 6),
        train_drawdown NUMERIC(10, 6),
        
        -- Out-of-sample performance
        test_return NUMERIC(10, 6),
        test_sharpe NUMERIC(10, 4),
        test_volatility NUMERIC(10, 6),
        test_drawdown NUMERIC(10, 6),
        
        -- Stability metrics
        parameter_stability NUMERIC(10, 4),
        sharpe_degradation NUMERIC(10, 4),
        
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE TABLE IF NOT EXISTS parameter_stability (
        strategy VARCHAR(20) NOT NULL,
        parameter_name VARCHAR(50) NOT NULL,
        
        -- For numeric parameters
        mean_value NUMERIC(10, 4),
        std_value NUMERIC(10, 4),
        cv_value NUMERIC(10, 4),
        min_value NUMERIC(10, 4),
        max_value NUMERIC(10, 4),
        
        -- For categorical parameters
        mode_value VARCHAR(50),
        consistency NUMERIC(6, 4),
        
        -- Stability flag
        is_stable BOOLEAN,
        
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (strategy, parameter_name)
    );
    
    CREATE INDEX IF NOT EXISTS idx_wf_strategy 
        ON walk_forward_results(strategy);
    CREATE INDEX IF NOT EXISTS idx_wf_sharpe 
        ON walk_forward_results(test_sharpe DESC);
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
    
    # Save optimization results
    for result in optimizer.optimization_results:
        with engine.connect() as conn:
            # Convert params to JSON string
            import json
            params_json = json.dumps(result['best_params'])
            
            insert_query = """
            INSERT INTO walk_forward_results (
                strategy, train_start, train_end, test_start, test_end,
                best_params,
                train_return, train_sharpe, train_volatility, train_drawdown,
                test_return, test_sharpe, test_volatility, test_drawdown,
                parameter_stability, sharpe_degradation
            ) VALUES (
                :strategy, :train_start, :train_end, :test_start, :test_end,
                :best_params::jsonb,
                :train_return, :train_sharpe, :train_volatility, :train_drawdown,
                :test_return, :test_sharpe, :test_volatility, :test_drawdown,
                :parameter_stability, :sharpe_degradation
            )
            """
            
            conn.execute(text(insert_query), {
                'strategy': strategy,
                'train_start': result['train_start'],
                'train_end': result['train_end'],
                'test_start': result['test_start'],
                'test_end': result['test_end'],
                'best_params': params_json,
                'train_return': result['train_performance']['total_return'],
                'train_sharpe': result['train_performance']['sharpe_ratio'],
                'train_volatility': result['train_performance']['volatility'],
                'train_drawdown': result['train_performance']['max_drawdown'],
                'test_return': result['test_performance']['total_return'] if result['test_performance'] else None,
                'test_sharpe': result['test_performance']['sharpe_ratio'] if result['test_performance'] else None,
                'test_volatility': result['test_performance']['volatility'] if result['test_performance'] else None,
                'test_drawdown': result['test_performance']['max_drawdown'] if result['test_performance'] else None,
                'parameter_stability': result['parameter_stability'],
                'sharpe_degradation': result['in_sample_sharpe'] - result['out_sample_sharpe']
            })
            conn.commit()
    
    # Save parameter stability
    for param_name, metrics in optimizer.parameter_stability.items():
        with engine.connect() as conn:
            if 'mean' in metrics:
                # Numeric parameter
                insert_query = """
                INSERT INTO parameter_stability (
                    strategy, parameter_name,
                    mean_value, std_value, cv_value, min_value, max_value,
                    is_stable
                ) VALUES (
                    :strategy, :parameter_name,
                    :mean_value, :std_value, :cv_value, :min_value, :max_value,
                    :is_stable
                ) ON CONFLICT (strategy, parameter_name) DO UPDATE SET
                    mean_value = EXCLUDED.mean_value,
                    std_value = EXCLUDED.std_value,
                    cv_value = EXCLUDED.cv_value,
                    min_value = EXCLUDED.min_value,
                    max_value = EXCLUDED.max_value,
                    is_stable = EXCLUDED.is_stable,
                    updated_at = CURRENT_TIMESTAMP
                """
                
                conn.execute(text(insert_query), {
                    'strategy': strategy,
                    'parameter_name': param_name,
                    'mean_value': metrics['mean'],
                    'std_value': metrics['std'],
                    'cv_value': metrics['cv'],
                    'min_value': metrics['min'],
                    'max_value': metrics['max'],
                    'is_stable': metrics['stable']
                })
            else:
                # Categorical parameter
                insert_query = """
                INSERT INTO parameter_stability (
                    strategy, parameter_name,
                    mode_value, consistency, is_stable
                ) VALUES (
                    :strategy, :parameter_name,
                    :mode_value, :consistency, :is_stable
                ) ON CONFLICT (strategy, parameter_name) DO UPDATE SET
                    mode_value = EXCLUDED.mode_value,
                    consistency = EXCLUDED.consistency,
                    is_stable = EXCLUDED.is_stable,
                    updated_at = CURRENT_TIMESTAMP
                """
                
                conn.execute(text(insert_query), {
                    'strategy': strategy,
                    'parameter_name': param_name,
                    'mode_value': str(metrics['mode']),
                    'consistency': metrics['consistency'],
                    'is_stable': metrics['stable']
                })
            
            conn.commit()

def analyze_optimization_results(optimizer):
    """Analyze and display walk-forward optimization results."""
    
    print("\n" + "=" * 80)
    print("WALK-FORWARD OPTIMIZATION RESULTS")
    print("=" * 80)
    
    if not optimizer.optimization_results:
        print("No optimization results available")
        return
    
    # Summary statistics
    in_sample_sharpes = [r['in_sample_sharpe'] for r in optimizer.optimization_results]
    out_sample_sharpes = [r['out_sample_sharpe'] for r in optimizer.optimization_results]
    
    print(f"\nStrategy: {optimizer.strategy_type}")
    print(f"Number of windows: {len(optimizer.optimization_results)}")
    print(f"\nIn-Sample Sharpe:  Mean={np.mean(in_sample_sharpes):.2f}, "
          f"Std={np.std(in_sample_sharpes):.2f}")
    print(f"Out-Sample Sharpe: Mean={np.mean(out_sample_sharpes):.2f}, "
          f"Std={np.std(out_sample_sharpes):.2f}")
    print(f"Sharpe Degradation: {np.mean(in_sample_sharpes) - np.mean(out_sample_sharpes):.2f}")
    
    # Parameter stability
    print("\nPARAMETER STABILITY:")
    for param_name, metrics in optimizer.parameter_stability.items():
        if 'mean' in metrics:
            stability = "✅ STABLE" if metrics['stable'] else "⚠️ UNSTABLE"
            print(f"  {param_name:20s}: {stability} | "
                  f"Mean={metrics['mean']:.2f}, CV={metrics['cv']:.2f}")
        else:
            stability = "✅ STABLE" if metrics['stable'] else "⚠️ UNSTABLE"
            print(f"  {param_name:20s}: {stability} | "
                  f"Mode={metrics['mode']}, Consistency={metrics['consistency']:.1%}")
    
    # Best window
    best_window = max(optimizer.optimization_results, key=lambda x: x['out_sample_sharpe'])
    print("\nBEST OUT-OF-SAMPLE WINDOW:")
    print(f"  Period: {best_window['test_start']} to {best_window['test_end']}")
    print(f"  Sharpe Ratio: {best_window['out_sample_sharpe']:.2f}")
    print(f"  Total Return: {best_window['test_performance']['total_return']*100:.1f}%")
    print(f"  Parameters: {best_window['best_params']}")

def main():
    """Main execution function."""
    start_time = time.time()
    log_script_start(logger, "walk_forward_optimization", "Running walk-forward parameter optimization")
    
    print("\n" + "=" * 80)
    print("WALK-FORWARD OPTIMIZATION")
    print("Robust Parameter Selection Through Rolling Windows")
    print("=" * 80)
    
    try:
        # Setup database
        load_dotenv()
        db_manager = DatabaseConnectionManager()
        engine = db_manager.get_engine()
        
        # Set optimization period
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=365*3)  # 3 years of data
        
        print(f"\n[INFO] Optimization period: {start_date} to {end_date}")
        
        # Run optimization for each strategy
        strategies = ['VALUE', 'GROWTH', 'DIVIDEND', 'BALANCED']
        
        for strategy in strategies:
            print(f"\n[INFO] Optimizing {strategy} strategy...")
            
            optimizer = WalkForwardOptimizer(engine, strategy)
            
            # Run walk-forward optimization
            # 12-month training, 3-month test, step 3 months
            results = optimizer.run_walk_forward(
                start_date, end_date,
                window_months=12,
                step_months=3
            )
            
            if results:
                # Analyze parameter stability
                optimizer.analyze_parameter_stability()
                
                # Save results
                save_optimization_results(engine, optimizer, strategy)
                
                # Display analysis
                analyze_optimization_results(optimizer)
        
        # Investment insights
        print("\n" + "=" * 80)
        print("WALK-FORWARD INSIGHTS")
        print("=" * 80)
        print("\nKey Findings:")
        print("  1. Stable parameters perform better out-of-sample")
        print("  2. Overfitting is reduced through rolling optimization")
        print("  3. Parameter adaptation captures regime changes")
        print("  4. Sharpe degradation indicates robustness")
        print("  5. Simple parameters often outperform complex ones")
        
        print("\nBest Practices:")
        print("  - Use stable parameters for live trading")
        print("  - Re-optimize quarterly with new data")
        print("  - Monitor out-of-sample performance")
        print("  - Prefer parameters with low CV (coefficient of variation)")
        print("  - Test across different market conditions")
        
        # Log completion
        duration = time.time() - start_time
        log_script_end(logger, "walk_forward_optimization", success=True, duration=duration)
        print(f"\n[SUCCESS] Walk-forward optimization completed in {duration:.1f} seconds")
        
    except Exception as e:
        logger.error(f"Script failed: {e}")
        log_script_end(logger, "walk_forward_optimization", success=False, duration=time.time()-start_time)
        raise

if __name__ == "__main__":
    main()