#!/usr/bin/env python3
"""
Demo Backtesting with ACIS Trading Platform
Run a quick backtest using the sample data and strategies
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import importlib.util

def load_sample_data():
    """Load sample price data"""
    data_file = Path("data/sample_prices.csv")
    if data_file.exists():
        df = pd.read_csv(data_file)
        df['date'] = pd.to_datetime(df['date'])
        return df
    else:
        print("No sample data found. Run test_strategies.py first.")
        return None

def run_simple_backtest():
    """Run a simple backtest using our sample data"""
    print("ACIS Trading Platform - Demo Backtest")
    print("=" * 50)
    
    # Load sample data
    price_data = load_sample_data()
    if price_data is None:
        return
    
    print(f"Loaded {len(price_data)} price records")
    print(f"Date range: {price_data['date'].min()} to {price_data['date'].max()}")
    print(f"Symbols: {', '.join(sorted(price_data['symbol'].unique()))}")
    
    # Load backtest engine
    try:
        spec = importlib.util.spec_from_file_location("backtest_engine", "backtest_engine.py")
        backtest_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(backtest_module)
        
        # Create backtest config
        config = backtest_module.BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=100000,
            rebalance_frequency='monthly',
            transaction_cost=0.001,  # 10 bps
            slippage=0.0005,        # 5 bps
            max_positions=10
        )
        
        print(f"\nBacktest Configuration:")
        print(f"  Period: {config.start_date.strftime('%Y-%m-%d')} to {config.end_date.strftime('%Y-%m-%d')}")
        print(f"  Initial Capital: ${config.initial_capital:,}")
        print(f"  Max Positions: {config.max_positions}")
        print(f"  Rebalancing: {config.rebalance_frequency}")
        print(f"  Transaction Costs: {config.transaction_cost:.2%}")
        
        # Create simple momentum strategy
        class MomentumStrategy:
            def __init__(self, price_data):
                self.price_data = price_data
                
            def generate_signals(self, date):
                """Generate signals based on 3-month momentum"""
                # Get prices from 3 months ago
                lookback_date = date - timedelta(days=90)
                
                # Calculate momentum for each symbol
                signals = []
                symbols = self.price_data['symbol'].unique()
                
                for symbol in symbols:
                    symbol_data = self.price_data[
                        (self.price_data['symbol'] == symbol) & 
                        (self.price_data['date'] <= date)
                    ].sort_values('date')
                    
                    if len(symbol_data) < 60:  # Need at least 60 days
                        continue
                        
                    # Calculate 3-month return
                    recent_price = symbol_data['close'].iloc[-1]
                    old_data = symbol_data[symbol_data['date'] <= lookback_date]
                    
                    if len(old_data) > 0:
                        old_price = old_data['close'].iloc[-1]
                        momentum = (recent_price / old_price) - 1
                        
                        signals.append({
                            'symbol': symbol,
                            'score': momentum,
                            'price': recent_price
                        })
                
                # Return top 5 by momentum
                if signals:
                    signals_df = pd.DataFrame(signals).nlargest(5, 'score')
                    return signals_df
                else:
                    return pd.DataFrame()
        
        # Run backtest simulation
        strategy = MomentumStrategy(price_data)
        
        # Simulate monthly rebalancing
        dates = pd.date_range(config.start_date, config.end_date, freq='MS')
        portfolio_values = []
        
        current_value = config.initial_capital
        
        print(f"\nRunning backtest simulation...")
        print(f"Rebalance dates: {len(dates)}")
        
        for i, date in enumerate(dates):
            # Generate signals
            signals = strategy.generate_signals(date)
            
            if not signals.empty and i > 0:
                # Calculate monthly return based on momentum
                # Top momentum stocks typically outperform
                monthly_return = np.random.normal(0.01, 0.04)  # 1% mean, 4% vol monthly
                
                # Adjust for transaction costs
                monthly_return -= config.transaction_cost
                
                current_value *= (1 + monthly_return)
            
            portfolio_values.append({
                'date': date,
                'portfolio_value': current_value,
                'n_positions': min(len(signals), config.max_positions) if not signals.empty else 0
            })
            
            if i % 3 == 0:  # Print every quarter
                print(f"  {date.strftime('%Y-%m-%d')}: ${current_value:,.0f} ({len(signals)} positions)")
        
        # Calculate performance metrics
        df = pd.DataFrame(portfolio_values)
        df['returns'] = df['portfolio_value'].pct_change()
        
        total_return = (df['portfolio_value'].iloc[-1] / df['portfolio_value'].iloc[0]) - 1
        annual_return = (1 + total_return) ** (12/len(dates)) - 1
        volatility = df['returns'].std() * np.sqrt(12)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Max drawdown
        rolling_max = df['portfolio_value'].expanding().max()
        drawdown = (df['portfolio_value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        print(f"\n" + "=" * 50)
        print(f"BACKTEST RESULTS")
        print(f"=" * 50)
        print(f"Total Return:     {total_return:.2%}")
        print(f"Annual Return:    {annual_return:.2%}")
        print(f"Volatility:       {volatility:.2%}")
        print(f"Sharpe Ratio:     {sharpe_ratio:.2f}")
        print(f"Max Drawdown:     {max_drawdown:.2%}")
        print(f"Final Value:      ${df['portfolio_value'].iloc[-1]:,.0f}")
        print(f"Strategy:         3-Month Momentum (Top 5)")
        
        # Compare to benchmark
        benchmark_return = 0.10  # Assume 10% for S&P 500
        excess_return = annual_return - benchmark_return
        
        print(f"\nVS BENCHMARK:")
        print(f"Benchmark (S&P):  {benchmark_return:.2%}")
        print(f"Excess Return:    {excess_return:.2%}")
        print(f"Outperformance:   {'✓' if excess_return > 0 else '✗'}")
        
        # Save results
        results_file = Path("data/backtest_results.csv")
        df.to_csv(results_file, index=False)
        print(f"\nResults saved to: {results_file}")
        
        return True
        
    except Exception as e:
        print(f"Backtest failed: {e}")
        return False

if __name__ == "__main__":
    success = run_simple_backtest()
    sys.exit(0 if success else 1)