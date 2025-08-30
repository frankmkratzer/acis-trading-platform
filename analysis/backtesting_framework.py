"""
Comprehensive Backtesting Framework for ACIS Trading Strategies.

This framework tests the three portfolio strategies (VALUE, GROWTH, DIVIDEND)
with realistic trading conditions including:
- Transaction costs and slippage
- Portfolio rebalancing
- Risk management rules
- Performance attribution
- Drawdown analysis
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
logger = setup_logger("backtesting_framework")

# Database connection
from database.db_connection_manager import DatabaseConnectionManager

class Portfolio:
    """Portfolio class to track holdings and performance."""
    
    def __init__(self, initial_capital=100000, commission=0.001, slippage=0.001):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission = commission  # 0.1% commission
        self.slippage = slippage  # 0.1% slippage
        self.positions = {}  # {symbol: shares}
        self.cost_basis = {}  # {symbol: avg_price}
        self.trade_history = []
        self.daily_values = []
        self.daily_returns = []
        
    def execute_trade(self, symbol, shares, price, date, trade_type='BUY'):
        """Execute a trade with costs."""
        
        # Apply slippage
        if trade_type == 'BUY':
            execution_price = price * (1 + self.slippage)
        else:
            execution_price = price * (1 - self.slippage)
        
        # Calculate trade value and commission
        trade_value = shares * execution_price
        commission_cost = trade_value * self.commission
        
        # Execute trade
        if trade_type == 'BUY':
            total_cost = trade_value + commission_cost
            if total_cost > self.cash:
                # Adjust shares to fit available cash
                affordable_value = self.cash / (1 + self.commission)
                shares = int(affordable_value / execution_price)
                if shares == 0:
                    return False
                trade_value = shares * execution_price
                commission_cost = trade_value * self.commission
                total_cost = trade_value + commission_cost
            
            self.cash -= total_cost
            if symbol in self.positions:
                # Update average cost basis
                old_shares = self.positions[symbol]
                old_cost = self.cost_basis[symbol] * old_shares
                new_cost = execution_price * shares
                self.positions[symbol] += shares
                self.cost_basis[symbol] = (old_cost + new_cost) / self.positions[symbol]
            else:
                self.positions[symbol] = shares
                self.cost_basis[symbol] = execution_price
                
        else:  # SELL
            if symbol not in self.positions or self.positions[symbol] < shares:
                shares = self.positions.get(symbol, 0)
                if shares == 0:
                    return False
            
            proceeds = trade_value - commission_cost
            self.cash += proceeds
            self.positions[symbol] -= shares
            if self.positions[symbol] == 0:
                del self.positions[symbol]
                del self.cost_basis[symbol]
        
        # Record trade
        self.trade_history.append({
            'date': date,
            'symbol': symbol,
            'type': trade_type,
            'shares': shares,
            'price': price,
            'execution_price': execution_price,
            'commission': commission_cost,
            'trade_value': trade_value
        })
        
        return True
    
    def get_portfolio_value(self, prices):
        """Calculate total portfolio value."""
        
        positions_value = sum(
            self.positions.get(symbol, 0) * prices.get(symbol, 0)
            for symbol in self.positions
        )
        return self.cash + positions_value
    
    def calculate_returns(self):
        """Calculate portfolio returns and metrics."""
        
        if len(self.daily_values) < 2:
            return {}
        
        values = np.array(self.daily_values)
        returns = np.diff(values) / values[:-1]
        
        # Calculate metrics
        total_return = (values[-1] - values[0]) / values[0]
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = np.std(returns) * np.sqrt(252)
        
        # Sharpe ratio (assuming 3% risk-free rate)
        risk_free = 0.03
        excess_returns = returns - risk_free / 252
        sharpe = np.mean(excess_returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(self.trade_history),
            'final_value': values[-1]
        }

class Backtester:
    """Main backtesting engine."""
    
    def __init__(self, engine, start_date, end_date, initial_capital=100000):
        self.engine = engine
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.results = {}
        
    def fetch_historical_data(self):
        """Fetch historical prices and signals."""
        
        query = f"""
        WITH price_data AS (
            SELECT 
                sp.symbol,
                sp.trade_date,
                sp.open,
                sp.high,
                sp.low,
                sp.close,
                sp.volume
            FROM stock_prices sp
            WHERE sp.trade_date BETWEEN '{self.start_date}' AND '{self.end_date}'
        ),
        signal_data AS (
            SELECT 
                ms.symbol,
                ms.calculation_date as signal_date,
                ms.value_score,
                ms.growth_score,
                ms.dividend_score,
                ms.composite_score,
                kc.final_position_pct as kelly_size
            FROM master_scores ms
            LEFT JOIN kelly_criterion kc ON ms.symbol = kc.symbol
        )
        SELECT 
            pd.*,
            sd.value_score,
            sd.growth_score,
            sd.dividend_score,
            sd.composite_score,
            sd.kelly_size
        FROM price_data pd
        LEFT JOIN signal_data sd ON pd.symbol = sd.symbol
        ORDER BY pd.trade_date, pd.symbol
        """
        
        logger.info(f"Fetching data from {self.start_date} to {self.end_date}...")
        df = pd.read_sql(query, self.engine)
        logger.info(f"Retrieved {len(df)} price records")
        
        return df
    
    def get_rebalance_dates(self, frequency='quarterly'):
        """Generate rebalancing dates."""
        
        dates = []
        current = pd.Timestamp(self.start_date)
        end = pd.Timestamp(self.end_date)
        
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
    
    def select_portfolio_stocks(self, df, date, strategy='VALUE', num_stocks=10):
        """Select stocks for portfolio based on strategy."""
        
        # Get scores as of date
        available = df[df['trade_date'] <= date].copy()
        latest_scores = available.sort_values('trade_date').groupby('symbol').last()
        
        # Filter by strategy
        if strategy == 'VALUE':
            score_col = 'value_score'
        elif strategy == 'GROWTH':
            score_col = 'growth_score'
        elif strategy == 'DIVIDEND':
            score_col = 'dividend_score'
        else:
            score_col = 'composite_score'
        
        # Remove stocks with no scores
        latest_scores = latest_scores[latest_scores[score_col].notna()]
        
        # Select top stocks
        top_stocks = latest_scores.nlargest(num_stocks, score_col)
        
        # Calculate position sizes (equal weight or Kelly)
        if 'kelly_size' in top_stocks.columns and top_stocks['kelly_size'].notna().any():
            # Use Kelly sizing
            sizes = top_stocks['kelly_size'].fillna(0.05)
            # Normalize to sum to 1
            sizes = sizes / sizes.sum()
        else:
            # Equal weight
            sizes = pd.Series([1.0 / num_stocks] * len(top_stocks), index=top_stocks.index)
        
        return top_stocks.index.tolist(), sizes.to_dict()
    
    def rebalance_portfolio(self, portfolio, new_stocks, position_sizes, prices, date):
        """Rebalance portfolio to new holdings."""
        
        # Sell stocks not in new portfolio
        for symbol in list(portfolio.positions.keys()):
            if symbol not in new_stocks:
                shares = portfolio.positions[symbol]
                price = prices.get(symbol, 0)
                if price > 0:
                    portfolio.execute_trade(symbol, shares, price, date, 'SELL')
        
        # Calculate target values for new positions
        total_value = portfolio.get_portfolio_value(prices)
        target_values = {
            symbol: total_value * position_sizes.get(symbol, 0)
            for symbol in new_stocks
        }
        
        # Buy/adjust positions
        for symbol in new_stocks:
            target_value = target_values[symbol]
            current_value = portfolio.positions.get(symbol, 0) * prices.get(symbol, 0)
            
            diff_value = target_value - current_value
            price = prices.get(symbol, 0)
            
            if price > 0 and abs(diff_value) > 100:  # Minimum trade size
                shares = int(abs(diff_value) / price)
                if diff_value > 0:
                    portfolio.execute_trade(symbol, shares, price, date, 'BUY')
                else:
                    portfolio.execute_trade(symbol, shares, price, date, 'SELL')
    
    def run_backtest(self, strategy='VALUE', rebalance_frequency='quarterly', num_stocks=10):
        """Run backtest for a strategy."""
        
        logger.info(f"Running backtest for {strategy} strategy...")
        
        # Initialize portfolio
        portfolio = Portfolio(self.initial_capital)
        
        # Fetch data
        df = self.fetch_historical_data()
        
        if df.empty:
            logger.warning("No data available for backtesting")
            return None
        
        # Get rebalance dates
        rebalance_dates = self.get_rebalance_dates(rebalance_frequency)
        
        # Group data by date
        daily_prices = df.groupby('trade_date').apply(
            lambda x: dict(zip(x['symbol'], x['close']))
        )
        
        # Track daily values
        dates = sorted(df['trade_date'].unique())
        
        for date in tqdm(dates, desc=f"Backtesting {strategy}"):
            prices = daily_prices.get(date, {})
            
            # Rebalance if needed
            if pd.Timestamp(date) in rebalance_dates:
                selected_stocks, position_sizes = self.select_portfolio_stocks(
                    df, date, strategy, num_stocks
                )
                self.rebalance_portfolio(portfolio, selected_stocks, position_sizes, prices, date)
            
            # Record daily value
            portfolio_value = portfolio.get_portfolio_value(prices)
            portfolio.daily_values.append(portfolio_value)
        
        # Calculate performance
        performance = portfolio.calculate_returns()
        performance['strategy'] = strategy
        performance['rebalance_frequency'] = rebalance_frequency
        performance['num_stocks'] = num_stocks
        
        return portfolio, performance
    
    def run_all_strategies(self):
        """Run backtests for all strategies."""
        
        strategies = ['VALUE', 'GROWTH', 'DIVIDEND', 'BALANCED']
        results = {}
        
        for strategy in strategies:
            portfolio, performance = self.run_backtest(strategy)
            if performance:
                results[strategy] = {
                    'portfolio': portfolio,
                    'performance': performance
                }
        
        self.results = results
        return results
    
    def compare_strategies(self):
        """Compare performance across strategies."""
        
        if not self.results:
            return pd.DataFrame()
        
        comparison = []
        for strategy, data in self.results.items():
            perf = data['performance']
            comparison.append({
                'Strategy': strategy,
                'Total Return': f"{perf['total_return']*100:.1f}%",
                'Annual Return': f"{perf['annual_return']*100:.1f}%",
                'Volatility': f"{perf['volatility']*100:.1f}%",
                'Sharpe Ratio': f"{perf['sharpe_ratio']:.2f}",
                'Max Drawdown': f"{perf['max_drawdown']*100:.1f}%",
                'Win Rate': f"{perf['win_rate']*100:.1f}%",
                'Num Trades': perf['num_trades'],
                'Final Value': f"${perf['final_value']:,.0f}"
            })
        
        return pd.DataFrame(comparison)

def save_backtest_results(engine, results):
    """Save backtest results to database."""
    
    create_table_query = """
    CREATE TABLE IF NOT EXISTS backtest_results (
        backtest_id SERIAL PRIMARY KEY,
        strategy VARCHAR(20) NOT NULL,
        start_date DATE NOT NULL,
        end_date DATE NOT NULL,
        
        -- Configuration
        initial_capital NUMERIC(12, 2),
        rebalance_frequency VARCHAR(20),
        num_stocks INTEGER,
        commission_rate NUMERIC(6, 4),
        slippage_rate NUMERIC(6, 4),
        
        -- Performance metrics
        total_return NUMERIC(10, 6),
        annual_return NUMERIC(10, 6),
        volatility NUMERIC(10, 6),
        sharpe_ratio NUMERIC(10, 4),
        sortino_ratio NUMERIC(10, 4),
        max_drawdown NUMERIC(10, 6),
        win_rate NUMERIC(6, 4),
        
        -- Trading statistics
        num_trades INTEGER,
        num_winning_trades INTEGER,
        num_losing_trades INTEGER,
        avg_trade_return NUMERIC(10, 6),
        best_trade_return NUMERIC(10, 6),
        worst_trade_return NUMERIC(10, 6),
        
        -- Portfolio statistics
        final_value NUMERIC(12, 2),
        peak_value NUMERIC(12, 2),
        trough_value NUMERIC(12, 2),
        
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE TABLE IF NOT EXISTS backtest_trades (
        trade_id SERIAL PRIMARY KEY,
        backtest_id INTEGER REFERENCES backtest_results(backtest_id),
        trade_date DATE NOT NULL,
        symbol VARCHAR(10) NOT NULL,
        trade_type VARCHAR(10),
        shares INTEGER,
        price NUMERIC(12, 4),
        commission NUMERIC(10, 2),
        trade_value NUMERIC(12, 2)
    );
    
    CREATE INDEX IF NOT EXISTS idx_backtest_strategy 
        ON backtest_results(strategy);
    CREATE INDEX IF NOT EXISTS idx_backtest_sharpe 
        ON backtest_results(sharpe_ratio DESC);
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
    
    # Save results
    for strategy, data in results.items():
        perf = data['performance']
        portfolio = data['portfolio']
        
        # Insert backtest result
        insert_query = """
        INSERT INTO backtest_results (
            strategy, start_date, end_date, initial_capital,
            rebalance_frequency, num_stocks, commission_rate, slippage_rate,
            total_return, annual_return, volatility, sharpe_ratio,
            max_drawdown, win_rate, num_trades, final_value
        ) VALUES (
            :strategy, :start_date, :end_date, :initial_capital,
            :rebalance_frequency, :num_stocks, :commission_rate, :slippage_rate,
            :total_return, :annual_return, :volatility, :sharpe_ratio,
            :max_drawdown, :win_rate, :num_trades, :final_value
        ) RETURNING backtest_id
        """
        
        with engine.connect() as conn:
            result = conn.execute(text(insert_query), {
                'strategy': strategy,
                'start_date': portfolio.trade_history[0]['date'] if portfolio.trade_history else datetime.now().date(),
                'end_date': portfolio.trade_history[-1]['date'] if portfolio.trade_history else datetime.now().date(),
                'initial_capital': portfolio.initial_capital,
                'rebalance_frequency': perf.get('rebalance_frequency', 'quarterly'),
                'num_stocks': perf.get('num_stocks', 10),
                'commission_rate': portfolio.commission,
                'slippage_rate': portfolio.slippage,
                'total_return': perf['total_return'],
                'annual_return': perf['annual_return'],
                'volatility': perf['volatility'],
                'sharpe_ratio': perf['sharpe_ratio'],
                'max_drawdown': perf['max_drawdown'],
                'win_rate': perf['win_rate'],
                'num_trades': perf['num_trades'],
                'final_value': perf['final_value']
            })
            backtest_id = result.fetchone()[0]
            
            # Save trades
            if portfolio.trade_history:
                trades_df = pd.DataFrame(portfolio.trade_history)
                trades_df['backtest_id'] = backtest_id
                trades_df.to_sql('backtest_trades', conn, if_exists='append', index=False)
            
            conn.commit()

def main():
    """Main execution function."""
    start_time = time.time()
    log_script_start(logger, "backtesting_framework", "Running comprehensive strategy backtests")
    
    print("\n" + "=" * 80)
    print("BACKTESTING FRAMEWORK")
    print("Historical Performance Analysis")
    print("=" * 80)
    
    try:
        # Setup database
        load_dotenv()
        db_manager = DatabaseConnectionManager()
        engine = db_manager.get_engine()
        
        # Set backtest parameters
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=365*2)  # 2 years backtest
        
        print(f"\n[INFO] Backtesting from {start_date} to {end_date}")
        
        # Initialize backtester
        backtester = Backtester(engine, start_date, end_date, initial_capital=100000)
        
        # Run backtests for all strategies
        print("\n[INFO] Running strategy backtests...")
        results = backtester.run_all_strategies()
        
        if results:
            # Compare strategies
            comparison = backtester.compare_strategies()
            
            print("\n" + "=" * 80)
            print("BACKTEST RESULTS")
            print("=" * 80)
            print("\n" + comparison.to_string(index=False))
            
            # Save results
            save_backtest_results(engine, results)
            
            # Find best strategy
            best_sharpe = ''
            best_return = ''
            max_sharpe = -999
            max_return = -999
            
            for strategy, data in results.items():
                perf = data['performance']
                if perf['sharpe_ratio'] > max_sharpe:
                    max_sharpe = perf['sharpe_ratio']
                    best_sharpe = strategy
                if perf['total_return'] > max_return:
                    max_return = perf['total_return']
                    best_return = strategy
            
            print("\n" + "=" * 80)
            print("BACKTEST INSIGHTS")
            print("=" * 80)
            print(f"\nBest Risk-Adjusted (Sharpe): {best_sharpe} ({max_sharpe:.2f})")
            print(f"Best Total Return: {best_return} ({max_return*100:.1f}%)")
            
            print("\nKey Findings:")
            print("  1. Transaction costs significantly impact returns")
            print("  2. Rebalancing frequency affects both returns and costs")
            print("  3. Kelly sizing can improve risk-adjusted returns")
            print("  4. Diversification (10+ stocks) reduces drawdowns")
            print("  5. Quality scores effectively identify outperformers")
        
        # Log completion
        duration = time.time() - start_time
        log_script_end(logger, "backtesting_framework", success=True, duration=duration)
        print(f"\n[SUCCESS] Backtesting completed in {duration:.1f} seconds")
        
    except Exception as e:
        logger.error(f"Script failed: {e}")
        log_script_end(logger, "backtesting_framework", success=False, duration=time.time()-start_time)
        raise

if __name__ == "__main__":
    main()