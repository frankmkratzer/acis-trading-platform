# =====================================
# 4. Backtesting Framework
# =====================================
"""
#!/usr/bin/env python3
# File: backtest_engine.py
# Purpose: Backtesting framework for strategy validation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class BacktestConfig:
    start_date: datetime
    end_date: datetime
    initial_capital: float = 100000
    rebalance_frequency: str = 'monthly'  # daily, weekly, monthly, quarterly
    transaction_cost: float = 0.001  # 10 bps
    slippage: float = 0.0005  # 5 bps
    max_positions: int = 20
    min_position_size: float = 0.01
    max_position_size: float = 0.10


class BacktestEngine:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.portfolio_value = []
        self.holdings = {}
        self.trades = []
        self.cash = config.initial_capital

    def run(self, strategy, price_data):
        """Run backtest for a given strategy"""
        dates = pd.date_range(
            self.config.start_date,
            self.config.end_date,
            freq='B'  # Business days
        )

        rebalance_dates = self._get_rebalance_dates(dates)

        for date in dates:
            # Check if rebalance day
            if date in rebalance_dates:
                signals = strategy.generate_signals(date)
                self._execute_rebalance(signals, price_data, date)

            # Update portfolio value
            portfolio_val = self._calculate_portfolio_value(price_data, date)
            self.portfolio_value.append({
                'date': date,
                'value': portfolio_val,
                'cash': self.cash,
                'n_holdings': len(self.holdings)
            })

        return self._calculate_metrics()

    def _get_rebalance_dates(self, dates):
        """Get rebalance dates based on frequency"""
        if self.config.rebalance_frequency == 'daily':
            return dates
        elif self.config.rebalance_frequency == 'weekly':
            return dates[dates.weekday == 0]  # Mondays
        elif self.config.rebalance_frequency == 'monthly':
            return dates[dates.is_month_start]
        elif self.config.rebalance_frequency == 'quarterly':
            return dates[(dates.month % 3 == 1) & dates.is_month_start]
        else:
            return dates[dates.is_month_start]

    def _execute_rebalance(self, signals, price_data, date):
        """Execute portfolio rebalance"""
        # Calculate target weights
        target_weights = self._calculate_target_weights(signals)

        # Get current prices
        current_prices = price_data[price_data['date'] == date].set_index('symbol')['close'].to_dict()

        # Calculate current portfolio value
        current_value = self.cash
        for symbol, shares in self.holdings.items():
            if symbol in current_prices:
                current_value += shares * current_prices[symbol]

        # Sell positions not in target or reduced weight
        for symbol in list(self.holdings.keys()):
            if symbol not in target_weights or target_weights[symbol] == 0:
                # Sell entire position
                if symbol in current_prices:
                    proceeds = self.holdings[symbol] * current_prices[symbol]
                    proceeds *= (1 - self.config.transaction_cost - self.config.slippage)
                    self.cash += proceeds

                    self.trades.append({
                        'date': date,
                        'symbol': symbol,
                        'action': 'SELL',
                        'shares': self.holdings[symbol],
                        'price': current_prices[symbol],
                        'value': proceeds
                    })

                del self.holdings[symbol]

        # Buy new positions or increase weights
        for symbol, target_weight in target_weights.items():
            target_value = current_value * target_weight
            current_holding_value = 0

            if symbol in self.holdings and symbol in current_prices:
                current_holding_value = self.holdings[symbol] * current_prices[symbol]

            trade_value = target_value - current_holding_value

            if abs(trade_value) > current_value * 0.001:  # Min trade size
                if symbol in current_prices:
                    shares_to_trade = trade_value / current_prices[symbol]

                    if trade_value > 0:  # Buy
                        cost = abs(trade_value) * (1 + self.config.transaction_cost + self.config.slippage)
                        if cost <= self.cash:
                            self.cash -= cost
                            self.holdings[symbol] = self.holdings.get(symbol, 0) + shares_to_trade

                            self.trades.append({
                                'date': date,
                                'symbol': symbol,
                                'action': 'BUY',
                                'shares': shares_to_trade,
                                'price': current_prices[symbol],
                                'value': cost
                            })

    def _calculate_target_weights(self, signals):
        """Calculate target portfolio weights from signals"""
        # Simple equal weight for top N stocks
        top_stocks = signals.nlargest(self.config.max_positions, 'score')

        weights = {}
        for _, row in top_stocks.iterrows():
            weights[row['symbol']] = 1.0 / len(top_stocks)

        return weights

    def _calculate_portfolio_value(self, price_data, date):
        """Calculate total portfolio value"""
        value = self.cash

        current_prices = price_data[price_data['date'] == date].set_index('symbol')['close'].to_dict()

        for symbol, shares in self.holdings.items():
            if symbol in current_prices:
                value += shares * current_prices[symbol]

        return value

    def _calculate_metrics(self):
        """Calculate backtest performance metrics"""
        df = pd.DataFrame(self.portfolio_value)
        df['returns'] = df['value'].pct_change()

        # Calculate metrics
        total_return = (df['value'].iloc[-1] / df['value'].iloc[0]) - 1

        # Annualized return
        n_years = (df['date'].iloc[-1] - df['date'].iloc[0]).days / 365.25
        annual_return = (1 + total_return) ** (1 / n_years) - 1

        # Volatility
        annual_vol = df['returns'].std() * np.sqrt(252)

        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0

        # Maximum drawdown
        rolling_max = df['value'].expanding().max()
        drawdown = (df['value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Win rate
        winning_days = (df['returns'] > 0).sum()
        total_days = len(df['returns'].dropna())
        win_rate = winning_days / total_days if total_days > 0 else 0

        # Calmar ratio
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'calmar_ratio': calmar,
            'total_trades': len(self.trades),
            'final_value': df['value'].iloc[-1],
            'portfolio_history': df
        }
