# =====================================
# 3. Enhanced Risk Management Module
# =====================================
"""
#!/usr/bin/env python3
# File: risk_management.py
# Purpose: Risk management and portfolio optimization
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import warnings

warnings.filterwarnings('ignore')


class RiskManager:
    def __init__(self,
                 max_position_size=0.10,
                 max_sector_weight=0.30,
                 min_positions=15,
                 max_positions=30,
                 target_volatility=0.15):

        self.max_position_size = max_position_size
        self.max_sector_weight = max_sector_weight
        self.min_positions = min_positions
        self.max_positions = max_positions
        self.target_volatility = target_volatility

    def calculate_portfolio_metrics(self, weights, returns, cov_matrix):
        """Calculate portfolio return and risk"""
        portfolio_return = np.sum(returns * weights)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = portfolio_return / portfolio_std if portfolio_std > 0 else 0

        return {
            'return': portfolio_return,
            'volatility': portfolio_std,
            'sharpe': sharpe_ratio
        }

    def optimize_portfolio(self, expected_returns, cov_matrix, constraints=None):
        """Optimize portfolio weights using mean-variance optimization"""
        n_assets = len(expected_returns)

        # Initial guess (equal weight)
        init_weights = np.array([1 / n_assets] * n_assets)

        # Constraints
        if constraints is None:
            constraints = []

        # Weights sum to 1
        constraints.append({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

        # Bounds for each weight (0 to max_position_size)
        bounds = tuple((0, self.max_position_size) for _ in range(n_assets))

        # Objective function (negative Sharpe ratio for minimization)
        def neg_sharpe(weights):
            metrics = self.calculate_portfolio_metrics(weights, expected_returns, cov_matrix)
            return -metrics['sharpe']

        # Optimize
        result = minimize(
            neg_sharpe,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if result.success:
            return result.x
        else:
            # Fall back to equal weight
            return init_weights

    def calculate_var(self, portfolio_value, returns, confidence_level=0.95):
        """Calculate Value at Risk"""
        if len(returns) == 0:
            return 0

        mean = np.mean(returns)
        std = np.std(returns)

        # Parametric VaR
        var_pct = norm.ppf(1 - confidence_level, mean, std)
        var_amount = portfolio_value * abs(var_pct)

        # Historical VaR
        hist_var_pct = np.percentile(returns, (1 - confidence_level) * 100)
        hist_var_amount = portfolio_value * abs(hist_var_pct)

        return {
            'parametric_var': var_amount,
            'historical_var': hist_var_amount,
            'var_pct': abs(var_pct)
        }

    def check_portfolio_constraints(self, holdings_df):
        """Check if portfolio meets all risk constraints"""
        checks = {}

        # Position size check
        max_weight = holdings_df['weight'].max()
        checks['max_position_ok'] = max_weight <= self.max_position_size

        # Number of positions
        n_positions = len(holdings_df)
        checks['position_count_ok'] = self.min_positions <= n_positions <= self.max_positions

        # Sector concentration
        sector_weights = holdings_df.groupby('sector')['weight'].sum()
        max_sector = sector_weights.max() if len(sector_weights) > 0 else 0
        checks['sector_concentration_ok'] = max_sector <= self.max_sector_weight

        # Diversification (Herfindahl index)
        herfindahl = np.sum(holdings_df['weight'] ** 2)
        checks['diversification_score'] = 1 - herfindahl  # Higher is better
        checks['well_diversified'] = herfindahl < 0.1  # No position > 31.6%

        return checks

    def calculate_position_size(self,
                                base_weight,
                                volatility,
                                score,
                                momentum=None):
        """Calculate position size with volatility and score adjustments"""

        # Volatility adjustment (inverse volatility weighting)
        if volatility > 0:
            vol_scalar = min(2.0, self.target_volatility / volatility)
        else:
            vol_scalar = 1.0

        # Score adjustment (higher score = larger position)
        score_scalar = 0.5 + (score / 100)  # 0.5 to 1.5x

        # Momentum adjustment (optional)
        momentum_scalar = 1.0
        if momentum is not None:
            if momentum > 0.20:  # Strong positive momentum
                momentum_scalar = 1.2
            elif momentum < -0.10:  # Negative momentum
                momentum_scalar = 0.8

        # Calculate final weight
        adjusted_weight = base_weight * vol_scalar * score_scalar * momentum_scalar

        # Apply position limit
        final_weight = min(adjusted_weight, self.max_position_size)

        return final_weight

    def rebalance_portfolio(self, current_holdings, target_holdings, prices):
        """Calculate trades needed to rebalance portfolio"""
        trades = []

        # Merge current and target
        all_symbols = set(current_holdings.keys()) | set(target_holdings.keys())

        for symbol in all_symbols:
            current_weight = current_holdings.get(symbol, 0)
            target_weight = target_holdings.get(symbol, 0)

            weight_diff = target_weight - current_weight

            if abs(weight_diff) > 0.001:  # 0.1% threshold
                trade = {
                    'symbol': symbol,
                    'current_weight': current_weight,
                    'target_weight': target_weight,
                    'weight_change': weight_diff,
                    'action': 'BUY' if weight_diff > 0 else 'SELL',
                    'shares': None  # Calculate based on portfolio value and price
                }

                if symbol in prices:
                    trade['price'] = prices[symbol]

                trades.append(trade)

        return pd.DataFrame(trades)