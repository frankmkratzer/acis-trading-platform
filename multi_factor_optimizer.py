# =====================================
# 3. MULTI-FACTOR PORTFOLIO OPTIMIZER
# =====================================
"""
#!/usr/bin/env python3
# File: multi_factor_optimizer.py
# Purpose: Advanced multi-factor portfolio optimization
"""
import pandas as pd
import numpy as np
from typing import Dict
from scipy.optimize import minimize

class MultiFactorOptimizer:
    """Multi-factor portfolio optimization with factor timing"""

    def __init__(self):
        self.factors = ['value', 'growth', 'momentum', 'quality', 'low_vol']
        self.lookback_days = 252
        self.rebalance_frequency = 'monthly'

    def calculate_factor_scores(self, universe: pd.DataFrame) -> pd.DataFrame:
        """Calculate all factor scores"""

        scores = pd.DataFrame(index=universe.index)

        # Value score
        scores['value'] = self.calculate_value_score(universe)

        # Growth score
        scores['growth'] = self.calculate_growth_score(universe)

        # Momentum score
        scores['momentum'] = self.calculate_momentum_score(universe)

        # Quality score
        scores['quality'] = self.calculate_quality_score(universe)

        # Low volatility score
        scores['low_vol'] = self.calculate_low_vol_score(universe)

        return scores

    def calculate_value_score(self, df: pd.DataFrame) -> pd.Series:
        """Enhanced value score"""

        # Multiple value metrics
        earnings_yield = df['earnings'] / df['market_cap']
        fcf_yield = df['free_cash_flow'] / df['market_cap']
        book_to_market = df['book_value'] / df['market_cap']
        ebitda_ev = df['ebitda'] / df['enterprise_value']

        # Z-score normalization
        scores = pd.DataFrame({
            'earnings_yield': self._zscore(earnings_yield),
            'fcf_yield': self._zscore(fcf_yield),
            'book_to_market': self._zscore(book_to_market),
            'ebitda_ev': self._zscore(ebitda_ev)
        })

        # Weighted average
        weights = [0.3, 0.3, 0.2, 0.2]
        return (scores * weights).sum(axis=1)

    def calculate_quality_score(self, df: pd.DataFrame) -> pd.Series:
        """Quality score based on profitability and stability"""

        # Profitability metrics
        roe = df['net_income'] / df['equity']
        roa = df['net_income'] / df['total_assets']
        gross_margin = df['gross_profit'] / df['revenue']

        # Stability metrics
        earnings_stability = 1 / df['earnings_volatility']
        debt_to_equity = df['total_debt'] / df['equity']

        # Combine metrics
        scores = pd.DataFrame({
            'roe': self._zscore(roe),
            'roa': self._zscore(roa),
            'gross_margin': self._zscore(gross_margin),
            'earnings_stability': self._zscore(earnings_stability),
            'debt_to_equity': self._zscore(-debt_to_equity)  # Lower is better
        })

        return scores.mean(axis=1)

    def calculate_factor_timing_signals(self,
                                        factor_returns: pd.DataFrame) -> pd.Series:
        """Generate factor timing signals"""

        signals = {}

        for factor in self.factors:
            if factor not in factor_returns.columns:
                continue

            # Calculate factor momentum
            factor_mom = factor_returns[factor].rolling(60).mean()

            # Calculate factor value spread
            factor_spread = self._calculate_factor_spread(factor)

            # Calculate factor volatility
            factor_vol = factor_returns[factor].rolling(20).std()

            # Combine signals
            signal = (
                    0.4 * (factor_mom > 0) +  # Positive momentum
                    0.3 * (factor_spread > factor_spread.median()) +  # Wide value spread
                    0.3 * (factor_vol < factor_vol.median())  # Low volatility
            )

            signals[factor] = signal.iloc[-1] if not signal.empty else 0.5

        return pd.Series(signals)

    def optimize_factor_portfolio(self,
                                  factor_scores: pd.DataFrame,
                                  factor_returns: pd.DataFrame,
                                  constraints: Dict = None) -> pd.Series:
        """Optimize multi-factor portfolio"""

        # Calculate factor exposures
        factor_exposures = self._calculate_factor_exposures(factor_scores)

        # Calculate factor expected returns
        factor_expected_returns = factor_returns.mean()

        # Calculate factor covariance
        factor_cov = factor_returns.cov()

        # Get factor timing signals
        timing_signals = self.calculate_factor_timing_signals(factor_returns)

        # Adjust expected returns by timing signals
        adjusted_returns = factor_expected_returns * (0.5 + timing_signals)

        # Optimize factor weights
        n_factors = len(self.factors)

        def objective(weights):
            # Expected return
            expected_return = weights @ adjusted_returns

            # Risk (variance)
            risk = weights @ factor_cov @ weights

            # Risk-adjusted return (Sharpe-like)
            return -expected_return / np.sqrt(risk) if risk > 0 else 0

        # Constraints
        cons = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Sum to 1
        ]

        if constraints:
            if 'max_factor_weight' in constraints:
                cons.append({
                    'type': 'ineq',
                    'fun': lambda x: constraints['max_factor_weight'] - np.max(x)
                })

        # Bounds
        bounds = [(0, 0.4) for _ in range(n_factors)]  # Max 40% per factor

        # Initial guess
        x0 = np.ones(n_factors) / n_factors

        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)

        if result.success:
            factor_weights = pd.Series(result.x, index=self.factors[:n_factors])
        else:
            factor_weights = pd.Series(x0, index=self.factors[:n_factors])

        # Convert factor weights to stock weights
        stock_weights = self._factor_weights_to_stock_weights(
            factor_weights,
            factor_scores,
            factor_exposures
        )

        return stock_weights

    def _calculate_factor_exposures(self, factor_scores: pd.DataFrame) -> pd.DataFrame:
        """Calculate stock exposures to each factor"""

        exposures = pd.DataFrame(index=factor_scores.index)

        for factor in factor_scores.columns:
            # Rank and normalize to [0, 1]
            exposures[factor] = factor_scores[factor].rank(pct=True)

        return exposures

    def _factor_weights_to_stock_weights(self,
                                         factor_weights: pd.Series,
                                         factor_scores: pd.DataFrame,
                                         factor_exposures: pd.DataFrame) -> pd.Series:
        """Convert factor weights to stock weights"""

        # Calculate composite score for each stock
        composite_scores = pd.Series(0, index=factor_scores.index)

        for factor, weight in factor_weights.items():
            if factor in factor_scores.columns:
                composite_scores += weight * factor_scores[factor]

        # Rank stocks by composite score
        ranks = composite_scores.rank(ascending=False)

        # Select top stocks (e.g., top 50)
        n_stocks = min(50, len(ranks))
        selected = ranks <= n_stocks

        # Calculate weights (could be equal weight or score-weighted)
        stock_weights = pd.Series(0, index=factor_scores.index)

        # Score-weighted within selected stocks
        selected_scores = composite_scores[selected]
        stock_weights[selected] = selected_scores / selected_scores.sum()

        return stock_weights

    def _zscore(self, series: pd.Series) -> pd.Series:
        """Calculate z-scores with outlier handling"""

        # Winsorize at 1st and 99th percentiles
        lower = series.quantile(0.01)
        upper = series.quantile(0.99)
        clipped = series.clip(lower, upper)

        # Calculate z-scores
        mean = clipped.mean()
        std = clipped.std()

        if std > 0:
            return (clipped - mean) / std
        else:
            return pd.Series(0, index=series.index)

    def _calculate_factor_spread(self, factor: str) -> pd.Series:
        """Calculate value spread for a factor"""

        # This would typically involve comparing top vs bottom quintile valuations
        # Placeholder implementation
        return pd.Series(np.random.random(252))