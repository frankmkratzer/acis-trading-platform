# =====================================
# 2. ADVANCED RISK MANAGEMENT SYSTEM
# =====================================
"""
#!/usr/bin/env python3
# File: advanced_risk_manager.py
# Purpose: Sophisticated risk management with regime detection and dynamic sizing
"""

import pandas as pd
import numpy as np
from typing import Dict
from scipy import stats
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from sklearn.mixture import GaussianMixture
import warnings
from typing import Dict

warnings.filterwarnings('ignore')


class AdvancedRiskManager:
    """Advanced risk management with multiple techniques"""

    def __init__(self):
        self.vol_target = 0.15  # 15% annual volatility target
        self.max_leverage = 1.5
        self.var_confidence = 0.95
        self.cvar_confidence = 0.95

    def calculate_risk_metrics(self, returns: pd.DataFrame) -> Dict:
        """Calculate comprehensive risk metrics"""

        metrics = {}

        # Basic statistics
        metrics['mean_return'] = returns.mean()
        metrics['volatility'] = returns.std()
        metrics['skewness'] = returns.skew()
        metrics['kurtosis'] = returns.kurtosis()

        # Value at Risk (parametric and historical)
        metrics['var_parametric'] = self.calculate_parametric_var(returns)
        metrics['var_historical'] = self.calculate_historical_var(returns)
        metrics['cvar'] = self.calculate_cvar(returns)

        # Maximum Drawdown
        metrics['max_drawdown'] = self.calculate_max_drawdown(returns)
        metrics['max_drawdown_duration'] = self.calculate_drawdown_duration(returns)

        # Risk-adjusted returns
        metrics['sharpe_ratio'] = self.calculate_sharpe_ratio(returns)
        metrics['sortino_ratio'] = self.calculate_sortino_ratio(returns)
        metrics['calmar_ratio'] = self.calculate_calmar_ratio(returns)

        # Tail risk
        metrics['tail_ratio'] = self.calculate_tail_ratio(returns)
        metrics['best_day'] = returns.max()
        metrics['worst_day'] = returns.min()

        return metrics

    def calculate_parametric_var(self, returns: pd.Series, confidence: float = None) -> float:
        """Calculate parametric VaR"""
        if confidence is None:
            confidence = self.var_confidence

        mean = returns.mean()
        std = returns.std()
        var = stats.norm.ppf(1 - confidence, mean, std)
        return abs(var)

    def calculate_historical_var(self, returns: pd.Series, confidence: float = None) -> float:
        """Calculate historical VaR"""
        if confidence is None:
            confidence = self.var_confidence

        var = returns.quantile(1 - confidence)
        return abs(var)

    def calculate_cvar(self, returns: pd.Series, confidence: float = None) -> float:
        """Calculate Conditional VaR (Expected Shortfall)"""
        if confidence is None:
            confidence = self.cvar_confidence

        var = self.calculate_historical_var(returns, confidence)
        cvar = returns[returns <= -var].mean()
        return abs(cvar) if not pd.isna(cvar) else var

    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())

    def calculate_drawdown_duration(self, returns: pd.Series) -> int:
        """Calculate maximum drawdown duration in days"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        is_drawdown = drawdown < 0
        drawdown_groups = is_drawdown.ne(is_drawdown.shift()).cumsum()
        drawdown_lengths = is_drawdown.groupby(drawdown_groups).sum()

        return int(drawdown_lengths.max()) if len(drawdown_lengths) > 0 else 0

    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free: float = 0) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free / 252  # Daily risk-free rate
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    def calculate_sortino_ratio(self, returns: pd.Series, risk_free: float = 0) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        excess_returns = returns - risk_free / 252
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_returns.std()

        if downside_std > 0:
            return np.sqrt(252) * excess_returns.mean() / downside_std
        return 0

    def calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio"""
        annual_return = (1 + returns.mean()) ** 252 - 1
        max_dd = self.calculate_max_drawdown(returns)

        if max_dd > 0:
            return annual_return / max_dd
        return 0

    def calculate_tail_ratio(self, returns: pd.Series, percentile: float = 0.05) -> float:
        """Calculate tail ratio (right tail / left tail)"""
        right_tail = returns.quantile(1 - percentile)
        left_tail = abs(returns.quantile(percentile))

        if left_tail > 0:
            return right_tail / left_tail
        return 0

    def detect_regime(self, returns: pd.DataFrame, n_regimes: int = 3) -> pd.Series:
        """Detect market regimes using Gaussian Mixture Models"""

        # Prepare features
        features = pd.DataFrame({
            'returns': returns.mean(axis=1),
            'volatility': returns.rolling(20).std().mean(axis=1),
            'correlation': returns.rolling(20).corr().mean().mean()
        }).dropna()

        # Fit GMM
        gmm = GaussianMixture(n_components=n_regimes, covariance_type='full', random_state=42)
        regimes = gmm.fit_predict(features)

        # Label regimes by volatility
        regime_vol = {}
        for i in range(n_regimes):
            mask = regimes == i
            regime_vol[i] = features.loc[mask, 'volatility'].mean()

        # Sort by volatility and relabel
        sorted_regimes = sorted(regime_vol.items(), key=lambda x: x[1])
        regime_map = {old: new for new, (old, _) in enumerate(sorted_regimes)}

        return pd.Series(
            [regime_map[r] for r in regimes],
            index=features.index,
            name='regime'
        )

    def calculate_dynamic_position_size(self,
                                        signal_strength: float,
                                        volatility: float,
                                        regime: int,
                                        correlation: float) -> float:
        """Calculate dynamic position size based on multiple factors"""

        # Base size from volatility targeting
        base_size = self.vol_target / (volatility * np.sqrt(252))

        # Adjust for signal strength (0 to 1)
        signal_adjusted = base_size * (0.5 + 0.5 * signal_strength)

        # Adjust for regime
        regime_multipliers = {
            0: 1.2,  # Low volatility regime - increase size
            1: 1.0,  # Normal regime
            2: 0.6  # High volatility regime - reduce size
        }
        regime_adjusted = signal_adjusted * regime_multipliers.get(regime, 1.0)

        # Adjust for correlation (reduce size when correlation is high)
        correlation_adjusted = regime_adjusted * (1 - 0.3 * abs(correlation))

        # Apply leverage limits
        final_size = min(correlation_adjusted, self.max_leverage)

        return final_size

    def optimize_portfolio_cvar(self,
                                expected_returns: np.array,
                                scenarios: np.array,
                                confidence: float = 0.95) -> np.array:
        """Optimize portfolio using CVaR (Conditional Value at Risk)"""

        n_assets = len(expected_returns)
        n_scenarios = len(scenarios)

        # Initial guess
        x0 = np.ones(n_assets + n_scenarios + 1) / n_assets
        x0[n_assets:] = 0

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x[:n_assets]) - 1},  # Weights sum to 1
            {'type': 'ineq', 'fun': lambda x: x[:n_assets]}  # Non-negative weights
        ]

        # Bounds
        bounds = [(0, 1) for _ in range(n_assets)]
        bounds += [(0, None) for _ in range(n_scenarios)]
        bounds += [(None, None)]  # VaR can be negative

        # Objective: Minimize CVaR
        def objective(x):
            weights = x[:n_assets]
            z = x[n_assets:n_assets + n_scenarios]
            var = x[-1]

            portfolio_returns = scenarios @ weights
            cvar = var + np.mean(z) / (1 - confidence)

            # Add constraint penalties
            for i in range(n_scenarios):
                if -portfolio_returns[i] - var > z[i]:
                    cvar += 1000  # Penalty for constraint violation

            return cvar

        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

        if result.success:
            return result.x[:n_assets]
        else:
            # Fallback to equal weight
            return np.ones(n_assets) / n_assets

    def calculate_risk_parity_weights(self, cov_matrix: np.array) -> np.array:
        """Calculate risk parity weights"""

        n_assets = len(cov_matrix)

        # Initial guess (equal weight)
        x0 = np.ones(n_assets) / n_assets

        # Objective: Equal risk contribution
        def objective(weights):
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            marginal_contrib = cov_matrix @ weights / portfolio_vol
            contrib = weights * marginal_contrib

            # Minimize variance of risk contributions
            return np.var(contrib)

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]

        # Bounds
        bounds = [(0.01, 0.5) for _ in range(n_assets)]  # Min 1%, Max 50% per asset

        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

        if result.success:
            return result.x
        else:
            return x0

    def calculate_kelly_fraction(self,
                                 win_rate: float,
                                 avg_win: float,
                                 avg_loss: float,
                                 kelly_fraction: float = 0.25) -> float:
        """Calculate Kelly criterion for position sizing"""

        if avg_loss == 0:
            return 0

        # Kelly formula: f = (p * b - q) / b
        # where p = win probability, q = loss probability, b = win/loss ratio
        b = abs(avg_win / avg_loss)
        q = 1 - win_rate

        kelly = (win_rate * b - q) / b

        # Apply Kelly fraction (typically 25% of full Kelly)
        conservative_kelly = kelly * kelly_fraction

        # Cap at reasonable levels
        return np.clip(conservative_kelly, 0, 0.25)