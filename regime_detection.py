# =====================================
# 4. REGIME DETECTION AND ADAPTATION
# =====================================
"""
#!/usr/bin/env python3
# File: regime_detection.py
# Purpose: Market regime detection and strategy adaptation
"""


class MarketRegimeDetector:
    """Detect and adapt to different market regimes"""

    def __init__(self):
        self.current_regime = None
        self.regime_history = []
        self.regime_models = {}

    def detect_regime_hmm(self, returns: pd.Series, n_regimes: int = 3) -> np.array:
        """Detect regimes using Hidden Markov Model"""

        from hmmlearn import hmm

        # Prepare data
        X = returns.values.reshape(-1, 1)

        # Fit HMM
        model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type="diag",
            n_iter=100
        )

        model.fit(X)

        # Predict regimes
        regimes = model.predict(X)

        # Sort regimes by volatility
        regime_vol = {}
        for i in range(n_regimes):
            mask = regimes == i
            regime_vol[i] = returns[mask].std()

        # Relabel: 0=low vol, 1=medium vol, 2=high vol
        sorted_regimes = sorted(regime_vol.items(), key=lambda x: x[1])
        regime_map = {old: new for new, (old, _) in enumerate(sorted_regimes)}

        regimes = np.array([regime_map[r] for r in regimes])

        return regimes

    def detect_regime_threshold(self, indicators: pd.DataFrame) -> pd.Series:
        """Detect regimes using threshold-based rules"""

        regimes = pd.Series(index=indicators.index, dtype=int)

        # Example: VIX-based regime
        if 'vix' in indicators.columns:
            vix = indicators['vix']

            # Low volatility regime
            regimes[vix < 15] = 0

            # Normal regime
            regimes[(vix >= 15) & (vix < 25)] = 1

            # High volatility regime
            regimes[vix >= 25] = 2

        # Example: Trend-based regime
        if 'sma_50' in indicators.columns and 'sma_200' in indicators.columns:
            golden_cross = indicators['sma_50'] > indicators['sma_200']

            # Adjust regime based on trend
            regimes[golden_cross] = np.minimum(regimes[golden_cross], 1)
            regimes[~golden_cross] = np.maximum(regimes[~golden_cross], 1)

        return regimes

    def detect_regime_clustering(self, features: pd.DataFrame, n_regimes: int = 4) -> np.array:
        """Detect regimes using clustering"""

        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)

        # Cluster
        kmeans = KMeans(n_clusters=n_regimes, random_state=42)
        regimes = kmeans.fit_predict(X_scaled)

        return regimes

    def adapt_strategy_to_regime(self, regime: int) -> Dict:
        """Adapt strategy parameters based on regime"""

        # Regime-specific parameters
        regime_params = {
            0: {  # Low volatility
                'position_size_multiplier': 1.5,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.05,
                'holding_period': 20,
                'signal_threshold': 0.6,
                'use_leverage': True,
                'max_leverage': 1.5
            },
            1: {  # Normal
                'position_size_multiplier': 1.0,
                'stop_loss_pct': 0.05,
                'take_profit_pct': 0.08,
                'holding_period': 15,
                'signal_threshold': 0.65,
                'use_leverage': False,
                'max_leverage': 1.0
            },
            2: {  # High volatility
                'position_size_multiplier': 0.5,
                'stop_loss_pct': 0.08,
                'take_profit_pct': 0.12,
                'holding_period': 10,
                'signal_threshold': 0.75,
                'use_leverage': False,
                'max_leverage': 1.0
            },
            3: {  # Crisis
                'position_size_multiplier': 0.2,
                'stop_loss_pct': 0.10,
                'take_profit_pct': 0.15,
                'holding_period': 5,
                'signal_threshold': 0.85,
                'use_leverage': False,
                'max_leverage': 1.0
            }
        }

        return regime_params.get(regime, regime_params[1])

    def calculate_regime_transition_matrix(self, regimes: np.array) -> np.array:
        """Calculate regime transition probabilities"""

        n_regimes = len(np.unique(regimes))
        transition_matrix = np.zeros((n_regimes, n_regimes))

        for i in range(len(regimes) - 1):
            current = regimes[i]
            next_regime = regimes[i + 1]
            transition_matrix[current, next_regime] += 1

        # Normalize rows to get probabilities
        for i in range(n_regimes):
            row_sum = transition_matrix[i].sum()
            if row_sum > 0:
                transition_matrix[i] /= row_sum

        return transition_matrix

    def predict_next_regime(self, current_regime: int,
                            transition_matrix: np.array) -> Tuple[int, float]:
        """Predict next regime based on transition probabilities"""

        probabilities = transition_matrix[current_regime]
        next_regime = np.argmax(probabilities)
        confidence = probabilities[next_regime]

        return next_regime, confidence