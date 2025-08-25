# =====================================
# 3. ADVANCED FEATURE ENGINEERING
# =====================================
"""
#!/usr/bin/env python3
# File: feature_engineering.py
# Purpose: Advanced feature engineering for trading models
"""


class AdvancedFeatureEngineer:
    """Advanced feature engineering for financial data"""

    def __init__(self):
        self.feature_names = []
        self.scaler = None

    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical analysis features"""

        features = df.copy()

        # Price-based features
        for period in [5, 10, 20, 50, 200]:
            # Simple Moving Average
            features[f'sma_{period}'] = df['close'].rolling(period).mean()

            # Exponential Moving Average
            features[f'ema_{period}'] = df['close'].ewm(span=period).mean()

            # Price relative to MA
            features[f'price_to_sma_{period}'] = df['close'] / features[f'sma_{period}']

        # Bollinger Bands
        for period in [20, 50]:
            sma = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()

            features[f'bb_upper_{period}'] = sma + 2 * std
            features[f'bb_lower_{period}'] = sma - 2 * std
            features[f'bb_width_{period}'] = features[f'bb_upper_{period}'] - features[f'bb_lower_{period}']
            features[f'bb_position_{period}'] = (df['close'] - features[f'bb_lower_{period}']) / features[
                f'bb_width_{period}']

        # RSI
        for period in [14, 28]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()

            rs = gain / loss
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']

        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        features['atr_14'] = true_range.rolling(14).mean()

        # Stochastic Oscillator
        for period in [14, 28]:
            low_min = df['low'].rolling(period).min()
            high_max = df['high'].rolling(period).max()

            features[f'stoch_{period}'] = 100 * (df['close'] - low_min) / (high_max - low_min)

        return features

    def create_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create market microstructure features"""

        features = df.copy()

        # Volume features
        features['volume_sma_20'] = df['volume'].rolling(20).mean()
        features['volume_ratio'] = df['volume'] / features['volume_sma_20']
        features['dollar_volume'] = df['close'] * df['volume']

        # Spread and liquidity
        if 'bid' in df.columns and 'ask' in df.columns:
            features['spread'] = df['ask'] - df['bid']
            features['spread_pct'] = features['spread'] / df['close']
            features['mid_price'] = (df['ask'] + df['bid']) / 2

        # Price impact
        features['price_impact'] = df['close'].diff() / df['volume'].shift()

        # VWAP
        features['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        features['price_to_vwap'] = df['close'] / features['vwap']

        # Order flow imbalance (if tick data available)
        if 'trade_direction' in df.columns:
            features['buy_volume'] = df[df['trade_direction'] == 1]['volume'].rolling(20).sum()
            features['sell_volume'] = df[df['trade_direction'] == -1]['volume'].rolling(20).sum()
            features['order_imbalance'] = (features['buy_volume'] - features['sell_volume']) / (
                        features['buy_volume'] + features['sell_volume'])

        # Amihud illiquidity
        features['amihud'] = np.abs(df['close'].pct_change()) / df['dollar_volume']
        features['amihud_ma'] = features['amihud'].rolling(20).mean()

        return features

    def create_alternative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create alternative data features"""

        features = df.copy()

        # Sentiment features (if available)
        if 'sentiment_score' in df.columns:
            features['sentiment_ma_5'] = df['sentiment_score'].rolling(5).mean()
            features['sentiment_std_5'] = df['sentiment_score'].rolling(5).std()
            features['sentiment_zscore'] = (df['sentiment_score'] - features['sentiment_ma_5']) / features[
                'sentiment_std_5']

        # News volume (if available)
        if 'news_count' in df.columns:
            features['news_volume_ma'] = df['news_count'].rolling(5).mean()
            features['news_spike'] = df['news_count'] / features['news_volume_ma']

        # Social media metrics (if available)
        if 'twitter_mentions' in df.columns:
            features['social_momentum'] = df['twitter_mentions'].pct_change(5)
            features['social_acceleration'] = features['social_momentum'].diff()

        # Options data (if available)
        if 'put_call_ratio' in df.columns:
            features['pcr_ma'] = df['put_call_ratio'].rolling(5).mean()
            features['pcr_extreme'] = np.abs(df['put_call_ratio'] - features['pcr_ma']) / features['pcr_ma']

        if 'implied_volatility' in df.columns:
            features['iv_rank'] = df['implied_volatility'].rolling(252).rank(pct=True)
            features['iv_percentile'] = df['implied_volatility'].rolling(252).apply(
                lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100
            )

        return features

    def create_interaction_features(self, df: pd.DataFrame,
                                    interactions: List[Tuple[str, str]]) -> pd.DataFrame:
        """Create interaction features between variables"""

        features = df.copy()

        for col1, col2 in interactions:
            if col1 in df.columns and col2 in df.columns:
                # Multiplication
                features[f'{col1}_x_{col2}'] = df[col1] * df[col2]

                # Division (with safety check)
                denominator = df[col2].replace(0, np.nan)
                features[f'{col1}_div_{col2}'] = df[col1] / denominator

                # Difference
                features[f'{col1}_minus_{col2}'] = df[col1] - df[col2]

        return features

    def create_polynomial_features(self, df: pd.DataFrame,
                                   columns: List[str],
                                   degree: int = 2) -> pd.DataFrame:
        """Create polynomial features"""

        from sklearn.preprocessing import PolynomialFeatures

        features = df.copy()

        if len(columns) > 0:
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            poly_features = poly.fit_transform(df[columns])

            # Get feature names
            poly_names = poly.get_feature_names_out(columns)

            # Add to dataframe
            for i, name in enumerate(poly_names):
                if name not in columns:  # Skip original features
                    features[f'poly_{name}'] = poly_features[:, i]

        return features

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""

        features = df.copy()

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

            # Calendar features
            features['day_of_week'] = df['date'].dt.dayofweek
            features['day_of_month'] = df['date'].dt.day
            features['week_of_year'] = df['date'].dt.isocalendar().week
            features['month'] = df['date'].dt.month
            features['quarter'] = df['date'].dt.quarter

            # Trading day features
            features['is_monday'] = (features['day_of_week'] == 0).astype(int)
            features['is_friday'] = (features['day_of_week'] == 4).astype(int)
            features['is_month_start'] = df['date'].dt.is_month_start.astype(int)
            features['is_month_end'] = df['date'].dt.is_month_end.astype(int)
            features['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(int)
            features['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)

            # Days until events (if available)
            if 'earnings_date' in df.columns:
                features['days_to_earnings'] = (pd.to_datetime(df['earnings_date']) - df['date']).dt.days

            if 'dividend_date' in df.columns:
                features['days_to_dividend'] = (pd.to_datetime(df['dividend_date']) - df['date']).dt.days

        return features

    def select_features(self, X: pd.DataFrame, y: pd.Series,
                        method: str = 'mutual_info',
                        n_features: int = 50) -> List[str]:
        """Select best features using various methods"""

        from sklearn.feature_selection import (
            mutual_info_regression,
            SelectKBest,
            RFE,
            f_regression
        )
        from sklearn.ensemble import RandomForestRegressor

        # Remove any columns with all NaN
        X_clean = X.dropna(axis=1, how='all')

        # Fill remaining NaN with median
        X_filled = X_clean.fillna(X_clean.median())

        if method == 'mutual_info':
            # Mutual information
            mi_scores = mutual_info_regression(X_filled, y)
            mi_scores = pd.Series(mi_scores, index=X_filled.columns)
            selected = mi_scores.nlargest(n_features).index.tolist()

        elif method == 'f_score':
            # F-score
            selector = SelectKBest(f_regression, k=n_features)
            selector.fit(X_filled, y)
            selected = X_filled.columns[selector.get_support()].tolist()

        elif method == 'rfe':
            # Recursive Feature Elimination
            estimator = RandomForestRegressor(n_estimators=100, random_state=42)
            selector = RFE(estimator, n_features_to_select=n_features)
            selector.fit(X_filled, y)
            selected = X_filled.columns[selector.support_].tolist()

        elif method == 'importance':
            # Tree-based feature importance
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_filled, y)

            importance = pd.Series(rf.feature_importances_, index=X_filled.columns)
            selected = importance.nlargest(n_features).index.tolist()

        else:
            selected = X_filled.columns[:n_features].tolist()

        logger.info(f"Selected {len(selected)} features using {method}")
        return selected