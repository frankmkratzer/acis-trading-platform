#!/usr/bin/env python3
"""
Complete Algorithmic Trading System
Missing Scripts and Enhanced Components
"""

# =====================================
# 1. train_ai_value_model.py
# =====================================
"""
#!/usr/bin/env python3
# File: train_ai_value_model.py
# Purpose: Train ML model for value stock selection
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import joblib
import logging
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
engine = create_engine(os.getenv("POSTGRES_URL"))

logging.basicConfig(
    filename="train_value_model.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class ValueModelTrainer:
    def __init__(self, lookback_days=1260, min_samples=100):
        self.lookback_days = lookback_days  # 5 years
        self.min_samples = min_samples
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None

    def load_training_data(self):
        """Load features and labels for training"""
        query = text("""
                     WITH features AS (SELECT f.symbol,
                                              f.fiscal_date                                               as as_of_date,
                                              -- Value metrics
                                              f.netincome::float / NULLIF(m.market_cap, 0) as earnings_yield, f.free_cf::float / NULLIF(m.market_cap, 0) as fcf_yield, f.totalrevenue::float / NULLIF(m.market_cap, 0) as sales_yield, f.totalshareholderequity::float / NULLIF(m.market_cap, 0) as book_to_market,

                    -- Quality metrics f.netincome::float / NULLIF(f.totalshareholderequity, 0) as roe, f.netincome::float / NULLIF(f.totalassets, 0) as roa, (f.totalassets - f.totalliabilities)::float / NULLIF(f.totalassets, 0) as equity_ratio, f.operatingcashflow::float / NULLIF(f.netincome, 0) as cash_conversion,

                    -- Growth metrics (year-over-year) (f.totalrevenue - lag(f.totalrevenue) OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date))::float 
                        / NULLIF(lag(f.totalrevenue) OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date), 0) as revenue_growth, (f.netincome - lag(f.netincome) OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date))::float 
                        / NULLIF(ABS(lag(f.netincome) OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date)), 0) as earnings_growth,

                    -- Market metrics m.pe_ratio,
                                              m.dividend_yield,
                                              LOG(m.market_cap + 1)                                       as log_market_cap,

                                              -- Sector dummies (simplified)
                                              CASE WHEN m.sector = 'Technology' THEN 1 ELSE 0 END         as is_tech,
                                              CASE WHEN m.sector = 'Financial Services' THEN 1 ELSE 0 END as is_financial,
                                              CASE WHEN m.sector = 'Healthcare' THEN 1 ELSE 0 END         as is_healthcare

                                       FROM fundamentals_annual f
                                                JOIN mv_symbol_with_metadata m ON f.symbol = m.symbol
                                       WHERE f.fiscal_date >= CURRENT_DATE -
                         INTERVAL ':lookback days'
                         AND m.market_cap
                        > 1000000000 -- $1B minimum
                         )
                        , labels AS (
                     SELECT
                         symbol, as_of_date, return_12m as forward_return
                     FROM forward_returns
                     WHERE return_12m IS NOT NULL
                         )
                     SELECT f.*,
                            l.forward_return as label
                     FROM features f
                              JOIN labels l ON f.symbol = l.symbol
                         AND f.as_of_date = l.as_of_date
                     WHERE f.earnings_yield IS NOT NULL
                       AND f.fcf_yield IS NOT NULL
                       AND ABS(f.earnings_yield) < 1 -- Remove outliers
                       AND ABS(f.fcf_yield) < 1
                     """)

        df = pd.read_sql(query, engine, params={"lookback": self.lookback_days})

        # Handle infinities and missing values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(thresh=len(df.columns) - 3)  # Allow some NaNs

        # Fill remaining NaNs with median
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].fillna(df[col].median())

        logger.info(f"Loaded {len(df)} samples for training")
        return df

    def prepare_features(self, df):
        """Prepare feature matrix and labels"""
        feature_cols = [
            'earnings_yield', 'fcf_yield', 'sales_yield', 'book_to_market',
            'roe', 'roa', 'equity_ratio', 'cash_conversion',
            'revenue_growth', 'earnings_growth',
            'pe_ratio', 'dividend_yield', 'log_market_cap',
            'is_tech', 'is_financial', 'is_healthcare'
        ]

        # Winsorize extreme values
        for col in feature_cols:
            if col in df.columns:
                lower = df[col].quantile(0.01)
                upper = df[col].quantile(0.99)
                df[col] = df[col].clip(lower, upper)

        X = df[feature_cols].values
        y = df['label'].values

        # Create binary classification target (outperform median)
        y_binary = (y > np.median(y)).astype(int)

        return X, y, y_binary, feature_cols

    def train_model(self, X, y, y_binary):
        """Train ensemble model"""
        # Use TimeSeriesSplit for proper validation
        tscv = TimeSeriesSplit(n_splits=5)

        # Train gradient boosting for regression
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42
        )

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Cross-validation
        scores = []
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            self.model.fit(X_train, y_train)
            score = self.model.score(X_val, y_val)
            scores.append(score)

        logger.info(f"Cross-validation R² scores: {scores}")
        logger.info(f"Mean CV R²: {np.mean(scores):.4f}")

        # Final training on all data
        self.model.fit(X_scaled, y)

        return self.model

    def calculate_feature_importance(self, feature_names):
        """Calculate and log feature importance"""
        if hasattr(self.model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

            logger.info("Feature Importance:")
            for _, row in importance.head(10).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")

            self.feature_importance = importance
            return importance

    def save_model(self, version=None):
        """Save trained model and scaler"""
        if version is None:
            version = datetime.now().strftime("v%Y%m%d_%H%M%S")

        model_path = f"models/value_model_{version}.pkl"
        scaler_path = f"models/value_scaler_{version}.pkl"

        os.makedirs("models", exist_ok=True)

        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

        # Save feature importance
        if self.feature_importance is not None:
            self.feature_importance.to_csv(f"models/value_features_{version}.csv", index=False)

        # Log model metadata to database
        with engine.begin() as conn:
            conn.execute(text("""
                              INSERT INTO ai_model_run_log (run_id, model_type, version, as_of_date,
                                                            features, hyperparameters, notes, created_at)
                              VALUES (:run_id, :model_type, :version, :as_of_date,
                                      :features, :hyperparameters, :notes, NOW())
                              """), {
                             "run_id": f"value_{version}",
                             "model_type": "value",
                             "version": version,
                             "as_of_date": datetime.now().date(),
                             "features": self.feature_importance.to_json() if self.feature_importance is not None else None,
                             "hyperparameters": {
                                 "n_estimators": 100,
                                 "learning_rate": 0.1,
                                 "max_depth": 5
                             },
                             "notes": "Gradient Boosting model for value stock selection"
                         })

        logger.info(f"Model saved: {model_path}")
        return version


def main():
    trainer = ValueModelTrainer()

    # Load data
    df = trainer.load_training_data()

    if len(df) < trainer.min_samples:
        logger.error(f"Insufficient data: {len(df)} samples")
        return

    # Prepare features
    X, y, y_binary, feature_names = trainer.prepare_features(df)

    # Train model
    model = trainer.train_model(X, y, y_binary)

    # Calculate feature importance
    trainer.calculate_feature_importance(feature_names)

    # Save model
    version = trainer.save_model()

    print(f"✅ Value model trained and saved (version: {version})")


if __name__ == "__main__":
    main()