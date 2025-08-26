#!/usr/bin/env python3
"""
AI Dividend Model Training
Train ML model for dividend stock selection using comprehensive dividend metrics
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import logging
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
engine = create_engine(os.getenv("POSTGRES_URL"))

logging.basicConfig(
    filename="train_dividend_model.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

class DividendModelTrainer:
    def __init__(self, lookback_days=1800, min_samples=100):  # 5 years
        self.lookback_days = lookback_days
        self.min_samples = min_samples
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None

    def load_training_data(self):
        """Load dividend-focused features and forward returns"""
        query = text("""
        WITH dividend_features AS (
            SELECT 
                f.symbol,
                f.fiscal_date as as_of_date,
                
                -- Dividend Metrics
                f.dividendpayout::float / NULLIF(f.netincome, 0) as payout_ratio,
                CASE 
                    WHEN f.dividendpayout > 0 AND m.current_price > 0 
                    THEN (f.dividendpayout::float * 4) / m.current_price  -- Annualized yield
                    ELSE 0 
                END as dividend_yield,
                f.free_cf::float / NULLIF(f.dividendpayout, 0) as fcf_dividend_coverage,
                f.operatingcashflow::float / NULLIF(f.dividendpayout, 0) as ocf_dividend_coverage,
                
                -- Dividend Growth (year-over-year)
                (f.dividendpayout - LAG(f.dividendpayout) OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date))::float
                    / NULLIF(LAG(f.dividendpayout) OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date), 0) as dividend_growth_1yr,
                
                -- Quality Metrics for Dividend Sustainability
                f.netincome::float / NULLIF(f.totalshareholderequity, 0) as roe,
                f.netincome::float / NULLIF(f.totalassets, 0) as roa,
                f.netincome::float / NULLIF(f.totalrevenue, 0) as net_margin,
                f.grossprofit::float / NULLIF(f.totalrevenue, 0) as gross_margin,
                f.operatingcashflow::float / NULLIF(f.totalrevenue, 0) as ocf_margin,
                
                -- Financial Stability 
                f.totalliabilities::float / NULLIF(f.totalshareholderequity, 0) as debt_to_equity,
                f.totalshareholderequity::float / NULLIF(f.totalassets, 0) as equity_ratio,
                (f.totalassets - f.totalliabilities)::float / NULLIF(f.totalassets, 0) as solvency_ratio,
                
                -- Business Quality
                f.free_cf::float / NULLIF(f.totalrevenue, 0) as fcf_margin,
                f.totalrevenue::float / NULLIF(f.totalassets, 0) as asset_turnover,
                
                -- Growth Sustainability
                (f.totalrevenue - LAG(f.totalrevenue) OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date))::float
                    / NULLIF(LAG(f.totalrevenue) OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date), 0) as revenue_growth,
                (f.netincome - LAG(f.netincome) OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date))::float
                    / NULLIF(ABS(LAG(f.netincome) OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date)), 0) as earnings_growth,
                
                -- Market Context
                m.pe_ratio,
                LOG(m.market_cap + 1) as log_market_cap,
                m.beta,
                
                -- Sector Classifications (dividend-paying sectors)
                CASE WHEN m.sector IN ('Utilities', 'Consumer Staples', 'Real Estate') THEN 1 ELSE 0 END as is_dividend_sector,
                CASE WHEN m.sector = 'Financial Services' THEN 1 ELSE 0 END as is_financial,
                CASE WHEN m.sector = 'Energy' THEN 1 ELSE 0 END as is_energy,
                CASE WHEN m.sector = 'Telecommunications' THEN 1 ELSE 0 END as is_telecom,
                
                -- Dividend History (stability indicator)
                COUNT(*) OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) as dividend_history_years
                
            FROM fundamentals_annual f
            JOIN mv_symbol_with_metadata m ON f.symbol = m.symbol
            WHERE f.fiscal_date >= CURRENT_DATE - INTERVAL ':lookback days'
                AND f.dividendpayout IS NOT NULL
                AND f.dividendpayout > 0  -- Only dividend-paying companies
                AND f.netincome IS NOT NULL
                AND f.free_cf IS NOT NULL
                AND m.market_cap > 1000000000  -- $1B minimum
                AND m.current_price > 5  -- Avoid penny stocks
        ),
        labels AS (
            SELECT
                symbol,
                as_of_date,
                return_12m as forward_return
            FROM forward_returns
            WHERE return_12m IS NOT NULL
        )
        SELECT f.*,
               l.forward_return as label
        FROM dividend_features f
        JOIN labels l ON f.symbol = l.symbol AND f.as_of_date = l.as_of_date
        WHERE f.payout_ratio IS NOT NULL
          AND f.payout_ratio BETWEEN 0.05 AND 0.95  -- Reasonable payout ratios
          AND f.dividend_yield BETWEEN 0.01 AND 0.15  -- 1% to 15% yield
          AND f.roe > 0.05  -- At least 5% ROE
          AND f.debt_to_equity < 3  -- Not overleveraged
          AND f.dividend_history_years >= 3  -- At least 3 years of dividends
        ORDER BY f.symbol, f.as_of_date
        """)

        df = pd.read_sql(query, engine, params={"lookback": self.lookback_days})
        
        if len(df) == 0:
            logger.error("No dividend training data found")
            return None
            
        logger.info(f"Loaded {len(df)} dividend training samples")
        
        # Handle missing values and outliers
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with median for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in ['symbol', 'as_of_date', 'label']:
                df[col] = df[col].fillna(df[col].median())
        
        # Remove extreme outliers (beyond 3 standard deviations)
        for col in ['payout_ratio', 'dividend_yield', 'roe', 'debt_to_equity']:
            if col in df.columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                df = df[(df[col] >= mean_val - 3*std_val) & (df[col] <= mean_val + 3*std_val)]
        
        logger.info(f"After cleaning: {len(df)} samples")
        return df

    def prepare_features(self, df):
        """Prepare feature matrix and target vector"""
        # Remove non-feature columns
        feature_columns = [col for col in df.columns 
                         if col not in ['symbol', 'as_of_date', 'label']]
        
        X = df[feature_columns].values
        y = df['label'].values
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Features: {feature_columns}")
        
        return X, y, feature_columns

    def train_model(self, X, y):
        """Train dividend prediction model with time series validation"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Use TimeSeriesSplit for proper backtesting
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Try multiple models
        models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42
            )
        }
        
        best_model = None
        best_score = float('-inf')
        best_name = None
        
        for name, model in models.items():
            scores = []
            
            for train_idx, val_idx in tscv.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                score = r2_score(y_val, y_pred)
                scores.append(score)
            
            avg_score = np.mean(scores)
            logger.info(f"{name}: Average R² = {avg_score:.4f}")
            
            if avg_score > best_score:
                best_score = avg_score
                best_model = model
                best_name = name
        
        # Train best model on full dataset
        logger.info(f"Training final {best_name} model on full dataset")
        best_model.fit(X_scaled, y)
        
        self.model = best_model
        
        # Store feature importance
        if hasattr(best_model, 'feature_importances_'):
            self.feature_importance = best_model.feature_importances_
        
        logger.info(f"Best model ({best_name}) R² score: {best_score:.4f}")
        return best_score

    def save_model(self):
        """Save trained model and scaler"""
        os.makedirs("models", exist_ok=True)
        
        version = datetime.now().strftime("v%Y%m%d")
        
        model_path = f"models/dividend_model_{version}.pkl"
        scaler_path = f"models/dividend_scaler_{version}.pkl"
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Scaler saved to {scaler_path}")
        
        # Log training run
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO ai_model_run_log (model_type, version, status, created_at)
                VALUES ('dividend', :version, 'completed', CURRENT_TIMESTAMP)
            """), {"version": version})
            conn.commit()
        
        return version

    def display_feature_importance(self, feature_columns):
        """Display feature importance"""
        if self.feature_importance is not None:
            importance_df = pd.DataFrame({
                'feature': feature_columns,
                'importance': self.feature_importance
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Feature Importances:")
            print("-" * 40)
            for _, row in importance_df.head(10).iterrows():
                print(f"{row['feature']:25s}: {row['importance']:.4f}")
            
            logger.info("Feature importance calculated and displayed")

def main():
    """Train the dividend AI model"""
    print("AI DIVIDEND MODEL TRAINING")
    print("=" * 50)
    
    trainer = DividendModelTrainer()
    
    try:
        # Load training data
        print("Loading dividend training data...")
        df = trainer.load_training_data()
        
        if df is None or len(df) < trainer.min_samples:
            print(f"ERROR: Insufficient training data ({len(df) if df is not None else 0} samples)")
            return False
        
        print(f"Loaded {len(df)} dividend training samples")
        print(f"Date range: {df['as_of_date'].min()} to {df['as_of_date'].max()}")
        print(f"Unique stocks: {df['symbol'].nunique()}")
        
        # Prepare features
        print("\nPreparing features...")
        X, y, feature_columns = trainer.prepare_features(df)
        
        # Train model
        print("\nTraining dividend model...")
        score = trainer.train_model(X, y)
        
        print(f"\nModel Performance:")
        print(f"  R² Score: {score:.4f}")
        print(f"  Features: {len(feature_columns)}")
        
        # Display feature importance
        trainer.display_feature_importance(feature_columns)
        
        # Save model
        print("\nSaving model...")
        version = trainer.save_model()
        
        print(f"\nDividend model training completed successfully!")
        print(f"Model version: {version}")
        print(f"Ready for dividend stock scoring and portfolio generation.")
        
        return True
        
    except Exception as e:
        print(f"Training error: {e}")
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()