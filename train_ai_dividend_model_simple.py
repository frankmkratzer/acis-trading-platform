#!/usr/bin/env python3
"""
Simplified AI Dividend Model Training
Train dividend model using available fundamental data only
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
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

class SimpleDividendModelTrainer:
    def __init__(self, lookback_days=1800, min_samples=50):
        self.lookback_days = lookback_days
        self.min_samples = min_samples
        self.model = None
        self.scaler = StandardScaler()

    def load_training_data(self):
        """Load simplified dividend training data using only fundamentals table"""
        query = text("""
        WITH dividend_features AS (
            SELECT 
                f.symbol,
                f.fiscal_date as as_of_date,
                
                -- Dividend Metrics (simplified)
                CASE 
                    WHEN f.netincome > 0 AND f.dividendpayout > 0
                    THEN f.dividendpayout::float / f.netincome 
                    ELSE 0 
                END as payout_ratio,
                
                f.dividendpayout::float as dividend_amount,
                
                -- Cash Flow Coverage
                CASE 
                    WHEN f.dividendpayout > 0 AND f.free_cf > 0
                    THEN f.free_cf::float / f.dividendpayout 
                    ELSE 0 
                END as fcf_coverage,
                
                CASE 
                    WHEN f.dividendpayout > 0 AND f.operatingcashflow > 0
                    THEN f.operatingcashflow::float / f.dividendpayout 
                    ELSE 0 
                END as ocf_coverage,
                
                -- Dividend Growth (year-over-year)
                (f.dividendpayout - LAG(f.dividendpayout) OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date))::float
                    / NULLIF(LAG(f.dividendpayout) OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date), 0) as dividend_growth_1yr,
                
                -- Quality Metrics
                CASE 
                    WHEN f.totalshareholderequity > 0 
                    THEN f.netincome::float / f.totalshareholderequity 
                    ELSE 0 
                END as roe,
                
                CASE 
                    WHEN f.totalassets > 0 
                    THEN f.netincome::float / f.totalassets 
                    ELSE 0 
                END as roa,
                
                CASE 
                    WHEN f.totalrevenue > 0 
                    THEN f.netincome::float / f.totalrevenue 
                    ELSE 0 
                END as net_margin,
                
                CASE 
                    WHEN f.totalrevenue > 0 AND f.grossprofit > 0
                    THEN f.grossprofit::float / f.totalrevenue 
                    ELSE 0 
                END as gross_margin,
                
                -- Financial Stability 
                CASE 
                    WHEN f.totalshareholderequity > 0 
                    THEN f.totalliabilities::float / f.totalshareholderequity 
                    ELSE 5 
                END as debt_to_equity,
                
                CASE 
                    WHEN f.totalassets > 0 
                    THEN f.totalshareholderequity::float / f.totalassets 
                    ELSE 0 
                END as equity_ratio,
                
                -- Business Quality
                CASE 
                    WHEN f.totalrevenue > 0 AND f.free_cf > 0
                    THEN f.free_cf::float / f.totalrevenue 
                    ELSE 0 
                END as fcf_margin,
                
                CASE 
                    WHEN f.totalassets > 0 
                    THEN f.totalrevenue::float / f.totalassets 
                    ELSE 0 
                END as asset_turnover,
                
                -- Growth Quality
                (f.totalrevenue - LAG(f.totalrevenue) OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date))::float
                    / NULLIF(LAG(f.totalrevenue) OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date), 0) as revenue_growth,
                
                (f.netincome - LAG(f.netincome) OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date))::float
                    / NULLIF(ABS(LAG(f.netincome) OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date)), 0) as earnings_growth,
                
                -- Size (log of revenue as proxy for market cap)
                LOG(GREATEST(f.totalrevenue, 1000000)) as log_revenue,
                
                -- Dividend Consistency (count of consecutive years with dividends)
                COUNT(CASE WHEN f.dividendpayout > 0 THEN 1 END) 
                    OVER (PARTITION BY f.symbol ORDER BY f.fiscal_date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) as dividend_years_5yr
                
            FROM fundamentals_annual f
            WHERE f.fiscal_date >= CURRENT_DATE - INTERVAL ':lookback days'
                AND f.dividendpayout IS NOT NULL
                AND f.dividendpayout > 0  -- Only dividend-paying companies
                AND f.netincome IS NOT NULL
                AND f.totalrevenue > 0
                AND f.totalassets > 0
                AND f.totalshareholderequity > 0
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
        WHERE f.payout_ratio BETWEEN 0.05 AND 0.90  -- Reasonable payout ratios
          AND f.roe > 0.03  -- At least 3% ROE
          AND f.debt_to_equity < 5  -- Not extremely leveraged
          AND f.dividend_years_5yr >= 2  -- At least 2 years in last 5
          AND ABS(f.forward_return) < 3  -- Remove extreme outliers
        ORDER BY f.symbol, f.as_of_date
        """)

        df = pd.read_sql(query, engine, params={"lookback": self.lookback_days})
        
        if len(df) == 0:
            logger.error("No dividend training data found")
            return None
            
        logger.info(f"Loaded {len(df)} dividend training samples")
        
        # Handle missing values and outliers
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with reasonable defaults
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in ['symbol', 'as_of_date', 'label']:
                if col in ['debt_to_equity']:
                    df[col] = df[col].fillna(2.0)  # Moderate debt
                elif col in ['roe', 'roa', 'net_margin']:
                    df[col] = df[col].fillna(0.05)  # Low but positive
                else:
                    df[col] = df[col].fillna(df[col].median())
        
        # Remove extreme outliers
        for col in ['payout_ratio', 'roe', 'debt_to_equity']:
            if col in df.columns:
                q99 = df[col].quantile(0.99)
                q01 = df[col].quantile(0.01)
                df = df[(df[col] >= q01) & (df[col] <= q99)]
        
        logger.info(f"After cleaning: {len(df)} samples")
        return df

    def prepare_features(self, df):
        """Prepare feature matrix and target vector"""
        feature_columns = [col for col in df.columns 
                         if col not in ['symbol', 'as_of_date', 'label']]
        
        X = df[feature_columns].values
        y = df['label'].values
        
        logger.info(f"Feature matrix shape: {X.shape}")
        return X, y, feature_columns

    def train_model(self, X, y):
        """Train simplified dividend model"""
        X_scaled = self.scaler.fit_transform(X)
        
        # Use simple RandomForest for robustness
        self.model = RandomForestRegressor(
            n_estimators=50,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        # Train on full dataset
        self.model.fit(X_scaled, y)
        
        # Cross-validation score estimate
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            temp_model = RandomForestRegressor(n_estimators=20, random_state=42)
            temp_model.fit(X_train, y_train)
            y_pred = temp_model.predict(X_val)
            score = r2_score(y_val, y_pred)
            scores.append(score)
        
        avg_score = np.mean(scores)
        logger.info(f"Cross-validation R² score: {avg_score:.4f}")
        
        return avg_score

    def save_model(self):
        """Save trained model and scaler"""
        os.makedirs("models", exist_ok=True)
        
        version = datetime.now().strftime("v%Y%m%d")
        
        model_path = f"models/dividend_model_{version}.pkl"
        scaler_path = f"models/dividend_scaler_{version}.pkl"
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        logger.info(f"Model saved to {model_path}")
        
        # Log training run
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO ai_model_run_log (model_type, version, status, created_at)
                VALUES ('dividend', :version, 'completed', CURRENT_TIMESTAMP)
                ON CONFLICT (model_type, version) DO UPDATE SET
                    status = 'completed',
                    created_at = CURRENT_TIMESTAMP
            """), {"version": version})
            conn.commit()
        
        return version

    def display_feature_importance(self, feature_columns):
        """Display feature importance"""
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Feature Importances:")
            print("-" * 40)
            for _, row in importance_df.head(10).iterrows():
                print(f"{row['feature']:25s}: {row['importance']:.4f}")

def main():
    """Train the simplified dividend AI model"""
    print("SIMPLIFIED AI DIVIDEND MODEL TRAINING")
    print("=" * 50)
    
    trainer = SimpleDividendModelTrainer()
    
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
        
        print(f"\nDividend model training completed!")
        print(f"Model version: {version}")
        print("Ready for dividend scoring.")
        
        return True
        
    except Exception as e:
        print(f"Training error: {e}")
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()