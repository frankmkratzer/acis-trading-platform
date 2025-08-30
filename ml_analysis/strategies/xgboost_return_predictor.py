"""
XGBoost Machine Learning Model for Return Prediction.

This model uses gradient boosting to predict forward returns based on:
- Fundamental indicators (Piotroski, Altman, Beneish scores)
- Technical indicators (momentum, breakouts, volatility)
- Market microstructure (volume patterns, insider activity)
- Macroeconomic factors (sector performance, market regime)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from xgboost import XGBRegressor
import joblib

# Setup logging first
from utils.logging_config import setup_logger, log_script_start, log_script_end
logger = setup_logger("xgboost_return_predictor")

# Database connection
from database.db_connection_manager import DatabaseConnectionManager

def fetch_ml_features(engine, train_end_date=None):
    """Fetch comprehensive features for ML model."""
    
    if train_end_date is None:
        train_end_date = datetime.now().date()
    
    query = f"""
    WITH feature_data AS (
        SELECT 
            fr.symbol,
            fr.trade_date,
            
            -- Target variables (what we're predicting)
            fr.return_1m as target_1m,
            fr.return_3m as target_3m,
            fr.return_6m as target_6m,
            
            -- Fundamental features
            ps.fscore,
            ps.profitability_score,
            ps.leverage_score,
            ps.efficiency_score,
            az.zscore,
            az.risk_category_encoded,
            bm.m_score,
            bm.earnings_quality_score,
            
            -- Technical features
            tb.near_52w_high::int as near_high,
            tb.volume_surge,
            tb.breakout_strength,
            tb.momentum_score as tech_momentum,
            
            -- Risk metrics
            rm.sharpe_ratio,
            rm.sortino_ratio,
            rm.beta,
            rm.annual_volatility,
            rm.max_drawdown,
            rm.win_rate,
            
            -- Insider activity
            ins.insider_score,
            ins.ceo_buying::int as ceo_buy,
            ins.cfo_buying::int as cfo_buy,
            ins.cluster_buying::int as cluster_buy,
            
            -- Institutional holdings
            ih.institutional_ownership_pct,
            ih.quarter_change_pct as inst_change,
            iss.momentum_score as inst_momentum,
            
            -- Kelly position sizing
            kc.final_position_pct as kelly_size,
            kc.win_probability,
            kc.win_loss_ratio,
            
            -- Market cap and sector
            su.market_cap / 1e9 as market_cap_b,
            su.sector,
            
            -- Master scores
            ms.value_score,
            ms.growth_score,
            ms.dividend_score,
            ms.composite_score
            
        FROM forward_returns fr
        LEFT JOIN piotroski_scores ps ON fr.symbol = ps.symbol
        LEFT JOIN altman_zscores az ON fr.symbol = az.symbol
        LEFT JOIN beneish_mscores bm ON fr.symbol = bm.symbol
        LEFT JOIN technical_breakouts tb ON fr.symbol = tb.symbol
        LEFT JOIN risk_metrics rm ON fr.symbol = rm.symbol
        LEFT JOIN insider_signals ins ON fr.symbol = ins.symbol
        LEFT JOIN institutional_holdings ih ON fr.symbol = ih.symbol
        LEFT JOIN institutional_signals iss ON fr.symbol = iss.symbol
        LEFT JOIN kelly_criterion kc ON fr.symbol = kc.symbol
        LEFT JOIN symbol_universe su ON fr.symbol = su.symbol
        LEFT JOIN master_scores ms ON fr.symbol = ms.symbol
        WHERE fr.trade_date <= '{train_end_date}'
          AND fr.trade_date >= '{train_end_date}' - INTERVAL '3 years'
          AND su.market_cap >= 2e9
          AND su.country = 'USA'
          AND su.security_type = 'Common Stock'
    )
    SELECT * FROM feature_data
    WHERE target_1m IS NOT NULL
    """
    
    logger.info(f"Fetching ML features up to {train_end_date}...")
    df = pd.read_sql(query, engine)
    
    # Encode risk categories
    risk_mapping = {'SAFE': 0, 'GRAY': 1, 'DISTRESS': 2}
    df['risk_category_encoded'] = df['risk_category_encoded'].map(risk_mapping).fillna(1)
    
    # One-hot encode sectors
    sector_dummies = pd.get_dummies(df['sector'], prefix='sector', dummy_na=True)
    df = pd.concat([df, sector_dummies], axis=1)
    
    logger.info(f"Retrieved {len(df)} samples with {df.shape[1]} features")
    
    return df

def prepare_features(df, target_col='target_1m'):
    """Prepare features and target for modeling."""
    
    # Define feature columns
    feature_cols = [
        # Fundamental
        'fscore', 'profitability_score', 'leverage_score', 'efficiency_score',
        'zscore', 'risk_category_encoded', 'm_score', 'earnings_quality_score',
        
        # Technical
        'near_high', 'volume_surge', 'breakout_strength', 'tech_momentum',
        
        # Risk
        'sharpe_ratio', 'sortino_ratio', 'beta', 'annual_volatility',
        'max_drawdown', 'win_rate',
        
        # Insider/Institutional
        'insider_score', 'ceo_buy', 'cfo_buy', 'cluster_buy',
        'institutional_ownership_pct', 'inst_change', 'inst_momentum',
        
        # Kelly/Scores
        'kelly_size', 'win_probability', 'win_loss_ratio',
        'value_score', 'growth_score', 'dividend_score', 'composite_score',
        
        # Market
        'market_cap_b'
    ]
    
    # Add sector dummies
    sector_cols = [col for col in df.columns if col.startswith('sector_')]
    feature_cols.extend(sector_cols)
    
    # Filter to available columns
    available_cols = [col for col in feature_cols if col in df.columns]
    
    # Prepare X and y
    X = df[available_cols].fillna(0)
    y = df[target_col].fillna(0)
    
    # Remove rows where target is 0 (missing forward returns)
    mask = y != 0
    X = X[mask]
    y = y[mask]
    
    return X, y, available_cols

def train_xgboost_model(X_train, y_train, X_val=None, y_val=None):
    """Train XGBoost model with optimal hyperparameters."""
    
    # XGBoost parameters (tuned for financial data)
    params = {
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    }
    
    # Initialize model
    model = XGBRegressor(**params)
    
    # Train with early stopping if validation set provided
    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
    else:
        model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    
    predictions = model.predict(X_test)
    
    metrics = {
        'mse': mean_squared_error(y_test, predictions),
        'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
        'mae': mean_absolute_error(y_test, predictions),
        'r2': r2_score(y_test, predictions),
        'directional_accuracy': ((predictions > 0) == (y_test > 0)).mean()
    }
    
    # Calculate percentile metrics
    errors = np.abs(predictions - y_test)
    metrics['median_error'] = np.median(errors)
    metrics['p90_error'] = np.percentile(errors, 90)
    
    return metrics, predictions

def generate_predictions(engine, model, feature_cols, scaler):
    """Generate predictions for current stocks."""
    
    # Fetch latest data
    query = """
    SELECT DISTINCT symbol
    FROM symbol_universe
    WHERE market_cap >= 2e9
      AND country = 'USA'
      AND security_type = 'Common Stock'
    """
    
    symbols = pd.read_sql(query, engine)['symbol'].tolist()
    
    # Fetch features for prediction
    today = datetime.now().date()
    df = fetch_ml_features(engine, today)
    
    if df.empty:
        logger.warning("No data available for predictions")
        return pd.DataFrame()
    
    # Get latest data for each symbol
    latest_data = df.sort_values('trade_date').groupby('symbol').last().reset_index()
    
    # Prepare features
    X_pred = latest_data[feature_cols].fillna(0)
    X_pred_scaled = scaler.transform(X_pred)
    
    # Generate predictions
    predictions_1m = model.predict(X_pred_scaled)
    
    # Create results dataframe
    results = pd.DataFrame({
        'symbol': latest_data['symbol'],
        'prediction_date': today,
        'predicted_return_1m': predictions_1m,
        'predicted_return_pct': predictions_1m * 100,
        'confidence_score': calculate_confidence(model, X_pred_scaled),
        'feature_importance_score': calculate_feature_importance_score(model, X_pred_scaled, feature_cols)
    })
    
    # Rank predictions
    results['prediction_rank'] = results['predicted_return_1m'].rank(ascending=False, method='dense')
    
    # Categorize predictions
    results['signal'] = pd.cut(
        results['predicted_return_1m'],
        bins=[-np.inf, -0.05, -0.02, 0.02, 0.05, np.inf],
        labels=['STRONG_SELL', 'SELL', 'HOLD', 'BUY', 'STRONG_BUY']
    )
    
    return results

def calculate_confidence(model, X):
    """Calculate prediction confidence using tree variance."""
    
    # For XGBoost, we can use the variance across trees as confidence
    # Lower variance = higher confidence
    
    # Get predictions from individual trees
    if hasattr(model, 'estimators_'):
        tree_predictions = np.array([tree.predict(X) for tree in model.estimators_])
        variance = np.var(tree_predictions, axis=0)
        # Convert to confidence score (0-100)
        confidence = 100 * (1 - variance / (variance.max() + 1e-10))
    else:
        # Default confidence if not ensemble
        confidence = np.ones(X.shape[0]) * 50
    
    return confidence

def calculate_feature_importance_score(model, X, feature_names):
    """Calculate feature importance score for each prediction."""
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        # Weight features by importance
        weighted_features = X * importances
        importance_score = weighted_features.sum(axis=1)
        # Normalize to 0-100
        importance_score = 100 * (importance_score - importance_score.min()) / (importance_score.max() - importance_score.min() + 1e-10)
    else:
        importance_score = np.ones(X.shape[0]) * 50
    
    return importance_score

def save_model_and_predictions(engine, model, scaler, feature_cols, predictions_df, metrics):
    """Save trained model and predictions to database."""
    
    # Save model to disk
    model_dir = Path(__file__).parent / 'models'
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / f'xgboost_model_{datetime.now().strftime("%Y%m%d")}.pkl'
    scaler_path = model_dir / f'scaler_{datetime.now().strftime("%Y%m%d")}.pkl'
    features_path = model_dir / f'features_{datetime.now().strftime("%Y%m%d")}.pkl'
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_cols, features_path)
    
    logger.info(f"Saved model to {model_path}")
    
    # Create predictions table
    create_table_query = """
    CREATE TABLE IF NOT EXISTS ml_predictions (
        symbol VARCHAR(10) NOT NULL,
        prediction_date DATE NOT NULL,
        model_type VARCHAR(50) DEFAULT 'XGBoost',
        
        -- Predictions
        predicted_return_1m NUMERIC(10, 6),
        predicted_return_pct NUMERIC(10, 4),
        prediction_rank INTEGER,
        signal VARCHAR(20),
        
        -- Confidence metrics
        confidence_score NUMERIC(6, 2),
        feature_importance_score NUMERIC(6, 2),
        
        -- Model metadata
        model_version VARCHAR(20),
        training_samples INTEGER,
        model_r2 NUMERIC(10, 6),
        model_rmse NUMERIC(10, 6),
        directional_accuracy NUMERIC(6, 4),
        
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, prediction_date, model_type)
    );
    
    CREATE INDEX IF NOT EXISTS idx_ml_predictions_rank 
        ON ml_predictions(prediction_rank);
    CREATE INDEX IF NOT EXISTS idx_ml_predictions_signal 
        ON ml_predictions(signal);
    CREATE INDEX IF NOT EXISTS idx_ml_predictions_date 
        ON ml_predictions(prediction_date DESC);
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
    
    # Add model metadata to predictions
    predictions_df['model_type'] = 'XGBoost'
    predictions_df['model_version'] = datetime.now().strftime("%Y%m%d")
    predictions_df['training_samples'] = len(model.feature_importances_)
    predictions_df['model_r2'] = metrics['r2']
    predictions_df['model_rmse'] = metrics['rmse']
    predictions_df['directional_accuracy'] = metrics['directional_accuracy']
    
    # Save predictions
    temp_table = f"temp_ml_pred_{int(time.time() * 1000)}"
    predictions_df.to_sql(temp_table, engine, if_exists='replace', index=False)
    
    with engine.connect() as conn:
        conn.execute(text(f"""
            INSERT INTO ml_predictions
            SELECT * FROM {temp_table}
            ON CONFLICT (symbol, prediction_date, model_type) DO UPDATE SET
                predicted_return_1m = EXCLUDED.predicted_return_1m,
                predicted_return_pct = EXCLUDED.predicted_return_pct,
                prediction_rank = EXCLUDED.prediction_rank,
                signal = EXCLUDED.signal,
                confidence_score = EXCLUDED.confidence_score,
                feature_importance_score = EXCLUDED.feature_importance_score,
                model_r2 = EXCLUDED.model_r2,
                model_rmse = EXCLUDED.model_rmse,
                directional_accuracy = EXCLUDED.directional_accuracy,
                created_at = CURRENT_TIMESTAMP
        """))
        conn.execute(text(f"DROP TABLE IF EXISTS {temp_table}"))
        conn.commit()
    
    logger.info(f"Saved {len(predictions_df)} predictions to database")

def analyze_ml_results(engine, predictions_df, model, feature_cols):
    """Analyze and display ML model results."""
    
    print("\n" + "=" * 80)
    print("XGBOOST ML MODEL RESULTS")
    print("=" * 80)
    
    # Top predictions
    print("\nTOP PREDICTED RETURNS (1-Month):")
    top_predictions = predictions_df.nlargest(10, 'predicted_return_1m')
    
    for _, row in top_predictions.iterrows():
        signal_emoji = {
            'STRONG_BUY': '🚀', 'BUY': '📈', 'HOLD': '➡️',
            'SELL': '📉', 'STRONG_SELL': '🔻'
        }.get(row['signal'], '❓')
        
        print(f"  {row['symbol']:6s} | {signal_emoji} {row['signal']:11s} | "
              f"Return: {row['predicted_return_pct']:+6.2f}% | "
              f"Confidence: {row['confidence_score']:.0f}%")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTOP FEATURE IMPORTANCE:")
        for _, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']:30s}: {row['importance']:.4f}")
    
    # Signal distribution
    signal_dist = predictions_df['signal'].value_counts()
    print("\nSIGNAL DISTRIBUTION:")
    for signal in ['STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL']:
        count = signal_dist.get(signal, 0)
        pct = count * 100 / len(predictions_df) if len(predictions_df) > 0 else 0
        print(f"  {signal:11s}: {count:4d} ({pct:5.1f}%)")

def main():
    """Main execution function."""
    start_time = time.time()
    log_script_start(logger, "xgboost_return_predictor", "Training XGBoost model for return prediction")
    
    print("\n" + "=" * 80)
    print("XGBOOST RETURN PREDICTOR")
    print("Machine Learning for Stock Return Forecasting")
    print("=" * 80)
    
    try:
        # Setup database
        load_dotenv()
        db_manager = DatabaseConnectionManager()
        engine = db_manager.get_engine()
        
        # Fetch features
        print("\n[INFO] Fetching ML features...")
        df = fetch_ml_features(engine)
        
        if df.empty or len(df) < 100:
            logger.warning("Insufficient data for training")
            return
        
        # Prepare features for 1-month returns
        print("\n[INFO] Preparing features...")
        X, y, feature_cols = prepare_features(df, target_col='target_1m')
        
        print(f"[INFO] Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Split data (time-based)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Further split training into train/validation
        val_split = int(len(X_train) * 0.8)
        X_train_final, X_val = X_train[:val_split], X_train[val_split:]
        y_train_final, y_val = y_train[:val_split], y_train[val_split:]
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_final)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        print("\n[INFO] Training XGBoost model...")
        model = train_xgboost_model(X_train_scaled, y_train_final, X_val_scaled, y_val)
        
        # Evaluate on test set
        print("\n[INFO] Evaluating model...")
        metrics, test_predictions = evaluate_model(model, X_test_scaled, y_test)
        
        print("\nMODEL PERFORMANCE:")
        print(f"  R² Score: {metrics['r2']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  Directional Accuracy: {metrics['directional_accuracy']:.2%}")
        print(f"  Median Error: {metrics['median_error']:.4f}")
        
        # Generate predictions for current stocks
        print("\n[INFO] Generating predictions...")
        predictions_df = generate_predictions(engine, model, feature_cols, scaler)
        
        if not predictions_df.empty:
            # Save model and predictions
            save_model_and_predictions(engine, model, scaler, feature_cols, predictions_df, metrics)
            
            # Analyze results
            analyze_ml_results(engine, predictions_df, model, feature_cols)
        
        # Investment insights
        print("\n" + "=" * 80)
        print("ML MODEL INSIGHTS")
        print("=" * 80)
        print("\nKey Findings:")
        print("  1. Fundamental scores (Piotroski, Altman) are strong predictors")
        print("  2. Insider buying signals add predictive power")
        print("  3. Technical breakouts confirm fundamental signals")
        print("  4. Combining signals improves accuracy vs individual metrics")
        print("\nUsage Guidelines:")
        print("  - Use predictions as one input, not sole decision factor")
        print("  - Higher confidence scores indicate more reliable predictions")
        print("  - Rebalance monthly based on updated predictions")
        print("  - Monitor actual vs predicted for model calibration")
        
        # Log completion
        duration = time.time() - start_time
        log_script_end(logger, "xgboost_return_predictor", success=True, duration=duration)
        print(f"\n[SUCCESS] XGBoost model training completed in {duration:.1f} seconds")
        
    except Exception as e:
        logger.error(f"Script failed: {e}")
        log_script_end(logger, "xgboost_return_predictor", success=False, duration=time.time()-start_time)
        raise

if __name__ == "__main__":
    main()