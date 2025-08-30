#!/usr/bin/env python3
"""
LSTM Deep Learning Model for Stock Return Prediction
Advanced time series prediction using Long Short-Term Memory networks

This model captures temporal dependencies in stock data that traditional ML misses:
- Sequential price patterns and momentum
- Volume-price relationships over time  
- Fundamental metric evolution
- Market regime changes

Architecture:
- Bi-directional LSTM layers for past/future context
- Attention mechanism for feature importance
- Ensemble with XGBoost for best of both worlds
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import text
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Deep Learning imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization, Attention
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

# Setup logging
from utils.logging_config import setup_logger, log_script_start, log_script_end
logger = setup_logger("lstm_return_predictor")

# Database connection
from database.db_connection_manager import DatabaseConnectionManager

class StockLSTMPredictor:
    """LSTM model for predicting stock returns"""
    
    def __init__(self, sequence_length=60, n_features=50, hidden_units=128):
        """
        Initialize LSTM predictor
        
        Args:
            sequence_length: Number of time steps to look back
            n_features: Number of features per time step
            hidden_units: Number of LSTM units
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.hidden_units = hidden_units
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def fetch_time_series_data(self, engine, symbol=None, start_date=None):
        """
        Fetch comprehensive time series data for LSTM training
        
        Returns sequences of features and targets
        """
        
        if start_date is None:
            start_date = datetime(1990, 1, 1).date()
            
        query = f"""
        WITH daily_features AS (
            SELECT 
                sp.symbol,
                sp.date as trade_date,
                
                -- Price features
                sp.open_price,
                sp.high_price,
                sp.low_price,
                sp.close_price,
                sp.adjusted_close,
                sp.volume,
                
                -- Price-derived features
                (sp.high_price - sp.low_price) / sp.open_price as daily_range,
                (sp.close_price - sp.open_price) / sp.open_price as daily_return,
                sp.volume * sp.close_price as dollar_volume,
                
                -- Technical indicators (need to be calculated or fetched)
                AVG(sp.close_price) OVER (PARTITION BY sp.symbol ORDER BY sp.date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as sma_20,
                AVG(sp.close_price) OVER (PARTITION BY sp.symbol ORDER BY sp.date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) as sma_50,
                AVG(sp.close_price) OVER (PARTITION BY sp.symbol ORDER BY sp.date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW) as sma_200,
                AVG(sp.volume) OVER (PARTITION BY sp.symbol ORDER BY sp.date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as volume_sma_20,
                
                -- Volatility
                STDDEV(sp.close_price) OVER (PARTITION BY sp.symbol ORDER BY sp.date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as volatility_20d,
                
                -- Forward returns (targets)
                LEAD(sp.close_price, 5) OVER (PARTITION BY sp.symbol ORDER BY sp.date) as price_5d,
                LEAD(sp.close_price, 21) OVER (PARTITION BY sp.symbol ORDER BY sp.date) as price_1m,
                LEAD(sp.close_price, 63) OVER (PARTITION BY sp.symbol ORDER BY sp.date) as price_3m,
                
                -- Market cap for scaling
                su.market_cap
                
            FROM stock_prices sp
            JOIN symbol_universe su ON sp.symbol = su.symbol
            WHERE sp.date >= '{start_date}'
                AND su.market_cap >= 2e9
                AND su.country = 'USA'
                {"AND sp.symbol = '" + symbol + "'" if symbol else ""}
            ORDER BY sp.symbol, sp.date
        ),
        feature_engineered AS (
            SELECT 
                *,
                -- Calculate returns
                (price_5d / close_price - 1) as return_5d,
                (price_1m / close_price - 1) as return_1m,
                (price_3m / close_price - 1) as return_3m,
                
                -- Relative strength
                close_price / NULLIF(sma_20, 0) as price_to_sma20,
                close_price / NULLIF(sma_50, 0) as price_to_sma50,
                close_price / NULLIF(sma_200, 0) as price_to_sma200,
                
                -- Volume indicators
                volume / NULLIF(volume_sma_20, 0) as relative_volume,
                
                -- Normalized volatility
                volatility_20d / NULLIF(close_price, 0) as normalized_volatility
                
            FROM daily_features
            WHERE sma_200 IS NOT NULL  -- Ensure we have enough history
        )
        SELECT * FROM feature_engineered
        WHERE return_3m IS NOT NULL  -- Ensure we have forward returns
        ORDER BY symbol, trade_date
        """
        
        logger.info(f"Fetching time series data from {start_date}...")
        df = pd.read_sql(query, engine)
        logger.info(f"Retrieved {len(df)} daily records for {df['symbol'].nunique()} stocks")
        
        return df
    
    def prepare_sequences(self, df, target_col='return_1m'):
        """
        Prepare sequences for LSTM training
        
        Args:
            df: DataFrame with time series data
            target_col: Column name for prediction target
            
        Returns:
            X: Array of sequences (samples, time_steps, features)
            y: Array of targets
        """
        
        feature_cols = [
            'open_price', 'high_price', 'low_price', 'close_price', 'volume',
            'daily_range', 'daily_return', 'dollar_volume',
            'price_to_sma20', 'price_to_sma50', 'price_to_sma200',
            'relative_volume', 'normalized_volatility'
        ]
        
        sequences = []
        targets = []
        
        # Group by symbol and create sequences
        for symbol, group in df.groupby('symbol'):
            if len(group) < self.sequence_length + 1:
                continue
                
            # Normalize features within each stock
            features = group[feature_cols].values
            target = group[target_col].values
            
            # Create sequences
            for i in range(len(features) - self.sequence_length):
                sequences.append(features[i:i + self.sequence_length])
                targets.append(target[i + self.sequence_length])
        
        X = np.array(sequences)
        y = np.array(targets)
        
        logger.info(f"Created {len(X)} sequences of length {self.sequence_length}")
        
        return X, y
    
    def build_model(self, input_shape):
        """
        Build LSTM model architecture
        
        Args:
            input_shape: Shape of input sequences (time_steps, features)
        """
        
        model = models.Sequential([
            # First LSTM layer with return sequences
            layers.LSTM(self.hidden_units, 
                       return_sequences=True,
                       input_shape=input_shape,
                       dropout=0.2,
                       recurrent_dropout=0.2),
            layers.BatchNormalization(),
            
            # Second LSTM layer
            layers.LSTM(self.hidden_units // 2,
                       return_sequences=True,
                       dropout=0.2,
                       recurrent_dropout=0.2),
            layers.BatchNormalization(),
            
            # Third LSTM layer
            layers.LSTM(self.hidden_units // 4,
                       dropout=0.2,
                       recurrent_dropout=0.2),
            layers.BatchNormalization(),
            
            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(1, activation='linear')
        ])
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        self.model = model
        
        logger.info("Model architecture built successfully")
        return model
    
    def build_attention_model(self, input_shape):
        """
        Build LSTM with attention mechanism
        
        More sophisticated architecture for capturing important time steps
        """
        
        inputs = layers.Input(shape=input_shape)
        
        # Bi-directional LSTM
        lstm_out, forward_h, forward_c, backward_h, backward_c = layers.Bidirectional(
            layers.LSTM(self.hidden_units, return_sequences=True, return_state=True),
            merge_mode='concat'
        )(inputs)
        
        # Attention mechanism
        attention = layers.Attention()([lstm_out, lstm_out])
        
        # Concatenate LSTM output and attention
        concat = layers.Concatenate()([lstm_out, attention])
        
        # Additional LSTM layer
        lstm_out2 = layers.LSTM(self.hidden_units // 2, return_sequences=False)(concat)
        
        # Dense layers
        dense1 = layers.Dense(64, activation='relu')(lstm_out2)
        dropout1 = layers.Dropout(0.3)(dense1)
        dense2 = layers.Dense(32, activation='relu')(dropout1)
        dropout2 = layers.Dropout(0.2)(dense2)
        
        # Output
        outputs = layers.Dense(1, activation='linear')(dropout2)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        self.model = model
        
        logger.info("Attention-based model built successfully")
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """
        Train the LSTM model
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        
        # Scale the data
        X_train_scaled = self.scaler_X.fit_transform(
            X_train.reshape(-1, X_train.shape[-1])
        ).reshape(X_train.shape)
        
        X_val_scaled = self.scaler_X.transform(
            X_val.reshape(-1, X_val.shape[-1])
        ).reshape(X_val.shape)
        
        y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1))
        y_val_scaled = self.scaler_y.transform(y_val.reshape(-1, 1))
        
        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        checkpoint = ModelCheckpoint(
            'lstm_best_model.h5',
            monitor='val_loss',
            save_best_only=True
        )
        
        # Train model
        history = self.model.fit(
            X_train_scaled, y_train_scaled,
            validation_data=(X_val_scaled, y_val_scaled),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr, checkpoint],
            verbose=1
        )
        
        logger.info(f"Training completed. Best val_loss: {min(history.history['val_loss']):.4f}")
        
        return history
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Input sequences
            
        Returns:
            Predictions (descaled)
        """
        
        X_scaled = self.scaler_X.transform(
            X.reshape(-1, X.shape[-1])
        ).reshape(X.shape)
        
        predictions_scaled = self.model.predict(X_scaled)
        predictions = self.scaler_y.inverse_transform(predictions_scaled)
        
        return predictions
    
    def ensemble_predict(self, X, xgboost_pred, lstm_weight=0.6):
        """
        Ensemble LSTM with XGBoost predictions
        
        Args:
            X: Input features
            xgboost_pred: XGBoost predictions
            lstm_weight: Weight for LSTM predictions
            
        Returns:
            Ensemble predictions
        """
        
        lstm_pred = self.predict(X)
        ensemble_pred = lstm_weight * lstm_pred + (1 - lstm_weight) * xgboost_pred
        
        return ensemble_pred

def main():
    """Main execution function"""
    
    start_time = datetime.now()
    log_script_start(logger, "lstm_return_predictor", "Training LSTM model for return prediction")
    
    try:
        # Setup
        load_dotenv()
        db_manager = DatabaseConnectionManager()
        engine = db_manager.get_engine()
        
        # Initialize predictor
        predictor = StockLSTMPredictor(
            sequence_length=60,  # 60 days lookback
            n_features=13,  # Number of features
            hidden_units=128
        )
        
        # Fetch data
        print("\n[1/5] Fetching time series data...")
        df = predictor.fetch_time_series_data(engine, start_date=datetime(2000, 1, 1).date())
        
        # Prepare sequences
        print("\n[2/5] Preparing sequences...")
        X, y = predictor.prepare_sequences(df, target_col='return_1m')
        
        # Split data (time series split)
        split_date = int(len(X) * 0.8)
        X_train, X_val = X[:split_date], X[split_date:]
        y_train, y_val = y[:split_date], y[split_date:]
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        # Build model
        print("\n[3/5] Building LSTM model...")
        predictor.build_attention_model(input_shape=(X.shape[1], X.shape[2]))
        print(predictor.model.summary())
        
        # Train model
        print("\n[4/5] Training model...")
        history = predictor.train(
            X_train, y_train,
            X_val, y_val,
            epochs=50,
            batch_size=64
        )
        
        # Evaluate
        print("\n[5/5] Evaluating model...")
        val_predictions = predictor.predict(X_val)
        val_mae = np.mean(np.abs(val_predictions.flatten() - y_val))
        val_rmse = np.sqrt(np.mean((val_predictions.flatten() - y_val) ** 2))
        
        print(f"\nValidation Metrics:")
        print(f"MAE: {val_mae:.4f}")
        print(f"RMSE: {val_rmse:.4f}")
        
        # Save model
        predictor.model.save('lstm_stock_predictor.h5')
        print("\nModel saved to lstm_stock_predictor.h5")
        
        # Performance summary
        positive_predictions = np.sum(val_predictions > 0) / len(val_predictions)
        actual_positive = np.sum(y_val > 0) / len(y_val)
        
        print(f"\nPrediction Statistics:")
        print(f"Predicted positive returns: {positive_predictions:.1%}")
        print(f"Actual positive returns: {actual_positive:.1%}")
        
        duration = (datetime.now() - start_time).total_seconds()
        log_script_end(logger, "lstm_return_predictor", success=True, duration=duration)
        
    except Exception as e:
        logger.error(f"LSTM training failed: {e}")
        raise

if __name__ == "__main__":
    main()