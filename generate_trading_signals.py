#!/usr/bin/env python3
"""
Generate ML-based trading signals
Uses trained models to generate actionable trading recommendations
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import joblib
import json
import logging

load_dotenv()
engine = create_engine(os.getenv("POSTGRES_URL"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TradingSignalGenerator:
    """Generate trading signals using ML models and rankings"""
    
    def __init__(self, model_path="models/", strategy_name="ml_ensemble_v1"):
        self.model_path = model_path
        self.strategy_name = strategy_name
        self.models = {}
        self.scaler = None
        self.feature_columns = []
        self.signals = []
        
    def load_models(self):
        """Load saved ML models"""
        logger.info("Loading ML models...")
        
        model_files = {
            'return_4w': 'quality_ranking_predictor_v1_return_4w_xgboost.pkl',
            'excess_4w': 'quality_ranking_predictor_v1_excess_4w_xgboost.pkl'
        }
        
        try:
            # Load models
            for key, filename in model_files.items():
                filepath = os.path.join(self.model_path, filename)
                if os.path.exists(filepath):
                    self.models[key] = joblib.load(filepath)
                    logger.info(f"  Loaded {key} model")
                else:
                    logger.warning(f"  Model file not found: {filepath}")
            
            # Load scaler
            scaler_path = os.path.join(self.model_path, 'quality_ranking_predictor_v1_scaler.pkl')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info("  Loaded feature scaler")
            
            # Load metadata
            metadata_path = os.path.join(self.model_path, 'quality_ranking_predictor_v1_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.feature_columns = metadata[0]['feature_columns']
                    logger.info(f"  Loaded {len(self.feature_columns)} feature columns")
                    
            return len(self.models) > 0
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def get_latest_rankings(self):
        """Get most recent rankings for signal generation"""
        query = """
        WITH latest_date AS (
            SELECT MAX(ranking_date) as max_date
            FROM stock_quality_rankings
        )
        SELECT 
            r.*,
            sp.close as current_price,
            sp.volume as current_volume,
            AVG(sp.close) OVER (PARTITION BY r.symbol ORDER BY sp.date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as sma_20,
            AVG(sp.close) OVER (PARTITION BY r.symbol ORDER BY sp.date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) as sma_50
        FROM stock_quality_rankings r
        JOIN latest_date ld ON r.ranking_date = ld.max_date
        LEFT JOIN stock_prices sp 
            ON r.symbol = sp.symbol 
            AND sp.date = r.ranking_date
        WHERE r.final_composite_score IS NOT NULL
        ORDER BY r.final_composite_score DESC
        """
        
        df = pd.read_sql_query(query, engine)
        logger.info(f"Loaded {len(df)} stocks from latest rankings")
        return df
    
    def prepare_features(self, df):
        """Prepare features for ML prediction"""
        
        # Calculate additional features
        df['price_to_sma20'] = df['current_price'] / df['sma_20']
        df['price_to_sma50'] = df['current_price'] / df['sma_50']
        
        # Normalize rankings
        for col in ['beat_sp500_ranking', 'excess_cash_flow_ranking', 'fundamentals_ranking',
                   'sentiment_ranking', 'value_ranking', 'breakout_ranking', 'growth_ranking']:
            if col in df.columns:
                max_rank = df[col].max()
                df[f'{col}_norm'] = 1.0 - (df[col] / max_rank)
        
        # Handle missing features
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0  # Default value for missing features
        
        return df
    
    def generate_predictions(self, df):
        """Generate return predictions using ML models"""
        
        if not self.models:
            logger.warning("No models loaded, using rule-based signals")
            return df
        
        # Prepare features
        X = df[self.feature_columns].fillna(0)
        
        # Scale features if scaler available
        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        # Generate predictions for each model
        predictions = {}
        for model_name, model in self.models.items():
            try:
                pred = model.predict(X_scaled)
                predictions[f'predicted_{model_name}'] = pred
                logger.info(f"  Generated predictions using {model_name}")
            except Exception as e:
                logger.error(f"  Error predicting with {model_name}: {e}")
        
        # Add predictions to dataframe
        for col, pred in predictions.items():
            df[col] = pred
        
        # Calculate ensemble prediction (average)
        pred_cols = [col for col in df.columns if col.startswith('predicted_')]
        if pred_cols:
            df['predicted_return'] = df[pred_cols].mean(axis=1)
            df['prediction_std'] = df[pred_cols].std(axis=1)
            df['prediction_confidence'] = 1 / (1 + df['prediction_std'])
        else:
            df['predicted_return'] = 0
            df['prediction_confidence'] = 0
        
        return df
    
    def generate_signals(self, df, top_n=50):
        """Generate trading signals based on predictions and rankings"""
        
        signals = []
        signal_date = datetime.now().date()
        
        # Filter to top candidates
        candidates = df.nlargest(top_n * 2, 'final_composite_score')
        
        for _, row in candidates.iterrows():
            signal = {
                'strategy_name': self.strategy_name,
                'signal_date': signal_date,
                'symbol': row['symbol'],
                'signal_type': 'HOLD',  # Default
                'signal_strength': 0,
                'primary_factor': None,
                'predicted_return_4w': row.get('predicted_return_4w', 0),
                'predicted_excess_4w': row.get('predicted_excess_4w', 0),
                'prediction_confidence': row.get('prediction_confidence', 0),
                'position_size_pct': 0,
                'current_price': row.get('current_price', 0)
            }
            
            # Determine signal type based on multiple factors
            score = 0
            factors = []
            
            # ML prediction factor
            if row.get('predicted_return', 0) > 5:
                score += 30
                factors.append('ML_prediction')
            
            # Composite score factor
            if row['final_composite_score'] > 70:
                score += 20
                factors.append('high_quality')
            
            # All-star stocks
            if row.get('is_all_star', False):
                score += 20
                factors.append('all_star')
            
            # Growth champion
            if row.get('is_growth_champion', False):
                score += 15
                factors.append('growth_champion')
            
            # Momentum breakout
            if row.get('is_momentum_breakout', False):
                score += 15
                factors.append('breakout')
            
            # Deep value
            if row.get('is_deep_value_star', False):
                score += 15
                factors.append('deep_value')
            
            # Sentiment momentum
            if row.get('sentiment_momentum', 0) > 10:
                score += 10
                factors.append('sentiment_momentum')
            
            # Determine signal based on score
            if score >= 60:
                signal['signal_type'] = 'BUY'
                signal['signal_strength'] = min(score, 100)
                signal['primary_factor'] = factors[0] if factors else 'composite_quality'
                
                # Calculate position size (simplified Kelly)
                confidence = row.get('prediction_confidence', 0.5)
                expected_return = row.get('predicted_return', 5) / 100
                signal['position_size_pct'] = min(
                    confidence * expected_return * 0.25,  # Fractional Kelly
                    0.05  # Max 5% per position
                ) * 100
                
                # Set risk parameters
                if signal['current_price'] > 0:
                    signal['stop_loss_price'] = signal['current_price'] * 0.92  # 8% stop loss
                    signal['take_profit_price'] = signal['current_price'] * 1.15  # 15% take profit
                    signal['risk_reward_ratio'] = 15 / 8  # 1.875
            
            elif score >= 40:
                signal['signal_type'] = 'HOLD'
                signal['signal_strength'] = score
            
            else:
                signal['signal_type'] = 'SELL'
                signal['signal_strength'] = 100 - score
            
            signals.append(signal)
        
        # Filter to top signals
        signals_df = pd.DataFrame(signals)
        buy_signals = signals_df[signals_df['signal_type'] == 'BUY'].nlargest(top_n, 'signal_strength')
        
        return buy_signals
    
    def save_signals(self, signals_df):
        """Save signals to database"""
        
        if signals_df.empty:
            logger.warning("No signals to save")
            return
        
        try:
            # Add metadata
            signals_df['created_at'] = datetime.now()
            
            # Save to database
            signals_df.to_sql(
                'strategy_signals',
                engine,
                if_exists='append',
                index=False,
                method='multi'
            )
            
            logger.info(f"Saved {len(signals_df)} signals to database")
            
        except Exception as e:
            logger.error(f"Error saving signals: {e}")
    
    def generate_report(self, signals_df):
        """Generate signal report"""
        
        print("\n" + "="*80)
        print(f"TRADING SIGNALS REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("="*80)
        
        if signals_df.empty:
            print("No buy signals generated")
            return
        
        print(f"\nStrategy: {self.strategy_name}")
        print(f"Total Buy Signals: {len(signals_df)}")
        
        print("\n" + "-"*80)
        print("TOP 20 BUY SIGNALS")
        print("-"*80)
        
        display_cols = [
            'symbol', 'signal_strength', 'predicted_return_4w', 
            'position_size_pct', 'primary_factor', 'current_price'
        ]
        
        for i, (_, row) in enumerate(signals_df.head(20).iterrows(), 1):
            print(f"\n{i}. {row['symbol']}")
            print(f"   Signal Strength: {row['signal_strength']:.0f}/100")
            print(f"   Predicted Return: {row['predicted_return_4w']:.1f}%")
            print(f"   Position Size: {row['position_size_pct']:.1f}%")
            print(f"   Primary Factor: {row['primary_factor']}")
            print(f"   Current Price: ${row['current_price']:.2f}")
            if row.get('stop_loss_price'):
                print(f"   Stop Loss: ${row['stop_loss_price']:.2f}")
                print(f"   Take Profit: ${row['take_profit_price']:.2f}")
        
        # Summary by primary factor
        print("\n" + "-"*80)
        print("SIGNALS BY PRIMARY FACTOR")
        print("-"*80)
        
        factor_summary = signals_df.groupby('primary_factor').agg({
            'symbol': 'count',
            'signal_strength': 'mean',
            'predicted_return_4w': 'mean'
        }).rename(columns={'symbol': 'count'})
        
        print(factor_summary.to_string())
        
        print("\n" + "="*80)


def main():
    """Main execution"""
    
    # Initialize generator
    generator = TradingSignalGenerator(
        model_path="models/",
        strategy_name="ml_ensemble_v1"
    )
    
    # Check if models exist, if not use rule-based
    models_loaded = generator.load_models()
    if not models_loaded:
        logger.warning("Using rule-based signal generation (no ML models found)")
        # Continue with rule-based signals
    
    # Get latest rankings
    rankings_df = generator.get_latest_rankings()
    
    if rankings_df.empty:
        logger.error("No rankings data available")
        return 1
    
    # Prepare features
    rankings_df = generator.prepare_features(rankings_df)
    
    # Generate predictions if models available
    if models_loaded:
        rankings_df = generator.generate_predictions(rankings_df)
    
    # Generate signals
    signals_df = generator.generate_signals(rankings_df, top_n=50)
    
    # Save signals
    generator.save_signals(signals_df)
    
    # Generate report
    generator.generate_report(signals_df)
    
    print(f"\n[SUCCESS] Generated {len(signals_df)} trading signals")
    
    return 0


if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    sys.exit(main())