#!/usr/bin/env python3
"""
Machine Learning Strategy Framework for ACIS Trading Platform
Develops and backtests ML/DL strategies using historical ranking data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import logging
import warnings
warnings.filterwarnings('ignore')

load_dotenv()
engine = create_engine(os.getenv("POSTGRES_URL"))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MLStrategyFramework:
    """
    Framework for developing ML-based trading strategies using ranking data
    """
    
    def __init__(self, model_name="ranking_predictor_v1"):
        self.model_name = model_name
        self.engine = engine
        self.models = {}
        self.feature_columns = None
        self.scaler = StandardScaler()
        self.performance_metrics = {}
        
    def load_training_data(self, start_date=None, end_date=None, min_history_weeks=52):
        """
        Load historical ranking and return data for training
        
        Args:
            start_date: Start of training period
            end_date: End of training period  
            min_history_weeks: Minimum weeks of history required per stock
        """
        logger.info("Loading training data from database...")
        
        query = """
        WITH feature_data AS (
            SELECT 
                r.symbol,
                r.ranking_date,
                
                -- Normalized rankings (0-1 scale)
                1.0 - (r.beat_sp500_ranking::float / NULLIF(MAX(r.beat_sp500_ranking) OVER (PARTITION BY r.ranking_date), 1)) as sp500_rank_norm,
                1.0 - (r.excess_cash_flow_ranking::float / NULLIF(MAX(r.excess_cash_flow_ranking) OVER (PARTITION BY r.ranking_date), 1)) as fcf_rank_norm,
                1.0 - (r.fundamentals_ranking::float / NULLIF(MAX(r.fundamentals_ranking) OVER (PARTITION BY r.ranking_date), 1)) as fund_rank_norm,
                1.0 - (r.sentiment_ranking::float / NULLIF(MAX(r.sentiment_ranking) OVER (PARTITION BY r.ranking_date), 1)) as sent_rank_norm,
                1.0 - (r.value_ranking::float / NULLIF(MAX(r.value_ranking) OVER (PARTITION BY r.ranking_date), 1)) as value_rank_norm,
                1.0 - (r.breakout_ranking::float / NULLIF(MAX(r.breakout_ranking) OVER (PARTITION BY r.ranking_date), 1)) as break_rank_norm,
                1.0 - (r.growth_ranking::float / NULLIF(MAX(r.growth_ranking) OVER (PARTITION BY r.ranking_date), 1)) as growth_rank_norm,
                
                -- Composite scores
                r.final_composite_score / 100.0 as composite_norm,
                r.sector_neutral_score / 100.0 as sector_neutral_norm,
                r.overall_data_confidence / 100.0 as confidence_norm,
                
                -- Ranking momentum (weekly changes)
                r.beat_sp500_ranking - LAG(r.beat_sp500_ranking, 1) OVER (PARTITION BY r.symbol ORDER BY r.ranking_date) as sp500_momentum,
                r.growth_ranking - LAG(r.growth_ranking, 1) OVER (PARTITION BY r.symbol ORDER BY r.ranking_date) as growth_momentum,
                r.final_composite_score - LAG(r.final_composite_score, 1) OVER (PARTITION BY r.symbol ORDER BY r.ranking_date) as score_momentum,
                
                -- Multi-week momentum
                r.final_composite_score - LAG(r.final_composite_score, 4) OVER (PARTITION BY r.symbol ORDER BY r.ranking_date) as score_momentum_4w,
                r.final_composite_score - LAG(r.final_composite_score, 12) OVER (PARTITION BY r.symbol ORDER BY r.ranking_date) as score_momentum_12w,
                
                -- Moving averages of composite score
                AVG(r.final_composite_score) OVER (PARTITION BY r.symbol ORDER BY r.ranking_date ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) as score_ma_4w,
                AVG(r.final_composite_score) OVER (PARTITION BY r.symbol ORDER BY r.ranking_date ROWS BETWEEN 11 PRECEDING AND CURRENT ROW) as score_ma_12w,
                
                -- Volatility of rankings
                STDDEV(r.final_composite_score) OVER (PARTITION BY r.symbol ORDER BY r.ranking_date ROWS BETWEEN 11 PRECEDING AND CURRENT ROW) as score_volatility,
                
                -- Strategy flags as features
                r.is_all_star::int as is_all_star,
                r.is_growth_champion::int as is_growth_champion,
                r.is_deep_value_star::int as is_deep_value,
                r.is_momentum_breakout::int as is_momentum,
                
                -- Fundamental metrics
                COALESCE(r.fcf_yield, 0) as fcf_yield,
                COALESCE(r.revenue_growth_10yr, 0) as revenue_growth,
                COALESCE(r.sharpe_ratio_10yr, 0) as sharpe_ratio,
                COALESCE(r.outperformance_consistency, 0) as consistency,
                
                -- Market regime indicators
                AVG(r.final_composite_score) OVER (PARTITION BY r.ranking_date) as market_avg_score,
                STDDEV(r.final_composite_score) OVER (PARTITION BY r.ranking_date) as market_dispersion,
                COUNT(*) FILTER (WHERE r.sentiment_score > 0) OVER (PARTITION BY r.ranking_date) / 
                    NULLIF(COUNT(*) OVER (PARTITION BY r.ranking_date), 0)::float as market_bullish_pct,
                
                -- Sector relative performance
                r.final_composite_score - AVG(r.final_composite_score) OVER (PARTITION BY r.ranking_date, r.sector) as sector_relative,
                RANK() OVER (PARTITION BY r.ranking_date, r.sector ORDER BY r.final_composite_score DESC) as sector_rank,
                
                -- Size category encoding
                CASE r.size_category
                    WHEN 'Mega' THEN 5
                    WHEN 'Large' THEN 4
                    WHEN 'Mid' THEN 3
                    WHEN 'Small' THEN 2
                    WHEN 'Micro' THEN 1
                    ELSE 3
                END as size_code,
                
                -- Sector encoding (simplified)
                CASE 
                    WHEN r.sector = 'Technology' THEN 1
                    WHEN r.sector = 'Healthcare' THEN 2
                    WHEN r.sector = 'Financials' THEN 3
                    WHEN r.sector = 'Consumer Cyclical' THEN 4
                    WHEN r.sector = 'Industrials' THEN 5
                    WHEN r.sector = 'Consumer Defensive' THEN 6
                    WHEN r.sector = 'Energy' THEN 7
                    WHEN r.sector = 'Utilities' THEN 8
                    WHEN r.sector = 'Real Estate' THEN 9
                    WHEN r.sector = 'Materials' THEN 10
                    WHEN r.sector = 'Communication' THEN 11
                    ELSE 0
                END as sector_code
                
            FROM stock_quality_rankings r
            WHERE r.ranking_date >= COALESCE(:start_date, '2015-01-01')
              AND r.ranking_date <= COALESCE(:end_date, CURRENT_DATE)
        ),
        target_data AS (
            SELECT 
                fr.symbol,
                fr.ranking_date,
                fr.forward_return as return_1w,
                fr.forward_excess_return as excess_1w
            FROM ml_forward_returns fr
            WHERE fr.horizon_weeks = 1
        ),
        target_data_4w AS (
            SELECT 
                fr.symbol,
                fr.ranking_date,
                fr.forward_return as return_4w,
                fr.forward_excess_return as excess_4w
            FROM ml_forward_returns fr
            WHERE fr.horizon_weeks = 4
        )
        SELECT 
            fd.*,
            td.return_1w,
            td.excess_1w,
            t4.return_4w,
            t4.excess_4w,
            -- Binary classification targets
            CASE WHEN t4.return_4w > 0 THEN 1 ELSE 0 END as positive_return_4w,
            CASE WHEN t4.excess_4w > 0 THEN 1 ELSE 0 END as beat_market_4w,
            CASE WHEN t4.return_4w > 5 THEN 1 ELSE 0 END as strong_return_4w
        FROM feature_data fd
        LEFT JOIN target_data td ON fd.symbol = td.symbol AND fd.ranking_date = td.ranking_date
        LEFT JOIN target_data_4w t4 ON fd.symbol = t4.symbol AND fd.ranking_date = t4.ranking_date
        WHERE td.return_1w IS NOT NULL 
          AND t4.return_4w IS NOT NULL
        ORDER BY fd.ranking_date, fd.symbol
        """
        
        params = {
            'start_date': start_date or '2015-01-01',
            'end_date': end_date or datetime.now()
        }
        
        self.df = pd.read_sql_query(query, self.engine, params=params)
        
        # Handle missing values
        self.df = self.df.fillna(0)
        
        # Store feature columns
        self.feature_columns = [col for col in self.df.columns 
                                if col not in ['symbol', 'ranking_date', 'return_1w', 'excess_1w', 
                                             'return_4w', 'excess_4w', 'positive_return_4w', 
                                             'beat_market_4w', 'strong_return_4w']]
        
        logger.info(f"Loaded {len(self.df)} samples with {len(self.feature_columns)} features")
        
        return self.df
    
    def prepare_train_test_split(self, test_size=0.2, val_size=0.1):
        """
        Split data chronologically for time series validation
        """
        # Sort by date
        self.df = self.df.sort_values('ranking_date')
        
        # Calculate split points
        n = len(self.df)
        train_end = int(n * (1 - test_size - val_size))
        val_end = int(n * (1 - test_size))
        
        # Split data
        self.train_df = self.df.iloc[:train_end]
        self.val_df = self.df.iloc[train_end:val_end]
        self.test_df = self.df.iloc[val_end:]
        
        logger.info(f"Train: {len(self.train_df)}, Val: {len(self.val_df)}, Test: {len(self.test_df)}")
        
        # Prepare features and targets
        self.X_train = self.train_df[self.feature_columns]
        self.X_val = self.val_df[self.feature_columns]
        self.X_test = self.test_df[self.feature_columns]
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        return self.X_train_scaled, self.X_val_scaled, self.X_test_scaled
    
    def train_return_predictor(self, target='return_4w'):
        """
        Train models to predict future returns
        """
        logger.info(f"Training return predictor for {target}...")
        
        # Get target values
        y_train = self.train_df[target].values
        y_val = self.val_df[target].values
        y_test = self.test_df[target].values
        
        # Train multiple models
        models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boost': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        }
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Train model
            model.fit(self.X_train_scaled, y_train)
            
            # Predictions
            train_pred = model.predict(self.X_train_scaled)
            val_pred = model.predict(self.X_val_scaled)
            test_pred = model.predict(self.X_test_scaled)
            
            # Calculate metrics
            results[name] = {
                'model': model,
                'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
                'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
                'train_r2': r2_score(y_train, train_pred),
                'val_r2': r2_score(y_val, val_pred),
                'test_r2': r2_score(y_test, test_pred),
                'val_predictions': val_pred,
                'test_predictions': test_pred
            }
            
            # Feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': self.feature_columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                results[name]['feature_importance'] = importance_df
                
                logger.info(f"Top 10 features for {name}:")
                print(importance_df.head(10))
        
        # Store results
        self.models[target] = results
        
        # Select best model based on validation performance
        best_model = min(results.items(), key=lambda x: x[1]['val_rmse'])
        logger.info(f"Best model: {best_model[0]} with Val RMSE: {best_model[1]['val_rmse']:.4f}")
        
        return results
    
    def create_trading_signals(self, threshold=5.0, confidence_threshold=0.6):
        """
        Generate trading signals based on model predictions
        
        Args:
            threshold: Minimum predicted return to generate buy signal
            confidence_threshold: Minimum confidence for signal generation
        """
        logger.info("Generating trading signals...")
        
        # Use best model predictions
        best_model_name = 'xgboost'  # Or select dynamically
        model = self.models['return_4w'][best_model_name]['model']
        
        # Get latest rankings for signal generation
        query = """
        SELECT * FROM stock_quality_rankings
        WHERE ranking_date = (SELECT MAX(ranking_date) FROM stock_quality_rankings)
        """
        
        current_df = pd.read_sql_query(query, self.engine)
        
        # Prepare features
        X_current = current_df[self.feature_columns].fillna(0)
        X_current_scaled = self.scaler.transform(X_current)
        
        # Generate predictions
        predictions = model.predict(X_current_scaled)
        
        # Calculate prediction confidence (using ensemble disagreement)
        all_predictions = []
        for model_name, model_info in self.models['return_4w'].items():
            pred = model_info['model'].predict(X_current_scaled)
            all_predictions.append(pred)
        
        all_predictions = np.array(all_predictions)
        pred_std = np.std(all_predictions, axis=0)
        confidence = 1 / (1 + pred_std)  # Higher confidence when models agree
        
        # Create signals
        signals = pd.DataFrame({
            'symbol': current_df['symbol'],
            'predicted_return': predictions,
            'confidence': confidence,
            'composite_score': current_df['final_composite_score'],
            'signal_date': datetime.now().date()
        })
        
        # Generate buy signals
        signals['signal'] = 'HOLD'
        signals.loc[
            (signals['predicted_return'] > threshold) & 
            (signals['confidence'] > confidence_threshold) &
            (signals['composite_score'] > 60),
            'signal'
        ] = 'BUY'
        
        # Strong buy for top predictions
        signals.loc[
            (signals['predicted_return'] > threshold * 2) & 
            (signals['confidence'] > confidence_threshold * 1.2) &
            (signals['composite_score'] > 70),
            'signal'
        ] = 'STRONG_BUY'
        
        # Calculate position sizing based on Kelly Criterion
        signals['position_size'] = self.calculate_position_size(
            signals['predicted_return'], 
            signals['confidence']
        )
        
        # Sort by predicted return
        signals = signals.sort_values('predicted_return', ascending=False)
        
        return signals
    
    def calculate_position_size(self, predicted_return, confidence, max_position=0.1):
        """
        Calculate position size using modified Kelly Criterion
        """
        # Simplified Kelly: f = (p * b - q) / b
        # where p = probability of win, b = odds, q = probability of loss
        
        # Estimate win probability from confidence
        p = confidence
        q = 1 - p
        
        # Estimate odds from predicted return
        b = predicted_return / 10  # Normalize to reasonable range
        
        # Kelly fraction
        kelly = np.maximum(0, (p * b - q) / b)
        
        # Apply fractional Kelly (25%) for safety
        position_size = kelly * 0.25
        
        # Cap at maximum position size
        position_size = np.minimum(position_size, max_position)
        
        return position_size
    
    def backtest_strategy(self, initial_capital=100000, max_positions=20):
        """
        Backtest the ML strategy using historical predictions
        """
        logger.info("Running strategy backtest...")
        
        # Get test predictions
        test_predictions = self.models['return_4w']['xgboost']['test_predictions']
        test_df = self.test_df.copy()
        test_df['predicted_return'] = test_predictions
        
        # Initialize portfolio
        portfolio = {
            'cash': initial_capital,
            'positions': {},
            'history': []
        }
        
        # Group by date for chronological backtesting
        for date in test_df['ranking_date'].unique():
            date_df = test_df[test_df['ranking_date'] == date]
            
            # Generate signals for this date
            buy_signals = date_df[date_df['predicted_return'] > 5].nlargest(max_positions, 'predicted_return')
            
            # Rebalance portfolio
            if len(buy_signals) > 0:
                # Sell existing positions not in new signals
                for symbol in list(portfolio['positions'].keys()):
                    if symbol not in buy_signals['symbol'].values:
                        # Sell position
                        position = portfolio['positions'].pop(symbol)
                        portfolio['cash'] += position['value']
                
                # Buy new positions
                position_size = portfolio['cash'] / max(len(buy_signals), 1)
                for _, signal in buy_signals.iterrows():
                    if signal['symbol'] not in portfolio['positions']:
                        portfolio['positions'][signal['symbol']] = {
                            'shares': position_size,
                            'value': position_size,
                            'entry_date': date
                        }
                        portfolio['cash'] -= position_size
            
            # Update portfolio value based on actual returns
            portfolio_value = portfolio['cash']
            for symbol, position in portfolio['positions'].items():
                # Get actual return for this position
                actual_return = date_df[date_df['symbol'] == symbol]['return_4w'].values
                if len(actual_return) > 0:
                    position['value'] = position['shares'] * (1 + actual_return[0] / 100)
                portfolio_value += position['value']
            
            # Record history
            portfolio['history'].append({
                'date': date,
                'value': portfolio_value,
                'num_positions': len(portfolio['positions']),
                'cash': portfolio['cash']
            })
        
        # Calculate performance metrics
        history_df = pd.DataFrame(portfolio['history'])
        
        if len(history_df) > 1:
            total_return = (history_df['value'].iloc[-1] / initial_capital - 1) * 100
            
            # Calculate Sharpe ratio
            history_df['returns'] = history_df['value'].pct_change()
            sharpe = history_df['returns'].mean() / history_df['returns'].std() * np.sqrt(52)  # Weekly
            
            # Calculate max drawdown
            cumulative = (1 + history_df['returns']).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min() * 100
            
            self.performance_metrics = {
                'total_return': total_return,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown,
                'final_value': history_df['value'].iloc[-1],
                'num_trades': len(portfolio['history'])
            }
            
            logger.info(f"Backtest Results:")
            logger.info(f"  Total Return: {total_return:.2f}%")
            logger.info(f"  Sharpe Ratio: {sharpe:.2f}")
            logger.info(f"  Max Drawdown: {max_drawdown:.2f}%")
            logger.info(f"  Final Value: ${history_df['value'].iloc[-1]:,.2f}")
        
        return history_df, self.performance_metrics
    
    def save_model(self, model_name=None):
        """Save trained models and metadata to disk"""
        if not model_name:
            model_name = self.model_name
            
        # Save models
        for target, models in self.models.items():
            for name, model_info in models.items():
                filename = f"models/{model_name}_{target}_{name}.pkl"
                joblib.dump(model_info['model'], filename)
                logger.info(f"Saved model to {filename}")
        
        # Save scaler
        joblib.dump(self.scaler, f"models/{model_name}_scaler.pkl")
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'feature_columns': self.feature_columns,
            'performance_metrics': self.performance_metrics,
            'training_date': datetime.now().isoformat()
        }
        
        pd.DataFrame([metadata]).to_json(f"models/{model_name}_metadata.json")
        logger.info(f"Model saved successfully")


def main():
    """Main execution for ML strategy development"""
    
    # Initialize framework
    ml_framework = MLStrategyFramework(model_name="quality_ranking_predictor_v1")
    
    # Load training data
    ml_framework.load_training_data(start_date='2015-01-01')
    
    # Prepare train/test split
    ml_framework.prepare_train_test_split(test_size=0.2, val_size=0.1)
    
    # Train models
    ml_framework.train_return_predictor(target='return_4w')
    ml_framework.train_return_predictor(target='excess_4w')
    
    # Generate current signals
    signals = ml_framework.create_trading_signals(threshold=5.0)
    
    print("\n" + "="*60)
    print("TOP BUY SIGNALS")
    print("="*60)
    print(signals[signals['signal'].isin(['BUY', 'STRONG_BUY'])].head(20))
    
    # Run backtest
    backtest_results, performance = ml_framework.backtest_strategy()
    
    print("\n" + "="*60)
    print("BACKTEST PERFORMANCE")
    print("="*60)
    for metric, value in performance.items():
        print(f"{metric}: {value:.2f}")
    
    # Save model
    ml_framework.save_model()


if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)
    main()