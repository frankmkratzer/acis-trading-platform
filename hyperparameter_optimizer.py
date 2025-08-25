# =====================================
# 1. HYPERPARAMETER OPTIMIZATION ENGINE
# =====================================
"""
#!/usr/bin/env python3
# File: hyperparameter_optimizer.py
# Purpose: Advanced hyperparameter optimization with multiple methods
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import optuna
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import joblib
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Callable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """Advanced hyperparameter optimization for trading models"""

    def __init__(self, model_type: str = 'gradient_boosting'):
        self.model_type = model_type
        self.best_params = None
        self.best_score = None
        self.optimization_history = []

    def optimize_with_optuna(self, X_train, y_train, n_trials: int = 100) -> Dict:
        """Optimize using Optuna (Bayesian optimization)"""

        def objective(trial):
            if self.model_type == 'gradient_boosting':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'min_samples_split': trial.suggest_int('min_samples_split', 10, 100),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 50),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'random_state': 42
                }
                model = GradientBoostingRegressor(**params)

            elif self.model_type == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 5, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 50),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.7]),
                    'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                    'random_state': 42
                }
                model = RandomForestRegressor(**params)

            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            scores = cross_val_score(
                model, X_train, y_train,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )

            return -scores.mean()  # Minimize negative MSE

        # Create study
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )

        # Optimize
        study.optimize(objective, n_trials=n_trials, n_jobs=1)

        self.best_params = study.best_params
        self.best_score = study.best_value

        # Save optimization history
        self.optimization_history = [
            {
                'trial': t.number,
                'value': t.value,
                'params': t.params
            }
            for t in study.trials
        ]

        logger.info(f"Optuna optimization complete. Best score: {self.best_score:.6f}")
        return self.best_params

    def optimize_with_hyperopt(self, X_train, y_train, max_evals: int = 100) -> Dict:
        """Optimize using Hyperopt (Tree-structured Parzen Estimator)"""

        if self.model_type == 'gradient_boosting':
            space = {
                'n_estimators': hp.choice('n_estimators', range(50, 500, 50)),
                'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
                'max_depth': hp.choice('max_depth', range(3, 11)),
                'min_samples_split': hp.choice('min_samples_split', range(10, 101, 10)),
                'min_samples_leaf': hp.choice('min_samples_leaf', range(5, 51, 5)),
                'subsample': hp.uniform('subsample', 0.5, 1.0),
                'max_features': hp.choice('max_features', ['sqrt', 'log2', None])
            }

        def objective(params):
            model = GradientBoostingRegressor(
                **params,
                random_state=42
            )

            tscv = TimeSeriesSplit(n_splits=5)
            scores = cross_val_score(
                model, X_train, y_train,
                cv=tscv,
                scoring='neg_mean_squared_error'
            )

            return {'loss': -scores.mean(), 'status': STATUS_OK}

        trials = Trials()
        best = fmin(
            objective,
            space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials
        )

        self.best_params = best
        self.best_score = min(trials.losses())

        logger.info(f"Hyperopt optimization complete. Best score: {self.best_score:.6f}")
        return best

    def optimize_with_bayesian(self, X_train, y_train, n_iter: int = 50) -> Dict:
        """Optimize using Scikit-Optimize (Gaussian Process)"""

        if self.model_type == 'gradient_boosting':
            search_space = {
                'n_estimators': Integer(50, 500),
                'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'max_depth': Integer(3, 10),
                'min_samples_split': Integer(10, 100),
                'min_samples_leaf': Integer(5, 50),
                'subsample': Real(0.5, 1.0),
                'max_features': Categorical(['sqrt', 'log2', None])
            }

            model = GradientBoostingRegressor(random_state=42)

        bayes_search = BayesSearchCV(
            model,
            search_space,
            n_iter=n_iter,
            cv=TimeSeriesSplit(n_splits=5),
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            random_state=42
        )

        bayes_search.fit(X_train, y_train)

        self.best_params = bayes_search.best_params_
        self.best_score = bayes_search.best_score_

        logger.info(f"Bayesian optimization complete. Best score: {self.best_score:.6f}")
        return self.best_params

    def walk_forward_optimization(self,
                                  X: np.array,
                                  y: np.array,
                                  window_size: int = 252,
                                  step_size: int = 21,
                                  optimization_method: str = 'optuna') -> List[Dict]:
        """Walk-forward optimization for robust parameter selection"""

        results = []
        n_samples = len(X)

        for start_idx in range(0, n_samples - window_size, step_size):
            end_idx = start_idx + window_size

            # Training window
            X_train = X[start_idx:end_idx]
            y_train = y[start_idx:end_idx]

            # Test window (next step_size samples)
            test_end = min(end_idx + step_size, n_samples)
            X_test = X[end_idx:test_end]
            y_test = y[end_idx:test_end]

            # Optimize on training window
            if optimization_method == 'optuna':
                params = self.optimize_with_optuna(X_train, y_train, n_trials=30)
            elif optimization_method == 'hyperopt':
                params = self.optimize_with_hyperopt(X_train, y_train, max_evals=30)
            else:
                params = self.optimize_with_bayesian(X_train, y_train, n_iter=20)

            # Evaluate on test window
            if self.model_type == 'gradient_boosting':
                model = GradientBoostingRegressor(**params, random_state=42)
            else:
                model = RandomForestRegressor(**params, random_state=42)

            model.fit(X_train, y_train)
            test_score = mean_squared_error(y_test, model.predict(X_test))

            results.append({
                'window_start': start_idx,
                'window_end': end_idx,
                'params': params,
                'test_score': test_score
            })

            logger.info(f"Window {start_idx}-{end_idx}: Test MSE = {test_score:.6f}")

        return results