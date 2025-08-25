# =====================================
# 2. ENSEMBLE MODEL BUILDER
# =====================================
"""
#!/usr/bin/env python3
# File: ensemble_models.py
# Purpose: Build ensemble models for better predictions
"""


class EnsembleModelBuilder:
    """Build and manage ensemble models"""

    def __init__(self):
        self.models = []
        self.weights = []
        self.meta_model = None

    def create_diverse_ensemble(self, X_train, y_train) -> List:
        """Create ensemble with diverse models"""

        models = [
            # Different algorithms
            ('gb_1', GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
            )),
            ('gb_2', GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=7, random_state=43
            )),
            ('rf_1', RandomForestRegressor(
                n_estimators=200, max_depth=10, random_state=42
            )),
            ('rf_2', RandomForestRegressor(
                n_estimators=300, max_depth=15, max_features='sqrt', random_state=43
            )),
        ]

        # Add XGBoost if available
        try:
            import xgboost as xgb
            models.append(('xgb', xgb.XGBRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
            )))
        except ImportError:
            pass

        # Add LightGBM if available
        try:
            import lightgbm as lgb
            models.append(('lgb', lgb.LGBMRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
            )))
        except ImportError:
            pass

        # Train all models
        trained_models = []
        for name, model in models:
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            trained_models.append((name, model))

        self.models = trained_models
        return trained_models

    def create_stacking_ensemble(self, X_train, y_train, X_val, y_val):
        """Create stacking ensemble with meta-learner"""

        # Train base models
        base_models = self.create_diverse_ensemble(X_train, y_train)

        # Generate predictions for meta-model training
        meta_features = []
        for name, model in base_models:
            pred = model.predict(X_val)
            meta_features.append(pred)

        meta_X = np.column_stack(meta_features)

        # Train meta-model (simple linear regression or neural network)
        from sklearn.linear_model import LinearRegression, Ridge

        self.meta_model = Ridge(alpha=1.0)
        self.meta_model.fit(meta_X, y_val)

        logger.info("Stacking ensemble created")
        return self

    def create_blending_ensemble(self, X_train, y_train, X_val, y_val):
        """Create blending ensemble with optimized weights"""

        # Train base models
        base_models = self.create_diverse_ensemble(X_train, y_train)

        # Get validation predictions
        val_preds = []
        for name, model in base_models:
            pred = model.predict(X_val)
            val_preds.append(pred)

        # Optimize blending weights
        def objective(weights):
            # Normalize weights
            weights = weights / weights.sum()

            # Weighted average prediction
            blended = np.zeros_like(val_preds[0])
            for i, pred in enumerate(val_preds):
                blended += weights[i] * pred

            # MSE
            return mean_squared_error(y_val, blended)

        from scipy.optimize import minimize

        n_models = len(base_models)
        initial_weights = np.ones(n_models) / n_models

        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=[(0, 1) for _ in range(n_models)],
            constraints={'type': 'eq', 'fun': lambda x: x.sum() - 1}
        )

        self.weights = result.x
        logger.info(f"Optimal blending weights: {self.weights}")

        return self

    def predict(self, X):
        """Make predictions with ensemble"""

        if self.meta_model is not None:
            # Stacking prediction
            meta_features = []
            for name, model in self.models:
                pred = model.predict(X)
                meta_features.append(pred)

            meta_X = np.column_stack(meta_features)
            return self.meta_model.predict(meta_X)

        elif len(self.weights) > 0:
            # Blending prediction
            predictions = []
            for name, model in self.models:
                pred = model.predict(X)
                predictions.append(pred)

            blended = np.zeros_like(predictions[0])
            for i, pred in enumerate(predictions):
                blended += self.weights[i] * pred

            return blended

        else:
            # Simple average
            predictions = []
            for name, model in self.models:
                pred = model.predict(X)
                predictions.append(pred)

            return np.mean(predictions, axis=0)
