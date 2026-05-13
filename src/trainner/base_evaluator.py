import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from src.utils.config import get_trainner_config

class BaseEvaluator:
    """
    Base class for all evaluators to provide shared logic for prediction and training.
    """
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y
        self.trainner_config = get_trainner_config()
        self.pipeline_config = self.trainner_config.get("pipeline", {})

    def init_train(self, initial_train_size=None):
        """Initial bootstrap training."""
        if initial_train_size is None:
            initial_train_size = self.pipeline_config.get("init_train_size", 1000)
            
        print(f"Initial training on {initial_train_size} samples...")
        if hasattr(self.X, 'iloc'):
            X_train_init = self.X.iloc[:initial_train_size]
        else:
            X_train_init = self.X[:initial_train_size]
            
        y_train_init = self.y[:initial_train_size]
        self._fit_model(X_train_init, y_train_init)
        return initial_train_size

    def predict_step(self, i):
        """Standard prediction step returning (y_true, y_pred, y_prob)."""
        if hasattr(self.X, 'iloc'):
            X_test = self.X.iloc[i:i+1]
        else:
            X_test = self.X[i:i+1]
        
        y_true = self.y[i]
        y_prob = self.model.predict_proba(X_test)[0][1]
        y_pred = self.model.predict(X_test)[0]
        
        return y_true, y_pred, y_prob

    def _fit_model(self, X, y):
        """Standard full model fit with support for categorical features."""
        from catboost import CatBoostClassifier
        X, y = self._ensure_two_classes(X, y)
        if isinstance(self.model, CatBoostClassifier):
            cat_features = list(X.select_dtypes(include=['category']).columns)
            self.model.fit(X, y, cat_features=cat_features)
        else:
            self.model.fit(X, y)

    def _ensure_two_classes(self, X_batch, y_batch, max_extra_samples=5):
        """Adds a few historical samples if a retraining batch has only one class."""
        unique_classes = np.unique(y_batch)
        if len(unique_classes) >= 2:
            return X_batch, y_batch

        missing_class = 1 - unique_classes[0]
        missing_indices = np.where(self.y == missing_class)[0]
        if len(missing_indices) == 0:
            return X_batch, y_batch

        end_pos = self._infer_batch_end_position(X_batch)
        if end_pos is not None:
            historical_missing_indices = missing_indices[missing_indices <= end_pos]
            if len(historical_missing_indices) > 0:
                missing_indices = historical_missing_indices

        extra_idx = missing_indices[:max_extra_samples]
        if hasattr(self.X, 'iloc'):
            X_extra = self.X.iloc[extra_idx]
        else:
            X_extra = self.X[extra_idx]
        y_extra = self.y[extra_idx]

        if isinstance(X_batch, pd.DataFrame):
            X_batch = pd.concat([X_batch, X_extra])
            y_batch = np.concatenate([y_batch, y_extra])
        else:
            X_batch = np.concatenate([X_batch, X_extra])
            y_batch = np.concatenate([y_batch, y_extra])

        return X_batch, y_batch

    def _infer_batch_end_position(self, X_batch):
        """Infers the last positional index of a batch sliced from self.X."""
        if not hasattr(X_batch, 'index') or not hasattr(self.X, 'index') or len(X_batch) == 0:
            return None

        last_label = X_batch.index[-1]
        try:
            loc = self.X.index.get_loc(last_label)
        except KeyError:
            return None

        if isinstance(loc, slice):
            return loc.stop - 1
        if isinstance(loc, np.ndarray):
            positions = np.where(loc)[0] if loc.dtype == bool else loc
            return int(positions[-1]) if len(positions) > 0 else None
        return int(loc)

    def _update_model(self, X_new, y_new, update_trees=50):
        """Performs library-specific incremental update with class consistency check."""
        
        # Ensure y_new contains both classes to avoid classifier label issues.
        X_new, y_new = self._ensure_two_classes(X_new, y_new)

        # Save original estimator count to restore later
        if isinstance(self.model, xgb.XGBClassifier):
            orig_estimators = self.model.n_estimators
            self.model.n_estimators = update_trees
            self.model.fit(X_new, y_new, xgb_model=self.model.get_booster())
            self.model.n_estimators = orig_estimators
            
        elif isinstance(self.model, lgb.LGBMClassifier):
            orig_estimators = self.model.n_estimators
            self.model.n_estimators = update_trees
            self.model.fit(X_new, y_new, init_model=self.model.booster_)
            self.model.n_estimators = orig_estimators
            
        elif isinstance(self.model, CatBoostClassifier):
            # CatBoost does not allow set_params on a fitted model.
            # We create a new instance with updated iterations for the incremental step.
            params = self.model.get_params()
            params['iterations'] = update_trees
            
            # Detect categorical columns
            cat_features = list(X_new.select_dtypes(include=['category']).columns)
            
            new_model = CatBoostClassifier(**params)
            new_model.fit(X_new, y_new, init_model=self.model, cat_features=cat_features)
            self.model = new_model
        
        else:
            # Fallback to standard fit if not supported
            self.model.fit(X_new, y_new)
