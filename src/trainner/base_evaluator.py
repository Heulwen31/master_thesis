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
        """Standard full model fit."""
        self.model.fit(X, y)

    def _update_model(self, X_new, y_new, update_trees=50):
        """Performs library-specific incremental update with class consistency check."""
        
        # Ensure y_new contains both classes to avoid scikit-learn LabelEncoder issues
        unique_classes = np.unique(y_new)
        if len(unique_classes) < 2:
            missing_class = 1 - unique_classes[0]
            # Find indices of the missing class in historical data
            hist_y = self.y[:len(self.X)]
            missing_indices = np.where(hist_y == missing_class)[0]
            
            if len(missing_indices) > 0:
                # Take up to 5 samples of the missing class
                extra_idx = missing_indices[:5]
                if hasattr(self.X, 'iloc'):
                    X_extra = self.X.iloc[extra_idx]
                else:
                    X_extra = self.X[extra_idx]
                y_extra = self.y[extra_idx]
                
                if isinstance(X_new, pd.DataFrame):
                    X_new = pd.concat([X_new, X_extra])
                    y_new = np.concatenate([y_new, y_extra])
                else:
                    X_new = np.concatenate([X_new, X_extra])
                    y_new = np.concatenate([y_new, y_extra])

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
            orig_params = self.model.get_params()
            orig_iterations = orig_params.get('iterations')
            self.model.set_params(iterations=update_trees)
            self.model.fit(X_new, y_new, init_model=self.model)
            self.model.set_params(iterations=orig_iterations)
        
        else:
            # Fallback to standard fit if not supported
            self.model.fit(X_new, y_new)
