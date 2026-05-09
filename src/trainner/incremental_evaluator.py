import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from src.utils.config import get_trainner_config

class IncrementalEvaluator:
    """
    Evaluates a model using Incremental Learning.
    Instead of retraining from scratch, it adds new trees to the existing ensemble.
    """
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y
        
        # Load configs
        trainner_config = get_trainner_config()
        self.pipeline_config = trainner_config.get("pipeline", {})
        self.incremental_config = trainner_config.get("incremental", {})
        
        self.update_interval = self.incremental_config.get("update_interval", 1000)
        self.update_trees = self.incremental_config.get("update_trees", 50)
        self.steps_since_last_update = 0

    def init_train(self, initial_train_size=None):
        """Initial bootstrap training with the full set of trees."""
        if initial_train_size is None:
            initial_train_size = self.pipeline_config.get("init_train_size", 1000)
            
        print(f"Initial training on {initial_train_size} samples...")
        if hasattr(self.X, 'iloc'):
            X_train_init = self.X.iloc[:initial_train_size]
        else:
            X_train_init = self.X[:initial_train_size]
            
        y_train_init = self.y[:initial_train_size]
        self.model.fit(X_train_init, y_train_init)
        
        self.steps_since_last_update = 0
        return initial_train_size

    def predict_step(self, i):
        """Predicts for a single step at index i."""
        if hasattr(self.X, 'iloc'):
            X_test = self.X.iloc[i:i+1]
        else:
            X_test = self.X[i:i+1]
        
        y_true = self.y[i]
        
        y_prob = self.model.predict_proba(X_test)[0][1]
        y_pred = self.model.predict(X_test)[0]
        
        return y_true, y_pred, y_prob

    def check_drift(self, is_correct):
        """Checks if the update interval has been reached."""
        self.steps_since_last_update += 1
        
        if self.steps_since_last_update >= self.update_interval:
            return True
        return False

    def retrain(self, i, retraining_start_idx):
        """Handles incremental update when the interval is reached."""
        # Only take the new data batch
        update_start_idx = retraining_start_idx
        
        if hasattr(self.X, 'iloc'):
            X_new = self.X.iloc[update_start_idx:i+1]
        else:
            X_new = self.X[update_start_idx:i+1]
            
        y_new = self.y[update_start_idx:i+1]
        
        self._update_model(X_new, y_new)
        
        # Reset interval counter
        self.steps_since_last_update = 0
        
        # The next update starts from current index + 1
        return update_start_idx

    def _update_model(self, X_new, y_new):
        """Performs library-specific incremental update with class consistency check."""
        
        # Ensure y_new contains both classes to avoid scikit-learn LabelEncoder issues
        # (Especially common in imbalanced fraud datasets)
        unique_classes = np.unique(y_new)
        if len(unique_classes) < 2:
            missing_class = 1 - unique_classes[0]
            # Find indices of the missing class in the historical data (up to current index)
            hist_y = self.y[:len(self.X)] # Simple slice
            missing_indices = np.where(hist_y == missing_class)[0]
            
            if len(missing_indices) > 0:
                # Take up to 5 samples of the missing class to stabilize the update
                extra_idx = missing_indices[:5]
                if hasattr(self.X, 'iloc'):
                    X_extra = self.X.iloc[extra_idx]
                else:
                    X_extra = self.X[extra_idx]
                y_extra = self.y[extra_idx]
                
                # Append to current batch
                if isinstance(X_new, pd.DataFrame):
                    X_new = pd.concat([X_new, X_extra])
                    y_new = np.concatenate([y_new, y_extra])
                else:
                    X_new = np.concatenate([X_new, X_extra])
                    y_new = np.concatenate([y_new, y_extra])

        # Save original estimator count to restore later
        if isinstance(self.model, xgb.XGBClassifier):
            orig_estimators = self.model.n_estimators
            self.model.n_estimators = self.update_trees
            self.model.fit(X_new, y_new, xgb_model=self.model.get_booster())
            self.model.n_estimators = orig_estimators
            
        elif isinstance(self.model, lgb.LGBMClassifier):
            orig_estimators = self.model.n_estimators
            self.model.n_estimators = self.update_trees
            self.model.fit(X_new, y_new, init_model=self.model.booster_)
            self.model.n_estimators = orig_estimators
            
        elif isinstance(self.model, CatBoostClassifier):
            orig_params = self.model.get_params()
            orig_iterations = orig_params.get('iterations')
            self.model.set_params(iterations=self.update_trees)
            self.model.fit(X_new, y_new, init_model=self.model)
            self.model.set_params(iterations=orig_iterations)
