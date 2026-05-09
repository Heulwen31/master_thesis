import numpy as np
from src.utils.config import get_trainner_config

class PeriodicEvaluator:
    """
    Evaluates a model using a fixed-interval periodic retraining baseline.
    Retrains on all historical data every N transactions.
    """
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y
        
        # Load configs
        trainner_config = get_trainner_config()
        self.pipeline_config = trainner_config.get("pipeline", {})
        periodic_config = trainner_config.get("periodic", {})
        
        self.retrain_interval = periodic_config.get("retrain_interval", 5000)
        self.steps_since_last_retrain = 0

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
        
        self.steps_since_last_retrain = 0
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
        """Checks if the periodic interval has been reached."""
        self.steps_since_last_retrain += 1
        
        if self.steps_since_last_retrain >= self.retrain_interval:
            return True
        return False

    def retrain(self, i, retraining_start_idx):
        """Handles retraining when the periodic interval is reached."""
        # For periodic baseline, retraining point is usually from the very beginning (index 0)
        # to simulate accumulating historical data.
        retraining_point = 0 
        
        if hasattr(self.X, 'iloc'):
            X_retrain = self.X.iloc[retraining_point:i+1]
        else:
            X_retrain = self.X[retraining_point:i+1]
            
        y_retrain = self.y[retraining_point:i+1]
        self._fit_model(X_retrain, y_retrain)
        
        # Reset interval counter
        self.steps_since_last_retrain = 0
        
        return retraining_point

    def _fit_model(self, X, y):
        """Internal model fit."""
        self.model.fit(X, y)
