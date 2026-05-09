import numpy as np
from src.trainner.addm import ADDM
from src.utils.config import get_trainner_config

class OnlineEvaluator:
    """
    Provides modular methods for online evaluation (prediction, drift detection, retraining).
    The main loop should be handled by an external runner (e.g., main.py).
    """
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y
        
        # Load configs
        trainner_config = get_trainner_config()
        self.pipeline_config = trainner_config.get("pipeline", {})
        addm_config = trainner_config.get("addm", {})
        
        self.addm = ADDM(
            hoeffding_delta=addm_config.get("hoeffding_delta", 0.01),
            min_window_size=addm_config.get("min_window_size", 30),
            entropy_threshold=addm_config.get("entropy_threshold", 0.99)
        )

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
        """Predicts for a single step at index i."""
        if hasattr(self.X, 'iloc'):
            X_test = self.X.iloc[i:i+1]
        else:
            X_test = self.X[i:i+1]
        
        y_true = self.y[i]
        
        # Get probability for ROC-AUC/PR-AUC
        y_prob = self.model.predict_proba(X_test)[0][1]
        y_pred = self.model.predict(X_test)[0]
        
        return y_true, y_pred, y_prob

    def check_drift(self, is_correct):
        """Updates ADDM and checks for drift."""
        return self.addm.add_result(is_correct)

    def retrain(self, i, retraining_start_idx):
        """Handles retraining when drift is detected."""
        retraining_point = self.addm.seek_retraining_timestamp(i)
        # Ensure we don't go back before the last retraining point
        retraining_point = max(retraining_start_idx, retraining_point)
        
        if hasattr(self.X, 'iloc'):
            X_retrain = self.X.iloc[retraining_point:i+1]
        else:
            X_retrain = self.X[retraining_point:i+1]
            
        y_retrain = self.y[retraining_point:i+1]
        self._fit_model(X_retrain, y_retrain)
        
        # Reset ADDM
        self.addm.reset()
        return retraining_point

    def _fit_model(self, X, y):
        """Internal model fit."""
        self.model.fit(X, y)
