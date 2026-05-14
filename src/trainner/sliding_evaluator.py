from src.trainner.base_evaluator import BaseEvaluator

class SlidingEvaluator(BaseEvaluator):
    """
    Evaluates a model using a fixed-size sliding window for periodic retraining.
    """
    def __init__(self, model, X, y):
        super().__init__(model, X, y)
        sliding_config = self.trainner_config.get("sliding", {})
        self.window_size = sliding_config.get("window_size", 1000)
        self.retrain_interval = sliding_config.get("retrain_interval", 200)
        self.steps_since_retrain = 0

    def check_drift(self, is_correct):
        """Checks if it's time for periodic retraining."""
        self.steps_since_retrain += 1
        if self.steps_since_retrain >= self.retrain_interval:
            return True
        return False

    def retrain(self, i, retraining_start_idx):
        """Handles full retraining using a fixed window size."""
        # Use last window_size samples
        retraining_point = max(0, i - self.window_size + 1)
        
        if hasattr(self.X, 'iloc'):
            X_retrain = self.X.iloc[retraining_point:i+1]
        else:
            X_retrain = self.X[retraining_point:i+1]
            
        y_retrain = self.y[retraining_point:i+1]
        self._fit_model(X_retrain, y_retrain)
        
        self.steps_since_retrain = 0
        return retraining_point
