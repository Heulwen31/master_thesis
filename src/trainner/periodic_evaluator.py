from src.trainner.base_evaluator import BaseEvaluator

class PeriodicEvaluator(BaseEvaluator):
    """
    Evaluates a model using a fixed-interval periodic retraining baseline.
    """
    def __init__(self, model, X, y):
        super().__init__(model, X, y)
        periodic_config = self.trainner_config.get("periodic", {})
        self.retrain_interval = periodic_config.get("retrain_interval", 5000)
        self.steps_since_last_retrain = 0

    def check_drift(self, is_correct):
        """Checks if the periodic interval has been reached."""
        self.steps_since_last_retrain += 1
        return self.steps_since_last_retrain >= self.retrain_interval

    def retrain(self, i, retraining_start_idx):
        """Handles full retraining from scratch."""
        retraining_point = 0 
        if hasattr(self.X, 'iloc'):
            X_retrain = self.X.iloc[retraining_point:i+1]
        else:
            X_retrain = self.X[retraining_point:i+1]
            
        y_retrain = self.y[retraining_point:i+1]
        self._fit_model(X_retrain, y_retrain)
        self.steps_since_last_retrain = 0
        return retraining_point
