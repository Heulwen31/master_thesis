from src.trainner.addm import ADDM
from src.trainner.base_evaluator import BaseEvaluator

class OnlineEvaluator(BaseEvaluator):
    """
    Evaluates a model using the ADDM (Adaptive sliding window based Drift Detection Method).
    """
    def __init__(self, model, X, y):
        super().__init__(model, X, y)
        addm_config = self.trainner_config.get("addm", {})
        self.addm = ADDM(
            hoeffding_delta=addm_config.get("hoeffding_delta", 0.01),
            min_window_size=addm_config.get("min_window_size", 30),
            entropy_threshold=addm_config.get("entropy_threshold", 1.0)
        )
        self.fallback_interval = addm_config.get("fallback_interval", 100)
        self.fallback_window = addm_config.get("fallback_window", "adaptive")
        self.fixed_train_size = addm_config.get("fixed_train_size", 1000)
        self.min_train_size = addm_config.get("min_train_size", 500)
        self.max_train_size = addm_config.get("max_train_size", 5000)
        self.steps_since_retrain = 0
        self.trigger_reason = None

    def check_drift(self, is_correct):
        """Updates ADDM and checks for drift."""
        drift_triggered = self.addm.add_result(is_correct)
        self.steps_since_retrain += 1

        if drift_triggered:
            self.trigger_reason = "drift"
            return True

        fallback_triggered = (
            self.fallback_interval > 0
            and self.steps_since_retrain >= self.fallback_interval
        )
        if fallback_triggered:
            self.trigger_reason = "fallback"
            return True

        return False

    def retrain(self, i, retraining_start_idx):
        """Handles full retraining when drift is detected."""
        if self.trigger_reason == "fallback":
            retraining_point = self._fallback_retraining_point(i)
        else:
            retraining_point = self.addm.seek_retraining_timestamp(i)
            retraining_point = max(retraining_start_idx, retraining_point)
        
        if hasattr(self.X, 'iloc'):
            X_retrain = self.X.iloc[retraining_point:i+1]
        else:
            X_retrain = self.X[retraining_point:i+1]
            
        y_retrain = self.y[retraining_point:i+1]
        self._fit_model(X_retrain, y_retrain)
        self.addm.reset()
        self.steps_since_retrain = 0
        self.trigger_reason = None
        return retraining_point

    def _fallback_retraining_point(self, i):
        """Returns a bounded recent retraining window for periodic fallback."""
        if self.fallback_window == "fixed":
            window_size = self.fixed_train_size
        else:
            window_size = max(self.min_train_size, len(self.addm.window))
            window_size = min(window_size, self.max_train_size)

        return max(0, i - window_size + 1)
