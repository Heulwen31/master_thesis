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

    def check_drift(self, is_correct):
        """Updates ADDM and checks for drift."""
        return self.addm.add_result(is_correct)

    def retrain(self, i, retraining_start_idx):
        """Handles full retraining when drift is detected."""
        retraining_point = self.addm.seek_retraining_timestamp(i)
        retraining_point = max(retraining_start_idx, retraining_point)
        
        if hasattr(self.X, 'iloc'):
            X_retrain = self.X.iloc[retraining_point:i+1]
        else:
            X_retrain = self.X[retraining_point:i+1]
            
        y_retrain = self.y[retraining_point:i+1]
        self._fit_model(X_retrain, y_retrain)
        self.addm.reset()
        return retraining_point
