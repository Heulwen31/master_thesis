from src.trainner.base_evaluator import BaseEvaluator

class IncrementalEvaluator(BaseEvaluator):
    """
    Evaluates a model using Incremental Learning.
    """
    def __init__(self, model, X, y):
        super().__init__(model, X, y)
        incremental_config = self.trainner_config.get("incremental", {})
        self.update_interval = incremental_config.get("update_interval", 1000)
        self.update_trees = incremental_config.get("update_trees", 50)
        self.steps_since_last_update = 0

    def check_drift(self, is_correct):
        """Checks if the update interval has been reached."""
        self.steps_since_last_update += 1
        return self.steps_since_last_update >= self.update_interval

    def retrain(self, i, retraining_start_idx):
        """Handles incremental update with the new batch."""
        update_start_idx = retraining_start_idx
        if hasattr(self.X, 'iloc'):
            X_new = self.X.iloc[update_start_idx:i+1]
        else:
            X_new = self.X[update_start_idx:i+1]
            
        y_new = self.y[update_start_idx:i+1]
        self._update_model(X_new, y_new, update_trees=self.update_trees)
        self.steps_since_last_update = 0
        return update_start_idx
