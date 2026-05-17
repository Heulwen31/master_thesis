from src.trainner.base_evaluator import BaseEvaluator
from src.trainner.periodic_evaluator import PeriodicEvaluator

class HybridEvaluator(BaseEvaluator):
    """
    Hybrid method that currently only uses Periodic evaluation logic.
    """
    def __init__(self, model, X, y):
        super().__init__(model, X, y)
        self.periodic_eval = PeriodicEvaluator(model, X, y)
        
    def init_train(self, initial_train_size=None):
        return self.periodic_eval.init_train(initial_train_size)

    def predict_step(self, i):
        return self.periodic_eval.predict_step(i)

    def check_drift(self, is_correct):
        return self.periodic_eval.check_drift(is_correct)

    def retrain(self, i, retraining_start_idx):
        return self.periodic_eval.retrain(i, retraining_start_idx)
