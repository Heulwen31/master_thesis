import numpy as np
import pandas as pd
from src.trainner.base_evaluator import BaseEvaluator


class HybridEvaluator(BaseEvaluator):
    """
    Hybrid Evaluator configured to only use Periodic retraining.
    
    Eliminates all incremental updates and gating logic, acting as a clean
    Periodic retraining pipeline under the Hybrid name.
    """
    def __init__(self, model, X, y):
        super().__init__(model, X, y)
        
        # Load periodic interval from configs
        periodic_config = self.trainner_config.get("periodic", {})
        self.retrain_interval = periodic_config.get("retrain_interval", 1000)
        
        # State tracking
        self.steps_since_last_retrain = 0
        self.active_triggers = []
        
        # Statistics
        self.count_periodic = 0

    def init_train(self, initial_train_size=None):
        """Bootstrap training for the single model."""
        if initial_train_size is None:
            initial_train_size = self.pipeline_config.get("init_train_size", 1000)
        
        if hasattr(self.X, 'iloc'):
            X_train = self.X.iloc[:initial_train_size]
        else:
            X_train = self.X[:initial_train_size]
        y_train = self.y[:initial_train_size]
        
        self._fit_model(X_train, y_train)
        self.steps_since_last_retrain = 0
        
        return initial_train_size

    def predict_step(self, i):
        """Single-model prediction directly."""
        if hasattr(self.X, 'iloc'):
            X_test_df = self.X.iloc[i:i+1]
        else:
            X_test_df = self.X[i:i+1]
            
        y_true = self.y[i]
        
        # Direct prediction from our single model
        proba = self.model.predict_proba(X_test_df)[0]
        y_prob = proba[1]
        y_pred = 1 if y_prob >= 0.5 else 0
        
        return y_true, y_pred, y_prob

    def print_stats(self):
        """Prints prediction model update statistics."""
        print("\n" + "═"*44)
        print(f"║ {'PERIODIC-ONLY HYBRID STATS':^40} ║")
        print("═"*44)
        print(f"║ Periodic Retrains   : {self.count_periodic:>10}          ║")
        print("═"*44)

    def check_drift(self, is_correct):
        """Checks if periodic retrain interval is met."""
        self.active_triggers = []
        self.steps_since_last_retrain += 1
        
        if self.steps_since_last_retrain >= self.retrain_interval:
            self.active_triggers.append("long")
            
        return bool(self.active_triggers)

    def retrain(self, i, retraining_start_idx):
        """Handles full periodic retraining from scratch."""
        retraining_points = []
        
        if "long" in self.active_triggers:
            retraining_point = 0 
            if hasattr(self.X, 'iloc'):
                X_retrain = self.X.iloc[retraining_point:i+1]
            else:
                X_retrain = self.X[retraining_point:i+1]
            y_retrain = self.y[retraining_point:i+1]
            
            print(f"\n[Hybrid-Periodic] Triggered Periodic Full Retrain at step {i} (samples 0 to {i}).")
            self._fit_model(X_retrain, y_retrain)
            
            self.steps_since_last_retrain = 0
            self.count_periodic += 1
            retraining_points.append(retraining_point)
            
        self.active_triggers = []
        return min(retraining_points) if retraining_points else retraining_start_idx
