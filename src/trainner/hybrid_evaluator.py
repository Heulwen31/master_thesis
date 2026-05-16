import numpy as np
import pandas as pd
import copy
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score, log_loss, average_precision_score
from src.utils.pso_gwo import PSOGWO
from src.trainner.base_evaluator import BaseEvaluator
from src.trainner.sliding_evaluator import SlidingEvaluator
from src.trainner.periodic_evaluator import PeriodicEvaluator
from src.trainner.incremental_evaluator import IncrementalEvaluator

class HybridEvaluator(BaseEvaluator):
    """
    Hybrid Adaptive Ensemble (HAE) optimized by PSO-GWO and Fraud Velocity.
    Manages two models (Long & Short) and combines them using weights 
    tuned by metaheuristic optimization and real-time fraud trends.
    """
    def __init__(self, model, X, y):
        super().__init__(model, X, y)
        hybrid_config = self.trainner_config.get("hybrid", {})
        self.short_term_method = hybrid_config.get("short_term_method", "incremental")
        self.weight_short = hybrid_config.get("weight_short", 1.0)
        self.weight_long = hybrid_config.get("weight_long", 1.0)
        
        # Metaheuristic Config
        self.use_metaheuristic = hybrid_config.get("use_metaheuristic", False)
        self.meta_metric = hybrid_config.get("metaheuristic_metric", "f1")
        self.meta_bounds = hybrid_config.get("metaheuristic_bounds", {})
        self.pso_gwo_config = hybrid_config.get("pso_gwo", {})
        self.opt_buffer = []  # Stores (y_true, p_short, p_long)
        
        # Trend-Aware Config
        trend_config = hybrid_config.get("trend_aware", {})
        self.trend_enabled = trend_config.get("enabled", False)
        self.trend_short_w = trend_config.get("short_window", 100)
        self.trend_long_w = trend_config.get("long_window", 500)
        self.trend_boost = trend_config.get("boost_factor", 1.5)
        self.trend_sensitivity = trend_config.get("sensitivity", 1.2)
        
        # 1. Long-term component (Stable model)
        self.model_long = copy.deepcopy(model)
        self.long_term_eval = PeriodicEvaluator(self.model_long, X, y)
        
        # 2. Short-term component (Adaptive model)
        self.model_short = copy.deepcopy(model)
        if self.short_term_method == "sliding":
            self.short_term_eval = SlidingEvaluator(self.model_short, X, y)
        else:
            self.short_term_eval = IncrementalEvaluator(self.model_short, X, y)
            
        self.active_triggers = []
        self.long_retraining_start_idx = None
        self.short_retraining_start_idx = None
        self.last_long_is_correct = None
        self.last_short_is_correct = None
        
        # Statistics
        self.count_ensemble = 0

    def init_train(self, initial_train_size=None):
        """Bootstrap training for both models."""
        if initial_train_size is None:
            initial_train_size = self.pipeline_config.get("init_train_size", 1000)
        
        size = self.long_term_eval.init_train(initial_train_size)
        self.model_short = copy.deepcopy(self.model_long)
        self.short_term_eval.model = self.model_short
        return size

    def predict_step(self, i):
        """Pure weighted ensemble prediction."""
        if hasattr(self.X, 'iloc'):
            X_test_df = self.X.iloc[i:i+1]
        else:
            X_test_df = self.X[i:i+1]
            
        y_true = self.y[i]

        proba_short = self.model_short.predict_proba(X_test_df)[0]
        proba_long = self.model_long.predict_proba(X_test_df)[0]
        p_short = proba_short[1]
        p_long = proba_long[1]
        
        pred_short = 1 if p_short >= 0.5 else 0
        pred_long = 1 if p_long >= 0.5 else 0
        self.last_short_is_correct = 1 if pred_short == y_true else 0
        self.last_long_is_correct = 1 if pred_long == y_true else 0
        
        # Save to optimization buffer
        if self.use_metaheuristic:
            self.opt_buffer.append((y_true, p_short, p_long))
            if len(self.opt_buffer) > 2000:
                self.opt_buffer.pop(0)
        
        # Dynamic Weights
        w_s = self.weight_short
        w_l = self.weight_long
        
        # Apply Trend-Aware Weight Boosting (Fraud Velocity)
        if self.trend_enabled and i > self.trend_long_w:
            recent_y = self.y[max(0, i-self.trend_short_w):i]
            rate_short = np.mean(recent_y) if len(recent_y) > 0 else 0
            
            baseline_y = self.y[max(0, i-self.trend_long_w):i]
            rate_long = np.mean(baseline_y) if len(baseline_y) > 0 else 0
            
            # Use sensitivity from config
            if rate_short > rate_long * self.trend_sensitivity: # Fraud increasing
                w_s *= self.trend_boost
            elif rate_short < rate_long * (1 / self.trend_sensitivity): # Fraud decreasing
                w_l *= self.trend_boost

        # Normalize weights
        total_w = w_s + w_l
        w_s_norm = w_s / total_w
        w_l_norm = w_l / total_w
        
        y_prob = w_s_norm * p_short + w_l_norm * p_long
        y_pred = 1 if y_prob >= 0.5 else 0
        self.count_ensemble += 1
            
        return y_true, y_pred, y_prob

    def print_stats(self):
        """Prints prediction model usage statistics."""
        print("\n" + "═"*40)
        print(f"║ {'HYBRID ENSEMBLE STATS':^36} ║")
        print("═"*40)
        print(f"║ Total Predicts  : {self.count_ensemble:>10}          ║")
        print(f"║ Optimized Weight Short: {self.weight_short:>8.3f}     ║")
        print(f"║ Optimized Weight Long : {self.weight_long:>8.3f}     ║")
        print("═"*40)

    def check_drift(self, is_correct):
        """Checks both triggers."""
        self.active_triggers = []
        long_is_correct = self.last_long_is_correct
        short_is_correct = self.last_short_is_correct

        if long_is_correct is None: long_is_correct = is_correct
        if short_is_correct is None: short_is_correct = is_correct

        long_triggered = self.long_term_eval.check_drift(long_is_correct)
        short_triggered = self.short_term_eval.check_drift(short_is_correct)

        if long_triggered: self.active_triggers.append("long")
        if short_triggered: self.active_triggers.append("short")

        return bool(self.active_triggers)

    def retrain(self, i, retraining_start_idx):
        """Handles retraining."""
        if self.long_retraining_start_idx is None: self.long_retraining_start_idx = retraining_start_idx
        if self.short_retraining_start_idx is None: self.short_retraining_start_idx = retraining_start_idx

        retraining_points = []

        if "long" in self.active_triggers:
            res_idx = self.long_term_eval.retrain(i, self.long_retraining_start_idx)
            self.model_long = self.long_term_eval.model
            retraining_points.append(res_idx)
            self.long_retraining_start_idx = i + 1

        if "short" in self.active_triggers:
            res_idx = self.short_term_eval.retrain(i, self.short_retraining_start_idx)
            self.model_short = self.short_term_eval.model
            retraining_points.append(res_idx)
            self.short_retraining_start_idx = i + 1

        self.active_triggers = []
        if self.use_metaheuristic and retraining_points:
            self.optimize_weights()

        return min(retraining_points) if retraining_points else retraining_start_idx

    def optimize_weights(self):
        """Uses Hybrid PSO-GWO to find optimal base weights."""
        if not self.use_metaheuristic or len(self.opt_buffer) < 50:
            return

        y_true = np.array([x[0] for x in self.opt_buffer])
        p_short = np.array([x[1] for x in self.opt_buffer])
        p_long = np.array([x[2] for x in self.opt_buffer])
        
        def objective(weights):
            w_s, w_l = weights
            total_w = w_s + w_l
            ws_n = w_s / total_w
            wl_n = w_l / total_w
            
            y_prob = ws_n * p_short + wl_n * p_long
            
            if self.meta_metric == "f1":
                # Optimize Average Precision (PR-AUC) instead of hard F1
                # This is much more stable for imbalanced fraud data
                return -average_precision_score(y_true, y_prob)
            else:
                return log_loss(y_true, np.clip(y_prob, 1e-15, 1-1e-15))

        b_short = self.meta_bounds.get("weight_short", [0.1, 5.0])
        b_long = self.meta_bounds.get("weight_long", [0.1, 5.0])
        
        optimizer = PSOGWO(
            obj_func=objective,
            bounds=[b_short, b_long],
            pop_size=self.pso_gwo_config.get("pop_size", 10),
            max_iter=self.pso_gwo_config.get("max_iter", 20),
            w_max=self.pso_gwo_config.get("w_max", 0.9),
            w_min=self.pso_gwo_config.get("w_min", 0.4),
            c1=self.pso_gwo_config.get("c1", 2.0),
            c2=self.pso_gwo_config.get("c2", 2.0)
        )
        
        best_weights, _ = optimizer.optimize()
        self.weight_short, self.weight_long = best_weights


