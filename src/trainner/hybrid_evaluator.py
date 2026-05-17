import numpy as np
import pandas as pd
import copy
from collections import deque
from src.trainner.base_evaluator import BaseEvaluator
from src.trainner.sliding_evaluator import SlidingEvaluator
from src.trainner.periodic_evaluator import PeriodicEvaluator
from src.trainner.incremental_evaluator import IncrementalEvaluator


class OnlineGatingNetwork:
    """
    Periodic-First Selective Override Gating Network.
    
    Core insight: The periodic model is a strong baseline because it trains on
    ALL historical data, giving it excellent probability calibration. The sliding
    model only helps when there's genuine concept drift that the periodic model
    hasn't adapted to yet.
    
    Strategy:
        1. DEFAULT to the periodic (long-term) model's prediction.
        2. Track rolling log-loss of BOTH models on a sliding window.
        3. Only override with the short-term model (or a blend) when it has
           demonstrated SIGNIFICANTLY lower log-loss recently.
        4. For fraud-suspect samples (p > threshold), use max(p_short, p_long)
           to never miss a strong fraud signal from either expert.
    
    This avoids the "dilution" problem where blending a well-calibrated model
    with a poorly-calibrated one degrades PR-AUC.
    """
    
    def __init__(self, window_size=200, warmup=200):
        self.warmup = warmup
        self.n_seen = 0
        self.window_size = window_size
        
        # Rolling log-loss tracking (the KEY metric for gating)
        self.short_losses = deque(maxlen=window_size)
        self.long_losses = deque(maxlen=window_size)
        
        # Rolling log-loss specifically on FRAUD samples
        self.short_fraud_losses = deque(maxlen=50)
        self.long_fraud_losses = deque(maxlen=50)
        
        # Track recent fraud recall (binary: did model catch fraud?)
        self.short_fraud_catches = deque(maxlen=50)
        self.long_fraud_catches = deque(maxlen=50)
    
    def predict_weight(self, p_short, p_long, X_sample):
        """
        Returns the final blended probability.
        
        Strategy: Periodic-First with Selective Override.
        The default is p_long. We only deviate when evidence is strong.
        """
        # During warmup, trust periodic completely
        if self.n_seen < self.warmup:
            return p_long
        
        # --- Fraud Signal Preservation ---
        # If EITHER model is confident about fraud, take the MAX.
        # This ensures we never dilute a strong fraud signal.
        # This is the single most important rule for PR-AUC.
        if p_short > 0.5 or p_long > 0.5:
            return max(p_short, p_long)
        
        # --- For non-fraud predictions: selective override ---
        # Calculate rolling performance advantage of short-term model
        if len(self.short_losses) >= 50:
            avg_short_loss = np.mean(self.short_losses)
            avg_long_loss = np.mean(self.long_losses)
            
            # Short-term model must be SIGNIFICANTLY better (>10% lower loss)
            # to justify overriding the well-calibrated periodic model
            improvement_ratio = (avg_long_loss - avg_short_loss) / (avg_long_loss + 1e-10)
            
            if improvement_ratio > 0.10:
                # Short-term is meaningfully better -> blend
                # More improvement = more weight on short-term
                alpha = min(0.6, improvement_ratio * 2)  # Cap at 0.6
                return alpha * p_short + (1 - alpha) * p_long
        
        # Default: trust the periodic model
        return p_long
    
    def update(self, p_short, p_long, y_true, pred_short, pred_long, X_sample):
        """Updates rolling loss statistics for both experts."""
        self.n_seen += 1
        
        # Calculate log-loss for this step
        eps = 1e-15
        p_s = np.clip(p_short, eps, 1 - eps)
        p_l = np.clip(p_long, eps, 1 - eps)
        
        loss_s = -(y_true * np.log(p_s) + (1 - y_true) * np.log(1 - p_s))
        loss_l = -(y_true * np.log(p_l) + (1 - y_true) * np.log(1 - p_l))
        
        self.short_losses.append(loss_s)
        self.long_losses.append(loss_l)
        
        # Track fraud-specific performance
        if y_true == 1:
            self.short_fraud_losses.append(loss_s)
            self.long_fraud_losses.append(loss_l)
            self.short_fraud_catches.append(1 if pred_short == 1 else 0)
            self.long_fraud_catches.append(1 if pred_long == 1 else 0)
    
    def reset_after_drift(self):
        """Clears rolling stats after model retrain."""
        self.short_losses.clear()
        self.long_losses.clear()
        self.short_fraud_losses.clear()
        self.long_fraud_losses.clear()
        self.short_fraud_catches.clear()
        self.long_fraud_catches.clear()


class HybridEvaluator(BaseEvaluator):
    """
    Hybrid Adaptive Ensemble: Periodic-First with Selective Override.
    
    Manages two models (Long-term & Short-term) and uses the Gating Network
    to decide when the short-term model's contribution is valuable enough
    to override the strong periodic baseline.
    
    Key design: The periodic model is treated as the "anchor" prediction.
    The short-term model only contributes when it has proven recent superiority.
    """
    def __init__(self, model, X, y):
        super().__init__(model, X, y)
        hybrid_config = self.trainner_config.get("hybrid", {})
        self.short_term_method = hybrid_config.get("short_term_method", "incremental")
        self.weight_short = hybrid_config.get("weight_short", 1.0)
        self.weight_long = hybrid_config.get("weight_long", 1.0)
        
        # Online Gating Network
        gating_config = hybrid_config.get("gating", {})
        self.gate = OnlineGatingNetwork(
            window_size=gating_config.get("window_size", 200),
            warmup=gating_config.get("warmup", 200)
        )
        
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
        self.count_long = 0
        self.count_short = 0
        self.count_ensemble = 0

    def init_train(self, initial_train_size=None):
        """Bootstrap training for both models."""
        if initial_train_size is None:
            initial_train_size = self.pipeline_config.get("init_train_size", 1000)
        
        size = self.long_term_eval.init_train(initial_train_size)
        # Sync model_short initially
        self.model_short = copy.deepcopy(self.model_long)
        self.short_term_eval.model = self.model_short
        
        return size

    def predict_step(self, i):
        """Gating-based prediction: periodic-first with selective override."""
        if hasattr(self.X, 'iloc'):
            X_test_df = self.X.iloc[i:i+1]
        else:
            X_test_df = self.X[i:i+1]
            
        y_true = self.y[i]

        # Get predictions from both expert models
        proba_short = self.model_short.predict_proba(X_test_df)[0]
        proba_long = self.model_long.predict_proba(X_test_df)[0]
        p_short = proba_short[1]
        p_long = proba_long[1]
        pred_short = 1 if p_short >= 0.5 else 0
        pred_long = 1 if p_long >= 0.5 else 0
        self.last_short_is_correct = 1 if pred_short == y_true else 0
        self.last_long_is_correct = 1 if pred_long == y_true else 0
        
        # Gate produces final probability
        y_prob = self.gate.predict_weight(p_short, p_long, X_test_df)
        
        # Track which expert dominated this prediction
        if abs(y_prob - p_short) < abs(y_prob - p_long):
            self.count_short += 1
        elif abs(y_prob - p_long) < abs(y_prob - p_short):
            self.count_long += 1
        else:
            self.count_ensemble += 1
        
        # Update gating model with the true label (learn from this step)
        self.gate.update(p_short, p_long, y_true, pred_short, pred_long, X_test_df)
        
        y_pred = 1 if y_prob >= 0.5 else 0
        return y_true, y_pred, y_prob

    def print_stats(self):
        """Prints prediction model usage statistics."""
        total = self.count_short + self.count_long + self.count_ensemble
        if total == 0: return
        
        print("\n" + "═"*44)
        print(f"║ {'HYBRID GATING STATS':^40} ║")
        print("═"*44)
        print(f"║ Short-term Led  : {self.count_short:>10} ({self.count_short/total:>6.1%}) ║")
        print(f"║ Long-term Led   : {self.count_long:>10} ({self.count_long/total:>6.1%}) ║")
        print(f"║ Blended         : {self.count_ensemble:>10} ({self.count_ensemble/total:>6.1%}) ║")
        print(f"║ Total Predicts  : {total:>10}          ║")
        print("═"*44)

    def check_drift(self, is_correct):
        """Checks both triggers."""
        self.active_triggers = []
        long_is_correct = self.last_long_is_correct
        short_is_correct = self.last_short_is_correct

        if long_is_correct is None:
            long_is_correct = is_correct
        if short_is_correct is None:
            short_is_correct = is_correct

        long_triggered = self.long_term_eval.check_drift(long_is_correct)
        short_triggered = self.short_term_eval.check_drift(short_is_correct)

        if long_triggered:
            self.active_triggers.append("long")
        if short_triggered:
            self.active_triggers.append("short")

        return bool(self.active_triggers)

    def retrain(self, i, retraining_start_idx):
        """Handles retraining for triggered components."""
        if self.long_retraining_start_idx is None:
            self.long_retraining_start_idx = retraining_start_idx
        if self.short_retraining_start_idx is None:
            self.short_retraining_start_idx = retraining_start_idx

        retraining_points = []

        if "long" in self.active_triggers:
            res_idx = self.long_term_eval.retrain(i, self.long_retraining_start_idx)
            self.model_long = self.long_term_eval.model
            retraining_points.append(res_idx)
            self.long_retraining_start_idx = i + 1
            # Reset gate rolling stats so it re-learns the new model's behavior
            self.gate.reset_after_drift()

        if "short" in self.active_triggers:
            res_idx = self.short_term_eval.retrain(i, self.short_retraining_start_idx)
            self.model_short = self.short_term_eval.model
            retraining_points.append(res_idx)
            self.short_retraining_start_idx = i + 1
            # Reset gate rolling stats
            self.gate.reset_after_drift()

        self.active_triggers = []
        return min(retraining_points) if retraining_points else retraining_start_idx
