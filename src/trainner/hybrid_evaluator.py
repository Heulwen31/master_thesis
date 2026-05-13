import numpy as np
import pandas as pd
import copy
from sklearn.neighbors import NearestNeighbors
from src.trainner.base_evaluator import BaseEvaluator
from src.trainner.online_evaluator import OnlineEvaluator
from src.trainner.periodic_evaluator import PeriodicEvaluator
from src.trainner.incremental_evaluator import IncrementalEvaluator

class HybridEvaluator(BaseEvaluator):
    """
    Hybrid Adaptive Ensemble with Similarity-based Switching (RAHE).
    Manages two models (Long & Short) and selects/combines them based on 
    the similarity of the test sample to their respective training data.
    """
    def __init__(self, model, X, y):
        super().__init__(model, X, y)
        hybrid_config = self.trainner_config.get("hybrid", {})
        self.short_term_method = hybrid_config.get("short_term_method", "incremental")
        self.thresh = hybrid_config.get("similarity_threshold", 0.1)
        self.k = hybrid_config.get("k_neighbors", 3)
        
        # 1. Long-term component (Stable model)
        self.model_long = copy.deepcopy(model)
        self.long_term_eval = PeriodicEvaluator(self.model_long, X, y)
        self.nn_long = NearestNeighbors(n_neighbors=self.k, n_jobs=-1)
        self.buffer_long_X = None
        
        # 2. Short-term component (Adaptive model)
        self.model_short = copy.deepcopy(model)
        if self.short_term_method == "addm":
            self.short_term_eval = OnlineEvaluator(self.model_short, X, y)
        else:
            self.short_term_eval = IncrementalEvaluator(self.model_short, X, y)
        self.nn_short = NearestNeighbors(n_neighbors=self.k, n_jobs=-1)
        self.buffer_short_X = None
            
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
        
        # Initialize buffers
        if hasattr(self.X, 'iloc'):
            X_init = self.X.iloc[:initial_train_size]
        else:
            X_init = self.X[:initial_train_size]
            
        # Fill NaNs for NN search (distance-based algorithms cannot handle NaNs)
        self.buffer_long_X = X_init.fillna(0).values if hasattr(X_init, 'fillna') else np.nan_to_num(X_init, nan=0)
        self.buffer_short_X = self.buffer_long_X.copy()
        
        # Fit NN indices
        self.nn_long.fit(self.buffer_long_X)
        self.nn_short.fit(self.buffer_short_X)
        
        return size

    def predict_step(self, i):
        """Similarity-based prediction logic."""
        if hasattr(self.X, 'iloc'):
            X_test_df = self.X.iloc[i:i+1]
        else:
            X_test_df = self.X[i:i+1]
            
        # Fill NaNs for query sample
        X_test_val = X_test_df.fillna(0).values if hasattr(X_test_df, 'fillna') else np.nan_to_num(X_test_df, nan=0)
        y_true = self.y[i]

        proba_short = self.model_short.predict_proba(X_test_df)[0]
        proba_long = self.model_long.predict_proba(X_test_df)[0]
        p_short = proba_short[1]
        p_long = proba_long[1]
        pred_short = 1 if p_short >= 0.5 else 0
        pred_long = 1 if p_long >= 0.5 else 0
        self.last_short_is_correct = 1 if pred_short == y_true else 0
        self.last_long_is_correct = 1 if pred_long == y_true else 0
        
        # 1. Search in Short Buffer
        d_short, _ = self.nn_short.kneighbors(X_test_val)
        min_d_short = d_short[0][0]
        
        # 2. Search in Long Buffer
        d_long, _ = self.nn_long.kneighbors(X_test_val)
        min_d_long = d_long[0][0]
        
        # Logic: Switching or Ensemble
        if min_d_short <= self.thresh:
            # Trusted Short Model
            y_prob = p_short
            self.count_short += 1
        elif min_d_long <= self.thresh:
            # Trusted Long Model
            y_prob = p_long
            self.count_long += 1
        else:
            # Combine based on inverse distance weights
            avg_d_short = np.mean(d_short[0])
            avg_d_long = np.mean(d_long[0])
            
            eps = 1e-8
            w_short = 1.0 / (avg_d_short + eps)
            w_long = 1.0 / (avg_d_long + eps)
            
            # Normalize weights
            total_w = w_short + w_long
            w_short /= total_w
            w_long /= total_w
            
            y_prob = w_short * p_short + w_long * p_long
            self.count_ensemble += 1
            
        y_pred = 1 if y_prob >= 0.5 else 0
        return y_true, y_pred, y_prob

    def print_stats(self):
        """Prints prediction model usage statistics."""
        total = self.count_short + self.count_long + self.count_ensemble
        if total == 0: return
        
        print("\n" + "═"*40)
        print(f"║ {'HYBRID PREDICTION STATS':^36} ║")
        print("═"*40)
        print(f"║ Short-term Only : {self.count_short:>10} ({self.count_short/total:>6.1%}) ║")
        print(f"║ Long-term Only  : {self.count_long:>10} ({self.count_long/total:>6.1%}) ║")
        print(f"║ Ensemble        : {self.count_ensemble:>10} ({self.count_ensemble/total:>6.1%}) ║")
        print(f"║ Total Predicts  : {total:>10}          ║")
        print("═"*40)

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
        """Handles retraining and updates the respective NN index."""
        if self.long_retraining_start_idx is None:
            self.long_retraining_start_idx = retraining_start_idx
        if self.short_retraining_start_idx is None:
            self.short_retraining_start_idx = retraining_start_idx

        retraining_points = []

        if "long" in self.active_triggers:
            # print(f"Hybrid: [LONG-TERM] Periodic retrain triggered at index {i}...")
            res_idx = self.long_term_eval.retrain(i, self.long_retraining_start_idx)
            self.model_long = self.long_term_eval.model
            retraining_points.append(res_idx)
            
            # Update Long Buffer and NN Index (from index 0 to i)
            if hasattr(self.X, 'iloc'):
                X_batch = self.X.iloc[res_idx:i+1]
            else:
                X_batch = self.X[res_idx:i+1]
            # Fill NaNs for NN
            self.buffer_long_X = X_batch.fillna(0).values if hasattr(X_batch, 'fillna') else np.nan_to_num(X_batch, nan=0)
            self.nn_long.fit(self.buffer_long_X)

            self.long_retraining_start_idx = i + 1

        if "short" in self.active_triggers:
            reason = getattr(self.short_term_eval, 'trigger_reason', None)
            reason_label = f" {reason}" if reason else ""
            # print(f"Hybrid: [SHORT-TERM]{reason_label} retrain triggered at index {i}...")
            res_idx = self.short_term_eval.retrain(i, self.short_retraining_start_idx)
            self.model_short = self.short_term_eval.model
            retraining_points.append(res_idx)
            
            # Update Short Buffer and NN Index
            if hasattr(self.X, 'iloc'):
                X_batch = self.X.iloc[res_idx:i+1]
            else:
                X_batch = self.X[res_idx:i+1]
            # Fill NaNs for NN
            self.buffer_short_X = X_batch.fillna(0).values if hasattr(X_batch, 'fillna') else np.nan_to_num(X_batch, nan=0)
            self.nn_short.fit(self.buffer_short_X)

            self.short_retraining_start_idx = i + 1

        self.active_triggers = []
        return min(retraining_points) if retraining_points else retraining_start_idx
