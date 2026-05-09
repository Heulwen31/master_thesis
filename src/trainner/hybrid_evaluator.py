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
            
        self.active_trigger = None

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
            
        self.buffer_long_X = X_init.values if hasattr(X_init, 'values') else X_init
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
            
        X_test_val = X_test_df.values if hasattr(X_test_df, 'values') else X_test_df
        y_true = self.y[i]
        
        # 1. Search in Short Buffer
        d_short, _ = self.nn_short.kneighbors(X_test_val)
        min_d_short = d_short[0][0]
        
        # 2. Search in Long Buffer
        d_long, _ = self.nn_long.kneighbors(X_test_val)
        min_d_long = d_long[0][0]
        
        # Logic: Switching or Ensemble
        if min_d_short <= self.thresh:
            # Trusted Short Model
            y_prob = self.model_short.predict_proba(X_test_df)[0][1]
        elif min_d_long <= self.thresh:
            # Trusted Long Model
            y_prob = self.model_long.predict_proba(X_test_df)[0][1]
        else:
            # Combine based on inverse distance weights
            # Avg distance of k neighbors
            avg_d_short = np.mean(d_short[0])
            avg_d_long = np.mean(d_long[0])
            
            eps = 1e-8
            w_short = 1.0 / (avg_d_short + eps)
            w_long = 1.0 / (avg_d_long + eps)
            
            # Normalize weights
            total_w = w_short + w_long
            w_short /= total_w
            w_long /= total_w
            
            p_short = self.model_short.predict_proba(X_test_df)[0][1]
            p_long = self.model_long.predict_proba(X_test_df)[0][1]
            y_prob = w_short * p_short + w_long * p_long
            
        y_pred = 1 if y_prob >= 0.5 else 0
        return y_true, y_pred, y_prob

    def check_drift(self, is_correct):
        """Checks both triggers."""
        if self.long_term_eval.check_drift(is_correct):
            self.active_trigger = self.long_term_eval
            return True
        if self.short_term_eval.check_drift(is_correct):
            self.active_trigger = self.short_term_eval
            return True
        return False

    def retrain(self, i, retraining_start_idx):
        """Handles retraining and updates the respective NN index."""
        if self.active_trigger == self.long_term_eval:
            print(f"Hybrid: [LONG-TERM] Periodic retrain triggered at index {i}...")
            res_idx = self.long_term_eval.retrain(i, retraining_start_idx)
            
            # Update Long Buffer and NN Index (from index 0 to i)
            if hasattr(self.X, 'iloc'):
                X_batch = self.X.iloc[res_idx:i+1]
            else:
                X_batch = self.X[res_idx:i+1]
            self.buffer_long_X = X_batch.values if hasattr(X_batch, 'values') else X_batch
            self.nn_long.fit(self.buffer_long_X)
            
            # Sync Short-term
            if hasattr(self.short_term_eval, 'addm'):
                self.short_term_eval.addm.reset()
            if hasattr(self.short_term_eval, 'steps_since_last_update'):
                self.short_term_eval.steps_since_last_update = 0
            return res_idx
        else:
            # Short-term retrain
            res_idx = self.short_term_eval.retrain(i, retraining_start_idx)
            
            # Update Short Buffer and NN Index
            if hasattr(self.X, 'iloc'):
                X_batch = self.X.iloc[res_idx:i+1]
            else:
                X_batch = self.X[res_idx:i+1]
            self.buffer_short_X = X_batch.values if hasattr(X_batch, 'values') else X_batch
            self.nn_short.fit(self.buffer_short_X)
            
            return res_idx
