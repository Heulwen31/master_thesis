import copy
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgb
from src.trainner.base_evaluator import BaseEvaluator


class HybridEvaluator(BaseEvaluator):
    """
    Historical + Adaptive Layer Gated Ensemble.
    
    Architecture:
    Historical Stable Model (trained on huge data)
            ↓
    Frozen / semi-frozen
            ↓
    Adaptive Layer (trained on recent data, copied from stable model)
            ↓
    Final Ensemble (Confidence-gated combination)
    """
    def __init__(self, model, X, y):
        super().__init__(model, X, y)
        
        # Load configurations
        periodic_config = self.trainner_config.get("periodic", {})
        self.retrain_interval = periodic_config.get("retrain_interval", 5000)
        
        incremental_config = self.trainner_config.get("incremental", {})
        self.update_interval = incremental_config.get("update_interval", 100)
        self.adaptive_window = incremental_config.get("adaptive_window", 1000)
        self.update_trees = incremental_config.get("update_trees", 20)
        
        hybrid_config = self.trainner_config.get("hybrid", {})
        self.gate_threshold = hybrid_config.get("gate_threshold", 0.02)
        self.weight_adaptive = hybrid_config.get("weight_adaptive", 0.3)
        
        # State tracking
        self.steps_since_last_retrain = 0
        self.steps_since_last_update = 0
        self.active_triggers = []
        
        # Active models
        self.adaptive_model = None
        
        # Statistics
        self.count_periodic = 0
        self.count_incremental = 0

    def init_train(self, initial_train_size=None):
        """Bootstrap training for the clean Historical Stable Model."""
        if initial_train_size is None:
            initial_train_size = self.pipeline_config.get("init_train_size", 1000)
        
        if hasattr(self.X, 'iloc'):
            X_train = self.X.iloc[:initial_train_size]
        else:
            X_train = self.X[:initial_train_size]
        y_train = self.y[:initial_train_size]
        
        # 1. Fit Historical Stable Model
        print(f"\n[Ensemble] Bootstrap training Historical Stable Model on {initial_train_size} samples...")
        self._fit_model(X_train, y_train)
        
        # 2. Make initial copy of Adaptive Layer
        self.adaptive_model = copy.deepcopy(self.model)
        
        self.steps_since_last_retrain = 0
        self.steps_since_last_update = 0
        
        return initial_train_size

    def predict_step(self, i):
        """Predict using the Final Ensemble of Stable + Adaptive Layer."""
        if hasattr(self.X, 'iloc'):
            X_test_df = self.X.iloc[i:i+1]
        else:
            X_test_df = self.X[i:i+1]
            
        y_true = self.y[i]
        
        # 1. Prediction from frozen Historical Stable Model
        p_stable = self.model.predict_proba(X_test_df)[0][1]
        
        # 2. Confidence Gating
        if p_stable < self.gate_threshold:
            # High-confidence normal sample: trust the frozen Historical model 100%
            y_prob = p_stable
        else:
            # Ambiguous or potential fraud sample: blend with Adaptive Layer
            p_adaptive = self.adaptive_model.predict_proba(X_test_df)[0][1]
            y_prob = (1 - self.weight_adaptive) * p_stable + self.weight_adaptive * p_adaptive
        
        y_pred = 1 if y_prob >= 0.5 else 0
        
        return y_true, y_pred, y_prob

    def check_drift(self, is_correct):
        """Checks if periodic retrain or short-term adaptive update interval is met."""
        self.active_triggers = []
        self.steps_since_last_retrain += 1
        self.steps_since_last_update += 1
        
        if self.steps_since_last_retrain >= self.retrain_interval:
            self.active_triggers.append("long")
        elif self.steps_since_last_update >= self.update_interval:
            self.active_triggers.append("short")
            
        return bool(self.active_triggers)

    def retrain(self, i, retraining_start_idx):
        """Handles periodic retraining of Stable model or Adaptive Layer copy-and-train updates."""
        retraining_points = []
        
        if "long" in self.active_triggers:
            # 1. Periodic full retraining of Historical Stable Model
            retraining_point = 0 
            if hasattr(self.X, 'iloc'):
                X_retrain = self.X.iloc[retraining_point:i+1]
            else:
                X_retrain = self.X[retraining_point:i+1]
            y_retrain = self.y[retraining_point:i+1]
            
            print(f"\n[Ensemble] Periodic Full Retrain of Historical Stable Model at step {i} (samples 0 to {i}).")
            self._fit_model(X_retrain, y_retrain)
            
            # Re-initialize Adaptive Layer from the new clean Stable Model
            self.adaptive_model = copy.deepcopy(self.model)
            
            self.steps_since_last_retrain = 0
            self.steps_since_last_update = 0
            self.count_periodic += 1
            retraining_points.append(retraining_point)
            
        elif "short" in self.active_triggers:
            # 2. Adaptive Layer copy-and-train update on recent window
            window_start = max(0, i - self.adaptive_window + 1)
            if hasattr(self.X, 'iloc'):
                X_recent = self.X.iloc[window_start:i+1]
            else:
                X_recent = self.X[window_start:i+1]
            y_recent = self.y[window_start:i+1]
            
            # Ensure both classes are present in the adaptive window to stabilize LightGBM
            X_recent, y_recent = self._ensure_two_classes(X_recent, y_recent, max_extra_samples=15)
            
            # Copy stable model to leverage all learned parameters
            self.adaptive_model = copy.deepcopy(self.model)
            
            # Update/Fine-tune the copy with recent data (Adaptive Layer)
            if isinstance(self.adaptive_model, lgb.LGBMClassifier):
                orig_estimators = self.adaptive_model.n_estimators
                orig_lr = self.adaptive_model.learning_rate
                
                current_trees = len(self.adaptive_model.booster_.dump_model()['tree_info'])
                self.adaptive_model.n_estimators = current_trees + self.update_trees
                self.adaptive_model.learning_rate = 0.05  # Gentle update step
                
                self.adaptive_model.fit(X_recent, y_recent, init_model=self.adaptive_model.booster_)
                
                self.adaptive_model.n_estimators = orig_estimators
                self.adaptive_model.learning_rate = orig_lr
                
            elif isinstance(self.adaptive_model, xgb.XGBClassifier):
                orig_estimators = self.adaptive_model.n_estimators
                orig_lr = self.adaptive_model.learning_rate
                
                current_trees = self.adaptive_model.get_booster().num_boosted_rounds()
                self.adaptive_model.n_estimators = current_trees + self.update_trees
                self.adaptive_model.learning_rate = 0.05
                
                self.adaptive_model.fit(X_recent, y_recent, xgb_model=self.adaptive_model.get_booster())
                
                self.adaptive_model.n_estimators = orig_estimators
                self.adaptive_model.learning_rate = orig_lr
                
            elif isinstance(self.adaptive_model, CatBoostClassifier):
                # CatBoost fine-tuning
                params = self.adaptive_model.get_params()
                params['iterations'] = self.update_trees
                params['learning_rate'] = 0.05
                cat_features = list(X_recent.select_dtypes(include=['category']).columns)
                
                new_model = CatBoostClassifier(**params)
                new_model.fit(X_recent, y_recent, init_model=self.adaptive_model, cat_features=cat_features)
                self.adaptive_model = new_model
            else:
                self.adaptive_model.fit(X_recent, y_recent)
                
            self.steps_since_last_update = 0
            self.count_incremental += 1
            retraining_points.append(window_start)
            
        self.active_triggers = []
        return min(retraining_points) if retraining_points else retraining_start_idx

    def print_stats(self):
        """Prints hybrid ensemble statistics."""
        print("\n" + "═"*44)
        print(f"║ {'HISTORICAL-ADAPTIVE GATED ENSEMBLE':^40} ║")
        print("═"*44)
        print(f"║ Stable Periodic Retrains : {self.count_periodic:>10}          ║")
        print(f"║ Adaptive Layer Updates   : {self.count_incremental:>10}          ║")
        print(f"║ Adaptive Window Size     : {self.adaptive_window:>10}          ║")
        print(f"║ Adaptive Trees Added     : {self.update_trees:>10}          ║")
        print(f"║ Confidence Gate Thresh   : {self.gate_threshold:>10.3f}          ║")
        print("═"*44)
