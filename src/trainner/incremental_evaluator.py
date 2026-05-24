import math
import numpy as np
from river import preprocessing

from src.trainner.base_evaluator import BaseEvaluator
from src.models.river_factory import RiverModelFactory


class IncrementalEvaluator(BaseEvaluator):
    """
    Prequential (test-then-train) evaluator using River incremental models.

    Optimizations for PRAUC:
      - Online feature scaling via StandardScaler
      - Class-weighted learning (inverse frequency, capped)
      - Adaptive decision threshold based on observed class prior
      - NaN imputation (configurable)
    """

    def __init__(self, model, X, y, dataset_name=None):
        super().__init__(model, X, y)
        config = self.trainner_config.get("incremental", {})

        # Apply dataset-specific overrides
        if dataset_name:
            overrides = config.get("dataset_overrides", {}).get(dataset_name, {})
            if overrides:
                base = {k: v for k, v in config.items() if k != "dataset_overrides"}
                config = {**base, **overrides}

        self.algorithm = config.get("algorithm", "ARF")
        self.impute_nan = config.get("impute_nan", True)

        self.scaler = preprocessing.StandardScaler()

        self.model = RiverModelFactory.get_model(
            self.algorithm,
            n_models=config.get("n_models", 20),
            lambda_value=config.get("lambda_value", 10),
            grace_period=config.get("grace_period", 200),
            split_criterion=config.get("split_criterion", "gini"),
            leaf_prediction=config.get("leaf_prediction", "nba"),
            max_depth=config.get("max_depth", None),
            seed=config.get("seed", 42),
        )

        self.fraud_weight_config = config.get("fraud_weight", "auto")
        self.initial_size = 0

        # Track class prior for adaptive weighting & thresholding
        self.n_samples_seen = 0
        self.n_fraud_seen = 0
        self.fraud_weight = 1.0
        self.class_prior = 0.0

    def init_train(self, initial_train_size=None):
        if initial_train_size is None:
            initial_train_size = self.pipeline_config.get("init_train_size", 10000)

        self.initial_size = initial_train_size

        if hasattr(self.X, "iloc"):
            X_init = self.X.iloc[:initial_train_size]
        else:
            X_init = self.X[:initial_train_size]
        y_init = self.y[:initial_train_size]

        # Compute class prior from bootstrap data
        self.n_fraud_seen = int(y_init.sum())
        self.n_samples_seen = initial_train_size
        self.class_prior = self.n_fraud_seen / self.n_samples_seen

        # Compute inverse-frequency weight (capped)
        self.fraud_weight = self._compute_fraud_weight()

        print(f"Bootstrapping {self.algorithm} model on {initial_train_size} samples...")
        print(f"  Class prior (fraud rate): {self.class_prior:.6f}")
        print(f"  Fraud sample weight: {self.fraud_weight:.1f}")

        for idx in range(initial_train_size):
            row = self._prepare_input(X_init.iloc[idx].to_dict())
            self.scaler.learn_one(row)
            row = self.scaler.transform_one(row)
            w = self.fraud_weight if y_init[idx] == 1 else 1.0
            self.model.learn_one(row, y_init[idx], weight=w)

        print("Bootstrap complete.")
        return initial_train_size

    def _compute_fraud_weight(self):
        if isinstance(self.fraud_weight_config, (int, float)) and self.fraud_weight_config != "auto":
            return max(1.0, float(self.fraud_weight_config))
        if self.n_samples_seen < 2 or self.class_prior <= 0:
            return 1.0
        if self.class_prior >= 0.5:
            return 1.0
        w = (1.0 - self.class_prior) / self.class_prior
        return float(np.clip(w, 1.0, 100.0))

    def _prepare_input(self, row_dict):
        if self.impute_nan:
            row_dict = {
                k: (0.0 if isinstance(v, float) and math.isnan(v) else v)
                for k, v in row_dict.items()
            }
        return row_dict

    def check_drift(self, is_correct):
        return False

    def predict_step(self, i):
        if hasattr(self.X, "iloc"):
            X_test_df = self.X.iloc[i : i + 1]
        else:
            X_test_df = self.X[i : i + 1]

        x_test_dict = self._prepare_input(X_test_df.iloc[0].to_dict())
        y_true = self.y[i]

        # Learn from previous step (test-then-train)
        if i > self.initial_size:
            prev_idx = i - 1
            if hasattr(self.X, "iloc"):
                X_prev_dict = (
                    self.X.iloc[prev_idx : prev_idx + 1].iloc[0].to_dict()
                )
            else:
                X_prev_dict = self.X[prev_idx : prev_idx + 1][0].to_dict()
            X_prev_dict = self._prepare_input(X_prev_dict)
            self.scaler.learn_one(X_prev_dict)
            X_prev_scaled = self.scaler.transform_one(X_prev_dict)
            w = self.fraud_weight if self.y[prev_idx] == 1 else 1.0
            self.model.learn_one(X_prev_scaled, self.y[prev_idx], weight=w)

            # Update running class prior
            self.n_samples_seen += 1
            if self.y[prev_idx] == 1:
                self.n_fraud_seen += 1
            # Exponential moving average for smoother prior
            self.class_prior = 0.99 * self.class_prior + 0.01 * (
                self.n_fraud_seen / self.n_samples_seen
            )

        # Scale test features (using pre-computed scaler stats, no data leak)
        x_test_scaled = self.scaler.transform_one(x_test_dict)

        # Predict probability for class 1
        p_prob = float(self.model.predict_proba_one(x_test_scaled).get(1, 0.0))

        # Adaptive threshold: at least 2x the class prior, floor of 0.05
        threshold = max(0.05, min(0.5, self.class_prior * 2.0))
        y_pred = 1 if p_prob >= threshold else 0

        return y_true, y_pred, p_prob

    def retrain(self, i, retraining_start_idx):
        return retraining_start_idx
