import math
from src.trainner.base_evaluator import BaseEvaluator
from src.models.river_factory import RiverModelFactory

class IncrementalEvaluator(BaseEvaluator):
    """
    Evaluates a model using River for incremental learning,
    configurable with different algorithms.
    """
    def __init__(self, model, X, y):
        super().__init__(model, X, y)
        config = self.trainner_config.get("incremental", {})
        
        self.algorithm = config.get("algorithm", "ARF")
        self.impute_nan = config.get("impute_nan", False)

        # Initialize model using factory
        self.model = RiverModelFactory.get_model(
            self.algorithm,
            n_models=config.get("n_models", 5),
            lambda_value=config.get("lambda_value", 6),
            grace_period=config.get("grace_period", 50),
            split_criterion=config.get("split_criterion", "info_gain")
        )

        self.initial_size = 0

    def init_train(self, initial_train_size=None):
        if initial_train_size is None:
            initial_train_size = self.pipeline_config.get("init_train_size", 1000)
        
        self.initial_size = initial_train_size
        
        # Prepare data
        if hasattr(self.X, "iloc"):
            X_init = self.X.iloc[:initial_train_size]
        else:
            X_init = self.X[:initial_train_size]
        y_init = self.y[:initial_train_size]

        print(f"Bootstrapping {self.algorithm} model on {initial_train_size} samples...")
        for idx in range(initial_train_size):
            row = self._prepare_input(X_init.iloc[idx].to_dict())
            self.model.learn_one(row, y_init[idx])
        
        print("Bootstrap complete.")
        return initial_train_size

    def _prepare_input(self, row_dict):
        if self.impute_nan:
            row_dict = {k: (0.0 if isinstance(v, float) and math.isnan(v) else v)
                        for k, v in row_dict.items()}
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

        # Learn from previous step
        if i > self.initial_size:
            prev_idx = i - 1
            if hasattr(self.X, "iloc"):
                X_prev_dict = self.X.iloc[prev_idx : prev_idx + 1].iloc[0].to_dict()
            else:
                X_prev_dict = self.X[prev_idx : prev_idx + 1][0].to_dict()
            self.model.learn_one(self._prepare_input(X_prev_dict), self.y[prev_idx])

        # Predict
        p_prob = float(self.model.predict_proba_one(x_test_dict).get(1, 0.0))
        y_pred = 1 if p_prob >= 0.5 else 0
        
        return y_true, y_pred, p_prob

    def retrain(self, i, retraining_start_idx):
        return retraining_start_idx
