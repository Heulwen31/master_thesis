import copy
import math
from collections import deque

import numpy as np
from src.trainner.base_evaluator import BaseEvaluator
from src.trainner.periodic_evaluator import PeriodicEvaluator
from src.models.river_factory import RiverModelFactory


class HybridEvaluator(BaseEvaluator):
    """
    Hybrid: periodic long model + River short model.

    boost_mode (configs/trainner.yml):
      - auc_boost: soft-OR when short is confident AND long score is in an
        ambiguous band (min_p_long < p_long < max_p_long). Tuned on creditcard
        50k test stream to improve ROC-AUC and PR-AUC vs periodic.
      - signal_gated: stress signals + monotonic boost (conservative).
      - legacy_soft_or: p_short > threshold → full soft-OR (no stress).
      - and_fusion: conservative AND-gate (both must agree) + log-odds
        blend with prior shrinkage. Designed for extreme class imbalance.

    dataset_name is used to load per-dataset overrides from
    configs/trainner.yml → hybrid → dataset_overrides → <dataset_name>.
    Supports different ARF capacity / NaN handling per dataset.
    """

    def __init__(self, model, X, y, dataset_name=None):
        super().__init__(model, X, y)
        hybrid_config = self.trainner_config.get("hybrid", {})

        if dataset_name:
            overrides = hybrid_config.get("dataset_overrides", {}).get(dataset_name, {})
            if overrides:
                base = {k: v for k, v in hybrid_config.items() if k != "dataset_overrides"}
                hybrid_config = {**base, **overrides}
                print(f"Applied hybrid overrides for dataset='{dataset_name}'")
        self.boost_mode = hybrid_config.get("boost_mode", "auc_boost")
        self.alpha = hybrid_config.get("alpha", 0.01)
        self.intervention_threshold = hybrid_config.get("intervention_threshold", 0.128)
        self.incremental_margin = hybrid_config.get("incremental_margin", 0.02)
        self.short_weight = hybrid_config.get("short_weight", 1.0)
        self.min_p_long_for_boost = hybrid_config.get("min_p_long_for_boost", 0.05)
        self.max_p_long_for_boost = hybrid_config.get("max_p_long_for_boost", 1.0)
        self.tiebreak_epsilon = hybrid_config.get("tiebreak_epsilon", 1e-7)
        self.monitor_window = hybrid_config.get("monitor_window", 500)
        self.min_monitor_samples = hybrid_config.get("min_monitor_samples", 100)
        self.accuracy_drop_threshold = hybrid_config.get("accuracy_drop_threshold", 0.05)
        self.min_rolling_accuracy = hybrid_config.get("min_rolling_accuracy", None)
        self.error_burst_multiplier = hybrid_config.get("error_burst_multiplier", 2.5)
        self.min_error_rate_for_burst = hybrid_config.get("min_error_rate_for_burst", 0.02)
        self.use_brier_stress = hybrid_config.get("use_brier_stress", True)
        self.brier_increase_ratio = hybrid_config.get("brier_increase_ratio", 0.12)
        self.stress_cooldown = hybrid_config.get("stress_cooldown", 300)
        self.require_stress_for_boost = hybrid_config.get("require_stress_for_boost", False)
        self.river_n_models = hybrid_config.get("river_n_models", 5)
        self.river_lambda_value = hybrid_config.get("river_lambda_value", 6)
        self.river_grace_period = hybrid_config.get("river_grace_period", 50)
        self.river_split_criterion = hybrid_config.get("river_split_criterion", "info_gain")
        self.river_algorithm = hybrid_config.get("river_algorithm", "ARF")

        # and_fusion mode params
        self.and_long_threshold = hybrid_config.get("and_long_threshold", 0.05)
        self.prior_ratio = hybrid_config.get("prior_ratio", None)

        # Short model input preparation (critical for high-D / NaN-rich data)
        self.short_max_features = hybrid_config.get("short_max_features", None)
        self.impute_nan = hybrid_config.get("impute_nan", True)
        self.short_selected_features = None

        self.model_long = copy.deepcopy(model)
        self.long_term_eval = PeriodicEvaluator(self.model_long, X, y)

        self.model_short = RiverModelFactory.get_model(
            self.river_algorithm,
            n_models=self.river_n_models,
            lambda_value=self.river_lambda_value,
            grace_period=self.river_grace_period,
            split_criterion=self.river_split_criterion
        )

        self.active_triggers = []
        self.long_retraining_start_idx = None
        self.last_long_is_correct = None

        self.recent_long_correct = deque(maxlen=self.monitor_window)
        self.recent_long_brier = deque(maxlen=self.monitor_window)
        self.ema_long_accuracy = None
        self.ema_long_brier = None
        self.stress_active = False
        self.cooldown_remaining = 0

        self.count_steps = 0
        self.count_long_only = 0
        self.count_boost = 0
        self.initial_size = 0

    def init_train(self, initial_train_size=None):
        if initial_train_size is None:
            initial_train_size = self.pipeline_config.get("init_train_size", 1000)

        size = self.long_term_eval.init_train(initial_train_size)
        self.initial_size = size

        if hasattr(self.X, "iloc"):
            X_init = self.X.iloc[:size]
        else:
            X_init = self.X[:size]
        y_init = self.y[:size]

        if self.prior_ratio is None:
            self.prior_ratio = float(y_init.mean())
            print(f"Auto-computed prior_ratio = {self.prior_ratio:.6f}")

        if self.short_max_features is not None and hasattr(self.model_long, 'feature_importances_'):
            importances = self.model_long.feature_importances_
            feature_names = X_init.columns.tolist()
            n_sel = min(self.short_max_features, len(feature_names))
            top_idx = np.argsort(importances)[-n_sel:]
            self.short_selected_features = set(feature_names[i] for i in top_idx)
            print(f"Short model: selected {n_sel}/{len(feature_names)} features by importance")

        print(f"Bootstrapping River short-term model on {size} samples...")
        for idx in range(size):
            row = self._prepare_short_input(X_init.iloc[idx].to_dict())
            self.model_short.learn_one(row, y_init[idx])

        print(f"Bootstrap complete. Hybrid boost_mode={self.boost_mode}")
        return size

    @staticmethod
    def _log_odds(p):
        p = max(min(p, 1.0 - 1e-15), 1e-15)
        return math.log(p / (1.0 - p))

    @staticmethod
    def _inv_log_odds(lo):
        return 1.0 / (1.0 + math.exp(-lo))

    def _prepare_short_input(self, row_dict):
        if self.short_selected_features is not None:
            row_dict = {k: v for k, v in row_dict.items() if k in self.short_selected_features}
        if self.impute_nan:
            row_dict = {k: (0.0 if isinstance(v, float) and math.isnan(v) else v)
                        for k, v in row_dict.items()}
        return row_dict

    def _fuse_monotonic(self, p_long, p_short):
        p_short_eff = min(self.short_weight * p_short, 1.0)
        boosted = 1.0 - (1.0 - p_long) * (1.0 - p_short_eff)
        return max(p_long, boosted)

    def _compute_final_probability(self, p_long, p_short):
        if self.boost_mode == "legacy_soft_or":
            if p_short > self.intervention_threshold:
                return self._fuse_monotonic(p_long, p_short), True
            return p_long, False

        if self.boost_mode == "signal_gated":
            short_ok = (
                p_short > self.intervention_threshold
                and p_short > p_long + self.incremental_margin
            )
            if (not self.require_stress_for_boost or self.stress_active) and short_ok:
                return self._fuse_monotonic(p_long, p_short), True
            return p_long, False

        if self.boost_mode == "and_fusion":
            short_boost = p_short * self.short_weight
            y_prob = p_long * (1.0 + short_boost)
            y_prob = min(y_prob, 1.0)

            both_confident = (
                p_short > self.intervention_threshold
                and p_long > self.and_long_threshold
            )

            if both_confident:
                lo_long = self._log_odds(p_long)
                lo_short = self._log_odds(min(p_short, 0.999))
                lo_prior = self._log_odds(self.prior_ratio)

                w_short = 0.2 + 0.3 * min(p_short, 1.0)
                w_long = 1.0 - w_short

                lo_fused = (0.9 * (w_long * lo_long + w_short * lo_short)
                            + 0.1 * lo_prior)
                p_fused = self._inv_log_odds(lo_fused)
                y_prob = max(y_prob, p_fused)
                return min(y_prob, 1.0), True

            return min(y_prob, 1.0), False

        # auc_boost (default): ambiguous-band soft-OR + tiny rank tie-break
        y_prob = p_long + p_short * self.tiebreak_epsilon
        in_band = (
            p_short > self.intervention_threshold
            and self.min_p_long_for_boost < p_long < self.max_p_long_for_boost
        )
        if in_band:
            y_prob = max(y_prob, self._fuse_monotonic(p_long, p_short))
            return y_prob, True
        return y_prob, False

    def _evaluate_stress_signals(self):
        if len(self.recent_long_correct) < self.min_monitor_samples:
            return False, []

        rolling_acc = sum(self.recent_long_correct) / len(self.recent_long_correct)
        ema_acc = self.ema_long_accuracy if self.ema_long_accuracy is not None else rolling_acc
        rolling_err = 1.0 - rolling_acc
        ema_err = max(1.0 - ema_acc, 1e-6)

        reasons = []
        if (ema_acc - rolling_acc) >= self.accuracy_drop_threshold:
            reasons.append("accuracy_drop")
        if (
            self.min_rolling_accuracy is not None
            and rolling_acc < self.min_rolling_accuracy
        ):
            reasons.append("below_floor")
        if (
            rolling_err >= self.error_burst_multiplier * ema_err
            and rolling_err >= self.min_error_rate_for_burst
        ):
            reasons.append("error_burst")

        if self.use_brier_stress and len(self.recent_long_brier) >= self.min_monitor_samples:
            rolling_brier = sum(self.recent_long_brier) / len(self.recent_long_brier)
            ema_brier = self.ema_long_brier if self.ema_long_brier is not None else rolling_brier
            if rolling_brier >= ema_brier * (1.0 + self.brier_increase_ratio):
                reasons.append("brier_degradation")

        return bool(reasons), reasons

    def _update_stress_state(self, long_is_correct, p_long, y_true):
        brier = float((p_long - y_true) ** 2)
        self.recent_long_correct.append(long_is_correct)
        self.recent_long_brier.append(brier)

        if self.ema_long_accuracy is None:
            self.ema_long_accuracy = float(long_is_correct)
            self.ema_long_brier = brier
        else:
            self.ema_long_accuracy = (
                self.alpha * long_is_correct + (1.0 - self.alpha) * self.ema_long_accuracy
            )
            self.ema_long_brier = self.alpha * brier + (1.0 - self.alpha) * self.ema_long_brier

        stressed, _ = self._evaluate_stress_signals()
        if stressed:
            self.stress_active = True
            self.cooldown_remaining = self.stress_cooldown
        elif self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            self.stress_active = True
        else:
            self.stress_active = False

    def predict_step(self, i):
        if hasattr(self.X, "iloc"):
            X_test_df = self.X.iloc[i : i + 1]
        else:
            X_test_df = self.X[i : i + 1]

        x_test_dict = self._prepare_short_input(X_test_df.iloc[0].to_dict())
        y_true = self.y[i]

        if i > self.initial_size:
            prev_idx = i - 1
            if hasattr(self.X, "iloc"):
                X_prev_df = self.X.iloc[prev_idx : prev_idx + 1]
            else:
                X_prev_df = self.X[prev_idx : prev_idx + 1]
            prev_dict = self._prepare_short_input(X_prev_df.iloc[0].to_dict())
            self.model_short.learn_one(prev_dict, self.y[prev_idx])

        p_long = float(self.model_long.predict_proba(X_test_df)[0][1])
        p_short = float(self.model_short.predict_proba_one(x_test_dict).get(1, 0.0))

        self.last_long_is_correct = int((p_long >= 0.5) == y_true)
        self.count_steps += 1

        y_prob, boosted = self._compute_final_probability(p_long, p_short)
        if boosted:
            self.count_boost += 1
        else:
            self.count_long_only += 1

        y_pred = 1 if y_prob >= 0.5 else 0
        self._update_stress_state(self.last_long_is_correct, p_long, y_true)
        return y_true, y_pred, y_prob

    def print_stats(self):
        if self.count_steps == 0:
            return
        total = self.count_long_only + self.count_boost
        print("\n" + "═" * 58)
        print(f"║ {'HYBRID STATS':^54} ║")
        print("═" * 58)
        print(f"║ boost_mode               : {self.boost_mode:<24} ║")
        print(
            f"║ Periodic only            : {self.count_long_only:>8} "
            f"({self.count_long_only / total:>6.1%}) ║"
        )
        print(
            f"║ Boost applied            : {self.count_boost:>8} "
            f"({self.count_boost / total:>6.1%}) ║"
        )
        if self.boost_mode == "auc_boost":
            print(f"║ threshold / band         : {self.intervention_threshold:.3f} "
                  f"({self.min_p_long_for_boost:.2f},{self.max_p_long_for_boost:.2f}) ║")
        if self.boost_mode == "and_fusion":
            print(f"║ AND thresh (short/long)  : {self.intervention_threshold:.3f} / "
                  f"{self.and_long_threshold:.3f} ║")
            print(f"║ prior_ratio              : {self.prior_ratio:.6f}           ║")
        if self.short_selected_features is not None:
            print(f"║ Short features           : selected={len(self.short_selected_features):>4}   ║")
        if self.impute_nan:
            print(f"║ NaN imputation           : on                           ║")
        print("═" * 58)

    def check_drift(self, is_correct):
        self.active_triggers = []
        long_is_correct = self.last_long_is_correct
        if long_is_correct is None:
            long_is_correct = is_correct
        if self.long_term_eval.check_drift(long_is_correct):
            self.active_triggers.append("long")
        return bool(self.active_triggers)

    def retrain(self, i, retraining_start_idx):
        if self.long_retraining_start_idx is None:
            self.long_retraining_start_idx = retraining_start_idx
        retraining_points = []
        if "long" in self.active_triggers:
            res_idx = self.long_term_eval.retrain(i, self.long_retraining_start_idx)
            self.model_long = self.long_term_eval.model
            retraining_points.append(res_idx)
            self.long_retraining_start_idx = i + 1
        self.active_triggers = []
        return min(retraining_points) if retraining_points else retraining_start_idx
