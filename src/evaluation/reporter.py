import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, average_precision_score

class DriftReporter:
    """
    Handles collecting metrics and generating a detailed report for the evaluation.
    """
    def __init__(self, dataset_name, model_name, method_name):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.method_name = method_name
        self.predictions = []
        self.probabilities = []
        self.ground_truth = []
        self.drifts = []

    def add_step(self, y_true, y_pred, y_prob):
        """Collects a single prediction result."""
        self.ground_truth.append(y_true)
        self.predictions.append(y_pred)
        self.probabilities.append(y_prob)

    def add_drift(self, index, retraining_point):
        """Records a drift event."""
        self.drifts.append({
            'detected_at': int(index),
            'retrained_from': int(retraining_point)
        })

    def _generate_visualizations(self, y_true, y_pred, y_prob, X_eval):
        """Generates diagnostic plots for error analysis."""
        os.makedirs("outputs/plots", exist_ok=True)
        is_correct = (y_pred == y_true)
        prefix = f"{self.method_name}_{self.dataset_name}"
        
        # 1. Error Timeline, Fraud Rate & Drift Points
        fig, ax1 = plt.subplots(figsize=(15, 6))
        
        # Accuracy plot (Primary Y-axis)
        rolling_acc = np.convolve(is_correct, np.ones(100)/100, mode='valid')
        ax1.plot(rolling_acc, color='#1f77b4', label='Rolling Accuracy (Window=100)', linewidth=1.5)
        ax1.set_xlabel("Sample Index", fontsize=10)
        ax1.set_ylabel("Accuracy", color='#1f77b4', fontsize=10)
        ax1.tick_params(axis='y', labelcolor='#1f77b4')
        ax1.grid(True, alpha=0.3)
        
        # Fraud Rate plot (Secondary Y-axis)
        ax2 = ax1.twinx()
        rolling_fraud = np.convolve(y_true, np.ones(100)/100, mode='valid')
        ax2.plot(rolling_fraud, color='#d62728', alpha=0.6, label='Rolling Fraud Rate', linewidth=1.2, linestyle='--')
        ax2.set_ylabel("Fraud Rate", color='#d62728', fontsize=10)
        ax2.tick_params(axis='y', labelcolor='#d62728')
        
        # Drift vertical lines
        if X_eval is not None:
            first_drift = True
            for drift in self.drifts:
                label = 'Drift Detected' if first_drift else ""
                ax1.axvline(x=drift['detected_at'] - (len(X_eval) - len(is_correct)), 
                           color='gray', alpha=0.4, linestyle=':', label=label)
                first_drift = False
                
        plt.title(f"Performance & Fraud Prevalence - {self.dataset_name} ({self.method_name.upper()})", fontsize=14, pad=20)
        
        # Combine legends from both axes
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        
        plt.tight_layout()
        plt.savefig(f"outputs/plots/timeline_{prefix}.png", bbox_inches='tight', dpi=150)
        plt.close()

        # 2. Probability Distribution for Errors vs Successes
        plt.figure(figsize=(10, 6))
        sns.kdeplot(y_prob[is_correct], label='Correct Predictions', fill=True, alpha=0.5)
        sns.kdeplot(y_prob[~is_correct], label='Errors', fill=True, alpha=0.5)
        plt.title(f"Prob Distribution: Correct vs Incorrect ({self.method_name.upper()})")
        plt.xlabel("Predicted Probability (Class 1)")
        plt.ylabel("Density")
        plt.legend()
        plt.savefig(f"outputs/plots/prob_dist_{prefix}.png")
        plt.close()

        # 3. Confidence vs Error (Margin analysis)
        confidence = np.abs(y_prob - 0.5) * 2
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=is_correct, y=confidence)
        plt.title(f"Confidence (Margin) for Correct vs Incorrect ({self.method_name.upper()})")
        plt.xticks([0, 1], ['Incorrect', 'Correct'])
        plt.ylabel("Confidence (|p-0.5|*2)")
        plt.savefig(f"outputs/plots/confidence_analysis_{prefix}.png")
        plt.close()

        # 4. Feature Analysis for Errors (Top 5 features)
        if X_eval is not None:
            num_cols = X_eval.select_dtypes(include=[np.number]).columns[:5]
            if len(num_cols) > 0:
                fig, axes = plt.subplots(1, len(num_cols), figsize=(20, 5))
                if len(num_cols) == 1: axes = [axes]
                
                for i, col in enumerate(num_cols):
                    sns.kdeplot(X_eval.iloc[is_correct][col], ax=axes[i], label='Correct', fill=True, alpha=0.3)
                    sns.kdeplot(X_eval.iloc[~is_correct][col], ax=axes[i], label='Error', fill=True, alpha=0.3)
                    axes[i].set_title(f"Dist: {col}")
                    axes[i].legend()
                
                plt.tight_layout()
                plt.savefig(f"outputs/plots/feature_error_analysis_{prefix}.png")
                plt.close()

    def generate_report(self, X_eval=None):
        """Calculates final metrics, saves to JSON, and prints a summary."""
        y_true = np.array(self.ground_truth)
        y_pred = np.array(self.predictions)
        y_prob = np.array(self.probabilities)
        
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        
        # Calculate AUC metrics
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
            pr_auc = float(average_precision_score(y_true, y_prob))
        except:
            roc_auc = 0.0
            pr_auc = 0.0

        results = {
            "dataset": self.dataset_name,
            "model": self.model_name,
            "method": self.method_name,
            "metrics": {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc
            },
            "total_drifts": len(self.drifts),
            "drift_details": self.drifts
        }

        # Save to JSON
        os.makedirs("outputs", exist_ok=True)
        filename = f"outputs/report_{self.method_name}_{self.dataset_name}_{self.model_name}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)

        # Generate Diagnostic Plots
        print(f"Generating diagnostic visualizations for {self.method_name}...")
        self._generate_visualizations(y_true, y_pred, y_prob, X_eval)

        print("\n" + "═"*60)
        print(f"║ {'FINAL EVALUATION REPORT':^56} ║")
        print("═"*60)
        print(f"║ Dataset      : {self.dataset_name:<41} ║")
        print(f"║ Model        : {self.model_name:<41} ║")
        print(f"║ Method       : {self.method_name:<41} ║")
        print("╟" + "─"*58 + "╢")
        print(f"║ {'METRICS':^56} ║")
        print("╟" + "─"*58 + "╢")
        print(f"║ Accuracy     : {acc:>41.4f} ║")
        print(f"║ Precision    : {prec:>41.4f} ║")
        print(f"║ Recall       : {rec:>41.4f} ║")
        print(f"║ F1-Score     : {f1:>41.4f} ║")
        print(f"║ ROC-AUC      : {roc_auc:>41.4f} ║")
        print(f"║ PR-AUC       : {pr_auc:>41.4f} ║")
        print("╟" + "─"*58 + "╢")
        print(f"║ {'DRIFT ANALYSIS':^56} ║")
        print("╟" + "─"*58 + "╢")
        print(f"║ Total Drifts : {len(self.drifts):>41} ║")
        print(f"║ Full report saved to: {os.path.basename(filename):<29} ║")
        print(f"║ Visualizations saved to: outputs/plots/             ║")
        print("═"*60 + "\n")
        
        return results


class ComparisonReporter:
    """
    Handles collecting metrics and generating comparison reports/plots for multiple methods.
    """
    def __init__(self, dataset_name, model_name, methods):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.methods = methods
        self.predictions = {m: [] for m in methods}
        self.probabilities = {m: [] for m in methods}
        self.ground_truth = {m: [] for m in methods}
        self.drifts = {m: [] for m in methods}
        self.initial_size = 0

    def add_step(self, method, y_true, y_pred, y_prob):
        """Collects a single prediction result for a specific method."""
        self.ground_truth[method].append(y_true)
        self.predictions[method].append(y_pred)
        self.probabilities[method].append(y_prob)

    def add_drift(self, method, index, retraining_point):
        """Records a drift event for a specific method."""
        self.drifts[method].append({
            'detected_at': int(index),
            'retrained_from': int(retraining_point)
        })

    def _generate_comparison_visualizations(self, results, X_eval):
        """Generates unified visual dashboards comparing all methods."""
        os.makedirs("outputs/plots", exist_ok=True)
        colors = {
            'sliding': '#1f77b4',     # blue
            'periodic': '#ff7f0e',    # orange
            'incremental': '#2ca02c', # green
            'hybrid': '#9467bd'       # purple
        }
        styles = {
            'sliding': {'linestyle': '-', 'linewidth': 3.0},
            'periodic': {'linestyle': '-', 'linewidth': 3.0},
            'incremental': {'linestyle': '--', 'linewidth': 1.5},
            'hybrid': {'linestyle': ':', 'linewidth': 2.0}
        }
        
        # 1. Dashboard (2x2)
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        
        # Top-Left: Rolling Accuracy (Window=1000)
        ax = axes[0, 0]
        y_true_first = np.array(self.ground_truth[self.methods[0]])
        rolling_fraud = np.convolve(y_true_first, np.ones(1000)/1000, mode='valid')
        ax_twin = ax.twinx()
        ax_twin.plot(rolling_fraud, color='grey', alpha=0.2, linestyle='--', label='Rolling Fraud Rate (Window=1000)')
        ax_twin.set_ylabel("Fraud Rate", color='grey', alpha=0.5)
        ax_twin.tick_params(axis='y', labelcolor='grey')
        
        for m in self.methods:
            is_correct = (np.array(self.predictions[m]) == np.array(self.ground_truth[m]))
            rolling_acc = np.convolve(is_correct, np.ones(1000)/1000, mode='valid')
            ax.plot(rolling_acc, color=colors[m], label=f'{m.upper()}', **styles[m])
            
        ax.set_title("Rolling Accuracy Over Time (Window=1000)", fontsize=12, fontweight='bold')
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Accuracy")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower left')
        
        # Top-Right: Cumulative Errors
        ax = axes[0, 1]
        for m in self.methods:
            is_incorrect = (np.array(self.predictions[m]) != np.array(self.ground_truth[m]))
            cum_errors = np.cumsum(is_incorrect)
            ax.plot(cum_errors, color=colors[m], label=f'{m.upper()}', **styles[m])
        ax.set_title("Cumulative Errors Over Time (Lower is Better)", fontsize=12, fontweight='bold')
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Cumulative Error Count")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        
        # Bottom-Left: Radar (Spider) Chart
        fig.delaxes(axes[1, 0])
        ax_radar = fig.add_subplot(2, 2, 3, polar=True)
        
        categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        ax_radar.set_theta_offset(np.pi / 2)
        ax_radar.set_theta_direction(-1)
        
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(categories, fontsize=9, fontweight='semibold')
        
        ax_radar.set_rlabel_position(0)
        ax_radar.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax_radar.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=7)
        ax_radar.set_ylim(0, 1.0)
        
        for m in self.methods:
            values = [
                results[m]['accuracy'],
                results[m]['precision'],
                results[m]['recall'],
                results[m]['f1_score'],
                results[m]['roc_auc']
            ]
            values += values[:1]
            ax_radar.plot(angles, values, color=colors[m], linewidth=1.5, linestyle='solid', label=m.upper())
            ax_radar.fill(angles, values, color=colors[m], alpha=0.08)
            
        ax_radar.set_title("Overall Metric Comparison (Radar Chart)", fontsize=12, fontweight='bold', pad=15)
        ax_radar.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=9)
        
        # Bottom-Right: Cumulative Retrains (Computational Cost)
        ax = axes[1, 1]
        for m in self.methods:
            n_eval_samples = len(self.predictions[m])
            retrain_timeline = np.zeros(n_eval_samples)
            
            for drift in self.drifts[m]:
                rel_idx = drift['detected_at'] - self.initial_size
                if 0 <= rel_idx < n_eval_samples:
                    retrain_timeline[rel_idx] = 1
            cum_retrains = np.cumsum(retrain_timeline)
            ax.plot(cum_retrains, color=colors[m], label=f'{m.upper()} (Total: {len(self.drifts[m])})', **styles[m])
            
        ax.set_title("Cumulative Retraining Events (Lower is Better)", fontsize=12, fontweight='bold')
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Cumulative Retrain Count")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        
        plt.suptitle(f"Method Comparison Dashboard - {self.dataset_name} ({self.model_name.upper()})", fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # Save dataset-specific dashboard
        specific_dash = f"outputs/plots/comparison_dashboard_{self.dataset_name}_{self.model_name}.png"
        plt.savefig(specific_dash, bbox_inches='tight', dpi=150)
        # Save generic latest dashboard
        plt.savefig("outputs/plots/comparison_dashboard.png", bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Comparison Dashboard saved to: {specific_dash}")
        
        # 2. Curves Plot (ROC & PR)
        from sklearn.metrics import roc_curve, precision_recall_curve
        fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(16, 7))
        
        for m in self.methods:
            y_true = np.array(self.ground_truth[m])
            y_prob = np.array(self.probabilities[m])
            
            # ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            ax_roc.plot(fpr, tpr, color=colors[m], label=f"{m.upper()} (AUC = {results[m]['roc_auc']:.4f})", **styles[m])
            
            # PR curve
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            ax_pr.plot(recall, precision, color=colors[m], label=f"{m.upper()} (AUC = {results[m]['pr_auc']:.4f})", **styles[m])
            
        ax_roc.plot([0, 1], [0, 1], color='navy', linestyle='--', alpha=0.5)
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('ROC Curve Comparison', fontsize=12, fontweight='bold')
        ax_roc.legend(loc="lower right")
        ax_roc.grid(True, alpha=0.3)
        
        ax_pr.set_xlim([0.0, 1.0])
        ax_pr.set_ylim([0.0, 1.05])
        ax_pr.set_xlabel('Recall')
        ax_pr.set_ylabel('Precision')
        ax_pr.set_title('Precision-Recall Curve Comparison', fontsize=12, fontweight='bold')
        ax_pr.legend(loc="lower left")
        ax_pr.grid(True, alpha=0.3)
        
        plt.suptitle(f"ROC & PR Curves Comparison - {self.dataset_name} ({self.model_name.upper()})", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save specific curve plot
        specific_curves = f"outputs/plots/comparison_curves_{self.dataset_name}_{self.model_name}.png"
        plt.savefig(specific_curves, bbox_inches='tight', dpi=150)
        # Save generic latest curve plot
        plt.savefig("outputs/plots/comparison_curves.png", bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Comparison Curves saved to: {specific_curves}")

    def generate_report(self, X_eval=None):
        """Calculates final metrics for all methods, saves to JSON, and prints comparative summary."""
        results = {}
        
        for m in self.methods:
            y_true = np.array(self.ground_truth[m])
            y_pred = np.array(self.predictions[m])
            y_prob = np.array(self.probabilities[m])
            
            acc = float(accuracy_score(y_true, y_pred))
            prec = float(precision_score(y_true, y_pred, zero_division=0))
            rec = float(recall_score(y_true, y_pred, zero_division=0))
            f1 = float(f1_score(y_true, y_pred, zero_division=0))
            
            try:
                roc_auc = float(roc_auc_score(y_true, y_prob))
                pr_auc = float(average_precision_score(y_true, y_prob))
            except:
                roc_auc = 0.0
                pr_auc = 0.0
                
            results[m] = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "total_drifts": len(self.drifts[m]),
                "drift_details": self.drifts[m]
            }

        # Save to JSON
        os.makedirs("outputs", exist_ok=True)
        filename = f"outputs/report_all_{self.dataset_name}_{self.model_name}.json"
        with open(filename, 'w') as f:
            json.dump({
                "dataset": self.dataset_name,
                "model": self.model_name,
                "methods_compared": self.methods,
                "results": results
            }, f, indent=4)
            
        # Print comparison table
        print("\n" + "═"*78)
        print(f"║ {'COMPARATIVE EVALUATION REPORT':^74} ║")
        print("═"*78)
        print(f"║ Dataset      : {self.dataset_name:<59} ║")
        print(f"║ Model        : {self.model_name:<59} ║")
        print("╟" + "─"*76 + "╢")
        print(f"║ {'Method':<12} │ {'Accuracy':<8} │ {'Precision':<9} │ {'Recall':<8} │ {'F1-Score':<8} │ {'ROC-AUC':<8} │ {'Drifts':<6} ║")
        print("╟" + "─"*76 + "╢")
        for m in self.methods:
            res = results[m]
            print(f"║ {m.upper():<12} │ {res['accuracy']:>8.4f} │ {res['precision']:>9.4f} │ {res['recall']:>8.4f} │ {res['f1_score']:>8.4f} │ {res['roc_auc']:>8.4f} │ {res['total_drifts']:>6} ║")
        print("═"*78 + "\n")

        # Generate Diagnostic Plots
        print("Generating diagnostic comparison visualizations...")
        self._generate_comparison_visualizations(results, X_eval)
        
        return results

