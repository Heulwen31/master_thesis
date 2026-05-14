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
    def __init__(self, dataset_name, model_name):
        self.dataset_name = dataset_name
        self.model_name = model_name
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
        
        # 1. Error Timeline & Drift Points
        plt.figure(figsize=(15, 5))
        plt.plot(np.convolve(is_correct, np.ones(100)/100, mode='valid'), label='Rolling Accuracy (Window=100)')
        if X_eval is not None:
            for drift in self.drifts:
                plt.axvline(x=drift['detected_at'] - (len(X_eval) - len(is_correct)), color='r', alpha=0.3, linestyle='--')
        plt.title(f"Evaluation Accuracy Timeline - {self.dataset_name}")
        plt.xlabel("Sample Index")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(f"outputs/plots/timeline_{self.dataset_name}.png")
        plt.close()

        # 2. Probability Distribution for Errors vs Successes
        plt.figure(figsize=(10, 6))
        sns.kdeplot(y_prob[is_correct], label='Correct Predictions', fill=True, alpha=0.5)
        sns.kdeplot(y_prob[~is_correct], label='Errors', fill=True, alpha=0.5)
        plt.title("Probability Distribution: Correct vs Incorrect")
        plt.xlabel("Predicted Probability (Class 1)")
        plt.ylabel("Density")
        plt.legend()
        plt.savefig(f"outputs/plots/prob_dist_{self.dataset_name}.png")
        plt.close()

        # 3. Confidence vs Error (Margin analysis)
        confidence = np.abs(y_prob - 0.5) * 2
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=is_correct, y=confidence)
        plt.title("Confidence (Margin) for Correct vs Incorrect Predictions")
        plt.xticks([0, 1], ['Incorrect', 'Correct'])
        plt.ylabel("Confidence (|p-0.5|*2)")
        plt.savefig(f"outputs/plots/confidence_analysis_{self.dataset_name}.png")
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
                plt.savefig(f"outputs/plots/feature_error_analysis_{self.dataset_name}.png")
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
        filename = f"outputs/report_{self.dataset_name}_{self.model_name}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)

        # Generate Diagnostic Plots
        print("Generating diagnostic visualizations...")
        self._generate_visualizations(y_true, y_pred, y_prob, X_eval)

        print("\n" + "═"*60)
        print(f"║ {'FINAL EVALUATION REPORT':^56} ║")
        print("═"*60)
        print(f"║ Dataset      : {self.dataset_name:<41} ║")
        print(f"║ Model        : {self.model_name:<41} ║")
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
        print(f"║ Full report saved to: {filename:<29} ║")
        print(f"║ Visualizations saved to: outputs/plots/             ║")
        print("═"*60 + "\n")
        
        return results
