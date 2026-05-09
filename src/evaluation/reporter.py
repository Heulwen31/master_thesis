import pandas as pd
import numpy as np
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
            'detected_at': index,
            'retrained_from': retraining_point
        })

    def generate_report(self):
        """Calculates final metrics and prints a premium summary."""
        y_true = np.array(self.ground_truth)
        y_pred = np.array(self.predictions)
        y_prob = np.array(self.probabilities)
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Calculate AUC metrics
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
            pr_auc = average_precision_score(y_true, y_prob)
        except:
            roc_auc = 0.0
            pr_auc = 0.0

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
        
        if self.drifts:
            print("║ Drift Details:                                           ║")
            for drift in self.drifts:
                detail = f"At {drift['detected_at']} (Retrained from {drift['retrained_from']})"
                print(f"║   • {detail:<52} ║")
        
        print("═"*60 + "\n")
        
        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "drifts": self.drifts
        }
