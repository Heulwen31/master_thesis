import argparse
import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.factory import create_model
from src.trainner.online_evaluator import OnlineEvaluator
from src.trainner.reporter import DriftReporter
from src.utils.config import get_trainner_config

def main():
    parser = argparse.ArgumentParser(description="Run the ADDM evaluation pipeline.")
    parser.add_argument("--dataset", type=str, default="ieee_cis", choices=["ieee_cis", "creditcard"], help="Dataset to evaluate on")
    parser.add_argument("--model", type=str, default="xgboost", choices=["xgboost", "lightgbm", "catboost"], help="Model to use")
    args = parser.parse_args()

    # Load config
    config = get_trainner_config()
    pipeline_config = config.get("pipeline", {})
    mode = pipeline_config.get("mode", "subsample")
    
    # 1. Load Data
    data_path = os.path.join("data", "processed", f"{args.dataset}.parquet")
    if not os.path.exists(data_path):
        print(f"❌ Error: Processed data not found at {data_path}. Please run preprocessing first.")
        sys.exit(1)
        
    print(f"Loading {args.dataset} data...")
    df = pd.read_parquet(data_path)
    
    if 'time' in df.columns:
        df = df.sort_values(by='time')
        df = df.drop(columns=['time'])
        
    if mode == "subsample":
        subsample_size = pipeline_config.get("subsample_size", 50000)
        print(f"⚠️  Running in SUBSAMPLE mode: taking first {subsample_size} records.")
        df = df.head(subsample_size)
    else:
        print(f"✅ Running in FULL mode: using all {len(df)} records.")

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')

    y = df['target'].values
    X_df = df.drop(columns=['target'])
    
    # 2. Initialize Components
    model = create_model(args.model)
    evaluator = OnlineEvaluator(model, X_df, y)
    reporter = DriftReporter(args.dataset, args.model)
    
    # 3. Initial Training (Warm-up)
    initial_size = evaluator.init_train()
    
    # 4. Evaluation Loop (Moved to main for better control)
    print("Starting prequential evaluation...")
    n_samples = len(X_df)
    retraining_start_idx = 0
    
    for i in range(initial_size, n_samples):
        # Step 1: Predict (Test)
        y_true, y_pred, y_prob = evaluator.predict_step(i)
        
        # Step 2: Record Metrics
        reporter.add_step(y_true, y_pred, y_prob)
        
        # Step 3: Check for Drift
        is_correct = 1 if y_pred == y_true else 0
        drift_detected = evaluator.check_drift(is_correct)
        
        if drift_detected:
            # Step 4: Retrain if needed
            retraining_point = evaluator.retrain(i, retraining_start_idx)
            reporter.add_drift(i, retraining_point)
            retraining_start_idx = i + 1
            
        if i % 1000 == 0:
            print(f"Processed samples: {i}/{n_samples}...", end="\r")

    # 5. Generate Final Report
    reporter.generate_report()

if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()
