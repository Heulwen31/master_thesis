import argparse
import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.factory import create_model
from src.trainner.sliding_evaluator import SlidingEvaluator
from src.trainner.periodic_evaluator import PeriodicEvaluator
from src.trainner.incremental_evaluator import IncrementalEvaluator
from src.trainner.hybrid_evaluator import HybridEvaluator
from src.evaluation.reporter import DriftReporter
from src.utils.config import get_trainner_config

def main():
    parser = argparse.ArgumentParser(description="Run the sliding window evaluation pipeline.")
    parser.add_argument("--dataset", type=str, default="ieee_cis", choices=["ieee_cis", "creditcard", "fraud_ecommerce"], help="Dataset to evaluate on")
    parser.add_argument("--model", type=str, default="xgboost", choices=["xgboost", "lightgbm", "catboost"], help="Model to use")
    parser.add_argument("--method", type=str, default="sliding", choices=["sliding", "periodic", "incremental", "hybrid", "all"], help="Evaluation method")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap rows after load (overrides pipeline subsample)")
    args = parser.parse_args()

    # Load config
    config = get_trainner_config()
    pipeline_config = config.get("pipeline", {})
    mode = pipeline_config.get("mode", "subsample")
    
    # 1. Load Data
    data_path = os.path.join("data", "processed", f"{args.dataset}.parquet")
    
    if not os.path.exists(data_path):
        print(f"❌ Error: Data not found at {data_path}. Please run preprocessing first.")
        sys.exit(1)
        
    print(f"Loading {args.dataset} data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    if 'time' in df.columns:
        df = df.sort_values(by='time')
        df = df.drop(columns=['time'])
        
    if args.max_samples is not None:
        df = df.head(args.max_samples)
        print(f"⚠️  Capped to {len(df)} records (--max-samples).")
    elif mode == "subsample":
        size = pipeline_config.get("subsample_size", 50000)
        print(f"⚠️  SUBSAMPLE mode: taking first {size} records.")
        df = df.head(size)
    else:
        print(f"✅ FULL mode: using {len(df)} records.")

    # Convert non-numeric columns to categorical codes for better compatibility
    cat_cols = df.select_dtypes(exclude=['number', 'bool']).columns
    for col in cat_cols:
        # Fill NaNs with a placeholder before encoding
        df[col] = df[col].fillna("Unknown").astype('category')
        # LightGBM prefers integers starting from 0
        df[col] = df[col].cat.codes.astype('int32')
        # Ensure it's still treated as category for models that support it
        df[col] = df[col].astype('category')

    y = df['target'].values
    X_df = df.drop(columns=['target'])
    
    # 2. Initialize Components & 3. Initial Training (Warm-up)
    if args.method == "all":
        from src.evaluation.reporter import ComparisonReporter
        methods = ["sliding", "periodic", "incremental", "hybrid"]
        evaluators = {}
        
        for m_name in methods:
            model_inst = create_model(args.model)
            if m_name == "sliding":
                evaluators[m_name] = SlidingEvaluator(model_inst, X_df, y)
            elif m_name == "periodic":
                evaluators[m_name] = PeriodicEvaluator(model_inst, X_df, y)
            elif m_name == "incremental":
                evaluators[m_name] = IncrementalEvaluator(model_inst, X_df, y)
            elif m_name == "hybrid":
                evaluators[m_name] = HybridEvaluator(model_inst, X_df, y, dataset_name=args.dataset)
        
        reporter = ComparisonReporter(args.dataset, args.model, methods)
        
        initial_size = None
        for m_name in methods:
            size = evaluators[m_name].init_train()
            if initial_size is None:
                initial_size = size
        
        reporter.initial_size = initial_size

        # 4. Evaluation Loop
        print(f"Starting prequential evaluation for ALL methods (Warm-up size: {initial_size})...")
        n_samples = len(X_df)
        retraining_start_idxs = {m_name: 0 for m_name in methods}
        
        for i in range(initial_size, n_samples):
            for m_name in methods:
                evaluator = evaluators[m_name]
                # Step 1: Predict (Test)
                y_true, y_pred, y_prob = evaluator.predict_step(i)
                
                # Step 2: Record Metrics
                reporter.add_step(m_name, y_true, y_pred, y_prob)
                
                # Step 3: Check for Drift
                is_correct = 1 if y_pred == y_true else 0
                drift_detected = evaluator.check_drift(is_correct)
                
                if drift_detected:
                    # Step 4: Retrain if needed
                    retraining_point = evaluator.retrain(i, retraining_start_idxs[m_name])
                    reporter.add_drift(m_name, i, retraining_point)
                    retraining_start_idxs[m_name] = i + 1
                    
            if i % 1000 == 0:
                print(f"Processed samples: {i}/{n_samples}...", end="\r")

        # 5. Generate Final Report
        if "hybrid" in evaluators and hasattr(evaluators["hybrid"], 'print_stats'):
            evaluators["hybrid"].print_stats()
            
        # Pass evaluation features for error analysis
        X_eval = X_df.iloc[initial_size:]
        reporter.generate_report(X_eval=X_eval)

    else:
        model = create_model(args.model)
        
        if args.method == "sliding":
            evaluator = SlidingEvaluator(model, X_df, y)
        elif args.method == "periodic":
            evaluator = PeriodicEvaluator(model, X_df, y)
        elif args.method == "incremental":
            evaluator = IncrementalEvaluator(model, X_df, y)
        elif args.method == "hybrid":
            evaluator = HybridEvaluator(model, X_df, y, dataset_name=args.dataset)
            
        reporter = DriftReporter(args.dataset, args.model, args.method)
        
        # 3. Initial Training (Warm-up)
        initial_size = evaluator.init_train()
        
        # 4. Evaluation Loop
        print(f"Starting prequential evaluation for {args.method}...")
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
        if args.method == "hybrid" and hasattr(evaluator, 'print_stats'):
            evaluator.print_stats()
            
        # Pass evaluation features for error analysis
        X_eval = X_df.iloc[initial_size:]
        reporter.generate_report(X_eval=X_eval)

if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()
