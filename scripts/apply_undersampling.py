#!/usr/bin/env python3
"""
Example script showing how to apply undersampling to fraud detection data.
Run this BEFORE or AFTER the main preprocessing pipeline.
"""

import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess.undersampling import (
    TimeAwareUndersampler,
    StratifiedTimeWindowUndersampler,
    RecencyBiasedUndersampler
)
from src.utils.config import get_data_config, get_trainner_config


def apply_undersampling_to_preprocessed(dataset_name, method="time_aware", sampling_ratio=0.5, keep_original=False):
    """
    Apply undersampling to already preprocessed parquet files.
    
    Args:
        dataset_name: Name of dataset
        method: Undersampling method
        sampling_ratio: Sampling ratio for majority class
        keep_original: If True, create backup before overwriting
    """
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "processed"
    )
    
    input_path = os.path.join(data_dir, f"{dataset_name}.parquet")
    output_path = os.path.join(data_dir, f"{dataset_name}.parquet")
    backup_path = os.path.join(data_dir, f"{dataset_name}_backup.parquet")
    
    if not os.path.exists(input_path):
        print(f"❌ File not found: {input_path}")
        return
    
    # Create backup if requested
    if keep_original and not os.path.exists(backup_path):
        import shutil
        shutil.copy(input_path, backup_path)
        print(f"💾 Backup created: {backup_path}")
    
    print(f"📊 Loading {dataset_name} from {input_path}...")
    df = pd.read_parquet(input_path)
    print(f"   Original shape: {df.shape}")
    print(f"   Class distribution:\n{df['target'].value_counts()}")
    
    # Split features and target
    X = df.drop(columns=['target'])
    y = df['target'].values
    
    # Apply undersampling
    print(f"\n🔄 Applying {method} undersampling (ratio={sampling_ratio})...")
    
    if method == "time_aware":
        undersampler = TimeAwareUndersampler(
            sampling_ratio=sampling_ratio,
            seed=42
        )
    elif method == "stratified_window":
        undersampler = StratifiedTimeWindowUndersampler(
            window_size=5000,
            sampling_ratio=sampling_ratio,
            seed=42
        )
    elif method == "recency_biased":
        undersampler = RecencyBiasedUndersampler(
            sampling_ratio=sampling_ratio,
            recency_weight=0.7,
            seed=42
        )
    else:
        print(f"❌ Unknown method: {method}")
        return
    
    X_sampled, y_sampled = undersampler.fit_transform(X, y)
    
    # Reconstruct dataframe
    df_sampled = pd.concat([X_sampled, pd.Series(y_sampled, name='target', index=X_sampled.index)], axis=1)
    
    print(f"\n   Sampled shape: {df_sampled.shape}")
    print(f"   Class distribution after undersampling:")
    print(f"{pd.Series(y_sampled).value_counts()}")
    print(f"   Reduction: {100 * (1 - len(df_sampled) / len(df)):.1f}%")
    
    # Save to file (overwrite original)
    print(f"\n💾 Saving undersampled data to {output_path}...")
    df_sampled.to_parquet(output_path, engine='pyarrow', index=False)
    print(f"✅ Done! Original file has been replaced with undersampled data.")


def apply_undersampling_with_preprocessing(dataset_name):
    """
    Example: integrate undersampling directly in preprocessing pipeline.
    This shows how to modify datasets.py to include undersampling.
    """
    from src.data.loader import get_loader
    
    loader = get_loader(dataset_name)
    config = get_data_config()
    undersample_cfg = config.get('undersampling', {})
    
    # Load and preprocess
    df = loader.load_raw()
    
    # Apply time conversion (assuming your preprocessing does this)
    # df = preprocess_time(df)  # Your time processing function
    
    if undersample_cfg.get('enabled', False):
        method = undersample_cfg.get('method', 'time_aware')
        sampling_ratio = undersample_cfg.get('sampling_ratio', 0.5)
        
        print(f"📊 {dataset_name} original shape: {df.shape}")
        
        X = df.drop(columns=['target'], errors='ignore')
        y = df['target'].values if 'target' in df.columns else None
        
        if y is not None:
            # Select undersampler
            if method == "time_aware":
                undersampler = TimeAwareUndersampler(sampling_ratio=sampling_ratio)
            elif method == "stratified_window":
                undersampler = StratifiedTimeWindowUndersampler(
                    window_size=undersample_cfg.get('window_size', 5000),
                    sampling_ratio=sampling_ratio
                )
            elif method == "recency_biased":
                undersampler = RecencyBiasedUndersampler(
                    sampling_ratio=sampling_ratio,
                    recency_weight=undersample_cfg.get('recency_weight', 0.7)
                )
            else:
                print(f"Unknown undersampling method: {method}")
                return df
            
            # Apply undersampling
            X_sampled, y_sampled = undersampler.fit_transform(X, y)
            
            # Reconstruct dataframe
            df = pd.concat([X_sampled, pd.Series(y_sampled, name='target')], axis=1)
            
            print(f"✅ After undersampling: {df.shape}")
            print(f"   Class distribution: {pd.Series(y_sampled).value_counts().to_dict()}")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Apply undersampling to fraud detection datasets")
    parser.add_argument("--dataset", type=str, default="creditcard",
                        choices=["creditcard", "ieee_cis", "fraud_ecommerce"],
                        help="Dataset name")
    parser.add_argument("--method", type=str, default="time_aware",
                        choices=["time_aware", "stratified_window", "recency_biased"],
                        help="Undersampling method")
    parser.add_argument("--ratio", type=float, default=0.5,
                        help="Sampling ratio for majority class (0 < ratio <= 1)")
    parser.add_argument("--keep-backup", action="store_true",
                        help="Create backup of original file before overwriting")
    
    args = parser.parse_args()
    
    apply_undersampling_to_preprocessed(args.dataset, args.method, args.ratio, args.keep_backup)
    
    print("\n" + "="*60)
    print("ℹ️  To use undersampled data in training:")
    print(f"   python main.py --dataset {args.dataset}  # (change data_path if needed)")
    print("="*60)
