"""
Helper module to integrate undersampling into the training pipeline.
"""

import pandas as pd
import numpy as np
from src.utils.config import get_data_config
from src.preprocess.undersampling import (
    TimeAwareUndersampler,
    StratifiedTimeWindowUndersampler,
    RecencyBiasedUndersampler
)


def get_undersampler(method="time_aware", config_dict=None):
    """
    Factory function to create undersampler based on method.
    
    Args:
        method: "time_aware", "stratified_window", or "recency_biased"
        config_dict: Optional config dictionary with parameters
        
    Returns:
        Undersampler instance
    """
    if config_dict is None:
        config_dict = {}
    
    sampling_ratio = config_dict.get('sampling_ratio', 0.5)
    seed = config_dict.get('seed', 42)
    
    if method == "time_aware":
        return TimeAwareUndersampler(
            sampling_ratio=sampling_ratio,
            seed=seed
        )
    elif method == "stratified_window":
        return StratifiedTimeWindowUndersampler(
            window_size=config_dict.get('window_size', 5000),
            sampling_ratio=sampling_ratio,
            seed=seed
        )
    elif method == "recency_biased":
        return RecencyBiasedUndersampler(
            sampling_ratio=sampling_ratio,
            recency_weight=config_dict.get('recency_weight', 0.7),
            seed=seed
        )
    else:
        raise ValueError(f"Unknown undersampling method: {method}")


def apply_undersampling_to_data(X_df, y, config=None, verbose=True):
    """
    Apply undersampling to data if enabled in config.
    
    Args:
        X_df: Feature dataframe
        y: Target labels
        config: Config dict with undersampling settings (from get_data_config())
        verbose: Print progress messages
        
    Returns:
        X_undersampled, y_undersampled (or original if disabled)
    """
    if config is None:
        config = get_data_config()
    
    undersample_cfg = config.get('undersampling', {})
    
    if not undersample_cfg.get('enabled', False):
        return X_df, y
    
    if verbose:
        print(f"📊 Undersampling enabled")
        print(f"   Original size: {len(X_df)}")
        print(f"   Class distribution: {np.bincount(y.astype(int))}")
    
    method = undersample_cfg.get('method', 'time_aware')
    undersampler = get_undersampler(method, undersample_cfg)
    
    X_undersampled, y_undersampled = undersampler.fit_transform(X_df, y)
    
    if verbose:
        print(f"   After undersampling: {len(X_undersampled)}")
        print(f"   Class distribution: {np.bincount(y_undersampled.astype(int))}")
        print(f"   Reduction: {100 * (1 - len(X_undersampled) / len(X_df)):.1f}%")
        print()
    
    return X_undersampled, y_undersampled


def print_undersampling_stats(X_before, y_before, X_after, y_after):
    """
    Print detailed statistics about undersampling impact.
    """
    print("📊 Undersampling Statistics:")
    print(f"  Original samples: {len(X_before)}")
    print(f"  After undersampling: {len(X_after)}")
    print(f"  Removed: {len(X_before) - len(X_after)} ({100 * (1 - len(X_after) / len(X_before)):.1f}%)")
    print(f"  ")
    print(f"  Before - Class distribution:")
    for label in np.unique(y_before):
        count = np.sum(y_before == label)
        pct = 100 * count / len(y_before)
        print(f"    Class {int(label)}: {count:6d} ({pct:5.2f}%)")
    print(f"  ")
    print(f"  After - Class distribution:")
    for label in np.unique(y_after):
        count = np.sum(y_after == label)
        pct = 100 * count / len(y_after)
        print(f"    Class {int(label)}: {count:6d} ({pct:5.2f}%)")
