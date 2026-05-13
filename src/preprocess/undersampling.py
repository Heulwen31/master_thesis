import numpy as np
import pandas as pd
from sklearn.utils import shuffle as sklearn_shuffle


class TimeAwareUndersampler:
    """
    Undersamples the majority class while respecting temporal order.
    Suitable for streaming data with concept drift.
    """
    
    def __init__(self, sampling_ratio=0.5, minority_label=1, seed=42):
        """
        Args:
            sampling_ratio: Fraction of majority class to keep (0 < ratio <= 1)
            minority_label: Label value for the minority class (default: 1 for fraud)
            seed: Random seed for reproducibility
        """
        self.sampling_ratio = sampling_ratio
        self.minority_label = minority_label
        self.majority_label = 1 - minority_label
        self.seed = seed
        
    def fit(self, X, y):
        """No fitting required for undersampling."""
        return self
    
    def transform(self, X, y):
        """
        Undersample majority class while preserving temporal order.
        
        Args:
            X: Features (DataFrame or array)
            y: Labels (array-like)
            
        Returns:
            X_sampled, y_sampled: Undersampled data
        """
        if isinstance(X, pd.DataFrame):
            indices = X.index.values
            X_array = X.values
        else:
            indices = np.arange(len(X))
            X_array = np.asarray(X)
        
        y_array = np.asarray(y)
        
        # Identify minority and majority indices
        minority_idx = np.where(y_array == self.minority_label)[0]
        majority_idx = np.where(y_array == self.majority_label)[0]
        
        # Calculate how many majority samples to keep
        n_majority_keep = int(len(majority_idx) * self.sampling_ratio)
        n_majority_keep = max(1, n_majority_keep)  # Keep at least 1
        
        # Randomly select majority samples to keep (preserves random state)
        np.random.seed(self.seed)
        selected_majority_idx = np.random.choice(
            majority_idx, size=min(n_majority_keep, len(majority_idx)), replace=False
        )
        
        # Combine indices: keep all minority + selected majority
        combined_idx = np.concatenate([minority_idx, selected_majority_idx])
        
        # Sort by original position to preserve temporal order
        combined_idx = np.sort(combined_idx)
        
        # Extract sampled data
        if isinstance(X, pd.DataFrame):
            X_sampled = X.iloc[combined_idx].reset_index(drop=True)
        else:
            X_sampled = X_array[combined_idx]
        
        y_sampled = y_array[combined_idx]
        
        return X_sampled, y_sampled
    
    def fit_transform(self, X, y):
        """Fit and transform in one step."""
        return self.transform(X, y)


class StratifiedTimeWindowUndersampler:
    """
    Undersamples within time windows to maintain temporal balance.
    More sophisticated for truly streaming scenarios.
    """
    
    def __init__(self, window_size=5000, sampling_ratio=0.5, minority_label=1, seed=42):
        """
        Args:
            window_size: Number of samples per time window
            sampling_ratio: Fraction of majority class to keep per window
            minority_label: Label for minority class (default: 1)
            seed: Random seed
        """
        self.window_size = window_size
        self.sampling_ratio = sampling_ratio
        self.minority_label = minority_label
        self.majority_label = 1 - minority_label
        self.seed = seed
    
    def fit(self, X, y):
        """No fitting required."""
        return self
    
    def transform(self, X, y):
        """
        Undersample within time windows.
        
        Args:
            X: Features (DataFrame or array)
            y: Labels
            
        Returns:
            X_sampled, y_sampled: Undersampled data
        """
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            is_df = True
        else:
            X_array = np.asarray(X)
            is_df = False
        
        y_array = np.asarray(y)
        n_samples = len(X_array)
        
        np.random.seed(self.seed)
        sampled_indices = []
        
        # Process data in windows
        for window_start in range(0, n_samples, self.window_size):
            window_end = min(window_start + self.window_size, n_samples)
            window_indices = np.arange(window_start, window_end)
            window_y = y_array[window_indices]
            
            # Get minority and majority in this window
            minority_in_window = np.where(window_y == self.minority_label)[0]
            majority_in_window = np.where(window_y == self.majority_label)[0]
            
            # Calculate how many majority to keep
            n_majority_keep = int(len(majority_in_window) * self.sampling_ratio)
            n_majority_keep = max(1, n_majority_keep)
            
            # Sample majority in this window
            if len(majority_in_window) > 0:
                selected = np.random.choice(
                    majority_in_window,
                    size=min(n_majority_keep, len(majority_in_window)),
                    replace=False
                )
                selected_indices = np.concatenate([minority_in_window, selected])
            else:
                selected_indices = minority_in_window
            
            # Map back to original indices
            sampled_indices.extend(window_indices[selected_indices])
        
        # Sort to preserve temporal order
        sampled_indices = np.sort(np.array(sampled_indices))
        
        # Extract sampled data
        if is_df:
            X_sampled = X.iloc[sampled_indices].reset_index(drop=True)
        else:
            X_sampled = X_array[sampled_indices]
        
        y_sampled = y_array[sampled_indices]
        
        return X_sampled, y_sampled
    
    def fit_transform(self, X, y):
        """Fit and transform in one step."""
        return self.transform(X, y)


class RecencyBiasedUndersampler:
    """
    Undersamples with bias towards recent samples.
    Better for concept drift scenarios where recent data is more important.
    """
    
    def __init__(self, sampling_ratio=0.5, minority_label=1, recency_weight=0.7, seed=42):
        """
        Args:
            sampling_ratio: Overall fraction of majority class to keep
            minority_label: Label for minority class (default: 1)
            recency_weight: How much to favor recent samples (0-1, higher = more recent)
            seed: Random seed
        """
        self.sampling_ratio = sampling_ratio
        self.minority_label = minority_label
        self.majority_label = 1 - minority_label
        self.recency_weight = recency_weight
        self.seed = seed
    
    def fit(self, X, y):
        """No fitting required."""
        return self
    
    def transform(self, X, y):
        """
        Undersample with exponential bias towards recent samples.
        
        Args:
            X: Features (DataFrame or array)
            y: Labels
            
        Returns:
            X_sampled, y_sampled: Undersampled data
        """
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            is_df = True
        else:
            X_array = np.asarray(X)
            is_df = False
        
        y_array = np.asarray(y)
        n_samples = len(X_array)
        
        # Identify majority and minority
        minority_idx = np.where(y_array == self.minority_label)[0]
        majority_idx = np.where(y_array == self.majority_label)[0]
        
        # Calculate sampling size
        n_majority_keep = int(len(majority_idx) * self.sampling_ratio)
        n_majority_keep = max(1, n_majority_keep)
        
        # Create exponential weights favoring recent samples
        weights = np.exp(self.recency_weight * np.arange(len(majority_idx)) / len(majority_idx))
        weights /= weights.sum()
        
        # Sample with probability weights
        np.random.seed(self.seed)
        selected_idx_pos = np.random.choice(
            len(majority_idx),
            size=min(n_majority_keep, len(majority_idx)),
            replace=False,
            p=weights
        )
        selected_majority_idx = majority_idx[selected_idx_pos]
        
        # Combine and sort
        combined_idx = np.concatenate([minority_idx, selected_majority_idx])
        combined_idx = np.sort(combined_idx)
        
        # Extract sampled data
        if is_df:
            X_sampled = X.iloc[combined_idx].reset_index(drop=True)
        else:
            X_sampled = X_array[combined_idx]
        
        y_sampled = y_array[combined_idx]
        
        return X_sampled, y_sampled
    
    def fit_transform(self, X, y):
        """Fit and transform in one step."""
        return self.transform(X, y)
