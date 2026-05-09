import math
import numpy as np

class ADDM:
    """
    Adaptive sliding window based Drift Detection Method (ADDM).
    Monitors a stream of prediction results (1 for correct, 0 for incorrect)
    to detect concept drift based on entropy and an adaptive sliding window.
    """
    def __init__(self, hoeffding_delta=0.01, min_window_size=30, entropy_threshold=1.0):
        self.delta = hoeffding_delta
        self.min_window_size = min_window_size
        self.entropy_threshold = entropy_threshold
        self.window = []
        
    def add_result(self, result: int) -> bool:
        """
        Adds a new prediction result and checks for drift.
        Returns True if drift is detected, False otherwise.
        """
        self.window.append(result)
        # Check drift BEFORE adapting window, because adapting might shrink the window 
        # so fast that we miss the entropy spike during the concept transition.
        is_drift = self._check_drift()
        if not is_drift:
            self._adapt_window()
        return is_drift
        
    def _adapt_window(self):
        """
        Dynamically adjusts the window size using the Hoeffding bound.
        Compares the mean of the older half to the newer half.
        If the difference exceeds the bound, it indicates the window contains
        multiple concepts, so we shrink it from the oldest elements.
        """
        if len(self.window) > self.min_window_size:
            n = len(self.window)
            half = n // 2
            
            # Split window into two halves
            w0 = self.window[:half]
            w1 = self.window[half:]
            
            mu0 = np.mean(w0)
            mu1 = np.mean(w1)
            
            # Hoeffding bound for the difference of two means
            epsilon = math.sqrt((1 / (2 * half)) * math.log(1 / self.delta))
            
            if abs(mu0 - mu1) > epsilon:
                # Concepts are different, shrink window by removing the oldest element
                self.window.pop(0)
                
    def _check_drift(self) -> bool:
        """
        Checks if the entropy of the current window exceeds the threshold.
        """
        if len(self.window) < self.min_window_size:
            return False
            
        p1 = np.mean(self.window)
        p0 = 1.0 - p1
        
        # Calculate entropy: -p1*log2(p1) - p0*log2(p0)
        # Add a small epsilon to avoid log(0)
        eps = 1e-15
        p1 = max(eps, min(1 - eps, p1))
        p0 = max(eps, min(1 - eps, p0))
        
        entropy = -p1 * math.log2(p1) - p0 * math.log2(p0)
        
        # Debug entropy

            
        if entropy >= self.entropy_threshold:
            return True
        return False
        
    def seek_retraining_timestamp(self, current_index: int) -> int:
        """
        MB-GT Algorithm Placeholder.
        Determines the exact timestamp to start retraining.
        Currently returns the start of the adaptive window as a baseline.
        """
        # The adaptive window contains the elements of the current concept.
        # The start of this window relative to the global stream is a good
        # approximation for the retraining point.
        retraining_point = current_index - len(self.window) + 1
        return max(0, retraining_point)

    def reset(self):
        """Resets the detector's state after a drift."""
        self.window = []
