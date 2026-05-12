import subprocess
import logging

logger = logging.getLogger(__name__)

def is_gpu_available():
    """
    Checks if an NVIDIA GPU is available in the environment.
    Uses nvidia-smi as a reliable check for system-wide GPU presence.
    """
    try:
        # Run nvidia-smi to check for GPU presence
        subprocess.check_output(['nvidia-smi'], stderr=subprocess.STDOUT)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Either nvidia-smi failed or is not installed
        return False

def get_gpu_params(model_type):
    """
    Returns the appropriate GPU acceleration parameters for different GBDT libraries.
    """
    if not is_gpu_available():
        return {}

    if model_type == "xgboost":
        # For modern XGBoost (2.0+), device='cuda' is the preferred way.
        # For older versions, tree_method='gpu_hist' is used.
        return {
            "device": "cuda",
            "tree_method": "hist" # Standard histogram algorithm but on GPU
        }
    
    elif model_type == "lightgbm":
        return {
            "device": "gpu",
            "gpu_platform_id": 0,
            "gpu_device_id": 0
        }
    
    elif model_type == "catboost":
        return {
            "task_type": "GPU",
            "devices": "0"
        }
    
    return {}
