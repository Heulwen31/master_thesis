import yaml
import os

def load_yaml(file_path):
    """Loads a YAML file from the given path."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def get_data_config():
    """Loads the data configuration."""
    # Assuming the config is in project_root/configs/data.yml
    # We can use an absolute path or relative to this file
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(base_dir, "configs", "data.yml")
    return load_yaml(config_path)
