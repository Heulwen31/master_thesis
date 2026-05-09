import pandas as pd
import os
from src.utils.config import get_data_config

class DataLoader:
    def __init__(self, dataset_name):
        self.config = get_data_config()
        if dataset_name not in self.config['datasets']:
            raise ValueError(f"Dataset {dataset_name} not found in configuration.")
        
        self.dataset_name = dataset_name
        self.dataset_cfg = self.config['datasets'][dataset_name]
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def load_raw(self):
        """Loads the raw dataset and renames columns according to mapping."""
        path = os.path.join(self.project_root, self.dataset_cfg['raw_path'])
        print(f"Loading {self.dataset_name} from {path}...")
        
        df = pd.read_csv(path)
        
        # Apply column mapping
        mapping = self.dataset_cfg.get('column_mapping', {})
        df = df.rename(columns=mapping)
        
        # If dataset is ieee_cis, we might want to join with identity
        if self.dataset_name == 'ieee_cis' and 'identity_path' in self.dataset_cfg:
            ident_path = os.path.join(self.project_root, self.dataset_cfg['identity_path'])
            if os.path.exists(ident_path):
                print(f"Joining with identity data from {ident_path}...")
                df_ident = pd.read_csv(ident_path)
                # Standardize identity ID if mapped
                df_ident = df_ident.rename(columns=mapping)
                df = df.merge(df_ident, on='id', how='left')
        
        # Ensure time is numeric (Kaggle datasets usually have it as seconds/offset)
        if 'time' in df.columns:
            df['time'] = pd.to_numeric(df['time'], errors='coerce')
            
        print(f"Loaded {self.dataset_name} with shape {df.shape}")
        return df

def get_loader(dataset_name):
    return DataLoader(dataset_name)
