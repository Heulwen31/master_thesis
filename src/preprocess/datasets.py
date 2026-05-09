import os
import pandas as pd
from datetime import datetime, timedelta
from src.data.loader import get_loader

class BasePreprocessor:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.loader = get_loader(dataset_name)
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.output_dir = os.path.join(self.project_root, "data", "processed")
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def process_time(self, df, reference_date="2024-01-01"):
        """Converts numerical time (seconds) to actual datetime starting from a reference date."""
        print(f"Converting time to datetime for {self.dataset_name} (unit: seconds)...")
        # Ensure it's treated as seconds ('s') not milliseconds
        df['time'] = pd.to_datetime(df['time'], unit='s', origin=pd.Timestamp(reference_date))
        return df

    def run(self):
        """Standard pipeline: load, process time, save to parquet."""
        df = self.loader.load_raw()
        df = self.process_time(df)
        
        output_path = os.path.join(self.output_dir, f"{self.dataset_name}.parquet")
        print(f"Saving processed data to {output_path}...")
        df.to_parquet(output_path, engine='pyarrow', index=False)
        print(f"Done processing {self.dataset_name}.")

class CreditCardPreprocessor(BasePreprocessor):
    def __init__(self):
        super().__init__("creditcard")

class IEEECISPreprocessor(BasePreprocessor):
    def __init__(self):
        super().__init__("ieee_cis")

    def run(self):
        """Standard pipeline: load, process time, drop id, save to parquet."""
        df = self.loader.load_raw()
        df = self.process_time(df)
        
        # Remove TransactionID (which is mapped to 'id')
        if 'id' in df.columns:
            print(f"Removing 'id' column for {self.dataset_name}...")
            df = df.drop(columns=['id'])
            
        output_path = os.path.join(self.output_dir, f"{self.dataset_name}.parquet")
        print(f"Saving processed data to {output_path}...")
        df.to_parquet(output_path, engine='pyarrow', index=False)
        print(f"Done processing {self.dataset_name}.")

class EcommerceFraudPreprocessor(BasePreprocessor):
    def __init__(self):
        super().__init__("fraud_ecommerce")

    def run(self):
        """Pipeline for Ecommerce Fraud: handles string dates, joins with IP mapping, drops id, saves to parquet."""
        # Load raw data
        path = os.path.join(self.project_root, self.loader.dataset_cfg['raw_path'])
        print(f"Loading {self.dataset_name} from {path}...")
        df = pd.read_csv(path)
        
        # Load Mapping data
        mapping_path = self.loader.dataset_cfg.get('mapping_path')
        if mapping_path:
            mapping_abs_path = os.path.join(self.project_root, mapping_path)
            if os.path.exists(mapping_abs_path):
                print(f"Loading IP to Country mapping from {mapping_abs_path}...")
                ip_df = pd.read_csv(mapping_abs_path)
                
                # Sort for merge_asof
                df = df.sort_values('ip_address')
                ip_df = ip_df.sort_values('lower_bound_ip_address')
                
                # Merge based on ranges
                print("Merging IP addresses with countries (this may take a moment)...")
                df = pd.merge_asof(
                    df, ip_df,
                    left_on='ip_address',
                    right_on='lower_bound_ip_address',
                    direction='backward'
                )
                
                # Validate range: ip must be <= upper_bound
                mask = df['ip_address'] <= df['upper_bound_ip_address']
                df.loc[~mask, 'country'] = "Unknown"
                
                # Cleanup auxiliary columns
                df = df.drop(columns=['lower_bound_ip_address', 'upper_bound_ip_address'])
        
        # Mapping standard columns
        mapping = self.loader.dataset_cfg.get('column_mapping', {})
        df = df.rename(columns=mapping)
        
        # Handle time: it's a string '2015-02-24 22:55:49'
        print(f"Converting string time to datetime for {self.dataset_name}...")
        df['time'] = pd.to_datetime(df['time'])
        
        # Drop ID and irrelevant columns
        id_col = self.loader.dataset_cfg.get('id_column')
        cols_to_drop = []
        if id_col and id_col in df.columns:
            cols_to_drop.append(id_col)
        elif 'id' in df.columns:
            cols_to_drop.append('id')
            
        if 'signup_time' in df.columns:
            cols_to_drop.append('signup_time')
            
        if cols_to_drop:
            print(f"Removing columns: {cols_to_drop}...")
            df = df.drop(columns=cols_to_drop)
            
        output_path = os.path.join(self.output_dir, f"{self.dataset_name}.parquet")
        print(f"Saving processed data to {output_path}...")
        df.to_parquet(output_path, engine='pyarrow', index=False)
        print(f"Done processing {self.dataset_name}.")
