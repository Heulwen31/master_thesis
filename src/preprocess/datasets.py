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
