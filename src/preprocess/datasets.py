import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import TargetEncoder
from sklearn.model_selection import KFold
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

    def _normalize_features(self, df):
        """Standardize numerical features using RobustScaler (best for fraud/imbalanced outliers)."""
        print(f"Normalizing numerical features for {self.dataset_name} using RobustScaler...")
        
        # Select numerical columns except target and time
        exclude = ['target', 'time']
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        num_cols = [col for col in num_cols if col not in exclude]
        
        if not num_cols:
            print("  No numerical columns to normalize.")
            return df
            
        scaler = RobustScaler()
        # Handle NaNs temporarily for scaling if any remain
        df[num_cols] = scaler.fit_transform(df[num_cols].fillna(df[num_cols].median()))
        print(f"  Normalized {len(num_cols)} features.")
        return df

    def run(self):
        """Standard pipeline: load, process time, normalize, save to parquet."""
        df = self.loader.load_raw()
        df = self.process_time(df)
        df = self._normalize_features(df)
        
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

    def _remove_high_null_features(self, df):
        """Remove features with null ratio above threshold."""
        null_threshold = self.loader.dataset_cfg.get('null_threshold', 0.8)
        print(f"Removing features with null ratio > {null_threshold} for {self.dataset_name}...")
        
        null_ratios = df.isnull().mean()
        high_null_cols = null_ratios[null_ratios > null_threshold].index.tolist()
        
        if high_null_cols:
            print(f"  Removing {len(high_null_cols)} features: {high_null_cols}")
            df = df.drop(columns=high_null_cols)
        else:
            print("  No features to remove.")
        
        return df

    def _encode_categorical(self, df):
        """Encode categorical columns based on cardinality with cross-validation for TargetEncoder."""
        print(f"Encoding categorical features for {self.dataset_name}...")
        
        # Identify categorical columns (exclude 'target' if present)
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        cat_cols = [col for col in cat_cols if col != 'target']
        
        if not cat_cols:
            print("No categorical columns to encode.")
            return df
        
        # Separate target for TargetEncoder
        has_target = 'target' in df.columns
        target = df['target'] if has_target else None
        
        for col in cat_cols:
            n_unique = df[col].nunique()
            print(f"  Column '{col}': {n_unique} unique values", end="")
            
            if n_unique < 3:
                # Label Encoding
                print(" -> Label Encoding")
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                
            elif 3 <= n_unique <= 5:
                # One-Hot Encoding
                print(" -> One-Hot Encoding")
                dummies = pd.get_dummies(df[col], prefix=col, dtype='int8')
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                
            else:
                # Target Encoding with Cross-Validation to avoid data leakage
                print(" -> Target Encoding with CV")
                if has_target and target is not None:
                    # Use KFold cross-validation for TargetEncoder
                    kf = KFold(n_splits=5, shuffle=True, random_state=42)
                    encoded_col = np.zeros(len(df))
                    
                    for train_idx, test_idx in kf.split(df):
                        te = TargetEncoder(random_state=42)
                        # Fit on train fold, transform on test fold
                        te.fit(df.iloc[train_idx][[col]], target.iloc[train_idx])
                        encoded_col[test_idx] = te.transform(df.iloc[test_idx][[col]]).ravel()
                    
                    df[col] = encoded_col
                else:
                    # Fallback: simple TargetEncoder if no target available
                    print("    (No target available, using simple TargetEncoder)")
                    te = TargetEncoder(random_state=42)
                    df[col] = te.fit_transform(df[[col]], target if has_target else pd.Series(0, index=df.index)).ravel()
        
        return df

    def run(self):
        """Standard pipeline: load, process time, drop id, remove high null features, encode categorical, normalize, save to parquet."""
        df = self.loader.load_raw()
        df = self.process_time(df)
        
        # Remove TransactionID (which is mapped to 'id')
        if 'id' in df.columns:
            print(f"Removing 'id' column for {self.dataset_name}...")
            df = df.drop(columns=['id'])
        
        # Remove features with high null ratio
        df = self._remove_high_null_features(df)
        
        # Encode categorical features with appropriate strategy
        df = self._encode_categorical(df)

        # Final normalization
        df = self._normalize_features(df)
            
        output_path = os.path.join(self.output_dir, f"{self.dataset_name}.parquet")
        print(f"Saving processed data to {output_path}...")
        df.to_parquet(output_path, engine='pyarrow', index=False)
        print(f"Done processing {self.dataset_name}.")

class EcommerceFraudPreprocessor(BasePreprocessor):
    def __init__(self):
        super().__init__("fraud_ecommerce")

    def _standardize_ip(self, df):
        if 'ip_address' in df.columns:
            print(f"Standardizing IP address values for {self.dataset_name}...")
            df['ip_address'] = pd.to_numeric(df['ip_address'], errors='coerce')
            if df['ip_address'].notna().any():
                min_ip = df['ip_address'].min()
                max_ip = df['ip_address'].max()
                if max_ip > min_ip:
                    df['ip_address'] = (df['ip_address'] - min_ip) / (max_ip - min_ip)
                else:
                    df['ip_address'] = 0.0
        return df

    def _log_transform_amount(self, df, amount_col):
        if amount_col in df.columns:
            print(f"Applying log transform to {amount_col} for {self.dataset_name}...")
            df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce').fillna(0.0)
            df[amount_col] = np.log1p(df[amount_col].clip(lower=0))
        return df

    def _encode_categorical(self, df):
        """Encode categorical columns based on cardinality with cross-validation for TargetEncoder."""
        print(f"Encoding categorical features for {self.dataset_name}...")
        
        # Identify categorical columns (exclude 'target' if present)
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        cat_cols = [col for col in cat_cols if col != 'target']
        
        if not cat_cols:
            print("No categorical columns to encode.")
            return df
        
        # Separate target for TargetEncoder
        has_target = 'target' in df.columns
        target = df['target'] if has_target else None
        
        for col in cat_cols:
            n_unique = df[col].nunique()
            print(f"  Column '{col}': {n_unique} unique values", end="")
            
            if n_unique < 3:
                # Label Encoding
                print(" -> Label Encoding")
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                
            elif 3 <= n_unique <= 5:
                # One-Hot Encoding
                print(" -> One-Hot Encoding")
                dummies = pd.get_dummies(df[col], prefix=col, dtype='int8')
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                
            else:
                # Target Encoding with Cross-Validation to avoid data leakage
                print(" -> Target Encoding with CV")
                if has_target and target is not None:
                    # Use KFold cross-validation for TargetEncoder
                    kf = KFold(n_splits=5, shuffle=True, random_state=42)
                    encoded_col = np.zeros(len(df))
                    
                    for train_idx, test_idx in kf.split(df):
                        te = TargetEncoder(random_state=42)
                        # Fit on train fold, transform on test fold
                        te.fit(df.iloc[train_idx][[col]], target.iloc[train_idx])
                        encoded_col[test_idx] = te.transform(df.iloc[test_idx][[col]]).ravel()
                    
                    df[col] = encoded_col
                else:
                    # Fallback: simple TargetEncoder if no target available
                    print("    (No target available, using simple TargetEncoder)")
                    te = TargetEncoder(random_state=42)
                    df[col] = te.fit_transform(df[[col]], target if has_target else pd.Series(0, index=df.index)).ravel()
        
        return df

    def run(self):
        """Pipeline for Ecommerce Fraud: handles string dates, joins with IP mapping, drops id, normalize, saves to parquet."""
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
        
        # Preprocess IP column and purchase amount
        df = self._standardize_ip(df)
        df = self._log_transform_amount(df, 'purchase_value')
        
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
        
        # Encode categorical features with appropriate strategy
        df = self._encode_categorical(df)

        # Final normalization
        df = self._normalize_features(df)
            
        output_path = os.path.join(self.output_dir, f"{self.dataset_name}.parquet")
        print(f"Saving processed data to {output_path}...")
        df.to_parquet(output_path, engine='pyarrow', index=False)
        print(f"Done processing {self.dataset_name}.")

