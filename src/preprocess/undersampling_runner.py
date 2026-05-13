import os
import sys
import pandas as pd
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.config import get_trainner_config

def apply_under_sampling(df, target_ratio=0.9, min_samples=50000):
    """
    Under-samples majority class (0) to occupy target_ratio of data.
    Keeps all minority (1) and respects min_samples.
    """
    minority = df[df['target'] == 1]
    majority = df[df['target'] == 0]
    m_count = len(minority)
    
    if m_count == 0:
        print("⚠️ No minority samples found. Skipping under-sampling.")
        return df
        
    # N = (R * M) / (1 - R)
    n_target = int((target_ratio * m_count) / (1 - target_ratio))
    
    # Respect minimum samples
    if (n_target + m_count) < min_samples:
        n_target = max(0, min_samples - m_count)
        
    n_target = min(n_target, len(majority))
    print(f"📊 Under-sampling: Minority={m_count}, Original Majority={len(majority)}")
    print(f"📊 Target Majority={n_target}, Final Total={n_target+m_count}")
    
    sampled_majority = majority.sample(n=n_target, random_state=42)
    return pd.concat([minority, sampled_majority]).sort_index()

def main():
    parser = argparse.ArgumentParser(description="Run under-sampling as a separate preprocess step.")
    parser.add_argument("--dataset", type=str, required=True, choices=["ieee_cis", "creditcard", "fraud_ecommerce"], help="Dataset to sample")
    args = parser.parse_args()

    # Load config
    config = get_trainner_config()
    sampling_cfg = config.get("pipeline", {}).get("sampling", {})
    
    input_path = os.path.join("data", "processed", f"{args.dataset}.parquet")
    output_path = input_path # Overwrite original file
    
    if not os.path.exists(input_path):
        print(f"❌ Error: Input file not found at {input_path}")
        sys.exit(1)
        
    print(f"📂 Loading data from {input_path}...")
    df = pd.read_parquet(input_path)
    
    df_sampled = apply_under_sampling(
        df,
        target_ratio=sampling_cfg.get("target_majority_ratio", 0.9),
        min_samples=sampling_cfg.get("min_total_samples", 50000)
    )
    
    print(f"💾 Saving sampled data to {output_path}...")
    df_sampled.to_parquet(output_path, engine='pyarrow', index=False)
    print("✅ Under-sampling complete!")

if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()
