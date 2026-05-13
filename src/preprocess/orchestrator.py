import sys
import os
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.preprocess.datasets import CreditCardPreprocessor, IEEECISPreprocessor, EcommerceFraudPreprocessor

def main():
    # Disable bytecode to keep it clean as requested by user
    sys.dont_write_bytecode = True
    
    parser = argparse.ArgumentParser(description="Run data preprocessing for datasets.")
    parser.add_argument("--dataset", type=str, default="all", choices=["creditcard", "ieee_cis", "fraud_ecommerce", "all"],
                        help="The dataset to process (creditcard, ieee_cis, fraud_ecommerce, or all)")
    
    args = parser.parse_args()
    
    print(f"Starting preprocessing phase for: {args.dataset}")
    
    # Process Credit Card
    if args.dataset in ["creditcard", "all"]:
        try:
            cc_preprocessor = CreditCardPreprocessor()
            cc_preprocessor.run()
        except Exception as e:
            print(f"Error processing Credit Card: {e}")

    # Process IEEE-CIS
    if args.dataset in ["ieee_cis", "all"]:
        try:
            ieee_preprocessor = IEEECISPreprocessor()
            ieee_preprocessor.run()
        except Exception as e:
            print(f"Error processing IEEE-CIS: {e}")

    # Process Fraud Ecommerce
    if args.dataset in ["fraud_ecommerce", "all"]:
        try:
            ecommerce_preprocessor = EcommerceFraudPreprocessor()
            ecommerce_preprocessor.run()
        except Exception as e:
            print(f"Error processing Fraud Ecommerce: {e}")

    print("\nPreprocessing complete.")

if __name__ == "__main__":
    main()
