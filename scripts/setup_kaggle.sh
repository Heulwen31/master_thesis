#!/bin/bash

# Environment Setup Script for Master Thesis
# Supports both Kaggle and Colab input paths.

set -e

function usage() {
    echo "Usage: $0 [INPUT_DIR]"
    echo "If INPUT_DIR is not provided, the script will try the default Kaggle input path." 
    echo "Example for Colab: ./scripts/setup_kaggle.sh /content/drive/MyDrive/fraud-detection"
    exit 1
}

INPUT_DIR="$1"
if [ -z "$INPUT_DIR" ]; then
    INPUT_DIR="/kaggle/input/datasets/trandongnam/fraud-detection"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_ROOT="$PROJECT_ROOT/data"
RAW_DIR="$DATA_ROOT/raw"
PROCESSED_DIR="$DATA_ROOT/processed"
OUTPUT_MODELS_DIR="$PROJECT_ROOT/outputs/models"
OUTPUT_RESULTS_DIR="$PROJECT_ROOT/outputs/results"

echo "🚀 Starting environment setup..."
echo "📁 Creating data folders..."
mkdir -p "$RAW_DIR/creditcard"
mkdir -p "$RAW_DIR/ieee_cis_fraud"
mkdir -p "$RAW_DIR/fraud_ecommerce"
mkdir -p "$PROCESSED_DIR"
mkdir -p "$OUTPUT_MODELS_DIR"
mkdir -p "$OUTPUT_RESULTS_DIR"

echo "📂 Copying datasets from $INPUT_DIR..."

# Credit Card Fraud
if [ -f "$INPUT_DIR/creditcard.csv" ]; then
    cp "$INPUT_DIR/creditcard.csv" "$RAW_DIR/creditcard/"
    echo "✅ Credit Card dataset copied."
else
    echo "⚠️  Credit Card dataset not found at $INPUT_DIR/creditcard.csv"
fi

# IEEE-CIS Fraud
if [ -f "$INPUT_DIR/train_transaction.csv" ]; then
    cp "$INPUT_DIR/train_transaction.csv" "$RAW_DIR/ieee_cis_fraud/"
    cp "$INPUT_DIR/train_identity.csv" "$RAW_DIR/ieee_cis_fraud/"
    echo "✅ IEEE-CIS dataset copied."
else
    echo "⚠️  IEEE-CIS dataset not found at $INPUT_DIR/train_transaction.csv"
fi

# Ecommerce Fraud
if [ -f "$INPUT_DIR/Fraud_Data.csv" ]; then
    cp "$INPUT_DIR/Fraud_Data.csv" "$RAW_DIR/fraud_ecommerce/"
    cp "$INPUT_DIR/IpAddress_to_Country.csv" "$RAW_DIR/fraud_ecommerce/"
    echo "✅ Ecommerce Fraud dataset copied."
else
    echo "⚠️  Ecommerce Fraud dataset not found at $INPUT_DIR/Fraud_Data.csv"
fi

echo "📦 Installing requirements..."
pip install -r "$PROJECT_ROOT/requirements.txt"

echo "✨ Setup complete!"
echo "Run preprocessing with: python scripts/run_preprocess.py --dataset <dataset>"
echo "Run training with: ./scripts/run_main.sh --dataset <dataset> --model <model> --method <method>"
