#!/bin/bash

# Kaggle Environment Setup Script for Master Thesis
# Author: Antigravity (AI Assistant)

echo "🚀 Starting Kaggle environment setup..."

# 1. Create directory structure
echo "📁 Creating data folders..."
mkdir -p data/raw/creditcard
mkdir -p data/raw/ieee_cis_fraud
mkdir -p data/raw/fraud_ecommerce
mkdir -p data/processed
mkdir -p outputs/models
mkdir -p outputs/results

# 2. Copy data from Kaggle input
# Custom paths provided by user
KAGLE_INPUT_DIR="/kaggle/input/datasets/trandongnam/fraud-detection"

echo "📂 Copying datasets from $KAGLE_INPUT_DIR..."

# Credit Card Fraud
if [ -f "$KAGLE_INPUT_DIR/creditcard.csv" ]; then
    cp "$KAGLE_INPUT_DIR/creditcard.csv" data/raw/creditcard/
    echo "✅ Credit Card dataset copied."
fi

# IEEE-CIS Fraud
if [ -f "$KAGLE_INPUT_DIR/train_transaction.csv" ]; then
    cp "$KAGLE_INPUT_DIR/train_transaction.csv" data/raw/ieee_cis_fraud/
    cp "$KAGLE_INPUT_DIR/train_identity.csv" data/raw/ieee_cis_fraud/
    echo "✅ IEEE-CIS dataset copied."
fi

# Ecommerce Fraud
if [ -f "$KAGLE_INPUT_DIR/Fraud_Data.csv" ]; then
    cp "$KAGLE_INPUT_DIR/Fraud_Data.csv" data/raw/fraud_ecommerce/
    cp "$KAGLE_INPUT_DIR/IpAddress_to_Country.csv" data/raw/fraud_ecommerce/
    echo "✅ Ecommerce Fraud dataset copied."
fi

# 3. Install dependencies
echo "📦 Installing requirements..."
pip install -r requirements.txt

echo "✨ Setup complete! You can now run the preprocessing or training scripts."
echo "Example: python main.py --mode train --dataset creditcard"
