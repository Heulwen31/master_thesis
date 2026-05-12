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
# NOTE: Paths below assume standard Kaggle input naming. 
# Adjust if your custom dataset has a different name.

echo "📂 Copying datasets from Kaggle input..."

# Credit Card Fraud
if [ -d "/kaggle/input/creditcardfraud" ]; then
    cp /kaggle/input/creditcardfraud/creditcard.csv data/raw/creditcard/
    echo "✅ Credit Card dataset copied."
fi

# IEEE-CIS Fraud
if [ -d "/kaggle/input/ieee-fraud-detection" ]; then
    cp /kaggle/input/ieee-fraud-detection/train_transaction.csv data/raw/ieee_cis_fraud/
    cp /kaggle/input/ieee-fraud-detection/train_identity.csv data/raw/ieee_cis_fraud/
    echo "✅ IEEE-CIS dataset copied."
fi

# Ecommerce Fraud (Assuming custom name or specific dataset)
if [ -d "/kaggle/input/fraud-ecommerce" ]; then
    cp /kaggle/input/fraud-ecommerce/Fraud_Data.csv data/raw/fraud_ecommerce/
    cp /kaggle/input/fraud-ecommerce/IpAddress_to_Country.csv data/raw/fraud_ecommerce/
    echo "✅ Ecommerce Fraud dataset copied."
fi

# 3. Install dependencies
echo "📦 Installing requirements..."
pip install -r requirements.txt

echo "✨ Setup complete! You can now run the preprocessing or training scripts."
echo "Example: python main.py --mode train --dataset creditcard"
