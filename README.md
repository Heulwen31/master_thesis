# Master Thesis - Fraud Detection Pipeline

This project implements a robust fraud detection pipeline for processing and modeling financial transaction data, featuring adaptive concept drift detection.

## Project Structure

- `configs/`: YAML configuration files for data, models, and training parameters.
- `data/`:
  - `raw/`: Original datasets (Credit Card, IEEE-CIS, Ecommerce Fraud).
  - `processed/`: Standardized Parquet files.
- `src/`: Core source code.
  - `data/`: Data loading utilities.
  - `preprocess/`: Dataset-specific preprocessing logic (handling joins, time conversion).
  - `trainner/`: Implementation of ADDM (Adaptive Concept Drift) and Periodic baselines.
  - `evaluation/`: Metrics collection and report generation.
  - `models/`: Factory for XGBoost, LightGBM, and CatBoost.
- `scripts/`: Automation scripts for preprocessing and running the pipeline.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Data**:
   Place your raw datasets in `data/raw/`. The pipeline supports:
   - **Credit Card Fraud** (Kaggle)
   - **IEEE-CIS Fraud Detection**
   - **Ecommerce Fraud** (with IP-to-Country mapping)

## Usage

### 1. Data Preprocessing
Standardize data, handle time formats, and join auxiliary tables (e.g., IP mapping).

```bash
# Process a specific dataset
python3 scripts/run_preprocess.py --dataset creditcard
python3 scripts/run_preprocess.py --dataset ieee_cis
python3 scripts/run_preprocess.py --dataset fraud_ecommerce

# Or use the shell script for all
./scripts/run_preprocess.sh
```

### 2. Running the Evaluation Pipeline
Run the online "test-then-train" evaluation with concept drift detection.

```bash
# Basic run (IEEE-CIS with XGBoost using ADDM)
./scripts/run_main.sh

# Advanced run (Select dataset, model, and method)
./scripts/run_main.sh --dataset fraud_ecommerce --model lightgbm --method addm
./scripts/run_main.sh --dataset creditcard --model catboost --method periodic
```

## Methods

- **ADDM (Adaptive sliding window based Drift Detection Method)**: Uses Information Entropy on an adaptive window (Hoeffding Bound) to detect concept changes in real-time.
- **Periodic**: A baseline that retrains the model every N transactions using all historical data.

## Configuration

- `configs/data.yml`: Column mapping and raw data paths.
- `configs/model.yml`: Hyperparameters for XGBoost, LightGBM, and CatBoost.
- `configs/trainner.yml`: ADDM thresholds, periodic intervals, and run modes (subsample vs full).

## Final Report
Every run generates a **Final Evaluation Report** including:
- Accuracy, Precision, Recall, F1-Score.
- ROC-AUC and PR-AUC.
- Detailed analysis of detected drift points and retraining events.
