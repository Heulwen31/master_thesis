# Master Thesis - Fraud Detection Pipeline

This project implements a robust fraud detection pipeline for processing and modeling financial transaction data.

## Project Structure

- `configs/`: YAML configuration files for data, models, and training.
- `data/`:
  - `raw/`: Original datasets (Credit Card, IEEE-CIS).
  - `processed/`: Standardized Parquet files (ignored by Git).
- `src/`: Core source code.
  - `data/`: Data loading and standardization utilities.
  - `preprocess/`: Dataset-specific preprocessing logic.
  - `utils/`: Common utilities (e.g., config loader).
- `scripts/`: Shell and Python scripts for automation.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Data**:
   Place your raw datasets in the `data/raw/` directory according to the paths defined in `configs/data.yml`.

## Usage

### 1. Data Preprocessing

Standardize column names, convert timestamps to actual datetime objects, and save to Parquet format.

```bash
# Process all datasets
./scripts/run_preprocess.sh

# Process a specific dataset
./scripts/run_preprocess.sh --dataset creditcard
./scripts/run_preprocess.sh --dataset ieee_cis
```

### 2. Configuration

You can unify column names and adjust paths in `configs/data.yml`. The current standardized names are:
- `time`: Standardized event time.
- `target`: Standardized fraud label.
- `id`: Standardized transaction ID.

## Notes

- Python bytecode (`__pycache__`) is disabled by default in scripts to maintain a clean workspace.
- All processed data is stored in Parquet format for optimized performance.
