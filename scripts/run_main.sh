#!/bin/bash

# ==============================================================================
# Pipeline Runner Script
# Usage: ./scripts/run_main.sh [--dataset DATASET] [--model MODEL]
# ==============================================================================

# Default values
DATASET="ieee_cis"
MODEL="xgboost"

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift ;;
        --model) MODEL="$2"; shift ;;
        -h|--help) 
            echo "Usage: ./scripts/run_main.sh [--dataset DATASET] [--model MODEL]"
            echo "  --dataset : Name of dataset to process (default: ieee_cis). Choices: ieee_cis, creditcard."
            echo "  --model   : Name of the model to evaluate (default: xgboost). Choices: xgboost, lightgbm, catboost."
            exit 0
            ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Ensure we are in the project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || exit 1

echo "================================================================="
echo "🚀 Starting ADDM Training & Evaluation Pipeline"
echo "================================================================="

# Set PYTHONPATH to include the project root
export PYTHONPATH="$PROJECT_ROOT"

# Run the main python script
python3 main.py --dataset "$DATASET" --model "$MODEL"

# Check exit status
if [ $? -eq 0 ]; then
    echo "✅ Pipeline completed successfully!"
else
    echo "❌ Pipeline failed! Please check the error logs above."
    exit 1
fi
