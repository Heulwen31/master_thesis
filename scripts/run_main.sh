#!/bin/bash

# ==============================================================================
# Pipeline Runner Script
# Usage: ./scripts/run_main.sh [--dataset DATASET] [--model MODEL]
# ==============================================================================

# Default values
DATASET="ieee_cis"
MODEL="xgboost"
METHOD="sliding"
MAX_SAMPLES=""

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift ;;
        --model) MODEL="$2"; shift ;;
        --method) METHOD="$2"; shift ;;
        --max-samples) MAX_SAMPLES="$2"; shift ;;
        -h|--help) 
            echo "Usage: ./scripts/run_main.sh [--dataset DATASET] [--model MODEL] [--method METHOD] [--max-samples MAX_SAMPLES]"
            echo "  --dataset     : Name of dataset to process (default: ieee_cis). Choices: ieee_cis, creditcard, fraud_ecommerce."
            echo "  --model       : Name of the model to evaluate (default: xgboost). Choices: xgboost, lightgbm, catboost."
            echo "  --method      : Evaluation method (default: sliding). Choices: sliding, periodic, incremental, hybrid, all."
            echo "  --max-samples : Cap rows after load"
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
echo "🚀 Starting Sliding Window Training & Evaluation Pipeline"
echo "================================================================="

# Set PYTHONPATH to include the project root
export PYTHONPATH="$PROJECT_ROOT"

# Run the main python script
if [ -n "$MAX_SAMPLES" ]; then
    python3 main.py --dataset "$DATASET" --model "$MODEL" --method "$METHOD" --max-samples "$MAX_SAMPLES"
else
    python3 main.py --dataset "$DATASET" --model "$MODEL" --method "$METHOD"
fi

# Check exit status
if [ $? -eq 0 ]; then
    echo "✅ Pipeline completed successfully!"
else
    echo "❌ Pipeline failed! Please check the error logs above."
    exit 1
fi
