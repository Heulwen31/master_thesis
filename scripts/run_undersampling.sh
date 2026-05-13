#!/bin/bash

# ==============================================================================
# Undersampling Script for Fraud Detection Datasets
# Usage: ./scripts/run_undersampling.sh [OPTIONS]
# ==============================================================================

# Default values
DATASET="creditcard"
METHOD="time_aware"
RATIO="0.5"
KEEP_BACKUP=""

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift ;;
        --method) METHOD="$2"; shift ;;
        --ratio) RATIO="$2"; shift ;;
        --keep-backup) KEEP_BACKUP="--keep-backup" ;;
        -h|--help) 
            echo "Usage: ./scripts/run_undersampling.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dataset DATASET    Dataset name (default: creditcard)"
            echo "                       Choices: creditcard, ieee_cis, fraud_ecommerce"
            echo "  --method METHOD      Undersampling method (default: time_aware)"
            echo "                       Choices: time_aware, stratified_window, recency_biased"
            echo "  --ratio RATIO        Sampling ratio for majority class (default: 0.5)"
            echo "                       Value between 0 and 1"
            echo "  --keep-backup        Create backup of original file before overwriting"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./scripts/run_undersampling.sh --dataset creditcard"
            echo "  ./scripts/run_undersampling.sh --dataset ieee_cis --method stratified_window --ratio 0.4"
            echo "  ./scripts/run_undersampling.sh --dataset fraud_ecommerce --keep-backup"
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
echo "🔄 Starting Undersampling Process"
echo "================================================================="
echo "Dataset: $DATASET"
echo "Method: $METHOD"
echo "Ratio: $RATIO"
echo "================================================================="

# Set PYTHONPATH to include the project root
export PYTHONPATH="$PROJECT_ROOT"

# Run the undersampling script
python3 scripts/apply_undersampling.py \
    --dataset "$DATASET" \
    --method "$METHOD" \
    --ratio "$RATIO" \
    $KEEP_BACKUP

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================="
    echo "✅ Undersampling completed successfully!"
    echo "================================================================="
    echo ""
    echo "ℹ️  Original file has been overwritten with undersampled data"
    if [ -n "$KEEP_BACKUP" ]; then
        echo "📁 Backup saved as: data/processed/${DATASET}_backup.parquet"
    fi
    echo ""
    echo "💡 Next: Run training with the undersampled dataset"
    echo "   ./scripts/run_main.sh --dataset $DATASET --model lightgbm --method periodic"
    echo "================================================================="
else
    echo ""
    echo "================================================================="
    echo "❌ Undersampling failed! Please check the error logs above."
    echo "================================================================="
    exit 1
fi
