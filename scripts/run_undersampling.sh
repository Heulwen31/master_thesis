#!/bin/bash

# Default values
DATASET="ieee_cis"

# Help message
function show_help {
    echo "Usage: ./scripts/run_undersampling.sh [options]"
    echo ""
    echo "Options:"
    echo "  -d, --dataset <name>    Specify dataset to sample (creditcard, ieee_cis, fraud_ecommerce) [default: ieee_cis]"
    echo "  -h, --help              Show this help message"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dataset)
            DATASET="$2"
            shift # past argument
            shift # past value
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Run the undersampling script
echo "Running under-sampling for dataset: $DATASET"
export PYTHONPATH=$PYTHONPATH:.
python3 src/preprocess/undersampling_runner.py --dataset "$DATASET"
