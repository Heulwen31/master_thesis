#!/bin/bash

# Default values
DATASET="all"

# Help message
function show_help {
    echo "Usage: ./scripts/run_preprocess.sh [options]"
    echo ""
    echo "Options:"
    echo "  -d, --dataset <name>    Specify dataset to process (creditcard, ieee_cis, all) [default: all]"
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

# Run the preprocessing script
echo "Running data preprocessing for dataset: $DATASET"
export PYTHONPATH=$PYTHONPATH:.
python3 scripts/run_preprocess.py --dataset "$DATASET"
