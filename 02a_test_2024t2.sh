#!/bin/bash

# Get arguments
dev_eval=$1

echo "Starting testing in eval mode for Scanner machine"

# Set absolute path to Python executable
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_PATH="${SCRIPT_DIR}/ml_env/Scripts/python.exe"

# Check if Python exists
if [ ! -f "$PYTHON_PATH" ]; then
    echo "Error: Python executable not found at $PYTHON_PATH"
    echo "Please create the virtual environment first"
    exit 1
fi

echo "Using Python at: $PYTHON_PATH"

# Set machine type to Scanner
MACHINE_TYPE="Scanner"

# Create results directory if it doesn't exist
mkdir -p results/${MACHINE_TYPE}

# Run testing
echo "Starting testing for machine type: ${MACHINE_TYPE}"
"$PYTHON_PATH" test.py \
    --dataset_dir="data/dcase2024t2/eval_data/raw" \
    --machine_type=${MACHINE_TYPE} \
    --input_dim=128 \
    --block_size=313 \
    --batch_size=32 \
    --results_dir="results/${MACHINE_TYPE}" \
    --model_dir="models"

if [ $? -eq 0 ]; then
    echo "Testing completed successfully"
    echo "Results saved in results/${MACHINE_TYPE}/"
    echo "1. performance_metrics.xlsx - Contains all performance metrics"
    echo "2. anomaly_scores.xlsx - Contains raw anomaly scores and labels"
else
    echo "Testing failed"
    exit 1
fi