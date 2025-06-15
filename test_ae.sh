#!/bin/bash

# Get arguments
dataset=$1
dev_eval=$2
id_1=$3
id_2=$4
id_3=$5
id_4=$6
id="$id_1 $id_2 $id_3 $id_4"

echo "Starting testing for dataset: ${dataset}"
echo "Dev/Eval: ${dev_eval}"
echo "ID: ${id}"

# Extract machine type from dataset name (e.g., DCASE2024T2Scanner -> Scanner)
machine_type=$(echo $dataset | sed 's/DCASE2024T2//')

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

# Create results directory if it doesn't exist
mkdir -p results/${machine_type}

# Run testing
echo "Starting testing for machine type: ${machine_type}"
"$PYTHON_PATH" train.py \
    --dataset_dir=data/dcase2024t2/eval_data/raw \
    --machine_type=${machine_type} \
    --input_dim=128 \
    --block_size=313 \
    --batch_size=32 \
    --epochs=1 \
    --save_interval=1

if [ $? -eq 0 ]; then
    echo "Testing completed successfully"
    echo "Results saved in results/${machine_type}/"
    echo "1. performance_metrics.xlsx - Contains all performance metrics"
    echo "2. anomaly_scores.xlsx - Contains raw anomaly scores and labels"
else
    echo "Testing failed"
    exit 1
fi 