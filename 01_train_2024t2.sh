#!/bin/bash

# Get arguments
dev_eval=$1

echo "Starting training in dev mode"

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

# List of machine types to train
MACHINE_TYPES=("ToyCar" "ToyConveyor" "fan" "gearbox" "pump" "slider" "valve")

# Train for each machine type
for machine_type in "${MACHINE_TYPES[@]}"; do
    echo "Training for machine type: ${machine_type}"
    
    # Create results directory if it doesn't exist
    mkdir -p results/${machine_type}
    
    # Run training
    "$PYTHON_PATH" train.py \
        --dataset_dir="data/dcase2024t2/dev_data/raw" \
        --machine_type=${machine_type} \
        --input_dim=128 \
        --block_size=313 \
        --batch_size=32 \
        --epochs=100 \
        --save_interval=10
        
    if [ $? -eq 0 ]; then
        echo "Training completed successfully for ${machine_type}"
    else
        echo "Training failed for ${machine_type}"
        exit 1
    fi
done

echo "All training completed successfully"
echo "Results saved in results/*/"
