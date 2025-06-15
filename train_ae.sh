#!/bin/bash

echo "Starting training for dataset: $1"

dataset=$1
dev_eval=$2

echo "Dataset: ${dataset}"
echo "Dev/Eval: ${dev_eval}"

# Extract machine type from dataset name (e.g., DCASE2024T2bearing -> bearing)
machine_type=$(echo $dataset | sed 's/DCASE2024T2//')

# Check if Python environment exists
if [ ! -f "ml_env/Scripts/python.exe" ]; then
    echo "Error: Python environment not found at ml_env/Scripts/python.exe"
    echo "Please create the virtual environment first"
    exit 1
fi

# Run training
echo "Starting training for machine type: ${machine_type}"
./ml_env/Scripts/python.exe train.py \
    --dataset_dir=data/dcase2024t2/dev_data/raw \
    --machine_type=${machine_type} \
    --input_dim=128 \
    --block_size=313 \
    --batch_size=32 \
    --epochs=100 \
    --save_interval=10

if [ $? -eq 0 ]; then
    echo "Training completed successfully"
else
    echo "Training failed"
    exit 1
fi
