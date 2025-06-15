#!/bin/bash

echo "Checking Python environment..."

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_PATH="${SCRIPT_DIR}/ml_env/Scripts/python.exe"

# Check if Python exists
if [ ! -f "$PYTHON_PATH" ]; then
    echo "Error: Python executable not found at $PYTHON_PATH"
    echo "Creating new virtual environment..."
    
    # Create virtual environment
    python -m venv ml_env
    
    # Install requirements
    "$PYTHON_PATH" -m pip install --upgrade pip
    "$PYTHON_PATH" -m pip install -r requirements.txt
fi

# Test Python execution
echo "Testing Python execution..."
"$PYTHON_PATH" -c "import torch; print('PyTorch version:', torch.__version__)"
"$PYTHON_PATH" -c "import numpy; print('NumPy version:', numpy.__version__)"
"$PYTHON_PATH" -c "import pandas; print('Pandas version:', pandas.__version__)"

# Check if model files exist
echo "Checking model files..."
if [ ! -d "models/saved_model/baseline" ]; then
    echo "Warning: Model directory not found"
    mkdir -p models/saved_model/baseline
fi

# Check if data directory exists
echo "Checking data directory..."
if [ ! -d "data/dcase2024t2/eval_data/raw" ]; then
    echo "Warning: Data directory not found"
    mkdir -p data/dcase2024t2/eval_data/raw
fi

echo "Environment check completed" 