#!/bin/bash

echo "Verifying Python environment setup..."

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
else
    echo "Python found at: $PYTHON_PATH"
    echo "Python version:"
    "$PYTHON_PATH" --version
    
    echo "Installed packages:"
    "$PYTHON_PATH" -m pip list
fi

# Check if model directory exists
if [ ! -d "models/saved_model/baseline" ]; then
    echo "Warning: Model directory not found at models/saved_model/baseline"
    mkdir -p models/saved_model/baseline
fi

# Check if data directory exists
if [ ! -d "data/dcase2024t2/eval_data/raw" ]; then
    echo "Warning: Data directory not found at data/dcase2024t2/eval_data/raw"
    mkdir -p data/dcase2024t2/eval_data/raw
fi

echo "Environment verification completed" 