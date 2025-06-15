#!/bin/bash

# Convert Windows line endings to Unix line endings
if [ -f "ml_env/Scripts/activate" ]; then
    dos2unix ml_env/Scripts/activate
fi

# Make sure Python path is correct
PYTHON_PATH="ml_env/Scripts/python.exe"
if [ ! -f "$PYTHON_PATH" ]; then
    echo "Error: Python executable not found at $PYTHON_PATH"
    echo "Please create the virtual environment first"
    exit 1
fi

echo "Environment setup completed" 