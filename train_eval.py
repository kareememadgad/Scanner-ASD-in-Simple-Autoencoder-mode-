import os
import subprocess
import shutil
import sys
import time

# Only train Scanner machine
machine = "DCASE2024T2Scanner"

print("\n" + "="*50)
print(f"Training {machine}")
print("="*50)

# Check if data directory exists
data_dir = os.path.join("data", "dcase2024t2", "eval_data", "raw", machine.replace("DCASE2024T2", ""))
if not os.path.exists(data_dir):
    print(f"Data directory not found: {data_dir}")
    sys.exit(1)

print(f"Data directory: {data_dir}")
print(f"Contents: {os.listdir(data_dir)}")

# Path to Python executable in virtual environment
venv_python = os.path.join("ml_env", "Scripts", "python.exe")

# Run training command with only supported parameters
cmd = [
    venv_python,
    "train.py",
    f"--dataset_dir=data/dcase2024t2/eval_data",
    f"--machine_type=Scanner",
    "--input_dim=640",
    "--block_size=640",
    "--batch_size=128",
    "--epochs=100",
    "--save_interval=10"
]

print(f"Running training command: {' '.join(cmd)}")

try:
    # Run training with output in real-time
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    
    # Print output in real-time
    for line in process.stdout:
        print(line, end='')
        
    # Wait for the process to complete
    process.wait()
    
    if process.returncode != 0:
        print("Training failed!")
        sys.exit(1)
    
    print("\nTraining completed successfully!")
    
    # Now run testing
    print("\n" + "="*50)
    print("Testing Scanner model")
    print("="*50)
    
    test_cmd = [
        venv_python,
        "test_eval_scanner.py"
    ]
    
    print(f"Running test command: {' '.join(test_cmd)}")
    
    test_process = subprocess.Popen(test_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    
    # Print test output in real-time
    for line in test_process.stdout:
        print(line, end='')
        
    # Wait for the test process to complete
    test_process.wait()
    
    if test_process.returncode != 0:
        print("Testing failed!")
        sys.exit(1)
    
    print("\nTesting completed successfully!")
    
except Exception as e:
    print(f"Error: {str(e)}")
    sys.exit(1) 