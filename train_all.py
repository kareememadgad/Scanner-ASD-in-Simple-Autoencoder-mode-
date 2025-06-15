import os
import subprocess
import sys

# List of all machines in the development dataset
machines = [
    "DCASE2024T2bearing",
    "DCASE2024T2fan",
    "DCASE2024T2gearbox",
    "DCASE2024T2slider",
    "DCASE2024T2ToyCar",
    "DCASE2024T2ToyTrain",
    "DCASE2024T2valve"
]

# Path to Python in virtual environment
venv_python = os.path.join(os.getcwd(), "ml_env", "Scripts", "python.exe")

# Run training for each machine
for machine in machines:
    print(f"\n{'='*50}")
    print(f"Training on {machine}")
    print(f"{'='*50}\n")
    
    # Create the tag
    tag = f"id(0_)"
    
    # Run train.py directly with virtual environment Python
    cmd = [
        venv_python,  # Use Python from virtual environment
        "train.py",
        f"--dataset={machine}",
        "-d",
        f"-tag={tag}",
        "--use_ids", "0",
        "--train_only"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    # Execute the command
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    
    # Print output in real-time
    for line in process.stdout:
        print(line, end='')
    
    # Wait for the process to complete
    process.wait()
    
    # Check if the process was successful
    if process.returncode != 0:
        print(f"Error training on {machine}. Exiting.")
        sys.exit(1)
    
    print(f"\nCompleted training on {machine}\n")

print("\nTraining completed for all machines!") 