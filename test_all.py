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

# Run testing for each machine
for machine in machines:
    print(f"\n{'='*50}")
    print(f"Testing on {machine}")
    print(f"{'='*50}\n")
    
    # Run train.py in test mode for this machine
    cmd = [
        venv_python,  # Use Python from virtual environment
        "train.py",
        f"--dataset={machine}",
        "-d",
        f"-tag=id(0_)",
        "--use_ids", "0",
        "--test_only"
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
        print(f"Error testing on {machine}. Exiting.")
        sys.exit(1)
    
    print(f"\nCompleted testing on {machine}\n")

print("\nTesting completed for all machines!") 