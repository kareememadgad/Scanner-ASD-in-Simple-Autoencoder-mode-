import os
import subprocess
import sys
import shutil

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

# Directory for Scanner test results
scanner_results_dir = "scanner_test_results"
os.makedirs(scanner_results_dir, exist_ok=True)

# Scanner model paths
scanner_model = "models/saved_model/baseline/DCASE2023T2-AE_DCASE2024T2Scanner_Eval_seed13711.pth"
scanner_score_distr = "models/saved_model/baseline/score_distr_DCASE2023T2-AE_DCASE2024T2Scanner_Eval_seed13711_mse.pickle"

# Run testing for each machine using Scanner model
for machine in machines:
    print(f"\n{'='*50}")
    print(f"Testing Scanner model on {machine}")
    print(f"{'='*50}\n")
    
    # Create target model directory
    target_model_dir = f"models/saved_model/baseline"
    os.makedirs(target_model_dir, exist_ok=True)
    
    # Copy Scanner model and score distribution to target location
    target_model_path = os.path.join(target_model_dir, f"DCASE2023T2-AE_{machine}_seed13711.pth")
    target_score_distr_path = os.path.join(target_model_dir, f"score_distr_DCASE2023T2-AE_{machine}_seed13711_mse.pickle")
    
    shutil.copy2(scanner_model, target_model_path)
    shutil.copy2(scanner_score_distr, target_score_distr_path)
    
    # Run train.py in test mode for this machine using Scanner model
    cmd = [
        venv_python,  # Use Python from virtual environment
        "train.py",
        f"--dataset={machine}",
        "-d",
        "--use_ids", "0",
        "--test_only",
        "--result_directory", scanner_results_dir
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    # Execute the command
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    
    # Print output in real-time
    for line in process.stdout:
        print(line, end='')
    
    # Wait for the process to complete
    process.wait()
    
    # Clean up - remove copied files
    if os.path.exists(target_model_path):
        os.remove(target_model_path)
    if os.path.exists(target_score_distr_path):
        os.remove(target_score_distr_path)
    
    # Check if the process was successful
    if process.returncode != 0:
        print(f"Error testing Scanner model on {machine}. Exiting.")
        sys.exit(1)
    
    print(f"\nCompleted testing Scanner model on {machine}\n")

print("\nTesting completed for Scanner model against all machines!") 