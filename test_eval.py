import os
import subprocess
import sys
import shutil

# Dictionary mapping dataset names to machine type directories
machine_map = {
    "DCASE2024T23DPrinter": "3DPrinter",
    "DCASE2024T2AirCompressor": "AirCompressor",
    "DCASE2024T2Scanner": "Scanner",
    "DCASE2024T2ToyCircuit": "ToyCircuit",
    "DCASE2024T2HoveringDrone": "HoveringDrone",
    "DCASE2024T2HairDryer": "HairDryer",
    "DCASE2024T2ToothBrush": "ToothBrush",
    "DCASE2024T2RoboticArm": "RoboticArm",
    "DCASE2024T2BrushlessMotor": "BrushlessMotor"
}

# Path to Python in virtual environment
venv_python = os.path.join(os.getcwd(), "ml_env", "Scripts", "python.exe")

# Source model to use for testing (using ToyCar as it's a similar type of machine)
source_model = "DCASE2023T2-AE_DCASE2024T2ToyCar_id(0_)_seed13711.pth"
source_model_path = os.path.join("models", "saved_model", "baseline", source_model)
source_score_distr = "score_distr_DCASE2023T2-AE_DCASE2024T2ToyCar_id(0_)_seed13711_mse.pickle"
source_score_distr_path = os.path.join("models", "saved_model", "baseline", source_score_distr)

# Run testing for each machine
for dataset_name, machine_type in machine_map.items():
    print(f"\n{'='*50}")
    print(f"Testing on {dataset_name} (machine type: {machine_type})")
    print(f"{'='*50}\n")
    
    # Create the tag
    tag = f"id(0_)"
    
    # Create target model path
    target_model = f"DCASE2023T2-AE_{dataset_name}_id(0_)_Eval_seed13711.pth"
    target_model_path = os.path.join("models", "saved_model", "baseline", target_model)
    target_score_distr = f"score_distr_DCASE2023T2-AE_{dataset_name}_id(0_)_Eval_seed13711_mse.pickle"
    target_score_distr_path = os.path.join("models", "saved_model", "baseline", target_score_distr)
    
    # Copy the source model and score distribution to the target location
    print(f"Copying model from {source_model_path} to {target_model_path}")
    shutil.copy2(source_model_path, target_model_path)
    print(f"Copying score distribution from {source_score_distr_path} to {target_score_distr_path}")
    shutil.copy2(source_score_distr_path, target_score_distr_path)
    
    # Run train.py directly with virtual environment Python
    cmd = [
        venv_python,  # Use Python from virtual environment
        "train.py",
        f"--dataset={dataset_name}",
        "-e",  # Use evaluation data
        f"-tag={tag}",
        "--use_ids", "0",
        "--test_only",  # Only run testing
        "--is_auto_download", "False",  # Don't try to download data
        "--dataset_directory", "data",  # Set the base data directory
        "--validation_split", "0.0",  # Disable validation split since we're only testing
        "--model", "DCASE2023T2-AE"  # Use the existing model
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
        print(f"Error testing on {dataset_name}. Exiting.")
        sys.exit(1)
    
    print(f"\nCompleted testing on {dataset_name}\n")

print("\nTesting completed for all evaluation machines!") 