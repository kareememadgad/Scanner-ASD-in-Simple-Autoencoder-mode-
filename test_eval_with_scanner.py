import os
import subprocess
import sys
import shutil

# Dictionary mapping dataset names to machine type directories
machine_map = {
    "DCASE2024T23DPrinter": "3DPrinter",
    "DCASE2024T2AirCompressor": "AirCompressor",
    "DCASE2024T2ToyCircuit": "ToyCircuit",
    "DCASE2024T2HoveringDrone": "HoveringDrone",
    "DCASE2024T2HairDryer": "HairDryer",
    "DCASE2024T2ToothBrush": "ToothBrush",
    "DCASE2024T2RoboticArm": "RoboticArm",
    "DCASE2024T2BrushlessMotor": "BrushlessMotor"
}

# Path to Python in virtual environment
venv_python = os.path.join(os.getcwd(), "ml_env", "Scripts", "python.exe")

# Source model (Scanner)
source_model = "DCASE2023T2-AE_DCASE2024T2Scanner_id(0_)_Eval_seed13711.pth"
source_model_path = os.path.join("models", "saved_model", "baseline", source_model)
source_score_distr = "score_distr_DCASE2023T2-AE_DCASE2024T2Scanner_id(0_)_Eval_seed13711_mse.pickle"
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
    os.makedirs(os.path.dirname(target_model_path), exist_ok=True)
    shutil.copy2(source_model_path, target_model_path)
    print(f"Copying score distribution from {source_score_distr_path} to {target_score_distr_path}")
    shutil.copy2(source_score_distr_path, target_score_distr_path)
    
    # Run train.py with test_only flag
    cmd = [
        venv_python,
        "train.py",
        f"--dataset={dataset_name}",
        "-e",  # Use evaluation data
        f"-tag={tag}",
        "--use_ids", "0",
        "--test_only",  # Only run testing
        "--model", "DCASE2023T2-AE",
        "--score", "MSE",
        "--seed", "13711",
        "--batch_size", "32",
        "--n_mels", "128",
        "--frames", "5",
        "--n_fft", "1024",
        "--hop_length", "512",
        "--power", "2.0",
        "--is_auto_download", "False",
        "--dataset_directory", "data",
        "--result_directory", "results",
        "--export_dir", "baseline"
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