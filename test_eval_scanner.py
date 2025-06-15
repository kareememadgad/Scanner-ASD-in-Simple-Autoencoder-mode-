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

# Source model to use for testing (using Scanner as the source model)
source_model = "scanner_model.pth"
source_model_path = os.path.join("models", source_model)

# Run testing for each machine
for dataset_name, machine_type in machine_map.items():
    print(f"\n{'='*50}")
    print(f"Testing on {dataset_name} (machine type: {machine_type})")
    print(f"{'='*50}\n")
    
    # Create target model path
    target_model = f"{machine_type}_model.pth"
    target_model_path = os.path.join("models", target_model)
    
    # Copy the source model to the target location
    print(f"Copying model from {source_model_path} to {target_model_path}")
    shutil.copy2(source_model_path, target_model_path)
    
    # Run train.py directly with virtual environment Python
    cmd = [
        venv_python,  # Use Python from virtual environment
        "train.py",
        f"--dataset={dataset_name}",
        f"--model={machine_type}_model",
        "--epochs=1",  # Just run one epoch for testing
        "--batch_size=32",
        "--lr=0.001",
        "--eval"  # Use evaluation data
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