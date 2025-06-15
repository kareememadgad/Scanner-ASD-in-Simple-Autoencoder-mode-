import os
import urllib.request
import zipfile

# Create directories
parent_dir = "data"
eval_data_dir = os.path.join(parent_dir, "dcase2024t2", "eval_data", "raw")
os.makedirs(eval_data_dir, exist_ok=True)

# List of machines
machines = [
    "3DPrinter",
    "AirCompressor",
    "Scanner",
    "ToyCircuit",
    "HoveringDrone",
    "HairDryer",
    "ToothBrush",
    "RoboticArm",
    "BrushlessMotor"
]

# Download and extract data for each machine
for machine in machines:
    print(f"\nDownloading data for {machine}...")
    zip_file = f"eval_data_{machine}_test.zip"
    url = f"https://zenodo.org/records/11363076/files/{zip_file}"
    zip_path = os.path.join(eval_data_dir, zip_file)
    
    # Download
    print(f"Downloading from {url}")
    try:
        urllib.request.urlretrieve(url, zip_path)
        print(f"Download complete: {zip_file}")
        
        # Extract
        print(f"Extracting {zip_file}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(eval_data_dir)
        print(f"Extraction complete for {machine}")
        
        # Remove zip file
        os.remove(zip_path)
        print(f"Removed zip file: {zip_file}")
    except Exception as e:
        print(f"Error processing {machine}: {str(e)}")

print("\nDownload complete for all machines!")

# Run the renaming script
import subprocess
subprocess.run(["python", "tools/rename_eval_wav.py", "--dataset_parent_dir=data", "--dataset_type=DCASE2024T2"]) 