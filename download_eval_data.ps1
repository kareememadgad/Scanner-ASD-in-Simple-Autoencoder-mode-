# Create directories
New-Item -ItemType Directory -Force -Path "data/dcase2024t2/eval_data/raw"

# Download and extract data for each machine type
$machine_types = @(
    "3DPrinter",
    "AirCompressor",
    "Scanner",
    "ToyCircuit",
    "HoveringDrone",
    "HairDryer",
    "ToothBrush",
    "RoboticArm",
    "BrushlessMotor"
)

foreach ($machine_type in $machine_types) {
    $url = "https://zenodo.org/records/11363076/files/eval_data_${machine_type}_test.zip"
    $output = "data/dcase2024t2/eval_data/raw/eval_data_${machine_type}_test.zip"
    
    Write-Host "Downloading $machine_type data..."
    Invoke-WebRequest -Uri $url -OutFile $output
    
    Write-Host "Extracting $machine_type data..."
    Expand-Archive -Path $output -DestinationPath "data/dcase2024t2/eval_data/raw" -Force
}

# Add reference labels to test data
python tools/rename_eval_wav.py --dataset_parent_dir=data --dataset_type=DCASE2024T2 