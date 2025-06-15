# Get arguments
param(
    [string]$dataset = "DCASE2024T2Scanner",
    [string]$dev_eval = "-d"
)

Write-Host "Starting training for dataset: ${dataset}"
Write-Host "Dev/Eval: ${dev_eval}"

# Extract machine type from dataset name (e.g., DCASE2024T2Scanner -> Scanner)
$machine_type = $dataset -replace "DCASE2024T2", ""

# Set absolute path to Python executable
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PYTHON_PATH = Join-Path $SCRIPT_DIR "ml_env\Scripts\python.exe"

# Check if Python exists
if (-not (Test-Path $PYTHON_PATH)) {
    Write-Host "Error: Python executable not found at $PYTHON_PATH"
    Write-Host "Please create the virtual environment first"
    exit 1
}

Write-Host "Using Python at: $PYTHON_PATH"

# Create results directory if it doesn't exist
$results_dir = Join-Path $SCRIPT_DIR "results\$machine_type"
New-Item -ItemType Directory -Force -Path $results_dir | Out-Null

# Run training
Write-Host "Starting training for machine type: ${machine_type}"
& $PYTHON_PATH train.py `
    --dataset_dir="data/dcase2024t2/eval_data/raw" `
    --machine_type=$machine_type `
    --input_dim=128 `
    --block_size=313 `
    --batch_size=32 `
    --epochs=100 `
    --save_interval=10

if ($LASTEXITCODE -eq 0) {
    Write-Host "Training completed successfully"
    Write-Host "Results saved in $results_dir"
    Write-Host "1. performance_metrics.xlsx - Contains all performance metrics"
    Write-Host "2. anomaly_scores.xlsx - Contains raw anomaly scores and labels"
} else {
    Write-Host "Training failed"
    exit 1
} 