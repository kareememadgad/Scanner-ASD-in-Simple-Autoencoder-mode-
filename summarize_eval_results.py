import os
import pandas as pd
import glob

# Directory containing the results
results_dir = "results/eval_data/baseline_MSE"

# List of evaluation machines
eval_machines = [
    "DCASE2024T23DPrinter",
    "DCASE2024T2AirCompressor",
    "DCASE2024T2Scanner",
    "DCASE2024T2ToyCircuit",
    "DCASE2024T2HoveringDrone",
    "DCASE2024T2HairDryer",
    "DCASE2024T2ToothBrush",
    "DCASE2024T2RoboticArm",
    "DCASE2024T2BrushlessMotor"
]

# Initialize lists to store results
machine_names = []
auc_source = []
auc_target = []
pauc = []
pauc_source = []
pauc_target = []
precision_source = []
precision_target = []
recall_source = []
recall_target = []
f1_source = []
f1_target = []

# Process each machine
for machine in eval_machines:
    # Find the ROC result file
    roc_file = glob.glob(os.path.join(results_dir, f"result_{machine}_test_seed13711_id(0_)_Eval_roc.csv"))[0]
    
    # Read the ROC results
    df = pd.read_csv(roc_file)
    
    # Get the arithmetic mean row
    mean_row = df[df['section'] == 'arithmetic mean'].iloc[0]
    
    # Append results
    machine_names.append(machine.replace("DCASE2024T2", ""))
    auc_source.append(mean_row['AUC (source)'])
    auc_target.append(mean_row['AUC (target)'])
    pauc.append(mean_row['pAUC'])
    pauc_source.append(mean_row['pAUC (source)'])
    pauc_target.append(mean_row['pAUC (target)'])
    precision_source.append(mean_row['precision (source)'])
    precision_target.append(mean_row['precision (target)'])
    recall_source.append(mean_row['recall (source)'])
    recall_target.append(mean_row['recall (target)'])
    f1_source.append(mean_row['F1 score (source)'])
    f1_target.append(mean_row['F1 score (target)'])

# Create summary DataFrame
summary_df = pd.DataFrame({
    'Machine': machine_names,
    'AUC (source)': auc_source,
    'AUC (target)': auc_target,
    'pAUC': pauc,
    'pAUC (source)': pauc_source,
    'pAUC (target)': pauc_target,
    'Precision (source)': precision_source,
    'Precision (target)': precision_target,
    'Recall (source)': recall_source,
    'Recall (target)': recall_target,
    'F1 score (source)': f1_source,
    'F1 score (target)': f1_target
})

# Calculate mean and std for each metric
mean_row = summary_df.mean(numeric_only=True)
std_row = summary_df.std(numeric_only=True)

# Add mean and std rows
summary_df.loc['Mean'] = ['Mean'] + list(mean_row)
summary_df.loc['Std'] = ['Std'] + list(std_row)

# Save to CSV
output_file = "scanner_evaluation_summary.csv"
summary_df.to_csv(output_file, index=False)
print(f"Summary saved to {output_file}")

# Print summary
print("\nEvaluation Machines Summary:")
print("=" * 80)
print(summary_df.to_string())
print("=" * 80) 