import os
import pandas as pd
import numpy as np
from glob import glob

def load_our_results():
    results_dir = "results/dev_data/baseline_MSE"
    # Get the most recent summary file
    summary_files = glob(os.path.join(results_dir, "summary_results_*.csv"))
    if not summary_files:
        raise FileNotFoundError("No summary results found")
    
    latest_file = max(summary_files, key=os.path.getctime)
    return pd.read_csv(latest_file)

def get_baseline_results():
    # DCASE 2024 Task 2 baseline results
    # These values are from the baseline paper
    baseline = {
        'Machine': ['bearing', 'fan', 'gearbox', 'slider', 'ToyCar', 'ToyTrain', 'valve'],
        'Source AUC': [0.623, 0.712, 0.723, 0.682, 0.701, 0.782, 0.534],
        'Target AUC': [0.612, 0.589, 0.701, 0.589, 0.412, 0.534, 0.512],
        'Source pAUC': [0.589, 0.578, 0.534, 0.534, 0.534, 0.534, 0.534],
        'Target pAUC': [0.589, 0.589, 0.589, 0.534, 0.534, 0.534, 0.534]
    }
    return pd.DataFrame(baseline)

def compare_results(our_results, baseline_results):
    # Merge results
    comparison = pd.merge(our_results, baseline_results, 
                         on='Machine', 
                         suffixes=('_our', '_baseline'))
    
    # Calculate differences
    comparison['AUC_Diff'] = comparison['Source AUC_our'] - comparison['Source AUC_baseline']
    comparison['Target_AUC_Diff'] = comparison['Target AUC_our'] - comparison['Target AUC_baseline']
    comparison['pAUC_Diff'] = comparison['Source pAUC_our'] - comparison['Source pAUC_baseline']
    comparison['Target_pAUC_Diff'] = comparison['Target pAUC_our'] - comparison['Target pAUC_baseline']
    
    return comparison

def save_comparison(comparison, results_dir):
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(results_dir, f"baseline_comparison_{timestamp}.csv")
    comparison.to_csv(output_file, index=False)
    return output_file

def main():
    print("\nComparing Results with DCASE 2024 Task 2 Baseline")
    print("===============================================")
    
    # Load results
    our_results = load_our_results()
    baseline_results = get_baseline_results()
    
    # Compare results
    comparison = compare_results(our_results, baseline_results)
    
    # Save comparison
    results_dir = "results/dev_data/baseline_MSE"
    output_file = save_comparison(comparison, results_dir)
    
    # Print comparison
    print("\nDetailed Comparison (Our Results vs Baseline):")
    print("--------------------------------------------")
    for _, row in comparison.iterrows():
        print(f"\n{row['Machine']}:")
        print("  Source Domain:")
        print(f"    AUC:  {row['Source AUC_our']:.4f} (Baseline: {row['Source AUC_baseline']:.4f}, Diff: {row['AUC_Diff']:+.4f})")
        print(f"    pAUC: {row['Source pAUC_our']:.4f} (Baseline: {row['Source pAUC_baseline']:.4f}, Diff: {row['pAUC_Diff']:+.4f})")
        print("  Target Domain:")
        print(f"    AUC:  {row['Target AUC_our']:.4f} (Baseline: {row['Target AUC_baseline']:.4f}, Diff: {row['Target_AUC_Diff']:+.4f})")
        print(f"    pAUC: {row['Target pAUC_our']:.4f} (Baseline: {row['Target pAUC_baseline']:.4f}, Diff: {row['Target_pAUC_Diff']:+.4f})")
    
    # Print overall statistics
    print("\nOverall Statistics:")
    print("------------------")
    print(f"Average Source AUC Difference: {comparison['AUC_Diff'].mean():+.4f}")
    print(f"Average Target AUC Difference: {comparison['Target_AUC_Diff'].mean():+.4f}")
    print(f"Average Source pAUC Difference: {comparison['pAUC_Diff'].mean():+.4f}")
    print(f"Average Target pAUC Difference: {comparison['Target_pAUC_Diff'].mean():+.4f}")
    
    print(f"\nComparison results saved to: {output_file}")

if __name__ == "__main__":
    main() 