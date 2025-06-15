import os
import pandas as pd
import numpy as np
from glob import glob
from datetime import datetime

def load_results():
    results_dir = "results/dev_data/baseline_MSE"
    machines = [
        "bearing", "fan", "gearbox", "slider", 
        "ToyCar", "ToyTrain", "valve"
    ]
    
    summary = []
    for machine in machines:
        # Find the result file for this machine
        result_files = glob(os.path.join(results_dir, f"result_DCASE2024T2{machine}_test_seed*_roc.csv"))
        if not result_files:
            print(f"No results found for {machine}")
            continue
            
        # Load the results
        df = pd.read_csv(result_files[0])
        
        # Calculate metrics
        metrics = {
            'Machine': machine,
            'Source AUC': df['AUC (source)'].iloc[0],
            'Target AUC': df['AUC (target)'].iloc[0],
            'Source pAUC': df['pAUC (source)'].iloc[0],
            'Target pAUC': df['pAUC (target)'].iloc[0],
            'Source F1': df['F1 score (source)'].iloc[0],
            'Target F1': df['F1 score (target)'].iloc[0],
            'Source Precision': df['precision (source)'].iloc[0],
            'Target Precision': df['precision (target)'].iloc[0],
            'Source Recall': df['recall (source)'].iloc[0],
            'Target Recall': df['recall (target)'].iloc[0]
        }
        summary.append(metrics)
    
    return pd.DataFrame(summary)

def save_summary(df, results_dir):
    # Create timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(results_dir, f"summary_results_{timestamp}.csv")
    
    # Save detailed results
    df.to_csv(summary_file, index=False)
    
    # Create and save a summary with averages
    summary_stats = pd.DataFrame({
        'Metric': ['Average Source AUC', 'Average Target AUC', 
                  'Average Source pAUC', 'Average Target pAUC',
                  'Average Source F1', 'Average Target F1',
                  'Average Source Precision', 'Average Target Precision',
                  'Average Source Recall', 'Average Target Recall'],
        'Value': [df['Source AUC'].mean(), df['Target AUC'].mean(),
                 df['Source pAUC'].mean(), df['Target pAUC'].mean(),
                 df['Source F1'].mean(), df['Target F1'].mean(),
                 df['Source Precision'].mean(), df['Target Precision'].mean(),
                 df['Source Recall'].mean(), df['Target Recall'].mean()]
    })
    
    stats_file = os.path.join(results_dir, f"summary_statistics_{timestamp}.csv")
    summary_stats.to_csv(stats_file, index=False)
    
    return summary_file, stats_file

def main():
    print("\nDCASE 2024 Task 2 Results Summary")
    print("=================================")
    
    df = load_results()
    
    # Sort by source AUC
    df = df.sort_values('Source AUC', ascending=False)
    
    # Save results
    results_dir = "results/dev_data/baseline_MSE"
    summary_file, stats_file = save_summary(df, results_dir)
    
    # Print summary
    print("\nResults by Machine (sorted by Source AUC):")
    print("----------------------------------------")
    for _, row in df.iterrows():
        print(f"\n{row['Machine']}:")
        print(f"  Source Domain:")
        print(f"    AUC:  {row['Source AUC']:.4f}")
        print(f"    pAUC: {row['Source pAUC']:.4f}")
        print(f"    F1:   {row['Source F1']:.4f}")
        print(f"  Target Domain:")
        print(f"    AUC:  {row['Target AUC']:.4f}")
        print(f"    pAUC: {row['Target pAUC']:.4f}")
        print(f"    F1:   {row['Target F1']:.4f}")
    
    # Print averages
    print("\nAverage Performance:")
    print("-------------------")
    print(f"Average Source AUC:  {df['Source AUC'].mean():.4f}")
    print(f"Average Target AUC:  {df['Target AUC'].mean():.4f}")
    print(f"Average Source pAUC: {df['Source pAUC'].mean():.4f}")
    print(f"Average Target pAUC: {df['Target pAUC'].mean():.4f}")
    print(f"Average Source F1:   {df['Source F1'].mean():.4f}")
    print(f"Average Target F1:   {df['Target F1'].mean():.4f}")
    
    print(f"\nDetailed results saved to: {summary_file}")
    print(f"Summary statistics saved to: {stats_file}")

if __name__ == "__main__":
    main() 