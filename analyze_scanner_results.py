import os
import pandas as pd
import glob

def load_results(results_dir):
    results = []
    
    # Find all result files
    result_files = glob.glob(os.path.join(results_dir, "dev_data", "baseline_MSE", "result_*_test_*.csv"))
    
    for file in result_files:
        # Extract machine name from filename
        machine = file.split("result_")[1].split("_test_")[0]
        
        # Read results
        df = pd.read_csv(file)
        
        # Extract metrics
        metrics = {
            'Machine': machine,
            'Source AUC': df['AUC (source)'].values[0],
            'Target AUC': df['AUC (target)'].values[0],
            'Source pAUC': df['pAUC (source)'].values[0],
            'Target pAUC': df['pAUC (target)'].values[0],
            'Source F1': df['F1 score (source)'].values[0],
            'Target F1': df['F1 score (target)'].values[0]
        }
        
        results.append(metrics)
    
    return pd.DataFrame(results)

def main():
    print("\nAnalyzing Scanner Model Performance Against All Machines")
    print("===================================================")
    
    # Load results
    results = load_results("scanner_test_results")
    
    # Print detailed results
    print("\nDetailed Results:")
    print("-----------------")
    for _, row in results.iterrows():
        print(f"\n{row['Machine']}:")
        print("  Source Domain:")
        print(f"    AUC:  {row['Source AUC']:.4f}")
        print(f"    pAUC: {row['Source pAUC']:.4f}")
        print(f"    F1:   {row['Source F1']:.4f}")
        print("  Target Domain:")
        print(f"    AUC:  {row['Target AUC']:.4f}")
        print(f"    pAUC: {row['Target pAUC']:.4f}")
        print(f"    F1:   {row['Target F1']:.4f}")
    
    # Print overall statistics
    print("\nOverall Statistics:")
    print("------------------")
    print(f"Average Source AUC:  {results['Source AUC'].mean():.4f}")
    print(f"Average Target AUC:  {results['Target AUC'].mean():.4f}")
    print(f"Average Source pAUC: {results['Source pAUC'].mean():.4f}")
    print(f"Average Target pAUC: {results['Target pAUC'].mean():.4f}")
    print(f"Average Source F1:   {results['Source F1'].mean():.4f}")
    print(f"Average Target F1:   {results['Target F1'].mean():.4f}")
    
    # Save results
    output_file = os.path.join("scanner_test_results", "scanner_performance_summary.csv")
    results.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main() 