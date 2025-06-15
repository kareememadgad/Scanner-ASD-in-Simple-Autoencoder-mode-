import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

def load_results(results_dir):
    results = []
    result_files = glob.glob(os.path.join(results_dir, "dev_data", "baseline_MSE", "result_*_test_*.csv"))
    
    for file in result_files:
        machine = file.split("result_")[1].split("_test_")[0]
        df = pd.read_csv(file)
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

def plot_auc_comparison(results):
    plt.figure(figsize=(12, 6))
    machines = results['Machine'].str.replace('DCASE2024T2', '')
    x = range(len(machines))
    
    plt.plot(x, results['Source AUC'], 'bo-', label='Source AUC', linewidth=2)
    plt.plot(x, results['Target AUC'], 'ro-', label='Target AUC', linewidth=2)
    
    plt.xticks(x, machines, rotation=45)
    plt.ylim(0.3, 0.8)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.title('AUC Comparison Across Machines')
    plt.xlabel('Machine Type')
    plt.ylabel('AUC Score')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join('scanner_test_results', 'auc_comparison.png'))
    plt.close()

def plot_pauc_comparison(results):
    plt.figure(figsize=(12, 6))
    machines = results['Machine'].str.replace('DCASE2024T2', '')
    x = range(len(machines))
    
    plt.plot(x, results['Source pAUC'], 'go-', label='Source pAUC', linewidth=2)
    plt.plot(x, results['Target pAUC'], 'mo-', label='Target pAUC', linewidth=2)
    
    plt.xticks(x, machines, rotation=45)
    plt.ylim(0.3, 0.7)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.title('pAUC Comparison Across Machines')
    plt.xlabel('Machine Type')
    plt.ylabel('pAUC Score')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join('scanner_test_results', 'pauc_comparison.png'))
    plt.close()

def plot_metrics_heatmap(results):
    # Prepare data for heatmap
    metrics = ['Source AUC', 'Target AUC', 'Source pAUC', 'Target pAUC']
    machines = results['Machine'].str.replace('DCASE2024T2', '')
    data = results[metrics].values.T
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlBu_r',
                xticklabels=machines,
                yticklabels=metrics,
                center=0.5,
                vmin=0.3,
                vmax=0.8)
    
    plt.title('Performance Metrics Heatmap')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join('scanner_test_results', 'metrics_heatmap.png'))
    plt.close()

def main():
    print("\nGenerating Performance Visualization Graphs")
    print("=========================================")
    
    # Load results
    results = load_results("scanner_test_results")
    
    # Create plots
    plot_auc_comparison(results)
    plot_pauc_comparison(results)
    plot_metrics_heatmap(results)
    
    print("\nGraphs have been generated and saved in the scanner_test_results directory:")
    print("1. auc_comparison.png - Line plot comparing Source and Target AUC")
    print("2. pauc_comparison.png - Line plot comparing Source and Target pAUC")
    print("3. metrics_heatmap.png - Heatmap of all performance metrics")

if __name__ == "__main__":
    main() 