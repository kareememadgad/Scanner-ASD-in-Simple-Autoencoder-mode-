import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_anomaly_scores(results_dir, machine):
    # Load anomaly scores
    score_file = os.path.join(results_dir, "dev_data", "baseline_MSE", 
                             f"anomaly_score_{machine}_section_00_test_seed13711.csv")
    
    # Load decision results to identify normal/anomaly
    decision_file = os.path.join(results_dir, "dev_data", "baseline_MSE",
                                f"decision_result_{machine}_section_00_test_seed13711.csv")
    
    # Read the files
    scores_df = pd.read_csv(score_file, header=None, names=['file', 'score'])
    decisions_df = pd.read_csv(decision_file, header=None, names=['file', 'decision'])
    
    # Merge scores with decisions
    df = pd.merge(scores_df, decisions_df, on='file')
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot normal samples
    normal_scores = df[df['decision'] == 0]['score']
    plt.hist(normal_scores, bins=30, alpha=0.5, label='Normal', color='blue', density=True)
    
    # Plot anomaly samples
    anomaly_scores = df[df['decision'] == 1]['score']
    plt.hist(anomaly_scores, bins=30, alpha=0.5, label='Anomaly', color='red', density=True)
    
    # Add labels and title
    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.title(f'Anomaly Score Distribution - {machine.replace("DCASE2024T2", "")}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    save_path = os.path.join(results_dir, "anomaly_plots")
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f'{machine}_anomaly_distribution.png'))
    plt.close()

def main():
    print("\nGenerating Anomaly Score Distribution Plots")
    print("=========================================")
    
    # List of machines
    machines = [
        "DCASE2024T2bearing",
        "DCASE2024T2fan",
        "DCASE2024T2gearbox",
        "DCASE2024T2slider",
        "DCASE2024T2ToyCar",
        "DCASE2024T2ToyTrain",
        "DCASE2024T2valve"
    ]
    
    # Generate plots for each machine
    for machine in machines:
        print(f"Generating plot for {machine}...")
        plot_anomaly_scores("scanner_test_results", machine)
    
    print("\nPlots have been generated and saved in scanner_test_results/anomaly_plots/")

if __name__ == "__main__":
    main() 