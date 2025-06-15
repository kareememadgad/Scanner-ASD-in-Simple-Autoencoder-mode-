import numpy as np
import os
import json
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def load_ground_truth(machine_type, dataset_dir):
    # Load ground truth labels from CSV
    gt_path = os.path.join(dataset_dir, machine_type, f'ground_truth_{machine_type}_section_00_test.csv')
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth file not found at {gt_path}")
    
    # Read CSV file
    gt_df = pd.read_csv(gt_path, header=None, names=['filename', 'label'])
    return gt_df

def calculate_metrics(scores, ground_truth):
    # Convert ground truth to numpy array
    gt = ground_truth['label'].values  # Use the 'label' column
    
    # Ensure scores and ground truth have the same length
    if len(scores) != len(gt):
        raise ValueError(f"Length mismatch: scores ({len(scores)}) != ground truth ({len(gt)})")
    
    # Calculate AUC-ROC
    auc_roc = roc_auc_score(gt, scores)
    
    # Calculate AUC-PR
    auc_pr = average_precision_score(gt, scores)
    
    # Calculate pAUC
    precision, recall, thresholds = precision_recall_curve(gt, scores)
    pauc = np.trapz(precision, recall)
    
    # Calculate F1 score at optimal threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    
    return {
        'AUC-ROC': float(auc_roc),
        'AUC-PR': float(auc_pr),
        'pAUC': float(pauc),
        'F1-score': float(optimal_f1),
        'Optimal threshold': float(optimal_threshold)
    }

def calculate_domain_metrics(scores, ground_truth, domain_info):
    # Calculate metrics for each domain
    domain_metrics = {}
    for domain in set(domain_info):
        domain_mask = np.array(domain_info) == domain
        domain_scores = scores[domain_mask]
        domain_gt = np.array([1 if label == 'anomaly' else 0 for label in ground_truth])[domain_mask]
        
        metrics = calculate_metrics(domain_scores, domain_gt)
        domain_metrics[domain] = metrics
    
    return domain_metrics

def main():
    # Load results
    result_path = 'results/Scanner_anomaly_scores.npy'
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"Results not found at {result_path}")
    
    scores = np.load(result_path)
    print(f"Loaded {len(scores)} anomaly scores")
    
    # Load ground truth and domain information
    dataset_dir = 'data/dcase2024t2/eval_data/raw'
    machine_type = 'Scanner'
    
    try:
        ground_truth = load_ground_truth(machine_type, dataset_dir)
        print(f"Loaded {len(ground_truth)} ground truth labels")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure the ground truth file exists at:", 
              os.path.join(dataset_dir, machine_type, f'ground_truth_{machine_type}_section_00_test.csv'))
        return
    
    # Calculate overall metrics
    try:
        overall_metrics = calculate_metrics(scores, ground_truth)
    except ValueError as e:
        print(f"Error calculating metrics: {e}")
        return
    
    # Print results
    print("\nOverall Performance Metrics:")
    print("=" * 50)
    for metric, value in overall_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'Machine Type': [machine_type],
        'AUC-ROC': [overall_metrics['AUC-ROC']],
        'AUC-PR': [overall_metrics['AUC-PR']],
        'pAUC': [overall_metrics['pAUC']],
        'F1-score': [overall_metrics['F1-score']]
    })
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Save to CSV
    results_df.to_csv(f'results/{machine_type}_metrics.csv', index=False)
    print(f"\nResults saved to results/{machine_type}_metrics.csv")
    
    # Save detailed metrics to JSON
    detailed_results = {
        'machine_type': machine_type,
        'overall_metrics': overall_metrics,
        'optimal_threshold': overall_metrics['Optimal threshold']
    }
    
    with open(f'results/{machine_type}_detailed_metrics.json', 'w') as f:
        json.dump(detailed_results, f, indent=4, cls=NumpyEncoder)
    
    print(f"Detailed metrics saved to results/{machine_type}_detailed_metrics.json")

if __name__ == '__main__':
    main() 