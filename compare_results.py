import numpy as np
import os
import argparse
from sklearn.metrics import roc_auc_score, average_precision_score
import json

def load_ground_truth(machine_type, dataset_dir):
    # Load ground truth labels
    gt_path = os.path.join(dataset_dir, machine_type, 'test', 'ground_truth.json')
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    return gt_data

def load_baseline_scores(machine_type):
    # Load baseline scores
    baseline_path = f'baseline_results/{machine_type}_anomaly_scores.npy'
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Baseline results not found at {baseline_path}")
    return np.load(baseline_path)

def evaluate_scores(scores, ground_truth):
    # Convert ground truth to numpy array
    gt = np.array([1 if label == 'anomaly' else 0 for label in ground_truth])
    
    # Calculate metrics
    auc_roc = roc_auc_score(gt, scores)
    auc_pr = average_precision_score(gt, scores)
    
    return {
        'AUC-ROC': auc_roc,
        'AUC-PR': auc_pr
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='data/dcase2024t2/eval_data/raw')
    parser.add_argument('--machine_type', type=str, default='Scanner')
    args = parser.parse_args()
    
    # Load our results
    our_scores_path = f'results/{args.machine_type}_anomaly_scores.npy'
    if not os.path.exists(our_scores_path):
        raise FileNotFoundError(f"Our results not found at {our_scores_path}")
    our_scores = np.load(our_scores_path)
    
    # Load baseline results
    try:
        baseline_scores = load_baseline_scores(args.machine_type)
    except FileNotFoundError:
        print("Baseline results not found. Skipping baseline comparison.")
        baseline_scores = None
    
    # Load ground truth
    ground_truth = load_ground_truth(args.machine_type, args.dataset_dir)
    
    # Evaluate our results
    our_metrics = evaluate_scores(our_scores, ground_truth)
    print("\nOur Results:")
    print(f"AUC-ROC: {our_metrics['AUC-ROC']:.4f}")
    print(f"AUC-PR: {our_metrics['AUC-PR']:.4f}")
    
    # Compare with baseline if available
    if baseline_scores is not None:
        baseline_metrics = evaluate_scores(baseline_scores, ground_truth)
        print("\nBaseline Results:")
        print(f"AUC-ROC: {baseline_metrics['AUC-ROC']:.4f}")
        print(f"AUC-PR: {baseline_metrics['AUC-PR']:.4f}")
        
        print("\nImprovement over baseline:")
        print(f"AUC-ROC: {((our_metrics['AUC-ROC'] - baseline_metrics['AUC-ROC']) / baseline_metrics['AUC-ROC'] * 100):.2f}%")
        print(f"AUC-PR: {((our_metrics['AUC-PR'] - baseline_metrics['AUC-PR']) / baseline_metrics['AUC-PR'] * 100):.2f}%")

if __name__ == '__main__':
    main() 