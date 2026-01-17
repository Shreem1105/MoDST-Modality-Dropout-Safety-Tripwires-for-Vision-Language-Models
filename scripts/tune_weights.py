import argparse
import yaml
import json
import os
from modst.tripwire.score import MoDSTScore
from modst.analysis.metrics import compute_metrics
import numpy as np

def calibrate(results_path):
    """
    Grid search for optimal alpha/beta/gamma weights based on AUC.
    Requires a results file from run_modst.py containing raw tripwire outputs.
    """
    with open(results_path, 'r') as f:
        data = json.load(f)

    best_auc = 0.0
    best_weights = None
    
    # Grid search
    print("Starting Weight Calibration...")
    for alpha in np.arange(0.1, 0.9, 0.1):
        for beta in np.arange(0.1, 0.9, 0.1):
            gamma = 1.0 - (alpha + beta)
            if gamma < 0: continue
            
            # Recompute total scores with new weights
            scorer = MoDSTScore(alpha, beta, gamma)
            temp_results = []
            for item in data:
                # We need to manually re-aggregate since we tracked component scores
                # Note: This implies we need component scores stored in results. 
                # Assuming results['scores'] has 'grounding_discrepancy', 'text_prior_collapse', 'safety_gap'
                s_g = item['scores']['grounding_discrepancy']
                s_f = item['scores']['text_prior_collapse']
                s_s = item['scores']['safety_gap']
                
                div = s_g * s_f # Contrastive Divergence
                
                total = alpha * div + beta * s_g + gamma * s_s
                
                new_item = item.copy()
                new_item['scores']['total'] = total
                temp_results.append(new_item)
                
            metrics = compute_metrics(temp_results)
            if metrics['auc'] > best_auc:
                best_auc = metrics['auc']
                best_weights = (alpha, beta, gamma)
                print(f"New Best: AUC={best_auc:.4f} | Weights={best_weights}")

    print(f"\nFinal Calibrated Weights: Alpha={best_weights[0]:.1f}, Beta={best_weights[1]:.1f}, Gamma={best_weights[2]:.1f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", type=str, required=True, help="Path to run_modst results json")
    args = parser.parse_args()
    
    calibrate(args.results_path)
