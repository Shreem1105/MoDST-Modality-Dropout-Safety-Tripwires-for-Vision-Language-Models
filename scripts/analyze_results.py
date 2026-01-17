import json
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def analyze(results_path, output_dir):
    with open(results_path, 'r') as f:
        results = json.load(f)
        
    print(f"Loaded {len(results)} results from {results_path}")
    
    # Extract scores
    scores = np.array([r['scores']['total'] for r in results])
    decisions = [r['decision']['action'] for r in results]
    
    # Basic Statistics
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    median_score = np.median(scores)
    
    # "Flagged" rate (Fallback)
    fallback_rate = decisions.count('fallback') / len(decisions)
    
    print("\n=== ICLR 2026 Workshop Experiment Analysis ===")
    print(f"Total Samples: {len(results)}")
    print(f"MoDST Score Mean:   {mean_score:.4f}")
    print(f"MoDST Score Median: {median_score:.4f}")
    print(f"MoDST Score Std:    {std_score:.4f}")
    print(f"Fallback Rate:      {fallback_rate:.2%} (Samples flagged as unsafe/ungrounded)")
    print("==============================================\n")
    
    # Generate Plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "score_distribution.png")
    
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(mean_score, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_score:.2f}')
    plt.axvline(0.5, color='orange', linestyle='dashed', linewidth=1, label='Threshold: 0.5')
    
    plt.xlabel('MoDST Tripwire Score (0=Safe, 1=Unsafe)')
    plt.ylabel('Number of Samples')
    plt.title('Distribution of MoDST Safety Scores on COCO-Val')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="outputs/results.json")
    parser.add_argument("--out", type=str, default="outputs")
    args = parser.parse_args()
    
    analyze(args.results, args.out)
