import matplotlib.pyplot as plt
import numpy as np

def plot_tripwire_distribution(results, output_path):
    labels = np.array([r['label'] for r in results])
    scores = np.array([r['scores']['total'] for r in results])
    
    plt.figure(figsize=(8, 5))
    plt.hist(scores[labels == 0], bins=20, alpha=0.5, label='Grounded (Safe)')
    plt.hist(scores[labels == 1], bins=20, alpha=0.5, label='Shifted (Unsafe)')
    plt.xlabel('MoDST Tripwire Score')
    plt.ylabel('Count')
    plt.title('Tripwire Score Distribution')
    plt.legend()
    plt.savefig(output_path)
    plt.close()

def plot_risk_coverage(results, output_path):
    # Sweep threshold
    scores = sorted([r['scores']['total'] for r in results])
    coverage = []
    risk = []
    
    for t in scores:
        answered = [r for r in results if r['scores']['total'] <= t]
        if not answered: continue
        cov = len(answered) / len(results)
        rsk = sum(1 for r in answered if r['label'] == 1) / len(answered)
        coverage.append(cov)
        risk.append(rsk)
        
    plt.figure(figsize=(8, 5))
    plt.plot(coverage, risk, marker='o')
    plt.xlabel('Coverage')
    plt.ylabel('Risk (Safety Violation Rate)')
    plt.title('MoDST Risk-Coverage Curve')
    plt.savefig(output_path)
    plt.close()
