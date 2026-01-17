import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from scipy.stats import gaussian_kde

# Set professional style (mimicking seaborn 'whitegrid' + 'ticks')
plt.style.use('fast')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['figure.dpi'] = 300

COLORS = {
    'primary': '#2563EB',    # Royal Blue
    'secondary': '#DC2626',  # Red
    'tertiary': '#10B981',   # Emerald
    'neutral': '#4B5563',    # Gray
    'accent': '#8B5CF6'      # Violet
}

def load_data(path):
    with open(path, 'r') as f:
        return json.load(f)

def plot_kde_distribution(results, output_dir):
    scores = np.array([r['scores']['total'] for r in results])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram underneath
    n, bins, patches = ax.hist(scores, bins=40, density=True, color=COLORS['primary'], alpha=0.15, edgecolor='none')
    
    # KDE Line
    density = gaussian_kde(scores)
    xs = np.linspace(0, 1, 200)
    ys = density(xs)
    ax.plot(xs, ys, color=COLORS['primary'], linewidth=3, label='MoDST Score Density')
    
    # Fill under KDE
    ax.fill_between(xs, ys, alpha=0.2, color=COLORS['primary'])
    
    # Threshold Line
    threshold = 0.5
    ax.axvline(threshold, color=COLORS['secondary'], linestyle='--', linewidth=2, label=f'Safety Threshold ({threshold})')
    
    # Annotations
    ax.text(0.1, max(ys)*0.8, "SAFE / GROUNDED", color=COLORS['neutral'], fontsize=12, fontweight='bold', alpha=0.7)
    ax.text(0.7, max(ys)*0.8, "UNSAFE / HALLUCINATED", color=COLORS['secondary'], fontsize=12, fontweight='bold', alpha=0.7)
    
    # Styling
    ax.set_xlim(0, 1)
    ax.set_xlabel('Tripwire Score Magnitude', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Hallucination Risk (MoDST)', fontsize=14, pad=15, fontweight='bold')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.legend(loc='upper right', frameon=False)
    
    out_path = os.path.join(output_dir, 'fig1_score_distribution_stunning.png')
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved {out_path}")
    plt.close()

def plot_component_breakdown(results, output_dir):
    # Split into Safe (Answered) vs Unsafe (Fallback)
    safe_samples = [r for r in results if r['decision']['action'] == 'answer']
    unsafe_samples = [r for r in results if r['decision']['action'] == 'fallback']
    
    components = ['grounding_discrepancy', 'text_prior_collapse', 'safety_gap']
    labels = ['Grounding\nDiscrepancy', 'Text Prior\nCollapse', 'Safety\nGap']
    
    avg_safe = [np.mean([r['scores'][c] for r in safe_samples]) for c in components]
    avg_unsafe = [np.mean([r['scores'][c] for r in unsafe_samples]) for c in components]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(components))
    width = 0.35
    
    # Bars
    rects1 = ax.bar(x - width/2, avg_safe, width, label='Safe (Answered)', color=COLORS['tertiary'], alpha=0.9)
    rects2 = ax.bar(x + width/2, avg_unsafe, width, label='Unsafe (Fallback)', color=COLORS['secondary'], alpha=0.9)
    
    # Styling
    ax.set_ylabel('Average Component Score', fontsize=12, fontweight='bold')
    ax.set_title('Why did MoDST flag samples? (Component Analysis)', fontsize=14, pad=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax.legend(fontsize=11)
    
    # Add values on top
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    
    ax.set_ylim(0, 1.1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    out_path = os.path.join(output_dir, 'fig2_component_breakdown_stunning.png')
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved {out_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results.json")
    parser.add_argument("--out", type=str, default="paper_visuals")
    args = parser.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    data = load_data(args.results)
    
    print("Generating Graph 1...")
    plot_kde_distribution(data, args.out)
    
    print("Generating Graph 2...")
    plot_component_breakdown(data, args.out)
    
    print("Done!")
