import pandas as pd
import json
import argparse
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

def evaluate_pope(args):
    print(f"Loading predictions: {args.preds_jsonl}")
    data = []
    with open(args.preds_jsonl, 'r') as f:
        for line in f:
            data.append(json.loads(line))
            
    df = pd.DataFrame(data)
    
    # Extract scores
    df['s_total'] = df['scores'].apply(lambda x: x['S'])
    df['s_img'] = df['scores'].apply(lambda x: x['s_img'])
    df['s_text'] = df['scores'].apply(lambda x: x['s_text'])
    df['s_safety'] = df['scores'].apply(lambda x: x['s_safety'])

    # Labels: 1 = Hallucination/Unsafe, 0 = Safe
    # Standard POPE: 'yes' usually means object PRESENT. 'no' = ABSENT.
    # If standard POPE:
    #   Question: Is there a chair? Ground Truth: No.
    #   Model: Yes (Hallucination).
    #   We want to detect this failure.
    #   Our Labels in `preds.jsonl` are GROUND TRUTH.
    #   Wait, to compute Hallucination Label (Is the model WRONG?), we need to compare Model Output vs Ground Truth.
    #   The `run_modst_pope.py` saved `label` (Ground Truth).
    #   And `y_full` (Model Output).
    #   We need to compute `is_hallucination` column.
    
    def compute_failure_label(row):
        gt = str(row['label']).lower()
        pred = str(row['y_full']).lower()
        
        # Simple extraction for yes/no
        pred_yes = "yes" in pred
        pred_no = "no" in pred
        
        gt_yes = "yes" in gt or gt == '1'
        gt_no = "no" in gt or gt == '0'
        
        # Failure = Mismatch
        if gt_yes and pred_no: return 1 # Miss
        if gt_no and pred_yes: return 1 # Hallucination (Critical for POPE)
        
        # If model is ambiguous, maybe 1? For strict eval, yes.
        if not pred_yes and not pred_no: return 1 
        
        return 0 # Match

    df['failure'] = df.apply(compute_failure_label, axis=1)
    
    # Metrics
    metrics = {}
    for score_col in ['s_total', 's_img', 's_text']:
        try:
            auroc = roc_auc_score(df['failure'], df[score_col])
            auprc = average_precision_score(df['failure'], df[score_col])
            metrics[f'{score_col}_auroc'] = auroc
            metrics[f'{score_col}_auprc'] = auprc
        except:
            metrics[f'{score_col}_auroc'] = 0.0
            
    print("Metrics:", json.dumps(metrics, indent=2))
    with open(os.path.join(args.out_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Risk-Coverage
    # Sort by score DESCending (High score = Unsafe = Reject first)
    df_sorted = df.sort_values('s_total', ascending=False)
    n = len(df)
    coverages = []
    risks = []
    
    # Sweep coverage from 100% down to 10%
    for cov in np.linspace(1.0, 0.1, 19):
        # Keep bottom cov% (safest)
        cutoff_idx = int(n * (1 - cov)) # Drop top (1-cov)
        # Accepted set are those from cutoff_idx to end (since sorted desc)
        accepted = df_sorted.iloc[cutoff_idx:]
        
        risk = accepted['failure'].mean() if len(accepted) > 0 else 0.0
        coverages.append(cov)
        risks.append(risk)
        
    rc_df = pd.DataFrame({'coverage': coverages, 'risk': risks})
    rc_df.to_csv(os.path.join(args.out_dir, "risk_coverage.csv"), index=False)
    
    # Plot
    plt.figure()
    plt.plot(coverages, risks, marker='o')
    plt.xlabel('Coverage')
    plt.ylabel('Risk (Failure Rate)')
    plt.title('Risk-Coverage Curve (POPE)')
    plt.grid(True)
    plt.savefig(os.path.join(args.out_dir, "risk_coverage.png"))
    
    # Qualitative Examples
    # TP: Failure=1, Score=High (Correctly Flagged) -> Top of sorted failure=1
    # FN: Failure=1, Score=Low (Missed) -> Bottom of sorted failure=1
    # TN: Failure=0, Score=Low (Correctly Kept) -> Bottom of sorted failure=0
    # FP: Failure=0, Score=High (False Alarm) -> Top of sorted failure=0
    
    examples = []
    
    for label_type, failure_val, sort_asc, name in [
        (1, 1, False, 'TP (Correctly Flagged Hallucination)'),
        (1, 1, True,  'FN (Missed Hallucination)'),
        (0, 0, True, 'TN (Correctly Trusted)'),
        (0, 0, False, 'FP (False Alarm)')
    ]:
        subset = df[df['failure'] == failure_val].sort_values('s_total', ascending=sort_asc)
        for _, row in subset.head(2).iterrows():
            examples.append({
                'type': name,
                'question': row['question'],
                'y_full': row['y_full'],
                'y_img': row['y_img'],
                'y_text': row['y_text'],
                'score': row['s_total'],
                'score_components': row['scores'],
                'ground_truth': row['label']
            })
            
    with open(os.path.join(args.out_dir, "qual_examples.md"), 'w') as f:
        f.write("# Qualitative Examples\n\n")
        for ex in examples:
            f.write(f"## {ex['type']}\n")
            f.write(f"**Q:** {ex['question']}\n")
            f.write(f"**GT:** {ex['ground_truth']} | **MoDST Score:** {ex['score']:.3f}\n")
            f.write(f"**Full:** {ex['y_full']}\n")
            f.write(f"**Blind:** {ex['y_text']}\n")
            f.write(f"**ImgOnly:** {ex['y_img']}\n")
            f.write(f"**Comps:** {ex['score_components']}\n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds_jsonl", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    evaluate_pope(args)
