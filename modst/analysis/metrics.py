import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve

def compute_metrics(results):
    """Computes AUC-ROC for tripwire scores against grounding labels."""
    scores = [r['scores']['total'] for r in results]
    # Assuming label 0 = correct grounding, 1 = failure (hallucination)
    labels = [r['label'] for r in results]
    
    if len(set(labels)) < 2:
        return {"auc": 0.0, "msg": "Insufficient label diversity"}

    auc = roc_auc_score(labels, scores)
    
    # Coverage vs Error
    # Helpfulness = proportion of samples where policy 'answered' correctly
    answered = [r for r in results if r['decision']['action'] == 'answer']
    safety_violations = sum(1 for r in answered if r['label'] == 1) # Answers on hallucinated samples
    
    return {
        "tripwire_auc": auc,
        "violation_rate": safety_violations / len(answered) if answered else 0.0,
        "coverage": len(answered) / len(results)
    }
