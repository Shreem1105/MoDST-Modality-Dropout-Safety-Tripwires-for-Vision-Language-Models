import argparse
import yaml
import json
import os
from tqdm import tqdm
from modst.models.vlm_wrapper import LlavaWrapper
from modst.datasets.base_dataset import BaseVLMDataset
from modst.datasets.perturbations import ShiftedDataset
from modst.tripwire.score import MoDSTScore
from modst.baselines.vanilla import MoDSTPolicy

def main(config):
    # Initialize components
    model = LlavaWrapper(load_in_4bit=config['quantization'])
    score_engine = MoDSTScore(**config['tripwire_weights'])
    policy = MoDSTPolicy(threshold=config['threshold'])
    
    # Load data
    base_ds = BaseVLMDataset(config['json_path'], config['img_dir'])
    if config['shift_type']:
        val_ds = ShiftedDataset(base_ds, config['shift_type'], config['severity'])
    else:
        val_ds = base_ds

    results = []
    print(f"Running MoDST Experiment: {config['experiment_name']}")
    
    for i in tqdm(range(min(len(val_ds), config['max_samples']))):
        item = val_ds[i]
        
        # 3-Pass Inference
        passes = model.run_modst_passes(item['prompt'], item['image'])
        
        # Tripwire Scoring
        scores = score_engine.compute(passes)
        
        # Action Policy
        decision = policy.act(scores['total'], passes)
        
        results.append({
            "id": item['id'],
            "label": item['label'],
            "passes": passes,
            "scores": scores,
            "decision": decision
        })

    # Save results
    os.makedirs(os.path.dirname(config['output_path']), exist_ok=True)
    with open(config['output_path'], 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {config['output_path']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    main(config)
