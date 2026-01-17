import argparse
import yaml
import json
import os
import time
from tqdm import tqdm
from modst.models.vlm_wrapper import LlavaWrapper, InstructBlipWrapper
from modst.datasets.base_dataset import BaseVLMDataset
from modst.datasets.perturbations import ShiftedDataset
from modst.tripwire.score import MoDSTScore
from modst.baselines.vanilla import MoDSTPolicy

def main(config):
    # Initialize components
    grounding_prompt = config.get('grounding_prompt', None)
    
    if "instructblip" in config['model_id'].lower():
        model = InstructBlipWrapper(model_id=config['model_id'], load_in_4bit=config['quantization'], grounding_prompt=grounding_prompt)
    else:
        model = LlavaWrapper(load_in_4bit=config['quantization'], grounding_prompt=grounding_prompt)

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
    
    max_limit = config['max_samples'] if config['max_samples'] > 0 else len(val_ds)
    for i in tqdm(range(min(len(val_ds), max_limit))):
        item = val_ds[i]
        
        start_time = time.time()
        # 3-Pass Inference
        passes = model.run_modst_passes(item['prompt'], item['image'])
        
        # Tripwire Scoring
        scores = score_engine.compute(passes)
        
        # Action Policy
        decision = policy.act(scores['total'], passes)
        latency = (time.time() - start_time) * 1000 # ms
        
        results.append({
            "id": item['id'],
            "label": item['label'],
            "passes": passes,
            "scores": scores,
            "decision": decision,
            "latency_ms": latency
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
