import json
import os
import glob

class PopeLoader:
    """
    Loader for the POPE (Polling Object Probing Evaluation) dataset.
    Normalizes various POPE formats into a standard iterable.
    """
    def __init__(self, dataset_path, images_root):
        self.dataset_path = dataset_path
        self.images_root = images_root
        self.data = self._load_data()
        
    def _load_data(self):
        # Handle directory of JSONs or single JSON
        if os.path.isdir(self.dataset_path):
            files = glob.glob(os.path.join(self.dataset_path, "*.json"))
        else:
            files = [self.dataset_path]
            
        normalized_data = []
        
        for fpath in files:
            content = []
            with open(fpath, 'r') as f:
                try:
                    # Try loading as standard JSON (list of dicts)
                    content = json.load(f)
                except json.JSONDecodeError:
                    # Fallback: Try loading as JSONL (line-separated dicts)
                    f.seek(0)
                    for line in f:
                        line = line.strip()
                        if line:
                            content.append(json.loads(line))
                
            # POPE often comes as a list of items
            # Expected formats:
            # 1. { "image": "...", "text": "...", "label": ... }
            # 2. { "question_id": ..., "image": ..., "text": ..., "label": ... }
            
            for idx, item in enumerate(content):
                # Resolve Image Path
                img_name = item.get('image')
                if not img_name:
                    continue # Skip if no image
                    
                abs_img_path = os.path.join(self.images_root, img_name)
                
                # Resolve Question
                question = item.get('text') or item.get('question') or item.get('query')
                
                # Resolve Label
                # POPE labels can be "yes"/"no" or 0/1 (where usually 'no' doesn't mean hallucination in POPE, 
                # but we need to map to "Hallucination Positive" vs "Negative").
                # Standard POPE: 'yes' = object present, 'no' = object absent.
                # MoDST Context: Label 1 = Hallucination (Failure), Label 0 = Grounded (Success).
                # If question is "Is there a [obj]?", and ground truth is "no", and model says "yes" -> Hallucination.
                # However, the dataset just gives the ground truth. 
                # We store the GROUND TRUTH label here.
                # Evaluation script will compare Model Answer vs Ground Truth.
                
                raw_label = item.get('label')
                # For output consistency, we keep raw_label but also try to normalize
                label_str = str(raw_label).lower()
                
                # Create ID
                item_id = str(item.get('question_id', f"{os.path.basename(fpath)}_{idx}"))
                
                normalized_data.append({
                    "id": item_id,
                    "image_path": abs_img_path,
                    "question": question,
                    "label": raw_label, # Store ground truth
                    "meta": item
                })
                
        print(f"[PopeLoader] Loaded {len(normalized_data)} items from {self.dataset_path}")
        return normalized_data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
