import os
import json
from PIL import Image
from torch.utils.data import Dataset
from typing import List, Optional

class BaseVLMDataset(Dataset):
    """Simple JSON-based dataset loader for VLMs."""
    def __init__(self, json_path: str, img_dir: str):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.img_dir = img_dir
        self.keys = list(self.data.keys())

    def __len__(self): return len(self.keys)

    def __getitem__(self, idx):
        item = self.data[self.keys[idx]]
        img_path = os.path.join(self.img_dir, item['filename'])
        image = Image.open(img_path).convert("RGB")
        return {
            "id": self.keys[idx],
            "image": image,
            "prompt": item['question'] if 'question' in item else item['caption'],
            "label": item.get('label', None)
        }
