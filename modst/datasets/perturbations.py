import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset

class Perturbations:
    @staticmethod
    def gaussian_noise(img: Image.Image, amount=0.1) -> Image.Image:
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, amount * 255, arr.shape)
        return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))

    @staticmethod
    def occlusion(img: Image.Image, size=50) -> Image.Image:
        arr = np.array(img).copy()
        h, w, _ = arr.shape
        y, x = np.random.randint(0, h-size), np.random.randint(0, w-size)
        arr[y:y+size, x:x+size, :] = 0
        return Image.fromarray(arr)

class ShiftedDataset(Dataset):
    """Applies perturbations to a base dataset on the fly."""
    def __init__(self, base_dataset, shift_type='noise', severity=0.1):
        self.base = base_dataset
        self.shift_type = shift_type
        self.severity = severity

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        img = item['image']
        if self.shift_type == 'noise':
            img = Perturbations.gaussian_noise(img, self.severity)
        elif self.shift_type == 'occlusion':
            img = Perturbations.occlusion(img, int(self.severity * 100))
        elif self.shift_type == 'blur':
            img = img.filter(ImageFilter.GaussianBlur(radius=self.severity * 5))
        
        item['image'] = img
        item['is_shifted'] = True
        return item
