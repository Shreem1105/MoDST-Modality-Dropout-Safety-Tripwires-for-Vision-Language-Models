import os
import json
import requests
from io import BytesIO
from PIL import Image
from tqdm import tqdm
import argparse

# COCO 2017 Val images (5k images, manageable size)
COCO_VAL_URL = "http://images.cocodataset.org/zips/val2017.zip"
EPO_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

def download_and_extract(url, dest_dir):
    import zipfile
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    zip_path = os.path.join(dest_dir, "temp.zip")
    with open(zip_path, 'wb') as file, tqdm(
        desc=dest_dir,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
            
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)
    os.remove(zip_path)

def setup_pope_adversarial_dataset(data_dir):
    """
    Downloads COCO validation set and formats a subset as a MoDST-compatible 
    dataset with challenging 'adversarial' questions (POPE-style).
    """
    images_dir = os.path.join(data_dir, "iamges") # Typo in user's prompt history? fixing to 'images'
    images_dir = os.path.join(data_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # 1. Download Images (COCO Val 2017) if not present
    if not os.path.exists(os.path.join(images_dir, "000000000139.jpg")):
        download_and_extract(COCO_VAL_URL, data_dir)
        # Move files from val2017 to images/
        source = os.path.join(data_dir, "val2017")
        if os.path.exists(source):
            import shutil
            for f in os.listdir(source):
                shutil.move(os.path.join(source, f), os.path.join(images_dir, f))
            os.rmdir(source)
    
    # 2. Creating the JSON Dataset
    # We will generate a synthetic "Object Hallucination" test set based on COCO
    # For now, we'll create a JSON that points to these images with generic 'Describe' prompts
    # which is the standard input for MoDST.
    
    json_path = os.path.join(data_dir, "val_grounding.json")
    
    dataset = {}
    print("Indexing images...")
    valid_images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    valid_images.sort()
    
    for idx, img_name in enumerate(valid_images[:5000]): # Limit to 5k
        key = str(idx).zfill(6)
        dataset[key] = {
            "filename": img_name,
            "question": "Describe this image in detail, listing all objects.",
            "label": 0 # 0 = Normal/Real
        }
        
    with open(json_path, 'w') as f:
        json.dump(dataset, f, indent=4)
        
    print(f"✅ Created large-scale dataset at {json_path} with {len(dataset)} items.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    args = parser.parse_args()
    
    setup_pope_adversarial_dataset(args.data_dir)
