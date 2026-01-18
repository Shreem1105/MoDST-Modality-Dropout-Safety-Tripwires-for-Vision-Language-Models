import os
import requests
import zipfile
import shutil
from tqdm import tqdm

def download_coco_2014():
    url = "http://images.cocodataset.org/zips/val2014.zip"
    data_dir = "data"
    images_dir = os.path.join(data_dir, "images")
    zip_path = os.path.join(data_dir, "val2014.zip")
    
    os.makedirs(images_dir, exist_ok=True)
    
    # 1. Download
    print(f"Downloading COCO 2014 Validation Images from {url}...")
    try:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        if os.path.exists(zip_path):
            # Simple resume check
            if os.path.getsize(zip_path) == total_size:
                print("  -> Zip already exists and matches size. Skipping download.")
            else:
                print("  -> Zip exists but size mismatch. Redownloading.")
                os.remove(zip_path)
        
        if not os.path.exists(zip_path):
            with open(zip_path, 'wb') as file, tqdm(
                desc=zip_path,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    bar.update(size)
    except Exception as e:
        print(f"Download Error: {e}")
        return

    # 2. Extract
    print("Extracting...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
    except Exception as e:
        print(f"Extraction Error: {e}")
        return

    # 3. Organize
    # Extracted folder is 'val2014'. We need to move images to 'data/images'
    source_dir = os.path.join(data_dir, "val2014")
    if os.path.exists(source_dir):
        print(f"Moving images from {source_dir} to {images_dir}...")
        for f in os.listdir(source_dir):
            if f.lower().endswith('.jpg'):
                src = os.path.join(source_dir, f)
                dst = os.path.join(images_dir, f)
                if not os.path.exists(dst):
                    shutil.move(src, dst)
        
        # Cleanup source dir if empty or just remove it
        shutil.rmtree(source_dir)
        print("Cleanup complete.")
    
    # Cleanup zip
    if os.path.exists(zip_path):
        os.remove(zip_path)
        
    print(f"✅ COCO 2014 Setup Complete. Images are in {images_dir}")

if __name__ == "__main__":
    download_coco_2014()
