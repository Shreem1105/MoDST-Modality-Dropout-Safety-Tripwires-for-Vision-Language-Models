import os
import json
from PIL import Image

def setup_demo():
    data_dir = "data"
    img_dir = os.path.join(data_dir, "images")
    json_path = os.path.join(data_dir, "val_grounding.json")

    if os.path.exists(json_path):
        print(f"Dataset found at {json_path}. Skipping demo generation.")
        return

    print("Dataset not found. Generating demo data needed to run the pipeline...")
    os.makedirs(img_dir, exist_ok=True)

    # Create a simple dummy image
    img = Image.new('RGB', (224, 224), color='red')
    dummy_img_path = os.path.join(img_dir, 'dummy.jpg')
    img.save(dummy_img_path)

    # Create a dummy JSON dataset
    data = {
        '001': {
            'filename': 'dummy.jpg',
            'question': 'Describe this image.',
            'label': 0
        },
        '002': {
            'filename': 'dummy.jpg',
            'question': 'Is this red?',
            'label': 1
        }
    }
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Created demo dataset at {json_path}")
    print(f"✅ Created dummy image at {dummy_img_path}")

if __name__ == "__main__":
    setup_demo()
