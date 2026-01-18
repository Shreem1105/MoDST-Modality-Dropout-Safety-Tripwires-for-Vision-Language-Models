import os
import requests
import zipfile
import shutil
import io

def download_and_setup_pope():
    # Candidates for the repository zip URL (branch naming)
    urls = [
        "https://github.com/lm-sys/POPE/archive/refs/heads/main.zip",
        "https://github.com/lm-sys/POPE/archive/refs/heads/master.zip"
    ]
    
    target_dir = "data/POPE"
    zip_path = "data/pope.zip"
    
    # 1. Cleanup
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    if os.path.exists(zip_path):
        os.remove(zip_path)
    os.makedirs("data", exist_ok=True)
    
    # 2. Download
    downloaded = False
    for url in urls:
        print(f"Trying to download from: {url}")
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                # Check if it looks like a zip (magic bytes or size)
                if int(response.headers.get('Content-Length', 0)) < 1000 and 'zip' not in response.headers.get('Content-Type', ''):
                    print(f"  -> File too small or wrong type. Probably 404 text.")
                    continue
                    
                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                print("  -> Download successful!")
                downloaded = True
                break
            else:
                print(f"  -> Failed with status code: {response.status_code}")
        except Exception as e:
            print(f"  -> Error: {e}")
            
    if not downloaded:
        print("CRITICAL ERROR: Could not download POPE zip from any known branch.")
        return

    # 3. Unzip
    print("Unzipping...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("data/")
            extracted_names = zip_ref.namelist()
            root_folder = extracted_names[0].split('/')[0]
    except zipfile.BadZipFile:
        print("Error: The downloaded file is not a valid zip.")
        return

    # 4. Rename
    extracted_path = os.path.join("data", root_folder)
    print(f"Extracted to: {extracted_path}")
    
    if os.path.exists(extracted_path):
        shutil.move(extracted_path, target_dir)
        print(f"Moved to: {target_dir}")
        print("✅ POPE Data Setup Complete!")
    else:
        print(f"Error: Could not find extracted folder {extracted_path}")
        
    # Cleanup zip
    os.remove(zip_path)

if __name__ == "__main__":
    download_and_setup_pope()
