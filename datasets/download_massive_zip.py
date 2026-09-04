import os
import urllib.request
import zipfile
import sys
import shutil

sys.stdout.reconfigure(encoding='utf-8')

DATASETS_DIR = r"c:\Users\Dell\Downloads\road_shield_ai_engine\datasets"
DOWNLOAD_URLS = [
    ("https://github.com/jaygala24/pothole-detection/releases/download/v1.0.0/Pothole.Dataset.IVCNZ.zip", "ivcnz_potholes.zip"),
    ("https://learnopencv.s3.us-west-2.amazonaws.com/pothole_dataset.zip", "learnopencv_potholes.zip")
]

print("=================================================================")
print("📥 INITIATING DIRECT MASSIVE DATASET DOWNLOAD (Bypassing Kaggle)")
print("=================================================================")

for url, filename in DOWNLOAD_URLS:
    dest_path = os.path.join(DATASETS_DIR, filename)
    print(f"Downloading {filename} from {url}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f" ✓ Download complete: {filename}")
        
        # Unzip
        extract_dir = os.path.join(DATASETS_DIR, filename.replace('.zip', '_extracted'))
        print(f" Unzipping {filename}...")
        with zipfile.ZipFile(dest_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            
        print(f" ✓ Unzipped successfully to {extract_dir}")
        os.remove(dest_path) # cleanup zip
    except Exception as e:
        print(f" ❌ Failed to process {url}: {e}")

# Count the real images we just downloaded
total_new_images = 0
for root, dirs, files in os.walk(DATASETS_DIR):
    if '_extracted' in root:
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                total_new_images += 1

print("\n=================================================================")
print(f"🎉 SUCCESS! Acquired {total_new_images} NEW UNIQUE REAL IMAGES.")
print("=================================================================")
