import os
import urllib.request
import zipfile
import glob
import random
import shutil

DOWNLOAD_URL = "https://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"
ZIP_PATH = "ModelNet10.zip"
EXTRACT_DIR = "ModelNet10"

TRAIN_DIR = "meshes/train"
TEST_DIR = "meshes/test"

def download_and_extract():
    if not os.path.exists(ZIP_PATH):
        print(f"Downloading {DOWNLOAD_URL} (this might take a minute, it's ~450MB)...")
        urllib.request.urlretrieve(DOWNLOAD_URL, ZIP_PATH)
        print("Download complete!")
    
    if not os.path.exists(EXTRACT_DIR):
        print("Extracting ZIP file...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(".")
        print("Extraction complete!")

def gather_meshes(num_train=50, num_test=30):
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    
    print("Gathering meshes from ModelNet10...")
    # ModelNet10 structure is ModelNet10/<category>/train/*.off and ModelNet10/<category>/test/*.off
    all_train_files = glob.glob(f"{EXTRACT_DIR}/*/train/*.off")
    all_test_files = glob.glob(f"{EXTRACT_DIR}/*/test/*.off")
    
    if len(all_train_files) < num_train or len(all_test_files) < num_test:
        raise ValueError("Not enough files found in the dataset!")
        
    random.shuffle(all_train_files)
    random.shuffle(all_test_files)
    
    selected_train = all_train_files[:num_train]
    selected_test = all_test_files[:num_test]
    
    print(f"Copying {num_train} meshes to {TRAIN_DIR}...")
    for f in selected_train:
        # e.g. ModelNet10/chair/train/chair_0001.off
        basename = os.path.basename(f)
        cat = f.split('/')[1]
        new_name = f"{cat}_{basename}"
        shutil.copy(f, os.path.join(TRAIN_DIR, new_name))
        
    print(f"Copying {num_test} meshes to {TEST_DIR}...")
    for f in selected_test:
        basename = os.path.basename(f)
        cat = f.split('/')[1]
        new_name = f"{cat}_{basename}"
        shutil.copy(f, os.path.join(TEST_DIR, new_name))
        
    print("Dataset construction complete!")
    print(f"Train meshes saved to: {os.path.abspath(TRAIN_DIR)}")
    print(f"Test meshes saved to: {os.path.abspath(TEST_DIR)}")

if __name__ == "__main__":
    download_and_extract()
    gather_meshes(num_train=50, num_test=30)
