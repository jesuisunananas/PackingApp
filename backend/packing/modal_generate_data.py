import os
import glob
from pathlib import Path

import modal

APP_NAME = "tetrisbot-data-gen"
PROJECT_DIR = "/root/project"
VOLUME_PATH = "/root/exp"

# Use a persistent volume so we can save the massive dataset directly in the cloud
volume = modal.Volume.from_name("tetrisbot-data-volume", create_if_missing=True)

def load_gitignore_patterns() -> list[str]:
    """Translate .gitignore entries into Modal ignore globs."""
    if not modal.is_local():
        return []

    root = Path(__file__).resolve().parents[0] # Assumes script is in root directory
    gitignore_path = root / ".gitignore"
    if not gitignore_path.is_file():
        return []

    patterns: list[str] = []
    for line in gitignore_path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#") or entry.startswith("!"):
            continue
        entry = entry.lstrip("/")
        if entry.endswith("/"):
            entry = entry.rstrip("/")
            patterns.append(f"**/{entry}/**")
        else:
            patterns.append(f"**/{entry}")
    return patterns

# Build image: use uv_sync() and copy the project directory
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libspatialindex-dev")  # Required for trimesh/rtree raycasting
    .uv_sync()
    .add_local_dir(".", remote_path=PROJECT_DIR, ignore=load_gitignore_patterns())
)

app = modal.App(APP_NAME)

env = {
    "PYTHONPATH": f"{PROJECT_DIR}", # Ensure imports from box, heuristics, etc. work
}

@app.function(image=image, env=env, timeout=1800)
def generate_episode(mesh_paths, bin_dims, items_per_episode, cell_size=0.05):
    import random
    import torch
    import numpy as np
    import os
    from box import MeshBox, Bin
    from heuristics import place_box_with_rule
    
    # CRITICAL FIX: Ensure the temporary directory exists in the fresh cloud container
    os.makedirs("/tmp/packing_meshes", exist_ok=True)
    
    boxes = []
    for _ in range(items_per_episode):
        if mesh_paths:
            path = random.choice(mesh_paths)
            try:
                # Executes the 6-pose raycasting natively inside the Modal worker
                box = MeshBox(path, cell_size=cell_size, fragility=random.uniform(0.1, 1.0))
                boxes.append(box)
            except Exception as e:
                # Failsafe for corrupted .off files
                print(f"Skipping corrupted mesh {path}: {e}")
                pass
                
    # Sort the boxes by true volume descending (the expert packing order)
    expert_boxes = sorted(boxes, key=lambda b: b.volume, reverse=True)
    
    # Initialize the Bin for this episode
    b = Bin(*bin_dims)
    episode_transitions = []
    
    for box in expert_boxes:
        # 1. State: The 2D height map before the object is placed
        # We must use .copy() so we don't just store a reference to the mutating array
        X_heightmap = torch.tensor(b.height_map.copy(), dtype=torch.float32)
        
        # 2. State: The 7D feature vector of the object being placed
        X_features = torch.tensor([
            box.length, box.width, box.height,
            box.volume, box.fragility,
            box.length / bin_dims[0],
            box.width / bin_dims[1]
        ], dtype=torch.float32)
        
        # 3. Apply the heuristic to get the expert placement
        placement = place_box_with_rule(box, b)
        
        if placement is not None:
            # placement from heuristics.py is (x, y, z, rot, pose_idx)
            x, y, z, rot, pose = placement
            
            # Action: The spatial coordinates and rotations chosen by the heuristic
            y_action = torch.tensor([x, y, rot, pose], dtype=torch.long)
            
            episode_transitions.append((X_heightmap, X_features, y_action))
        else:
            # If the expert heuristic fails to fit the box, the episode terminates early
            break
            
    return episode_transitions

# The coordinator function: runs remotely, orchestrates the workers, and writes to the Volume
@app.function(volumes={VOLUME_PATH: volume}, timeout=60 * 60 * 2, env=env, image=image, cpu=4.0)
def coordinator_generation():
    import torch
    
    BIN_DIMS = (30, 30, 50)
    ITEMS_PER_EPISODE = 8
    
    # Paths are now relative to the remote PROJECT_DIR
    train_meshes = glob.glob(os.path.join(PROJECT_DIR, "meshes", "ModelNet10", "*", "train", "*.off"))
    test_meshes = glob.glob(os.path.join(PROJECT_DIR, "meshes", "ModelNet10", "*", "test", "*.off"))
    
    if not train_meshes:
        print("Error: Could not find train meshes in the remote container.")
        return

    print("Distributing 45,000 train episodes across Modal workers...")
    train_args = [(train_meshes, BIN_DIMS, ITEMS_PER_EPISODE) for _ in range(45000)]
    train_data = list(generate_episode.starmap(train_args))
    
    train_save_path = f"{VOLUME_PATH}/bc_dataset_train.pt"
    torch.save(train_data, train_save_path)
    print(f"Saved {len(train_data)} train episodes to {train_save_path}!")
    
    if test_meshes:
        print("Distributing 5,000 test episodes across Modal workers...")
        test_args = [(test_meshes, BIN_DIMS, ITEMS_PER_EPISODE) for _ in range(5000)]
        test_data = list(generate_episode.starmap(test_args))
        
        test_save_path = f"{VOLUME_PATH}/bc_dataset_test.pt"
        torch.save(test_data, test_save_path)
        print(f"Saved {len(test_data)} test episodes to {test_save_path}!")
        
    volume.commit() # Ensure data is persisted

@app.local_entrypoint()
def main():
    print("Initiating remote data generation coordinator...")
    coordinator_generation.remote()
    print("Generation complete! Data is safely stored in the Modal Volume.")