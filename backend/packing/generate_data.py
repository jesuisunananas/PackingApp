import os
import glob
import random
import torch
import numpy as np
import trimesh
import multiprocessing as mp
from box import MeshBox, Box, Bin

# Generate 50 episodes of 8 items each to keep runtime < 2 mins for the demo
# NUM_EPISODES = 50
ITEMS_PER_EPISODE = 8
CELL_SIZE = 0.05
BIN_DIMS = (30, 30, 50) # 1.5m x 1.5m x 2.5m

def get_mesh_paths(split="train"):
    # Target: meshes/ModelNet10/<category>/<split>/*.off
    search_path = os.path.join("meshes", "ModelNet10", "*", split, "*.off")
    return glob.glob(search_path)

def generate_episode(mesh_list, bin_dims, items_per_episode):
    boxes = []
    for _ in range(items_per_episode):
        if mesh_list:
            mesh_path = random.choice(mesh_list)
            try:
                # Load quickly without raycasting Heightmaps
                mesh = trimesh.load(mesh_path, force='mesh')
                bounds = mesh.bounds
                size = bounds[1] - bounds[0]
                
                L = int(np.ceil(size[0] / CELL_SIZE))
                W = int(np.ceil(size[1] / CELL_SIZE))
                H = size[2]
                vol = float(mesh.volume) if mesh.is_watertight else float(size[0]*size[1]*size[2])
                
                fragility = random.uniform(0.1, 1.0)
                box = Box(L, W, H, fragility=fragility)
                
                # override generic volume with mesh volume
                box._volume = vol
                boxes.append(box)
            except Exception:
                # Fallback to random cuboid if the mesh is corrupted or fails to load
                L = random.randint(3, 8)
                W = random.randint(3, 8)
                H = random.randint(3, 8)
                frag = random.uniform(0.1, 1.0)
                boxes.append(Box(L, W, H, fragility=frag))
        else:
            L = random.randint(3, 8)
            W = random.randint(3, 8)
            H = random.randint(3, 8)
            frag = random.uniform(0.1, 1.0)
            boxes.append(Box(L, W, H, fragility=frag))
    
    # Build state tensor (N, 7)
    X = []
    for b in boxes:
        X.append([
            b.length, b.width, b.height,
            b.volume,
            b.fragility,
            b.length / bin_dims[0],
            b.width / bin_dims[1]
        ])
        
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    # Determine expert ordering: Volume descending
    expert_indices = sorted(range(len(boxes)), key=lambda i: boxes[i].volume, reverse=True)
    y_tensor = torch.tensor(expert_indices, dtype=torch.long)
    
    return X_tensor, y_tensor

if __name__ == "__main__":
    train_meshes = get_mesh_paths("train")
    test_meshes = get_mesh_paths("test")
    
    if not train_meshes:
        print("Warning: No train meshes found. Falling back to random cuboids.")
    if not test_meshes:
        print("Warning: No test meshes found. Falling back to random cuboids.")

    cpu_cores = mp.cpu_count()
    print(f"Using {cpu_cores} cores for generation.")

    with mp.Pool(cpu_cores) as pool:
        print("Generating 45,000 train sequences...")
        train_args = [(train_meshes, BIN_DIMS, ITEMS_PER_EPISODE) for _ in range(45000)]
        train_data = pool.starmap(generate_episode, train_args)
        torch.save(train_data, "bc_dataset_train.pt")
        print("Saved offline BC dataset to bc_dataset_train.pt!")
        
        print("Generating 5,000 test sequences...")
        test_args = [(test_meshes, BIN_DIMS, ITEMS_PER_EPISODE) for _ in range(5000)]
        test_data = pool.starmap(generate_episode, test_args)
        torch.save(test_data, "bc_dataset_test.pt")
        print("Saved offline BC dataset to bc_dataset_test.pt!")

# train_meshes = get_mesh_paths("train")
# test_meshes = get_mesh_paths("test")

# def generate_dataset():
#     meshes_dir = "meshes/train"
#     if not os.path.exists(meshes_dir):
#         # Fallback to general meshes if they didn't run download script
#         meshes_dir = "meshes"
        
#     obj_files = glob.glob(os.path.join(meshes_dir, '*.obj'))
#     obj_files += glob.glob(os.path.join(meshes_dir, '*.off'))
    
#     if not obj_files:
#         print("No meshes found. generating random cuboids instead.")
#         obj_files = []

#     dataset = []
    
#     print(f"Generating {NUM_EPISODES} offline experiences...")
#     for ep in range(NUM_EPISODES):
#         boxes = []
#         for _ in range(ITEMS_PER_EPISODE):
#             if obj_files:
#                 mesh_path = random.choice(obj_files)
#                 # Load quickly without raycasting Heightmaps
#                 mesh = trimesh.load(mesh_path, force='mesh')
#                 bounds = mesh.bounds
#                 size = bounds[1] - bounds[0]
                
#                 L = int(np.ceil(size[0] / CELL_SIZE))
#                 W = int(np.ceil(size[1] / CELL_SIZE))
#                 H = size[2]
#                 vol = float(mesh.volume) if mesh.is_watertight else float(size[0]*size[1]*size[2])
                
#                 fragility = random.uniform(0.1, 1.0)
#                 box = Box(L, W, H, fragility=fragility)
#                 # override generic volume with mesh volume
#                 box._volume = vol
#                 boxes.append(box)
#             else:
#                 L = random.randint(3, 8)
#                 W = random.randint(3, 8)
#                 H = random.randint(3, 8)
#                 frag = random.uniform(0.1, 1.0)
#                 boxes.append(Box(L, W, H, fragility=frag))
        
#         # Build state tensor (N, 7)
#         X = []
#         for b in boxes:
#             X.append([
#                 b.length, b.width, b.height,
#                 b.volume,
#                 b.fragility,
#                 b.length / BIN_DIMS[0],
#                 b.width / BIN_DIMS[1]
#             ])
            
#         X_tensor = torch.tensor(X, dtype=torch.float32)
        
#         # Determine expert ordering: Volume descending
#         # sort boxes by volume (b.volume)
#         expert_indices = sorted(range(len(boxes)), key=lambda i: boxes[i].volume, reverse=True)
#         y_tensor = torch.tensor(expert_indices, dtype=torch.long)
        
#         dataset.append((X_tensor, y_tensor))
        
#         if (ep + 1) % 5 == 0:
#             print(f"  Generated {ep + 1}/{NUM_EPISODES}")
            
#     torch.save(dataset, "bc_dataset.pt")
#     print("Saved offline BC dataset to bc_dataset.pt!")

# if __name__ == "__main__":
#     generate_dataset()
