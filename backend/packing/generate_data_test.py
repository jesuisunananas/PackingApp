import os
import glob
import random
import torch
import numpy as np
import trimesh
import multiprocessing as mp
import matplotlib.pyplot as plt
from box import MeshBox, Box, Bin
import heuristics
from main import visualize_bin_pybullet

# --- TEST RUN PARAMETERS ---
NUM_TRAIN_EPISODES = 10  # Shrunk for testing
NUM_TEST_EPISODES = 2    # Shrunk for testing
ITEMS_PER_EPISODE = 8
CELL_SIZE = 0.05
BIN_DIMS = (30, 30, 50) # L, W, H

def get_mesh_paths(split="train"):
    search_path = os.path.join("meshes", "ModelNet10", "*", split, "*.off")
    return glob.glob(search_path)

def visualize_transition(heightmap, action, episode_idx, step_idx):
    """Saves a 2D plot of the heightmap and the expert's chosen action."""
    os.makedirs('debug_renders', exist_ok=True)
    x, y, rot, pose = action
    
    plt.figure(figsize=(6, 6))
    plt.imshow(heightmap, cmap='viridis', origin='lower', vmin=0, vmax=BIN_DIMS[2])
    plt.colorbar(label='Height (Cells)')
    
    # Draw red X at the placement coordinate
    plt.scatter(x, y, color='red', marker='X', s=100, label=f'Action (x:{x}, y:{y})')
    
    plt.title(f'Episode {episode_idx} | Step {step_idx}\nRot: {rot}, Pose: {pose}')
    plt.legend()
    plt.savefig(f'debug_renders/ep_{episode_idx}_step_{step_idx}.png')
    plt.close()

def generate_episode(mesh_list, bin_dims, items_per_episode, episode_idx):
    boxes = []
    # 1. Generate Boxes using MeshBox
    for _ in range(items_per_episode):
        if mesh_list:
            mesh_path = random.choice(mesh_list)
            file_size_mb = os.path.getsize(mesh_path) / (1024 * 1024)
            if file_size_mb > 15.0:
                print(f"Skipping {mesh_path}: File too large ({file_size_mb:.1f} MB)")
                continue
            try:
                # Instantiate YOUR MeshBox instead of a generic Box
                box = MeshBox(mesh_path, cell_size=CELL_SIZE)
                boxes.append(box)
            except Exception as e:
                # If a mesh fails to raycast, skip it or add a fallback
                print(f"FAILED to load mesh {mesh_path}: {e}")
                continue 
        else:
            # Fallback if no meshes exist at all
            L, W, H = random.randint(3, 8), random.randint(3, 8), random.randint(3, 8)
            boxes.append(Box(L, W, H, fragility=random.uniform(0.1, 1.0)))
    
    # Sort boxes by volume descending (Expert Strategy)
    expert_boxes = sorted(boxes, key=lambda b: b.volume, reverse=True)
    
    # 2. Simulate the Bin (Spatial State)
    grid_L, grid_W, grid_H = bin_dims
    bin_env = Bin(grid_L, grid_W, grid_H)
    transitions = []
    
    # 3. Generate Sequential Transitions
    for step_idx, b in enumerate(expert_boxes):
        # A. Record Current State (BEFORE placement)
        # We use .copy() so we don't accidentally save a reference that changes later
        current_heightmap = torch.tensor(bin_env.height_map.copy(), dtype=torch.float32)
        
        vol = b.volume if hasattr(b, 'volume') else b._volume
        features = torch.tensor([
            b.length, b.width, b.height,
            vol,
            b.fragility,
            b.length / grid_L,
            b.width / grid_W
        ], dtype=torch.float32)
        
        # B & E. Calculate Expert Action & Update State
        # place_box_with_rule handles finding the optimal spot AND updating bin_env.height_map
        best_candidate = heuristics.place_box_with_rule(b, bin_env)
        
        if best_candidate is None:
            # The heuristic couldn't fit the box in the bin!
            # Terminate this episode early, saving only the successful transitions
            break
            
        best_x, best_y, best_z, best_rot, best_pose = best_candidate
        action = torch.tensor([best_x, best_y, best_rot, best_pose], dtype=torch.long)
        
        # C. Save Transition
        transitions.append((current_heightmap, features, action))
        
        # D. Visualize (Optional)
        if episode_idx == 0:
            print("\nOpening 3D PyBullet Visualizer! (Close the window to continue generation)")
            visualize_bin_pybullet(bin_env, cell_size=CELL_SIZE, gui=True)
            visualize_transition(current_heightmap.numpy(), [best_x, best_y, best_rot, best_pose], episode_idx, step_idx)

    return transitions

if __name__ == "__main__":
    train_meshes = get_mesh_paths("train")
    test_meshes = get_mesh_paths("test")
    
    cpu_cores = 1
    print(f"Using {cpu_cores} cores for small test generation.")

    with mp.Pool(cpu_cores) as pool:
        print(f"Generating {NUM_TRAIN_EPISODES} train sequences...")
        train_args = [(train_meshes, BIN_DIMS, ITEMS_PER_EPISODE, i) for i in range(NUM_TRAIN_EPISODES)]
        train_data = pool.starmap(generate_episode, train_args)
        torch.save(train_data, "bc_dataset_train.pt")
        
        print(f"Generating {NUM_TEST_EPISODES} test sequences...")
        test_args = [(test_meshes, BIN_DIMS, ITEMS_PER_EPISODE, i) for i in range(NUM_TEST_EPISODES)]
        test_data = pool.starmap(generate_episode, test_args)
        torch.save(test_data, "bc_dataset_test.pt")
        
    print("Saved test datasets! Check the 'debug_renders' folder for your visuals.")