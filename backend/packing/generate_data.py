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

# --- PRODUCTION RUN PARAMETERS ---
NUM_TRAIN_EPISODES = 8000#45000
NUM_TEST_EPISODES = 1000#5000
ITEMS_PER_EPISODE = 20#8
MIN_EPISODE_LENGTH = 5#3  # Drop episodes that fail to pack at least 3 items
CELL_SIZE = 0.05
BIN_DIMS = (30, 30, 50) # L, W, H

def get_mesh_paths(split="train"):
    search_path = os.path.join("meshes", "ModelNet10", "*", split, "*.off")
    return glob.glob(search_path)

def visualize_transition(heightmap, action, episode_idx, step_idx):
    os.makedirs('debug_renders', exist_ok=True)
    x, y, rot, pose = action
    
    plt.figure(figsize=(6, 6))
    plt.imshow(heightmap, cmap='viridis', origin='lower', vmin=0, vmax=BIN_DIMS[2] * CELL_SIZE)
    plt.colorbar(label='Height (Physical Meters)')
    
    plt.scatter(x, y, color='red', marker='X', s=100, label=f'Action (x:{x}, y:{y})')
    
    plt.title(f'Episode {episode_idx} | Step {step_idx}\nRot: {rot}, Pose: {pose}')
    plt.legend()
    plt.savefig(f'debug_renders/ep_{episode_idx}_step_{step_idx}.png')
    plt.close()

def generate_episode(mesh_list, bin_dims, items_per_episode, episode_idx):
    boxes = []
    
    for _ in range(items_per_episode):
        if mesh_list:
            mesh_path = random.choice(mesh_list)
            file_size_mb = os.path.getsize(mesh_path) / (1024 * 1024)
            if file_size_mb > 15.0:
                continue 
                
            try:
                save = (episode_idx == 0)
                box = MeshBox(mesh_path, cell_size=CELL_SIZE, save_mesh=save)
                boxes.append(box)
            except Exception:
                continue 
        else:
            L, W, H = random.randint(3, 8), random.randint(3, 8), random.randint(3, 8)
            boxes.append(Box(L, W, H, fragility=random.uniform(0.1, 1.0)))
    
    expert_boxes = sorted(boxes, key=lambda b: b.volume, reverse=True)
    
    grid_L, grid_W, grid_H = bin_dims
    bin_env = Bin(grid_L, grid_W, grid_H)
    transitions = []
    
    for step_idx, b in enumerate(expert_boxes):
        current_heightmap = torch.tensor(bin_env.height_map.copy(), dtype=torch.float32)
        
        # Current Object Features
        vol = b.volume if hasattr(b, 'volume') else b._volume
        features = torch.tensor([
            b.length, b.width, b.height, vol, b.fragility,
            b.length / grid_L, b.width / grid_W, b.height / grid_H 
        ], dtype=torch.float32)
        
        best_candidate = heuristics.place_box_with_rule(b, bin_env)
        
        if best_candidate is None:
            # If the heuristic fails to place it, we give a penalty and end the episode
            action = torch.tensor([0, 0, 0, 0], dtype=torch.long) # Dummy action
            reward = -1.0 
            next_heightmap = current_heightmap # State didn't change
            next_features = torch.zeros_like(features) # No next object
            done = True
            transitions.append((current_heightmap, features, action, reward, next_heightmap, next_features, done))
            break 
            
        best_x, best_y, best_z, best_rot, best_pose = best_candidate
        action = torch.tensor([best_x, best_y, best_rot, best_pose], dtype=torch.long)
        
        # --- NEW: CALCULATE REWARD ---
        # Use your heuristics to define how good this placement was
        compactness = heuristics.compute_compactness(bin_env)
        pyramid_score = heuristics.compute_pyramid(bin_env)
        reward = compactness + pyramid_score 
        
        # --- NEW: GET NEXT STATE ---
        done = (step_idx == len(expert_boxes) - 1)
        next_heightmap = torch.tensor(bin_env.height_map.copy(), dtype=torch.float32)
        
        if not done:
            next_b = expert_boxes[step_idx + 1]
            next_vol = next_b.volume if hasattr(next_b, 'volume') else next_b._volume
            next_features = torch.tensor([
                next_b.length, next_b.width, next_b.height, next_vol, next_b.fragility,
                next_b.length / grid_L, next_b.width / grid_W, next_b.height / grid_H
            ], dtype=torch.float32)
        else:
            next_features = torch.zeros_like(features) # Empty features if episode is done
            
        # APPEND FULL TUPLE
        transitions.append((current_heightmap, features, action, reward, next_heightmap, next_features, done))

    return transitions

# Wrapper function required for pool.imap_unordered
def generate_episode_wrapper(args):
    return generate_episode(*args)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    train_meshes = get_mesh_paths("train")
    test_meshes = get_mesh_paths("test")
    
    cpu_cores = mp.cpu_count()
    print(f"Using {cpu_cores} cores for full dataset generation.\n")

    # --- CHUNK SETTINGS ---
    CHUNK_SIZE = 1000 # Save a file every 1000 successful episodes

    with mp.Pool(cpu_cores) as pool:
        # --- TRAIN GENERATION ---
        print(f"--- Generating {NUM_TRAIN_EPISODES} Train Sequences ---")
        train_args = [(train_meshes, BIN_DIMS, ITEMS_PER_EPISODE, i) for i in range(NUM_TRAIN_EPISODES)]
        
        train_data = []
        train_dropped = 0
        train_chunk_idx = 0
        
        for i, ep in enumerate(pool.imap_unordered(generate_episode_wrapper, train_args)):
            if len(ep) >= MIN_EPISODE_LENGTH:
                train_data.append(ep)
            else:
                train_dropped += 1
                
            if (i + 1) % 500 == 0:
                print(f"  -> Processed {i + 1}/{NUM_TRAIN_EPISODES} | Dropped: {train_dropped}")
                
            # --- SAVE CHECKPOINT ---
            if len(train_data) >= CHUNK_SIZE:
                file_name = f"bc_dataset_train_chunk_{train_chunk_idx}.pt"
                torch.save(train_data, file_name)
                print(f"💾 CHECKPOINT: Saved {file_name}!")
                
                train_data = [] # Clear memory!
                train_chunk_idx += 1
                
        # Save whatever is left over at the end
        if len(train_data) > 0:
            torch.save(train_data, f"bc_dataset_train_chunk_{train_chunk_idx}.pt")
            print("💾 CHECKPOINT: Saved final train chunk!\n")
            
            
        # --- TEST GENERATION ---
        print(f"--- Generating {NUM_TEST_EPISODES} Test Sequences ---")
        test_args = [(test_meshes, BIN_DIMS, ITEMS_PER_EPISODE, i) for i in range(NUM_TEST_EPISODES)]
        
        test_data = []
        test_dropped = 0
        test_chunk_idx = 0
        
        for i, ep in enumerate(pool.imap_unordered(generate_episode_wrapper, test_args)):
            if len(ep) >= MIN_EPISODE_LENGTH:
                test_data.append(ep)
            else:
                test_dropped += 1
                
            if (i + 1) % 500 == 0:
                print(f"  -> Processed {i + 1}/{NUM_TEST_EPISODES} | Dropped: {test_dropped}")
                
            # --- SAVE CHECKPOINT ---
            if len(test_data) >= CHUNK_SIZE:
                file_name = f"bc_dataset_test_chunk_{test_chunk_idx}.pt"
                torch.save(test_data, file_name)
                print(f"💾 CHECKPOINT: Saved {file_name}!")
                
                test_data = []
                test_chunk_idx += 1

        if len(test_data) > 0:
            torch.save(test_data, f"bc_dataset_test_chunk_{test_chunk_idx}.pt")
            print("💾 CHECKPOINT: Saved final test chunk!\n")