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

NUM_TRAIN_EPISODES = 20
NUM_TEST_EPISODES = 5
ITEMS_PER_EPISODE = 20
MIN_EPISODE_LENGTH = 5
CELL_SIZE = 0.05
BIN_DIMS = (30, 30, 50)
CHUNK_SIZE = 10

def visualize_transition(heightmap, action, episode_idx, step_idx, grid_L, grid_W):
    os.makedirs('debug_renders', exist_ok=True)

    norm_x, norm_y, norm_rot, norm_pose = action.tolist()

    plot_x = (norm_x + 1.0) / 2.0 * grid_L
    plot_y = (norm_y + 1.0) / 2.0 * grid_W

    rot = (norm_rot + 1.0) / 2.0 * 3.0
    pose = (norm_pose + 1.0) / 2.0 * 5.0

    plt.figure(figsize=(6, 6))

    if isinstance(heightmap, torch.Tensor):
        heightmap = heightmap.numpy()

    plt.imshow(heightmap, cmap='viridis', origin='lower', vmin=0, vmax=BIN_DIMS[2] * CELL_SIZE)
    plt.colorbar(label='Height (Physical Meters)')

    plt.scatter(plot_x, plot_y, color='red', marker='X', s=100, label=f'Action (x:{plot_x:.1f}, y:{plot_y:.1f})')

    plt.title(f'Episode {episode_idx} | Step {step_idx}\nRot: {rot:.1f}, Pose: {pose:.1f}')
    plt.legend()
    plt.savefig(f'debug_renders/ep_{episode_idx}_step_{step_idx}.png')
    plt.close()

def get_mesh_paths(split="train"):
    search_path = os.path.join("meshes", "ModelNet10", "*", split, "*.off")
    return glob.glob(search_path)

def generate_episode(mesh_list, bin_dims, items_per_episode, episode_idx):
    def set_seed(seed=10):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    set_seed(episode_idx)
    os.makedirs("/tmp/packing_meshes", exist_ok=True)

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

    for b in boxes:
        noise_factor = random.uniform(0.85, 1.15)
        vol = b.volume if hasattr(b, 'volume') else b._volume
        b._sorting_weight = vol * noise_factor

    expert_boxes = sorted(boxes, key=lambda b: b._sorting_weight, reverse=True)

    grid_L, grid_W, grid_H = bin_dims
    bin_env = Bin(grid_L, grid_W, grid_H)
    transitions = []

    for step_idx, b in enumerate(expert_boxes):
        current_heightmap = torch.tensor(bin_env.height_map.copy(), dtype=torch.float32)

        vol = b.volume if hasattr(b, 'volume') else b._volume
        features = torch.tensor([
            b.length, b.width, b.height, vol, b.fragility,
            b.length / grid_L, b.width / grid_W, b.height / grid_H
        ], dtype=torch.float32)

        best_candidate = heuristics.place_box_with_rule(b, bin_env)

        if best_candidate is None:
            action = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
            reward = -1.0
            next_heightmap = current_heightmap
            next_features = torch.zeros_like(features)
            done = True
            transitions.append((current_heightmap, features, action, reward, next_heightmap, next_features, done))
            break

        best_x, best_y, best_z, best_rot, best_pose = best_candidate

        norm_x = (best_x / grid_L) * 2.0 - 1.0
        norm_y = (best_y / grid_W) * 2.0 - 1.0
        norm_rot = (best_rot / 3.0) * 2.0 - 1.0
        norm_pose = (best_pose / 5.0) * 2.0 - 1.0

        action = torch.tensor([norm_x, norm_y, norm_rot, norm_pose], dtype=torch.float32)

        if episode_idx == 0:
            visualize_transition(current_heightmap, action, episode_idx, step_idx, grid_L, grid_W)

        compactness = heuristics.compute_compactness(bin_env)
        pyramid_score = heuristics.compute_pyramid(bin_env)
        reward = compactness + pyramid_score

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
            next_features = torch.zeros_like(features)

        transitions.append((current_heightmap, features, action, reward, next_heightmap, next_features, done))

    return transitions

def generate_episode_wrapper(args):
    return generate_episode(*args)

def save_chunk(data_list, filename):
    flat_transitions = [t for ep in data_list for t in ep]
    chunk_dict = {
        "X_heightmap": torch.stack([t[0] for t in flat_transitions]),
        "X_features": torch.stack([t[1] for t in flat_transitions]),
        "y_action": torch.stack([t[2] for t in flat_transitions]),
        "reward": torch.tensor([t[3] for t in flat_transitions], dtype=torch.float32),
        "next_heightmap": torch.stack([t[4] for t in flat_transitions]),
        "next_features": torch.stack([t[5] for t in flat_transitions]),
        "done": torch.tensor([t[6] for t in flat_transitions], dtype=torch.bool)
    }
    torch.save(chunk_dict, filename)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    train_meshes = get_mesh_paths("train")
    test_meshes = get_mesh_paths("test")

    cpu_cores = mp.cpu_count()
    print(f"Using {cpu_cores} cores for local dataset generation.\n")

    with mp.Pool(cpu_cores) as pool:

        train_args = [(train_meshes, BIN_DIMS, ITEMS_PER_EPISODE, i) for i in range(NUM_TRAIN_EPISODES)]

        train_data = []
        train_dropped = 0
        train_chunk_idx = 0

        for i, ep in enumerate(pool.imap_unordered(generate_episode_wrapper, train_args)):
            if len(ep) >= MIN_EPISODE_LENGTH:
                train_data.append(ep)
            else:
                train_dropped += 1

            if (i + 1) % 10 == 0:
                print(f"  -> Processed {i + 1}/{NUM_TRAIN_EPISODES} | Dropped: {train_dropped}")

            if len(train_data) >= CHUNK_SIZE:
                save_chunk(train_data, f"bc_dataset_train_chunk_continuous_{train_chunk_idx}.pt")
                train_data = []
                train_chunk_idx += 1

        if len(train_data) > 0:
            save_chunk(train_data, f"bc_dataset_train_chunk_continuous_{train_chunk_idx}.pt")

        test_args = [(test_meshes, BIN_DIMS, ITEMS_PER_EPISODE, i) for i in range(NUM_TEST_EPISODES)]

        test_data = []
        test_dropped = 0
        test_chunk_idx = 0

        for i, ep in enumerate(pool.imap_unordered(generate_episode_wrapper, test_args)):
            if len(ep) >= MIN_EPISODE_LENGTH:
                test_data.append(ep)
            else:
                test_dropped += 1

            if (i + 1) % 10 == 0:
                print(f"  -> Processed {i + 1}/{NUM_TEST_EPISODES} | Dropped: {test_dropped}")

            if len(test_data) >= CHUNK_SIZE:
                save_chunk(test_data, f"bc_dataset_test_chunk_continuous_{test_chunk_idx}.pt")
                test_data = []
                test_chunk_idx += 1

        if len(test_data) > 0:
            save_chunk(test_data, f"bc_dataset_test_chunk_continuous_{test_chunk_idx}.pt")