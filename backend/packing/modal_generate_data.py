import os
import glob
from pathlib import Path
import modal
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning, module='trimesh.triangles')

APP_NAME = "tetrisbot-data-gen"
VOLUME_NAME = "tetrisbot-data-volume"
PROJECT_DIR = "/root/project"
VOLUME_PATH = "/vol"
DEFAULT_CPU = 4.0
DEFAULT_TIMEOUT_SECONDS = 60 * 60 * 6

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

def load_gitignore_patterns() -> list[str]:
    """Robust gitignore parser adopted from the reference script."""
    if not modal.is_local():
        return []

    root = Path(__file__).resolve().parents[0]
    gitignore_path = root / ".gitignore"

    raw_entries = [
        ".git/",
        ".venv/",
        "__pycache__/",
        ".pytest_cache/",
        ".mypy_cache/",
        ".ruff_cache/",
        "*.pyc",
        "*.zip",
        ".DS_Store",
    ]

    if gitignore_path.is_file():
        for line in gitignore_path.read_text(encoding="utf-8").splitlines():
            entry = line.strip()
            if not entry or entry.startswith("#") or entry.startswith("!"):
                continue
            raw_entries.append(entry)

    patterns: list[str] = []
    for entry in raw_entries:
        entry = entry.lstrip("/")
        if entry.endswith("/"):
            entry = entry.rstrip("/")
            patterns.extend([f"{entry}/**", f"**/{entry}/**"])
        elif "/" in entry:
            patterns.extend([entry, f"{entry}/**"])
        else:
            patterns.extend([entry, f"**/{entry}"])

    patterns.extend([".git/**", "**/.git/**"])
    return sorted(set(patterns))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libspatialindex-dev")
    .uv_sync()
    .add_local_dir(".", remote_path=PROJECT_DIR, ignore=load_gitignore_patterns())
)

app = modal.App(APP_NAME)

env = {
    "PYTHONPATH": PROJECT_DIR,
    "PYTHONUNBUFFERED": "1",
}

NUM_TRAIN_EPISODES = 50000
NUM_TEST_EPISODES = 5000
ITEMS_PER_EPISODE = 30
MIN_EPISODE_LENGTH = 10
CELL_SIZE = 0.05
BIN_DIMS = (30, 30, 50)
CHUNK_SIZE = 5000

@app.function(image=image, env=env, timeout=1800)
def generate_episode(mesh_list, bin_dims, items_per_episode, episode_idx):
    """The worker function: executes on hundreds of isolated CPU containers."""
    import random
    import torch
    import os
    from box import MeshBox, Box, Bin
    import heuristics
    import numpy as np

    def set_seed(seed=10):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

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
                fragility = random.uniform(0.1, 1.0)
                box = MeshBox(mesh_path, cell_size=CELL_SIZE, save_mesh=save, fragility=fragility)
                boxes.append(box)
            except Exception:
                continue
        else:
            L, W, H = random.randint(3, 8), random.randint(3, 8), random.randint(3, 8)
            boxes.append(Box(L, W, H, fragility=random.uniform(0.1, 1.0)))

    for b in boxes:

        vol = b.volume if hasattr(b, 'volume') else b._volume
        b._sorting_weight = (vol * 0.5) + ((1.0 - b.fragility) * 0.5)

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

        compactness = heuristics.compute_compactness(bin_env)
        pyramid_score = heuristics.compute_pyramid(bin_env)
        frag_penalty = heuristics.compute_fragility_penalty(bin_env, base_scaling=1.0, heavy_factor=2.0, fragile_quantile=0.25, alpha=1.0)
        reward = compactness + pyramid_score - frag_penalty

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

@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=DEFAULT_TIMEOUT_SECONDS,
    env=env,
    image=image,
    cpu=DEFAULT_CPU
)
def coordinator_generation():
    """The master orchestrator: maps out the parallel tasks and saves chunked data."""
    import torch

    train_meshes = glob.glob(os.path.join(PROJECT_DIR, "meshes", "ModelNet10", "*", "train", "*.off"))
    test_meshes = glob.glob(os.path.join(PROJECT_DIR, "meshes", "ModelNet10", "*", "test", "*.off"))

    if not train_meshes:
        print("Warning: Could not find train meshes in the remote container. Falling back to random Boxes.")

    existing_chunks = glob.glob(f"{VOLUME_PATH}/bc_dataset_train_chunk_continuous_*.pt")

    if existing_chunks:
        highest_chunk = max([int(f.split('_')[-1].split('.')[0]) for f in existing_chunks])
        train_chunk_idx = highest_chunk + 1
    else:
        train_chunk_idx = 0

    episodes_already_done = train_chunk_idx * CHUNK_SIZE
    episodes_remaining = NUM_TRAIN_EPISODES - episodes_already_done

    train_args = [(train_meshes, BIN_DIMS, ITEMS_PER_EPISODE, i)
                  for i in range(episodes_already_done, NUM_TRAIN_EPISODES)]

    train_data = []
    train_dropped = 0

    for i, ep in enumerate(generate_episode.starmap(train_args)):
        if len(ep) >= MIN_EPISODE_LENGTH:
            train_data.append(ep)
        else:
            train_dropped += 1

        if (i + 1) % 500 == 0:
            print(f"  -> Processed {i + 1}/{NUM_TRAIN_EPISODES} | Dropped: {train_dropped}")

        if len(train_data) >= CHUNK_SIZE:

            flat_transitions = [t for ep in train_data for t in ep]

            chunk_dict = {
                "X_heightmap": torch.stack([t[0] for t in flat_transitions]),
                "X_features": torch.stack([t[1] for t in flat_transitions]),
                "y_action": torch.stack([t[2] for t in flat_transitions]),
                "reward": torch.tensor([t[3] for t in flat_transitions], dtype=torch.float32),
                "next_heightmap": torch.stack([t[4] for t in flat_transitions]),
                "next_features": torch.stack([t[5] for t in flat_transitions]),
                "done": torch.tensor([t[6] for t in flat_transitions], dtype=torch.bool)
            }

            file_name = f"{VOLUME_PATH}/bc_dataset_train_chunk_continuous_{train_chunk_idx}.pt"
            torch.save(chunk_dict, file_name)
            volume.commit()

            train_data = []
            train_chunk_idx += 1

    if len(train_data) > 0:
        flat_transitions = [t for ep in train_data for t in ep]
        chunk_dict = {
            "X_heightmap": torch.stack([t[0] for t in flat_transitions]),
            "X_features": torch.stack([t[1] for t in flat_transitions]),
            "y_action": torch.stack([t[2] for t in flat_transitions]),
            "reward": torch.tensor([t[3] for t in flat_transitions], dtype=torch.float32),
            "next_heightmap": torch.stack([t[4] for t in flat_transitions]),
            "next_features": torch.stack([t[5] for t in flat_transitions]),
            "done": torch.tensor([t[6] for t in flat_transitions], dtype=torch.bool)
        }
        file_name = f"{VOLUME_PATH}/bc_dataset_train_chunk_continuous_{train_chunk_idx}.pt"
        torch.save(chunk_dict, file_name)
        volume.commit()

    test_args = [(test_meshes, BIN_DIMS, ITEMS_PER_EPISODE, i) for i in range(NUM_TEST_EPISODES)]

    test_data = []
    test_dropped = 0
    test_chunk_idx = 0

    for i, ep in enumerate(generate_episode.starmap(test_args)):
        if len(ep) >= MIN_EPISODE_LENGTH:
            test_data.append(ep)
        else:
            test_dropped += 1

        if (i + 1) % 500 == 0:
            print(f"  -> Processed {i + 1}/{NUM_TEST_EPISODES} | Dropped: {test_dropped}")

        if len(test_data) >= CHUNK_SIZE:

            flat_transitions = [t for ep in test_data for t in ep]

            chunk_dict = {
                "X_heightmap": torch.stack([t[0] for t in flat_transitions]),
                "X_features": torch.stack([t[1] for t in flat_transitions]),
                "y_action": torch.stack([t[2] for t in flat_transitions]),
                "reward": torch.tensor([t[3] for t in flat_transitions], dtype=torch.float32),
                "next_heightmap": torch.stack([t[4] for t in flat_transitions]),
                "next_features": torch.stack([t[5] for t in flat_transitions]),
                "done": torch.tensor([t[6] for t in flat_transitions], dtype=torch.bool)
            }
            file_name = f"{VOLUME_PATH}/bc_dataset_test_chunk_continuous_{test_chunk_idx}.pt"
            torch.save(chunk_dict, file_name)
            volume.commit()

            test_data = []
            test_chunk_idx += 1

    if len(test_data) > 0:
        flat_transitions = [t for ep in test_data for t in ep]
        chunk_dict = {
            "X_heightmap": torch.stack([t[0] for t in flat_transitions]),
            "X_features": torch.stack([t[1] for t in flat_transitions]),
            "y_action": torch.stack([t[2] for t in flat_transitions]),
            "reward": torch.tensor([t[3] for t in flat_transitions], dtype=torch.float32),
            "next_heightmap": torch.stack([t[4] for t in flat_transitions]),
            "next_features": torch.stack([t[5] for t in flat_transitions]),
            "done": torch.tensor([t[6] for t in flat_transitions], dtype=torch.bool)
        }
        file_name = f"{VOLUME_PATH}/bc_dataset_test_chunk_continuous_{test_chunk_idx}.pt"
        torch.save(chunk_dict, file_name)
        volume.commit()

@app.local_entrypoint()
def main():
    print("Initiating remote data generation coordinator...")
    coordinator_generation.remote()
    print(f"Generation complete! Data is safely stored in the '{VOLUME_NAME}' Modal Volume.")