import os
import glob
import random
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import modal

APP_NAME = "tetrisbot-train-online-sac"
VOLUME_NAME = "tetrisbot-data-volume"
PROJECT_DIR = "/root/project"
VOLUME_PATH = "/vol"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

def load_gitignore_patterns() -> list[str]:
    if not modal.is_local():
        return []
    root = Path(__file__).resolve().parents[0]
    gitignore_path = root / ".gitignore"
    raw_entries = [".git/", ".venv/", "__pycache__/", "*.pyc", "*.zip", ".DS_Store"]
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
            patterns.extend([f"{entry.rstrip('/')}/**", f"**/{entry.rstrip('/')}/**"])
        elif "/" in entry:
            patterns.extend([entry, f"{entry}/**"])
        else:
            patterns.extend([entry, f"**/{entry}"])
    patterns.extend([".git/**", "**/.git/**"])
    return sorted(set(patterns))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libspatialindex-dev")

    .pip_install("pybullet", "trimesh", "torch", "numpy", "rtree", "matplotlib")
    .add_local_dir(".", remote_path=PROJECT_DIR, ignore=load_gitignore_patterns())
)

app = modal.App(APP_NAME)
env = {
    "PYTHONPATH": PROJECT_DIR,
    "PYTHONUNBUFFERED": "1"
}

def set_seed(seed=10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
GAMMA = 0.95
TAU = 0.005
ALPHA = 0.5
BATCH_SIZE = 64
MEMORY_SIZE = 100000

@app.function(
    volumes={VOLUME_PATH: volume},
    image=image,
    env=env,
    gpu="A10G",
    timeout=60 * 60 * 24
)
def train_online_sac_remote():
    import matplotlib.pyplot as plt
    from model_continuous import MDNSpatialActor, TwinContinuousCritic
    from box import MeshBox
    from pybullet_env import PyBulletPackingEnv, BIN_L, BIN_W, CELL_SIZE

    set_seed(10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting SAC Online Training on {device}...")

    actor = MDNSpatialActor(feature_dim=8, map_size=30, action_dim=4, num_mixtures=5).to(device)
    critic = TwinContinuousCritic(feature_dim=8, map_size=30, action_dim=4).to(device)
    target_critic = TwinContinuousCritic(feature_dim=8, map_size=30, action_dim=4).to(device)

    bc_model_path = f"{VOLUME_PATH}/actor_bc_continuous.pt"
    if os.path.exists(bc_model_path):
        print("Loading Warm-Started BC Actor Weights...")
        actor.load_state_dict(torch.load(bc_model_path, map_location=device))
    else:
        print("Warning: BC Actor not found. Starting from scratch.")

    target_critic.load_state_dict(critic.state_dict())

    actor_optim = optim.Adam(actor.parameters(), lr=LR_ACTOR)
    critic_optim = optim.Adam(critic.parameters(), lr=LR_CRITIC)

    target_entropy = -1.0
    log_alpha = torch.tensor([-2.0], requires_grad=True, device=device)
    alpha_optim = optim.Adam([log_alpha], lr=3e-4)

    scaler_path = f"{VOLUME_PATH}/feature_scaler.pt"
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Missing {scaler_path}. Run modal_train_bc.py first!")
    scaler_dict = torch.load(scaler_path, map_location=device, weights_only=True)
    global_mean = scaler_dict["mean"].to(device)
    global_std = scaler_dict["std"].to(device)

    replay_buffer = deque(maxlen=MEMORY_SIZE)
    print("\nPre-filling Replay Buffer with Offline BC chunks...")
    chunk_files = glob.glob(f"{VOLUME_PATH}/bc_dataset_train_chunk_continuous_*.pt")

    loaded_transitions = 0
    selected_chunks = random.sample(chunk_files, min(5, len(chunk_files)))
    for f in selected_chunks:
        chunk = torch.load(f, map_location='cpu', weights_only=False)
        for i in range(len(chunk["reward"])):
            replay_buffer.append((
                chunk["X_heightmap"][i],
                chunk["X_features"][i],
                chunk["y_action"][i],
                chunk["reward"][i].clone().detach().float(),
                chunk["next_heightmap"][i],
                chunk["next_features"][i],
                chunk["done"][i].clone().detach().float()
            ))
            loaded_transitions += 1
    print(f"Buffer pre-filled with {loaded_transitions} high-quality expert transitions.")

    env = PyBulletPackingEnv(render=False)

    def get_random_boxes(num_boxes):
        test_meshes = glob.glob(os.path.join(PROJECT_DIR, "meshes", "ModelNet10", "*", "test", "*.off"))
        boxes = []
        while len(boxes) < num_boxes:
            mesh_path = random.choice(test_meshes)
            file_size_mb = os.path.getsize(mesh_path) / (1024 * 1024)
            if file_size_mb > 15.0:
                continue

            box = MeshBox(mesh_path, cell_size=CELL_SIZE)
            box.fragility = random.uniform(0.1, 1.0)
            boxes.append(box)

        for b in boxes:
            vol = b.volume if hasattr(b, 'volume') else b._volume
            b._sorting_weight = (vol * 0.5) + ((1.0 - b.fragility) * 0.5)

        expert_boxes = sorted(boxes, key=lambda b: b._sorting_weight, reverse=True)
        return expert_boxes

    episode_rewards = []
    moving_averages = []

    global_step = 0

    for episode in range(1, 1001):
        boxes = get_random_boxes(10)
        h, f = env.reset(boxes)
        episode_reward = 0
        done = False

        while not done:
            h_tensor = h.unsqueeze(0).to(device)
            f_tensor = f.unsqueeze(0).to(device)
            f_tensor = (f_tensor - global_mean) / global_std

            with torch.no_grad():
                if global_step < 2000:
                    pi, mu, sigma = actor(h_tensor, f_tensor)
                    best_mixture_idx = torch.argmax(pi, dim=-1)
                    B = pi.shape[0]
                    batch_indices = torch.arange(B)
                    action_tensor = mu[batch_indices, best_mixture_idx, :]
                else:
                    action_tensor, _ = actor.sample_action(h_tensor, f_tensor)
                action_array = action_tensor.squeeze(0).cpu().numpy()

            norm_x, norm_y, norm_rot, norm_pose = action_array
            x_idx = int(np.clip(((norm_x + 1.0) / 2.0) * BIN_L, 0, BIN_L - 1))
            y_idx = int(np.clip(((norm_y + 1.0) / 2.0) * BIN_W, 0, BIN_W - 1))
            rot_idx = int(np.clip(((norm_rot + 1.0) / 2.0) * 4, 0, 3))
            pose_idx = int(np.clip(((norm_pose + 1.0) / 2.0) * 6, 0, 5))

            phys_action = (x_idx, y_idx, rot_idx, pose_idx)

            next_h, next_f, reward, done = env.step(phys_action)
            episode_reward += reward

            replay_buffer.append((h, f, torch.tensor(action_array, dtype=torch.float32),
                                  torch.tensor(reward, dtype=torch.float32),
                                  next_h, next_f, torch.tensor(float(done), dtype=torch.float32)))

            h, f = next_h, next_f
            global_step += 1

            if len(replay_buffer) >= BATCH_SIZE:
                batch = random.sample(replay_buffer, BATCH_SIZE)

                bh = torch.stack([t[0] for t in batch]).to(device)
                bf = torch.stack([t[1] for t in batch]).to(device)
                ba = torch.stack([t[2] for t in batch]).to(device)
                br = torch.stack([t[3] for t in batch]).to(dtype=torch.float32, device=device).unsqueeze(1)
                bnh = torch.stack([t[4] for t in batch]).to(device)
                bnf = torch.stack([t[5] for t in batch]).to(device)
                bd = torch.stack([t[6] for t in batch]).to(dtype=torch.float32, device=device).unsqueeze(1)

                bf = (bf - global_mean) / global_std
                bnf = (bnf - global_mean) / global_std

                alpha = log_alpha.exp().detach()

                with torch.no_grad():
                    next_action, next_log_prob = actor.sample_action(bnh, bnf)
                    target_q1, target_q2 = target_critic(bnh, bnf, next_action)
                    target_q = torch.min(target_q1, target_q2) - alpha * next_log_prob
                    target_q = br + GAMMA * (1 - bd) * target_q

                current_q1, current_q2 = critic(bh, bf, ba)
                critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

                critic_optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                critic_optim.step()

                if global_step > 2000:
                    curr_action, curr_log_prob = actor.sample_action(bh, bf)
                    q1_curr, q2_curr = critic(bh, bf, curr_action)
                    q_curr = torch.min(q1_curr, q2_curr)

                    actor_loss = ((alpha * curr_log_prob) - q_curr).mean()

                    actor_optim.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                    actor_optim.step()

                    alpha_loss = -(log_alpha * (curr_log_prob + target_entropy).detach()).mean()
                    alpha_optim.zero_grad()
                    alpha_loss.backward()
                    alpha_optim.step()

                for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                    target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

        episode_rewards.append(episode_reward)

        window_size = 50
        if len(episode_rewards) >= window_size:
            moving_avg = np.mean(episode_rewards[-window_size:])
        else:
            moving_avg = np.mean(episode_rewards)
        moving_averages.append(moving_avg)

        if episode % 10 == 0:
            print(f"Episode {episode} | Reward: {episode_reward:.2f} | 50-Ep Avg: {moving_avg:.2f} | Buffer: {len(replay_buffer)}")

        if episode % 100 == 0:
            torch.save(actor.state_dict(), f"{VOLUME_PATH}/actor_sac_physics_ep{episode}.pt")
            volume.commit()

    print("\nGenerating training reward plot...")
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(range(1, len(episode_rewards) + 1), episode_rewards, alpha=0.3, color='blue', label='Raw Episode Reward')

    ax.plot(range(1, len(moving_averages) + 1), moving_averages, color='red', linewidth=2, label='50-Episode Moving Average')

    ax.set_title('SAC Online Physics Training Rewards')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.legend()
    ax.grid(True)

    plot_path = f"{VOLUME_PATH}/sac_online_rewards_alt.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    volume.commit()
    print(f"Saved reward plot to '{plot_path}'!")

@app.local_entrypoint()
def main():
    print("Initiating remote SAC Online Physics training...")
    train_online_sac_remote.remote()
    print("Training complete! The plot and models are safely stored in the Modal Volume.")