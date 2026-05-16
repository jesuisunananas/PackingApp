import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import modal
import numpy as np
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau

APP_NAME = "tetrisbot-train-bc"
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
    .uv_sync()
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

from model_continuous import MDNSpatialActor, SpatialSoftmax
import torch.distributions as D

class ContinuousBCDataset(Dataset):
    def __init__(self, data_pattern):
        print(f"Loading datasets matching {data_pattern}...")
        chungus_files = glob.glob(data_pattern)
        print(f"Found {len(chungus_files)} files.")

        if len(chungus_files) == 0:
            raise FileNotFoundError(f"No chunks found matching {data_pattern} inside the volume!")

        h_maps, feats, acts = [], [], []

        for f in chungus_files:
            chunk = torch.load(f, weights_only=False)
            h_maps.append(chunk["X_heightmap"])
            feats.append(chunk["X_features"])
            acts.append(chunk["y_action"])

        self.X_heightmap = torch.cat(h_maps)
        self.X_features = torch.cat(feats)
        self.y_action = torch.cat(acts)

        print(f"Loaded {len(self.X_heightmap)} total continuous state-action transitions.")

    def __len__(self):
        return len(self.X_heightmap)

    def __getitem__(self, idx):

        return self.X_heightmap[idx], self.X_features[idx], self.y_action[idx]

def mdn_loss(pi, mu, sigma, target):
    """
    pi shape: [B, K]
    mu shape: [B, K, A]
    sigma shape: [B, K, A]
    target shape: [B, A]
    """

    target = target.unsqueeze(1)

    normal_dist = D.Normal(loc=mu, scale=sigma)

    log_prob = normal_dist.log_prob(target)

    log_prob = log_prob.sum(dim=-1)

    log_pi = torch.log(pi + 1e-8)
    weighted_log_prob = log_pi + log_prob

    loss = -torch.logsumexp(weighted_log_prob, dim=1)

    return loss.mean()

@app.function(
    volumes={VOLUME_PATH: volume},
    image=image,
    env=env,
    gpu="A10G",
    timeout=60 * 60 * 12
)
def train_bc_remote():
    set_seed(10)

    train_dataset = ContinuousBCDataset(f"{VOLUME_PATH}/bc_dataset_train_chunk_continuous*.pt")
    test_dataset = ContinuousBCDataset(f"{VOLUME_PATH}/bc_dataset_test_chunk_continuous*.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Calculating Global Feature Normalization Stats...")
    all_f = []
    for i in range(len(train_dataset)):
        all_f.append(train_dataset[i][1])
    all_f = torch.stack(all_f)
    global_mean = all_f.mean(dim=0).to(device)
    global_std = (all_f.std(dim=0) + 1e-6).to(device)

    scaler_dict = {"mean": global_mean.cpu(), "std": global_std.cpu()}
    torch.save(scaler_dict, f"{VOLUME_PATH}/feature_scaler.pt")
    volume.commit()
    print(f"Saved Global Scaler to {VOLUME_PATH}/feature_scaler.pt")

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    actor = MDNSpatialActor(feature_dim=8, map_size=30, action_dim=4, num_mixtures=5).to(device)
    optimizer = optim.Adam(actor.parameters(), lr=3e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)

    epochs = 20
    train_losses = []
    test_losses = []

    train_soft_accuracies = []
    eval_soft_accuracies = []
    SOFT_MATCH_THRESHOLD = 0.10

    best_test_loss = float('inf')

    print("\nStarting Continuous Behavioral Cloning (MDN NLL Regression)...")
    for epoch in range(epochs):

        actor.train()
        total_train_loss = 0.0
        train_soft_acc_count = 0
        train_samples = 0

        for X_heightmap, X_features, y_action in train_loader:
            X_heightmap = X_heightmap.to(device)
            X_features = X_features.to(device)
            y_action = y_action.to(device)

            X_features = (X_features - global_mean) / global_std

            optimizer.zero_grad()
            pi, mu, sigma = actor(X_heightmap, X_features)
            loss = mdn_loss(pi, mu, sigma, y_action)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()

            with torch.no_grad():
                best_mixture_idx = torch.argmax(pi, dim=-1)
                B = pi.shape[0]
                batch_indices = torch.arange(B)
                best_pred = mu[batch_indices, best_mixture_idx, :]

                distances = torch.norm(best_pred[:, 0:2] - y_action[:, 0:2], dim=1)
                train_soft_acc_count += (distances < SOFT_MATCH_THRESHOLD).sum().item()
                train_samples += B

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_soft_accuracies.append((train_soft_acc_count / train_samples) * 100.0)

        actor.eval()
        total_test_loss = 0.0

        total_l2_distance = 0.0
        eval_soft_acc_count = 0
        eval_samples = 0

        with torch.no_grad():
            for X_heightmap, X_features, y_action in test_loader:
                X_heightmap = X_heightmap.to(device)
                X_features = X_features.to(device)
                y_action = y_action.to(device)

                X_features = (X_features - global_mean) / global_std

                pi, mu, sigma = actor(X_heightmap, X_features)
                loss = mdn_loss(pi, mu, sigma, y_action)
                total_test_loss += loss.item()

                best_mixture_idx = torch.argmax(pi, dim=-1)

                B = pi.shape[0]
                batch_indices = torch.arange(B)
                best_predicted_action = mu[batch_indices, best_mixture_idx, :]

                distances = torch.norm(best_predicted_action[:, 0:2] - y_action[:, 0:2], dim=1)
                total_l2_distance += distances.sum().item()

                eval_soft_acc_count += (distances < SOFT_MATCH_THRESHOLD).sum().item()
                eval_samples += B

        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        avg_l2_distance = total_l2_distance / eval_samples
        eval_soft_accuracy = (eval_soft_acc_count / eval_samples) * 100.0
        eval_soft_accuracies.append(eval_soft_accuracy)

        print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Eval Loss: {avg_test_loss:.4f} | XY Error: {avg_l2_distance:.3f} | Train Soft Acc: {train_soft_accuracies[-1]:.1f}% | Eval Soft Acc: {eval_soft_accuracy:.1f}%")

        scheduler.step(avg_test_loss)

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            model_path = f"{VOLUME_PATH}/actor_bc_continuous.pt"
            torch.save(actor.state_dict(), model_path)
            volume.commit()
            print(f"New best Continuous Actor saved! (Eval Loss: {best_test_loss:.4f})")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(range(1, epochs+1), train_losses, marker='o', linestyle='-', color='b', label='Train Loss')
    ax1.plot(range(1, epochs+1), test_losses, marker='x', linestyle='--', color='r', label='Eval Loss')
    ax1.set_title('MDN Spatial Actor Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Negative Log-Likelihood')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(range(1, epochs+1), train_soft_accuracies, marker='o', linestyle='-', color='g', label='Train Soft Accuracy')
    ax2.plot(range(1, epochs+1), eval_soft_accuracies, marker='x', linestyle='--', color='purple', label='Eval Soft Accuracy')
    ax2.set_title(f'Spatial Packing Accuracy (Threshold: {SOFT_MATCH_THRESHOLD})')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    metric_path = f"{VOLUME_PATH}/bc_training_metrics_continuous.png"
    plt.savefig(metric_path)
    print(f"Saved training metrics to '{metric_path}'!")
    volume.commit()

@app.local_entrypoint()
def main():
    print("Initiating remote BC Continuous training...")
    train_bc_remote.remote()
    print("Training complete! The best model and plots are safely stored in the Modal Volume.")