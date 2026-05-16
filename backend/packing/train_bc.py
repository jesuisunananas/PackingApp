import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import glob
import random
import numpy as np

from model_continuous import ContinuousSpatialActor

def set_seed(seed=10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class ContinuousBCDataset(Dataset):
    def __init__(self, data_pattern):
        print(f"Loading datasets matching {data_pattern}...")
        chungus_files = glob.glob(data_pattern)
        print(f"Found {len(chungus_files)} files.")

        if len(chungus_files) == 0:
            raise FileNotFoundError(f"No chunks found matching {data_pattern} locally!")

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

def train_bc():
    set_seed(10)

    train_dataset = ContinuousBCDataset("bc_dataset_train_chunk_continuous*.pt")
    test_dataset = ContinuousBCDataset("bc_dataset_test_chunk_continuous*.pt")

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Training locally on device: {device}")

    actor = ContinuousSpatialActor(feature_dim=8, map_size=30, action_dim=4).to(device)

    optimizer = optim.Adam(actor.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    epochs = 20
    train_losses = []
    test_losses = []

    best_test_loss = float('inf')

    print("\nStarting Local Continuous Behavioral Cloning (MSE Regression)...")
    for epoch in range(epochs):

        actor.train()
        total_train_loss = 0.0

        for X_heightmap, X_features, y_action in train_loader:
            X_heightmap = X_heightmap.to(device)
            X_features = X_features.to(device)
            y_action = y_action.to(device)

            X_mean = X_features.mean(dim=0, keepdim=True)
            X_std = X_features.std(dim=0, keepdim=True) + 1e-6
            X_features = (X_features - X_mean) / X_std

            optimizer.zero_grad()

            mu, _ = actor(X_heightmap, X_features)

            loss = criterion(mu, y_action)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        actor.eval()
        total_test_loss = 0.0

        with torch.no_grad():
            for X_heightmap, X_features, y_action in test_loader:
                X_heightmap = X_heightmap.to(device)
                X_features = X_features.to(device)
                y_action = y_action.to(device)

                X_mean = X_features.mean(dim=0, keepdim=True)
                X_std = X_features.std(dim=0, keepdim=True) + 1e-6
                X_features = (X_features - X_mean) / X_std

                mu, _ = actor(X_heightmap, X_features)
                loss = criterion(mu, y_action)
                total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        print(f"Epoch {epoch+1:02d} | Train MSE: {avg_train_loss:.4f} | Eval MSE: {avg_test_loss:.4f}")

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(actor.state_dict(), "actor_bc_continuous_local.pt")

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(range(1, epochs+1), train_losses, marker='o', linestyle='-', color='b', label='Train MSE Loss')
    ax1.plot(range(1, epochs+1), test_losses, marker='x', linestyle='--', color='r', label='Eval MSE Loss')
    ax1.set_title('Local Continuous Spatial Regression Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Mean Squared Error')
    ax1.legend()
    ax1.grid(True)

    plt.tight_layout()
    metric_path = "bc_training_metrics_continuous_local.png"
    plt.savefig(metric_path)
    print(f"Saved training metrics locally to '{metric_path}'!")
    print(f"Best local model saved with Eval MSE: {best_test_loss:.4f}")

if __name__ == "__main__":
    train_bc()