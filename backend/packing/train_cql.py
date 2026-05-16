import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import glob
import copy

from model_cnn import SpatialQNetwork, SpatialMatchingPolicy
from train_bc import BCDataset

def train_cql():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_dim = 8
    hidden_dim = 64
    num_rotations = 4
    num_poses = 6
    gamma = 0.95
    tau = 0.005
    cql_alpha = 1.0
    lr = 3e-4
    batch_size = 32
    epochs = 10

    q_net = SpatialQNetwork(feature_dim, hidden_dim, num_rotations, num_poses).to(device)
    target_q_net = copy.deepcopy(q_net).to(device)

    bc_policy = SpatialMatchingPolicy(feature_dim, hidden_dim, num_rotations, num_poses).to(device)
    bc_policy.load_state_dict(torch.load('policy_bc_spatial.pt', map_location=device))
    bc_policy.eval()

    q_net.load_state_dict(torch.load('policy_bc_spatial.pt', map_location=device), strict=False)
    print("Pre-initialized Q-Network with BC weights.")

    optimizer = optim.Adam(q_net.parameters(), lr=lr)

    train_loader = DataLoader(BCDataset("bc_dataset_train_chunk_*.pt"), batch_size=batch_size, shuffle=True)

    history_total_loss = []
    history_bellman_loss = []
    history_cql_penalty = []

    print("\nStarting Offline CQL Training...")
    for epoch in range(epochs):
        q_net.train()

        epoch_total = 0.0
        epoch_bellman = 0.0
        epoch_cql = 0.0

        for batch in train_loader:
            h, f, action, reward, next_h, next_f, done = [x.to(device) for x in batch]
            reward = reward.float()
            done = done.float()
            B, L, W = h.shape

            with torch.no_grad():
                next_q_values = target_q_net(next_h, next_f)
                max_next_q, _ = next_q_values.view(B, -1).max(dim=1)
                target_q = reward + (gamma * max_next_q * (1 - done.float()))

            current_q_all = q_net(h, f).view(B, -1)

            x, y, rot, pose = action[:, 0], action[:, 1], action[:, 2], action[:, 3]
            act_idx = (((rot * num_poses) + pose) * L + y) * W + x
            current_q_taken = current_q_all.gather(1, act_idx.unsqueeze(1)).squeeze(1)

            bellman_loss = F.mse_loss(current_q_taken, target_q)

            cql_penalty = (torch.logsumexp(current_q_all, dim=1) - current_q_taken).mean()

            loss = bellman_loss + cql_alpha * cql_penalty

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
            optimizer.step()

            for param, target_param in zip(q_net.parameters(), target_q_net.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

            epoch_total += loss.item()
            epoch_bellman += bellman_loss.item()
            epoch_cql += cql_penalty.item()

        avg_total = epoch_total / len(train_loader)
        avg_bellman = epoch_bellman / len(train_loader)
        avg_cql = epoch_cql / len(train_loader)

        history_total_loss.append(avg_total)
        history_bellman_loss.append(avg_bellman)
        history_cql_penalty.append(avg_cql)

        print(f"Epoch {epoch+1:02d} | Total Loss: {avg_total:.4f} | Bellman MSE: {avg_bellman:.4f} | CQL Penalty: {avg_cql:.4f}")

    torch.save(q_net.state_dict(), 'policy_cql_spatial.pt')
    print("\nCQL Training Complete! Saved 'policy_cql_spatial.pt'.")

    print("Generating training curves...")
    plt.figure(figsize=(10, 6))

    plt.plot(range(1, epochs+1), history_total_loss, label='Total Loss', marker='o', color='black')
    plt.plot(range(1, epochs+1), history_bellman_loss, label='Bellman Loss (MSE)', marker='s', linestyle='--', color='blue')
    plt.plot(range(1, epochs+1), history_cql_penalty, label='CQL Penalty', marker='^', linestyle=':', color='red')

    plt.title('Discrete CQL Training Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig('cql_training_metrics.png')
    print("Saved training metrics to 'cql_training_metrics.png'!")

if __name__ == "__main__":
    train_cql()