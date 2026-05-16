import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
import glob

from model_continuous import MDNSpatialActor

def verify():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_dim = 8
    map_size = 30
    action_dim = 4
    num_mixtures = 5

    actor = MDNSpatialActor(feature_dim, map_size, action_dim, num_mixtures).to(device)

    model_path = 'actor_bc_continuous.pt'
    try:
        actor.load_state_dict(torch.load(model_path, map_location=device))
        actor.eval()
    except Exception as e:
        print(f"Failed to load {model_path}: {e}")
        return

    test_files = glob.glob("bc_dataset_test_chunk_continuous_*.pt")
    if not test_files:
        print("No test chunks found!")
        return

    random_chunk = torch.load(random.choice(test_files), map_location='cpu', weights_only=False)

    num_transitions = random_chunk['X_heightmap'].shape[0]
    idx = random.randint(0, num_transitions - 1)

    h = random_chunk['X_heightmap'][idx]
    f = random_chunk['X_features'][idx]
    action = random_chunk['y_action'][idx]

    with torch.no_grad():
        h_tensor = h.unsqueeze(0).to(device)
        f_tensor = f.unsqueeze(0).to(device)
        pi, mu, sigma = actor(h_tensor, f_tensor)

        pi = pi[0].cpu().numpy()
        mu = mu[0].cpu().numpy()
        sigma = sigma[0].cpu().numpy()

    ex_norm_x, ex_norm_y, ex_norm_rot, ex_norm_pose = action.numpy()

    ex_x = ((ex_norm_x + 1.0) / 2.0) * 30
    ex_y = ((ex_norm_y + 1.0) / 2.0) * 30

    X, Y = np.meshgrid(np.linspace(0, 30, 300), np.linspace(0, 30, 300))
    Z = np.zeros_like(X, dtype=float)

    for m in range(num_mixtures):
        weight = pi[m]

        mx = ((mu[m, 0] + 1.0) / 2.0) * 30
        my = ((mu[m, 1] + 1.0) / 2.0) * 30

        sx = (sigma[m, 0] / 2.0) * 30
        sy = (sigma[m, 1] / 2.0) * 30

        sx = max(sx, 1.5)
        sy = max(sy, 1.5)

        pdf = weight * np.exp(-((X - mx)**2 / (2 * sx**2) + (Y - my)**2 / (2 * sy**2)))
        Z += pdf

    fig, ax1 = plt.subplots(figsize=(6, 5))

    ax1.imshow(h.cpu().numpy(), cmap='viridis', origin='lower')
    ax1.scatter(ex_x, ex_y, color='red', marker='X', s=100, label='Expert Action')
    ax1.set_title(f"Current Bin State\nExpert target: (X: {ex_x:.1f}, Y: {ex_y:.1f})")
    ax1.legend()

    plt.tight_layout()
    plt.savefig('bc_mdn.png')

if __name__ == "__main__":
    verify()
