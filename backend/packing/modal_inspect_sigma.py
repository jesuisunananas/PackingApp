import torch
import modal
from torch.utils.data import DataLoader
from model_continuous import MDNSpatialActor
from modal_train_bc import ContinuousBCDataset, VOLUME_NAME, VOLUME_PATH, PROJECT_DIR, env, image

app = modal.App("tetrisbot-inspect-sigma")
volume = modal.Volume.from_name(VOLUME_NAME)

@app.function(volumes={VOLUME_PATH: volume}, image=image, env=env)
def inspect_sigma_remote():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inspecting on device: {device}\n")

    actor = MDNSpatialActor(feature_dim=8, map_size=30, action_dim=4, num_mixtures=5).to(device)
    model_path = f"{VOLUME_PATH}/actor_bc_continuous.pt"

    try:
        actor.load_state_dict(torch.load(model_path, map_location=device))
        actor.eval()
        print(f"Successfully loaded checkpoint: {model_path}")
    except FileNotFoundError:
        print(f"Error: Could not find {model_path}. Did you train the model yet?")
        return

    scaler_path = f"{VOLUME_PATH}/feature_scaler.pt"
    scaler = torch.load(scaler_path, map_location=device)
    global_mean = scaler["mean"].to(device)
    global_std = scaler["std"].to(device)

    test_dataset = ContinuousBCDataset(f"{VOLUME_PATH}/bc_dataset_test_chunk_continuous_0.pt")
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    X_heightmap, X_features, y_action = next(iter(test_loader))
    X_heightmap = X_heightmap.to(device)
    X_features = (X_features.to(device) - global_mean) / global_std

    with torch.no_grad():
        pi, mu, sigma = actor(X_heightmap, X_features)

    print("Action space dims: [X, Y, Rotation, Pose]")
    print(f"Expected action range: Roughly [-1.0, 1.0]\n")

    best_mixture_idx = torch.argmax(pi, dim=-1)
    B = pi.shape[0]
    batch_indices = torch.arange(B)

    active_sigmas = sigma[batch_indices, best_mixture_idx, :]

    mean_sigma = active_sigmas.mean(dim=0)
    min_sigma = active_sigmas.min(dim=0).values
    max_sigma = active_sigmas.max(dim=0).values

    action_names = ["X", "Y", "Rotation", "Pose"]
    for i, name in enumerate(action_names):
        print(f"{name} Dimension:")
        print(f"  Mean Sigma: {mean_sigma[i].item():.5f}")
        print(f"  Min  Sigma: {min_sigma[i].item():.5f}")
        print(f"  Max  Sigma: {max_sigma[i].item():.5f}")

    print("> 0.10: Very healthy. SAC will have plenty of room to explore.")
    print("0.01 - 0.10: Getting narrow, but usually acceptable for a warm-start.")
    print("< 0.01: COLLAPSED. The model is near-deterministic. SAC may get stuck.")

@app.local_entrypoint()
def main():
    inspect_sigma_remote.remote()