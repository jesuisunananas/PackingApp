import os
import glob
import random
import time
import torch
import numpy as np
import pybullet as p

from model_continuous import MDNSpatialActor
from box import MeshBox
from pybullet_env import PyBulletPackingEnv, BIN_L, BIN_W, CELL_SIZE

MODEL_PATH = "actor_sac_physics_ep1000.pt"
NUM_EPISODES = 5
ITEMS_PER_EPISODE = 10

def get_random_boxes(num_boxes):
    test_meshes = glob.glob(os.path.join("meshes", "ModelNet10", "*", "test", "*.off"))
    if not test_meshes:
        raise FileNotFoundError("Could not find test meshes. Check your 'meshes' directory path.")

    boxes = []
    for _ in range(num_boxes):
        mesh_path = random.choice(test_meshes)
        box = MeshBox(mesh_path, cell_size=CELL_SIZE)
        box.fragility = random.uniform(0.1, 1.0)
        boxes.append(box)

    for b in boxes:
        vol = b.volume if hasattr(b, 'volume') else b._volume
        b._sorting_weight = (vol * 0.5) + ((1.0 - b.fragility) * 0.5)

    expert_boxes = sorted(boxes, key=lambda b: b._sorting_weight, reverse=True)
    return expert_boxes

def visualize_agent():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading trained SAC policy on {device}...")

    actor = MDNSpatialActor(feature_dim=8, map_size=30, action_dim=4, num_mixtures=5).to(device)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Could not find model weights at {MODEL_PATH}")

    actor.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    actor.eval()

    scaler_path = "feature_scaler.pt"
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Missing {scaler_path}. Run: modal volume get tetrisbot-data-volume feature_scaler.pt .")

    scaler_dict = torch.load(scaler_path, map_location=device, weights_only=True)
    global_mean = scaler_dict["mean"].to(device)
    global_std = scaler_dict["std"].to(device)

    print("\nLaunching PyBullet GUI...")
    env = PyBulletPackingEnv(render=True)

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

    for episode in range(1, NUM_EPISODES + 1):
        print(f"\n{'='*40}")
        print(f"{'='*40}")

        boxes = get_random_boxes(ITEMS_PER_EPISODE)
        h, f = env.reset(boxes)
        episode_reward = 0
        done = False

        while not done:
            h_tensor = h.unsqueeze(0).to(device)
            f_tensor = f.unsqueeze(0).to(device)
            f_tensor = (f_tensor - global_mean) / global_std

            with torch.no_grad():
                pi, mu, sigma = actor(h_tensor, f_tensor)
                best_mixture_idx = torch.argmax(pi, dim=-1)

                B = pi.shape[0]
                batch_indices = torch.arange(B)
                best_action = mu[batch_indices, best_mixture_idx, :]

                action_array = best_action.squeeze(0).cpu().numpy()

            norm_x, norm_y, norm_rot, norm_pose = action_array

            x_idx = int(np.clip(((norm_x + 1.0) / 2.0) * BIN_L, 0, BIN_L - 1))
            y_idx = int(np.clip(((norm_y + 1.0) / 2.0) * BIN_W, 0, BIN_W - 1))
            rot_idx = int(np.clip(((norm_rot + 1.0) / 2.0) * 4, 0, 3))
            pose_idx = int(np.clip(((norm_pose + 1.0) / 2.0) * 6, 0, 5))

            phys_action = (x_idx, y_idx, rot_idx, pose_idx)

            current_box = env.episode_boxes[env.current_step]
            fragility = getattr(current_box, 'fragility', 0.0)

            next_h, next_f, reward, done = env.step(phys_action)
            episode_reward += reward
            h, f = next_h, next_f

            time.sleep(0.5)

        final_max_height = next_h.max().item()

        compactness = 1.0 - (final_max_height / 2.5)

        print(f"\nEpisode {episode}")

        input("\n Press Enter to reset for the next episode...")

if __name__ == "__main__":
    visualize_agent()