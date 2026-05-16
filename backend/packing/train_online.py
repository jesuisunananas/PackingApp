import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import glob
import os
from collections import deque
import numpy as np

from model_cnn import SpatialQNetwork
from box import MeshBox
from pybullet_env import PyBulletPackingEnv, BIN_L, BIN_W, CELL_SIZE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LR = 1e-4
GAMMA = 0.95
BATCH_SIZE = 16
MEMORY_SIZE = 2000
EPSILON_START = 0.3
EPSILON_END = 0.05
EPSILON_DECAY = 1000

def get_random_boxes(num_boxes):
    test_meshes = glob.glob(os.path.join("meshes", "ModelNet10", "*", "test", "*.off"))
    boxes = []
    for _ in range(num_boxes):
        mesh_path = random.choice(test_meshes)
        boxes.append(MeshBox(mesh_path, cell_size=CELL_SIZE))
    return boxes

def train_online():

    q_net = SpatialQNetwork(8, 64, 4, 6).to(device)
    target_net = SpatialQNetwork(8, 64, 4, 6).to(device)

    print("Loading Warm-Started CQL Weights...")
    q_net.load_state_dict(torch.load('policy_cql_spatial.pt', map_location=device))
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=LR)

    env = PyBulletPackingEnv(render=False)
    replay_buffer = deque(maxlen=MEMORY_SIZE)

    epsilon = EPSILON_START
    global_step = 0

    for episode in range(1, 501):
        boxes = get_random_boxes(10)
        h, f = env.reset(boxes)
        episode_reward = 0

        done = False
        while not done:
            h_tensor = h.unsqueeze(0).to(device)
            f_tensor = f.unsqueeze(0).to(device)

            with torch.no_grad():
                q_values = q_net(h_tensor, f_tensor).cpu()

            box = boxes[env.current_step]
            grid_l, grid_w = box.length, box.width

            for r in range(4):
                r_grid_l, r_grid_w = (grid_w, grid_l) if r % 2 == 1 else (grid_l, grid_w)
                max_x = max(0, BIN_L - int(r_grid_l))
                max_y = max(0, BIN_W - int(r_grid_w))
                q_values[:, r, :, max_x:, :] = -float('inf')
                q_values[:, r, :, :, max_y:] = -float('inf')

            if random.random() < epsilon:

                valid_indices = torch.where(q_values[0] != -float('inf'))
                random_choice = random.randint(0, len(valid_indices[0]) - 1)
                rot_idx = valid_indices[0][random_choice].item()
                pose_idx = valid_indices[1][random_choice].item()
                x_idx = valid_indices[2][random_choice].item()
                y_idx = valid_indices[3][random_choice].item()
            else:

                best_flat_idx = q_values.argmax().item()
                rot_idx = (best_flat_idx // (6 * 30 * 30)) % 4
                pose_idx = (best_flat_idx // (30 * 30)) % 6
                x_idx = (best_flat_idx // 30) % 30
                y_idx = best_flat_idx % 30

            action = (x_idx, y_idx, rot_idx, pose_idx)

            next_h, next_f, reward, done = env.step(action)
            episode_reward += reward

            flat_action = (((rot_idx * 6) + pose_idx) * 30 + y_idx) * 30 + x_idx
            replay_buffer.append((h, f, flat_action, reward, next_h, next_f, done))

            h, f = next_h, next_f
            global_step += 1

            epsilon = max(EPSILON_END, epsilon - (EPSILON_START - EPSILON_END) / EPSILON_DECAY)

            if len(replay_buffer) >= BATCH_SIZE:
                batch = random.sample(replay_buffer, BATCH_SIZE)
                bh = torch.stack([t[0] for t in batch]).to(device)
                bf = torch.stack([t[1] for t in batch]).to(device)
                ba = torch.tensor([t[2] for t in batch], dtype=torch.long).to(device)
                br = torch.tensor([t[3] for t in batch], dtype=torch.float32).to(device)
                bnh = torch.stack([t[4] for t in batch]).to(device)
                bnf = torch.stack([t[5] for t in batch]).to(device)
                bd = torch.tensor([t[6] for t in batch], dtype=torch.float32).to(device)

                curr_q = q_net(bh, bf).view(BATCH_SIZE, -1).gather(1, ba.unsqueeze(1)).squeeze()
                with torch.no_grad():
                    max_next_q = target_net(bnh, bnf).view(BATCH_SIZE, -1).max(1)[0]
                    target_q = br + GAMMA * max_next_q * (1 - bd)

                loss = F.mse_loss(curr_q, target_q)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
                optimizer.step()

                for target_param, param in zip(target_net.parameters(), q_net.parameters()):
                    target_param.data.copy_(0.005 * param.data + (1.0 - 0.005) * target_param.data)

        print(f"Episode {episode} | Reward: {episode_reward:.2f} | Epsilon: {epsilon:.3f}")

        if episode % 50 == 0:
            torch.save(q_net.state_dict(), f'policy_online_physics_ep{episode}.pt')

if __name__ == "__main__":
    train_online()