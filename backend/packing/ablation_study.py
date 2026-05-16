import os
import glob
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

from model_continuous import MDNSpatialActor
from box import MeshBox, Bin
from pybullet_env import PyBulletPackingEnv, BIN_L, BIN_W, BIN_H, CELL_SIZE
import heuristics

SAC_MODEL_PATH = "actor_sac_physics_ep1000.pt"
BC_MODEL_PATH = "actor_bc_continuous.pt"
SCALER_PATH = "feature_scaler.pt"

SEEDS = [20, 100, 500, 1000, 1500]
EPISODES_PER_SEED = 10
ITEMS_PER_EPISODE = 10

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_random_boxes(num_boxes):
    test_meshes = glob.glob(os.path.join("meshes", "ModelNet10", "*", "test", "*.off"))
    boxes = []
    for _ in range(num_boxes):
        mesh_path = random.choice(test_meshes)
        file_size_mb = os.path.getsize(mesh_path) / (1024 * 1024)
        if file_size_mb > 15.0:
            continue
        fragility = random.uniform(0.1, 1.0)
        box = MeshBox(mesh_path, cell_size=CELL_SIZE, fragility=fragility)
        boxes.append(box)

    for b in boxes:
        vol = b.volume if hasattr(b, 'volume') else b._volume
        b._sorting_weight = (vol * 0.5) + ((1.0 - b.fragility) * 0.5)
    expert_boxes = sorted(boxes, key=lambda b: b._sorting_weight, reverse=True)
    return expert_boxes

def run_heuristic_episode(boxes, env):
    env.reset(boxes)
    heuristic_bin = Bin(BIN_L, BIN_W, BIN_H)

    total_reward = 0
    success = True

    for i, box in enumerate(boxes):
        best_candidate = heuristics.place_box_with_rule(box, heuristic_bin)

        if best_candidate is None:
            phys_action = (0, 0, 0, 0)
        else:
            x_idx, y_idx, _, rot_idx, pose_idx = best_candidate
            phys_action = (x_idx, y_idx, rot_idx, pose_idx)

        _, _, reward, done = env.step(phys_action)
        total_reward += reward

        if reward <= -15.0:
            success = False

        if done:
            break

    return total_reward, success

def run_agent_episode(boxes, env, actor, global_mean, global_std, device):
    h, f = env.reset(boxes)
    total_reward = 0
    success = True
    done = False

    while not done:
        h_tensor = h.unsqueeze(0).to(device)
        f_tensor = f.unsqueeze(0).to(device)
        f_tensor = (f_tensor - global_mean) / global_std

        with torch.no_grad():
            pi, mu, sigma = actor(h_tensor, f_tensor)
            best_mixture_idx = torch.argmax(pi, dim=-1)
            best_action = mu[0, best_mixture_idx, :]
            action_array = best_action.squeeze(0).cpu().numpy()

        norm_x, norm_y, norm_rot, norm_pose = action_array
        x_idx = int(np.clip(((norm_x + 1.0) / 2.0) * BIN_L, 0, BIN_L - 1))
        y_idx = int(np.clip(((norm_y + 1.0) / 2.0) * BIN_W, 0, BIN_W - 1))
        rot_idx = int(np.clip(((norm_rot + 1.0) / 2.0) * 4, 0, 3))
        pose_idx = int(np.clip(((norm_pose + 1.0) / 2.0) * 6, 0, 5))

        phys_action = (x_idx, y_idx, rot_idx, pose_idx)

        next_h, next_f, reward, done = env.step(phys_action)
        total_reward += reward

        if reward <= -15.0:
            success = False

        h, f = next_h, next_f

    return total_reward, success

def plot_ablation_results(results, total_runs):
    methods = []
    mean_rewards = []
    std_rewards = []
    success_rates = []

    for method_name, data in results.items():
        if not data["rewards"]: continue
        methods.append(method_name.replace("_", " ").upper())
        mean_rewards.append(np.mean(data["rewards"]))
        std_rewards.append(np.std(data["rewards"]))
        success_rates.append((data["successes"] / total_runs) * 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    x_pos = np.arange(len(methods))
    bars1 = ax1.bar(x_pos, mean_rewards, yerr=std_rewards, capsize=10, color=['#e74c3c', '#f39c12', '#2ecc71'], alpha=0.8)
    ax1.set_ylabel('Average Physics Reward')
    ax1.set_title('Ablation Study: Average Reward per Episode')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(methods)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + (5 if yval > 0 else -5), f'{yval:.1f}', ha='center', va='bottom' if yval > 0 else 'top', fontweight='bold')

    bars2 = ax2.bar(x_pos, success_rates, color=['#e74c3c', '#f39c12', '#2ecc71'], alpha=0.8)
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title('Ablation Study: Physics Success Rate')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(methods)
    ax2.set_ylim(0, 105)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('ablation_results.png', dpi=300)

def run_ablation():

    device = torch.device("cpu")

    bc_actor = MDNSpatialActor(feature_dim=8, map_size=30, action_dim=4, num_mixtures=5).to(device)
    if os.path.exists(BC_MODEL_PATH):
        bc_actor.load_state_dict(torch.load(BC_MODEL_PATH, map_location=device, weights_only=True))
        bc_actor.eval()
    else:
        print(f"Warning: {BC_MODEL_PATH} not found. BC evaluation will be skipped.")
        bc_actor = None

    sac_actor = MDNSpatialActor(feature_dim=8, map_size=30, action_dim=4, num_mixtures=5).to(device)
    sac_actor.load_state_dict(torch.load(SAC_MODEL_PATH, map_location=device, weights_only=True))
    sac_actor.eval()

    scaler_dict = torch.load(SCALER_PATH, map_location=device, weights_only=True)
    global_mean = scaler_dict["mean"].to(device)
    global_std = scaler_dict["std"].to(device)

    env = PyBulletPackingEnv(render=False)

    results = {
        "heuristic": {"rewards": [], "successes": 0},
        "bc_only": {"rewards": [], "successes": 0},
        "sac_full": {"rewards": [], "successes": 0}
    }

    total_runs = len(SEEDS) * EPISODES_PER_SEED

    for seed in SEEDS:
        for i in range(EPISODES_PER_SEED):
            set_seed(seed * 100 + i)
            boxes = get_random_boxes(ITEMS_PER_EPISODE)
            h_rew, h_succ = run_heuristic_episode(boxes, env)

            if bc_actor is not None:
                bc_rew, bc_succ = run_agent_episode(boxes, env, bc_actor, global_mean, global_std, device)
            else:
                bc_rew, bc_succ = 0.0, False

            s_rew, s_succ = run_agent_episode(boxes, env, sac_actor, global_mean, global_std, device)

            results["heuristic"]["rewards"].append(h_rew)
            results["heuristic"]["successes"] += 1 if h_succ else 0

            results["bc_only"]["rewards"].append(bc_rew)
            results["bc_only"]["successes"] += 1 if bc_succ else 0

            results["sac_full"]["rewards"].append(s_rew)
            results["sac_full"]["successes"] += 1 if s_succ else 0

            print(f"  Ep {i+1:02d} | Heuristic: {h_rew:6.2f} | BC-Only: {bc_rew:6.2f} | SAC-Full: {s_rew:6.2f}")

    for method_name, data in results.items():
        if not data["rewards"]: continue

        mean_rew = np.mean(data["rewards"])
        std_rew = np.std(data["rewards"])
        succ_rate = (data["successes"] / total_runs) * 100

        print(f"[{method_name.upper()}]")
        print(f"  Average Reward: {mean_rew:5.2f} ± {std_rew:5.2f}")
        print(f"  Success Rate:   {succ_rate:.1f}%")
        print("-" * 40)

    plot_ablation_results(results, total_runs)

if __name__ == "__main__":
    run_ablation()
