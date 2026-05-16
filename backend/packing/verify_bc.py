import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import glob

from model_cnn import SpatialQNetwork

def verify_vibe():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_dim = 8
    hidden_dim = 64
    num_rotations = 4
    num_poses = 6

    policy = SpatialQNetwork(feature_dim, hidden_dim, num_rotations, num_poses).to(device)

    policy.load_state_dict(torch.load('policy_cql_spatial.pt', map_location=device))
    policy.eval()

    test_files = glob.glob("bc_dataset_test_chunk_*.pt")
    if not test_files:
        print("No test chunks found!")
        return

    random_chunk = torch.load(random.choice(test_files), weights_only=False)
    random_episode = random.choice(random_chunk)

    h, f, action, r, next_h, next_f, done = random.choice(random_episode)

    with torch.no_grad():

        q_values = policy(h.unsqueeze(0).to(device), f.unsqueeze(0).to(device))

        probs = F.softmax(q_values.view(1, -1), dim=1).view(num_rotations, num_poses, 30, 30)

    ex_x, ex_y, ex_rot, ex_pose = action

    ai_heatmap = probs[ex_rot, ex_pose].cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.imshow(h.cpu().numpy(), cmap='viridis', origin='lower')
    ax1.scatter(ex_x, ex_y, color='red', marker='X', s=100, label='Expert Action')
    ax1.set_title(f"Current Bin State\nExpert target: ({ex_x}, {ex_y})")
    ax1.legend()

    im = ax2.imshow(ai_heatmap, cmap='hot', origin='lower')
    plt.colorbar(im, ax=ax2, label='Confidence (Softmax of Q)')
    ax2.set_title(f"CQL Spatial Q-Values\n(Rot: {ex_rot}, Pose: {ex_pose})")

    plt.tight_layout()
    plt.savefig('bc_vibe_check_cql.png')

if __name__ == "__main__":
    verify_vibe()