import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import glob

# --- UPDATE 1: Import the Q-Network instead of the Policy ---
from model_cnn import SpatialQNetwork 

def verify_vibe():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load the trained CQL Network
    feature_dim = 8
    hidden_dim = 64
    num_rotations = 4
    num_poses = 6
    
    # --- UPDATE 2: Use SpatialQNetwork so names like 'q_value_conv' match the file ---
    policy = SpatialQNetwork(feature_dim, hidden_dim, num_rotations, num_poses).to(device)
    
    # Load the CQL weights
    policy.load_state_dict(torch.load('policy_cql_spatial.pt', map_location=device))
    policy.eval()
    
    # 2. Grab a random transition from the test set
    test_files = glob.glob("bc_dataset_test_chunk_*.pt")
    if not test_files:
        print("No test chunks found!")
        return
        
    random_chunk = torch.load(random.choice(test_files), weights_only=False)
    random_episode = random.choice(random_chunk)
    
    # Unpack the 7-item MDP tuple you generated
    h, f, action, r, next_h, next_f, done = random.choice(random_episode)
    
    # 3. Get Q-Value Prediction
    with torch.no_grad():
        # Forward pass: (B, Rot, Pose, L, W)
        q_values = policy(h.unsqueeze(0).to(device), f.unsqueeze(0).to(device))
        
        # For visualization, we convert Q-values to a "probability-like" heatmap
        # using softmax so the highest Q-value stands out clearly.
        probs = F.softmax(q_values.view(1, -1), dim=1).view(num_rotations, num_poses, 30, 30)
    
    # 4. Find the Expert's Choice
    ex_x, ex_y, ex_rot, ex_pose = action
    
    # 5. Get the heatmap for the expert's chosen orientation
    ai_heatmap = probs[ex_rot, ex_pose].cpu().numpy()
    
    # --- PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: The Bin State
    ax1.imshow(h.cpu().numpy(), cmap='viridis', origin='lower')
    ax1.scatter(ex_x, ex_y, color='red', marker='X', s=100, label='Expert Action')
    ax1.set_title(f"Current Bin State\nExpert target: ({ex_x}, {ex_y})")
    ax1.legend()

    # Plot 2: CQL Heatmap
    im = ax2.imshow(ai_heatmap, cmap='hot', origin='lower')
    plt.colorbar(im, ax=ax2, label='Confidence (Softmax of Q)')
    ax2.set_title(f"CQL Spatial Q-Values\n(Rot: {ex_rot}, Pose: {ex_pose})")
    
    plt.tight_layout()
    plt.savefig('bc_vibe_check_cql.png')
    print("Vibe check complete! Open 'bc_vibe_check_cql.png' to see the heatmap.")

if __name__ == "__main__":
    verify_vibe()