import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import glob

# Import the new Spatial Matching Network
from model_cnn import SpatialMatchingPolicy

class BCDataset(Dataset):
    def __init__(self, data_pattern="bc_dataset_train_chunk_*.pt"):
        print(f"Loading datasets matching {data_pattern}...")
        chungus_files = glob.glob(data_pattern)
        print(len(chungus_files))
        #raw_episodes = torch.load(data_path)
        
        # The new dataset format is a list of episodes.
        # Each episode is a list of transitions: (X_heightmap, X_features, y_action)
        # We need to flatten this into a single list for the DataLoader
        self.transitions = []

        for f in chungus_files:
            raw_episodes = torch.load(f, weights_only=False)
            for episode in raw_episodes:
                for transition in episode:
                    self.transitions.append(transition)
                
        print(f"Loaded {len(self.transitions)} total state-action transitions from {data_pattern}")
        
    def __len__(self):
        return len(self.transitions)
        
    def __getitem__(self, idx):
        return self.transitions[idx]

def train_bc():
    # Load Train and Test datasets
    train_dataset = BCDataset("bc_dataset_train_chunk_*.pt")
    test_dataset = BCDataset("bc_dataset_test_chunk_*.pt")
    
    # Batch size can be larger since we aren't dealing with sequential RNNS
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Hyperparameters
    feature_dim = 8
    hidden_dim = 64
    num_rotations = 4
    num_poses = 6
    
    # Initialize the new Spatial Matching Architecture
    policy = SpatialMatchingPolicy(
        feature_dim=feature_dim, 
        hidden_dim=hidden_dim, 
        num_rotations=num_rotations, 
        num_poses=num_poses
    )
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    policy.to(device)
    
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)
    
    # Standard Cross-Entropy for 1D flattened classes
    criterion = nn.CrossEntropyLoss()
    
    epochs = 20
    train_losses = []
    test_losses = []
    eval_accuracies = []
    
    print("\nStarting Offline Behavior Cloning (Row Matching)...")
    for epoch in range(epochs):
        # --- TRAINING ---
        policy.train()
        total_train_loss = 0.0
        
        for X_heightmap, X_features, y_action, reward, next_h, next_f, done in train_loader:
            X_heightmap = X_heightmap.to(device)
            X_features = X_features.to(device)
            y_action = y_action.to(device)
            
            # Normalize Object Features
            # X_mean = X_features.mean(dim=0, keepdim=True)
            # X_std = X_features.std(dim=0, keepdim=True) + 1e-6
            # X_features = (X_features - X_mean) / X_std
            
            optimizer.zero_grad()
            
            # Forward pass -> (B, Rot, Pose, L, W)
            logits = policy(X_heightmap, X_features)
            
            B, R, P, L, W = logits.shape
            
            # Flatten logits to (B, Classes)
            logits_flat = logits.view(B, -1)
            
            # Extract target actions: [x, y, rot, pose]
            x_target = y_action[:, 0]
            y_target = y_action[:, 1]
            rot_target = y_action[:, 2]
            pose_target = y_action[:, 3]
            
            # Calculate the 1D index using multi-dimensional striding
            target_indices = (((rot_target * P) + pose_target) * L + y_target) * W + x_target
            
            loss = criterion(logits_flat, target_indices)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # --- EVALUATION ---
        policy.eval()
        total_test_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for X_heightmap, X_features, y_action, reward, next_h, next_f, done in test_loader:
                X_heightmap = X_heightmap.to(device)
                X_features = X_features.to(device)
                y_action = y_action.to(device)
                
                # Normalize using batch statistics
                # X_mean = X_features.mean(dim=0, keepdim=True)
                # X_std = X_features.std(dim=0, keepdim=True) + 1e-6
                # X_features = (X_features - X_mean) / X_std
                
                logits = policy(X_heightmap, X_features)
                B, R, P, L, W = logits.shape
                logits_flat = logits.view(B, -1)
                
                x_target = y_action[:, 0]
                y_target = y_action[:, 1]
                rot_target = y_action[:, 2]
                pose_target = y_action[:, 3]
                
                target_indices = (((rot_target * P) + pose_target) * L + y_target) * W + x_target
                
                loss = criterion(logits_flat, target_indices)
                total_test_loss += loss.item()
                
                # Calculate Accuracy (Did the network predict the exact expert action?)
                preds = logits_flat.argmax(dim=1)
                correct_predictions += (preds == target_indices).sum().item()
                total_samples += B
                
        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        
        accuracy = correct_predictions / total_samples
        eval_accuracies.append(accuracy)
        
        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Eval Loss: {avg_test_loss:.4f} | Eval Acc: {accuracy*100:.2f}%")

    print("\nTraining Complete! Plotting learning curves...")
    
    # Plotting Loss and Accuracy side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss Curve
    ax1.plot(range(1, epochs+1), train_losses, marker='o', linestyle='-', color='b', label='Train Loss')
    ax1.plot(range(1, epochs+1), test_losses, marker='x', linestyle='--', color='r', label='Eval Loss')
    ax1.set_title('Spatial BC Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Accuracy Curve
    ax2.plot(range(1, epochs+1), eval_accuracies, marker='^', linestyle='-', color='g', label='Eval Accuracy')
    ax2.set_title('Exact Match Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('bc_training_metrics.png')
    print("Saved training metrics to 'bc_training_metrics.png'!")
    
    # Save model
    torch.save(policy.state_dict(), 'policy_bc_spatial.pt')
    print("Saved pretrained BC policy to 'policy_bc_spatial.pt'!")

if __name__ == "__main__":
    train_bc()