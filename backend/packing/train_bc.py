import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from model import PointerNetPolicy

class BCDataset(Dataset):
    def __init__(self, data_path="bc_dataset.pt"):
        # Load elements of (X_tensor, y_tensor)
        self.data = torch.load(data_path)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]

def train_bc():
    print("Loading BC Dataset...")
    dataset = BCDataset("bc_dataset.pt")
    
    # We will use batch_size 1 since Pointer Net seq length is dynamic across sequences in general, 
    # but here they are all technically ITEMS_PER_EPISODE. 
    # Let's batch them anyway!
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Feature dim is 7 (length, width, height, vol, frag, l_ratio, w_ratio)
    feature_dim = 7
    hidden_dim = 128
    
    policy = PointerNetPolicy(feature_dim, hidden_dim)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    
    # CrossEntropyLoss expects logits of shape (B, C, d1, d2...)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 20
    epoch_losses = []
    
    print("Starting Offline Behavior Cloning...")
    for epoch in range(epochs):
        policy.train()
        total_loss = 0.0
        
        for batch_idx, (X, y) in enumerate(dataloader):
            # X: (B, N, 7)
            # Normalize X to prevent vanishing gradients across wildly varying CAD dimensions
            X_mean = X.mean(dim=(0, 1), keepdim=True)
            X_std = X.std(dim=(0, 1), keepdim=True) + 1e-6
            X = (X - X_mean) / X_std
            
            # y: (B, N) target indices
            
            optimizer.zero_grad()
            
            # Forward pass with target indices for teacher forcing
            # Returns logits of shape (B, Decode_steps, N)
            indices, logits = policy(X, target_indices=y)
            
            # PyTorch CrossEntropy expects classes as channel dimension: (B, C, sequence_length)
            # Logits is (B, SeqLen, Classes). So we transpose to (B, Classes, SeqLen)
            logits = logits.transpose(1, 2)
            
            loss = criterion(logits, y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

    print("Training Complete! Plotting learning curve...")
    plt.figure(figsize=(8,5))
    plt.plot(range(1, epochs+1), epoch_losses, marker='o', linestyle='-', color='b')
    plt.title('Behavior Cloning Loss Curve (Volume-Descending Expert)')
    plt.xlabel('Epoch')
    plt.ylabel('Average Cross-Entropy Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('bc_training_curve.png')
    print("Saved training curve to 'bc_training_curve.png'!")
    
    # Save model
    torch.save(policy.state_dict(), 'policy_bc.pt')
    print("Saved pretrained BC policy to 'policy_bc.pt'!")

if __name__ == "__main__":
    train_bc()
