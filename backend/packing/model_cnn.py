import torch
import torch.nn as nn
import torch.nn.functional as F

class TerrainEncoder(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        # Use padding to maintain the L x W spatial resolution
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, heightmap):
        # heightmap shape: (B, L, W)
        # Add a channel dimension for the CNN: (B, 1, L, W)
        x = heightmap.unsqueeze(1)
        return self.net(x)

class ObjectEncoder(nn.Module):
    def __init__(self, feature_dim=7, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, features):
        # features shape: (B, 7)
        return self.net(features)

class SpatialMatchingPolicy(nn.Module):
    def __init__(self, feature_dim=7, hidden_dim=64, num_rotations=4, num_poses=6):
        super().__init__()
        self.num_rotations = num_rotations
        self.num_poses = num_poses
        
        self.terrain_enc = TerrainEncoder(hidden_dim)
        self.obj_enc = ObjectEncoder(feature_dim, hidden_dim)

        # Fusion network: Inputs are concatenated along the channel dimension
        # hidden_dim (terrain) + hidden_dim (object) = hidden_dim * 2
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            # Output a heatmap channel for every combination of rotation and pose
            nn.Conv2d(hidden_dim, num_rotations * num_poses, kernel_size=1)
        )

    def forward(self, heightmap, features):
        B, L, W = heightmap.shape

        # 1. Encode spatial terrain -> (B, hidden_dim, L, W)
        t_enc = self.terrain_enc(heightmap)

        # 2. Encode object features -> (B, hidden_dim)
        o_enc = self.obj_enc(features)

        # 3. Broadcast object features across the spatial grid
        # (B, hidden_dim) -> (B, hidden_dim, 1, 1) -> (B, hidden_dim, L, W)
        o_enc_broadcast = o_enc.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, L, W)

        # 4. Concatenate terrain and object features -> (B, hidden_dim * 2, L, W)
        fused = torch.cat([t_enc, o_enc_broadcast], dim=1)

        # 5. Output logits -> (B, 24, L, W)
        logits = self.fusion_conv(fused)

        # Reshape to explicitly separate rotations and poses: (B, Rot, Pose, L, W)
        logits = logits.view(B, self.num_rotations, self.num_poses, L, W)
        
        return logits

class SpatialCritic(nn.Module):
    def __init__(self, feature_dim=7, hidden_dim=64):
        super().__init__()
        self.terrain_enc = TerrainEncoder(hidden_dim)
        self.obj_enc = ObjectEncoder(feature_dim, hidden_dim)

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, heightmap, features):
        # Encode terrain and pool the spatial dimensions to get a single vector per batch
        t_enc = self.terrain_enc(heightmap)       # (B, hidden_dim, L, W)
        t_pooled = t_enc.mean(dim=(2, 3))         # (B, hidden_dim)

        # Encode object
        o_enc = self.obj_enc(features)            # (B, hidden_dim)

        # Concatenate and pass to value head
        fused = torch.cat([t_pooled, o_enc], dim=1) # (B, hidden_dim * 2)
        
        return self.value_head(fused).squeeze(-1)   # (B,)