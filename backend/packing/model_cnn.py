import torch
import torch.nn as nn
import torch.nn.functional as F

class TerrainEncoder(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, heightmap):

        x = heightmap.unsqueeze(1)
        return self.net(x)

class ObjectEncoder(nn.Module):
    def __init__(self, feature_dim=8, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

    def forward(self, features):

        return self.net(features)

class SpatialMatchingPolicy(nn.Module):
    def __init__(self, feature_dim=8, hidden_dim=64, num_rotations=4, num_poses=6):
        super().__init__()
        self.num_rotations = num_rotations
        self.num_poses = num_poses

        self.terrain_enc = TerrainEncoder(hidden_dim)
        self.obj_enc = ObjectEncoder(feature_dim, hidden_dim)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(hidden_dim, num_rotations * num_poses, kernel_size=1)
        )

    def forward(self, heightmap, features):
        B, L, W = heightmap.shape

        t_enc = self.terrain_enc(heightmap)

        o_enc = self.obj_enc(features)

        o_enc_broadcast = o_enc.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, L, W)

        fused = torch.cat([t_enc, o_enc_broadcast], dim=1)

        logits = self.fusion_conv(fused)

        logits = logits.view(B, self.num_rotations, self.num_poses, L, W)

        return logits

class SpatialCritic(nn.Module):
    def __init__(self, feature_dim=8, hidden_dim=64):
        super().__init__()
        self.terrain_enc = TerrainEncoder(hidden_dim)
        self.obj_enc = ObjectEncoder(feature_dim, hidden_dim)

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, heightmap, features):

        t_enc = self.terrain_enc(heightmap)
        t_pooled = t_enc.mean(dim=(2, 3))

        o_enc = self.obj_enc(features)

        fused = torch.cat([t_pooled, o_enc], dim=1)

        return self.value_head(fused).squeeze(-1)

class SpatialQNetwork(nn.Module):
    def __init__(self, feature_dim=8, hidden_dim=64, num_rotations=4, num_poses=6):
        super().__init__()
        self.num_rotations = num_rotations
        self.num_poses = num_poses

        self.terrain_enc = TerrainEncoder(hidden_dim)
        self.obj_enc = ObjectEncoder(feature_dim, hidden_dim)

        self.q_value_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(hidden_dim, num_rotations * num_poses, kernel_size=1)
        )

    def forward(self, heightmap, features):
        B, L, W = heightmap.shape
        t_enc = self.terrain_enc(heightmap)
        o_enc = self.obj_enc(features)

        o_enc_broadcast = o_enc.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, L, W)
        fused = torch.cat([t_enc, o_enc_broadcast], dim=1)

        q_values = self.q_value_conv(fused)
        q_values = q_values.view(B, self.num_rotations, self.num_poses, L, W)

        return q_values