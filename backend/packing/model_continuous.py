import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SpatialSoftmax(nn.Module):
    """Maps 2D Convolutional feature maps directly to 1D spatial coordinates [-1, 1]."""
    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width

        pos_x, pos_y = np.meshgrid(np.linspace(-1., 1., self.width), np.linspace(-1., 1., self.height))
        pos_x = torch.from_numpy(pos_x.reshape(self.height * self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height * self.width)).float()

        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def forward(self, feature):
        B, C, H, W = feature.shape
        feature_flat = feature.view(B, C, H * W)

        softmax_attention = F.softmax(feature_flat, dim=-1)

        expected_x = torch.sum(self.pos_x * softmax_attention, dim=-1, keepdim=True)
        expected_y = torch.sum(self.pos_y * softmax_attention, dim=-1, keepdim=True)

        expected_xy = torch.cat([expected_x, expected_y], dim=2)
        return expected_xy.view(B, C * 2)

class MDNSpatialActor(nn.Module):
    """MDN Actor: Outputs K continuous action distributions (Pi, Mu, Sigma)"""
    def __init__(self, feature_dim=8, map_size=30, action_dim=4, num_mixtures=5):
        super().__init__()
        self.num_mixtures = num_mixtures
        self.action_dim = action_dim

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 16, kernel_size=1)
        )

        self.spatial_softmax = SpatialSoftmax(height=map_size, width=map_size)

        fc_input_dim = (16 * 2) + feature_dim

        self.mlp = nn.Sequential(
            nn.Linear(fc_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.pi_head = nn.Linear(128, num_mixtures)

        self.mu_head = nn.Linear(128, num_mixtures * action_dim)

        self.log_std_head = nn.Linear(128, num_mixtures * action_dim)

    def forward(self, heightmap, features):
        if heightmap.dim() == 3:
            heightmap = heightmap.unsqueeze(1)

        cnn_out = self.cnn(heightmap)
        spatial_coords = self.spatial_softmax(cnn_out)

        combined = torch.cat([spatial_coords, features], dim=-1)
        latent = self.mlp(combined)

        pi = F.softmax(self.pi_head(latent), dim=-1)

        mu = self.mu_head(latent)
        mu = mu.view(-1, self.num_mixtures, self.action_dim)
        mu = torch.tanh(mu)

        log_std = self.log_std_head(latent)
        log_std = torch.clamp(log_std, min=-4.0, max=2.0)
        log_std = log_std.view(-1, self.num_mixtures, self.action_dim)
        sigma = torch.exp(log_std)

        return pi, mu, sigma

    def sample_action(self, heightmap, features):
        """Samples an action and correctly computes the GMM Log Probability."""
        pi, mu, sigma = self.forward(heightmap, features)

        categorical = torch.distributions.Categorical(probs=pi)
        mixture_idx = categorical.sample()

        B = pi.shape[0]
        batch_indices = torch.arange(B)
        chosen_mu = mu[batch_indices, mixture_idx, :]
        chosen_sigma = sigma[batch_indices, mixture_idx, :]

        normal_dist = torch.distributions.Normal(chosen_mu, chosen_sigma)
        raw_action = normal_dist.rsample()
        action = torch.tanh(raw_action)

        raw_action_expanded = raw_action.unsqueeze(1)

        all_dists = torch.distributions.Normal(mu, sigma)

        log_probs_all = all_dists.log_prob(raw_action_expanded).sum(dim=-1)

        log_pi = torch.log(pi + 1e-8)

        full_gmm_log_prob = torch.logsumexp(log_pi + log_probs_all, dim=1)

        action = torch.tanh(raw_action)
        squash_correction = torch.log(1 - action**2 + 1e-6).sum(dim=-1)

        final_log_prob = full_gmm_log_prob - squash_correction

        return action, final_log_prob.unsqueeze(1)

class TwinContinuousCritic(nn.Module):
    """SAC Twin Critics: Predicts the Q-value of a specific continuous action"""
    def __init__(self, feature_dim=8, map_size=30, action_dim=4):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 16, kernel_size=1)
        )
        self.spatial_softmax = SpatialSoftmax(height=map_size, width=map_size)

        fc_input_dim = (16 * 2) + feature_dim + action_dim

        self.mlp1 = nn.Sequential(
            nn.Linear(fc_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(fc_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, heightmap, features, action):
        if heightmap.dim() == 3:
            heightmap = heightmap.unsqueeze(1)

        cnn_out = self.cnn(heightmap)
        spatial_coords = self.spatial_softmax(cnn_out)

        combined = torch.cat([spatial_coords, features, action], dim=-1)

        q1 = self.mlp1(combined)
        q2 = self.mlp2(combined)

        return q1, q2