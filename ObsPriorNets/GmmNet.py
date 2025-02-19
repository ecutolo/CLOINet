import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn

class GmmNet(pl.LightningModule):
    """
    GmmNet implements a simple Gaussian Mixture Model approach, where each batch is processed
    to learn mixture weights, means, and (diagonal) covariances for a set of points.

    This can produce an output (obs_prior) which is a 2-channel map (mean and variance) 
    on a grid, as well as the GMM parameters.
    """

    def __init__(self, grid_size, num_gaussians=10, hidden_dim=128, gridded_obs=True):
        """
        Args:
            grid_size (tuple): (W, H) size for the output map.
            num_gaussians (int): Number of GMM components.
            hidden_dim (int): Dimensionality of the hidden layer for the transformer-like block.
            gridded_obs (bool): If True, uses an internal function to extract observation points 
                                from a mask; otherwise, it is assumed obs is already in point form.
        """
        super(GmmNet, self).__init__()
        self.num_gaussians = num_gaussians
        self.gridded_obs = gridded_obs 
        self.prior_fields_n = 2  # We'll produce a 2-channel output (mean, variance)
        self.grid_size = grid_size
        
        # Precompute a normalized grid of (x, y) positions
        W, H = self.grid_size
        w_coords, h_coords = torch.meshgrid(
            torch.arange(W) / W, 
            torch.arange(H) / H
        )
        pos = torch.stack((w_coords.flatten(), h_coords.flatten()), dim=1)
        self.pos = pos.repeat(1, 1, 1)  # May need adaptation if used for larger batch

        # Feature extraction
        input_dim = 3  # We'll embed x, y, and z (value) => shape=3
        self.feature_expansion = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Multi-head attention for set-based input
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4)
        
        # Networks predicting GMM parameters (means, covs, weights)
        self.fc_means = nn.Sequential(
            nn.Linear(hidden_dim, num_gaussians * input_dim),
            nn.ReLU(),
            nn.Linear(num_gaussians * input_dim, num_gaussians * input_dim),
            nn.Sigmoid()
        )
        
        self.fc_covs = nn.Sequential(
            nn.Linear(hidden_dim, num_gaussians * input_dim),
            nn.ReLU(),
            nn.Linear(num_gaussians * input_dim, num_gaussians * input_dim),
            nn.Sigmoid()
        )
        
        self.fc_weights = nn.Sequential(
            nn.Linear(hidden_dim, num_gaussians * input_dim),
            nn.ReLU(),
            nn.Linear(num_gaussians * input_dim, num_gaussians),
        )
        
        self.softmax = nn.Softmax(dim=-1)

    def extract_obs_points(self, target: torch.Tensor, masks: torch.Tensor, P: int = 30) -> torch.Tensor:
        """
        Extracts up to P observation points from each mask, normalizing the coordinates
        and the associated values to [0, 1].

        Args:
            target (torch.Tensor): shape (B, W, H)
            masks (torch.Tensor): shape (B, W, H) with 1 for valid obs, else 0
            P (int): number of repeated samples to ensure a consistent shape.

        Returns:
            torch.Tensor: shape (B, P, 3), with (x, y, value) in each row.
        """
        B, W, H = target.shape
        # For each batch element, gather coordinates and values where mask=1
        padded_obs = torch.zeros((B, P, 3), device=masks.device)
        
        for b in range(B):
            non_zero_coords = torch.nonzero(masks[b])  # shape (M,2)
            non_zero_values = target[b][non_zero_coords[:, 0], non_zero_coords[:, 1]]
            
            # Normalize the observation values
            val_min = non_zero_values.min()
            val_max = non_zero_values.max()
            denom = val_max - val_min if val_max != val_min else 1.0
            non_zero_values = (non_zero_values - val_min) / denom
            
            # Build (x, y, val)
            obs_points = torch.cat([
                non_zero_coords[:, 0].unsqueeze(-1) / W,
                non_zero_coords[:, 1].unsqueeze(-1) / H,
                non_zero_values.unsqueeze(-1)
            ], dim=1)
            
            # If fewer points than P, we tile them
            observations_n = obs_points.shape[0]
            if observations_n == 0:
                # If no nonzero coords, remain zeros (or handle differently if needed)
                continue
            num_repeats = (P // observations_n) + 1
            repeated_sequence = obs_points.repeat(num_repeats, 1)
            padded_obs[b] = repeated_sequence[:P]
        
        return padded_obs

    def forward(self, obs: torch.Tensor, masks: torch.Tensor = None):
        """
        Forward pass that converts either gridded observations (if self.gridded_obs == True)
        or direct point observations into GMM parameters, then produces mean/variance maps.

        Args:
            obs (torch.Tensor): shape (B, W, H) if gridded_obs=True, else (B, P, 3).
            masks (torch.Tensor): shape (B, W, H), 1 for valid obs, else 0 (optional).

        Returns:
            obs_prior (torch.Tensor): shape (B, 2, W, H). The 2 channels are the conditional
                                      mean and variance of the predicted field.
            gmm_params (tuple): A tuple containing:
                                (obs_points, weights, means, covs).
        """
        if self.gridded_obs:
            # Convert grid + mask => set of point observations
            obs_points = self.extract_obs_points(obs, masks)
        else:
            # If not gridded, assume obs is already shape (B, P, 3)
            obs_points = obs

        batch_size = obs_points.shape[0]
        W, H = self.grid_size

        # Transform obs_points => (P, B, input_dim) for attention
        x = obs_points.transpose(0, 1)  # shape => (P, B, 3)
        
        # Feature expansion
        x = self.feature_expansion(x)  # shape => (P, B, hidden_dim)
        
        # Multi-head attention
        attn_output, _ = self.attention(x, x, x)  # shape => (P, B, hidden_dim)
        
        # Pool across P dimension
        attn_output = attn_output.mean(dim=0)  # shape => (B, hidden_dim)

        # Predict GMM parameters
        means = self.fc_means(attn_output).view(batch_size, self.num_gaussians, 3)
        covs = self.fc_covs(attn_output).view(batch_size, self.num_gaussians, 3)
        weights = self.fc_weights(attn_output)  # shape => (B, num_gaussians)
        weights = self.softmax(weights)

        pos = self.pos.to(obs.device)
        
        # Compute mean, variance for each point in the (W,H) grid
        obs_map, var_map = self.compute_conditional_mean_variance(pos, weights, means, covs)
        var_map = var_map.reshape((batch_size, 1, W, H))
        obs_map = obs_map.reshape((batch_size, 1, W, H))
        
        # Combine into a 2-channel prior
        obs_prior = torch.cat((obs_map, var_map), dim=1)
        gmm_params = (obs_points, weights, means, covs)
        return obs_prior, gmm_params

    def compute_conditional_mean_variance(self, points_2d: torch.Tensor, 
                                          weights: torch.Tensor,
                                          means: torch.Tensor,
                                          covs_diag: torch.Tensor):
        """
        Compute conditional mean & variance along the z-dimension, given x,y coordinates.

        Args:
            points_2d (torch.Tensor): shape (batch_size, N, 2).
            weights (torch.Tensor): shape (batch_size, num_gaussians).
            means (torch.Tensor): shape (batch_size, num_gaussians, 3).
            covs_diag (torch.Tensor): shape (batch_size, num_gaussians, 3).
        
        Returns:
            conditional_mean (torch.Tensor): shape (batch_size, N).
            conditional_variance (torch.Tensor): shape (batch_size, N).
        """
        # Cov terms for x,y
        covs_xx_diag = covs_diag[:, :, :2] + 1e-10
        covs_zz_diag = covs_diag[:, :, 2] + 1e-10
        inv_covs_xx_diag = 1.0 / covs_xx_diag

        # points_2d => shape (B, N, 2)
        diff = points_2d.unsqueeze(2) - means[:, :, :2].unsqueeze(1)  # => (B, N, G, 2)

        # -1/2 * (diff^2 / Sigma_xx)
        exponent = -0.5 * torch.sum(diff**2 * inv_covs_xx_diag.unsqueeze(1), dim=-1) 
        # normalizer => 1 / (2*pi*sqrt(detSigma_xx))
        normalizer = 1.0 / (2 * torch.pi * torch.sqrt(torch.prod(covs_xx_diag, dim=-1) + 1e-10))
        gauss_pdf = normalizer.unsqueeze(1) * torch.exp(exponent)  # => (B, N, G)

        # Weighted responsibilities
        gamma = weights.unsqueeze(1) * gauss_pdf
        gamma /= gamma.sum(dim=-1, keepdim=True) + 1e-10

        # Compute mean along z
        means_z = means[:, :, 2]  # => (B, G)
        conditional_mean = torch.sum(gamma * means_z.unsqueeze(1), dim=-1)  # => (B, N)

        # Compute variance along z
        means_z_expanded = means_z.unsqueeze(1)        # (B, 1, G)
        conditional_mean_expanded = conditional_mean.unsqueeze(-1)  # (B, N, 1)
        covs_zz_diag_expanded = covs_zz_diag.unsqueeze(1)           # (B, 1, G)

        variance_term = covs_zz_diag_expanded + (means_z_expanded - conditional_mean_expanded)**2
        conditional_variance = torch.sum(gamma * variance_term, dim=-1)  # => (B, N)

        # Optional normalization fo

