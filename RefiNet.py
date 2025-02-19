import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F


class RandomChannelShuffle(nn.Module):
    """
    Applies a random shuffle (zero-out) to channels in the training phase
    with probability p. This can act as a form of regularization.
    """
    def __init__(self, p=0.5):
        super(RandomChannelShuffle, self).__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            batch_size, num_channels, height, width = x.size()
            # Generate random binary mask for each channel in the batch
            mask = torch.bernoulli(torch.full((batch_size, num_channels), self.p)).to(x.device)
            # Zero out channels randomly
            shuffled_x = x * mask.unsqueeze(-1).unsqueeze(-1)
            return shuffled_x
        else:
            return x


class RefiNet(pl.LightningModule):
    """
    RefiNet refines combined cluster memberships from different sources (surface and observation
    clusters) into a smaller set of refined clusters. It also calculates an entropy-based loss
    to encourage discrete cluster assignments.
    """

    def __init__(self, proposed_clusters_n: int, ref_clusters_n: int, k=4):
        """
        Args:
            proposed_clusters_n (int): Number of initial combined clusters from different inputs.
            ref_clusters_n (int): Number of refined clusters.
            k (int): Downscaling factor for convolution filter size.
        """
        super().__init__()
        self.shuffle_layer = RandomChannelShuffle(p=0.5)
        self.features = nn.Sequential(
            nn.Conv2d(proposed_clusters_n, 64 // k, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64 // k, 128 // k, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128 // k, ref_clusters_n, kernel_size=1)
        )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, proposed_clusters: torch.Tensor) -> torch.Tensor:
        """
        Args:
            proposed_clusters (torch.Tensor): Shape (B, C, W, H), i.e. combined cluster memberships.

        Returns:
            torch.Tensor: Shape (B, ref_clusters_n, W, H), refined cluster memberships.
        """
        # Apply random channel shuffle for regularization
        x = self.shuffle_layer(proposed_clusters)
        # Pass through convolutional layers
        refined_clusters = self.features(x)
        # Convert to probability-like distribution per cluster
        refined_clusters = self.softmax(refined_clusters)
        return refined_clusters
    
    def loss(self, ref_clusters: torch.Tensor) -> torch.Tensor:
        """
        Computes entropy-based loss over refined clusters to encourage "sharper" distributions.

        Args:
            ref_clusters (torch.Tensor): Shape (B, [D,] ref_clusters_n, W, H),
                                         depending on how you pass it in.

        Returns:
            torch.Tensor: Scalar mean entropy loss.
        """
        # Negative sum(p*log p) is the standard entropy measure
        ref_entropy = (-ref_clusters * torch.log(ref_clusters)).sum(dim=-3)
        return ref_entropy.mean()

