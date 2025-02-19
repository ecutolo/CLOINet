import pytorch_lightning as pl
import torch
from torch import nn

from .utils import *
from .rfcm import RFCM_loss


class CluNet(pl.LightningModule):
    """
    CluNet is a clustering network:
      - It uses convolutional layers to extract features
      - Produces cluster membership probabilities via a softmax
      - Uses an RFCM-based loss for unsupervised clustering
    """

    def __init__(self, input_fields_n=1, clusters_n=30, learning_rate=0.001, k=8):
        """
        Args:
            input_fields_n (int): Number of channels in the input feature map.
            clusters_n (int): Number of clusters to output.
            learning_rate (float): Learning rate for the network.
            k (int): Downscaling factor for the number of filters.
        """
        super().__init__()
        self.lr = learning_rate
        
        self.features = nn.Sequential(
            nn.Conv2d(input_fields_n, 64 // k, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64 // k, 128 // k, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128 // k, clusters_n, kernel_size=1)
        )
        
        self.RFCM_loss = RFCM_loss()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that outputs cluster membership probabilities.

        Args:
            x (torch.Tensor): Shape (B, input_fields_n, W, H).
        
        Returns:
            torch.Tensor: Shape (B, clusters_n, W, H), membership probabilities.
        """
        x = self.features(x)
        return self.softmax(x)

    def loss(self, fields: torch.Tensor, clusters: torch.Tensor) -> torch.Tensor:
        """
        Computes the RFCM loss across each field slice in `fields`.

        Args:
            fields (torch.Tensor): Input fields, shape (B, S, W, H).
            clusters (torch.Tensor): Cluster membership map, shape (B, C, W, H).

        Returns:
            torch.Tensor: Scalar RFCM loss.
        """
        total_loss = 0
        # fields.transpose(0,1) => shape (S, B, W, H)
        # We compute the RFCM loss for each 'field' individually
        for field in fields.transpose(0, 1):
            total_loss += self.RFCM_loss(clusters, field.unsqueeze(1))
        return total_loss
    
    def training_step(self, batch, batch_idx):
        # An example training step if used stand-alone
        x, _ = batch  # adjust if there's a target
        clusters = self(x)
        loss_val = self.loss(x, clusters)
        return loss_val
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

