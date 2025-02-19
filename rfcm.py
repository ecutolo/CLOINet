import math
import torch
from torch.nn import functional as F
from torch import nn


class RFCM_loss(nn.Module):
    """
    RFCM_loss implements the Robust Fuzzy C-Means loss as described in:
    'Learning Fuzzy Clustering for SPECT/CT Segmentation via Convolutional Neural Networks.'
    by Junyu Chen, et al. (2021).

    It is an unsupervised method that doesn't require ground truth labels but uses the original
    image itself to guide cluster assignments. The fuzzy_factor and regularizer_wt control
    the fuzziness and a neighborhood smoothness prior, respectively.
    """
    def __init__(self, fuzzy_factor=2, regularizer_wt=0.0008):
        """
        Args:
            fuzzy_factor (float): Exponent for controlling fuzzy overlap.
            regularizer_wt (float): Weighting factor for the regularization term.
        """
        super().__init__()
        self.fuzzy_factor = fuzzy_factor
        self.wt = regularizer_wt

    def forward(self, y_pred: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """
        Computes the RFCM loss given predicted membership maps and the input images.

        Args:
            y_pred (torch.Tensor): Shape (B, C, ..., W, H) after softmax. Membership probabilities.
            image (torch.Tensor): Shape (B, 1, ..., W, H). The original image or slice.

        Returns:
            torch.Tensor: Scalar RFCM loss value.
        """
        dim = len(list(y_pred.shape)[2:])
        assert dim in (2, 3), 'Only 2D or 3D data is supported by this RFCM implementation.'

        # Number of clusters
        num_clus = y_pred.shape[1]
        # Flatten membership probabilities => (B, C, V)
        pred = y_pred.reshape(y_pred.shape[0], num_clus, math.prod(y_pred.shape[2:]))
        # Flatten images => (B, V)
        img = image.reshape(image.shape[0], math.prod(image.shape[2:]))

        # Create a kernel for local neighbor smoothing
        if dim == 3:
            # 3D kernel
            kernel = torch.ones((1, 1, 3, 3, 3), dtype=torch.float, device=y_pred.device)
            kernel[:, :, 1, 1, 1] = 0
        else:
            # 2D kernel
            kernel = torch.ones((1, 1, 3, 3), dtype=torch.float, device=y_pred.device)
            kernel[:, :, 1, 1] = 0

        J_1 = 0.0  # Intra-cluster variance term
        J_2 = 0.0  # Neighborhood regularization term

        # Compute for each cluster
        for i in range(num_clus):
            # Membership for cluster i, exponentiated by fuzzy_factor
            mem = pred[:, i, ...] ** self.fuzzy_factor  # shape (B, V)

            # Compute cluster center v_k
            numerator = torch.sum(img * mem, dim=1, keepdim=True)
            denominator = torch.sum(mem, dim=1, keepdim=True)
            v_k = numerator / denominator  # shape (B, 1)

            # Part 1 of the objective: (img - v_k)^2 weighted by membership
            J_1 += mem * (img - v_k) ** 2

            # Part 2: neighborhood penalty term
            # For each cluster i, sum membership from other clusters j => convolved
            mem_i = mem.reshape(image.shape)
            J_in = 0.0
            for j in range(num_clus):
                if i == j:
                    continue
                mem_j = (pred[:, j, ...] ** self.fuzzy_factor).reshape(image.shape)
                if dim == 3:
                    res = F.conv3d(mem_j, kernel, padding=1)  # 3D
                else:
                    res = F.conv2d(mem_j, kernel, padding=1)  # 2D
                res = res.reshape(res.shape[0], -1)  # flatten
                J_in += res
            # Weighted by mem_i
            J_2 += mem_i.reshape(mem_i.shape[0], -1) * J_in

        # Combine and average out
        loss_val = torch.mean(J_1) + self.wt * torch.mean(J_2)
        return loss_val

