import pytorch_lightning as pl
import torch
from torch import nn

from .utils import mask2D_to_3D, my_cdist
from .ObsPriorNets import ObsPriorNet  # Assumed import, adjust if named differently
from .CluNet import CluNet
from .RefiNet import RefiNet

class CLOINet(pl.LightningModule):
    """
    CLOINet is a PyTorch Lightning module that coordinates:
      1) Clustering of surface fields (srf_clustering_net).
      2) Clustering of observed/prior fields (obs_clustering_net).
      3) Refinement of combined clusters (clusters_refiner_net).
      4) Final interpolation to predict outputs.

    It also computes several losses:
      - L1 loss
      - Loss terms associated with the clustering (RFCM loss)
      - Refinement entropy loss
      - Observation prior network loss
    """

    def __init__(
        self, 
        obs_prior_net: nn.Module,
        srf_fields_n: int = 1,
        srf_clusters_n: int = 30,
        obs_clusters_n: int = 10,
        ref_clusters_n: int = 10, 
        learning_rate: float = 0.01
    ):
        """
        Initializes the CLOINet.

        Args:
            obs_prior_net (nn.Module): Network that takes the raw observational data
                                       and produces an initial prior (e.g. ObsPriorNet).
            srf_fields_n (int): Number of surface fields (input channels).
            srf_clusters_n (int): Number of clusters for surface fields.
            obs_clusters_n (int): Number of clusters for observation prior.
            ref_clusters_n (int): Number of refined clusters.
            learning_rate (float): Base learning rate for the optimizer.
        """
        super().__init__()
        self.params = {
            'OC': obs_clusters_n,
            'SF': srf_fields_n,
            'SC': srf_clusters_n,
            'RC': ref_clusters_n
        }
        self.lr = learning_rate
        
        # Networks
        self.obs_prior_net = obs_prior_net  # Produces an observation-based prior
        self.srf_clustering_net = CluNet(input_fields_n=srf_fields_n, clusters_n=srf_clusters_n)
        self.obs_clustering_net = CluNet(input_fields_n=self.obs_prior_net.prior_fields_n,
                                         clusters_n=obs_clusters_n)
        self.clusters_refiner_net = RefiNet(
            proposed_clusters_n=srf_clusters_n + obs_clusters_n,
            ref_clusters_n=ref_clusters_n
        )

        # Simple L1 loss for the final prediction
        self.loss = nn.L1Loss()

    def forward(self, surface_fields: torch.Tensor, masks: torch.Tensor, obs_fields: torch.Tensor):
        """
        Forward pass for the CLOINet. Computes:

        1) Observation prior.
        2) Surface field clusters.
        3) Observation clusters.
        4) Combined refined clusters.
        5) Final interpolation to produce predictions.

        Args:
            surface_fields (torch.Tensor): Shape (B, S, W, H), e.g. (batch, #surfaceFields, width, height).
            masks (torch.Tensor): Shape (B, 1, W, H) or (B, D, W, H) used to indicate observed data points.
            obs_fields (torch.Tensor): Shape (B, D, W, H), e.g. (batch, #obsChannels, width, height).

        Returns:
            pred (torch.Tensor): Interpolated predictions, shape (B, D, W, H).
            ref_clusters (torch.Tensor): Refined clusters, shape (B, D, RC, W, H).
            srf_clusters (torch.Tensor): Surface field clusters, shape (B, SC, W, H).
            obs_clusters (torch.Tensor): Observation prior clusters, shape (B*D, OC, W, H).
            obs_prior (torch.Tensor): Output of obs_prior_net, shape (B*D, something, W, H).
            obs_param (any): Additional parameters returned by obs_prior_net.
        """
        B, S, W, H = surface_fields.shape
        B, D, W, H = obs_fields.shape
        SC = self.params['SC']
        RC = self.params['RC']

        # Expand mask to match observed data dimension if necessary
        masks = masks.repeat(1, D, 1, 1)  # Now shape (B, D, W, H)

        # 1) Observation prior
        obs_prior, obs_param = self.obs_prior_net(obs_fields, masks)  # shape (B*D, X, W, H)

        # 2) Clustering for surface fields
        srf_clusters = self.srf_clustering_net(surface_fields)  # shape (B, SC, W, H)

        # 3) Clustering for observation prior
        obs_clusters = self.obs_clustering_net(obs_prior)  # shape (B*D, OC, W, H)

        # 4) Combine surface & observation clusters and refine
        combined_clusters = srf_clusters.unsqueeze(1).repeat(1, D, 1, 1, 1)
        combined_clusters = combined_clusters.reshape(B * D, SC, W, H)
        combined_clusters = torch.cat((combined_clusters, obs_clusters), dim=1)  # shape (B*D, SC+OC, W, H)

        ref_clusters = self.clusters_refiner_net(combined_clusters)
        ref_clusters = ref_clusters.reshape((B, D, RC, W, H))

        # 5) Interpolation/prediction
        pred = torch.ones((B, D, W, H), device=masks.device)
        for b_idx, b_clusters, b_masks, b_obs in zip(range(B), ref_clusters, masks, obs_fields):
            for d_idx, d_clusters, d_mask, d_obs in zip(range(D), b_clusters, b_masks, b_obs):
                # Build correlation matrix using Jensen-Shannon divergence
                cov2D = self.compute_correlation_matrix(d_clusters)
                # Interpolate using the correlation matrix and known observations
                pred[b_idx, d_idx] = self.interpolate(cov2D, d_obs.t(), d_mask)

        return pred, ref_clusters, srf_clusters, obs_clusters, obs_prior, obs_param
    
    def compute_correlation_matrix(self, clusters: torch.Tensor) -> torch.Tensor:
        """
        Builds a correlation (or similarity) matrix from a cluster probability map using
        Jensen-Shannon divergence.

        Args:
            clusters (torch.Tensor): Shape (C, W, H), cluster probability distributions.

        Returns:
            torch.Tensor: Pairwise similarity matrix (W*H, W*H).
        """
        C, W, H = clusters.shape
        # Flatten from (C, W, H) => (W*H, C)
        clusters_2d = clusters.permute(1, 2, 0).reshape((W * H, C))

        # Jensen-Shannon Divergence => efficient_jensen_shannon_divergence
        cov2D = efficient_jensen_shannon_divergence(clusters_2d, clusters_2d)
        return cov2D

    def interpolate(self, cov2D: torch.Tensor, obs: torch.Tensor, mask: torch.Tensor, eps: float = 0.001) -> torch.Tensor:
        """
        Performs interpolation of the observation data using the covariance (similarity) matrix.

        Args:
            cov2D (torch.Tensor): Shape (W*H, W*H), similarity matrix.
            obs (torch.Tensor): Shape (H, W), or transposed if needed.
            mask (torch.Tensor): Shape (W, H) with 1s for valid observations and 0s otherwise.
            eps (float): Small diagonal term to ensure invertibility.

        Returns:
            torch.Tensor: Interpolated 2D field of shape (W, H).
        """
        W_, H_ = mask.shape
        O = int(torch.sum(mask))

        # Convert the 2D mask into a 3D mask matrix used for interpolation
        mask3D = mask2D_to_3D(mask, dev=obs.device).reshape((W_ * H_, O))
        obs1D = obs.reshape((W_ * H_))

        # Covariance in observed indices + small identity for stability
        a2D = mask3D.t() @ cov2D @ mask3D + torch.diag(torch.ones(O, device=obs.device) * eps)
        ainv2D = torch.linalg.inv(a2D)

        # Interpolate: cov2D@mask3D@ainv2D@(obs1D@mask3D)
        pred1D = cov2D @ mask3D @ ainv2D @ (obs1D @ mask3D)
        return pred1D.reshape((H_, W_)).t()

    def compute_losses(self, indata, outdata):
        """
        Computes various losses for the model.

        Args:
            indata: A tuple (srf_fields, masks, target).
            outdata: A tuple (pred, ref_clusters, srf_clusters, obs_clusters, obs_prior, obs_prior_params).
        
        Returns:
            dict: Dictionary of scalar loss values.
        """
        srf_fields, masks, target = indata
        pred, ref_clusters, srf_clusters, obs_clusters, obs_prior, obs_prior_params = outdata    

        losses = {}
        # Loss from observation prior net
        losses['OBS_PRIOR'] = self.obs_prior_net.loss(*obs_prior_params)
        # Loss from observation clustering
        losses['OBS_RFCM'] = self.obs_clustering_net.loss(obs_prior, obs_clusters)
        # Loss from surface clustering
        losses['SRF_RFCM'] = self.srf_clustering_net.loss(srf_fields, srf_clusters)
        # Loss from cluster refiner
        losses['REFI'] = self.clusters_refiner_net.loss(ref_clusters)
        # L1 reconstruction loss
        losses['L1'] = self.loss(pred, target)
        # For convenience, partial L1 losses on first channel, etc.
        losses['L1_5'] = self.loss(pred[:, 0], target[:, 0])
        losses['L1_100'] = self.loss(pred[:, 1], target[:, 1])
        losses['L1_150'] = self.loss(pred[:, 2], target[:, 2])
        
        # Combined total
        losses['TOT'] = (
            0 * losses['REFI'] +
            1 * losses['OBS_PRIOR'] +
            1 * losses['OBS_RFCM'] +
            1 * losses['SRF_RFCM'] +
            1 * losses['L1']
        )
        return losses

    def training_step(self, batch, batch_idx):
        output = self.forward(*batch)
        losses = self.compute_losses(batch, output)
        self.log_dict({key + "_train_loss": loss for key, loss in losses.items()})
        return losses['TOT']

    def validation_step(self, batch, batch_idx):
        output = self.forward(*batch)
        losses = self.compute_losses(batch, output)
        self.log_dict({key + "_val_loss": loss for key, loss in losses.items()})
        return losses['TOT']

    def predict_step(self, batch, batch_idx):
        # Used for production/inference
        output = self.forward(*batch)
        return output

    def configure_optimizers(self):
        """
        Sets up optimizers and (optionally) learning rate schedulers.
        """
        optimizer = torch.optim.Adam([
            {'params': self.srf_clustering_net.parameters(), 'lr': 0.01},
            {'params': self.obs_clustering_net.parameters(), 'lr': 0.01},
            {'params': self.clusters_refiner_net.parameters(), 'lr': 0.001},
            {'params': self.obs_prior_net.parameters(), 'lr': 0.001}
        ])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }

    def save(self, filename: str):
        """
        Saves the state_dict of the model along with its parameters for future inference.
        """
        model_state = {
            "state_dict": self.state_dict(),
            "params": self.params
        }
        torch.save(model_state, filename)

