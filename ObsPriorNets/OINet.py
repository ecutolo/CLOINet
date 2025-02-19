import torch
from torch import nn
import pytorch_lightning as pl
from .EmpCovNet import EmpCovNet

class OIPrior(pl.LightningModule):
    """
    OIPrior composes multiple OINet models to build a prior field. 
    Each OINet can interpolate data differently and produce a result.
    """

    def __init__(self, grid_size, oi_fields_n=1, lr=0.001):
        """
        Args:
            grid_size (tuple): The (W, H) shape, plus a third dimension for modeling (W, H, D).
            oi_fields_n (int): Number of OINet models to stack. 
            lr (float): Learning rate, if needed.
        """
        super().__init__()
        self.prior_fields_n = oi_fields_n  # The total channels produced
        # Expand grid_size to 3D by appending (3,)
        grid_size = grid_size + (3,)

        # Create multiple OINet models
        self.oi_models = nn.ModuleList([OINet(grid_size) for _ in range(oi_fields_n)])
        
        # Hard-coded examples of fixed correlation lengths in each submodel
        # (User-specific logic)
        self.oi_models[0].fixed_clen = torch.tensor(grid_size) / 2
        self.oi_models[1].fixed_clen = torch.tensor(grid_size) / 4
        self.oi_models[2].fixed_clen = torch.tensor(grid_size) / 8
        self.oi_models[3].fixed_clen = torch.tensor(grid_size) / 16

    def forward(self, obs: torch.Tensor, masks: torch.Tensor):
        """
        Forward pass that runs each OINet and concatenates the results.

        Args:
            obs (torch.Tensor): shape (B, D, W, H).
            masks (torch.Tensor): shape (B, 1, W, H).

        Returns:
            preds (torch.Tensor): shape (B*D, oi_fields_n, W, H) 
            clen (Any): correlation length or related param from the last OINet call.
        """
        B, D, W, H = obs.shape
        masks = masks[:, 0:1]  # Possibly picking only the first channel

        # Collect each submodel's prediction
        preds_list = []
        clen = None
        for model in self.oi_models:
            pred, clen = model(obs, masks)
            preds_list.append(pred.reshape(B * D, 1, W, H))

        # Stack all submodel predictions
        preds = torch.cat(preds_list, dim=1)
        return preds, clen

    def loss(self, *params):
        """
        Override any additional losses if needed. Currently returns 0.
        """
        return 0

class OINet(pl.LightningModule):
    """
    OINet is a 3D interpolation network using exponential correlation (or Gaussian kernel) 
    to interpolate from known observations.
    """

    def __init__(
        self, 
        grid_size,
        clen_model=False, 
        irregular_z=False, 
        surface_fields_n=0, 
        lr=0.01, 
        autoz=False
    ):
        """
        Args:
            grid_size (tuple): (W, H, D) dimension for the 3D field.
            clen_model (bool or nn.Module): If not bool, used to learn correlation length from data.
            irregular_z (bool or torch.Tensor): If True/False or a custom z-vector for the 3rd dimension.
            surface_fields_n (int): Additional surface field channels that may be processed via a small conv net.
            lr (float): Learning rate.
            autoz (bool): Whether to automatically set z correlation length or keep it fixed.
        """
        super().__init__()
        self.grid_size = grid_size
        self.srf_fields_n = surface_fields_n
        self.clen_model = clen_model
        self.clen_scaling = nn.Parameter(torch.ones(3))
        self.clen_norm = nn.Sigmoid()  # normalizes the correlation length

        self.autoz = autoz
        self.loss = nn.L1Loss()
        self.lr = lr
        
        # The 3D coordinates
        N, M, D = self.grid_size
        x = torch.linspace(0, N, N)
        y = torch.linspace(0, M, M)
        if isinstance(irregular_z, bool):
            z = torch.linspace(0, D, D)
        else:
            z = irregular_z

        # If we have surface fields, create a conv net to reduce them to a single channel
        if surface_fields_n > 0:
            D += 1  # extra dimension from surface fields
            z = torch.cat((torch.tensor([0.0]), z))  # example logic
            self.srf_preproc = nn.Sequential(
                nn.Conv2d(surface_fields_n, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, kernel_size=1)
            )

        # Create a grid of points => shape (N, M, D, 3)
        xx, yy, zz = torch.meshgrid(x, y, z)
        self.grid_points = torch.stack((xx, yy, zz), dim=-1)

    def get_mask_tensor(self, mask: torch.Tensor, dev: torch.device = torch.device('cpu')):
        """
        Converts a 3D mask (I, J, K) => dense representation (I, J, K, O),
        where O is the count of nonzero mask elements.
        """
        O = int(torch.sum(mask))
        I, J, K = mask.shape
        coords = torch.nonzero(mask, as_tuple=False)
        # coords => shape (O, 3) => (x, y, z)
        mask_coord = torch.stack([
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            torch.arange(O, device=dev)
        ])
        mask3D = torch.sparse_coo_tensor(mask_coord, 
                                         torch.ones(O, device=dev),
                                         (I, J, K, O))
        return mask3D.to_dense()

    def forward(self, obs: torch.Tensor, masks: torch.Tensor, surface_fields: torch.Tensor = None, eps: float = 0.001):
        """
        Forward pass that first optionally processes surface fields,
        then interpolates the observations in 3D using an exponential kernel.

        Args:
            obs (torch.Tensor): shape (B, D, W, H).
            masks (torch.Tensor): shape (B, D, W, H).
            surface_fields (torch.Tensor): optional surface data shape (B, surface_fields_n, W, H).
            eps (float): small value added to diagonal of covariance for numerical stability.
        
        Returns:
            pred (torch.Tensor): shape (B, D, W, H) final interpolation result.
            clen (torch.Tensor): correlation length used for interpolation.
        """
        B, D, W, H = obs.shape
        
        # Expand masks across each channel dimension if needed
        masks = masks.repeat(1, D, 1, 1)
        obs = torch.where(masks.bool(), obs, torch.zeros_like(obs))

        # If we have surface fields, process them into an extra channel
        if self.srf_fields_n > 0 and surface_fields is not None:
            surface_fields_conv = self.srf_preproc(surface_fields)
            D += 1
            s = 2  # sampling stride for surface
            surface_fields_mask = torch.zeros((B, 1, W, H), device=obs.device)
            surface_fields_mask[:, :, ::s, ::s] = 1  # downsampled
            surface_fields_processed = torch.zeros((B, 1, W, H), device=obs.device)
            surface_fields_processed[:, :, ::s, ::s] = surface_fields_conv[:, :, ::s, ::s]

            # Concatenate
            masks = torch.cat((surface_fields_mask, masks), dim=1)
            obs = torch.cat(
                (torch.where(surface_fields_mask.bool(), surface_fields_processed, torch.zeros_like(surface_fields_processed)), 
                 obs), 
                dim=1
            )

        # Allocate a container for the interpolated results
        pred = torch.ones((B, D, W, H), device=obs.device)
        
        # Interpolate each batch element separately
        for b_idx, b_masks, b_obs in zip(range(B), masks, obs):
            pred[b_idx], clen = self.interpolate(b_obs, b_masks, eps=eps)

        return pred, clen

    def interpolate(self, obs: torch.Tensor, mask: torch.Tensor, eps: float = 0.001):
        """
        Core interpolation logic. 
        Builds a correlation matrix among observed points, inverts it, 
        then multiplies by the known data to predict the entire 3D field.

        Args:
            obs (torch.Tensor): shape (D, W, H).
            mask (torch.Tensor): shape (D, W, H).
            eps (float): Diagonal stability term.
        
        Returns:
            pred (torch.Tensor): shape (W, H, D).
            clen (torch.Tensor): correlation length vector used in the kernel.
        """
        grid_points = self.grid_points.to(obs.device)
        mask_tensor = self.get_mask_tensor(mask, dev=obs.device)

        # Weighted sum of coordinates for points that are observed
        obs_points = torch.einsum('ijkp,kijo->op', grid_points, mask_tensor.float())
        # Observed data
        obs_values = torch.einsum('kij,kijo->o', obs, mask_tensor.float())

        # Distances among observed points
        obs_obs_dist = obs_points.unsqueeze(1) - obs_points.unsqueeze(0)
        # Distances from all grid points to the observed points
        all_obs_dist = grid_points.view(-1, 3).unsqueeze(1) - obs_points.unsqueeze(0)

        # Determine correlation lengths (clen)
        if isinstance(self.clen_model, bool):
            # If no custom model, use a fixed correlation length scaled by a sigmoid factor
            clen = self.clen_norm(self.clen_scaling) * self.fixed_clen.to(obs.device)
        else:
            # If there's a model, e.g. EmpCovNet, run it
            # (Custom logic for distance => correlation length)
            clen_list = []
            for i in range(3):
                dist_i = obs_obs_dist[:, :, i]
                dist_i_norm = dist_i / (dist_i.max() + 1e-10)
                predicted = self.clen_model(dist_i_norm.unsqueeze(0), 
                                            obs_values.unsqueeze(0)) * dist_i.max()
                clen_list.append(predicted)
            clen = torch.tensor([c.item() for c in clen_list], device=obs.device)
            if not self.autoz:
                clen[2] = obs_obs_dist[:, :, 2].max()
            clen = clen * self.clen_scaling

        # Exponential kernel among observed points
        cov_obs_obs = torch.exp(-1.0 * (obs_obs_dist**2 / (2 * clen**2)).sum(dim=-1))
        cov_obs_obs += torch.diag(torch.ones(cov_obs_obs.shape[0], device=obs.device) * eps)
        cov_obs_obs_inv = torch.linalg.inv(cov_obs_obs)

        # Exponential kernel from all points to observed points
        cov_obs_all = torch.exp(-1.0 * (all_obs_dist**2 / (2 * clen**2)).sum(dim=-1))
        cov_obs_all = cov_obs_all.view(self.grid_points.shape[:-1] + obs_points.shape[:-1])  # => (W, H, D, #obs)

        # Interpolation: pred = C(all,obs) * C(obs,obs)^{-1} * obs_values
        pred_3d = torch.einsum('ijko,qo,q->kij', cov_obs_all, cov_obs_obs_inv, obs_values)
        return pred_3d, clen

    def predict_step(self, batch, batch_idx):
        """
        Example predict step if you have a data batch of (surface_fields, obs_masks, target).
        """
        surface_fields, obs_masks, target = batch
        pred, clen = self.forward(target, obs_masks, surface_fields=surface_fields)
        # Omit the surface channel if used => skip_surface
        skip_surface = int(self.srf_fields_n > 0)
        return pred[:, skip_surface:], clen

    def training_step(self, batch, batch_idx):
        surface_fields, obs_masks, target = batch
        pred, clen = self.forward(target, obs_masks, surface_fields=surface_fields)
        skip_surface = int(self.srf_fields_n > 0)
        loss_val = self.loss(pred[:, skip_surface:], target)
        self.log_dict({"L1_train_loss": loss_val})
        return loss_val

    def validation_step(self, batch, batch_idx):
        surface_fields, obs_masks, target = batch
        pred, clen = self.forward(target, obs_masks, surface_fields=surface_fields)
        skip_surface = int(self.srf_fields_n > 0)
        loss_val = self.loss(pred[:, skip_surface:], target)
        self.log_dict({"L1_val_loss": loss_val})
        return loss_val

    def configure_optimizers(self):
        """
        Set different learning rates for the surface preproc or correlation length model
        if they exist.
        """
        if self.srf_fields_n > 0:
            optimizer = torch.optim.Adam([
                {'params': self.srf_preproc.parameters(), 'lr': self.lr},
                {'params': self.clen_model.parameters(), 'lr': 0.01}
            ])
        else:
            optimizer = torch.optim.Adam([{'params': self.parameters(), 'lr': 0.01}])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }

