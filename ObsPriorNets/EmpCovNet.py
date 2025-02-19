import torch
from torch import nn
import pytorch_lightning as pl

class EmpCovNet(pl.LightningModule):
    """
    EmpCovNet is a simple network that:
    1. Processes an input distance matrix and point values per batch element.
    2. Learns to predict a single parameter (or scalar output) for each element in the batch.
    
    Each batch element is handled independently in a loop.
    """

    def __init__(self, k: int = 2):
        """
        Args:
            k (int): Downscaling factor for the hidden dimension.
        """
        super(EmpCovNet, self).__init__()
        # Embedding layers for distance matrix and point values
        self.embed_layer1 = nn.Sequential(
            nn.Linear(1, 16 // k),
            nn.ReLU(inplace=True),
            nn.Linear(16 // k, 32 // k),
        )
        self.embed_layer2 = nn.Sequential(
            nn.Linear(1, 16 // k),
            nn.ReLU(inplace=True),
            nn.Linear(16 // k, 32 // k),
        )

        # Combined estimator for final scalar prediction
        self.clen_estimator = nn.Sequential(
            nn.Linear(64 // k, 32 // k),
            nn.ReLU(inplace=True),
            nn.Linear(32 // k, 16 // k),
            nn.ReLU(inplace=True),
            nn.Linear(16 // k, 1),
            nn.Sigmoid()
        )

        # Keep track of training losses
        self.train_loss_values = []

    def forward(self, distance_matrix: torch.Tensor, point_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that processes each item in the batch independently.

        Args:
            distance_matrix (torch.Tensor): A list (or batch) of distance matrices. 
                                            Each entry i has shape (n_i, n_i).
            point_values (torch.Tensor): A list (or batch) of point value arrays. 
                                         Each entry i has shape (n_i,).

        Returns:
            torch.Tensor: Stacked scalar predictions for each batch entry, shape (batch_size, 1, 1).
        """
        batch_size = len(distance_matrix)
        predictions = []

        for i in range(batch_size):
            single_distance_matrix = distance_matrix[i]  # (n, n)
            single_point_values = point_values[i]        # (n,)
            n = single_distance_matrix.shape[0]

            # Flatten the distance matrix and embed to higher dimensions
            x1 = single_distance_matrix.view(n * n, 1)
            x1 = self.embed_layer1(x1).view(n * n, 1, -1)
            # Mean pooling over the sequence dimension
            x1 = x1.mean(dim=0)  # shape => (1, hidden_dim)

            # Flatten the point values and embed
            x2 = single_point_values.view(n, 1)
            x2 = self.embed_layer2(x2).view(n, 1, -1)
            # Mean pooling over the sequence dimension
            x2 = x2.mean(dim=0)  # shape => (1, hidden_dim)

            # Concatenate
            x = torch.cat([x1, x2], dim=1)  # shape => (1, combined_dim)

            # Final fully connected prediction
            pred = self.clen_estimator(x)   # shape => (1, 1)
            predictions.append(pred)

        # Stack predictions across the batch: shape => (batch_size, 1, 1)
        return torch.stack(predictions, dim=0)

    def training_step(self, batch, batch_idx):
        """
        In each training step, compute the prediction for each item in the batch
        and accumulate MSE loss.
        """
        distance_matrices, point_values, truth_params = batch
        losses = []
        
        for i in range(len(distance_matrices)):
            distance_matrix = distance_matrices[i].unsqueeze(0)
            point_value = point_values[i].unsqueeze(0)
            truth_param = truth_params[i].unsqueeze(0)

            predicted_params = self(distance_matrix, point_value)
            loss = nn.MSELoss()(predicted_params, truth_param)
            losses.append(loss)
        
        avg_loss = torch.stack(losses).mean()
        self.train_loss_values.append(avg_loss.item())
        self.log('train_loss', avg_loss)
        return avg_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        """
        Prediction step used in inference. 
        Note that we don't have ground truth params here.
        """
        distance_matrices, point_values, _ = batch
        predicted_params = self(distance_matrices, point_values)
        return predicted_params

    def configure_optimizers(self):
        """
        Configures the optimizer for training.
        """
        return torch.optim.Adam(self.parameters(), lr=0.001)

