import torch
import numpy as np

def efficient_jensen_shannon_divergence(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    Computes the pairwise Jensen-Shannon Divergence (JSD) between two tensors x1 and x2.
    The inputs are expected to have shape (N, D) and (M, D), respectively.

    Args:
        x1 (torch.Tensor): Tensor of shape (N, D).
        x2 (torch.Tensor): Tensor of shape (M, D).

    Returns:
        torch.Tensor: Pairwise JSD matrix of shape (N, M).
    """
    # Compute average distributions
    m = 0.5 * (x1.unsqueeze(1) + x2.unsqueeze(0))

    # Compute KL divergence for all pairs
    kl_x1_m = torch.sum(x1.unsqueeze(1) * torch.log(x1.unsqueeze(1) / m), dim=-1)
    kl_x2_m = torch.sum(x2.unsqueeze(0) * torch.log(x2.unsqueeze(0) / m), dim=-1)

    # Compute Jensen-Shannon Divergence
    jsd = 0.5 * kl_x1_m + 0.5 * kl_x2_m
    return jsd

def fill_zeros_with_repeats(x: torch.Tensor, P: int = 30) -> torch.Tensor:
    """
    Fills zero columns with repeated data from the first non-zero columns.

    Args:
        x (torch.Tensor): Shape (B, S, E). Possibly containing zero columns.
        P (int): Desired number of columns to fill.

    Returns:
        torch.Tensor: Result of shape (B, P, E) with repeated data in place of zeros.
    """
    B, S, E = x.shape
    new_x = torch.zeros((B, P, E), device=x.device)

    for b in range(B):
        # Identify columns where x[b, :, 2] != 0 as non-zero
        non_zero_cols = x[b][x[b, :, 2] != 0]
        if non_zero_cols.numel() == 0:
            # If all columns are zero, keep them zero (or handle differently as needed)
            continue
        # Calculate the number of repeats needed to fill up to P
        num_repeats = (P // non_zero_cols.shape[0]) + 1
        repeated_sequence = non_zero_cols.repeat(num_repeats, 1)
        new_x[b] = repeated_sequence[:P, :]

    return new_x


@torch.jit.script
def my_cdist(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    Custom pairwise distance function similar to torch.cdist, computing Euclidean distance
    in a more manual way to allow for JIT scripting.

    Args:
        x1 (torch.Tensor): Shape (N, D).
        x2 (torch.Tensor): Shape (M, D).

    Returns:
        torch.Tensor: Distances of shape (N, M).
    """
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
    # Clamp for numerical stability, then sqrt
    res = res.clamp_min_(1e-30).sqrt_()
    return res


@torch.jit.script
def rot_mat(phi: torch.Tensor) -> torch.Tensor:
    """
    Creates a 2D rotation matrix for each value in phi (in radians).
    Phi is first multiplied by 3.14 (approx pi), which suggests it's scaling degrees to radians,
    or some custom factor.

    Args:
        phi (torch.Tensor): Scalar or tensor of angles.

    Returns:
        torch.Tensor: Rotation matrix of shape (2, 2, ...) depending on the shape of phi.
    """
    phi = phi * 3.14
    rot = torch.stack([
        torch.stack([torch.cos(phi), -torch.sin(phi)]),
        torch.stack([torch.sin(phi),  torch.cos(phi)])
    ])
    return rot


def kmeans(X: torch.Tensor, n_clusters: int, device=0, tol=1e-4):
    """
    A simple PyTorch-based KMeans implementation with random initialization.

    Args:
        X (torch.Tensor): Data of shape (N, D).
        n_clusters (int): Number of clusters to form.
        device (int): Unused here unless you want to specify device placement.
        tol (float): Tolerance to stop the iterative updates.

    Returns:
        tuple: (choice_cluster, centers, std_dev)
            choice_cluster (torch.Tensor): Shape (N,), cluster assignment.
            centers (torch.Tensor): Shape (n_clusters, D), cluster centers.
            std_dev (torch.Tensor): Shape (n_clusters, D), max deviation in each cluster dimension.
    """
    num_points = len(X)
    indices = torch.randperm(num_points)[:n_clusters]
    centers = X[indices].clone()
    stds = torch.zeros_like(centers)

    while True:
        # Distances to cluster centers
        dist = my_cdist(X, centers)
        # Assign points to nearest cluster
        choice_cluster = torch.argmin(dist, dim=1)
        old_centers = centers.clone()

        # Update centers
        for idx in range(n_clusters):
            members = (choice_cluster == idx).nonzero(as_tuple=True)[0]
            selected = X[members] if len(members) > 0 else None
            if selected is None or selected.shape[0] == 0:
                # If a cluster gets no members, re-sample a center at random
                selected = X[torch.randint(num_points, (1,))]
            centers[idx] = selected.mean(dim=0)

        # Check for convergence
        center_shift = (centers - old_centers).pow(2).sum(dim=1).sqrt().sum()
        if center_shift.pow(2) < tol:
            break

    # Compute max deviations in each cluster
    for idx in range(n_clusters):
        members = (choice_cluster == idx).nonzero(as_tuple=True)[0]
        if len(members) > 0:
            selected = X[members]
            stds[idx] = (selected - centers[idx]).abs().max(axis=0).values

    return choice_cluster, centers, stds


def mask2D_to_3D(mask: torch.Tensor, dev: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    Converts a 2D mask into a 3D sparse matrix representation suitable for interpolation.

    Args:
        mask (torch.Tensor): Shape (W, H), contains 1 or 0 entries.
        dev (torch.device): Target device for the output tensor.

    Returns:
        torch.Tensor: Dense tensor of shape (M, N, O) with O=number of ones in the mask.
    """
    O = int(torch.sum(mask))
    W, H = mask.shape
    # Coordinates of the nonzero entries: (y, x)
    coords = torch.nonzero(mask, as_tuple=False)
    # Re-arrange coords => (x, y, index)
    mask_coord = torch.stack([coords[:, 1], coords[:, 0], torch.arange(O, device=dev)])
    # Sparse assignment
    mask3D = torch.sparse_coo_tensor(
        mask_coord, torch.ones(O, device=dev),
        (H, W, O)
    )
    return mask3D.to_dense()


def repeat_to_size(tensor: torch.Tensor, dimension: int, target_size: int) -> torch.Tensor:
    """
    Repeats the input tensor along a specified dimension until it reaches the target size.

    Args:
        tensor (torch.Tensor): The input tensor.
        dimension (int): The dimension along which to repeat.
        target_size (int): The target size along the specified dimension.

    Returns:
        torch.Tensor: The repeated tensor trimmed to `target_size`.
    """
    assert target_size >= tensor.size(dimension), (
        "Target size must be larger or equal to the input size along the specified dimension."
    )
    # Calculate repetition factor
    rep_factor = (target_size + tensor.size(dimension) - 1) // tensor.size(dimension)
    repeated_tensor = torch.cat([tensor] * rep_factor, dim=dimension)
    # Slice the tensor down to the target size
    index = [slice(None)] * tensor.dim()
    index[dimension] = slice(target_size)
    repeated_tensor = repeated_tensor[tuple(index)]
    return repeated_tensor

