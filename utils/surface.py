from typing import Optional
import torch

def compute_surface_from_points(points: torch.Tensor, 
                                poly_point_incidence: torch.Tensor, 
                                vertex_poly_incidence: torch.Tensor=None,
                                eps: float=1e-6,
                                thresh: Optional[float]=None,
                                strategy=None) -> dict:
    """
    A fully-differentiable function to compute polygon centroids and normals from a point cloud and polygon-point incidence matrix.
    Optionally computes vertex coordinates from vertex-polygon incidence matrix.

    Required inputs:
        - points: coordinates of surface point cloud (B, N, 3)
        - poly_point_incidence: polygon-point incidence plus indicator column (B, K, N+1)
    Optional inputs:
        - vertex_poly_incidence: vertex-polygon incidence plus indicator column (B, M, K+1)
        - eps: numerical stability
        - thresh: if provided, thresholds the incidence matrices to binary values
    Returns dict with 2-3 tensors:
        centroids: centroids of each polygon (B, K, 3)
        normals: normals of each polygon (B, K, 3)
        vertices: best-fit coordinates of intersections of polygons (B, M, 3)
    """

    ### Address padding
    points = torch.nan_to_num(points, nan=0.0)

    if thresh is not None:
        poly_point_incidence = (poly_point_incidence > thresh).float()
        if vertex_poly_incidence is not None:
            vertex_poly_incidence = (vertex_poly_incidence > thresh).float()

    indicator = poly_point_incidence[:, :, -1:]  # (B, K, 1)
    incidence_matrix = poly_point_incidence[:, :, :-1].float()  # (B, K, N)

    B, K, N = incidence_matrix.shape

    weighted_coord_sums = torch.einsum('bkn,bnd->bkd', incidence_matrix, points)
    weight_sums = incidence_matrix.sum(dim=-1, keepdim=True).float() + eps
    centroids = weighted_coord_sums / weight_sums

    delta = points.unsqueeze(1) - centroids.unsqueeze(2) ### (B, K, N, 3)
    covariance = torch.einsum('bkn,bknd,bknl->bkdl', incidence_matrix, delta, delta)  # (B, K, 3, 3)
    
    ### Stability to take care of empty polygons
    M_eps = eps * torch.eye(3, device=covariance.device).view(1, 1, 3, 3)
    covariance = covariance + M_eps

    ### Smallest eigenvalue of covariance matrix corresponds to normal eigenvector
    _evalues, evectors = torch.linalg.eigh(covariance)  # (B, K, 3), (B, K, 3, 3)
    normals = evectors[:, :, :, 0]

    tries = 0
    while normals.isnan().any():
        ### Retry with larger eps
        new_eps = (10**tries) * eps
        print(f"Warning: NaN detected in normals computation. Trying with eps={new_eps:.1e}...")
        M_eps = new_eps * torch.eye(3, device=covariance.device).view(1, 1, 3, 3)
        covariance = covariance + M_eps
        _evalues, evectors = torch.linalg.eigh(covariance)  # (B, K, 3), (B, K, 3, 3)
        normals = evectors[:, :, :, 0]
        tries += 1
        if tries > 5:
            print("Error: Unable to compute normals without NaNs. Will set to zero.")
            where_nan = normals.isnan()
            normals[where_nan] = 0.0
            break

    ### Work out vertex coords
    if vertex_poly_incidence is not None:
        vertex_indicator = vertex_poly_incidence[:, :, -1:]  # (B, M, 1)
        vertex_incidence_matrix = vertex_poly_incidence[:, :, :-1].float()  # (B, M, K)

        B, M, K = vertex_incidence_matrix.shape

        n = normals.unsqueeze(1).expand(-1, M, -1, -1)  # (B, M, K, 3)
        c = centroids.unsqueeze(1).expand(-1, M, -1, -1)  # (B, M, K, 3)
        
        ### A matrix
        w = vertex_incidence_matrix.unsqueeze(-1) # (B, M, K, 1)
        A = torch.einsum('bmkd,bmkl->bmdl', w * n, n) # (B, M, 3, 3)

        M_eps = eps * torch.eye(3, device=A.device).view(1, 1, 3, 3)
        A = A + M_eps

        ### b vector
        n_dot_c = (n * c).sum(dim=-1)  # (B, M, K)
        b = torch.einsum('bmk,bmk,bmkd->bmd', vertex_incidence_matrix, n_dot_c, n)

        ### Type matching
        target_dtype = points.dtype
        A = A.to(target_dtype)
        b = b.to(target_dtype)

        ### Solve linear system A*v = b
        if strategy is None or strategy == 'lstsq':
            vertex_coords = torch.linalg.lstsq(A, b).solution  # (B, M, 3)
        elif strategy == 'solve':
            vertex_coords = torch.linalg.solve(A, b)  # (B, M, 3)

        ### Mask
        vertices = vertex_coords * vertex_indicator
    else:
        vertices = None

    ### Mask out centroids and normals for empty polygons
    centroids = centroids * indicator
    normals = normals * indicator

    if thresh is None:
        thresh = 0.5

    return {
        'centroids': centroids,
        'normals': normals,
        'vertices': vertices,
        'polygon_mask': (indicator.squeeze(-1) > thresh),
        'vertex_mask': (vertex_indicator.squeeze(-1) > thresh) if vertex_poly_incidence is not None else None
    }