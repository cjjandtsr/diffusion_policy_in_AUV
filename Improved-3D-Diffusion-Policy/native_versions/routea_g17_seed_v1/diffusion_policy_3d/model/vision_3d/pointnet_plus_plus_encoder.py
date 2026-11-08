"""
pointnet_plus_plus_encoder.py — PointNet++ with 2 Set Abstraction levels + distance-weighted dual pooling
==========================================================================================================
Replaces MultiStagePointNet (flat per-point MLP + global max-pooling) with hierarchical encoder.

Architecture:
  SA1: FPS(N→512) + BallQuery(r=1.0m, k=32) + PointNet(6→64→64→128)  # relative_xyz(3) + original_xyz(3)
  SA2: FPS(512→128) + BallQuery(r=4.0m, k=64) + PointNet(131→128→128→256)  # feat_sa1 + relative_xyz
  Global: Conv1d(256→256) + BN + ReLU + distance-weighted dual pooling (max+avg) + Linear(512→128)

Interface: forward(x[B, N, 3]) → global_feat[B, 128], fully compatible with MultiStagePointNetEncoder

Placement: diffusion_policy_3d/model/vision_3d/pointnet_plus_plus_encoder.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import cprint


# ───────── Farthest Point Sampling ─────────
# Prefer pytorch3d CUDA implementation (single kernel, <5ms for 4096→512).
# Install: pip install pytorch3d (requires matching PyTorch/CUDA version).
# Fallback to pure PyTorch if pytorch3d not available (~77ms on 4096→512).
_FPS_CUDA = False
try:
    from pytorch3d.ops import sample_farthest_points as _sample_farthest_points
    _FPS_CUDA = True
except ImportError:
    pass


def farthest_point_sample(xyz, npoint):
    """
    xyz: [B, N, 3]
    returns: [B, npoint] indices of sampled centroids

    Uses pytorch3d CUDA kernel when available (<5ms), otherwise pure PyTorch (~77ms).
    """
    if _FPS_CUDA:
        _, idx = _sample_farthest_points(xyz, K=npoint)
        return idx.long()

    # ── Pure PyTorch fallback ──
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.zeros(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(B, device=device), farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, dim=-1).indices

    return centroids


# ───────── Ball Query ─────────
def ball_query(xyz, centroids_xyz, radius, n_sample):
    """
    xyz: [B, N, 3]
    centroids_xyz: [B, npoint, 3]
    radius: float — search radius (meters, body frame)
    n_sample: int — max neighbors per centroid
    returns: [B, npoint, n_sample] indices of grouped points

    When fewer than n_sample neighbors fall within radius, out-of-radius positions
    are filled with the nearest neighbor (standard PointNet++ practice).
    This prevents pseudo-neighbors with large relative coordinate offsets from corrupting MLP input.
    """
    B, N, _ = xyz.shape
    npoint = centroids_xyz.shape[1]

    dist = torch.cdist(centroids_xyz, xyz, p=2)  # [B, npoint, N]

    # Take the n_sample nearest neighbors
    group_idx = dist.topk(k=n_sample, dim=-1, largest=False).indices  # [B, npoint, n_sample]

    # Check if selected points are within radius; fill out-of-radius with nearest neighbor
    group_dist = torch.gather(dist, dim=-1, index=group_idx)  # [B, npoint, n_sample]
    nn_idx = group_idx[:, :, 0:1]  # [B, npoint, 1] nearest neighbor per centroid
    out_of_radius = group_dist > radius
    group_idx = torch.where(out_of_radius, nn_idx.expand_as(group_idx), group_idx)

    return group_idx


# ───────── Set Abstraction Layer ─────────
class SetAbstraction(nn.Module):
    """
    1. FPS downsamples to n_centroids centroids
    2. Ball Query finds n_sample neighbors around each centroid
    3. PointNet (1x1 Conv + BN + ReLU sequence) extracts local features
    4. Max-pool aggregates neighbor features → output [B, n_centroids, mlp_out[-1]]

    in_channels: input point feature dimension (EXCLUDING relative_xyz).
                 SA1 uses 3 (xyz), SA2 uses 128 (SA1 output).
                 MLP actual input = in_channels + 3 (relative_xyz always concatenated).
    """
    def __init__(self, n_centroids, radius, n_sample, in_channels, mlp_dims):
        super().__init__()
        self.n_centroids = n_centroids
        self.radius = radius
        self.n_sample = n_sample

        layers = []
        actual_in = in_channels + 3  # relative_xyz always concatenated
        for i, out_dim in enumerate(mlp_dims):
            in_dim = actual_in if i == 0 else mlp_dims[i - 1]
            layers.append(nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(out_dim))
            layers.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, features):
        """
        xyz: [B, N, 3] — input point coordinates
        features: [B, N, C_in] — input point features (xyz itself for SA1, SA1 output for SA2)
        returns:
            new_xyz: [B, n_centroids, 3] — subsampled centroid coordinates
            new_features: [B, n_centroids, C_out] — aggregated local features
        """
        B, N, _ = xyz.shape

        # 1. FPS
        fps_idx = farthest_point_sample(xyz, self.n_centroids)  # [B, n_centroids]
        fps_idx_expanded = fps_idx.unsqueeze(-1).expand(-1, -1, 3)  # [B, n_centroids, 3]
        new_xyz = torch.gather(xyz, dim=1, index=fps_idx_expanded)  # [B, n_centroids, 3]

        # 2. Ball Query
        group_idx = ball_query(xyz, new_xyz, self.radius, self.n_sample)
        # [B, n_centroids, n_sample]

        # 3. Group features
        batch_indices = torch.arange(B, device=xyz.device).view(B, 1, 1).expand(-1, self.n_centroids, self.n_sample)

        # neighbor relative coordinates
        grouped_xyz = xyz[batch_indices, group_idx]  # [B, n_centroids, n_sample, 3]
        grouped_xyz_relative = grouped_xyz - new_xyz.unsqueeze(2)

        # neighbor features
        grouped_features = features[batch_indices, group_idx]  # [B, n_centroids, n_sample, C_in]

        # concat relative coords + features
        grouped_concat = torch.cat([grouped_xyz_relative, grouped_features], dim=-1)
        # SA1: [B, 512, 32, 3+3=6]  SA2: [B, 128, 64, 3+128=131]

        # 4. PointNet: [B, C_in+3, n_centroids, n_sample] → [B, C_out, n_centroids, n_sample]
        grouped_concat = grouped_concat.permute(0, 3, 1, 2)
        grouped_concat = self.mlp(grouped_concat)

        # 5. Max-pool over neighbors
        new_features = grouped_concat.max(dim=-1).values  # [B, C_out, n_centroids]
        new_features = new_features.permute(0, 2, 1)  # [B, n_centroids, C_out]

        return new_xyz, new_features


# ───────── PointNet++ Encoder ─────────
class PointNetPlusPlusEncoder(nn.Module):
    """
    PointNet++ encoder: 2 Set Abstraction levels + distance-weighted dual pooling

    Parameters:
        out_channels: output feature dimension (default 128, compatible with MultiStagePointNet)
        sa1_n_centroids: SA1 centroid count (default 512)
        sa1_radius: SA1 ball query radius, meters (default 1.0)
        sa1_n_sample: SA1 neighbors per centroid (default 32)
        sa2_n_centroids: SA2 centroid count (default 128)
        sa2_radius: SA2 ball query radius, meters (default 4.0)
        sa2_n_sample: SA2 neighbors per centroid (default 64)
        use_distance_weight: enable distance-weighted pooling (default True)

    Keeps h_dim, num_layers, **kwargs for MultiStagePointNet init signature compatibility (unused).
    """

    def __init__(self,
                 h_dim=128,
                 out_channels=128,
                 num_layers=4,
                 sa1_n_centroids=512,
                 sa1_radius=1.0,
                 sa1_n_sample=32,
                 sa2_n_centroids=128,
                 sa2_radius=4.0,
                 sa2_n_sample=64,
                 use_distance_weight=True,
                 **kwargs):
        super().__init__()
        self.out_channels = out_channels
        self.use_distance_weight = use_distance_weight

        # SA1: local detail (small radius)
        # in_channels=3: point xyz as initial features; MLP actual input = 3(feat) + 3(relative_xyz) = 6
        self.sa1 = SetAbstraction(
            n_centroids=sa1_n_centroids,
            radius=sa1_radius,
            n_sample=sa1_n_sample,
            in_channels=3,
            mlp_dims=[64, 64, 128],
        )

        # SA2: regional context (large radius)
        # in_channels=128: SA1 output features; MLP actual input = 128(feat) + 3(relative_xyz) = 131
        self.sa2 = SetAbstraction(
            n_centroids=sa2_n_centroids,
            radius=sa2_radius,
            n_sample=sa2_n_sample,
            in_channels=128,
            mlp_dims=[128, 128, 256],
        )

        # Global: 1x1 Conv + BN + ReLU → distance-weighted dual pooling → Linear projection
        self.conv_global = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.proj_pool = nn.Linear(512, out_channels)

        cprint(f'[PointNet++] SA1: {sa1_n_centroids} centroids, r={sa1_radius}m, k={sa1_n_sample}', 'cyan')
        cprint(f'[PointNet++] SA2: {sa2_n_centroids} centroids, r={sa2_radius}m, k={sa2_n_sample}', 'cyan')
        cprint(f'[PointNet++] distance_weight: {use_distance_weight}', 'cyan')

    def forward(self, x):
        """
        x: [B, N, 3] — body-frame xyz point cloud
        returns: [B, out_channels] — global feature vector
        """
        # SA1: N → 512  (xyz serves as both coordinates and features)
        xyz_sa1, feat_sa1 = self.sa1(x, x)  # feat_sa1: [B, 512, 128]

        # SA2: 512 → 128
        xyz_sa2, feat_sa2 = self.sa2(xyz_sa1, feat_sa1)  # feat_sa2: [B, 128, 256]

        # Global pooling
        # feat_sa2: [B, 128, 256] → permute → [B, 256, 128] → conv → [B, 256, 128]
        feat = feat_sa2.permute(0, 2, 1)
        feat = self.conv_global(feat)  # [B, 256, 128]

        if self.use_distance_weight:
            distances = torch.norm(xyz_sa2, dim=-1)  # [B, 128]
            weights = 1.0 / (1.0 + distances)        # [B, 128]
            weights = weights / weights.sum(dim=-1, keepdim=True)  # normalize
            feat_weighted = feat * weights.unsqueeze(1)  # [B, 256, 128]
        else:
            feat_weighted = feat

        # Dual pooling: Max (saliency) + Avg (robustness)
        feat_max = feat_weighted.max(dim=-1).values   # [B, 256]
        feat_avg = feat_weighted.sum(dim=-1)           # [B, 256]
        global_feat = self.proj_pool(torch.cat([feat_max, feat_avg], dim=-1))  # [B, out_channels]

        return global_feat
