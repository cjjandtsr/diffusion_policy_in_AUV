"""
safety_guide.py — Phase ②.5 Safety-Guided Diffusion: inference-time safety guidance

Design: Phase2_5_Safety_Guidance_FM_Adaptation_Design.md
Route 1: FM velocity-space classifier guidance (gradient through UNet)
Route 2: Trajectory-space gradient descent (CGD-style, fallback)

Reuses cost function primitives from collision_loss.py (k-NN soft-min + softplus-hinge).
"""

import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


def _build_kdtree(point_cloud_world: np.ndarray):
    """Build scipy KD-Tree from world-frame point cloud (B=1 only)."""
    from scipy.spatial import cKDTree
    # point_cloud_world: (N, 3) — world-frame xyz
    valid = np.isfinite(point_cloud_world).all(axis=1)
    pc = point_cloud_world[valid]
    if len(pc) < 8:
        return None, pc
    return cKDTree(pc), pc


def _knn_via_kdtree(waypoints_np: np.ndarray, kdtree, pc: np.ndarray, k: int, device, dtype):
    """Query KD-Tree for k nearest neighbors, return distances as tensor (B, H, k).

    Args:
        waypoints_np: (H, 3) numpy array (B=1)
        kdtree: scipy cKDTree
        pc: (N, 3) valid world-frame points
        k: number of neighbors
    Returns:
        dists: (1, H, k) tensor with L2 distances
        neighbors: (1, H, k, 3) tensor with neighbor positions
    """
    k_actual = min(k, len(pc))
    dists_np, idx_np = kdtree.query(waypoints_np, k=k_actual)
    # dists_np: (H, k_actual), idx_np: (H, k_actual)
    neighbors_np = pc[idx_np]  # (H, k_actual, 3)
    dists = torch.tensor(dists_np, device=device, dtype=dtype).unsqueeze(0)  # (1, H, k)
    neighbors = torch.tensor(neighbors_np, device=device, dtype=dtype).unsqueeze(0)  # (1, H, k, 3)
    return dists, neighbors


def compute_safety_cost_on_x1hat(
    x1_hat: torch.Tensor,
    point_cloud_world: torch.Tensor,
    cur_pos: torch.Tensor,
    d_safe: float = 0.35,
    beta: float = 20.0,
    k_nn: int = 8,
    tau_softmin: float = 5.0,
    horizon_decay_tau: float = 8.0,
    dpos_slice: tuple = (0, 3),
    return_diagnostics: bool = False,
):
    """Compute collision safety cost on predicted clean trajectory x̂₁.

    This is the DIFFERENTIABLE cost function used for safety guidance.
    Uses KD-Tree for neighbor lookup (non-differentiable) but computes
    distances from neighbor positions (differentiable w.r.t. waypoints).

    Args:
        x1_hat: (B, H, action_dim) — predicted clean trajectory (normalized action space)
        point_cloud_world: (B, N, 3) — world-frame point cloud xyz
        cur_pos: (B, 3) — current drone position (world frame)
        d_safe: safety distance threshold (m), Phase ②.5 = 0.35m
        beta: softplus sharpness
        k_nn: number of nearest neighbors
        tau_softmin: soft-min temperature
        horizon_decay_tau: temporal decay τ for w_k = exp(-k/τ)
        dpos_slice: (start, end) indices for dpos channels in action
        return_diagnostics: if True, return (cost, diagnostics dict)

    Returns:
        cost: scalar tensor — safety cost (differentiable)
        (optional) diagnostics: dict
    """
    from diffusion_policy_3d.model.common.collision_loss import (
        cumsum_dpos_to_waypoints,
    )

    B, H, Da = x1_hat.shape
    device = x1_hat.device
    dtype = x1_hat.dtype

    # Input NaN guard
    if not torch.isfinite(x1_hat).all():
        cost = torch.tensor(0.0, device=device)
        if return_diagnostics:
            return cost, {}
        return cost

    # Extract dpos channels
    s, e = dpos_slice
    dpos_world = x1_hat[..., s:e]  # (B, H, 3)

    # cumsum → world waypoints
    waypoints = cumsum_dpos_to_waypoints(dpos_world, cur_pos.to(device))  # (B, H, 3)

    # Handle point cloud: B=1 for inference
    if point_cloud_world.dim() == 2:
        point_cloud_world = point_cloud_world.unsqueeze(0)  # (1, N, 3)
    if point_cloud_world.shape[-1] > 3:
        pc_xyz = point_cloud_world[..., :3]
    else:
        pc_xyz = point_cloud_world

    # For B=1 inference: use KD-Tree for efficient neighbor lookup
    if B == 1:
        wp_np = waypoints[0].detach().cpu().numpy()  # (H, 3) numpy
        pc_np = pc_xyz[0].detach().cpu().numpy()     # (N, 3) numpy
        kdtree, pc_valid = _build_kdtree(pc_np)

        if kdtree is None:
            cost = torch.tensor(0.0, device=device)
            if return_diagnostics:
                return cost, {'kd_failed': True}
            return cost

        dists, neighbors = _knn_via_kdtree(wp_np, kdtree, pc_valid, k_nn, device, dtype)
        # dists: (1, H, k), neighbors: (1, H, k, 3) — both differentiable w.r.t. nothing yet
        # But! dists are computed from numpy → they're detached from the computation graph.
        # We need to recompute distances from waypoints to neighbors to get gradients.
        # The neighbors were selected by KD-Tree (non-diff), but distance computation is diff.

        # Recompute differentiable distances from waypoints to KD-Tree-selected neighbors
        # neighbors is a constant tensor (detached from graph), waypoints carries the gradient
        wp_expanded = waypoints.unsqueeze(2)  # (1, H, 1, 3)
        nb_expanded = neighbors                # (1, H, k, 3)
        dists_diff = torch.sqrt(
            ((wp_expanded - nb_expanded) ** 2).sum(dim=-1) + 1e-12
        )  # (1, H, k)
        d_knn = dists_diff
    else:
        # B > 1: fallback to full cdist (expensive but correct)
        pc_xyz = pc_xyz.to(device)
        dists = torch.cdist(waypoints, pc_xyz, p=2.0)  # (B, H, N)
        k_actual = min(k_nn, pc_xyz.shape[1])
        d_knn, _ = torch.topk(dists, k=k_actual, dim=-1, largest=False)  # (B, H, k)
        d_knn = torch.sqrt(d_knn * d_knn + 1e-12)

    # Soft-min distance (bias-corrected, matching collision_loss.py)
    k_actual = d_knn.shape[-1]
    log_k_over_tau = math.log(k_actual) / tau_softmin
    d_soft = -(1.0 / tau_softmin) * torch.logsumexp(-tau_softmin * d_knn, dim=-1) + log_k_over_tau  # (B, H)
    d_soft = d_soft.clamp(min=0.0)

    # Softplus-hinge cost
    margin = d_safe - d_soft  # (B, H), positive = violation
    margin = margin.clamp(max=2.0 * d_safe)
    c_per_waypoint = F.softplus(beta * margin) / beta  # (B, H)
    c_per_waypoint = c_per_waypoint.clamp(max=2.0 * d_safe)

    # Horizon weighting: w_k = exp(-k/τ), normalized
    k_idx = torch.arange(H, device=device, dtype=c_per_waypoint.dtype)
    w_k = torch.exp(-k_idx / horizon_decay_tau)  # (H,)
    w_k = w_k / w_k.sum()

    # Weighted average → scalar cost
    cost_per_sample = (c_per_waypoint * w_k.unsqueeze(0)).sum(dim=1)  # (B,)
    cost = cost_per_sample.mean()

    if not torch.isfinite(cost):
        cost = torch.tensor(0.0, device=device)
        if return_diagnostics:
            return cost, {'nan_fallback': True}
        return cost

    if return_diagnostics:
        with torch.no_grad():
            diag = {
                'd_soft_p50': torch.quantile(d_soft, 0.50).item(),
                'd_soft_p05': torch.quantile(d_soft, 0.05).item(),
                'd_soft_min': d_soft.min().item(),
                'c_mean': c_per_waypoint.mean().item(),
                'c_max': c_per_waypoint.max().item(),
                'violating_share': (d_soft < d_safe).float().mean().item(),
            }
        return cost, diag
    return cost


def compute_schedule_weight(k: int, num_steps: int, schedule: str = 'cosine_mid') -> float:
    """Compute safety guidance schedule weight α for step k.

    Args:
        k: current step index (0-based)
        num_steps: total number of FM steps (4)
        schedule: 'cosine_mid', 'linear_up', or 'const'

    Returns:
        alpha: schedule weight in [0, 1]

    FM 4-step mapping (cosine_mid):
        k=0 (t≈0.00): α=0.0   — preserve anchor
        k=1 (t≈0.25): α=0.5   — gradual
        k=2 (t≈0.50): α=1.0   — peak correction
        k=3 (t≈0.75): α=0.5   — exit
    """
    if schedule == 'const':
        return 1.0
    elif schedule == 'linear_up':
        return (k + 1) / num_steps
    elif schedule == 'cosine_mid':
        # Design doc: peak at middle steps, zero/taper at ends
        # FM 4-step: [0.0, 0.5, 1.0, 0.5]
        # DDIM 5-step: [0.0, 0.5, 1.0, 0.5, 0.0]
        if num_steps == 4:
            weights = [0.0, 0.5, 1.0, 0.5]
        elif num_steps == 5:
            weights = [0.0, 0.5, 1.0, 0.5, 0.0]
        else:
            # General cos²: α_k = cos²(π·(k − mid) / (2·mid))
            mid = (num_steps - 1) / 2.0
            weights = [
                max(0.0, min(1.0, math.cos(math.pi * (i - mid) / (2.0 * mid)) ** 2))
                for i in range(num_steps)
            ]
        return weights[k]
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


def l2_clip_gradient(grad_x: torch.Tensor, g_max: float = 1.0) -> torch.Tensor:
    """L2-norm clip per waypoint (per (T, D) dimension group).

    Args:
        grad_x: (B, T, Da) gradient tensor
        g_max: max L2 norm per waypoint

    Returns:
        clipped gradient
    """
    # Per-waypoint L2 norm: (B, T, 1)
    grad_norm = grad_x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = (g_max / grad_norm).clamp(max=1.0)
    return grad_x * scale


# ──────────────────────────────────────────────────────────────
# Route 1: Compute guidance gradient (for injection into velocity)
# ──────────────────────────────────────────────────────────────

def compute_safety_guidance_route1(
    x_t: torch.Tensor,
    t_k: torch.Tensor,
    v_cfg_fn,  # callable: v_cfg_fn(x) -> velocity (B, T, Da)
    point_cloud_world: torch.Tensor,
    cur_pos: torch.Tensor,
    s_safe: float,
    schedule_weight: float,
    d_safe: float,
    beta: float,
    k_nn: int,
    tau_softmin: float,
    horizon_decay_tau: float,
    g_max: float,
) -> torch.Tensor:
    """Route 1: Compute velocity-space guidance gradient (through UNet).

    Re-runs CFG velocity with grad enabled, computes cost on x̂₁,
    backprops through UNet to get ∇_{x_t} c(x̂₁).

    Args:
        x_t: (B, T, Da) — current noisy trajectory (in normalized action space)
        t_k: scalar — current FM time
        v_cfg_fn: callable x_t → v_cfg (CFG velocity, with UNet forward)
        point_cloud_world: (B, N, 3) — world-frame point cloud
        cur_pos: (B, 3) — current drone position
        s_safe: guidance strength
        schedule_weight: α(t) ∈ [0, 1]
        d_safe, beta, k_nn, tau_softmin, horizon_decay_tau: cost params
        g_max: L2 gradient clip

    Returns:
        guidance: (B, T, Da) — velocity correction term to SUBTRACT from v_cfg
    """
    if s_safe <= 0 or schedule_weight <= 0:
        return torch.zeros_like(x_t)

    # Second forward pass with grad enabled
    x_t_grad = x_t.detach().requires_grad_(True)
    v_cfg_grad = v_cfg_fn(x_t_grad)  # (B, T, Da) — through UNet

    # Predicted clean trajectory (FM convention: t=1 is clean)
    # x̂₁ = x_t + (1-t) * v_t
    x1_hat = x_t_grad + (1.0 - t_k) * v_cfg_grad

    # Safety cost
    cost = compute_safety_cost_on_x1hat(
        x1_hat=x1_hat,
        point_cloud_world=point_cloud_world,
        cur_pos=cur_pos,
        d_safe=d_safe,
        beta=beta,
        k_nn=k_nn,
        tau_softmin=tau_softmin,
        horizon_decay_tau=horizon_decay_tau,
    )

    # Gradient w.r.t. x_t_grad (through UNet + cost)
    if not cost.requires_grad:
        return torch.zeros_like(x_t)

    grad_x = torch.autograd.grad(cost, x_t_grad, create_graph=False)[0]

    if grad_x is None or not torch.isfinite(grad_x).all():
        return torch.zeros_like(x_t)

    # L2 clip per waypoint
    grad_x = l2_clip_gradient(grad_x, g_max=g_max)

    # Scale and return (this will be SUBTRACTED from v_cfg)
    return s_safe * schedule_weight * grad_x


# ──────────────────────────────────────────────────────────────
# Route 2: Trajectory-space guidance (CGD-style, no UNet backward)
# ──────────────────────────────────────────────────────────────

def compute_safety_guidance_route2(
    x_t: torch.Tensor,
    t_k: torch.Tensor,
    v_cfg: torch.Tensor,
    point_cloud_world: torch.Tensor,
    cur_pos: torch.Tensor,
    eta_0: float,
    schedule_weight: float,
    d_safe: float,
    beta: float,
    k_nn: int,
    tau_softmin: float,
    horizon_decay_tau: float,
    g_max: float,
) -> torch.Tensor:
    """Route 2: Trajectory-space guidance (CGD-style).

    v_cfg is detached — gradient only through x̂₁ = x_t + (1-t)*v_cfg,
    where ∂x̂₁/∂x_t = I (identity). No UNet backward needed.

    Args:
        x_t: (B, T, Da) — current noisy trajectory
        t_k: scalar — current FM time
        v_cfg: (B, T, Da) — CFG velocity (DETACHED from graph)
        Other args same as Route 1.
        eta_0: trajectory step size (replaces s_safe)

    Returns:
        delta_x: (B, T, Da) — trajectory correction to SUBTRACT from x_t
    """
    if eta_0 <= 0 or schedule_weight <= 0:
        return torch.zeros_like(x_t)

    # Predicted clean trajectory (v_cfg detached)
    x1_hat = x_t + (1.0 - t_k) * v_cfg.detach()

    # Safety cost (gradient only through x_t → x1_hat → cost)
    cost = compute_safety_cost_on_x1hat(
        x1_hat=x1_hat,
        point_cloud_world=point_cloud_world,
        cur_pos=cur_pos,
        d_safe=d_safe,
        beta=beta,
        k_nn=k_nn,
        tau_softmin=tau_softmin,
        horizon_decay_tau=horizon_decay_tau,
    )

    if not cost.requires_grad:
        return torch.zeros_like(x_t)

    grad_x = torch.autograd.grad(cost, x_t, create_graph=False)[0]

    if grad_x is None or not torch.isfinite(grad_x).all():
        return torch.zeros_like(x_t)

    # L2 clip per waypoint
    grad_x = l2_clip_gradient(grad_x, g_max=g_max)

    # Trajectory step: x_t ← x_t - eta * grad
    return eta_0 * schedule_weight * grad_x
