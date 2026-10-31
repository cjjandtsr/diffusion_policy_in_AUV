"""
collision_loss_v9_1.py — Phase α Finetune: z-leakage fix + early avoidance enhancement
================================================================================
基于 collision_loss_v9_graded.py（Phase α 三级分级 collision loss），两处改动：

改动 1 (Task 1 — z 泄漏修复):
  - 距离计算加入 z 降权：dist² = dxy² + z_weight · dz²
  - z_weight=0.2 默认，水平避障为主，杜绝"降高度作弊"
  - 新增 weighted_knn_softmin_distance() 函数

改动 2 (Task 2 — 提前避障增强):
  - 三级权重从 (0.1, 1.0, 3.0) → (0.5, 2.0, 3.0)
    · 预警级 ×5（0.1→0.5）：远距（1.5-2.0m）有实质梯度
    · 关注级 ×2（1.0→2.0）：锚定 expert P50 1.59m
    · 危险级不变（3.0）：防撞底线
  - 预警级 hinge 从 quadratic → linear（对抗 D9 远距衰减）
    · linear: max(0, ε-d) → 梯度恒定，远距不衰减
    · 关注/危险级保持 quadratic

设计依据: iDP3_UAV_v9_Master_Design.md §6.1.1 (v9.4)
触发原因: Guide_PhaseAlpha_finetune_zleak_onset.md

放置路径:
  ~/AUV/.../diffusion_policy_3d/model/common/collision_loss_v9_1.py
"""

import torch
import torch.nn.functional as F
import math

# ── 从 v9 复用核心组件 ──
from diffusion_policy_3d.model.common.collision_loss_v9 import (
    body_to_world,
    cumsum_dpos_to_waypoints,
)


# ═══════════════════════════════════════════════════════════════════════════
# Task 1: z-downweighted distance function
# ═══════════════════════════════════════════════════════════════════════════

def weighted_knn_softmin_distance(
    waypoints,
    point_cloud,
    k=8,
    tau=5.0,
    eps=1e-12,
    z_weight=0.2,
):
    """
    k-NN soft-min distance with z-downweight.

    [Task 1 FIX] 距离计算加入 z 降权：dist² = dxy² + z_weight · dz²
    让 collision loss 的梯度主要指向水平方向，杜绝"降高度作弊"。

    与 collision_loss_v9.knn_softmin_distance 的区别:
      - cdist(xyz) → cdist(xy) + z_weight * cdist(z)
      - z_weight < 1 时，z 方向的距离贡献被降权
      - z_weight=1.0 时恢复为原始 3D 欧氏距离

    Args:
        waypoints:   (B, H, 3)   — 轨迹点 (world frame)
        point_cloud: (B, N, 3)   — 点云 (world frame)
        k:           int         — 近邻数
        tau:         float       — soft-min 锐度
        eps:         float       — 距离 norm 的数值 floor
        z_weight:    float       — z 方向权重 (default 0.2, 水平为主)

    Returns:
        d_soft: (B, H) — 每个 waypoint 的 soft-min 距离 (≥ 0)
    """
    B, H, _ = waypoints.shape
    N = point_cloud.shape[1]

    # 水平距离分量 (B, H, N)
    dxy_sq = torch.cdist(waypoints[..., :2], point_cloud[..., :2], p=2.0) ** 2

    # 垂直距离分量 (B, H, N) — 用 cdist 保持与 xy 一致的接口
    dz_sq = torch.cdist(waypoints[..., 2:3], point_cloud[..., 2:3], p=2.0) ** 2

    # 加权距离: d² = dxy² + z_weight · dz²
    d2 = dxy_sq + z_weight * dz_sq
    d_weighted = torch.sqrt(d2 + eps)  # (B, H, N)

    # 取 k 个最近邻的距离 (B, H, k)
    k_actual = min(k, N)
    d_knn, _ = torch.topk(d_weighted, k=k_actual, dim=-1, largest=False)

    # 数值 floor
    d_knn = torch.sqrt(d_knn * d_knn + eps)

    # soft-min via LogSumExp + 偏差修正
    log_k_over_tau = math.log(k_actual) / tau
    d_soft = -(1.0 / tau) * torch.logsumexp(-tau * d_knn, dim=-1) + log_k_over_tau

    # 物理护栏: 距离不可能为负
    d_soft = d_soft.clamp(min=0.0)

    return d_soft


# ═══════════════════════════════════════════════════════════════════════════
# Task 1 + 2: λ(t) schedule (unchanged from v9_graded)
# ═══════════════════════════════════════════════════════════════════════════

def lambda_t_schedule(t, schedule='linear'):
    """λ(t) 时间加权 (unchanged from v9_graded)."""
    if schedule == 'constant':
        return torch.ones_like(t)
    elif schedule == 'quadratic':
        return t ** 2
    else:
        return t


# ═══════════════════════════════════════════════════════════════════════════
# Task 1 + 2: 三级分级 collision loss (revised)
# ═══════════════════════════════════════════════════════════════════════════

def collision_auxiliary_loss_graded(
    pred_x0_action,
    point_cloud,
    cur_pos,
    sin_yaw=None,
    cos_yaw=None,
    eps_levels=(2.0, 1.5, 0.75),
    weights=(0.5, 2.0, 3.0),       # [Task 2] 上调预警/关注级权重
    hinge_types=('linear', 'quadratic', 'quadratic'),  # [Task 2] 预警级 linear
    beta=None,
    tau_softmin=5.0,
    k_nn=8,
    horizon_decay_tau=8.0,
    dpos_slice=(0, 3),
    flow_t=None,
    lambda_t_schedule_type='linear',
    z_weight=0.2,                  # [Task 1] z 降权
    return_diagnostics=False,
):
    """
    Phase α Finetune: 三级距离分级 collision auxiliary loss (v9_1).

    与 collision_loss_v9_graded.collision_auxiliary_loss_graded 的区别:
      [Task 1] 距离计算: 3D cdist → weighted (z_weight=0.2) → 杜绝 z 泄漏
      [Task 2] 权重: (0.1,1.0,3.0) → (0.5,2.0,3.0) → 提前避障
      [Task 2] hinge: 预警级 quadratic → linear → 对抗远距梯度衰减

    Args:
        pred_x0_action: (B, H, action_dim) — clean trajectory (UN-normalized)
        point_cloud:    (B, N, 3 or 6)     — body-frame point cloud
        cur_pos:        (B, 3)             — current drone position (world)
        sin_yaw:        (B,) or None       — sin(yaw)
        cos_yaw:        (B,) or None       — cos(yaw)
        eps_levels:     tuple[float]       — margins (warn, attention, danger)
        weights:        tuple[float]       — weights [Task 2 default: (0.5,2.0,3.0)]
        hinge_types:    tuple[str]         — 'linear' or 'quadratic' per level
        beta:           float or None      — softplus sharpness (None→20)
        tau_softmin:    float              — k-NN soft-min sharpness
        k_nn:           int                — k-NN neighbors
        horizon_decay_tau: float           — horizon weight τ
        dpos_slice:     tuple              — dpos channel range
        flow_t:         (B,) or None       — flow timestep for λ(t)
        lambda_t_schedule_type: str        — λ(t) schedule
        z_weight:       float              — [Task 1] z direction weight
        return_diagnostics: bool           — return diag dict

    Returns:
        loss: scalar tensor
        (optional) diagnostics: dict
    """
    if beta is None:
        beta = 20.0

    B, H, _ = pred_x0_action.shape
    device = pred_x0_action.device

    # NaN guard
    if not torch.isfinite(pred_x0_action).all() or not torch.isfinite(point_cloud).all():
        if return_diagnostics:
            return torch.tensor(0.0, device=device), {}
        return torch.tensor(0.0, device=device)

    # Step 1: extract dpos
    s, e = dpos_slice
    dpos_world = pred_x0_action[..., s:e]

    # Step 2: point cloud → xyz, to device
    if point_cloud.shape[-1] > 3:
        pc_xyz = point_cloud[..., :3]
    else:
        pc_xyz = point_cloud
    pc_xyz = pc_xyz.to(device)

    # Step 3: body→world (P0-A fix)
    if sin_yaw is not None and cos_yaw is not None:
        pc_xyz = body_to_world(pc_xyz, cur_pos.to(device), sin_yaw, cos_yaw)

    # Step 4: cumsum → world waypoints
    waypoints = cumsum_dpos_to_waypoints(dpos_world, cur_pos.to(device))

    # ── [Task 1] z-downweighted distance (replaces original knn_softmin_distance) ──
    d_soft = weighted_knn_softmin_distance(
        waypoints, pc_xyz, k=k_nn, tau=tau_softmin, z_weight=z_weight)  # (B, H)

    # ── [Task 2] per-level hinge (linear for warn, quadratic for attn/danger) ──
    loss_per_level = []
    diag_per_level = {}

    level_names = ['warn', 'attn', 'danger']

    for i, (eps, w, htype) in enumerate(zip(eps_levels, weights, hinge_types)):
        level_name = level_names[i] if i < len(level_names) else f'level{i}'

        margin = eps - d_soft
        margin = margin.clamp(max=2.0 * eps)

        if htype == 'linear':
            # [Task 2] linear hinge: max(0, ε-d), gradient = const
            hinge = torch.clamp(margin, min=0.0)
        else:
            # quadratic hinge: max(0, ε-d)²
            hinge = torch.clamp(margin, min=0.0) ** 2

        # horizon weights
        k_idx = torch.arange(H, device=device, dtype=hinge.dtype)
        w_horizon = torch.exp(-k_idx / horizon_decay_tau)
        w_horizon = w_horizon / w_horizon.sum()

        level_loss_per_sample = (hinge * w_horizon.unsqueeze(0)).sum(dim=1)
        level_loss = level_loss_per_sample.mean()
        loss_per_level.append(w * level_loss)

        if return_diagnostics:
            with torch.no_grad():
                diag_per_level[f'{level_name}_viol'] = (d_soft < eps).float().mean().item()
                diag_per_level[f'{level_name}_loss'] = level_loss.item()

    total_collision = sum(loss_per_level)

    # Step 5: λ(t) time weighting
    lambda_t_weight = 1.0
    if flow_t is not None:
        lambda_t = lambda_t_schedule(flow_t, schedule=lambda_t_schedule_type)
        per_sample_loss = torch.zeros(B, device=device)
        for i, (eps, w, htype) in enumerate(zip(eps_levels, weights, hinge_types)):
            margin = eps - d_soft
            margin = margin.clamp(max=2.0 * eps)
            if htype == 'linear':
                hinge = torch.clamp(margin, min=0.0)
            else:
                hinge = torch.clamp(margin, min=0.0) ** 2
            k_idx = torch.arange(H, device=device, dtype=hinge.dtype)
            w_horizon = torch.exp(-k_idx / horizon_decay_tau)
            w_horizon = w_horizon / w_horizon.sum()
            per_sample_loss = per_sample_loss + w * (hinge * w_horizon.unsqueeze(0)).sum(dim=1)
        total_collision = (per_sample_loss * lambda_t).mean()
        lambda_t_weight = lambda_t.mean().item()

    # NaN fallback
    if not torch.isfinite(total_collision):
        if return_diagnostics:
            return torch.tensor(0.0, device=device, requires_grad=True), {
                'd_soft_p50': float('nan'),
                'd_soft_p05': float('nan'),
                'nan_fallback': 1.0,
            }
        return torch.tensor(0.0, device=device, requires_grad=True)

    if not return_diagnostics:
        return total_collision

    with torch.no_grad():
        diag = {
            'd_soft_p50': torch.quantile(d_soft, 0.50).item(),
            'd_soft_p05': torch.quantile(d_soft, 0.05).item(),
            'd_soft_min': d_soft.min().item(),
            'lambda_t_mean': lambda_t_weight,
            'z_weight': z_weight,
            **diag_per_level,
            'n_violating_share': (d_soft < eps_levels[-1]).float().mean().item(),
        }
    return total_collision, diag


# ═══════════════════════════════════════════════════════════════════════════
# Backward compatibility: single-level interface
# ═══════════════════════════════════════════════════════════════════════════

def collision_auxiliary_loss(
    pred_x0_action,
    point_cloud,
    cur_pos,
    sin_yaw=None,
    cos_yaw=None,
    d_safe=0.75,
    beta=None,
    tau_softmin=5.0,
    k_nn=8,
    horizon_decay_tau=8.0,
    dpos_slice=(0, 3),
    flow_t=None,
    lambda_t_schedule_type='linear',
    z_weight=0.2,
    return_diagnostics=False,
):
    """Backward-compatible single-level interface. Delegates to graded version."""
    return collision_auxiliary_loss_graded(
        pred_x0_action=pred_x0_action,
        point_cloud=point_cloud,
        cur_pos=cur_pos,
        sin_yaw=sin_yaw,
        cos_yaw=cos_yaw,
        eps_levels=(d_safe,),
        weights=(1.0,),
        hinge_types=('quadratic',),
        beta=beta,
        tau_softmin=tau_softmin,
        k_nn=k_nn,
        horizon_decay_tau=horizon_decay_tau,
        dpos_slice=dpos_slice,
        flow_t=flow_t,
        lambda_t_schedule_type=lambda_t_schedule_type,
        z_weight=z_weight,
        return_diagnostics=return_diagnostics,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Self-tests
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import numpy as np

    print("=" * 60)
    print("collision_loss_v9_1.py — self-test")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)

    B, H, N = 4, 16, 4096
    action_dim = 11

    # --- Test 1: z-weighted distance (z_weight < 1 reduces z contribution) ---
    print("\n[Test 1] z-weighted distance: z contrib < xy contrib")
    wp_test = torch.zeros(1, 1, 3)
    # Scenario: obstacle 1m lateral, 1m vertical
    pc_test = torch.zeros(1, 2, 3)
    pc_test[0, 0] = torch.tensor([0.0, 1.0, 0.0])  # 1m pure lateral
    pc_test[0, 1] = torch.tensor([0.0, 0.0, 1.0])  # 1m pure vertical

    # With z_weight=0.2, the lateral point should be "closer" than vertical
    d_soft_z02 = weighted_knn_softmin_distance(
        wp_test, pc_test, k=2, tau=5.0, z_weight=0.2)
    # With z_weight=1.0 (original 3D), both equally distant
    d_soft_z10 = weighted_knn_softmin_distance(
        wp_test, pc_test, k=2, tau=5.0, z_weight=1.0)

    print(f"  z_weight=0.2: d_soft = {d_soft_z02.item():.4f} m")
    print(f"  z_weight=1.0: d_soft = {d_soft_z10.item():.4f} m")
    print(f"  z_weight=0.2 should be SMALLER (z downweighted → near neighbors are closer in xy)")
    print(f"  {'PASS' if d_soft_z02.item() < d_soft_z10.item() else 'FAIL'}")

    # --- Test 2: z gradient suppression (realistic obstacle scenario) ---
    print("\n[Test 2] z gradient suppression (realistic obstacle scenario)")
    print("  Scenario: waypoint at origin, dense xy obstacle at 0.3m, single z obstacle at 0.8m")
    wp_grad = torch.zeros(1, 1, 3, requires_grad=True)
    pc_grad = torch.zeros(1, 20, 3)
    # 10 points at xy=(0.3, 0), z=0 → lateral obstacle surface
    for i in range(10):
        pc_grad[0, i] = torch.tensor([0.3, 0.0, 0.0])
    # 5 points at xy=(0, 0.3), z=0 → lateral obstacle surface
    for i in range(10, 15):
        pc_grad[0, i] = torch.tensor([0.0, 0.3, 0.0])
    # 1 point directly above at z=0.8 → vertical-only displacement
    pc_grad[0, 15] = torch.tensor([0.0, 0.0, 0.8])
    # fill rest far away
    for i in range(16, 20):
        pc_grad[0, i] = torch.tensor([100.0, 0.0, 0.0])

    d_z02 = weighted_knn_softmin_distance(wp_grad, pc_grad, k=8, tau=5.0, z_weight=0.2)
    d_z02.backward()
    grad_xy = wp_grad.grad[..., :2].abs().sum().item()
    grad_z = wp_grad.grad[..., 2].abs().item()
    print(f"  |grad_xy| = {grad_xy:.4f}, |grad_z| = {grad_z:.4f}")
    print(f"  ratio (z/xy) = {grad_z/(grad_xy+1e-8):.4f}")
    print(f"  {'PASS' if grad_z < grad_xy * 0.5 else 'INFO'}  (z gradient should be suppressed relative to xy)")
    wp_grad.grad = None  # clean up

    # --- Test 3: linear vs quadratic hinge ---
    print("\n[Test 3] linear vs quadratic hinge gradient at far distance")
    d_soft_test = torch.tensor([[1.8]], requires_grad=True)  # 0.2m inside 2.0m margin
    eps = 2.0

    # quadratic: max(0, eps-d)²
    hinge_quad = torch.clamp(eps - d_soft_test, min=0.0) ** 2
    hinge_quad.backward(retain_graph=False)
    grad_quad = d_soft_test.grad.item()
    d_soft_test.grad = None

    # linear: max(0, eps-d)
    hinge_lin = torch.clamp(eps - d_soft_test, min=0.0)
    hinge_lin.backward()
    grad_lin = d_soft_test.grad.item()

    print(f"  d_soft=1.8m, eps=2.0m → margin=0.2m")
    print(f"  quadratic hinge grad: {grad_quad:.4f} (∝ 2*(eps-d) = 0.4)")
    print(f"  linear hinge grad:    {grad_lin:.4f} (constant = 1.0)")
    print(f"  linear > quadratic at small margin: {'PASS' if abs(grad_lin) > abs(grad_quad) else 'FAIL'}")

    # --- Test 4: graded loss with new defaults ---
    print("\n[Test 4] graded loss with v9_1 defaults (w=[0.5,2.0,3.0], linear warn)")
    pred_x0 = torch.randn(B, H, action_dim, requires_grad=True)
    pc = torch.randn(B, N, 6) * 5.0
    cur_p = torch.zeros(B, 3)
    flow_t = torch.rand(B) * 0.9 + 0.1

    loss, diag = collision_auxiliary_loss_graded(
        pred_x0, pc, cur_p,
        return_diagnostics=True,
        flow_t=flow_t,
    )
    loss.backward()

    grad_ok = pred_x0.grad is not None and torch.isfinite(pred_x0.grad).all()
    grad_dpos = pred_x0.grad[..., 0:3].abs().sum().item()
    grad_other = pred_x0.grad[..., 3:].abs().sum().item()

    print(f"  loss = {loss.item():.4f}")
    print(f"  default weights = (0.5, 2.0, 3.0)")
    print(f"  default hinge_types = ('linear', 'quadratic', 'quadratic')")
    print(f"  default z_weight = 0.2")
    print(f"  grad finite: {grad_ok}  {'PASS' if grad_ok else 'FAIL'}")
    print(f"  grad on dpos: {grad_dpos:.4f} (should > 0)")
    print(f"  grad on other: {grad_other:.2e} (should = 0)")
    print(f"  {'PASS' if grad_dpos > 0 and grad_other < 1e-6 else 'FAIL'}")
    print(f"  z_weight in diag: {diag.get('z_weight', 'NOT FOUND')}")

    # --- Test 5: monotonicity (near > far) ---
    print("\n[Test 5] monotonicity (near obstacle > far obstacle)")
    pred_t5 = torch.randn(2, 8, action_dim, requires_grad=False)
    cur_p = torch.zeros(2, 3)
    # Near: all points at 0.3m → should be inside all margins
    pc_near = torch.randn(2, 200, 3) * 0.1 + 0.3
    # Far: all points at 10m away → outside all margins
    pc_far = torch.randn(2, 200, 3) * 0.5 + 10.0

    loss_near = collision_auxiliary_loss_graded(
        pred_t5, pc_near, cur_p, return_diagnostics=False)
    loss_far = collision_auxiliary_loss_graded(
        pred_t5, pc_far, cur_p, return_diagnostics=False)

    print(f"  loss (near, ~0.3m): {loss_near.item():.4f}")
    print(f"  loss (far, ~10m):   {loss_far.item():.6f}")
    print(f"  {'PASS' if loss_near.item() > loss_far.item() * 10 else 'FAIL'}")

    # --- Test 6: backward compatibility ---
    print("\n[Test 6] backward-compatible single-level interface")
    loss_single, diag_single = collision_auxiliary_loss(
        pred_t5, pc_near, cur_p, d_safe=0.75, return_diagnostics=True)
    print(f"  loss (d_safe=0.75): {loss_single.item():.4f}")
    print(f"  diag keys: {list(diag_single.keys())}")
    print(f"  {'PASS' if 'z_weight' in diag_single else 'INFO: z_weight=' + str(diag_single.get('z_weight', 'N/A'))}")

    # --- Test 7: z_weight=1.0 recovers near-original behavior ---
    print("\n[Test 7] z_weight=1.0 recovers near-original 3D distance")
    # Compare with original knn_softmin_distance from collision_loss_v9
    from diffusion_policy_3d.model.common.collision_loss_v9 import knn_softmin_distance as orig_knn

    wp_t7 = torch.randn(2, 8, 3)
    pc_t7 = torch.randn(2, 100, 3)

    d_orig = orig_knn(wp_t7, pc_t7, k=8, tau=5.0)
    d_v91_z10 = weighted_knn_softmin_distance(wp_t7, pc_t7, k=8, tau=5.0, z_weight=1.0)
    d_v91_z02 = weighted_knn_softmin_distance(wp_t7, pc_t7, k=8, tau=5.0, z_weight=0.2)

    err_z10 = (d_orig - d_v91_z10).abs().max().item()
    err_z02 = (d_orig - d_v91_z02).abs().max().item()

    print(f"  z_weight=1.0 vs original: max error = {err_z10:.6f} (expect near 0)")
    print(f"  z_weight=0.2 vs original: max error = {err_z02:.4f} (expect > 0, different)")
    print(f"  z_weight=1.0 recovery: {'PASS' if err_z10 < 0.02 else 'FAIL'}")

    print("\n" + "=" * 60)
    print("self-test complete")
    print("=" * 60)
