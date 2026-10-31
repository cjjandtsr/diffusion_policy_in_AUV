"""
collision_loss.py — Phase ② Constraint-Aware BC 的 collision auxiliary loss
================================================================================
设计依据: iDP3_UAV_Design_Action_Plan_v1.2.md §5.2 + 附录 A

核心思想:
  把 obstacle constraint 从"策略外 projection"搬到"策略内训练 loss"。
  对模型预测的 clean trajectory x̂₀ 计算 k-NN soft-min 距离到点云,
  用 softplus-hinge 在 d_safe 阈值附近惩罚, 作为 BC loss 的辅助项。

数学形式 (与 v1.2 附录 A 对齐):
  1. x̂₀ 的 dpos 通道 cumsum → world waypoints
  2. body-frame 点云先经 body→world 变换: p_world = R(yaw) @ p_body + cur_pos
     (v1.2 §5.2 :348-349 明确要求; 调用方必须传 sin_yaw/cos_yaw)
  3. 每个 waypoint 计 k-NN soft-min 距离
  4. softplus-hinge: L = softplus(β·(d_safe − d_soft)) / β
  5. 横向加权: w_k = exp(−k/τ) 强调早期 waypoint

[BUG FIX 2026-08-14] 坐标系一致性修复 (V11 审计):
  原实现直接 torch.cdist(world_waypoints, body_pc) 混用坐标系 ——
  waypoints 是 world frame 绝对坐标, 点云是 body frame 相对坐标,
  yaw≠0 时距离系统性失真, cur_pos 远离原点时惩罚完全失效。
  修复: 增加可微 body_to_world() 变换, collision_auxiliary_loss
  新增 sin_yaw/cos_yaw 参数 (None 时跳过变换, 兼容旧调用)。
  设计依据: iDP3_UAV_Design_Action_Plan_v1.2 §5.2 自始明确要求该变换,
  原文件头"augment 自洽"声称不成立 (drone_dataset.py 增广只旋转 world 分量)。

关键: 训练侧 d_safe = 0.75m (ORANGE verdict, 对抗 EGO 0.3m inflation 的擦边演示)

放置路径:
  ~/AUV/Improved-3D-Diffusion-Policy/diffusion_policy_3d/model/common/collision_loss.py
  (或与 phase1a_policy.py 同目录, 按 import 习惯调整)
"""

import torch
import torch.nn.functional as F


def body_to_world(pc_body, cur_pos_world, sin_yaw, cos_yaw):
    """
    把 body-frame 点云转到 world frame。

    绕 z 轴旋转（与训练增广、goal_body 旋转不变约定一致）：
        p_world = R(yaw) @ p_body + cur_pos_world
        R(yaw) = [[c, -s, 0],
                  [s,  c, 0],
                  [0,  0, 1]]

    [FIX 2026-08-14] 移植自 collision_loss_v9.py（Phase 0 P0-A 坐标系修复版）。
    设计依据: iDP3_UAV_Design_Action_Plan_v1.2 §5.2 (:348-349) ——
    waypoints 是 world frame 绝对坐标, 点云是 body frame 相对坐标,
    直接 cdist 在 yaw≠0 时距离系统性失真, 必须先变换到同一坐标系。

    Args:
        pc_body:        (B, N, 3) — body-frame 点云
        cur_pos_world:  (B, 3)    — 无人机当前世界位置
        sin_yaw:        (B,)      — 当前偏航角正弦 (agent_pos[..., 7])
        cos_yaw:        (B,)      — 当前偏航角余弦 (agent_pos[..., 8])

    Returns:
        pc_world: (B, N, 3) — world-frame 点云

    注意：使用 sin/cos 直接构造旋转矩阵，避免 atan2→sin/cos 的数值往返。
          所有操作可微，不破坏梯度链路。
    """
    B, N, _ = pc_body.shape
    c = cos_yaw.view(B, 1, 1)  # (B, 1, 1)
    s = sin_yaw.view(B, 1, 1)

    x = pc_body[..., 0:1]  # (B, N, 1)
    y = pc_body[..., 1:2]
    z = pc_body[..., 2:3]

    # R(yaw) @ p_body
    xw = c * x - s * y
    yw = s * x + c * y
    zw = z  # z 轴不旋转（与 augmentation 约定一致）

    pc_world_local = torch.cat([xw, yw, zw], dim=-1)  # (B, N, 3)
    pc_world = pc_world_local + cur_pos_world.view(B, 1, 3)

    return pc_world


def cumsum_dpos_to_waypoints(dpos_world, cur_pos):
    """
    把 step-wise world-frame dpos 累加成 world-frame waypoints。

    Args:
        dpos_world: (B, H, 3) — action 前 3 维, world-frame step-wise delta
        cur_pos:    (B, 3)    — 当前 drone 位置 (world frame), agent_pos[:, -1, 0:3]

    Returns:
        waypoints: (B, H, 3) — 累加后的 world-frame 绝对位置

    说明:
        x_t = cur_pos + Σ_{i=1..t} dpos_world[i]
        cumsum 可微, backprop 时每个 delta 收到其下游所有 waypoint 的梯度和。
        这个梯度分布的"早期 delta 梯度偏大"由 horizon weights w_k 抵消。
    """
    # (B, H, 3) cumsum along H
    cumulative = torch.cumsum(dpos_world, dim=1)
    waypoints = cur_pos.unsqueeze(1) + cumulative
    return waypoints


def knn_softmin_distance(waypoints, point_cloud, k=8, tau=5.0, eps=1e-12):
    """
    计算每个 waypoint 到点云的 k-NN soft-min 距离。

    [BUG FIX 2026-05-15] 修复 LogSumExp soft-min 的系统性低估
    ─────────────────────────────────────────────────────────
    原版本: d_soft = -1/τ · log Σ exp(-τ d_(l))
    问题:   当 k 个邻居距离都约为 d 时:
              logsumexp(-τd, ..., -τd) = log(k) - τd
              d_soft = -1/τ · (log(k) - τd) = d - log(k)/τ
            即 soft-min 比真实 min **系统性低估 log(k)/τ**
            (τ=5, k=8 时偏差 = log(8)/5 = 0.416m, 可导致 d_soft 变负数)
    修复:   d_soft = -1/τ · logsumexp(-τ d_(l)) + log(k)/τ
            (k 个邻居全相等时, d_soft 精确等于 d, 无偏差)
    并:     d_soft clamp 到 [0, +inf) (物理护栏: 距离不可能为负)
            τ 默认值按 v1.2 §5.2 锁定为 5.0 (偏差修正项 log(k)/τ 已消除系统性低估)
    ─────────────────────────────────────────────────────────

    Args:
        waypoints:   (B, H, 3)   — 轨迹点
        point_cloud: (B, N, 3)   — 点云 (与 waypoints 同坐标系!)
        k:           int         — 近邻数
        tau:         float       — soft-min 锐度 (τ→∞ 趋近 hard min)
        eps:         float       — 距离 norm 的数值 floor, 防零距离奇异

    Returns:
        d_soft: (B, H) — 每个 waypoint 的 soft-min 距离 (≥ 0)

    数学 (修正后):
        d_i = ||waypoint − p_i||  (对每个点云点)
        取 k 个最近: {d_(1), ..., d_(k)}
        d_soft = -1/τ · logsumexp(-τ · d_(l)) + log(k)/τ
              ≈ min(d_(l))  (τ 足够大时)
              ≥ 0           (clamp 强制)

    实现注意:
        - torch.cdist 算全量距离矩阵 (B, H, N), 然后 topk
        - topk 的索引选择不可微, 但选出来的距离值可微 (标准 Chamfer trick)
        - N=4096, H=16: cdist 矩阵 (B, 16, 4096), B=60 时约 60×16×4096×4byte ≈ 15MB, 可接受
    """
    import math

    B, H, _ = waypoints.shape
    N = point_cloud.shape[1]

    # 全量距离矩阵 (B, H, N)
    dists = torch.cdist(waypoints, point_cloud, p=2.0)  # (B, H, N)

    # 取 k 个最近邻的距离 (B, H, k)
    k_actual = min(k, N)
    d_knn, _ = torch.topk(dists, k=k_actual, dim=-1, largest=False)  # (B, H, k)

    # 数值 floor (防 sqrt(0) 梯度奇异)
    d_knn = torch.sqrt(d_knn * d_knn + eps)

    # soft-min via LogSumExp + 偏差修正
    #   d_soft = -1/τ · logsumexp(-τ d_knn) + log(k)/τ
    log_k_over_tau = math.log(k_actual) / tau  # 偏差修正项
    d_soft = -(1.0 / tau) * torch.logsumexp(-tau * d_knn, dim=-1) + log_k_over_tau  # (B, H)

    # 物理护栏: 距离不可能为负 (即使修正后, τ 有限时仍有微小残余偏差可能小于 0)
    d_soft = d_soft.clamp(min=0.0)

    return d_soft


def collision_auxiliary_loss(
    pred_x0_action,
    point_cloud,
    cur_pos,
    d_safe=0.75,
    beta=None,
    tau_softmin=5.0,
    k_nn=8,
    horizon_decay_tau=8.0,
    dpos_slice=(0, 3),
    return_diagnostics=False,
    sin_yaw=None,
    cos_yaw=None,
):
    """
    Phase ② collision auxiliary loss 主函数。

    [BUG FIX 2026-05-15] c_per_waypoint 加上限 clamp (防单点爆炸)
    [v1.2 §5.2] tau_softmin 默认值锁定为 5.0
    [BUG FIX 2026-08-14] 坐标系一致性: 新增 body→world 点云变换 (sin_yaw/cos_yaw)

    Args:
        pred_x0_action: (B, H, action_dim) — 模型预测的 clean trajectory (UN-normalized)
                        注意: 必须是反归一化后的 action, 因为距离计算需要物理米制
        point_cloud:    (B, N, 3 或 6)     — 点云 (body frame; 传入本函数后变换到 world frame)
                        若 6 维 (xyz+rgb), 自动取前 3 维
        cur_pos:        (B, 3)             — 当前 drone 位置 (world frame)
        d_safe:         float             — 安全距离阈值 (m). Phase ② = 0.75m
        beta:           float or None     — softplus 锐度. None 时默认 = 1/0.05 = 20.0
        tau_softmin:    float             — k-NN soft-min 锐度 (默认 20.0, bug fix 后值)
        k_nn:           int               — k-NN 近邻数
        horizon_decay_tau: float          — 横向加权 w_k = exp(-k/τ) 的 τ
        dpos_slice:     tuple             — action 中 dpos_world 的通道范围, 默认 (0,3)
        return_diagnostics: bool          — True 时额外返回诊断 dict
        sin_yaw:        (B,) or None      — [FIX 2026-08-14] 当前偏航角正弦 (agent_pos[..., 7])
        cos_yaw:        (B,) or None      — [FIX 2026-08-14] 当前偏航角余弦 (agent_pos[..., 8])

    Returns:
        loss: scalar tensor — collision auxiliary loss
        (可选) diagnostics: dict

    坐标系说明 (v1.2 §5.2):
        waypoints 由 dpos_world cumsum 得到, 是 world frame 绝对坐标;
        point_cloud 是 body frame 相对坐标。两者必须同坐标系才能 cdist。
        修复后: pc_world = R(yaw) @ pc_body + cur_pos (可微, 见 body_to_world)。
        sin_yaw/cos_yaw 为 None 时跳过变换 (兼容旧调用, 但不保证距离正确)。
    """
    if beta is None:
        beta = 20.0  # 1/0.05m 过渡宽度

    B, H, _ = pred_x0_action.shape
    device = pred_x0_action.device

    # 入口 NaN 防护: 输入含 NaN/Inf 时直接返回 0
    if not torch.isfinite(pred_x0_action).all() or not torch.isfinite(point_cloud).all():
        if return_diagnostics:
            return torch.tensor(0.0, device=device), {}
        return torch.tensor(0.0, device=device)

    # 取 dpos_world 通道
    s, e = dpos_slice
    dpos_world = pred_x0_action[..., s:e]  # (B, H, 3)

    # 点云取前 3 维 (xyz)
    if point_cloud.shape[-1] > 3:
        pc_xyz = point_cloud[..., :3]
    else:
        pc_xyz = point_cloud
    pc_xyz = pc_xyz.to(device)

    # [FIX 2026-08-14] body→world 变换: 点云是 body frame, waypoints 是 world frame。
    #   必须旋转平移到 world frame 再算距离 (v1.2 §5.2), 否则 yaw≠0 时距离系统性失真、
    #   cur_pos 远离原点时惩罚完全失效 (实测 cur_pos=(10,8,2) 时 loss=0.0)。
    if sin_yaw is not None and cos_yaw is not None:
        pc_xyz = body_to_world(
            pc_xyz, cur_pos.to(device),
            sin_yaw.to(device), cos_yaw.to(device))

    # cumsum → world waypoints
    waypoints = cumsum_dpos_to_waypoints(dpos_world, cur_pos.to(device))  # (B, H, 3)

    # k-NN soft-min 距离
    d_soft = knn_softmin_distance(
        waypoints, pc_xyz, k=k_nn, tau=tau_softmin)  # (B, H)

    # softplus-hinge 惩罚
    # c_t = softplus(β·(d_safe − d_soft)) / β
    #   d_soft >> d_safe → c ≈ 0
    #   d_soft << d_safe → c ≈ (d_safe − d_soft)
    #
    # 数值稳定性: 当 d_soft << d_safe 时, β·(d_safe − d_soft) 可能很大,
    #   softplus(x) ≈ x (x >> 0), 梯度 ≈ 1, 无问题
    # 当 d_soft >> d_safe 时, β·(d_safe − d_soft) 很负,
    #   softplus(x) ≈ exp(x) ≈ 0, 梯度 ≈ exp(x) ≈ 0, 无问题
    # 关键: β 控制过渡区锐度, 梯度在过渡区内最大为 1 (softplus 的 sigmoid 导数 ≤ 1)
    # 但 β 放大了输入, 使过渡区更窄, 梯度变化更陡 → 需要限制 β·(d_safe − d_soft) 的范围
    margin = d_safe - d_soft  # (B, H) 正值=违规, 负值=安全
    # clamp margin 防止极端值: 当 d_soft 为负或极小时 margin 会爆炸
    margin = margin.clamp(max=2.0 * d_safe)  # 最多惩罚到 2×d_safe 距离
    c_per_waypoint = F.softplus(beta * margin) / beta  # (B, H)
    # [BUG FIX 2026-05-15] 单 waypoint c 的上限 ≈ d_safe (margin clamp 后 softplus 的上界)
    #   双重护栏: 即使 margin clamp 失效, c 也不会超过这个值
    c_per_waypoint = c_per_waypoint.clamp(max=2.0 * d_safe)

    # 横向加权 w_k = exp(-k / τ)
    k_idx = torch.arange(H, device=device, dtype=c_per_waypoint.dtype)
    w_k = torch.exp(-k_idx / horizon_decay_tau)  # (H,)
    w_k = w_k / w_k.sum()  # 归一化, 保证 loss 尺度与 d_safe 同量级

    # 加权平均 (per batch, 再 batch 平均)
    loss_per_sample = (c_per_waypoint * w_k.unsqueeze(0)).sum(dim=1)  # (B,)
    loss = loss_per_sample.mean()

    # [BUG FIX 2026-05-15] 输出 NaN/Inf 兜底: 任何环节产生 NaN 都不让它进总 loss
    if not torch.isfinite(loss):
        if return_diagnostics:
            return torch.tensor(0.0, device=device, requires_grad=True), {
                'd_soft_p50': float('nan'), 'd_soft_p05': float('nan'),
                'd_soft_min': float('nan'),
                'c_per_waypoint_mean': float('nan'),
                'c_per_waypoint_p95': float('nan'),
                'n_violating_share': 0.0,
                'nan_fallback': 1.0,  # 标记: 这一 batch loss 被兜底为 0
            }
        return torch.tensor(0.0, device=device, requires_grad=True)

    if not return_diagnostics:
        return loss

    # 诊断信息
    with torch.no_grad():
        diag = {
            'd_soft_p50': torch.quantile(d_soft, 0.50).item(),
            'd_soft_p05': torch.quantile(d_soft, 0.05).item(),
            'd_soft_min': d_soft.min().item(),
            'c_per_waypoint_mean': c_per_waypoint.mean().item(),
            'c_per_waypoint_p95': torch.quantile(c_per_waypoint, 0.95).item(),
            # 距离 < d_safe 的 waypoint 比例 (违规率)
            'n_violating_share': (d_soft < d_safe).float().mean().item(),
        }
    return loss, diag


# ──────────────────────────────────────────────────────────────────────────
# Tweedie x̂₀ 恢复 (从 ε-prediction)
# ──────────────────────────────────────────────────────────────────────────

def recover_x0_from_eps(noisy, pred_eps, timesteps, noise_scheduler):
    """
    从 ε-prediction 恢复 clean trajectory x̂₀ (Tweedie estimate)。

    Args:
        noisy:      (B, H, Da) — 加噪后的轨迹
        pred_eps:   (B, H, Da) — 模型预测的噪声 (ε-prediction)
        timesteps:  (B,)       — 每个样本的扩散时间步
        noise_scheduler:       — diffusers 的 scheduler (提供 alphas_cumprod)

    Returns:
        x0_hat: (B, H, Da) — 预测的 clean trajectory (NORMALIZED space, 与 noisy 同空间)

    数学 (DDPM 标准 Tweedie):
        x̂₀ = (noisy − √(1−ᾱ_t)·ε_pred) / √ᾱ_t

    注意:
        - 返回的 x̂₀ 在 NORMALIZED 空间 (因为 noisy 是 normalized action 加噪)
        - 调用方需要用 normalizer['action'].unnormalize() 转回物理米制
        - 才能传给 collision_auxiliary_loss (距离计算需要物理坐标)
    """
    device = noisy.device
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)  # (T_full,)

    # 取每个样本对应 timestep 的 ᾱ_t
    alpha_bar = alphas_cumprod[timesteps]  # (B,)
    alpha_bar = alpha_bar.view(-1, 1, 1)   # (B, 1, 1) for broadcast

    sqrt_alpha_bar = torch.sqrt(alpha_bar)
    sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

    x0_hat = (noisy - sqrt_one_minus_alpha_bar * pred_eps) / sqrt_alpha_bar
    return x0_hat


# ──────────────────────────────────────────────────────────────────────────
# 单元测试 / numpy reference 验证
# ──────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import numpy as np

    print("=" * 60)
    print("collision_loss.py — self-test")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)

    B, H, N = 4, 16, 4096
    action_dim = 11

    # --- Test 1: cumsum_dpos_to_waypoints ---
    print("\n[Test 1] cumsum_dpos_to_waypoints")
    dpos = torch.randn(B, H, 3) * 0.1
    cur_pos = torch.tensor([[0.0, 0.0, 2.0]] * B)
    wp = cumsum_dpos_to_waypoints(dpos, cur_pos)
    # numpy reference
    wp_np_ref = cur_pos.numpy()[:, None, :] + np.cumsum(dpos.numpy(), axis=1)
    err = np.abs(wp.numpy() - wp_np_ref).max()
    print(f"  cumsum max error vs numpy: {err:.2e}  {'PASS' if err < 1e-5 else 'FAIL'}")

    # --- Test 2: knn_softmin_distance ---
    print("\n[Test 2] knn_softmin_distance (sparse-neighbor scenario)")
    # 构造: waypoint 在原点, 点云有 2 个点近 (0.5m / 0.6m), 其余远
    # [说明] 偏差修正项 log(k)/τ ≈ 0.42m 在邻居稀疏时会"过修正" — 真实 min=0.5m
    #        但报告偏高. 这是 trade-off: dense-neighbor (真实障碍场景) 精确,
    #        sparse-neighbor 偏高. **关键: 永远不会报负数** (修复前会).
    #        密集场景见 Test 6 (与 pilot 实际场景一致).
    wp_test = torch.zeros(1, 1, 3)
    pc_test = torch.randn(1, N, 3) * 10.0 + 20.0  # 全部很远
    pc_test[0, 0] = torch.tensor([0.5, 0.0, 0.0])  # 一个点在 0.5m
    pc_test[0, 1] = torch.tensor([0.6, 0.0, 0.0])  # 一个点在 0.6m
    d_soft = knn_softmin_distance(wp_test, pc_test, k=8, tau=5.0)
    print(f"  d_soft = {d_soft.item():.4f} (期望 ≥ 0, 含偏差修正)")
    print(f"  非负性: {'PASS' if d_soft.item() >= 0 else 'FAIL'}")
    print(f"  范围内: {'PASS' if 0.45 <= d_soft.item() <= 1.0 else 'FAIL'}")

    # --- Test 3: collision_auxiliary_loss 梯度流 ---
    print("\n[Test 3] collision_auxiliary_loss 梯度反传")
    pred_x0 = torch.randn(B, H, action_dim, requires_grad=True)
    pc = torch.randn(B, N, 6) * 5.0  # xyz+rgb
    cur_p = torch.zeros(B, 3)
    loss, diag = collision_auxiliary_loss(
        pred_x0, pc, cur_p, d_safe=0.75, return_diagnostics=True)
    loss.backward()
    grad_ok = pred_x0.grad is not None and torch.isfinite(pred_x0.grad).all()
    # 梯度应该只在 dpos 通道 (0:3) 非零, 其他通道为 0
    grad_dpos = pred_x0.grad[..., 0:3].abs().sum().item()
    grad_other = pred_x0.grad[..., 3:].abs().sum().item()
    print(f"  loss = {loss.item():.4f}")
    print(f"  grad finite: {grad_ok}  {'PASS' if grad_ok else 'FAIL'}")
    print(f"  grad on dpos channels: {grad_dpos:.4f} (应 > 0)")
    print(f"  grad on other channels: {grad_other:.2e} (应 = 0)")
    print(f"  {'PASS' if grad_dpos > 0 and grad_other < 1e-6 else 'FAIL'}")
    print(f"  diagnostics: {diag}")

    # --- Test 4: d_safe 单调性 (近障碍 loss 大) ---
    print("\n[Test 4] d_safe 单调性")
    # 场景 A: waypoint 紧贴障碍 (距离 0.1m)
    wp_close = torch.zeros(1, H, 3)
    pc_close = torch.ones(1, N, 3) * 0.1  # 所有点在 0.1m 附近
    loss_close = collision_auxiliary_loss(
        torch.zeros(1, H, action_dim), pc_close,
        torch.zeros(1, 3), d_safe=0.75)
    # 场景 B: waypoint 远离障碍 (距离 5m)
    pc_far = torch.ones(1, N, 3) * 5.0
    loss_far = collision_auxiliary_loss(
        torch.zeros(1, H, action_dim), pc_far,
        torch.zeros(1, 3), d_safe=0.75)
    print(f"  loss (紧贴障碍 0.1m): {loss_close.item():.4f}")
    print(f"  loss (远离障碍 5m):   {loss_far.item():.6f}")
    print(f"  {'PASS' if loss_close.item() > loss_far.item() * 10 else 'FAIL'}"
          f" (近障碍 loss 应远大于远障碍)")

    # --- Test 5: recover_x0_from_eps ---
    print("\n[Test 5] recover_x0_from_eps (Tweedie)")
    # mock noise scheduler
    class MockScheduler:
        def __init__(self):
            T = 50
            betas = torch.linspace(1e-4, 0.02, T)
            alphas = 1.0 - betas
            self.alphas_cumprod = torch.cumprod(alphas, dim=0)
    sched = MockScheduler()
    x0_true = torch.randn(B, H, action_dim)
    eps_true = torch.randn(B, H, action_dim)
    ts = torch.randint(0, 50, (B,))
    ab = sched.alphas_cumprod[ts].view(-1, 1, 1)
    noisy = torch.sqrt(ab) * x0_true + torch.sqrt(1 - ab) * eps_true
    # 用真实 eps 恢复, 应得到真实 x0
    x0_rec = recover_x0_from_eps(noisy, eps_true, ts, sched)
    rec_err = (x0_rec - x0_true).abs().max().item()
    print(f"  x0 恢复误差 (用真实 eps): {rec_err:.2e}")
    print(f"  {'PASS' if rec_err < 1e-4 else 'FAIL'}")

    # --- Test 6: soft-min 偏差修复（pilot NaN bug regression test）---
    print("\n[Test 6] soft-min 偏差修复 (pilot NaN bug regression test)")
    print("  [背景] 修复前 τ=5, k=8 个邻居都在 0.1m 时, soft-min 报 -0.316m (负数!)")
    print("         导致 softplus(β·(0.75−(−0.3))) = softplus(21) ≈ 21, 梯度爆炸 NaN")
    print()

    # 场景: 8 个邻居全部紧贴 0.1m (与 pilot 现场一致)
    wp_clustered = torch.zeros(1, 1, 3)
    pc_clustered = torch.randn(1, N, 3) * 100.0 + 50.0  # 大部分点远
    # 8 个点都在距离 0.1m 处 (各方向)
    for i, dir_vec in enumerate([
        [0.10, 0, 0], [-0.10, 0, 0], [0, 0.10, 0], [0, -0.10, 0],
        [0, 0, 0.10], [0, 0, -0.10], [0.07, 0.07, 0], [-0.07, -0.07, 0]
    ]):
        pc_clustered[0, i] = torch.tensor(dir_vec)

    # §5.2 τ=5.0 (偏差修正 + clamp 保证非负)
    d_soft_fixed = knn_softmin_distance(wp_clustered, pc_clustered, k=8, tau=5.0)
    err_vs_true_min = abs(d_soft_fixed.item() - 0.10)
    print(f"  修复后 d_soft (τ=5, k=8, 真实 min=0.10m): {d_soft_fixed.item():.4f} m")
    print(f"  误差 vs 真实 min: {err_vs_true_min:.4f} m")
    print(f"  非负性: d_soft ≥ 0 → {'PASS' if d_soft_fixed.item() >= 0 else 'FAIL'}")
    print(f"  精度: 误差 < 0.5m → {'PASS' if err_vs_true_min < 0.5 else 'FAIL'}")

    # 测试 collision loss 在该情景下不爆炸 (pilot 现场: d_safe=0.75, β=20)
    pred_x0_pilot = torch.zeros(1, 16, 11, requires_grad=True)
    cur_p_pilot = torch.zeros(1, 3)
    loss_pilot, diag_pilot = collision_auxiliary_loss(
        pred_x0_pilot, pc_clustered.expand(1, N, 3),
        cur_p_pilot, d_safe=0.75, beta=20.0, tau_softmin=5.0,
        return_diagnostics=True)
    print(f"  pilot 情景 loss: {loss_pilot.item():.4f}")
    print(f"  pilot 情景 d_soft_p05: {diag_pilot.get('d_soft_p05', float('nan')):.4f} m"
          f" (修复前应为 ~-0.30m)")
    loss_pilot.backward()
    grad_max = pred_x0_pilot.grad.abs().max().item()
    grad_finite = torch.isfinite(pred_x0_pilot.grad).all().item()
    print(f"  pilot 情景 grad finite: {grad_finite}, max |grad|: {grad_max:.4f}")
    print(f"  {'PASS' if grad_finite and grad_max < 100 else 'FAIL'}"
          f" (梯度有限且不爆炸)")

    print("\n" + "=" * 60)
    print("self-test 完成")
    print("=" * 60)