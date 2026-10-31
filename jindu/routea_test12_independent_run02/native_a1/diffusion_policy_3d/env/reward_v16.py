"""
reward_v16.py — Reward v1.6 (E038 修订): 2-Component Action-Dependent Reward

Design: DPPO_Structural_Refactoring_v2_0.md §3.1（F6 定案 2026-08-03）
Based on CaRL 2025: ≤3 components better than ≥5 components.
DPPO paper uses standard Gym dense reward (1 component).

  r = r_progress + r_smooth + terminal

2026-08-03 E038 修订：r_safety（绝对/delta/gate-delta 全形态）删除——
r_safety 家族三连败定案（E031 空旷巡航 / E036 逃离陷阱 / E037 绕圈不完成）；
C2 目标=障碍密集区 → 任何 clearance 型 shaping 都惩罚任务行为（设计 v2.0 §3.1）。
clearance 计算保留（metrics d_soft_h/d_min，供 monitor 行为判据）。

2026-08-04 E039 修订（G13 校准，设计 v1.1 §4.2，D4 已裁决）：r_collision −50 → −20。
E038 判读 = 接近目标区期望成本（−50·P_collision）> progress 收益 → 漂移是经济理性；
−20 使穿障到达经济可行（reach/collision 比 2:1 → 5:1）。r_reach 保持 100。
备份：program/backup/reward_v16_pre_e039.py

主分量 action-dependent:
  - r_progress uses action_chunk to estimate forward progress
  - r_smooth uses action_chunk jerk (2nd-order dpos differences)
"""

import math
import numpy as np
from typing import Dict, Optional, Tuple


# ═══════════════════════════════════════════════════════════════
# Default configuration
# ═══════════════════════════════════════════════════════════════

REWARD_V16_DEFAULTS = {
    # ── r_progress: goal-directed progress (action-dependent) ──
    "w_progress": 2.0,
    "r_reach": 100.0,
    # E039 (G13 校准, 2026-08-04, 设计 v1.1 §4.2): −50 → −20（D4 裁决；reach/collision 5:1）
    "r_collision": -20.0,

    # ── clearance metrics (E038: r_safety 已删，保留计算供 monitor) ──
    "h_future": 8,                # lookahead steps
    "softmin_beta": 8.0,
    "time_decay": 0.15,

    # ── r_smooth: action-dependent jerk penalty ──
    "w_smooth": 0.2,
}


# ═══════════════════════════════════════════════════════════════
# Geometry helpers (minimal, reused from v14)
# ═══════════════════════════════════════════════════════════════

def _reconstruct_future_positions(pos_world: np.ndarray, action_chunk: np.ndarray,
                                   h_future: int) -> np.ndarray:
    """Reconstruct future positions from action delta cumsum."""
    from diffusion_policy_3d.env.traj_utils import DPOS_W, POS

    if action_chunk.shape[0] < h_future:
        h_future = action_chunk.shape[0]

    # R5.8-C C7: the "absolute-position" branch is gone — POS is the
    # first step's DELTA in the raw 11-dim action (traj_utils), so it
    # was dead unless the drone sat within 1 mm of the origin, and then
    # it returned an all-zero path (corrupting d_soft_h).
    dpos = action_chunk[:h_future, DPOS_W]
    cum = np.cumsum(dpos, axis=0)
    return pos_world.reshape(1, 3) + cum


def _temporal_softmin_clearance(future_positions: np.ndarray, obstacles,
                                 beta: float = 8.0, time_decay: float = 0.15) -> float:
    """Temporal soft-min clearance over future waypoints."""
    from diffusion_policy_3d.env.collision_geom import compute_obstacle_clearance

    H = len(future_positions)
    raw_weights = np.array([math.exp(-time_decay * i) for i in range(H)])
    weights = raw_weights / max(raw_weights.sum(), 1e-12)

    distances = np.zeros(H, dtype=np.float64)
    for i in range(H):
        d_i, _, _ = compute_obstacle_clearance(
            float(future_positions[i, 0]), float(future_positions[i, 1]),
            float(future_positions[i, 2]), obstacles, check_3d=True)
        distances[i] = d_i if d_i is not None else 100.0

    exp_terms = weights * np.exp(-beta * distances)
    sum_exp = np.sum(exp_terms)
    if sum_exp < 1e-30:
        return float(np.min(distances))
    return float(max(-1.0 / beta * math.log(sum_exp), 0.0))


def _compute_instant_clearance(pos, obstacles) -> float:
    """Single-point obstacle clearance."""
    from diffusion_policy_3d.env.collision_geom import compute_obstacle_clearance
    d, _, _ = compute_obstacle_clearance(
        float(pos[0]), float(pos[1]), float(pos[2]), obstacles, check_3d=True)
    return d if d is not None else 100.0


# ═══════════════════════════════════════════════════════════════
# Main reward function
# ═══════════════════════════════════════════════════════════════

def compute_reward_v16(
    s_t: dict,
    action_chunk: np.ndarray,
    s_tp1: dict,
    obstacle_json=None,
    goal_world: Optional[np.ndarray] = None,
    cfg: Optional[dict] = None,
) -> Tuple[float, dict, dict]:
    """Reward v1.6: 2-component action-dependent reward (E038: r_safety 已删, F6).

    Args:
        s_t: current state with 'pos_world'(3,), 'yaw'(scalar), 'vel_world'(3,)
        action_chunk: (H, 11) action output
        s_tp1: next state with 'pos_world'(3,), 'yaw'(scalar), 'vel_world'(3,),
               'collided'(bool)
        obstacle_json: obstacle list for clearance queries
        goal_world: (3,) goal position
        cfg: configuration overrides

    Returns:
        (reward, terms, metrics)
    """
    if cfg is None:
        cfg = {}
    c = {**REWARD_V16_DEFAULTS, **cfg}

    pos_t = s_t.get("pos_world", np.zeros(3))
    pos_tp1 = s_tp1.get("pos_world", np.zeros(3))
    gw = goal_world if goal_world is not None else np.array([8.0, 0.0, 2.0])
    obstacles = obstacle_json if obstacle_json is not None else []

    dist_t = float(np.linalg.norm(pos_t - gw))
    dist_tp1 = float(np.linalg.norm(pos_tp1 - gw))
    # R5.8-C (N5/P1-2): reached is ENV-AUTHORITATIVE — the env's own
    # termination criterion (dist < goal_tol_xy AND |dz| < goal_tol_z
    # AND v_xy < 1.0), not a distance-only re-inference. The env
    # terminates on the step it becomes reached, so the bonus is
    # terminal-gated by construction (the "park at z+0.4 and harvest
    # 100/step" exploit dies here).
    reached = bool(s_tp1.get("reached", False))
    collided = s_tp1.get("collided", False)

    terms = {"progress": 0.0, "smooth": 0.0}  # E038: "safety" 键删除（F6）
    metrics = {"d_soft_h": -1.0, "d_min": 100.0, "dist": dist_tp1}

    # ═══════════════════════════════════════════════════════════
    # 1. r_progress — action-dependent goal progress
    # ═══════════════════════════════════════════════════════════
    # Use actual displacement Δdist, which IS action-dependent
    # (policy's action_chunk determines where the drone goes)
    r_progress = c["w_progress"] * (dist_t - dist_tp1)

    # Terminal bonuses (only at episode end)
    # R5.8-C (N5 counterexample): collision + goal co-fire must NEVER
    # pay the reach bonus — a crashed episode is not a success.
    if reached and not collided:
        r_progress += c["r_reach"]
    if collided:
        r_progress += c["r_collision"]

    terms["progress"] = r_progress
    metrics["dist"] = dist_tp1

    # ═══════════════════════════════════════════════════════════
    # 2. clearance metrics（E038 F6: r_safety 已删除——见文件头）
    #    仅保留计算供 monitor 行为判据（d_soft 漂移检测）
    # ═══════════════════════════════════════════════════════════
    future_pos = _reconstruct_future_positions(pos_t, action_chunk, c["h_future"])
    d_soft_h = _temporal_softmin_clearance(
        future_pos, obstacles, beta=c["softmin_beta"], time_decay=c["time_decay"])
    d_soft_h = float(np.clip(d_soft_h, 0.01, 50.0))
    d_min = _compute_instant_clearance(pos_tp1, obstacles)
    d_min = float(np.clip(d_min, 0.01, 50.0))

    metrics["d_soft_h"] = d_soft_h
    metrics["d_min"] = d_min

    # ═══════════════════════════════════════════════════════════
    # 3. r_smooth — action-dependent jerk penalty
    # ═══════════════════════════════════════════════════════════
    if len(action_chunk) >= 3:
        from diffusion_policy_3d.env.traj_utils import DPOS_W
        dpos = action_chunk[:, DPOS_W]  # (H, 3)
        # 1st-order: velocity (H-1, 3)
        vel = np.diff(dpos, axis=0)
        # 2nd-order: acceleration/jerk (H-2, 3)
        jerk = np.diff(vel, axis=0)
        jerk_norm = float(np.mean(np.abs(jerk)))
        r_smooth = -c["w_smooth"] * jerk_norm / (1.0 + jerk_norm)
    else:
        r_smooth = -0.05  # minimal penalty when can't compute
    terms["smooth"] = r_smooth

    # ═══════════════════════════════════════════════════════════
    # 4. Total
    # ═══════════════════════════════════════════════════════════
    reward = float(sum(terms.values()))
    return reward, terms, metrics
