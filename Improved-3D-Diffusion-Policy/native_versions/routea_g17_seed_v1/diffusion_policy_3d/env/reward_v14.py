"""
reward_v14.py — Reward v1.4: Adaptive-Threshold Multi-Objective Reward

Design: 4 components replacing v1.3's 8 components.
  r = r_goal + r_safety + r_explore + r_recovery

Key design change from v1.3:
  - Episode-adaptive thresholds (EMA of d_soft^H) instead of fixed absolute values
  - Looser stall detection so recovery signals actually activate
  - No offline_density dependency for exploration
  - Removed: r_pbrs (negative), r_guard (folded), r_flip (folded)
"""

import math
import numpy as np
from collections import deque
from typing import Dict, Optional, Tuple, Any, Callable

# ═══════════════════════════════════════════════════════════════
# Default configuration
# ═══════════════════════════════════════════════════════════════

REWARD_V14_DEFAULTS = {
    # ── r_goal (same as v1.3 r_base) ──
    "w_goal": 2.0,
    "r_reach": 100.0,
    "r_collision": 50.0,
    "w_clearance": 2.5,
    "clear_cap": 1.0,
    "w_alt": 20.0,
    "z_min": 1.0,
    "w_smooth": 0.01,
    "w_z": 0.5,

    # ── r_safety (future clearance with adaptive reference) ──
    "w_safe": 3.0,
    "s_safe": 1.0,             # tanh scale (m)
    "ema_alpha": 0.1,          # EMA smoothing for d_ref
    "h_future": 8,             # future steps for d_soft^H
    "softmin_beta": 8.0,
    "time_decay": 0.15,
    "epsilon_back": 0.05,      # min progress to qualify for r_safety

    # ── r_explore (novelty in goal-aligned grid) ──
    "w_explore": 0.5,
    "explore_bin": (2.0, 2.0, 1.0),  # (x_parallel, x_perp, z) bin sizes
    "explore_safe_threshold": 1.5,    # min d_soft^H to qualify
    "explore_episode_cap": 5.0,

    # ── r_recovery (stall detection + escape) ──
    "w_stall": 0.5,            # per-step penalty when stalled
    "w_escape": 2.0,           # one-time bonus on escape
    "stall_window": 8,         # detection window (steps)
    "stall_progress": 0.5,     # max progress over window to flag stall (m)
    "stall_d_soft": 2.0,       # max d_soft^H over window to flag stall (m)
    "stall_n_flip": 1,         # min lateral sign changes for stall
    "epsilon_recover_progress": 0.5,  # min progress for escape
    "epsilon_recover_clearance": 0.15, # min clearance improvement for escape
}


# ═══════════════════════════════════════════════════════════════
# History management
# ═══════════════════════════════════════════════════════════════

def create_reward_history_v14(cfg: Optional[dict] = None) -> dict:
    """Create fresh episode history for v1.4 reward."""
    if cfg is None:
        cfg = {}
    c = {**REWARD_V14_DEFAULTS, **cfg}
    w = max(c["stall_window"], 8)
    return {
        # r_safety: EMA of d_soft^H
        "d_soft_h_ema": -1.0,  # -1 = uninitialized
        # r_recovery: sliding windows
        "progress": deque(maxlen=w),
        "d_soft_h_vals": deque(maxlen=w),
        "d_min_vals": deque(maxlen=w),
        "m_t_vals": deque(maxlen=w),      # lateral sign history
        # stall state
        "was_stalled": False,
        "stall_segment_len": 0,
        "stall_start_d_soft_h": 100.0,
        "escape_paid": False,
        # r_explore: grid visit counter
        "visit_count": {},
        "new_cells_this_episode": set(),
        "explore_accumulated": 0.0,
        "total_steps": 0,
    }


# ═══════════════════════════════════════════════════════════════
# Geometry helpers (reuse from v1.3 or reimplement minimal versions)
# ═══════════════════════════════════════════════════════════════

def _reconstruct_future_positions(pos_world: np.ndarray, action_chunk: np.ndarray,
                                   h_future: int) -> np.ndarray:
    """Reconstruct future positions from action delta cumsum."""
    from diffusion_policy_3d.env.traj_utils import DPOS_W, POS

    if action_chunk.shape[0] < h_future:
        h_future = action_chunk.shape[0]

    # Check if action_chunk contains absolute positions
    if action_chunk.shape[1] >= 10 and np.any(np.abs(action_chunk[0, POS]) > 0):
        first_pos = action_chunk[0, POS]
        if np.linalg.norm(first_pos - pos_world) < 1e-3:
            return action_chunk[:h_future, POS].copy()

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


def _compute_signed_lateral(future_positions: np.ndarray, pos_world: np.ndarray,
                             goal_world: np.ndarray) -> float:
    """Signed lateral displacement (positive = left of goal direction)."""
    goal_dir = goal_world - pos_world
    goal_dist = np.linalg.norm(goal_dir[:2])
    if goal_dist < 1e-6:
        return 0.0
    goal_unit = goal_dir / goal_dist
    signed_values = []
    for i in range(len(future_positions)):
        to_point = future_positions[i] - pos_world
        proj = np.dot(to_point, goal_unit)
        lateral_vec = to_point - proj * goal_unit
        lateral_norm = np.linalg.norm(lateral_vec)
        cross_z = goal_unit[0] * lateral_vec[1] - goal_unit[1] * lateral_vec[0]
        sign = 1.0 if cross_z >= 0 else -1.0
        signed_values.append(sign * lateral_norm)
    return float(np.mean(signed_values)) if signed_values else 0.0


def _compute_smoothness(action_chunk: np.ndarray) -> float:
    """Trajectory smoothness from dpos 1st-order differences."""
    if len(action_chunk) < 2:
        return 0.0
    from diffusion_policy_3d.env.traj_utils import DPOS_W
    dpos = action_chunk[:, DPOS_W]
    return float(np.sum(np.abs(np.diff(dpos, axis=0))))


def _compute_explore_cell(pos_world, goal_world, bin_size):
    """Goal-aligned cell index for exploration grid."""
    rel = pos_world - goal_world
    goal_norm = np.linalg.norm(goal_world)
    if goal_norm < 1e-6:
        return (0, 0, 0)
    goal_unit = goal_world / goal_norm
    x_parallel = np.dot(rel, goal_unit)
    x_perp = np.linalg.norm(rel[:2] - x_parallel * goal_unit[:2])
    cross_z = goal_unit[0] * rel[1] - goal_unit[1] * rel[0]
    x_perp_signed = x_perp if cross_z >= 0 else -x_perp
    return (int(round(x_parallel / bin_size[0])),
            int(round(x_perp_signed / bin_size[1])),
            int(round(pos_world[2] / bin_size[2])))


def _compute_m_t(signed_lateral: float, epsilon_m: float = 0.10) -> int:
    """Lateral sign {-1, 0, +1}."""
    if signed_lateral > epsilon_m:
        return 1
    elif signed_lateral < -epsilon_m:
        return -1
    return 0


# ═══════════════════════════════════════════════════════════════
# Main reward function
# ═══════════════════════════════════════════════════════════════

def compute_reward_v14(
    s_t: dict,
    action_chunk: np.ndarray,
    s_tp1: dict,
    obstacle_json: Any = None,
    goal_world: Optional[np.ndarray] = None,
    history: Optional[dict] = None,
    cfg: Optional[dict] = None,
) -> Tuple[float, dict, dict, dict]:
    """Reward v1.4: 4-component adaptive multi-objective reward.

    Args:
        s_t: current state with 'pos_world'(3,), 'yaw'(scalar), 'vel_world'(3,)
        action_chunk: (H, 11) FM output action
        s_tp1: next state with 'pos_world'(3,), 'yaw'(scalar), 'vel_world'(3,),
               'collided'(bool)
        obstacle_json: obstacle list for clearance queries
        goal_world: (3,) goal position
        history: episode history dict (mutated in-place)
        cfg: configuration overrides

    Returns:
        (reward, terms, metrics, updated_history)
    """
    # ── Config ──
    if cfg is None:
        cfg = {}
    c = {**REWARD_V14_DEFAULTS, **cfg}

    # ── State extraction ──
    pos_t = s_t.get("pos_world", np.zeros(3))
    pos_tp1 = s_tp1.get("pos_world", np.zeros(3))
    z_t = float(pos_t[2])
    z_tp1 = float(pos_tp1[2])
    gw = goal_world if goal_world is not None else np.array([8.0, 0.0, 2.0])

    if history is None:
        history = create_reward_history_v14(cfg)
    h = history

    # ── Init outputs ──
    terms = {"goal": 0.0, "safety": 0.0, "explore": 0.0, "recovery": 0.0}
    metrics = {
        "d_soft_h": -1.0, "d_ref": -1.0, "d_min": 100.0,
        "stall_active": False, "stall_recovered": False,
        "low_density_cell": False, "novel_cell": False,
    }

    obstacles = obstacle_json if obstacle_json is not None else []

    # ═══════════════════════════════════════════════════════════
    # 1. Compute shared geometric quantities
    # ═══════════════════════════════════════════════════════════
    future_pos = _reconstruct_future_positions(pos_t, action_chunk, c["h_future"])
    d_soft_h = _temporal_softmin_clearance(
        future_pos, obstacles, beta=c["softmin_beta"], time_decay=c["time_decay"])
    # Clip to reasonable range (avoid inf-inf=NaN when no obstacles nearby)
    d_soft_h = float(np.clip(d_soft_h, 0.01, 50.0))
    d_min = _compute_instant_clearance(pos_tp1, obstacles)
    d_min = float(np.clip(d_min, 0.01, 50.0))
    signed_lat = _compute_signed_lateral(future_pos, pos_t, gw)
    m_t = _compute_m_t(signed_lat, epsilon_m=0.10)

    metrics["d_soft_h"] = d_soft_h
    metrics["d_min"] = d_min

    # ── Update EMA of d_soft^H (for adaptive reference) ──
    if h["d_soft_h_ema"] < 0:
        h["d_soft_h_ema"] = d_soft_h  # initialize
    else:
        h["d_soft_h_ema"] = (c["ema_alpha"] * d_soft_h
                             + (1.0 - c["ema_alpha"]) * h["d_soft_h_ema"])
    d_ref = h["d_soft_h_ema"]
    metrics["d_ref"] = d_ref

    dist_t = float(np.linalg.norm(pos_t - gw))
    dist_tp1 = float(np.linalg.norm(pos_tp1 - gw))
    # R5.8-C (N5): env-authoritative reached — the env's own
    # termination criterion, never a distance-only re-inference.
    reached = bool(s_tp1.get("reached", False))
    collided = s_tp1.get("collided", False)

    # ═══════════════════════════════════════════════════════════
    # 2. r_goal
    # ═══════════════════════════════════════════════════════════
    r_goal = 0.0
    r_goal += c["w_goal"] * (dist_t - dist_tp1)              # goal progress
    r_goal += (c["r_reach"] if (reached and not collided) else 0.0)               # terminal: success
    r_goal -= c["r_collision"] if collided else 0.0           # terminal: collision
    r_goal += c["w_clearance"] * min(float(d_min), c["clear_cap"])  # instant clearance
    if z_t < c["z_min"]:
        r_goal -= c["w_alt"]                                   # altitude penalty
    r_goal -= c["w_smooth"] * _compute_smoothness(action_chunk)  # smoothness
    if goal_world is not None:
        r_goal -= c["w_z"] * abs(z_tp1 - float(gw[2]))        # z deviation
    terms["goal"] = r_goal

    # ═══════════════════════════════════════════════════════════
    # 3. r_safety (adaptive-threshold future clearance)
    # ═══════════════════════════════════════════════════════════
    r_safety = 0.0
    progress = dist_t - dist_tp1
    if progress > -c["epsilon_back"] and d_ref > 0:
        # tanh: positive when d_soft^H > d_ref, negative when below
        r_safety = c["w_safe"] * math.tanh((d_soft_h - d_ref) / c["s_safe"])
    terms["safety"] = r_safety

    # ═══════════════════════════════════════════════════════════
    # 4. r_explore (grid-based novelty)
    # ═══════════════════════════════════════════════════════════
    r_explore = 0.0
    if d_soft_h > c["explore_safe_threshold"] and not collided:
        cell = _compute_explore_cell(pos_tp1, gw, c["explore_bin"])
        cell_key = str(cell)
        n_visit = h["visit_count"].get(cell_key, 0)
        novelty = 1.0 / math.sqrt(n_visit + 1)
        r_explore = c["w_explore"] * novelty
        # Episode cap
        remaining = c["explore_episode_cap"] - h["explore_accumulated"]
        r_explore = min(r_explore, max(0, remaining))
        if n_visit == 0:
            metrics["novel_cell"] = True
            h["new_cells_this_episode"].add(cell_key)
        h["visit_count"][cell_key] = n_visit + 1
        h["explore_accumulated"] += max(0, r_explore)
    terms["explore"] = r_explore

    # ═══════════════════════════════════════════════════════════
    # 5. r_recovery (stall detection + escape)
    # ═══════════════════════════════════════════════════════════
    r_recovery = 0.0

    # Update sliding windows
    h["progress"].append(dist_t - dist_tp1)
    h["d_soft_h_vals"].append(d_soft_h)
    h["d_min_vals"].append(d_min)
    h["m_t_vals"].append(m_t)

    # Stall detection
    stall_window = c["stall_window"]
    progress_hist = list(h["progress"])[-stall_window:] if len(h["progress"]) >= stall_window else []
    d_soft_hist = list(h["d_soft_h_vals"])[-stall_window:] if len(h["d_soft_h_vals"]) >= stall_window else []
    m_t_hist = list(h["m_t_vals"])[-stall_window:] if len(h["m_t_vals"]) >= stall_window else []

    total_progress = sum(progress_hist) if progress_hist else 999.0
    min_d_soft_window = min(d_soft_hist) if d_soft_hist else 100.0

    # Count lateral sign changes
    n_flip = 0
    non_zero = [m for m in m_t_hist if m != 0]
    for i in range(1, len(non_zero)):
        if non_zero[i] != non_zero[i-1]:
            n_flip += 1

    stall_active = (len(progress_hist) >= stall_window
                    and total_progress < c["stall_progress"]
                    and min_d_soft_window < c["stall_d_soft"]
                    and n_flip > c["stall_n_flip"])
    metrics["stall_active"] = stall_active

    # Stall state machine
    if not h["was_stalled"] and stall_active:
        h["was_stalled"] = True
        h["stall_segment_len"] = 1
        h["stall_start_d_soft_h"] = d_soft_h
        h["escape_paid"] = False
    elif h["was_stalled"] and stall_active:
        h["stall_segment_len"] += 1
    elif h["was_stalled"] and not stall_active:
        # Exiting stall: check escape
        if h["stall_segment_len"] >= 2:
            progress_ok = total_progress > c["epsilon_recover_progress"]
            clearance_ok = (d_soft_h - h["stall_start_d_soft_h"]
                            > c["epsilon_recover_clearance"])
            if progress_ok and clearance_ok:
                h["escape_paid"] = True
                metrics["stall_recovered"] = True
        h["was_stalled"] = False
        h["stall_segment_len"] = 0

    if stall_active:
        r_recovery -= c["w_stall"]

    if h["escape_paid"]:
        r_recovery += c["w_escape"]
        h["escape_paid"] = False  # one-shot

    terms["recovery"] = r_recovery

    # ═══════════════════════════════════════════════════════════
    # 6. Update history counters
    # ═══════════════════════════════════════════════════════════
    h["total_steps"] += 1

    # ═══════════════════════════════════════════════════════════
    # 7. Total
    # ═══════════════════════════════════════════════════════════
    reward = float(sum(terms.values()))
    return reward, terms, metrics, h
