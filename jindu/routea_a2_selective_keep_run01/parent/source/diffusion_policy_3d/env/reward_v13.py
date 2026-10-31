"""
reward_v13.py — Reward v1.3: Anticipatory Avoidance, Support Expansion, Dense-Obstacle Recovery

Design document: sheji/Reward_v1_3_Anticipatory_Support_DenseObstacle_Design_v1_2.md

r_t = r_base + r_pbrs + r_onset + r_fc + r_trap + r_flip + r_sup + r_guard

所有 numpy 操作，不需要 PyTorch（env 中 reward 在 CPU numpy 域计算）。
"""

import math, json, os
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple, Any, Callable

# Debug flag for high-frequency debug prints (default off for 10k training)
_DEBUG_PRINT = False

# 默认 offline density 路径（P0-1）
# 由 tools/build_offline_density.py 生成
# reward_v13.py → env/ → diffusion_policy_3d/ → project root
DEFAULT_DENSITY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data', 'offline_density.json'
)


# ═══════════════════════════════════════════════════════════════
# §0. 默认配置 (与设计文档 v1.2 一致)
# ═══════════════════════════════════════════════════════════════

REWARD_V13_DEFAULTS = {
    # ── base ──
    "w_goal": 2.0,
    "r_reach": 100.0,
    "r_collision": 50.0,
    "w_clearance": 2.5,
    "clear_cap": 1.0,
    "w_alt": 20.0,
    "z_min": 1.0,
    "w_smooth": 0.01,
    "w_z": 0.5,
    "w_yaw": 0.0,

    # ── future clearance (shared) ──
    "h_future": 8,
    "softmin_beta": 8.0,
    "time_decay": 0.15,
    "d_safe": 0.45,
    "d_ref": 0.75,

    # ── risk gate ──
    "tau_risk": 1.5,
    "s_tau": 0.4,
    "d_risk": 0.9,
    "s_d": 0.15,
    "risk_active_threshold": 0.35,

    # ── onset ──
    "w_onset": 0.25,
    "onset_kappa": 0.3,
    "onset_eta": 0.03,
    "onset_step_cap": 0.35,
    "onset_episode_cap": 8.0,
    "n_base": 10,
    "low_risk_threshold": 0.2,

    # ── PBRS ──
    "lambda_goal_pbrs": 0.2,
    "lambda_free_pbrs": 0.4,
    "pbrs_goal_clip": 5.0,
    "pbrs_free_clip": 2.0,
    "pbrs_step_cap": 0.4,
    "free_cone_angle_deg": 30.0,

    # ── future_clearance reward ──
    "w_future_clearance": 0.35,
    "future_clearance_scale": 0.25,
    "future_clearance_step_cap": 0.35,
    "future_clearance_episode_cap": 10.0,
    "epsilon_back": 0.05,

    # ── trap ──
    "trap_window": 6,
    "epsilon_progress": 0.25,
    "d_trap": 0.65,
    "n_flip_max": 2,
    "w_stall": 0.10,
    "w_escape": 3.0,
    "epsilon_recover_progress": 0.5,
    "epsilon_recover_clearance": 0.15,
    "min_trap_len": 2,

    # ── flip ──
    "w_flip": 0.35,
    "flip_episode_floor": -5.0,
    "epsilon_m": 0.10,
    "clearance_recovered": 0.9,

    # ── support ──
    "support_bin": (1.0, 0.5, 0.5),
    "w_support": 0.08,
    "support_safe": 0.75,
    "support_progress": 0.2,
    "w_support_episode": 0.5,
    "support_k_max": 8,
    "support_episode_cap": 6.0,

    # ── guard ──
    "w_eff": 0.02,
    "guard_risk_threshold": 0.2,
    "guard_d_soft_threshold": 1.0,
    "eta_eff": 0.04,
    "w_loop": 1.0,
    "loop_window": 8,
    "loop_sign_changes": 4,
    "loop_progress": 0.2,
    "loop_dense_false_positive_threshold": 0.05,
}


# ═══════════════════════════════════════════════════════════════
# §1. 几何基础函数
# ═══════════════════════════════════════════════════════════════

def reconstruct_future_positions(
    pos_world: np.ndarray,
    action_chunk: np.ndarray,
    h_future: int
) -> np.ndarray:
    """从 11d action chunk 的 dpos_world cumsum 重建未来位置。

    Args:
        pos_world: (3,) 当前世界坐标
        action_chunk: (H, 11) FM 输出的动作（或 world waypoints 含 POS 列）
                      H >= h_future
        h_future: 未来步数（≤ action 长度）

    Returns:
        future_positions: (h_future, 3) 世界坐标
    """
    from .traj_utils import DPOS_W, POS

    if action_chunk.shape[0] < h_future:
        h_future = action_chunk.shape[0]

    # 检查 action_chunk 是否包含绝对位置（POS 列非零）
    # 若已包含绝对位置（通过 action_to_world 转换过），直接取 POS 列的前 h_future 行
    if action_chunk.shape[1] >= 10 and np.any(np.abs(action_chunk[0, POS]) > 0):
        # 需要检查是否是 absolute waypoints: 检查第一行 POS 列与 pos_world 接近
        first_pos = action_chunk[0, POS]
        if np.linalg.norm(first_pos - pos_world) < 1e-3:
            # 已经是绝对 waypoints
            return action_chunk[:h_future, POS].copy()
        elif np.any(np.abs(action_chunk[:, DPOS_W]).max(axis=0) > 0):
            # 有 dpos 列 → 是 delta action
            pass

    # 从 dpos 累加
    dpos = action_chunk[:h_future, DPOS_W]
    cum = np.cumsum(dpos, axis=0)
    return pos_world.reshape(1, 3) + cum


def temporal_softmin_clearance(
    future_positions: np.ndarray,
    obstacles_or_pointcloud: Any,
    beta: float = 8.0,
    time_decay: float = 0.15,
    k_nn: int = 8,
    clearance_fn: Optional[Callable] = None,
) -> float:
    """对未来轨迹 waypoints 计算时间维 soft-min clearance d_soft^H。

    d_soft^H = -1/β · log Σ ω_i · exp(-β · d_i)

    其中 d_i 是第 i 个 waypoint 的最近障碍距离，
    ω_i = exp(-time_decay · i) 后归一化。

    Args:
        future_positions: (H, 3) 未来轨迹世界坐标
        obstacles_or_pointcloud: 障碍列表或点云（传入 clearance_fn 使用）
        beta: soft-min 锐度（β→∞ 趋近 min）
        time_decay: 时间衰减系数（近端权重大）
        k_nn: 近邻数（仅当 clearance_fn 是 k-NN 时）
        clearance_fn: 距离查询函数，签名 (pos: (3,)) -> float
                      若 None，尝试用 collision_geom.compute_obstacle_clearance

    Returns:
        d_soft_h: float, temporal soft-min clearance
    """
    H = len(future_positions)

    # 时间权重
    raw_weights = np.array([math.exp(-time_decay * i) for i in range(H)])
    if raw_weights.sum() < 1e-12:
        weights = np.ones(H) / H
    else:
        weights = raw_weights / raw_weights.sum()

    # 每个 waypoint 的最近距离
    distances = np.zeros(H, dtype=np.float64)
    for i in range(H):
        if clearance_fn is not None:
            d_i, _, _ = clearance_fn(*future_positions[i])
            distances[i] = d_i if d_i is not None else 100.0
        else:
            # 默认使用 collision_geom 的 compute_obstacle_clearance
            from .collision_geom import compute_obstacle_clearance
            d_i, _, _ = compute_obstacle_clearance(
                float(future_positions[i, 0]),
                float(future_positions[i, 1]),
                float(future_positions[i, 2]),
                obstacles_or_pointcloud, check_3d=True)
            distances[i] = d_i if d_i is not None else 100.0

    # 安全下限（避免 log(0)）
    d_min_nonzero = max(np.min(distances), 1e-8)
    _ = d_min_nonzero  # unused but kept for reference

    # temporal soft-min with bias correction
    # d_soft = -1/β · log(Σ ω_i · exp(-β · d_i))
    # 无 bias correction（因为 sum(ω_i) = 1）
    exp_terms = weights * np.exp(-beta * distances)
    sum_exp = np.sum(exp_terms)
    if sum_exp < 1e-30:
        return float(np.min(distances))
    d_soft = -1.0 / beta * math.log(sum_exp)
    return float(max(d_soft, 0.0))


def compute_lateral_displacement(
    future_positions: np.ndarray,
    pos_world: np.ndarray,
    goal_world: np.ndarray,
) -> float:
    """计算未来轨迹相对 goal direction 的平均 lateral displacement。

    将每个 future position 投影到 goal direction 法向平面，
    取 lateral 分量的大小均值。

    Args:
        future_positions: (H, 3) 未来轨迹位置
        pos_world: (3,) 当前位置
        goal_world: (3,) 目标位置

    Returns:
        lateral_disp: float, 平均 lateral displacement (米)
    """
    goal_dir = goal_world - pos_world
    goal_dist = np.linalg.norm(goal_dir)
    if goal_dist < 1e-6:
        return 0.0
    goal_unit = goal_dir / goal_dist

    lateral_dists = []
    for i in range(len(future_positions)):
        to_point = future_positions[i] - pos_world
        # 沿 goal 方向投影量
        proj = np.dot(to_point, goal_unit)
        # 法向分量（lateral）
        lateral_vec = to_point - proj * goal_unit
        lateral_dists.append(np.linalg.norm(lateral_vec))

    return float(np.mean(lateral_dists))


def compute_m_t(
    lateral_disp: float,
    epsilon_m: float = 0.10,
) -> int:
    """从 lateral displacement 返回绕行符号 m_t ∈ {-1, 0, +1}。

    使用 deadband 过滤小位移：
    m_t = +1  if lateral_disp > epsilon_m
          -1  if lateral_disp < -epsilon_m (带符号)
          0   otherwise

    由于 compute_lateral_displacement 返回的是标量大小，这里使用
    displacement 的带符号版本。若需要符号，需传入 signed 版本。

    Args:
        lateral_disp: 带符号的 lateral displacement（正=左，负=右）
        epsilon_m: deadband 阈值

    Returns:
        m_t: {-1, 0, +1}
    """
    if lateral_disp > epsilon_m:
        return 1
    elif lateral_disp < -epsilon_m:
        return -1
    return 0


def compute_free_cone_stats(
    pos_world: np.ndarray,
    goal_world: np.ndarray,
    obstacles_or_pointcloud: Any,
    angle_deg: float = 30.0,
    clearance_fn: Optional[Callable] = None,
) -> Dict[str, float]:
    """计算以 goal direction 为中心的锥体内 free distance 统计量。

    在锥体内采样固定射线集，查询每条射线的 first-hit free distance，
    返回 min / p20 / median。

    Args:
        pos_world: (3,) 当前位置
        goal_world: (3,) 目标位置
        obstacles_or_pointcloud: 障碍列表或点云
        angle_deg: 锥体半角（度）
        clearance_fn: 距离查询函数，签名 (pos: (3,)) -> float
                      若 None，使用 collision_geom

    Returns:
        stats: {"min": float, "p20": float, "median": float}
    """
    goal_dir = goal_world - pos_world
    goal_dist = np.linalg.norm(goal_dir[:2])  # 水平面距离
    if goal_dist < 1e-6:
        return {"min": 10.0, "p20": 10.0, "median": 10.0}

    angle_rad = math.radians(angle_deg)
    cos_limit = math.cos(angle_rad)

    # 射线方向: 在水平面和竖直面内采样
    yaw_offsets = np.linspace(-angle_deg, angle_deg, 5)
    pitch_offsets = np.linspace(-angle_deg / 2, angle_deg / 2, 3)

    # 计算射线
    rays = []
    for yaw_d in yaw_offsets:
        for pitch_d in pitch_offsets:
            yaw_r = math.radians(yaw_d)
            pitch_r = math.radians(pitch_d)
            # 方向向量
            dx = math.cos(yaw_r) * math.cos(pitch_r)
            dy = math.sin(yaw_r) * math.cos(pitch_r)
            dz = math.sin(pitch_r)
            ray_dir = np.array([dx, dy, dz])
            rays.append(ray_dir)

    if not rays:
        return {"min": 10.0, "p20": 10.0, "median": 10.0}

    # 沿射线采样查询 free distance（离散步进）
    max_range = 10.0
    step = 0.1
    n_steps = int(max_range / step)

    free_dists = []
    for ray in rays:
        pt = pos_world.copy()
        dist = max_range
        for _ in range(n_steps):
            pt = pt + step * ray
            if pt[2] < 0.0:  # 进入地面
                dist = np.linalg.norm(pt - pos_world)
                break
            if clearance_fn is not None:
                d_i, _, _ = clearance_fn(*pt)
            else:
                from .collision_geom import compute_obstacle_clearance
                d_i, _, _ = compute_obstacle_clearance(
                    float(pt[0]), float(pt[1]), float(pt[2]),
                    obstacles_or_pointcloud, check_3d=True)
            if d_i is not None and d_i < 0.2:  # 接近障碍
                dist = np.linalg.norm(pt - pos_world)
                break
        free_dists.append(dist)

    if not free_dists:
        return {"min": 10.0, "p20": 10.0, "median": 10.0}

    free_dists = np.array(free_dists)
    return {
        "min": float(np.min(free_dists)),
        "p20": float(np.percentile(free_dists, 20)),
        "median": float(np.median(free_dists)),
    }


def detect_progress_loop(
    progress_history: deque,
    window: int = 8,
    sign_changes: int = 3,
    min_net_progress: float = 0.2,
) -> bool:
    """检测 progress oscillation（低风险区无效来回）。

    最近 `window` 个 RL step 内 Delta progress 符号变化次数 >= sign_changes，
    且窗口净前进 < min_net_progress，判定为 loop。

    Args:
        progress_history: deque of delta progress values（最近值）
        window: 检测窗口
        sign_changes: 符号变化次数阈值
        min_net_progress: 窗口净前进下限（米）

    Returns:
        is_loop: bool
    """
    if len(progress_history) < window:
        return False

    recent = list(progress_history)[-window:]

    # 符号变化计数
    signs = [1 if p > 1e-4 else (-1 if p < -1e-4 else 0) for p in recent]
    changes = 0
    for i in range(1, len(signs)):
        if signs[i] != 0 and signs[i] != signs[i - 1]:
            changes += 1
        # 处理 +1 -> 0 -> -1 跨零变化
        if i >= 2 and signs[i] != 0 and signs[i - 1] == 0 and signs[i] != signs[i - 2]:
            changes += 1

    if changes < sign_changes:
        return False

    # 窗口净前进
    net_progress = sum(recent)
    return net_progress < min_net_progress


# ═══════════════════════════════════════════════════════════════
# §2. History / episode state 管理
# ═══════════════════════════════════════════════════════════════

def create_reward_history(
    trap_window: int = 6,
    loop_window: int = 8,
    n_base: int = 10,
    flip_episode_floor: float = -5.0,
) -> dict:
    """创建空的 reward v1.3 history / episode state。

    reset() 时必须调用此函数重新初始化。

    Args:
        trap_window: trap 检测窗口
        loop_window: loop 检测窗口
        n_base: 低风险基线窗口
        flip_episode_floor: flip 单回合累计下限

    Returns:
        history: dict 含所有 episode 内状态缓存
    """
    return {
        "progress": deque(maxlen=max(trap_window, loop_window, n_base)),
        "d_soft_h": deque(maxlen=max(trap_window, loop_window)),
        "d_min": deque(maxlen=max(trap_window, loop_window)),
        "risk_gate": deque(maxlen=max(trap_window, loop_window, n_base)),
        "u_lat_abs": deque(maxlen=max(trap_window, loop_window)),
        "yaw_rate_abs": deque(maxlen=max(trap_window, loop_window)),
        "m_t": deque(maxlen=max(trap_window, loop_window)),
        # ── trap state ──
        "trap_active": False,
        "trap_start_step": -1,
        "trap_start_d_soft_h": 100.0,
        "trap_escape_paid": False,
        "trap_segment_len": 0,
        # ── episode caps ──
        "episode_caps": {
            "onset": 0.0,
            "future_clearance": 0.0,
            "support": 0.0,
            "flip": flip_episode_floor if flip_episode_floor < 0 else -abs(flip_episode_floor),
        },
        # ── support ──
        "online_visit_count": {},
        "low_density_count_this_episode": 0,
        "new_support_cells_this_episode": set(),
        # ── history of signed lateral displacement ──
        "signed_lateral": deque(maxlen=loop_window),
        # ── loop detection ──
        "loop_count_this_episode": 0,
        "total_steps": 0,
    }


def update_trap_state(
    history: dict,
    trap_active: bool,
    progress_ok: bool,
    d_soft_h: float,
    cfg: dict,
) -> dict:
    """更新 trap segment 状态追踪。

    Args:
        history: 当前 episode history dict（会原地修改）
        trap_active: 当前 step 是否 trap
        progress_ok: progress 是否满足逃逸条件
        d_soft_h: 当前 future clearance
        cfg: reward 配置

    Returns:
        history: 更新后的 history（原地修改）
    """
    prev_active = history["trap_active"]
    trap_window = cfg.get("trap_window", 6)
    min_trap_len = cfg.get("min_trap_len", 2)

    if not prev_active and trap_active:
        # 进入新 trap segment
        history["trap_active"] = True
        history["trap_start_step"] = history.get("total_steps", 0)
        history["trap_start_d_soft_h"] = d_soft_h
        history["trap_escape_paid"] = False
        history["trap_segment_len"] = 1

    elif prev_active and trap_active:
        # 持续 trap
        history["trap_segment_len"] += 1

    elif prev_active and not trap_active:
        # 退出 trap segment — 检查是否发放 escape
        if history["trap_segment_len"] >= min_trap_len:
            if progress_ok and (d_soft_h - history["trap_start_d_soft_h"]
                                > cfg.get("epsilon_recover_clearance", 0.15)):
                history["trap_escape_paid"] = True  # 标记可发
        history["trap_active"] = False
        history["trap_segment_len"] = 0

    return history


# ═══════════════════════════════════════════════════════════════
# §3. 支持域密度工具
# ═══════════════════════════════════════════════════════════════

def compute_support_cell(
    pos_world: np.ndarray,
    goal_world: np.ndarray,
    bin_size: tuple = (1.0, 0.5, 0.5),
) -> tuple:
    """计算 goal-aligned 相对坐标 cell 索引。

    坐标系: x_parallel(沿 goal 方向), x_perp(水平法向), z(高度)

    Args:
        pos_world: (3,) 当前位置
        goal_world: (3,) 目标位置
        bin_size: (x_parallel_bin, x_perp_bin, z_bin)

    Returns:
        cell: (int, int, int) cell 索引
    """
    rel = pos_world - goal_world
    # x_parallel: 沿 goal 到 pos 方向上的投影
    goal_norm = np.linalg.norm(goal_world)
    if goal_norm < 1e-6:
        return (0, 0, 0)
    goal_unit = goal_world / goal_norm
    x_parallel = np.dot(rel, goal_unit)
    # x_perp: 水平法向
    x_perp = np.linalg.norm(rel[:2] - x_parallel * goal_unit[:2])
    # 符号：在 goal 左侧为正
    cross_z = goal_unit[0] * rel[1] - goal_unit[1] * rel[0]
    x_perp_signed = x_perp if cross_z >= 0 else -x_perp

    cell = (
        int(round(x_parallel / bin_size[0])),
        int(round(x_perp_signed / bin_size[1])),
        int(round(pos_world[2] / bin_size[2])),
    )
    return cell


def build_offline_density(
    dataset_positions: np.ndarray,
    goals: np.ndarray,
    bin_size: tuple = (1.0, 0.5, 0.5),
) -> dict:
    """从离线数据集构建 cell visit count 直方图。

    用于计算 p_D(c) — 离线数据中 cell c 的访问频率。

    Args:
        dataset_positions: (N, 3) 离线数据位置
        goals: (N, 3) 对应 goal 位置（或每个 episode 的 goal）
        bin_size: cell 尺寸

    Returns:
        density: dict {"cell_counts": {cell: count}, "total": N,
                 "p20_threshold": float, "p50_threshold": float}
    """
    counts = {}
    for i in range(len(dataset_positions)):
        cell = compute_support_cell(dataset_positions[i], goals[min(i, len(goals) - 1)], bin_size)
        counts[cell] = counts.get(cell, 0) + 1

    all_counts = sorted(counts.values())
    p20 = np.percentile(all_counts, 20) if all_counts else 0
    p50 = np.percentile(all_counts, 50) if all_counts else 0

    return {
        "cell_counts": counts,
        "total": len(dataset_positions),
        "p20_threshold": float(p20),
        "p50_threshold": float(p50),
        "bin_size": bin_size,
    }


def load_offline_density(path: Optional[str] = None) -> Optional[dict]:
    """从 JSON 文件加载 offline density。

    Args:
        path: JSON 路径。None 则使用 DEFAULT_DENSITY_PATH。

    Returns:
        density dict（cell_counts key 已恢复为 tuple），若文件不存在则返回 None。
    """
    if path is None:
        path = DEFAULT_DENSITY_PATH
    if not os.path.isfile(path):
        return None

    with open(path) as f:
        raw = json.load(f)

    # cell_counts key 从 str → tuple(int,int,int)
    cell_counts = {}
    for k_str, v in raw.get('cell_counts', {}).items():
        # k_str 形如 "(1, 2, 3)" 或 "(-8, 1, 4)"
        parts = k_str.strip('()').split(',')
        if len(parts) == 3:
            cell = (int(parts[0]), int(parts[1]), int(parts[2]))
            cell_counts[cell] = v

    density = {
        "cell_counts": cell_counts,
        "total": raw.get('total', 0),
        "p20_threshold": raw.get('p20_threshold', 0.0),
        "p50_threshold": raw.get('p50_threshold', 0.0),
        "bin_size": tuple(raw.get('bin_size', [1.0, 0.5, 0.5])),
    }
    return density


# ═══════════════════════════════════════════════════════════════
# §4. 主 Reward 函数
# ═══════════════════════════════════════════════════════════════

def compute_reward_v13(
    s_t: dict,
    action_chunk: np.ndarray,
    s_tp1: dict,
    pointcloud: Optional[np.ndarray] = None,
    obstacle_json: Any = None,
    goal_world: Optional[np.ndarray] = None,
    history: Optional[dict] = None,
    offline_density: Optional[dict] = None,
    cfg: Optional[dict] = None,
    clearance_fn: Optional[Callable] = None,
) -> Tuple[float, dict, dict, dict]:
    """Reward v1.3 主函数。

    Args:
        s_t: 当前状态字段，至少含 'pos_world'(3,), 'yaw'(scalar),
             'vel_world'(3,), 'z'(scalar)
        action_chunk: (H, 11) FM 输出动作
        s_tp1: 下一状态字段
        pointcloud: (N, 3) 点云（可选，与 obstacle_json 至少一个）
        obstacle_json: 障碍列表（可选）
        goal_world: (3,) 目标世界坐标
        history: episode history dict（会原地修改）
        offline_density: 离线密度字典（支持域）
        cfg: 配置字典（合并 defaults）
        clearance_fn: 距离查询函数（可选）

    Returns:
        (reward, terms, metrics, updated_history)
        reward: float
        terms: dict[str, float]
        metrics: dict[str, Any]
        updated_history: dict
    """
    # ── 配置 ──
    if cfg is None:
        cfg = {}
    c = {**REWARD_V13_DEFAULTS, **cfg}

    # ── 提取字段 ──
    pos_t = s_t.get("pos_world", s_t.get("pos", np.zeros(3)))
    pos_tp1 = s_tp1.get("pos_world", s_tp1.get("pos", np.zeros(3)))
    yaw_t = s_t.get("yaw", 0.0)
    vel_t = s_t.get("vel_world", s_t.get("vel", np.zeros(3)))
    z_t = float(pos_t[2])
    z_tp1 = float(pos_tp1[2])
    gw = goal_world if goal_world is not None else np.array([8.0, 0.0, 2.0])

    if history is None:
        history = create_reward_history(
            trap_window=c["trap_window"],
            loop_window=c["loop_window"],
            n_base=c["n_base"],
            flip_episode_floor=c["flip_episode_floor"],
        )
    h = history  # shorthand

    # ── 初始化 terms / metrics ──
    terms = {"base": 0.0, "pbrs": 0.0, "onset": 0.0,
             "future_clearance": 0.0, "trap": 0.0, "flip": 0.0,
             "support": 0.0, "guard": 0.0}
    metrics = {
        "d_soft_h": -1.0, "risk_gate": 0.0, "ttc": 100.0, "ttc_valid": False,
        "onset_active": False, "trap_active": False, "trap_recovered": False,
        "side_flip_count": 0, "loop_detected": False, "loop_detected_rate": 0.0,
        "support_cell_density": 0.0, "low_density_cell": False,
        "l_free_cone_min": -1.0, "l_free_cone_p20": -1.0, "l_free_cone_median": -1.0,
    }

    # ═══════════════════════════════════════════════════════════
    # 1) 重建未来位置
    # ═══════════════════════════════════════════════════════════
    future_pos = reconstruct_future_positions(pos_t, action_chunk, c["h_future"])
    # 未来位置 trajectory world frame
    # 用于 compute_obstacle_clearance
    obstacles = obstacle_json if obstacle_json is not None else []

    # ═══════════════════════════════════════════════════════════
    # 2) 计算几何量 (共享给多个 term)
    # ═══════════════════════════════════════════════════════════

    # d_soft^H: temporal soft-min over future waypoints
    d_soft_h = temporal_softmin_clearance(
        future_pos, obstacles, beta=c["softmin_beta"],
        time_decay=c["time_decay"], clearance_fn=clearance_fn)
    metrics["d_soft_h"] = d_soft_h
    if hasattr(clearance_fn, '__call__'):
        pass  # custom clearance_fn used
    elif d_soft_h < 0.01 and obstacles and _DEBUG_PRINT:
        import traceback
        print(f"[DEBUG:temporal_softmin] d_soft_h={d_soft_h:.4f} obstacle_json={type(obstacles).__name__} "
              f"len={len(obstacles) if hasattr(obstacles, '__len__') else '?'} future_pos[0]={future_pos[0]} "
              f"future_pos[-1]={future_pos[-1]}")

    # 即时 clearance（当前点）
    if clearance_fn is not None:
        d_min, _, _ = clearance_fn(*pos_t)
    else:
        from .collision_geom import compute_obstacle_clearance
        d_min, _, _ = compute_obstacle_clearance(
            float(pos_t[0]), float(pos_t[1]), float(pos_t[2]),
            obstacles, check_3d=True)
    if d_min is None:
        d_min = 100.0

    # TTC 估计
    vel_mag = float(np.linalg.norm(vel_t))
    ttc_val = False
    ttc = 100.0
    if vel_mag > 0.1 and d_min < 10.0:
        # 沿当前速度方向到障碍的简化 TTC
        # 近似：沿速度方向向外投影，查最近距离变化率
        v_dir = vel_t / vel_mag
        look_ahead = pos_t + v_dir * 2.0  # 前看 2m
        if clearance_fn is not None:
            d_ahead, _, _ = clearance_fn(*look_ahead)
        else:
            d_ahead, _, _ = compute_obstacle_clearance(
                float(look_ahead[0]), float(look_ahead[1]), float(look_ahead[2]),
                obstacles, check_3d=True)
        if d_ahead is not None and d_ahead < 10.0:
            # 粗略 TTC: 沿速度方向距离 / 速度
            d_diff = max(d_ahead - d_min, 0.01)
            vel_along = vel_mag
            ttc = d_diff / max(vel_along, 0.1)
            ttc_val = True
    metrics["ttc"] = ttc
    metrics["ttc_valid"] = ttc_val

    # Risk gate
    # g_risk = sigmoid((tau_risk - TTC)/s_tau) · sigmoid((d_risk - d_soft_h)/s_d)
    if ttc_val:
        risk_ttc = 1.0 / (1.0 + math.exp((ttc - c["tau_risk"]) / c["s_tau"]))
    else:
        risk_ttc = 0.0  # TTC 不可用时降级到 0
    risk_dist = 1.0 / (1.0 + math.exp((d_soft_h - c["d_risk"]) / c["s_d"]))
    risk_gate = risk_ttc * risk_dist
    # 若 TTC 无效，退化为 distance-only risk gate
    if not ttc_val:
        risk_gate = risk_dist
    metrics["risk_gate"] = risk_gate

    # 共用的 lateral displacement
    signed_lateral = _compute_signed_lateral(future_pos, pos_t, gw)
    u_lat_abs = abs(signed_lateral)
    m_t_val = compute_m_t(signed_lateral, epsilon_m=c["epsilon_m"])

    # ═══════════════════════════════════════════════════════════
    # 3) r_base
    # ═══════════════════════════════════════════════════════════
    dist_t = float(np.linalg.norm(pos_t - gw))
    dist_tp1 = float(np.linalg.norm(pos_tp1 - gw))
    reached = dist_tp1 < 0.5  # 假设 goal_tol_xy=0.5
    collided = s_tp1.get("collided", False)

    r_base = 0.0
    # — goal progress
    r_base += c["w_goal"] * (dist_t - dist_tp1)
    # — terminal signals
    r_base += c["r_reach"] if reached else 0.0
    r_base -= c["r_collision"] if collided else 0.0
    # — immediate clearance (底线条)
    r_base += c["w_clearance"] * min(float(d_min), c["clear_cap"])
    # — altitude penalty
    if z_t < c["z_min"]:
        r_base -= c["w_alt"]
    # — smoothness penalty
    r_base -= c["w_smooth"] * _compute_smoothness(action_chunk)
    # — z deviation
    if "goal_world" in s_t or "goal_vel" in s_t:
        gz = goal_world[2] if goal_world is not None else 2.0
        r_base -= c["w_z"] * abs(z_tp1 - gz)  # 用 tp1 的 z
    else:
        r_base -= c["w_z"] * abs(z_tp1 - 2.0)  # 默认 goal z=2
    # — yaw rate penalty
    yaw_rate = abs(s_tp1.get("yaw", 0.0) - yaw_t)
    terms["base"] = r_base

    # ═══════════════════════════════════════════════════════════
    # 4) r_pbrs (state-only free-space potential)
    # ═══════════════════════════════════════════════════════════
    phi_t = compute_free_cone_stats(
        pos_t, gw, obstacles, angle_deg=c["free_cone_angle_deg"],
        clearance_fn=clearance_fn)
    phi_tp1 = compute_free_cone_stats(
        pos_tp1, gw, obstacles, angle_deg=c["free_cone_angle_deg"],
        clearance_fn=clearance_fn)

    # 记录 free cone metrics
    metrics["l_free_cone_min"] = phi_tp1["min"]
    metrics["l_free_cone_p20"] = phi_tp1["p20"]
    metrics["l_free_cone_median"] = phi_tp1["median"]

    phi_t_val = (c["lambda_goal_pbrs"] * max(0.0, phi_t["p20"])
                 + c["lambda_free_pbrs"] * max(0.0, phi_t["p20"]))
    phi_tp1_val = (c["lambda_goal_pbrs"] * max(0.0, phi_tp1["p20"])
                   + c["lambda_free_pbrs"] * max(0.0, phi_tp1["p20"]))

    # Clamp per component
    phi_t_val = min(phi_t_val, c["pbrs_goal_clip"] + c["pbrs_free_clip"])
    phi_tp1_val = min(phi_tp1_val, c["pbrs_goal_clip"] + c["pbrs_free_clip"])

    gamma = c.get("gamma", 0.99)
    r_pbrs_raw = gamma * phi_tp1_val - phi_t_val
    # Clamp per step
    r_pbrs = max(-c["pbrs_step_cap"], min(c["pbrs_step_cap"], r_pbrs_raw))
    terms["pbrs"] = r_pbrs

    # ═══════════════════════════════════════════════════════════
    # 5) r_onset (风险条件化提前避障)
    # ═══════════════════════════════════════════════════════════
    r_onset = 0.0
    onset_active = risk_gate > c["risk_active_threshold"]
    metrics["onset_active"] = onset_active

    if onset_active:
        # Δ|u_lat| over 低风险基线
        # 收集低风险步骤的 lateral displacement
        base_lat = 0.0
        valid_base_steps = 0
        for rg, ul in zip(h["risk_gate"], h["u_lat_abs"]):
            if rg < c["low_risk_threshold"]:
                base_lat += ul
                valid_base_steps += 1
        if valid_base_steps >= 3:
            base_lat /= valid_base_steps
        elif valid_base_steps > 0:
            base_lat /= valid_base_steps
        else:
            # 没有足够低风险基线，使用 episode 均值
            all_ul = list(h["u_lat_abs"])
            if all_ul:
                base_lat = np.mean(all_ul)
            else:
                base_lat = u_lat_abs  # 本 step 作为基线（不下发 onset）

        delta_lat = u_lat_abs - base_lat
        # 同时考虑 yaw rate 变化
        # （简化：使用当前 yaw_rate - 低风险基线 yaw_rate）
        base_yaw = 0.0
        valid_yaw_base = 0
        for rg, yr in zip(h["risk_gate"], h["yaw_rate_abs"]):
            if rg < c["low_risk_threshold"]:
                base_yaw += yr
                valid_yaw_base += 1
        if valid_yaw_base >= 3:
            base_yaw /= valid_yaw_base
        delta_yaw = yaw_rate - base_yaw

        onset_raw = max(0.0, delta_lat + c["onset_kappa"] * delta_yaw - c["onset_eta"])

        if onset_raw > 0:
            # 检查 d_soft^H 是否提升（从 s_t 到 s_tp1 的 future clearance）
            # 注意: d_soft_h 是基于当前 action 的 future clearance
            # 我们比较 s_tp1 时重新计算的 future clearance
            future_pos_tp1 = reconstruct_future_positions(
                pos_tp1, action_chunk, c["h_future"])
            d_soft_tp1 = temporal_softmin_clearance(
                future_pos_tp1, obstacles, beta=c["softmin_beta"],
                time_decay=c["time_decay"], clearance_fn=clearance_fn)
            clearance_improvement = d_soft_tp1 - d_soft_h
            if clearance_improvement > 0:
                r_onset = c["w_onset"] * risk_gate * onset_raw
                r_onset = max(0.0, min(r_onset, c["onset_step_cap"]))
                # Episode cap
                remaining = c["onset_episode_cap"] - h["episode_caps"]["onset"]
                r_onset = min(r_onset, remaining)

    terms["onset"] = r_onset

    # ═══════════════════════════════════════════════════════════
    # 6) r_future_clearance (action-conditioned)
    # ═══════════════════════════════════════════════════════════
    r_fc = 0.0
    # 检查 progress 是否未大幅倒退
    progress_fc = dist_t - dist_tp1
    if progress_fc > -c["epsilon_back"]:
        # tanh-scaled future clearance bonus
        fc_raw = math.tanh((d_soft_h - c["d_ref"]) / c["future_clearance_scale"])
        r_fc = c["w_future_clearance"] * fc_raw
        r_fc = max(-c["future_clearance_step_cap"],
                   min(c["future_clearance_step_cap"], r_fc))
        # Episode cap
        remaining = c["future_clearance_episode_cap"] - h["episode_caps"]["future_clearance"]
        r_fc = max(-remaining, min(r_fc, remaining))

    terms["future_clearance"] = r_fc

    # ═══════════════════════════════════════════════════════════
    # 7) r_trap (dense obstacle stall / escape)
    # ═══════════════════════════════════════════════════════════
    r_trap = 0.0
    trap_window = c["trap_window"]

    # 判断 trap 条件
    progress_history = list(h["progress"])[-trap_window:] if len(h["progress"]) >= trap_window else []
    total_progress = sum(progress_history) if progress_history else -1.0  # 空历史 → 不通过任何 progress 门控
    d_min_history = list(h["d_min"])[-trap_window:] if len(h["d_min"]) >= trap_window else []
    min_d_over_window = min(d_min_history) if d_min_history else 100.0
    m_t_history = list(h["m_t"])[-trap_window:] if len(h["m_t"]) >= trap_window else []

    # N_flip: 统计 m_t 符号变化
    n_flip = 0
    non_zero_m = [m for m in m_t_history if m != 0]
    for i in range(1, len(non_zero_m)):
        if non_zero_m[i] != non_zero_m[i - 1]:
            n_flip += 1

    trap_active = (total_progress < c["epsilon_progress"]
                   and min_d_over_window < c["d_trap"]
                   and n_flip > c["n_flip_max"])
    metrics["trap_active"] = trap_active

    # 更新 trap state
    progress_ok = total_progress > c["epsilon_recover_progress"]
    h = update_trap_state(h, trap_active, progress_ok, d_soft_h, c)

    if trap_active:
        # Stall penalty
        r_trap -= c["w_stall"]

    # Escape: 刚退出 trap segment 且满足 recovery
    if h["trap_escape_paid"] and not trap_active:
        trap_recovered = True
        h["trap_escape_paid"] = False  # 一次发放
        r_trap += c["w_escape"]
    else:
        trap_recovered = False

    metrics["trap_recovered"] = trap_recovered
    terms["trap"] = r_trap

    # ═══════════════════════════════════════════════════════════
    # 8) r_flip (anti-flip)
    # ═══════════════════════════════════════════════════════════
    r_flip_val = 0.0
    clearance_rec = d_soft_h > c["clearance_recovered"] or risk_gate < c["risk_active_threshold"]

    # m_t 历史（含当前）
    m_t_prev = h["m_t"][-1] if len(h["m_t"]) > 0 else 0
    if (risk_gate > c["risk_active_threshold"]
            and not clearance_rec
            and m_t_val != 0
            and m_t_val != m_t_prev):
        r_flip_val -= c["w_flip"]
        # Episode floor
        current_flip = h["episode_caps"]["flip"]
        if current_flip - c["w_flip"] < c["flip_episode_floor"]:
            r_flip_val = 0.0  # 已达下限

    terms["flip"] = r_flip_val

    # ═══════════════════════════════════════════════════════════
    # 9) r_support (support expansion)
    # ═══════════════════════════════════════════════════════════
    r_support = 0.0
    low_density_cell = False
    support_cell_density = 0.0

    if offline_density is not None:
        cell = compute_support_cell(pos_tp1, gw, c["support_bin"])
        cell_counts = offline_density.get("cell_counts", {})
        total_cells = offline_density.get("total", 1)

        # p_D(c): 离线密度
        count_c = cell_counts.get(cell, 0)
        density_c = count_c / max(total_cells, 0.01)
        support_cell_density = density_c

        # tau_D: p20 或 p50
        tau_d = offline_density.get("p20_threshold", 0)
        if tau_d == 0:
            tau_d = offline_density.get("p50_threshold", 0.001)

        # 低密度 cell
        low_density = count_c < tau_d
        metrics["low_density_cell"] = low_density
        metrics["support_cell_density"] = density_c

        if low_density and not collided:
            # DEBUG: trace gates (off by default for 10k training)
            if _DEBUG_PRINT:
                _gate1 = d_soft_h > c["support_safe"]
                _gate2 = total_progress > c["support_progress"]
                if hasattr(h, '__len__') and not hasattr(h, 'keys'):
                    pass  # not a dict
                if not _gate1:
                    print(f"[SUPPORT GATE] blocked by d_soft_h={d_soft_h:.3f} <= safe={c['support_safe']}")
                elif not _gate2:
                    print(f"[SUPPORT GATE] blocked by total_progress={total_progress:.2f} <= prog={c['support_progress']}")
                    print(f"  progress history: {list(h['progress'])}")
                else:
                    print(f"[SUPPORT GATE] ALL PASS: cell={cell} count={count_c} tau_d={tau_d}")

            # 安全性门控
            if d_soft_h > c["support_safe"]:
                # Progress 门控
                if total_progress > c["support_progress"]:
                    # Novelty 衰减
                    visit_key = f"{cell}"
                    n_visit = h["online_visit_count"].get(visit_key, 0)
                    novelty = 1.0 / math.sqrt(n_visit + 1)

                    r_support = c["w_support"] * novelty
                    # Episode cap
                    remaining = c["support_episode_cap"] - h["episode_caps"]["support"]
                    r_support = min(r_support, remaining)

                    # Episode success bonus (在 episode 结束时发放)
                    h["new_support_cells_this_episode"].add(visit_key)

                    h["online_visit_count"][visit_key] = n_visit + 1
                    low_density_cell = True

    terms["support"] = r_support

    # ═══════════════════════════════════════════════════════════
    # 10) r_guard (efficiency guard)
    # ═══════════════════════════════════════════════════════════
    r_guard = 0.0

    # — 低风险绕远惩罚（仅当无障碍物交互时启用，避免惩罚合法绕行）
    if risk_gate < c["guard_risk_threshold"] and d_soft_h > c["guard_d_soft_threshold"]:
        detour_penalty = max(0.0, u_lat_abs + c["onset_kappa"] * yaw_rate - c["eta_eff"])
        r_guard -= c["w_eff"] * detour_penalty

    # — loop detection
    loop_detected = detect_progress_loop(
        h["progress"], window=c["loop_window"],
        sign_changes=c["loop_sign_changes"],
        min_net_progress=c["loop_progress"])
    if loop_detected:
        r_guard -= c["w_loop"]
        h["loop_count_this_episode"] = h.get("loop_count_this_episode", 0) + 1

    loop_detected_rate = (h.get("loop_count_this_episode", 0)
                          / max(h.get("total_steps", 1), 1))
    metrics["loop_detected"] = loop_detected
    metrics["loop_detected_rate"] = loop_detected_rate
    terms["guard"] = r_guard

    # ═══════════════════════════════════════════════════════════
    # 11) 更新 history
    # ═══════════════════════════════════════════════════════════
    step_progress = dist_t - dist_tp1
    h["progress"].append(step_progress)
    h["d_soft_h"].append(d_soft_h)
    h["d_min"].append(float(d_min))
    h["risk_gate"].append(risk_gate)
    h["u_lat_abs"].append(u_lat_abs)
    h["yaw_rate_abs"].append(yaw_rate)
    h["m_t"].append(m_t_val)
    h["signed_lateral"].append(signed_lateral)
    h["total_steps"] = h.get("total_steps", 0) + 1

    # 更新 episode caps
    h["episode_caps"]["onset"] += max(0, r_onset)
    h["episode_caps"]["future_clearance"] += r_fc
    h["episode_caps"]["support"] += max(0, r_support)
    h["episode_caps"]["flip"] += r_flip_val

    # ═══════════════════════════════════════════════════════════
    # 12) 汇总
    # ═══════════════════════════════════════════════════════════
    reward = sum(terms.values())
    metrics["side_flip_count"] = n_flip

    return float(reward), terms, metrics, h


# ═══════════════════════════════════════════════════════════════
# §5. 内部辅助函数
# ═══════════════════════════════════════════════════════════════

def _compute_signed_lateral(
    future_positions: np.ndarray,
    pos_world: np.ndarray,
    goal_world: np.ndarray,
) -> float:
    """计算带符号的 lateral displacement（正=left of goal direction）。

    使用 cross product 的 z 分量判断左右。
    返回 lateral displacement 的均值，左为正、右为负。
    """
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
        # 使用 cross 积的 z 分量判断正负
        cross_z = goal_unit[0] * lateral_vec[1] - goal_unit[1] * lateral_vec[0]
        sign = 1.0 if cross_z >= 0 else -1.0
        signed_values.append(sign * lateral_norm)

    return float(np.mean(signed_values)) if signed_values else 0.0


def _compute_smoothness(action_chunk: np.ndarray) -> float:
    """计算轨迹平滑度惩罚。

    使用 dpos 的一阶差分绝对值之和。
    """
    if len(action_chunk) < 2:
        return 0.0
    from .traj_utils import DPOS_W
    dpos = action_chunk[:, DPOS_W]
    diffs = np.diff(dpos, axis=0)
    return float(np.sum(np.abs(diffs)))


def compute_episode_terminal_support_bonus(
    history: dict,
    cfg: Optional[dict] = None,
    reached: bool = True,
) -> float:
    """Episode 结束时计算 support success bonus。

    R_sup_success = w_sup_ep · min(K_new_success, K_max) · 1[reached]

    Args:
        history: episode history dict
        cfg: 配置（覆盖 defaults）
        reached: 是否成功到达 goal

    Returns:
        bonus: float
    """
    if cfg is None:
        cfg = {}
    c = {**REWARD_V13_DEFAULTS, **cfg}
    if not reached:
        return 0.0
    k_new = len(history.get("new_support_cells_this_episode", set()))
    bonus = c["w_support_episode"] * min(k_new, c["support_k_max"])
    return bonus
