"""
kinematic_uav_env.py — 运动学闭环 RL 采样环境 (Phase β Stage 0-RL Route B)

替代 GazeboUAVEnv：纯张量运动学转移，无 ROS/SITL/subprocess。
- 完美执行: p_{t+1} = p_t + Δp (取 action_to_world 后第 8 步绝对位置)
- 滚动重规划: 执行 8 步 → 新状态 → 重合成点云 → 偏离自然涌现 (内生)
- 人为扰动: episode 中概率注入推偏/执行噪声 (强化恢复, 可配置)

DSRL-SAC 方法不变：本 env 的 step(action) 接收 (8,11) world-frame delta action,
与 GazeboUAVEnv 完全一致 (wrapper/base_policy 无需改)。

设计依据: Design_PhaseBeta_kinematic_env_v2.1
复用: collision_geom(几何碰撞), traj_utils(action_to_world/quat_to_yaw),
      synth_pc(点云合成), reward PBRS 逻辑(+高度门控 v1.3)

坐标系: 运动学环境 pos 直接是 world frame (无 SPAWN_OFFSET 的 local↔world)。
        但 synth_pc.synthesize 期望的坐标需与训练时一致 —— 见 _get_obs_frame 注释。
"""

import os, math, json, hashlib, heapq
from dataclasses import replace
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
from typing import Mapping, Optional, Dict, List, Tuple, Set
from typing import Dict, Optional

from .traj_utils import (
    action_to_world, normalize_angle,
    DPOS_W, VEL_W, ACC_W, DYAW_CW, YAW_RT, POS, VEL, ACC, YAW)
from .collision_geom import (
    GeometryPurpose,
    load_obstacles,
    compute_obstacle_clearance,
    segment_obstacle_collision,
    signed_clearance,
)
from .reward_v13 import (compute_reward_v13, create_reward_history,
                         REWARD_V13_DEFAULTS, compute_episode_terminal_support_bonus,
                         load_offline_density)
from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.terminal_contract import (
    reached_goal,
    resolve_terminal,
)


# ═══════════════════════════════════════════════════════════════
# 路径 / 常量 (与 GazeboUAVEnv 一致)
# ═══════════════════════════════════════════════════════════════
AUV_ROOT = Path(os.path.expanduser("~/AUV"))
WORLDS_DIR = AUV_ROOT / "PX4-Autopilot/Tools/sitl_gazebo/worlds/ood_experiment_worlds/layerA"
TRAIN_WORLDS_DIR = AUV_ROOT / "PX4-Autopilot/Tools/sitl_gazebo/worlds/worlds_v8"
WORLD_ENV_MAP = {4: "env_004_obs19", 8: "env_008_obs20",
                 11: "env_011_obs23", 12: "env_012_obs22", 18: "env_018_obs25"}

# ★坐标系: GazeboUAVEnv 中 synth_pc 输入 = drone_pos + SPAWN_OFFSET(-8,0,0)
# 因为 Gazebo 的 odom 是 spawn-local，synth_pc 期望 world。运动学环境若直接用
# world pos 生成障碍/目标，则 synth_pc 输入就是 world pos，无需 offset。
# 关键: 障碍 json、goal、drone_pos 必须同一 frame。本环境统一用 world frame，
# 与 obstacles.json (world frame) 一致 → synth_pc 直接喂 world pos，SPAWN_OFFSET=0。
SPAWN_OFFSET = np.array([0.0, 0.0, 0.0], dtype=np.float32)

DEFAULT_GOAL = np.array([8.0, 0.0, 2.0], dtype=np.float32)  # world frame goal


# ═══════════════════════════════════════════════════════════════
# 训练 world 池扫描 (不硬编码, 自动收集)
# ═══════════════════════════════════════════════════════════════

def scan_world_pool(worlds_dir: str):
    """扫描目录所有 obstacles.json, 返回 [(world_name, json_path), ...]
    按名称排序, 不含 OOD world (OOD 在 layerA/, 训练在 worlds_v8/, 目录隔离).
    """
    import glob as _glob
    pattern = os.path.join(worlds_dir, "env_*.obstacles.json")
    paths = sorted(_glob.glob(pattern))
    pool = []
    for p in paths:
        name = os.path.basename(p).replace(".obstacles.json", "")
        pool.append((name, p))
    return pool


# ═══════════════════════════════════════════════════════════════
# 运动学闭环环境
# ═══════════════════════════════════════════════════════════════

class KinematicUAVEnv(gym.Env):
    """运动学闭环 UAV 规划层 RL 采样环境。

    step(action): action=(8,11) world-frame delta (与 GazeboUAVEnv 同义)。
                  完美执行 8 步 → 新状态 → 重合成点云 → reward → done。
    """
    metadata = {"render_modes": []}

    def __init__(self, world_id: int = 4, goal_world: np.ndarray = None,
                 synth_pc=None, rl_action_steps: int = 8,
                 n_exec_steps: int = 4,          # ★每step实际执行的步数 (滚动重规划)
                 # ── 多 world 训练 (二选一: world_pool 或 world_id) ──
                 world_pool=None,          # [(name, json_path), ...] 训练 world 池
                 synth_pc_pool=None,       # {name: SynthPC} 预加载; None=按需建
                 synth_pc_factory=None,    # callable(json_path)->SynthPC, 按需建用
                 max_episode_steps: int = 60,
                 z_min: float = 1.0, z_max: float = 5.0,
                 collision_thresh: float = 0.10,   # ★ 判撞阈值
                 goal_tol_xy: float = 0.5, goal_tol_z: float = 0.3,
                 gamma: float = 0.99, lambda_c: float = 0.1,
                 d_safe: float = 0.75, beta_avoid: float = 20.0,
                 # ── reward v1.3: 高度门控 ──
                 h_min: float = 0.3,
                 # ── 人为扰动 (初版开启) ──
                 disturb_enable: bool = True,
                 disturb_prob: float = 0.2,
                 disturb_pos_scale: float = 0.3,   # 推偏脉冲幅度(米), 初值小, 见步骤2标定
                 exec_noise_scale: float = 0.0,    # 执行噪声(米/步), 默认关, 可开
                 # ── 非顺利中间态初始化比例 ──
                 mid_state_init_ratio: float = 0.3,
                 mid_state_pos_scale: float = 0.5,  # 中间态偏离幅度, 见步骤2标定
                 # ── reward 增量 (初版默认关闭) ──
                 use_stuck_penalty: bool = False,
                 use_escape_reward: bool = False,
                 lambda_stuck: float = 0.5, n_stuck: int = 10, stuck_eps: float = 0.05,
                 lambda_escape: float = 3.0,
                 # ── reward v1.3 (默认启用) ──
                 reward_version: str = "v13",
                 reward_v13_cfg: Optional[dict] = None,
                 offline_density: Optional[dict] = None,
                 # ── start/goal curriculum sampling (v1.3) ──
                 curriculum_stage: str = 'C2',       # 'C0' | 'C1' | 'C2'
                 episode_role: Optional[str] = None, # anticipatory|dense_recovery|support_expansion|yaw_lateral_recovery
                 seed: Optional[int] = None,
                 max_reset_attempts: int = 50,
                 reward_v17_cfg: Optional[dict] = None,
                 terminate_at_space_limits: bool = True,
                 space_cost_budget: Optional[float] = None):

        super().__init__()
        self.goal_w = (goal_world if goal_world is not None else DEFAULT_GOAL.copy()).astype(np.float32)
        self.rl_action_steps = rl_action_steps
        self.n_exec_steps = n_exec_steps         # ★实际执行步数 (=4)
        assert 1 <= n_exec_steps <= rl_action_steps, \
            f"n_exec_steps({n_exec_steps}) 必须 ∈ [1, rl_action_steps({rl_action_steps})]"
        self.max_episode_steps = max_episode_steps
        self.z_min, self.z_max = z_min, z_max
        self.z_target = (z_min + z_max) / 2.0
        self.goal_tol_xy, self.goal_tol_z = goal_tol_xy, goal_tol_z
        self.traj_dt = 0.3
        self.terminate_at_space_limits = bool(terminate_at_space_limits)
        self.space_cost_budget = space_cost_budget
        if space_cost_budget is not None and (
                reward_version != "v2" or not np.isfinite(space_cost_budget) or space_cost_budget < 0.):
            raise ValueError("space cost requires v2 and a finite non-negative budget")
        self.n_obs_steps = 8   # ★n8: 必须 8, global_cond=192×8=1536

        # ── world 模式: 多 world 训练 / 单 world 评估 ──
        self.world_pool = world_pool
        self.synth_pc_pool = synth_pc_pool
        self.synth_pc_factory = synth_pc_factory
        self._multi_world = world_pool is not None
        if self._multi_world:
            # 训练: reset 时随机选 world, 此处先占位 (用池中第一个初始化)
            assert len(world_pool) > 0, "world_pool 为空"
            self.world_id, self.obstacles_json = world_pool[0][0], world_pool[0][1]
            self.synth_pc = None  # reset 时按 world 设置
        else:
            # 单 world (评估/sanity): 兼容原逻辑
            self.world_id = world_id
            self.obstacles_json = str(WORLDS_DIR / f"{WORLD_ENV_MAP[world_id]}.obstacles.json")
            self.synth_pc = synth_pc

        # reward
        self.gamma, self.lambda_c = gamma, lambda_c
        self.d_safe, self.beta_avoid = d_safe, beta_avoid
        self.h_min = h_min

        # 扰动
        self.disturb_enable = disturb_enable
        self.disturb_prob = disturb_prob
        self.disturb_pos_scale = disturb_pos_scale
        self.exec_noise_scale = exec_noise_scale

        # 中间态初始化
        self.mid_state_init_ratio = mid_state_init_ratio
        self.mid_state_pos_scale = mid_state_pos_scale

        # reward 增量
        self.use_stuck_penalty = use_stuck_penalty
        self.use_escape_reward = use_escape_reward
        self.lambda_stuck, self.n_stuck, self.stuck_eps = lambda_stuck, n_stuck, stuck_eps
        self.lambda_escape = lambda_escape

        # ── start/goal curriculum sampling (v1.3) ──
        self.curriculum_stage = curriculum_stage
        self._episode_role = episode_role
        if isinstance(max_reset_attempts, bool) or int(max_reset_attempts) != max_reset_attempts or int(max_reset_attempts) < 1:
            raise ValueError("max_reset_attempts must be a positive integer")
        self.max_reset_attempts = int(max_reset_attempts)

        # reward v1.3
        self.reward_version = reward_version
        self.reward_v13_cfg = {**REWARD_V13_DEFAULTS, **(reward_v13_cfg or {})}
        if reward_version == "v2" and (
            self.n_exec_steps != 4 or self.rl_action_steps != 8
            or self.max_episode_steps != 60 or collision_thresh != .10
        ):
            raise ValueError("R1 requires H4 execution, 8x11 policy, 60 decisions and 0.10 m contact")
        if reward_version == "v17":
            from .reward_v17 import resolve_config
            from .reward_v17_support.density import validate_density
            self.reward_v17_cfg = resolve_config(reward_v17_cfg)
            validate_density(offline_density)
            if (self.n_exec_steps != self.reward_v17_cfg["H_e"]
                    or self.max_episode_steps != self.reward_v17_cfg["max_episode_steps"]
                    or self.gamma != self.reward_v17_cfg["gamma_pbrs"]):
                raise ValueError("v17 environment horizon/gamma mismatch")
        # P0-1: 自动加载 offline density（若未显式传入）
        if offline_density is None and reward_version == "v2":
            self.offline_density = None
        elif offline_density is None:
            self.offline_density = load_offline_density()
        else:
            self.offline_density = offline_density
        self._reward_history = None
        self._last_reward_terms = None
        self._last_reward_metrics = None

        # P1-3/4: Episode-level monitoring (init default, reset() sets per-episode)
        self._episode_fallback_type = None
        self._episode_fallback_reasons = []
        self._episode_reward_term_active = {}
        self._episode_dense_struct_traversal = False
        self._episode_antic_risk_onset = False
        self._episode_support_lowdensity_cov = 0.0
        self._last_rejection_histogram = {}
        self._occ_grid_cache = None

        # 障碍几何 (world frame, 碰撞检测用). 多world时reset会重载.
        self.obstacles = load_obstacles(self.obstacles_json)
        # ★坐标系: BC 训练数据中 state 的 pos_w 是 odom 坐标 (world - spawn),
        #   PC 是 world 坐标。运动学环境需复现此约定。
        self._spawn_offset = self._load_spawn_offset(self.obstacles_json)
        self.drone_radius = 0.35
        self.collision_thresh = collision_thresh

        # obs/act space (与 GazeboUAVEnv 一致)
        self.observation_space = spaces.Dict({
            'point_cloud': spaces.Box(-50, 50, (self.n_obs_steps, 4096, 3), np.float32),
            'agent_pos': spaces.Box(-np.inf, np.inf, (self.n_obs_steps, 9), np.float32)})
        self.action_space = spaces.Box(-1, 1, (rl_action_steps, 11), np.float32)

        self._rng = np.random.default_rng(seed)

        # 内部状态 (纯内存, 无 ROS)
        self.episode_step = 0
        self._drone_pos = np.zeros(3, np.float32)
        self._drone_yaw = 0.0
        self._drone_vel = np.zeros(3, np.float32)
        self._pc_history = np.zeros((self.n_obs_steps, 4096, 3), np.float32)
        self._state_history = np.zeros((self.n_obs_steps, 9), np.float32)
        self._pc_meta_history = []
        self._observation_frame_index = 0
        self._episode_identity_hash = None
        self._last_pc_meta = None
        self._prev_phi = 0.0
        self._prev_dist = None
        self._count_stuck = 0
        self._done_reason = "none"

    # ═══════════════════════════════════════════════════════════
    # reset (纯内存, 无 SITL)
    # ═══════════════════════════════════════════════════════════

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.episode_step = 0
        self._count_stuck = 0
        self._prev_dist = None

        # P1-3/4: Episode-level monitoring fields
        self._episode_fallback_type = None       # None | 'fallback_validated' | 'fallback_invalid'
        self._episode_fallback_reasons = []      # list of failed gate names
        self._episode_reward_term_active = {}    # {term_name: bool} — per reward term nonzero
        self._episode_dense_struct_traversal = False  # dense: did path cross bottleneck/trap/branch?
        self._episode_antic_risk_onset = False   # anticipatory: was onset_time triggered?
        self._episode_support_lowdensity_cov = 0.0  # support: low-density cell coverage fraction
        self._last_rejection_histogram = {}      # rejection reasons from _sample_start_goal

        # 0) 多 world 训练: 随机选一个 world (obstacles + synth_pc 同源切换)
        if self._multi_world:
            idx = int(self._rng.integers(len(self.world_pool)))
            name, json_path = self.world_pool[idx]
            self._set_world(name, json_path)

        # 1) 训练随机采样；formal paired evaluation 只能通过显式 options
        #    消费冻结起终点。A* 永远仅记录诊断，不参与准入。
        fixed = (options or {}).get("fixed_start_goal") if isinstance(options, Mapping) else None
        if fixed is not None:
            if not isinstance(fixed, Mapping) or set(fixed) != {"start_pos", "start_yaw", "goal_pos"}:
                raise ValueError("fixed_start_goal must contain start_pos/start_yaw/goal_pos")
            start_pos = np.asarray(fixed["start_pos"], dtype=np.float32)
            self.goal_w = np.asarray(fixed["goal_pos"], dtype=np.float32)
            start_yaw = float(fixed["start_yaw"])
            if not math.isfinite(start_yaw):
                raise ValueError("fixed start_yaw must be finite")
            reasons = self._candidate_rejection_reasons(
                start_pos, self.goal_w, self._episode_role or "anticipatory"
            )
            if reasons:
                raise ValueError("fixed_start_goal failed physical admission: {}".format(reasons))
            self._last_astar_diagnostic = self._compute_astar_diagnostic(
                start_pos, self.goal_w, self._episode_role or "anticipatory"
            )
            self._last_rejection_histogram = {"fixed_valid": 1}
        else:
            start_pos, start_yaw, self.goal_w = self._prepare_reset_candidate(
                self._episode_role or 'anticipatory', apply_mid_state=True)

        self._drone_pos = start_pos.astype(np.float32)
        self._drone_yaw = float(start_yaw)
        self._drone_vel = np.zeros(3, np.float32)

        # ── P0-2 (rework #3): Episode-local RNG seed for reproducible
        # point cloud ── hashlib.blake2b over the canonical JSON payload;
        # the string world_id enters VERBATIM (never Python hash(), whose
        # process-random salt breaks cross-process reproduction —
        # RULING-7 P0-2). int world_id (legacy) stays int.
        try:
            _wid = int(self.world_id)
        except (ValueError, TypeError):
            _wid = str(self.world_id)
        _payload_data = {
            "world_id": _wid,
            "goal_w": [round(float(v), 4) for v in self.goal_w],
            "drone_pos": [round(float(v), 4) for v in self._drone_pos],
        }
        if seed is not None:
            _payload_data["seed"] = int(seed)
        _payload = json.dumps(_payload_data, sort_keys=True).encode("utf-8")
        _digest = hashlib.blake2b(_payload, digest_size=4).digest()
        self._episode_rng_seed = int.from_bytes(_digest, "little")
        self._episode_identity_hash = hashlib.sha256(_payload).hexdigest()
        self._observation_frame_index = 0
        self._pc_meta_history = []

        # 3) 合成初始点云 → 重复填 8 帧 (reset 时无人机未动, 同一帧 ×8)
        obs = self._get_obs_frame()
        for frame_index in range(self.n_obs_steps):
            meta = None
            if self.reward_version in ("v17", "v2"):
                meta = replace(self._last_pc_meta, decision_index=frame_index)
            self._push_obs(obs, pointcloud_meta=meta)
        self._observation_frame_index = self.n_obs_steps - 1

        # 4) 初始化 Φ
        self._prev_phi = self._compute_phi()
        self._reward_history = create_reward_history(
            trap_window=self.reward_v13_cfg["trap_window"],
            loop_window=self.reward_v13_cfg["loop_window"],
            n_base=self.reward_v13_cfg["n_base"],
            flip_episode_floor=self.reward_v13_cfg["flip_episode_floor"],
        )
        if self.reward_version == "v17":
            from .reward_v17_support.environment import reset_reward
            reset_reward(self)
        elif self.reward_version == "v2":
            from .reward_v2_support.environment import reset_reward_v2
            reset_reward_v2(self)
        return self._get_obs_dict(), {
            'world_id': self.world_id,
            'astar_diagnostic': dict(self._last_astar_diagnostic),
            'pointcloud_meta_history': tuple(self._pc_meta_history),
            'episode_identity_hash': self._episode_identity_hash,
            'decision_index': self._observation_frame_index,
        }

    @staticmethod
    def _load_spawn_offset(obstacles_json):
        """从 obstacles.json 读取 spawn_world, 返回 [x, y, 0] 偏移."""
        import json as _json
        with open(obstacles_json) as f:
            data = _json.load(f)
        sw = data.get('spawn_world', [-8.0, 0.0])
        return np.array([sw[0], sw[1], 0.0], dtype=np.float32)

    def _world_to_odom(self, world_pos):
        """world 坐标 → odom 坐标 (state 用, 匹配 BC 训练分布)."""
        return world_pos - self._spawn_offset

    def _set_world(self, name, json_path):
        """切换 world: obstacles 几何 + synth_pc 点云合成器 必须同源同时切."""
        self.world_id = name
        self.obstacles_json = json_path
        self.obstacles = load_obstacles(json_path)   # 碰撞几何
        self._spawn_offset = self._load_spawn_offset(json_path)  # ★坐标系偏移
        self._occ_grid_cache = None  # invalidate occupancy grid cache
        # synth_pc: 预加载池取 / 按需建 (factory) — 必须与 obstacles 同 world
        if self.synth_pc_pool is not None:
            self.synth_pc = self.synth_pc_pool[name]
        elif self.synth_pc_factory is not None:
            self.synth_pc = self.synth_pc_factory(json_path)
        # 若两者都 None (单 world 模式不会进这分支), synth_pc 保持 __init__ 设的

    # ═══════════════════════════════════════════════════════════
    # Start/Goal 采样 (v1.3: role-aware curriculum + rejection)
    # ═══════════════════════════════════════════════════════════

    _ROLE_SAMPLING_PARAMS = {
        'anticipatory':         {'d_xy': (10, 24), 'start_cl': 0.60, 'goal_cl': 2.50},
        'dense_recovery':       {'d_xy': (5, 18),  'start_cl': 0.60, 'goal_cl': 1.50},
        'support_expansion':    {'d_xy': (8, 26),  'start_cl': 0.80, 'goal_cl': 2.00},
        'yaw_lateral_recovery': {'d_xy': (6, 18),  'start_cl': 0.80, 'goal_cl': 2.00},
    }

    def _curriculum_xy_range(self):
        if self.curriculum_stage == 'C0':
            return (-10.0, 10.0, -6.0, 6.0)
        elif self.curriculum_stage == 'C1':
            return (-12.0, 12.0, -12.0, 12.0)
        else:
            return (-14.0, 14.0, -14.0, 14.0)

    def _curriculum_z_range(self):
        if self.curriculum_stage == 'C0':
            return (1.3, 3.3)
        elif self.curriculum_stage == 'C1':
            return (1.0, 3.5)
        else:
            return (1.0, self.z_max)

    def _get_role_params(self, role):
        rp = self._ROLE_SAMPLING_PARAMS.get(role)
        if rp is None:
            rp = self._ROLE_SAMPLING_PARAMS['anticipatory']
        return rp['d_xy'][0], rp['d_xy'][1], rp['start_cl'], rp['goal_cl']

    def _draw_start_goal_candidate(self, role):
        x_min, x_max, y_min, y_max = self._curriculum_xy_range()
        z_low, z_high = self._curriculum_z_range()
        start_x = self._rng.uniform(x_min, x_max)
        start_y = self._rng.uniform(y_min, y_max)
        start_z = self._rng.uniform(z_low + 0.3, z_high - 0.3)
        goal_x = self._rng.uniform(x_min, x_max)
        goal_y = self._rng.uniform(y_min, y_max)
        goal_z = self._rng.uniform(z_low + 0.5, z_high - 0.5)
        yaw = math.atan2(goal_y - start_y, goal_x - start_x)
        yaw += self._rng.uniform(-0.3, 0.3)
        return (
            np.array([start_x, start_y, start_z], dtype=np.float32),
            normalize_angle(yaw),
            np.array([goal_x, goal_y, goal_z], dtype=np.float32),
        )

    def _candidate_rejection_reasons(self, start, goal, role):
        start = np.asarray(start)
        goal = np.asarray(goal)
        if (start.shape != (3,) or goal.shape != (3,)
                or not np.isfinite(start).all() or not np.isfinite(goal).all()):
            return ['nonfinite']
        x_min, x_max, y_min, y_max = self._curriculum_xy_range()
        z_low, z_high = self._curriculum_z_range()
        reasons = []
        for label, point in (('start', start), ('goal', goal)):
            if not (x_min <= point[0] <= x_max and y_min <= point[1] <= y_max
                    and z_low <= point[2] <= z_high):
                reasons.append('bounds_' + label)
        if self.reward_version == "v2" and float(np.linalg.norm(
                start.astype(np.float64) - goal.astype(np.float64))) <= 0.5:
            reasons.append('goal_sphere_start')
        _, _, start_cl, goal_cl = self._get_role_params(role)
        if not reasons:
            start_clearance, _, _ = compute_obstacle_clearance(
                float(start[0]), float(start[1]), float(start[2]), self.obstacles)
            goal_clearance, _, _ = compute_obstacle_clearance(
                float(goal[0]), float(goal[1]), float(goal[2]), self.obstacles)
            if start_clearance < start_cl:
                reasons.append('clearance_start')
            if goal_clearance < goal_cl:
                reasons.append('clearance_goal')
        return reasons

    def _compute_astar_diagnostic(self, start, goal, role):
        try:
            raw = self._astar_3d(
                self._get_occupancy_grid(),
                *[float(value) for value in start],
                *[float(value) for value in goal])
            reachable, path_length, bottleneck, z_min, z_max, path = raw
            chord = float(np.linalg.norm(np.asarray(goal) - np.asarray(start)))
            stretch = (float(path_length) / max(chord, 0.1)) if reachable else None
            return {
                'diagnostic_only': True,
                'status': 'reachable' if reachable else 'unreachable',
                'reachable': bool(reachable),
                'path_length_m': float(path_length) if np.isfinite(path_length) else None,
                'path_stretch': stretch,
                'bottleneck_clearance_m': float(bottleneck),
                'path_z_min_m': float(z_min),
                'path_z_max_m': float(z_max),
                'path_nodes': len(path),
                'role': role,
            }
        except Exception as exc:
            return {
                'diagnostic_only': True,
                'status': 'error',
                'reachable': None,
                'error_type': type(exc).__name__,
                'role': role,
            }

    def _prepare_reset_candidate(self, role, apply_mid_state):
        from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.terminal_contract import ResetExhausted
        rejection_histogram = {}
        for _attempt in range(self.max_reset_attempts):
            start, yaw, goal = self._draw_start_goal_candidate(role)
            if apply_mid_state and self._rng.random() < self.mid_state_init_ratio:
                start, yaw = self._perturb_to_mid_state(start, yaw)
            reasons = self._candidate_rejection_reasons(start, goal, role)
            if reasons:
                for reason in reasons:
                    rejection_histogram[reason] = rejection_histogram.get(reason, 0) + 1
                continue
            self._last_astar_diagnostic = self._compute_astar_diagnostic(start, goal, role)
            self._last_rejection_histogram = dict(rejection_histogram)
            self._last_rejection_histogram['success'] = 1
            self._episode_fallback_type = None
            self._episode_fallback_reasons = []
            return start.astype(np.float32), float(yaw), goal.astype(np.float32)
        self._last_rejection_histogram = dict(rejection_histogram)
        raise ResetExhausted(
            getattr(self, 'task_id', self.world_id),
            role,
            self.max_reset_attempts,
            rejection_histogram,
        )

    def _sample_start_goal(self, role=None):
        role = role or self._episode_role or 'anticipatory'
        return self._prepare_reset_candidate(role, apply_mid_state=False)

    # ═══════════════════════════════════════════════════════════
    # Occupancy Grid + 3D A* (P1-2: reachability + path_stretch)
    # ═══════════════════════════════════════════════════════════

    _OCC_RES = 0.5  # 0.5m grid resolution

    def _get_occupancy_grid(self):
        """Lazy-build occupancy grid from obstacles, cached per world."""
        if hasattr(self, '_occ_grid_cache') and self._occ_grid_cache is not None:
            return self._occ_grid_cache

        self._occ_grid_cache = self._build_occupancy_grid()
        return self._occ_grid_cache

    def _build_occupancy_grid(self):
        """Build the diagnostic A* grid from the shared primitive oracle."""
        res = self._OCC_RES
        x_min, x_max, y_min, y_max = self._curriculum_xy_range()
        z_min_val, z_max_val = self._curriculum_z_range()
        # Expand slightly for safety margin
        x_min, x_max = x_min - 1.0, x_max + 1.0
        y_min, y_max = y_min - 1.0, y_max + 1.0
        z_min_val, z_max_val = max(z_min_val - 0.5, 0.0), z_max_val + 0.5

        nx = max(int((x_max - x_min) / res) + 1, 2)
        ny = max(int((y_max - y_min) / res) + 1, 2)
        nz = max(int((z_max_val - z_min_val) / res) + 1, 2)

        # grid[y][x][z] — True = free, False = blocked
        grid = np.ones((ny, nx, nz), dtype=bool)

        admission_margin = 0.15
        for obs in self.obstacles:
            cx, cy, cz = obs.x, obs.y, obs.z
            if obs.shape == "sphere":
                xy_extent = obs.radius + admission_margin
                z_extent = obs.radius + admission_margin
            elif obs.shape == "cylinder":
                xy_extent = obs.radius + admission_margin
                z_extent = obs.height / 2.0
            elif obs.shape == "box":
                xy_extent = math.hypot(obs.size_x, obs.size_y) / 2.0 + admission_margin
                z_extent = obs.size_z / 2.0
            else:
                raise ValueError("unknown obstacle shape: {!r}".format(obs.shape))

            # Bounding box in grid indices
            ix0 = max(0, int((cx - xy_extent - x_min) / res))
            ix1 = min(nx - 1, int((cx + xy_extent - x_min) / res))
            iy0 = max(0, int((cy - xy_extent - y_min) / res))
            iy1 = min(ny - 1, int((cy + xy_extent - y_min) / res))
            iz0 = max(0, int((cz - z_extent - z_min_val) / res))
            iz1 = min(nz - 1, int((cz + z_extent - z_min_val) / res))

            for iy in range(iy0, iy1 + 1):
                for ix in range(ix0, ix1 + 1):
                    wx = x_min + (ix + 0.5) * res
                    wy = y_min + (iy + 0.5) * res
                    if obs.shape != "sphere":
                        clearance = signed_clearance(
                            (wx, wy, cz),
                            obs,
                            purpose=GeometryPurpose.ADMISSION_ENDPOINT,
                            body_radius_m=0.0,
                        )
                        if clearance < admission_margin:
                            grid[iy, ix, iz0:iz1 + 1] = False
                        continue
                    for iz in range(iz0, iz1 + 1):
                        wz = z_min_val + (iz + 0.5) * res
                        if signed_clearance(
                            (wx, wy, wz), obs,
                            purpose=GeometryPurpose.ADMISSION_ENDPOINT,
                            body_radius_m=0.0,
                        ) < admission_margin:
                            grid[iy, ix, iz] = False

        # Cache grid metadata
        self._occ_grid_meta = {
            'x_min': x_min, 'y_min': y_min, 'z_min': z_min_val,
            'res': res, 'nx': nx, 'ny': ny, 'nz': nz,
        }
        return grid

    def _world_to_grid(self, x, y, z):
        """Convert world coordinates to grid indices."""
        m = self._occ_grid_meta
        ix = int((x - m['x_min']) / m['res'])
        iy = int((y - m['y_min']) / m['res'])
        iz = int((z - m['z_min']) / m['res'])
        return ix, iy, iz

    def _grid_to_world(self, ix, iy, iz):
        """Convert grid indices to world coordinates."""
        m = self._occ_grid_meta
        wx = m['x_min'] + (ix + 0.5) * m['res']
        wy = m['y_min'] + (iy + 0.5) * m['res']
        wz = m['z_min'] + (iz + 0.5) * m['res']
        return wx, wy, wz

    def _astar_3d(self, occ_grid, sx, sy, sz, gx, gy, gz):
        """3D A* search on occupancy grid.
        Returns (reachable: bool, path_len: float, bottleneck_clr: float,
                 z_min: float, z_max: float, path: list).
        """
        m = self._occ_grid_meta
        six, siy, siz = self._world_to_grid(sx, sy, sz)
        gix, giy, giz = self._world_to_grid(gx, gy, gz)

        # Clamp to grid bounds
        six = max(0, min(m['nx'] - 1, six))
        siy = max(0, min(m['ny'] - 1, siy))
        siz = max(0, min(m['nz'] - 1, siz))
        gix = max(0, min(m['nx'] - 1, gix))
        giy = max(0, min(m['ny'] - 1, giy))
        giz = max(0, min(m['nz'] - 1, giz))

        if not occ_grid[siy, six, siz] or not occ_grid[giy, gix, giz]:
            return False, 0.0, 0.0, 0.0, 0.0, []

        start = (six, siy, siz)
        goal = (gix, giy, giz)
        if start == goal:
            return True, 0.0, 2.0, sz, gz, [start]

        dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

        def _heuristic(a, b):
            return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)

        open_set = [(0.0, 0, start)]
        g_score = {start: 0.0}
        parent: Dict[Tuple, Optional[Tuple]] = {start: None}
        visited: Set[Tuple] = set()

        max_iter = m['nx'] * m['ny'] * m['nz'] // 5
        iters = 0

        while open_set and iters < max_iter:
            iters += 1
            _, _, current = heapq.heappop(open_set)
            if current in visited:
                continue
            visited.add(current)

            if current == goal:
                # Reconstruct path
                path = []
                cur = goal
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()

                path_len = 0.0
                for i in range(1, len(path)):
                    di = 0.0
                    for dim in range(3):
                        di += (path[i][dim] - path[i-1][dim]) ** 2
                    path_len += math.sqrt(di) * m['res']

                return True, path_len, 1.0, sz, gz, path

            ci, cj, ck = current
            for d in dirs:
                ni, nj, nk = ci + d[0], cj + d[1], ck + d[2]
                if not (0 <= ni < m['nx'] and 0 <= nj < m['ny'] and 0 <= nk < m['nz']):
                    continue
                if not occ_grid[nj, ni, nk]:
                    continue
                neighbor = (ni, nj, nk)
                move_cost = m['res'] if d[2] == 0 else math.hypot(m['res'], m['res'])
                tentative_g = g_score[current] + move_cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + _heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, iters, neighbor))
                    parent[neighbor] = current

        return False, 0.0, 0.0, 0.0, 0.0, []

    def _perturb_to_mid_state(self, pos, yaw):
        """把起点推到'非顺利中间态': 高度偏/朝向偏/位置偏.
        ★幅度 mid_state_pos_scale 参考 §8 步骤2 FM 内生偏离 P95."""
        pos = pos.copy()
        pos[0] += self._rng.uniform(-self.mid_state_pos_scale, self.mid_state_pos_scale)
        pos[1] += self._rng.uniform(-self.mid_state_pos_scale, self.mid_state_pos_scale)
        pos[2] += self._rng.uniform(-self.mid_state_pos_scale, self.mid_state_pos_scale)
        pos[2] = np.clip(pos[2], self.z_min, self.z_max)
        yaw = normalize_angle(yaw + self._rng.uniform(-0.5, 0.5))
        return pos, yaw

    # ═══════════════════════════════════════════════════════════
    # step (纯张量运动学, 完美执行 + 扰动)
    # ═══════════════════════════════════════════════════════════

    def step(self, action: np.ndarray):
        """action=(8,11) world-frame delta. 完美执行 8 步 → 新状态."""
        if self.reward_version == "v17":
            from .reward_v17_support.environment import step_v17
            return step_v17(self, action)
        if self.reward_version == "v2":
            from .reward_v2_support.environment import step_v2
            return step_v2(self, action)
        phi_prev = self._prev_phi

        # ★ 保存 pre-step 状态 (s_t), 供 reward v1.3 区分 s_t 与 s_tp1
        _pre_pos = self._drone_pos.copy()
        _pre_yaw = self._drone_yaw
        _pre_vel = self._drone_vel.copy()

        # 1) delta action → 绝对 world waypoints (复用 action_to_world)
        world_wp = action_to_world(action, self._drone_pos,
                                    self._drone_vel, self._drone_yaw)

        # 2) 完美执行: 取第 n_exec_steps 步(最后一个执行步)的绝对位置/yaw
        #    world_wp[k, POS] 是第 k 步累积后的绝对 world 位置
        last_idx = self.n_exec_steps - 1             # ★ = 3, 只推进到第4步
        new_pos = world_wp[last_idx, POS].copy()
        new_yaw = float(world_wp[last_idx, YAW])
        # 速度 = 最后一步的 world 速度分量 (供下次 action_to_world 用)
        new_vel = world_wp[last_idx, VEL].copy()

        # 3) 执行噪声 (可选, 模拟执行不完美; 默认关 exec_noise_scale=0)
        if self.exec_noise_scale > 0:
            new_pos = new_pos + self._rng.normal(0, self.exec_noise_scale, 3).astype(np.float32)

        # 4) 人为扰动 (概率注入推偏脉冲, 强化恢复能力)
        if self.disturb_enable and self._rng.random() < self.disturb_prob:
            impulse = self._rng.normal(0, self.disturb_pos_scale, 3).astype(np.float32)
            new_pos = new_pos + impulse

        self._drone_pos = new_pos.astype(np.float32)
        self._drone_yaw = normalize_angle(new_yaw)
        self._drone_vel = new_vel.astype(np.float32)

        # 5) 新状态 → 每 waypoint 合成点云 + 推入 history
        #    BC 训练帧距=0.3s。每 step 只执行 n_exec_steps 个 waypoint,
        #    每 waypoint 产生一帧观测 (与 Gazebo 每时间步记一帧一致)。
        # ★ 对 k == last_idx, 使用扰动后的 self._drone_pos 而非未扰动的 world_wp,
        #    确保 obs history 与 env 内部状态一致 (Item 6 fix).
        for k in range(self.n_exec_steps):           # ★ range(4): 每时间步push一帧, 共4帧
            if k == last_idx:
                wp_pos = self._drone_pos.copy()
                wp_yaw = self._drone_yaw
            else:
                wp_pos = world_wp[k, POS]
                wp_yaw = float(world_wp[k, YAW])
            # P0-2: per-waypoint seed = episode seed + step index (unique yet reproducible)
            _wp_seed = self._episode_rng_seed + k  if hasattr(self, '_episode_rng_seed') else None
            pc = self.synth_pc.synthesize(wp_pos, wp_yaw,
                                          num_points=4096, add_noise=True,
                                          rng_seed=_wp_seed)
            odom_pos = self._world_to_odom(wp_pos)
            goal_odom = self._world_to_odom(self.goal_w)
            s9 = self._build_state(odom_pos, wp_yaw, goal_odom)
            self._push_obs({'point_cloud': pc.astype(np.float32),
                            'agent_pos': s9.astype(np.float32)})
        obs_dict = self._get_obs_dict()

        # 6) Contact path: pre-step position -> executed waypoints -> actual end.
        # Positions only; separate allocation leaves policy/reward world_wp intact.
        # The noisy end REPLACES the nominal last point, just as obs history does;
        # do not invent a nominal-end -> noisy-end extra leg or check the unused tail.
        self._last_path = np.vstack((_pre_pos, world_wp[:self.n_exec_steps, POS]))
        self._last_path[-1] = self._drone_pos
        if self.reward_version == "v13":
            reward, self._last_reward_terms, self._last_reward_metrics, self._reward_history = \
                self._compute_reward_v13(phi_prev, world_wp, _pre_pos, _pre_yaw, _pre_vel)
            facts = self._last_transition_facts
        else:
            # P1-4: Track per-term activity (OR across steps)
            if self._last_reward_terms:
                for term_name, term_val in self._last_reward_terms.items():
                    if term_name not in self._episode_reward_term_active:
                        self._episode_reward_term_active[term_name] = False
                    if abs(term_val) > 1e-8:
                        self._episode_reward_term_active[term_name] = True
            # P1-4: Track role-specific semantic indicators
            if self._last_reward_metrics:
                onset = self._last_reward_metrics.get("onset_time", None)
                if onset is not None and onset > 0:
                    self._episode_antic_risk_onset = True
                if self._last_reward_metrics.get("trap_active", False):
                    self._episode_dense_struct_traversal = True
                ld_cov = self._last_reward_metrics.get("low_density_coverage", 0.0)
                if ld_cov > self._episode_support_lowdensity_cov:
                    self._episode_support_lowdensity_cov = ld_cov
            reward, _ = self._compute_reward(phi_prev)
            facts = self._last_transition_facts
        self.episode_step += 1
        terminated = facts.terminated
        truncated = facts.truncated

        info = {'step': self.episode_step, 'pos': self._drone_pos.copy(),
                'z_world': float(self._drone_pos[2]),
                'transition_facts': facts,
                'done_reason': facts.done_reason}
        if self.reward_version == "v13":
            info["reward_version"] = "v1.3"
            if self._last_reward_terms is not None:
                info["reward_terms"] = dict(self._last_reward_terms)
            if self._last_reward_metrics is not None:
                info["reward_metrics"] = dict(self._last_reward_metrics)
        if terminated or truncated:
            dist = float(np.linalg.norm(self._drone_pos - self.goal_w))
            clr, _, _ = self._geom_clearance()
            info['dist'] = dist
            info['clr'] = clr
            # P1-3/4: Episode-level role semantic monitoring
            info['episode_fallback_type'] = self._episode_fallback_type
            info['episode_fallback_reasons'] = list(self._episode_fallback_reasons)
            info['episode_reward_term_active'] = dict(self._episode_reward_term_active)
            info['episode_dense_struct_traversal'] = self._episode_dense_struct_traversal
            info['episode_antic_risk_onset'] = self._episode_antic_risk_onset
            info['episode_support_lowdensity_cov'] = self._episode_support_lowdensity_cov
        return obs_dict, reward, terminated, truncated, info

    # ═══════════════════════════════════════════════════════════
    # 观测 (复用 GazeboUAVEnv 逻辑)
    # ═══════════════════════════════════════════════════════════

    def _get_obs_frame(self):
        # ★坐标系约定 (匹配 BC 训练数据):
        #   synth_pc 输入 = world 坐标 (障碍物 json 与点云同 frame)
        #   state 的 pos_w = odom 坐标 (world - spawn_offset), 与 Gazebo odom 一致
        synth_pos = self._drone_pos.copy()
        synth_pos[2] = max(synth_pos[2], 0.05)  # 不低于地面 5cm
        # P0-2: episode-local RNG for reproducible point cloud
        _obs_seed = self._episode_rng_seed if hasattr(self, '_episode_rng_seed') else None
        if self.reward_version in ("v17", "v2"):
            pc, self._last_pc_meta = self.synth_pc.generate_with_metadata(
                synth_pos,
                self._drone_yaw,
                num_points=4096,
                add_noise=True,
                rng_seed=_obs_seed,
                episode_identity_hash=self._episode_identity_hash,
                decision_index=self._observation_frame_index,
            )
        else:
            pc = self.synth_pc.synthesize(
                synth_pos,
                self._drone_yaw,
                num_points=4096,
                add_noise=True,
                rng_seed=_obs_seed,
            )
        odom_pos = self._world_to_odom(self._drone_pos)
        goal_odom = self._world_to_odom(self.goal_w)
        s9 = self._build_state(odom_pos, self._drone_yaw, goal_odom)
        return {'point_cloud': pc.astype(np.float32), 'agent_pos': s9.astype(np.float32)}

    def _push_obs(self, obs, pointcloud_meta=None):
        self._pc_history = np.roll(self._pc_history, -1, axis=0)
        self._pc_history[-1] = obs['point_cloud']
        self._state_history = np.roll(self._state_history, -1, axis=0)
        self._state_history[-1] = obs['agent_pos']
        if pointcloud_meta is not None:
            self._pc_meta_history = (
                self._pc_meta_history + [pointcloud_meta]
            )[-self.n_obs_steps:]

    def _get_obs_dict(self):
        return {'point_cloud': self._pc_history.copy(),
                'agent_pos': self._state_history.copy()}

    @staticmethod
    def _build_state(pos_w, yaw, goal_w):
        """9D state [pos_world(3), goal_body(3), dist, sin_yaw, cos_yaw].
        与 GazeboUAVEnv._build_state_v65 一致 (但直接用 yaw, 不经 quat)."""
        dx, dy, dz = goal_w[0]-pos_w[0], goal_w[1]-pos_w[1], goal_w[2]-pos_w[2]
        c, s = np.cos(yaw), np.sin(yaw)
        gbx = c*dx + s*dy
        gby = -s*dx + c*dy
        gbz = dz
        gd = float(np.sqrt(dx*dx + dy*dy + dz*dz))
        return np.array([pos_w[0], pos_w[1], pos_w[2],
                         gbx, gby, gbz, gd,
                         np.sin(yaw), np.cos(yaw)], np.float32)

    # ═══════════════════════════════════════════════════════════
    # Reward (PBRS v1.2 + 高度门控 v1.3)
    # ═══════════════════════════════════════════════════════════

    def _compute_reward_v13(self, phi_prev, world_wp, pre_pos, pre_yaw, pre_vel):
        """Reward v1.3 计算 (design: Reward_v1_3_..._Design_v1_2.md).
        pre_pos/pre_yaw/pre_vel = s_t (before step), self._drone_* = s_tp1 (after step)."""
        c = self.reward_v13_cfg
        pos_t = self._drone_pos.copy()

        facts = self._resolve_transition_facts(
            getattr(self, '_last_path', None), self.episode_step + 1
        )
        self._last_transition_facts = facts

        if facts.terminated:
            reached = facts.reached
            bonus = c["r_reach"] if reached else -c["r_collision"]
            support_bonus = compute_episode_terminal_support_bonus(
                self._reward_history, c, reached)
            reward = bonus + support_bonus
            terms = {"base": bonus, "pbrs": 0.0, "onset": 0.0,
                     "future_clearance": 0.0, "trap": 0.0, "flip": 0.0,
                     "support": support_bonus, "guard": 0.0}
            metrics = {"d_soft_h": 0.0, "risk_gate": 0.0, "ttc": 100.0,
                       "ttc_valid": False, "onset_active": False,
                       "trap_active": False, "trap_recovered": False,
                       "side_flip_count": 0, "loop_detected": False,
                       "loop_detected_rate": 0.0, "support_cell_density": 0.0,
                       "low_density_cell": False, "l_free_cone_min": -1.0,
                       "l_free_cone_p20": -1.0, "l_free_cone_median": -1.0,
                       "terminated": True, "truncated": False,
                       "done_reason": facts.done_reason}
            return float(reward), terms, metrics, self._reward_history

        # 非 terminal: 调用 reward_v13 核心
        # s_t = pre-step state, s_tp1 = post-step state
        s_t = {"pos_world": pre_pos.copy(), "yaw": float(pre_yaw),
               "vel_world": pre_vel.copy(), "z": float(pre_pos[2])}
        s_tp1 = {"pos_world": pos_t, "yaw": float(self._drone_yaw),
                 "vel_world": self._drone_vel.copy(), "z": float(pos_t[2]),
                 "collided": False}

        reward, terms, metrics, updated_history = compute_reward_v13(
            s_t, world_wp, s_tp1,
            pointcloud=None,
            obstacle_json=self.obstacles,
            goal_world=self.goal_w,
            history=self._reward_history,
            offline_density=self.offline_density,
            cfg=c,
        )
        metrics["terminated"] = False
        metrics["truncated"] = facts.truncated
        metrics["done_reason"] = facts.done_reason
        return float(reward), terms, metrics, updated_history

    def _compute_phi(self):
        """势能 Φ(s) = -(1-w)·dist - (1-w)·g·e_yaw - h(w)·|z-z_target|
        高度门控 h(w) = h_min + (1-h_min)(1-w_avoid)  [v1.3]"""
        dist = float(np.linalg.norm(self._drone_pos - self.goal_w))
        min_clr, _, _ = self._geom_clearance()
        w_avoid = 1.0 / (1.0 + math.exp(5.0 * (min_clr - self.d_safe)))

        # progress + yaw 门控 (可趋零)
        g = dist if dist < 1.0 else 1.0
        gdir = self.goal_w[:2] - self._drone_pos[:2]
        gnorm = np.linalg.norm(gdir) + 1e-6
        dh = np.array([np.cos(self._drone_yaw), np.sin(self._drone_yaw)])
        e_yaw = 1.0 - np.dot(dh, gdir / gnorm)

        # ★高度门控 v1.3: h(w) ∈ [h_min, 1], 降权不归零
        h_w = self.h_min + (1.0 - self.h_min) * (1.0 - w_avoid)

        return (-(1 - w_avoid) * dist
                - (1 - w_avoid) * g * e_yaw
                - h_w * abs(self._drone_pos[2] - self.z_target))

    def _compute_reward(self, phi_prev):
        phi_next = self._compute_phi()
        self._prev_phi = phi_next
        dist = float(np.linalg.norm(self._drone_pos - self.goal_w))
        facts = self._resolve_transition_facts(
            getattr(self, '_last_path', None), self.episode_step + 1
        )
        self._last_transition_facts = facts

        if facts.terminated:
            r = -phi_prev + (10.0 if facts.reached else -5.0)
            return float(r), True

        # PBRS
        r = self.gamma * phi_next - phi_prev

        # lambda_c 边距惩罚 (v1.2 既有, 避障直接惩罚, 不进 Φ; 用几何 clearance)
        clr, _, _ = self._geom_clearance()
        margin = self.d_safe - clr
        if margin > 0:
            x = self.beta_avoid * margin
            r -= self.lambda_c * (math.log(1 + math.exp(min(x, 20))) / self.beta_avoid)

        # ── 增量2: 持续偏离温和累积 (默认关) ──
        progress = 0.0 if self._prev_dist is None else (self._prev_dist - dist)
        self._prev_dist = dist
        if self.use_stuck_penalty:
            if progress < self.stuck_eps:
                self._count_stuck += 1
            else:
                # 增量3: escape (默认关) — 之前 stuck 且现在有 progress
                if self.use_escape_reward and self._count_stuck > 0:
                    r += self.lambda_escape
                self._count_stuck = 0
            r -= self.lambda_stuck * min(1.0, self._count_stuck / self.n_stuck)

        return float(r), False

    def _resolve_transition_facts(self, path_wp, next_step):
        """Compute the sole terminal record before reward or adapter consumption."""

        path = np.asarray(path_wp if path_wp is not None else [self._drone_pos])
        finite = (
            path.ndim == 2
            and path.shape[1] == 3
            and np.isfinite(path).all()
            and np.isfinite(self._drone_pos).all()
            and np.isfinite(self.goal_w).all()
        )
        if not finite:
            facts = resolve_terminal(
                reached=False, collision=False, ground_crash=False,
                out_of_bounds=False, altitude_exceed=False,
                invalid_state=True, time_limit=False,
                raw_done_reason="invalid_state",
            )
            self._done_reason = "invalid_state"
            return facts

        self._done_reason = "none"
        contact = bool(self._check_collision(path_wp=path))
        raw_contact = str(self._done_reason)
        ground = bool(np.any(path[:, 2] < 0.2))
        collision = bool(contact and raw_contact != "crash_ground")
        out = bool(np.any(np.abs(path[:, :2]) > 30.0))
        altitude = bool(np.any(path[:, 2] > self.z_max + 1.0))
        reached = reached_goal(self._drone_pos, self.goal_w, 0.5)
        raw = (
            raw_contact if contact else
            "out_of_bounds" if out else
            "altitude_exceed" if altitude else
            "reached" if reached else
            "none"
        )
        facts = resolve_terminal(
            reached=reached,
            collision=collision,
            ground_crash=ground,
            out_of_bounds=out,
            altitude_exceed=altitude,
            invalid_state=False,
            time_limit=next_step >= self.max_episode_steps,
            raw_done_reason=raw,
        )
        self._done_reason = raw if facts.terminated else (
            "time_limit" if facts.truncated else "none"
        )
        return facts

    # ═══════════════════════════════════════════════════════════
    # 碰撞 (几何, 复用 collision_geom; 删 synth_pc clr<0.15 和 z骤降)
    # ═══════════════════════════════════════════════════════════

    def _geom_clearance(self):
        """几何 clearance (world frame, 表面距离, 含飞越豁免)."""
        return compute_obstacle_clearance(
            float(self._drone_pos[0]), float(self._drone_pos[1]),
            float(self._drone_pos[2]), self.obstacles, check_3d=True)

    def _check_collision(self, path_wp=None):
        """检查已执行路径的点、地面与相邻点之间的线段，不修改轨迹。

        沿用 collision_thresh 水平判据、0.1m直立半高和严格垂直分离豁免。
        地面沿用 z<0.2 的既有阈值；线性段的最低高度必在端点取得。
        返回接触事实；DPPO adapter 负责让碰撞优先于到达目标。

        Args:
            path_wp: (N,3+) 有序world位置；step传入起点及实际执行点。
                None仅查当前点，不读取历史路径，避免跨episode残留。
        """
        # Preserve existing point checks, then test the space between points.
        check_pts = []
        if path_wp is not None:
            for k in range(len(path_wp)):
                check_pts.append(path_wp[k, POS])
        check_pts.append(self._drone_pos)

        for pt in check_pts:
            min_clr, name, is_flyover = compute_obstacle_clearance(
                float(pt[0]), float(pt[1]), float(pt[2]),
                self.obstacles, check_3d=True)
            # 飞越豁免: 垂直越障合法, 不按 drone_radius 误判 (支持垂直越障)
            if is_flyover:
                # 飞越时只查真穿透 (min_clr 已是垂直间隙, <0 才是真撞顶)
                if min_clr < 0.0:
                    self._done_reason = f'penetrate_flyover({name})'
                    return True
                continue
            # 判撞阈值: surface-to-surface clearance < thresh 才算撞
            if min_clr < self.collision_thresh:
                self._done_reason = f'collide({name},clr={min_clr:.2f}<{self.collision_thresh})'
                return True

        # Any visited ground contact persists even if the final point is airborne.
        if any(float(pt[2]) < 0.2 for pt in check_pts):
            self._done_reason = 'crash_ground'
            return True

        for start, end in zip(check_pts, check_pts[1:]):
            hit = segment_obstacle_collision(start, end, self.obstacles, self.collision_thresh)
            if hit is not None:
                clearance, name = hit
                self._done_reason = f'collide({name},segment_clr={clearance:.2f}<{self.collision_thresh})'
                return True
        return False

    def _check_early_death(self):
        p = self._drone_pos
        if abs(p[0]) > 30 or abs(p[1]) > 30:
            self._done_reason = 'out_of_bounds'
            return True
        if p[2] > self.z_max + 1.0:
            self._done_reason = 'altitude_exceed'
            return True
        return False

    def close(self):
        pass
