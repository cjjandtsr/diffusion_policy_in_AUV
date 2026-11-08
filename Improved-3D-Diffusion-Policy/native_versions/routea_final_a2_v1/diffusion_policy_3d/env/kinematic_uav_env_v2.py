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

import os, math, json, hashlib
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

from .pointcloud_provenance import frame_hit_count, frame_sha256
from diffusion_policy_3d.model.common.synth_pointcloud import (
    DepthCameraIntrinsics,
)
from .traj_utils import (
    action_to_world, normalize_angle,
    DPOS_W, VEL_W, ACC_W, DYAW_CW, YAW_RT, POS, VEL, ACC, YAW)
from .collision_geom import load_obstacles, compute_obstacle_clearance
from .episode_outcome import (
    EpisodeOutcome,
)

if TYPE_CHECKING:
    # Type-only: never imported at runtime (rl → env import cycle).
    from diffusion_policy_3d.rl.dppo_uav.validated_synth_adapter import (
        RayGeometryReport,
    )

# R5.8-C (F-2): early-death terminal penalty — separate from
# r_collision (-20); the policy must not prefer flying out of bounds
# over a real collision.
EARLY_DEATH_PENALTY = -5.0


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
                 collision_thresh: float = 0.35,   # ★ 判撞阈值 = drone_radius (F450 frame 0.45m/2 + 旋翼余量). 补偿 kinematic 无法模拟碰撞后翻滚(翻车)的缺陷
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
                 # ── reward v1.4 (默认启用) ──
                 reward_version: str = 'v14',         # 'v14' | 'v16' | 'legacy'
                 # ── start/goal curriculum sampling (v1.3) ──
                 curriculum_stage: str = 'C2',       # 'C0' | 'C1' | 'C2'
                 episode_role: Optional[str] = None, # anticipatory|dense_recovery|support_expansion|yaw_lateral_recovery
                 seed: Optional[int] = None):

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
        self.reward_version = reward_version
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
        # R7-5 provenance histories — row-aligned with _pc_history (index
        # 0 = oldest pushed frame). _push_obs rolls all of them
        # atomically; the RL observation-window gate cross-checks the
        # recorded values against its own recompute, so drift fails
        # closed. RayGeometryReport is pickle-safe (checkpointed).
        self._cam_far = float(DepthCameraIntrinsics().far)
        self._point_cloud_valid_history = np.zeros(
            self.n_obs_steps, dtype=bool
        )
        self._ray_report_history: List[Optional[RayGeometryReport]] = [
            None
        ] * self.n_obs_steps
        self._point_cloud_frame_hash_history: List[str] = [
            ""
        ] * self.n_obs_steps
        self._prev_phi = 0.0
        self._prev_dist = None
        self._count_stuck = 0
        self._done_reason = "none"
        self._disturb_schedule = None  # [(step, impulse_vec), ...] for deterministic replay
        self._disturb_idx = 0
        self._episode_config = None    # stored for export

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
        self._disturb_idx = 0
        # R5.8-C (F-3): a fresh episode must never report the previous
        # episode's done_reason (stale "collision" on timeout episodes).
        self._done_reason = "none"

        # ── paired eval: 使用预生成的 episode_config ──
        if options and 'episode_config' in options:
            cfg = options['episode_config']
            self._episode_config = cfg
            # World
            if self._multi_world:
                wname = cfg['world_name']
                # find in pool
                for n, jp in self.world_pool:
                    if n == wname:
                        self._set_world(n, jp)
                        break
            # Goal / start
            self.goal_w = np.array(cfg['goal_pos'], dtype=np.float32)
            self._drone_pos = np.array(cfg['start_pos'], dtype=np.float32)
            self._drone_yaw = float(cfg['start_yaw'])
            self._drone_vel = np.zeros(3, np.float32)
            # Disturb schedule: list of [step, impulse_x, impulse_y, impulse_z]
            self._disturb_schedule = cfg.get('disturb_events', [])

            # P0-2 (rework #3): Episode-local RNG seed (stable blake2b,
            # never Python hash()) — the string world_id enters the
            # canonical JSON payload verbatim, so the seed is identical
            # across processes and PYTHONHASHSEED values (RULING-7 P0-2).
            _payload_data = {
                "world_id": int(self.world_id) if (isinstance(self.world_id, int) or str(self.world_id).isdigit()) else str(self.world_id),
                "goal_w": [round(float(v), 4) for v in self.goal_w],
                "drone_pos": [round(float(v), 4) for v in self._drone_pos],
            }
            if seed is not None:
                _payload_data["seed"] = int(seed)
            _payload = json.dumps(_payload_data, sort_keys=True).encode("utf-8")
            _digest = hashlib.blake2b(_payload, digest_size=4).digest()
            self._episode_rng_seed = int.from_bytes(_digest, "little")
        else:
            self._episode_config = None
            self._disturb_schedule = None
            # 0) 多 world 训练: 随机选一个 world
            if self._multi_world:
                idx = int(self._rng.integers(len(self.world_pool)))
                name, json_path = self.world_pool[idx]
                self._set_world(name, json_path)
            # 1) 随机起点/goal (role-aware curriculum, v1.3)
            start_pos, start_yaw, self.goal_w = self._sample_start_goal()
            # 2) 中间态初始化
            if self._rng.random() < self.mid_state_init_ratio:
                start_pos, start_yaw = self._perturb_to_mid_state(start_pos, start_yaw)
            self._drone_pos = start_pos.astype(np.float32)
            self._drone_yaw = float(start_yaw)
            self._drone_vel = np.zeros(3, np.float32)

            # P0-2 (rework #3): stable blake2b, never Python hash()
            _payload_data = {
                "world_id": int(self.world_id) if (isinstance(self.world_id, int) or str(self.world_id).isdigit()) else str(self.world_id),
                "goal_w": [round(float(v), 4) for v in self.goal_w],
                "drone_pos": [round(float(v), 4) for v in self._drone_pos],
            }
            if seed is not None:
                _payload_data["seed"] = int(seed)
            _payload = json.dumps(_payload_data, sort_keys=True).encode("utf-8")
            _digest = hashlib.blake2b(_payload, digest_size=4).digest()
            self._episode_rng_seed = int.from_bytes(_digest, "little")

        # 3) 合成初始点云
        obs = self._get_obs_frame()
        for _ in range(self.n_obs_steps):
            self._push_obs(obs)

        # 4) 初始化 Φ (legacy) / v1.4 history
        self._prev_phi = self._compute_phi()
        if self.reward_version == 'v14':
            from diffusion_policy_3d.env.reward_v14 import create_reward_history_v14
            self._v14_history = create_reward_history_v14()
        elif self.reward_version == 'v16':
            self._v14_history = {}  # v16 doesn't use history
        info = {'world_id': self.world_id}
        if self._episode_config:
            info['episode_id'] = self._episode_config.get('episode_id', '')
        return self._get_obs_dict(), info

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
        # synth_pc: 预加载池取 / 按需建 (factory) — 必须与 obstacles 同 world
        if self.synth_pc_pool is not None:
            self.synth_pc = self.synth_pc_pool[name]
        elif self.synth_pc_factory is not None:
            self.synth_pc = self.synth_pc_factory(json_path)
        # 若两者都 None (单 world 模式不会进这分支), synth_pc 保持 __init__ 设的

    # ═══════════════════════════════════════════════════════════
    # Start/Goal 采样 (v1.3: role-aware curriculum + rejection)
    # ═══════════════════════════════════════════════════════════

    # Role-aware sampling parameters (design doc §4.1.0 v1.3)
    _ROLE_SAMPLING_PARAMS = {
        'anticipatory':         {'d_xy': (10, 24), 'start_cl': 0.60, 'goal_cl': 2.50},
        'dense_recovery':       {'d_xy': (5, 18),  'start_cl': 0.60, 'goal_cl': 1.50},
        'support_expansion':    {'d_xy': (8, 26),  'start_cl': 0.80, 'goal_cl': 2.00},
        'yaw_lateral_recovery': {'d_xy': (6, 18),  'start_cl': 0.80, 'goal_cl': 2.00},
    }

    def _curriculum_xy_range(self):
        """Return (x_min, x_max, y_min, y_max) for current curriculum stage."""
        if self.curriculum_stage == 'C0':
            return (-10.0, 10.0, -6.0, 6.0)
        elif self.curriculum_stage == 'C1':
            return (-12.0, 12.0, -12.0, 12.0)
        else:  # C2
            return (-14.0, 14.0, -14.0, 14.0)

    def _curriculum_z_range(self):
        """Return (z_low, z_high) for current curriculum stage."""
        if self.curriculum_stage == 'C0':
            return (1.3, 3.3)
        elif self.curriculum_stage == 'C1':
            return (1.0, 3.5)
        else:  # C2
            return (1.0, self.z_max)

    def _get_role_params(self, role):
        """Return (d_xy_min, d_xy_max, start_cl, goal_cl) for role."""
        rp = self._ROLE_SAMPLING_PARAMS.get(role)
        if rp is None:
            rp = self._ROLE_SAMPLING_PARAMS['anticipatory']
        return rp['d_xy'][0], rp['d_xy'][1], rp['start_cl'], rp['goal_cl']

    def _sample_start_goal(self, role=None):
        """Unified start/goal sampling with role-aware curriculum + rejection.

        Returns:
            (start_pos, start_yaw, goal_pos) — all world frame, np.float32
        """
        role = role or self._episode_role or 'anticipatory'
        d_xy_min, d_xy_max, start_cl, goal_cl = self._get_role_params(role)
        x_min, x_max, y_min, y_max = self._curriculum_xy_range()
        z_low, z_high = self._curriculum_z_range()
        obstacles = self.obstacles

        for _attempt in range(50):
            # Sample start
            start_x = self._rng.uniform(x_min, x_max)
            start_y = self._rng.uniform(y_min, y_max)
            start_z = self._rng.uniform(z_low + 0.3, z_high - 0.3)

            # Sample goal
            goal_x = self._rng.uniform(x_min, x_max)
            goal_y = self._rng.uniform(y_min, y_max)
            goal_z = self._rng.uniform(z_low + 0.5, z_high - 0.5)

            # Distance checks
            d_xy = math.hypot(goal_x - start_x, goal_y - start_y)
            if d_xy < d_xy_min or d_xy > d_xy_max:
                continue
            d_3d = math.hypot(d_xy, goal_z - start_z)
            if d_3d < 4.0:  # Unified 3D floor
                continue

            # Clearance checks
            start_cl_actual, _, _ = compute_obstacle_clearance(
                start_x, start_y, start_z, obstacles)
            if start_cl_actual < start_cl:
                continue

            goal_cl_actual, _, _ = compute_obstacle_clearance(
                goal_x, goal_y, goal_z, obstacles)
            if goal_cl_actual < goal_cl:
                continue

            # Yaw: face goal + small perturbation
            yaw = math.atan2(goal_y - start_y, goal_x - start_x)
            yaw += self._rng.uniform(-0.3, 0.3)

            start_pos = np.array([start_x, start_y, start_z], dtype=np.float32)
            goal_pos = np.array([goal_x, goal_y, goal_z], dtype=np.float32)
            return start_pos, normalize_angle(yaw), goal_pos

        # Fallback: use nominal positions
        return self._fallback_start_goal()

    def _fallback_start_goal(self):
        """Fallback to nominal start/goal when sampling fails."""
        start = np.array([-8.0, 0.0, self.z_target], dtype=np.float32)
        goal = DEFAULT_GOAL.copy().astype(np.float32)
        d = goal[:2] - start[:2]
        yaw = math.atan2(d[1], d[0])
        return start, yaw, goal

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
        phi_prev = self._prev_phi

        # 0) Capture s_t (for v1.4 reward)
        pos_t_before = self._drone_pos.copy()
        yaw_t_before = float(self._drone_yaw)
        vel_t_before = self._drone_vel.copy()

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

        # 4) 人为扰动: 优先用 deterministic schedule (paired eval), 否则随机
        if self.disturb_enable:
            if self._disturb_schedule is not None and self._disturb_idx < len(self._disturb_schedule):
                # Deterministic replay from seed bank
                evt = self._disturb_schedule[self._disturb_idx]
                if int(evt[0]) == self.episode_step:
                    impulse = np.array(evt[1:4], dtype=np.float32)
                    new_pos = new_pos + impulse
                    self._disturb_idx += 1
            elif self._rng.random() < self.disturb_prob:
                # Random (normal training / seed bank generation)
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
            pc = self.synth_pc.synthesize(wp_pos, wp_yaw,
                                          num_points=4096, add_noise=True,
                                          rng_seed=self._episode_rng_seed + k)
            odom_pos = self._world_to_odom(wp_pos)
            goal_odom = self._world_to_odom(self.goal_w)
            s9 = self._build_state(odom_pos, wp_yaw, goal_odom)
            self._push_obs({'point_cloud': pc.astype(np.float32),
                            'agent_pos': s9.astype(np.float32)})
        obs_dict = self._get_obs_dict()

        # 6) reward
        self._last_path = world_wp[:self.n_exec_steps]   # ★只检实际执行的 4 步路径

        if self.reward_version == 'v14':
            reward, terminated = self._compute_reward_v14(action, pos_t_before, yaw_t_before, vel_t_before)
        elif self.reward_version == 'v16':
            reward, terminated = self._compute_reward_v16(action, pos_t_before, yaw_t_before, vel_t_before)
        else:
            reward, terminated = self._compute_reward(phi_prev)
        self.episode_step += 1
        truncated = self.episode_step >= self.max_episode_steps

        # early death (出界/超高) — 运动学下仍保留边界
        # R5.8-C (F-2): early death is its OWN outcome bucket with a
        # small terminal penalty — it is NOT a collision (the policy
        # must not dodge collision penalties by flying out of bounds).
        early = False
        if not terminated:
            early = self._check_early_death()
            if early:
                terminated = True
                reward = float(reward) + EARLY_DEATH_PENALTY

        # R5.8-C (N5): the SINGLE truth source — outcome carries the
        # normalized fixed-enum done_reason; every consumer (buffer /
        # metrics / trace / evaluator) reads THIS object, never the raw
        # free-form strings (collision subtypes collapse into
        # "collision").
        reason = self._done_reason
        if reason.startswith(("collide", "penetrate")):
            reason = "collision"
        if truncated and reason in ("none",):
            reason = "timeout"
        outcome = EpisodeOutcome(
            success=bool(terminated and reason == "reached"),
            collision=bool(reason == "collision" or reason == "crash_ground"),
            time_limit=bool(truncated),
            out_of_bounds=bool(reason == "out_of_bounds"),
            altitude_exceed=bool(reason == "altitude_exceed"),
            terminated=bool(terminated),
            truncated=bool(truncated),
            done_reason=reason,
        )

        info = {'step': self.episode_step, 'pos': self._drone_pos.copy(),
                'z_world': float(self._drone_pos[2]),
                'outcome': outcome.to_dict()}
        if terminated or truncated:
            dist = float(np.linalg.norm(self._drone_pos - self.goal_w))
            clr, _, _ = self._geom_clearance()
            info['done_reason'] = reason
            info['dist'] = dist
            info['clr'] = clr
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
        pc = self.synth_pc.synthesize(synth_pos,
                                       self._drone_yaw, num_points=4096, add_noise=True,
                                       rng_seed=self._episode_rng_seed)
        odom_pos = self._world_to_odom(self._drone_pos)
        goal_odom = self._world_to_odom(self.goal_w)
        s9 = self._build_state(odom_pos, self._drone_yaw, goal_odom)
        return {'point_cloud': pc.astype(np.float32), 'agent_pos': s9.astype(np.float32)}

    def _push_obs(self, obs):
        self._pc_history = np.roll(self._pc_history, -1, axis=0)
        self._pc_history[-1] = obs['point_cloud']
        self._state_history = np.roll(self._state_history, -1, axis=0)
        self._state_history[-1] = obs['agent_pos']
        # R7-5: per-frame provenance rolled with the frame. The ray
        # report is read here — last_report() still holds THIS frame's
        # report because _push_obs always runs right after the synthesize
        # that produced the frame (reset's repeated pushes replay the
        # same frame and the same report). The gate cross-checks the
        # recorded valid flag and hash against its own recompute.
        report = None
        synth = getattr(self, 'synth_pc', None)
        if synth is not None and hasattr(synth, 'last_report'):
            report = synth.last_report()
        points = np.asarray(obs['point_cloud'], dtype=np.float32)
        self._point_cloud_valid_history = np.roll(
            self._point_cloud_valid_history, -1
        )
        self._point_cloud_valid_history[-1] = self._frame_valid(
            points, report
        )
        self._ray_report_history = self._roll_list(
            self._ray_report_history, report
        )
        self._point_cloud_frame_hash_history = self._roll_list(
            self._point_cloud_frame_hash_history, frame_sha256(points)
        )

    @staticmethod
    def _roll_list(items: list, newest) -> list:
        """Roll a python-list history like np.roll (newest last)."""
        return items[1:] + [newest]

    def _frame_valid(self, points, ray_report) -> bool:
        """Env-side mirror of the gate's hard conditions (R7-5).

        Same five checks as PointCloudValidityGate.validate (shape /
        finite / hit-or-passing-report / unique floor / all-far
        collapse). The production window gate cross-checks this recorded
        flag against its own recompute — drift between the two fails
        closed. Kept independent of the gate module: the env is the
        lower layer and must not import rl.
        """
        if points.shape != (4096, 3):
            return False
        if not np.isfinite(points).all():
            return False
        hit = frame_hit_count(points, self._cam_far)
        if hit <= 0 and (
            ray_report is None or not getattr(ray_report, 'passes', False)
        ):
            return False
        if len(np.unique(points, axis=0)) < 2:
            return False
        if np.allclose(points, self._cam_far):
            return False
        return True

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
        reached = (dist < self.goal_tol_xy
                   and abs(self._drone_pos[2] - self.goal_w[2]) < self.goal_tol_z)
        collision = self._check_collision(path_wp=getattr(self, '_last_path', None))
        terminated = reached or collision

        if terminated:
            # R5.8-C: collision wins over reached — a crash at the goal
            # is a collision, never a success.
            self._done_reason = (
                'reached' if (reached and not collision) else 'collision'
            )
            r = -phi_prev + (10.0 if reached else -5.0)
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

    # ═══════════════════════════════════════════════════════════
    # Reward v1.4 (adaptive-threshold multi-objective)
    # ═══════════════════════════════════════════════════════════

    def _compute_reward_v14(self, action, pos_t, yaw_t, vel_t):
        """v1.4: r = r_goal + r_safety + r_explore + r_recovery."""
        from diffusion_policy_3d.env.reward_v14 import compute_reward_v14

        # Check collision + reached
        collision = self._check_collision(path_wp=getattr(self, '_last_path', None))
        dist = float(np.linalg.norm(self._drone_pos - self.goal_w))
        v_xy = math.hypot(self._drone_vel[0], self._drone_vel[1])
        reached = (dist < self.goal_tol_xy
                   and abs(self._drone_pos[2] - self.goal_w[2]) < self.goal_tol_z
                   and v_xy < 1.0)  # E-03: 对齐Gazebo Monitor deploy口径

        s_t = {'pos_world': pos_t, 'yaw': yaw_t, 'vel_world': vel_t, 'z': float(pos_t[2])}
        s_tp1 = {'pos_world': self._drone_pos.copy(), 'yaw': float(self._drone_yaw),
                 'vel_world': self._drone_vel.copy(), 'z': float(self._drone_pos[2]),
                 'collided': collision,
                 # R5.8-C: the reward consumes the env's OWN reached
                 # criterion — no distance-only re-inference (N5/P1-2).
                 'reached': reached}

        reward, terms, metrics, self._v14_history = compute_reward_v14(
            s_t=s_t, action_chunk=action, s_tp1=s_tp1,
            obstacle_json=self.obstacles, goal_world=self.goal_w,
            history=self._v14_history)

        terminated = reached or collision
        if terminated:
            # R5.8-C: collision wins over reached — a crash at the goal
            # is a collision, never a success.
            self._done_reason = (
                'reached' if (reached and not collision) else 'collision'
            )
        self._collision_detected = collision

        # Store v1.4 terms/metrics for external access (eval scripts)
        self._last_v14_terms = terms
        self._last_v14_metrics = metrics

        return float(reward), terminated

    def _compute_reward_v16(self, action, pos_t, yaw_t, vel_t):
        """v1.6 (E038 修订): r = r_progress + r_smooth + terminal（r_safety 已删，F6）。"""
        from diffusion_policy_3d.env.reward_v16 import compute_reward_v16

        collision = self._check_collision(path_wp=getattr(self, '_last_path', None))
        dist = float(np.linalg.norm(self._drone_pos - self.goal_w))
        v_xy = math.hypot(self._drone_vel[0], self._drone_vel[1])
        reached = (dist < self.goal_tol_xy
                   and abs(self._drone_pos[2] - self.goal_w[2]) < self.goal_tol_z
                   and v_xy < 1.0)

        s_t = {'pos_world': pos_t, 'yaw': yaw_t, 'vel_world': vel_t, 'z': float(pos_t[2])}
        s_tp1 = {'pos_world': self._drone_pos.copy(), 'yaw': float(self._drone_yaw),
                 'vel_world': self._drone_vel.copy(), 'z': float(self._drone_pos[2]),
                 'collided': collision,
                 # R5.8-C: the reward consumes the env's OWN reached
                 # criterion — no distance-only re-inference (N5/P1-2).
                 'reached': reached}

        reward, terms, metrics = compute_reward_v16(
            s_t=s_t, action_chunk=action, s_tp1=s_tp1,
            obstacle_json=self.obstacles, goal_world=self.goal_w)

        terminated = reached or collision
        if terminated:
            # R5.8-C: collision wins over reached — a crash at the goal
            # is a collision, never a success.
            self._done_reason = (
                'reached' if (reached and not collision) else 'collision'
            )
        self._collision_detected = collision

        self._last_v14_terms = terms
        self._last_v14_metrics = metrics

        return float(reward), terminated

    # ═══════════════════════════════════════════════════════════
    # 碰撞 (几何, 复用 collision_geom; 删 synth_pc clr<0.15 和 z骤降)
    # ═══════════════════════════════════════════════════════════

    def _geom_clearance(self):
        """几何 clearance (world frame, 表面距离, 含飞越豁免)."""
        return compute_obstacle_clearance(
            float(self._drone_pos[0]), float(self._drone_pos[1]),
            float(self._drone_pos[2]), self.obstacles, check_3d=True)

    def _check_collision(self, path_wp=None):
        """几何碰撞检测 (纯数学, 比物理引擎更需严密).

        漏洞修复:
        - 漏洞1: 用 drone_radius 阈值 (机身有体积, 非质点). 中心离表面 <
          drone_radius 即撞 (中心穿表面 min_clr<0 是质点判据, 会漏 0.35m).
          飞越豁免不受影响 (is_flyover 时 min_clr 是垂直间隙, drone_radius
          是水平半径, 语义不同, 不扣).
        - 漏洞2: 路径采样防穿墙 (离散 step 从 A 跳到 B, 薄障碍可能被跨过).
          查 A→B 路径上所有中间 waypoint, 任一撞即撞.

        Args:
            path_wp: (N,3+) 本 step 的中间 waypoint 路径 (world_wp). None 则只查当前点.
        """
        # 待查点: 路径采样点 (防穿墙) + 当前终点
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

        # 坠地
        if self._drone_pos[2] < 0.2:
            self._done_reason = 'crash_ground'
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

    def export_episode_config(self, episode_id: str) -> dict:
        """Export current episode config for seed bank (call after reset)."""
        cfg = {
            'episode_id': episode_id,
            'world_name': str(self.world_id) if self._multi_world else WORLD_ENV_MAP.get(self.world_id, str(self.world_id)),
            'goal_pos': self.goal_w.tolist(),
            'start_pos': self._drone_pos.tolist(),
            'start_yaw': float(self._drone_yaw),
            'z_range': [float(self.z_min), float(self.z_max)],
            'disturb_enable': self.disturb_enable,
        }
        if self._episode_config:
            cfg['disturb_events'] = self._episode_config.get('disturb_events', [])
        return cfg

    @property
    def drone_pos(self):
        return self._drone_pos.copy()

    @property
    def drone_yaw(self):
        return float(self._drone_yaw)