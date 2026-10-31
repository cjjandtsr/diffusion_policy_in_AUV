"""
traj_utils.py — 抽自 ml_planner_gazebo.py 的纯数学函数 (无 ROS 依赖)

可被 RL 节点和 planner 共用。
"""

import numpy as np
from threading import Lock
from typing import Optional, Tuple


# ═══════════════════════════════════════════════════════════════
# action 索引常量 (与 planner 一致)
# ═══════════════════════════════════════════════════════════════
DPOS_W  = slice(0, 3)
VEL_W   = slice(3, 6)
ACC_W   = slice(6, 9)
DYAW_CW = 9
YAW_RT  = 10

POS  = slice(0, 3)
VEL  = slice(3, 6)
ACC  = slice(6, 9)
YAW  = 9


# ═══════════════════════════════════════════════════════════════
# 四元数 → yaw (selfless, 抽自 L1532)
# ═══════════════════════════════════════════════════════════════

def quat_to_yaw(q_xyzw):
    x, y, z, w = q_xyzw
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


# ═══════════════════════════════════════════════════════════════
# 角度归一化
# ═══════════════════════════════════════════════════════════════

def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


# ═══════════════════════════════════════════════════════════════
# action delta → absolute world waypoints (抽自 L1092-1130)
# ═══════════════════════════════════════════════════════════════

def action_to_world(raw_action: np.ndarray,
                    pos_w: np.ndarray,
                    vel_world: np.ndarray,
                    yaw: float) -> np.ndarray:
    """delta action (N, 11) → absolute world-frame waypoints (N, 11)

    Args:
        raw_action: (N, 11) [dpos_w(3), dvel_w(3), acc_w(3), dyaw, yaw_rate]
        pos_w: (3,) current world position
        vel_world: (3,) current world velocity
        yaw: float current yaw

    Returns:
        world_action: (N, 11) absolute waypoints
            columns: POS(0:3), VEL(3:6), ACC(6:9), YAW(9), YAW_RT(10)
    """
    N = len(raw_action)
    positions = np.zeros((N, 3), dtype=np.float32)
    yaws = np.zeros(N, dtype=np.float32)
    cur_pos = pos_w.copy()
    cur_yaw = yaw
    for k in range(N):
        cur_pos = cur_pos + raw_action[k, DPOS_W]
        cur_yaw = normalize_angle(cur_yaw + float(raw_action[k, DYAW_CW]))
        positions[k] = cur_pos
        yaws[k] = cur_yaw

    world_action = np.zeros_like(raw_action, dtype=np.float32)
    world_action[:, POS]    = positions
    world_action[:, VEL]    = raw_action[:, VEL_W]
    world_action[:, ACC]    = raw_action[:, ACC_W]
    world_action[:, YAW]    = yaws
    world_action[:, YAW_RT] = raw_action[:, YAW_RT]
    return world_action


# ═══════════════════════════════════════════════════════════════
# TrajectoryManager (抽自 L244-283, 纯数学, 不含 rospy.Time)
# ═══════════════════════════════════════════════════════════════

class TrajectoryManager:
    """轨迹插值器: set waypoints → sample(t) → 线性插值 pos/vel/acc/yaw/yr"""

    MAX_JUMP_M = 0.5

    def __init__(self, traj_dt: float = 0.3):
        self.lock = Lock()
        self.waypoints: Optional[np.ndarray] = None
        self.start_time: Optional[float] = None
        self.traj_dt = traj_dt
        self.duration = 0.0

    def set_trajectory(self, world_action: np.ndarray,
                       traj_dt: float,
                       cur_pos: Optional[np.ndarray] = None,
                       t_start: Optional[float] = None):
        """存入绝对 world waypoints

        Args:
            world_action: (N, 11+) waypoints
            traj_dt: 步长时间 (0.3)
            cur_pos: 当前实际位置, 若与 wapoints[0] 偏差大则平移对齐
            t_start: 起始时间 (秒). None=用 time.time()
        """
        with self.lock:
            traj = world_action.copy()
            if cur_pos is not None:
                delta = cur_pos - traj[0, POS]
                if np.linalg.norm(delta) > self.MAX_JUMP_M:
                    traj[:, POS] += delta
            self.waypoints = traj
            import time
            self.start_time = t_start if t_start is not None else time.time()
            self.traj_dt = traj_dt
            self.duration = (len(traj) - 1) * traj_dt

    def sample(self, t_now: float) -> Optional[Tuple]:
        """给定当前时间, 线性插值返回 (pos, vel, acc, yaw, yr, progress)

        Returns None if no trajectory set.
        """
        with self.lock:
            if self.waypoints is None:
                return None
            t = max(0.0, t_now - self.start_time)
            progress = t / self.duration if self.duration > 0 else 1.0
            t_idx = t / self.traj_dt
            i = int(t_idx)
            frac = t_idx - i
            N = len(self.waypoints)
            if i >= N - 1:
                wp = self.waypoints[-1]
                return (wp[POS].copy(), np.zeros(3), np.zeros(3),
                        float(wp[YAW]), 0.0, progress)
            wp0, wp1 = self.waypoints[i], self.waypoints[i + 1]
            return (wp0[POS] * (1 - frac) + wp1[POS] * frac,
                    wp0[VEL] * (1 - frac) + wp1[VEL] * frac,
                    wp0[ACC] * (1 - frac) + wp1[ACC] * frac,
                    float(wp0[YAW] * (1 - frac) + wp1[YAW] * frac),
                    float(wp0[YAW_RT] * (1 - frac) + wp1[YAW_RT] * frac),
                    progress)

    def is_valid(self) -> bool:
        with self.lock:
            return self.waypoints is not None

    def clear(self):
        with self.lock:
            self.waypoints = None
