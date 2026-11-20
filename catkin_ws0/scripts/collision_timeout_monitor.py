#!/usr/bin/env python3
"""
collision_timeout_monitor.py — External collision/timeout/success monitor
=========================================================================
iDP3-UAV OOD Experiment — Task 1 (P0)

Independent ROS node that monitors a closed-loop trial. Does NOT modify
the deployment node (ml_planner_gazebo.py). Subscribes to Gazebo model
states, detects collisions, enforces timeout, and records dual-threshold
success (deploy口径 + FlightBench口径).

Architecture:
  - Reads .obstacles.json for obstacle geometry (from OOD world generator)
  - Subscribes to /gazebo/model_states for drone + obstacle positions
  - Computes per-obstacle clearance → detects collision
  - Subscribes to /mavros/local_position/odom for drone velocity
  - Tracks min_clearance across entire episode
  - Enforces 60s timeout per waypoint
  - Dual success判定: deploy (0.5m xy / 0.3m z / 1.0 m/s) + FlightBench (1.5m xy)
  - Outputs per-trial JSON with all metrics

Coordinate systems:
  - goal_x/y/z: LOCAL frame (NED, origin = takeoff point), same as planner waypoint
  - obstacles: WORLD frame (from .obstacles.json / Gazebo)
  - Collision check: WORLD frame (model_states vs obstacles)
  - Success check: LOCAL frame (odometry vs goal) — matches planner's判定

Usage:
  # goal is in LOCAL coordinates (same as ml_planner_gazebo_fm.launch waypoint)
  rosrun ego_planner collision_timeout_monitor.py \
    --obstacles_json env_014_obs14.obstacles.json \
    --goal_x 12.0 --goal_y 0.0 --goal_z 2.0 \
    --drone_radius 0.35 --timeout 60.0 \
    --output trial_001.json
"""

import argparse
import json
import math
import os
import signal
import sys
import time
import threading
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose, Twist, Point, Vector3
from nav_msgs.msg import Odometry
from std_msgs.msg import Header


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ObstacleSpec:
    """Obstacle geometry loaded from .obstacles.json."""
    name: str
    shape: str  # "cylinder" or "box"
    x: float
    y: float
    z: float       # center z (= h/2 for ground-sitting)
    yaw: float = 0.0
    # cylinder
    radius: float = 0.0
    height: float = 0.0
    # box
    size_x: float = 0.0
    size_y: float = 0.0
    size_z: float = 0.0

    @property
    def bounding_r(self) -> float:
        """Bounding circle radius in XY plane."""
        if self.shape == "cylinder":
            return self.radius
        return math.hypot(self.size_x, self.size_y) / 2.0

    @classmethod
    def from_dict(cls, d: dict) -> "ObstacleSpec":
        return cls(
            name=d.get("name", "unknown"),
            shape=d.get("shape", "cylinder"),
            x=d.get("x", 0.0),
            y=d.get("y", 0.0),
            z=d.get("z", d.get("height", 1.0) / 2 if d.get("shape") == "cylinder"
                     else d.get("size_z", 1.0) / 2),
            yaw=d.get("yaw", 0.0),
            radius=d.get("radius", 0.0),
            height=d.get("height", 0.0),
            size_x=d.get("size_x", 0.0),
            size_y=d.get("size_y", 0.0),
            size_z=d.get("size_z", 0.0),
        )


@dataclass
class TrialResult:
    """Complete result for one trial."""
    world_id: str = ""  # 2026-11-07: 任务表 world_id 是 world 名字符串, 非数字
    trial: int = 0
    tier: str = ""
    filename: str = ""

    # Outcome
    success_deploy: bool = False
    success_flightbench: bool = False
    collision: bool = False
    timeout: bool = False
    lost: bool = False  # drone flew out of bounds

    # Timing
    start_time: float = 0.0
    end_time: float = 0.0
    time_to_goal: float = 0.0

    # Clearance
    min_clearance: float = float("inf")
    clearance_history: List[float] = field(default_factory=list)

    # Final state
    final_dist_xy: float = 0.0
    final_dist_z: float = 0.0
    final_vel_xy: float = 0.0
    final_x: float = 0.0
    final_y: float = 0.0
    final_z: float = 0.0

    # Intermediate checkpoints (optional)
    checkpoint_times: List[float] = field(default_factory=list)
    checkpoint_dists: List[float] = field(default_factory=list)
    n_collisions: int = 0
    crash_reason: str = ""   # "" / "obstacle" / "flipped" / "crashed_ground"

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            if k == "clearance_history":
                # Sample to keep JSON size manageable
                if len(v) > 1000:
                    step = len(v) // 500
                    d[k] = [v[i] for i in range(0, len(v), max(1, step))]
                else:
                    d[k] = v
            else:
                d[k] = v
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Collision detection
# ─────────────────────────────────────────────────────────────────────────────

def point_to_oriented_rect_distance(
    px: float, py: float,
    cx: float, cy: float,
    sx: float, sy: float,
    yaw: float,
) -> float:
    """
    Shortest distance from point (px,py) to an oriented rectangle.
    Rectangle: center (cx,cy), half-extents (sx/2, sy/2), rotated by yaw.

    Transforms point to box-local frame, then computes distance to
    axis-aligned rectangle.
    """
    # Transform point to box-local coordinates
    dx = px - cx
    dy = py - cy
    cos_y = math.cos(-yaw)
    sin_y = math.sin(-yaw)
    lx = dx * cos_y - dy * sin_y
    ly = dx * sin_y + dy * cos_y

    hx = sx / 2.0
    hy = sy / 2.0

    # Distance to axis-aligned rectangle in local frame
    cx_dist = max(abs(lx) - hx, 0.0)
    cy_dist = max(abs(ly) - hy, 0.0)
    return math.hypot(cx_dist, cy_dist)


def compute_obstacle_clearance(
    drone_x: float, drone_y: float, drone_z: float,
    obstacles: List[ObstacleSpec],
    check_3d: bool = True,
) -> Tuple[float, Optional[str], bool]:
    """
    Compute minimum clearance from drone to all obstacles.

    Args:
        drone_x, drone_y, drone_z: drone position (world frame)
        obstacles: list of obstacle specs
        check_3d: if True, consider vertical overlap (drone above obstacle
                  is NOT a collision)

    Returns:
        (min_clearance, closest_obstacle_name, closest_is_flyover)
        min_clearance = surface-to-surface distance (negative if penetrating)
        closest_is_flyover = True 表示最近障碍处于"无人机从顶部上方>5cm飞越"状态,
                             此时即使 XY 重叠也不算碰撞 (选项1: 飞越完全豁免)
    """
    min_clear = float("inf")
    closest_name = None
    closest_flyover = False

    for obs in obstacles:
        is_flyover = False
        if obs.shape == "cylinder":
            # XY distance from obstacle center to drone
            d_xy = math.hypot(drone_x - obs.x, drone_y - obs.y)
            clear_xy = d_xy - obs.radius

            if check_3d:
                # Vertical check: drone above or below obstacle
                obs_top = obs.z + obs.height / 2.0
                obs_bot = obs.z - obs.height / 2.0
                drone_radius_z = 0.1   # iris 机身垂直半高(实测约0.1m, 原0.3过大致误报)
                FLYOVER_MARGIN = 0.05  # 无人机底部高出障碍顶 >5cm 即视为合法飞越
                drone_top = drone_z + drone_radius_z
                drone_bot = drone_z - drone_radius_z

                # 飞越判定: 无人机底部高出障碍顶 >5cm (或整体在障碍底下方)
                if drone_bot > (obs_top - FLYOVER_MARGIN) or drone_top < (obs_bot + FLYOVER_MARGIN):
                    # 合法垂直飞越: clearance 报真实垂直间隙, 标记 flyover (碰撞判定豁免)
                    is_flyover = True
                    dz = min(
                        abs(drone_bot - obs_top),
                        abs(drone_top - obs_bot),
                    )
                    clear_3d = math.hypot(max(clear_xy, 0.0), dz)
                elif clear_xy >= 0:
                    # Horizontal separation but vertical overlap
                    clear_3d = clear_xy
                else:
                    # Penetrating in XY and vertically overlapping → collision
                    clear_3d = clear_xy  # negative
            else:
                clear_3d = clear_xy

            if clear_3d < min_clear:
                min_clear = clear_3d
                closest_name = obs.name
                closest_flyover = is_flyover

        elif obs.shape == "box":
            # XY distance to oriented rectangle
            d_surf = point_to_oriented_rect_distance(
                drone_x, drone_y,
                obs.x, obs.y,
                obs.size_x, obs.size_y,
                obs.yaw,
            )

            if check_3d:
                obs_top = obs.z + obs.size_z / 2.0
                obs_bot = obs.z - obs.size_z / 2.0
                drone_radius_z = 0.1   # iris 机身垂直半高(与cylinder段一致)
                FLYOVER_MARGIN = 0.05  # 5cm 飞越容差
                drone_top = drone_z + drone_radius_z
                drone_bot = drone_z - drone_radius_z

                if drone_bot > (obs_top - FLYOVER_MARGIN) or drone_top < (obs_bot + FLYOVER_MARGIN):
                    is_flyover = True
                    dz = min(
                        abs(drone_bot - obs_top),
                        abs(drone_top - obs_bot),
                    )
                    clear_3d = math.hypot(max(d_surf, 0.0), dz)
                elif d_surf >= 0:
                    clear_3d = d_surf
                else:
                    clear_3d = d_surf
            else:
                clear_3d = d_surf

            if clear_3d < min_clear:
                min_clear = clear_3d
                closest_name = obs.name
                closest_flyover = is_flyover

    return min_clear, closest_name, closest_flyover


# ─────────────────────────────────────────────────────────────────────────────
# Load obstacles
# ─────────────────────────────────────────────────────────────────────────────

def load_obstacles(json_path: str) -> List[ObstacleSpec]:
    """Load obstacle specs from .obstacles.json."""
    with open(json_path) as f:
        data = json.load(f)
    return [ObstacleSpec.from_dict(d) for d in data.get("obstacles", [])]


# ─────────────────────────────────────────────────────────────────────────────
# Monitor node
# ─────────────────────────────────────────────────────────────────────────────

class CollisionTimeoutMonitor:
    """
    ROS node that monitors a single closed-loop trial.

    Subscribes to:
      - /gazebo/model_states (drone + obstacle poses)
      - /mavros/local_position/odom (drone velocity)

    Detects:
      - collision (drone penetrates any obstacle)
      - timeout (elapsed > timeout_s since trial start)
      - success_deploy (dxy<0.5, dz<0.3, v_xy<1.0, debounced 5 frames)
      - success_flightbench (dxy<1.5, no collision)
      - lost (drone outside ±25 xy or z<-0.5 or z>10)
        [2026-11-07] xy 界由 ±15 放宽到 ±25: 任务 world 的 obstacle_region 是 ±14,
        起点 y 可到 13.1, ±15 只剩 <2 m 余量, 悬停漂移即误判 lost。
        [2026-11-07] z 上界由 20 收到 10: 任务表 start/goal z ∈ [1.42, 4.70],
        与 xy=25 同为"任务工作包络"型越界判据, z 上界收到 10 m。
    """

    def __init__(
        self,
        obstacles: List[ObstacleSpec],
        goal: Tuple[float, float, float],
        drone_collision_radius: float = 0.35,
        timeout_s: float = 60.0,
        debounce_frames: int = 5,
        control_freq: float = 50.0,
        bounds_xy: float = 25.0,
        bounds_z_max: float = 10.0,
    ):
        self.obstacles = obstacles
        self.goal = goal
        self.drone_collision_radius = drone_collision_radius
        self.timeout_s = timeout_s
        self.debounce_frames = debounce_frames
        # 越界判据 (world 系 |x|,|y| 上限)。任务 world 的 obstacle_region 为 ±14,
        # 起点 y 可达 13.1, 故默认 25 而非原 15 (2026-11-07 单次试飞实测误判)。
        self.bounds_xy = bounds_xy
        # 越界判据 (world 系 z 上界)。原实现写死 20 m, 对 z≤4.7 的任务无判别力;
        # 收到 10 m 与 xy=25 一起构成任务工作包络 (2026-11-07)。
        self.bounds_z_max = bounds_z_max

        # State
        self._lock = threading.Lock()
        self._drone_x = 0.0      # world frame (from model_states)
        self._drone_y = 0.0
        self._drone_z = 0.0
        self._drone_vx = 0.0
        self._drone_vy = 0.0
        self._drone_vz = 0.0
        self._drone_roll = 0.0   # 姿态 (rad), 用于翻车检测
        self._drone_pitch = 0.0
        self._local_x = 0.0      # local frame (from odom, NED origin=takeoff)
        self._local_y = 0.0
        self._local_z = 0.0
        self._drone_name = "iris_D435i"
        self._latest_odom_stamp = None
        self._latest_model_stamp = None

        # Results
        self.result = TrialResult()
        self._start_time = None
        self._goal_reach_consec = 0
        self._prev_dxy = float("inf")
        self._min_clearance = float("inf")
        self._n_collisions = 0
        self._last_collision_time = -1.0

        # Termination
        self._done = False
        self._done_reason = ""

        # Trajectory recording
        self._record_traj = True  # always record when monitor runs
        self._state_history: List[Tuple[float, float, float, float]] = []  # (t, lx, ly, lz)
        self._pos_cmd_history: List[Tuple[float, float, float, float, float, float, float]] = []  # (t, px,py,pz,vx,vy,vz)

        # Subscribers
        self._model_sub = rospy.Subscriber(
            "/gazebo/model_states", ModelStates,
            self._model_states_cb, queue_size=1,
        )
        self._odom_sub = rospy.Subscriber(
            "/mavros/local_position/odom", Odometry,
            self._odom_cb, queue_size=1,
        )
        # EGO /planning/pos_cmd for trajectory recording
        try:
            from quadrotor_msgs.msg import PositionCommand
            self._pos_cmd_sub = rospy.Subscriber(
                "/planning/pos_cmd", PositionCommand,
                self._pos_cmd_cb, queue_size=10,
            )
            rospy.loginfo("[Monitor] Subscribed to /planning/pos_cmd for trajectory recording")
        except ImportError:
            rospy.logwarn("[Monitor] quadrotor_msgs not available, skipping pos_cmd recording")
            self._pos_cmd_sub = None

    def _model_states_cb(self, msg: ModelStates):
        """Extract drone position from /gazebo/model_states."""
        try:
            idx = msg.name.index(self._drone_name)
        except ValueError:
            # Try common alternative names
            for alt in ["iris", "iris_iris", "uav"]:
                try:
                    idx = msg.name.index(alt)
                    self._drone_name = alt
                    break
                except ValueError:
                    continue
            else:
                return  # drone not found in model states yet

        pose = msg.pose[idx]
        twist = msg.twist[idx]
        # 四元数 → roll/pitch (翻车检测)
        q = pose.orientation
        import math as _m
        sinr = 2.0 * (q.w * q.x + q.y * q.z)
        cosr = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        roll = _m.atan2(sinr, cosr)
        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        sinp = max(-1.0, min(1.0, sinp))
        pitch = _m.asin(sinp)
        with self._lock:
            self._drone_x = pose.position.x
            self._drone_y = pose.position.y
            self._drone_z = pose.position.z
            self._drone_vx = twist.linear.x
            self._drone_vy = twist.linear.y
            self._drone_vz = twist.linear.z
            self._drone_roll = roll
            self._drone_pitch = pitch
            self._latest_model_stamp = rospy.Time.now()

    def _odom_cb(self, msg: Odometry):
        """Extract drone LOCAL position + velocity from odometry."""
        with self._lock:
            # Local frame position (NED, origin = takeoff point)
            self._local_x = msg.pose.pose.position.x
            self._local_y = msg.pose.pose.position.y
            self._local_z = msg.pose.pose.position.z
            self._drone_vx = msg.twist.twist.linear.x
            self._drone_vy = msg.twist.twist.linear.y
            self._drone_vz = msg.twist.twist.linear.z
            self._latest_odom_stamp = msg.header.stamp
            # Record state history for trajectory analysis
            if self._record_traj:
                t = msg.header.stamp.to_sec() if msg.header.stamp else time.time()
                self._state_history.append((t, self._local_x, self._local_y, self._local_z))

    def _pos_cmd_cb(self, msg):
        """Record EGO planned trajectory from /planning/pos_cmd."""
        if self._record_traj:
            t = msg.header.stamp.to_sec() if msg.header.stamp else time.time()
            self._pos_cmd_history.append((
                t,
                msg.position.x, msg.position.y, msg.position.z,
                msg.velocity.x, msg.velocity.y, msg.velocity.z,
            ))

    def _check_lost(self) -> bool:
        """Check if drone is out of bounds."""
        x, y, z = self._drone_x, self._drone_y, self._drone_z
        # 2026-11-07: ±15 → ±25。任务 world 的 obstacle_region 为 ±14, 起点 y 可达 13.1,
        # 无人机悬停/漂移 1.9 m 就会越过 15 被误判 lost (任务 000113 实测)。
        if abs(x) > self.bounds_xy or abs(y) > self.bounds_xy:
            return True
        if z < -0.5 or z > self.bounds_z_max:
            return True
        return False

    def _update(self) -> Optional[str]:
        """
        Update monitor state. Called at ~50Hz in spin loop.

        Returns:
            None if trial continues, or termination reason string.
        """
        now = time.time()
        if self._start_time is None:
            self._start_time = now
            self.result.start_time = now

        elapsed = now - self._start_time

        with self._lock:
            # World frame (for collision check against obstacles)
            wx = self._drone_x
            wy = self._drone_y
            wz = self._drone_z
            # Local frame (for success check — matches planner's coordinate system)
            lx = self._local_x
            ly = self._local_y
            lz = self._local_z
            vx = self._drone_vx
            vy = self._drone_vy
            roll = self._drone_roll
            pitch = self._drone_pitch

        gx, gy, gz = self.goal  # LOCAL coordinates (same as planner waypoint)

        # ── Check: 坠地 / 翻车 (无人机自身状态异常, 非障碍碰撞) ──
        # 起飞缓冲: 前 8s 不判 (起飞初期 z 低、姿态可能瞬时大)
        import math as _m
        if elapsed > 8.0:
            # 翻车: roll 或 pitch 绝对值 > 90° (1.571 rad), 即彻底侧翻/倒扣
            if abs(roll) > 1.571 or abs(pitch) > 1.571:
                rospy.logwarn(f"[Monitor] 翻车! roll={_m.degrees(roll):.0f}° pitch={_m.degrees(pitch):.0f}°")
                self.result.collision = True
                self.result.n_collisions = self._n_collisions
                self.result.crash_reason = "flipped"
                return "collision"
            # 坠地: world z < 0.3m (倒在地面)
            if wz < 0.1:
                rospy.logwarn(f"[Monitor] 坠地! world_z={wz:.2f}m")
                self.result.collision = True
                self.result.crash_reason = "crashed_ground"
                return "collision"

        # ── Check: Lost ──
        if self._check_lost():
            self.result.lost = True
            # 2026-11-07 修复: 原来直接 return, 早于下方 final_* 赋值, 使 final_dist_xy
            # 保持默认 0.0, 再被 run() 末尾的 `dxy<1.5 → success_flightbench=True`
            # 误记成"到达成功"。此处补记最终位置与距离。
            self.result.final_x = wx
            self.result.final_y = wy
            self.result.final_z = wz
            self.result.final_dist_xy = math.hypot(lx - gx, ly - gy)
            self.result.final_dist_z = abs(lz - gz)
            self.result.final_vel_xy = math.hypot(vx, vy)
            self.result.min_clearance = self._min_clearance
            return "lost"

        # ── Check: Timeout ──
        if elapsed > self.timeout_s:
            self.result.timeout = True
            return "timeout"

        # ── Check: Collision (WORLD frame against obstacles) ──
        clearance, closest, is_flyover = compute_obstacle_clearance(
            wx, wy, wz, self.obstacles, check_3d=True,
        )
        self.result.clearance_history.append(clearance)
        # DEBUG: log every 100th frame to verify position vs clearance
        if len(self.result.clearance_history) % 100 == 1:
            rospy.loginfo_throttle(
                5.0,
                f"[Monitor DEBUG] world=({wx:.2f},{wy:.2f},{wz:.2f}) "
                f"local=({lx:.2f},{ly:.2f},{lz:.2f}) "
                f"clearance={clearance:.3f}m closest={closest} "
                f"flyover={is_flyover} n_obs={len(self.obstacles)}"
            )
        if clearance < self._min_clearance:
            self._min_clearance = clearance

        # 障碍碰撞 (放宽): 只在明显穿模时判, 贴近/擦过不判.
        # 主要的炸机判定靠上面的姿态(翻车>90°)和坠地(z<0.3)检测.
        #   穿模深度 < -0.1m (无人机中心嵌入障碍表面 0.1m 以上) 才算撞
        COLLISION_PENETRATION = -0.10
        if clearance < COLLISION_PENETRATION and not is_flyover:
            if now - self._last_collision_time > 0.3:
                self._n_collisions += 1
                self._last_collision_time = now
            if self._n_collisions >= 2:
                self.result.collision = True
                self.result.n_collisions = self._n_collisions
                self.result.crash_reason = "obstacle"
                return "collision"
        else:
            if self._n_collisions > 0 and clearance > 0:
                self._n_collisions = 0

        # ── Check: Success, deploy口径 (LOCAL frame — matches planner) ──
        dxy = math.hypot(lx - gx, ly - gy)
        dz_abs = abs(lz - gz)
        v_xy = math.hypot(vx, vy)

        in_box = (dxy < 0.5) and (dz_abs < 0.3)
        slow_enough = (v_xy < 1.0)

        if in_box and slow_enough:
            self._goal_reach_consec += 1
        else:
            self._goal_reach_consec = 0

        if self._goal_reach_consec >= self.debounce_frames:
            self.result.success_deploy = True
            return "success_deploy"

        # ── Check: Success (FlightBench口径, looser) ──
        if dxy < 1.5 and not self.result.collision and elapsed > 2.0:
            self.result.success_flightbench = True

        # Track progress
        self._prev_dxy = dxy
        self.result.final_dist_xy = dxy
        self.result.final_dist_z = dz_abs
        self.result.final_vel_xy = v_xy
        self.result.final_x = wx
        self.result.final_y = wy
        self.result.final_z = wz
        self.result.min_clearance = self._min_clearance

        return None  # continue

    def run(self, rate_hz: float = 50.0) -> TrialResult:
        """
        Block until trial terminates (success / collision / timeout / lost).

        Returns TrialResult with all metrics. Gracefully handles Ctrl+C.
        """
        rate = rospy.Rate(rate_hz)
        rospy.loginfo(
            f"[Monitor] Starting trial. Goal=({self.goal[0]:.1f}, "
            f"{self.goal[1]:.1f}, {self.goal[2]:.1f}), "
            f"timeout={self.timeout_s}s, obs={len(self.obstacles)}"
        )

        try:
            while not rospy.is_shutdown() and not self._done:
                reason = self._update()
                if reason is not None:
                    self._done = True
                    self._done_reason = reason
                    break
                rate.sleep()
        except (rospy.ROSInterruptException, KeyboardInterrupt):
            rospy.loginfo("[Monitor] Interrupted — saving partial results")
            self._done_reason = "interrupted"

        # Finalize result
        now = time.time()
        self.result.end_time = now
        self.result.time_to_goal = now - self._start_time if self._start_time else 0.0
        self.result.min_clearance = self._min_clearance
        self.result.n_collisions = self._n_collisions
        self.result.clearance_history = list(self.result.clearance_history)

        # FlightBench success: dxy < 1.5 and no collision
        # 2026-11-07: 排除 lost —— 越界逃逸不算到达 (原实现会把 lost 记成成功)
        if (self.result.final_dist_xy < 1.5
                and not self.result.collision
                and not self.result.lost):
            self.result.success_flightbench = True

        rospy.loginfo(
            f"[Monitor] Trial ended: reason={self._done_reason}, "
            f"deploy_success={self.result.success_deploy}, "
            f"fb_success={self.result.success_flightbench}, "
            f"collision={self.result.collision}, timeout={self.result.timeout}, "
            f"lost={self.result.lost}, "
            f"T={self.result.time_to_goal:.1f}s, "
            f"min_clear={self.result.min_clearance:.3f}m, "
            f"final_dxy={self.result.final_dist_xy:.3f}m"
        )

        return self.result

    def write_trajectory_csvs(self, base_path: str):
        """Write state_history and pos_cmd_history to CSV files."""
        import csv as _csv
        # State CSV (drone actual position over time)
        if self._state_history:
            state_path = base_path.replace('.json', '_state.csv')
            with open(state_path, 'w', newline='') as f:
                w = _csv.writer(f)
                w.writerow(['timestamp', 'px', 'py', 'pz'])
                for t, lx, ly, lz in self._state_history:
                    w.writerow([f'{t:.6f}', f'{lx:.6f}', f'{ly:.6f}', f'{lz:.6f}'])
            rospy.loginfo(f"[Monitor] State history saved: {len(self._state_history)} frames → {state_path}")
        # Pos_cmd CSV (EGO planned trajectory)
        if self._pos_cmd_history:
            cmd_path = base_path.replace('.json', '_pos_cmd.csv')
            with open(cmd_path, 'w', newline='') as f:
                w = _csv.writer(f)
                w.writerow(['timestamp', 'cmd_px', 'cmd_py', 'cmd_pz', 'cmd_vx', 'cmd_vy', 'cmd_vz'])
                for row in self._pos_cmd_history:
                    w.writerow([f'{v:.6f}' for v in row])
            rospy.loginfo(f"[Monitor] Pos_cmd history saved: {len(self._pos_cmd_history)} entries → {cmd_path}")

    def shutdown(self):
        self._model_sub.unregister()
        self._odom_sub.unregister()
        if hasattr(self, '_pos_cmd_sub') and self._pos_cmd_sub:
            self._pos_cmd_sub.unregister()


# ─────────────────────────────────────────────────────────────────────────────
# Standalone helper: detect goal from mavros/local_position/odom
# ─────────────────────────────────────────────────────────────────────────────

def wait_for_drone_appear(timeout: float = 10.0, drone_name: str = "iris") -> bool:
    """Wait until drone appears in /gazebo/model_states."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            msg = rospy.wait_for_message(
                "/gazebo/model_states", ModelStates, timeout=2.0
            )
            if drone_name in msg.name:
                return True
            # Try alternatives
            for alt in ["iris_iris", "uav"]:
                if alt in msg.name:
                    return True
        except rospy.ROSException:
            pass
    return False


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Collision/Timeout/Success monitor for closed-loop trials"
    )
    parser.add_argument("--obstacles_json", required=True,
                        help="Path to .obstacles.json for this world")
    parser.add_argument("--goal_x", type=float, default=12.0)
    parser.add_argument("--goal_y", type=float, default=0.0)
    parser.add_argument("--goal_z", type=float, default=2.0)
    parser.add_argument("--drone_radius", type=float, default=0.35,
                        help="Drone collision radius (meters)")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="Timeout per waypoint (seconds)")
    parser.add_argument("--debounce_frames", type=int, default=5,
                        help="Success debounce frames")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: stdout)")
    parser.add_argument("--world_id", type=str, default="",
                        help="World id (任务表为 world 名字符串)")
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--tier", type=str, default="unknown")
    parser.add_argument("--filename", type=str, default="")
    parser.add_argument("--drone_name", type=str, default="iris_D435i",
                        help="Drone model name in Gazebo")
    parser.add_argument("--bounds_xy", type=float, default=25.0,
                        help="Lost 判据: world 系 |x|,|y| 上限 (米), 默认 25")
    parser.add_argument("--bounds_z_max", type=float, default=10.0,
                        help="Lost 判据: world 系 z 上界 (米), 默认 10")
    args = parser.parse_args()

    rospy.init_node("collision_timeout_monitor", anonymous=True)

    # Load obstacles
    obstacles = load_obstacles(args.obstacles_json)
    rospy.loginfo(f"[Monitor] Loaded {len(obstacles)} obstacles from {args.obstacles_json}")

    # Wait for drone
    if not wait_for_drone_appear(drone_name=args.drone_name):
        rospy.logerr("[Monitor] Drone not found in model_states. Is Gazebo running?")
        sys.exit(1)
    rospy.loginfo("[Monitor] Drone confirmed in Gazebo model_states")

    # Create monitor
    monitor = CollisionTimeoutMonitor(
        obstacles=obstacles,
        goal=(args.goal_x, args.goal_y, args.goal_z),
        drone_collision_radius=args.drone_radius,
        timeout_s=args.timeout,
        debounce_frames=args.debounce_frames,
        bounds_xy=args.bounds_xy,
        bounds_z_max=args.bounds_z_max,
    )

    # Set metadata
    monitor.result.world_id = args.world_id
    monitor.result.trial = args.trial
    monitor.result.tier = args.tier
    monitor.result.filename = args.filename

    # Run
    result = None
    try:
        result = monitor.run()
    except Exception as e:
        rospy.logerr(f"[Monitor] Fatal error: {e}")
        result = monitor.result  # save whatever we have
    finally:
        monitor.shutdown()

    # Output — always, even on Ctrl+C
    result_dict = result.to_dict() if result else monitor.result.to_dict()
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        # Write trajectory CSVs first
        monitor.write_trajectory_csvs(args.output)
        with open(args.output, "w") as f:
            json.dump(result_dict, f, indent=2)
        print(f"Result saved to {args.output}")
        print(f"  success_deploy={result_dict.get('success_deploy')}, "
              f"collision={result_dict.get('collision')}, "
              f"timeout={result_dict.get('timeout')}, "
              f"final_dxy={result_dict.get('final_dist_xy', 0):.2f}m, "
              f"min_clearance={result_dict.get('min_clearance', 0):.2f}m")
    else:
        print(json.dumps(result_dict, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())