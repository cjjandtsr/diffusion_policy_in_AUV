"""
gazebo_uav_env.py — RL 采样 gym 环境 (rospy 直连版, Route B v2)

RL 节点 = idp3 conda 里的 ROS 节点: sub odom → 合成点云+FM → pub /planning/pos_cmd
无 ZMQ, 无桥接。与 planner 同环境。

关键: gym step 同步语义 ↔ ROS 异步回调 — 后台 spin 线程 + 主线程时间轮询.
"""

import os, sys, time, math, subprocess, threading, collections, hashlib, json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
from typing import Dict, Tuple, Optional

import rospy
from nav_msgs.msg import Odometry
from mavros_msgs.msg import State as MavrosState
from quadrotor_msgs.msg import PositionCommand

from .traj_utils import (
    action_to_world, quat_to_yaw, TrajectoryManager, DPOS_W, VEL_W, ACC_W, YAW, YAW_RT)
from .collision_geom import load_obstacles, compute_obstacle_clearance


# ═══════════════════════════════════════════════════════════════
# 路径
# ═══════════════════════════════════════════════════════════════
AUV_ROOT = Path(os.path.expanduser("~/AUV"))
CATKIN_WS = AUV_ROOT / "catkin_ws0"
PX4_DIR = AUV_ROOT / "PX4-Autopilot"
WORLDS_DIR = PX4_DIR / "Tools/sitl_gazebo/worlds/ood_experiment_worlds/layerA"
TAKEOFF_SCRIPT = CATKIN_WS / "scripts/takeoff_only.sh"
STOP_SITL_SCRIPT = CATKIN_WS / "scripts/stop_sitl.sh"
START_SITL_SCRIPT = CATKIN_WS / "scripts/start_sitl_rl.sh"
DEVNULL = subprocess.DEVNULL

SPAWN_OFFSET = np.array([-8.0, 0.0, 0.0], dtype=np.float32)
DEFAULT_GOAL = np.array([12.0, 0.0, 2.0], dtype=np.float32)

WORLD_ENV_MAP = {4: "env_004_obs19", 8: "env_008_obs20",
                 11: "env_011_obs23", 12: "env_012_obs22", 18: "env_018_obs25"}


# ═══════════════════════════════════════════════════════════════
# Gym 环境 (rospy 节点)
# ═══════════════════════════════════════════════════════════════

class GazeboUAVEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, world_id: int = 4, goal_world: np.ndarray = None,
                 synth_pc=None, rl_action_steps: int = 8,
                 max_episode_steps: int = 60, z_min: float = 1.0, z_max: float = 5.0,
                 goal_tol_xy: float = 0.5, goal_tol_z: float = 0.3,
                 gamma: float = 0.99, lambda_c: float = 0.1,
                 d_safe: float = 0.75, beta_avoid: float = 20.0,
                 pos_gain=(3.5, 3.5, 3.5), vel_gain=(3.5, 3.5, 3.5)):

        super().__init__()
        self.world_id = world_id
        self.obstacles_json = str(WORLDS_DIR / f"{WORLD_ENV_MAP[world_id]}.obstacles.json")
        # 加载障碍几何（碰撞检测用，与 synth_pc 同源 json，纯几何无 ROS）
        self.obstacles = load_obstacles(self.obstacles_json)
        self.goal_w = (goal_world if goal_world is not None else DEFAULT_GOAL.copy()).astype(np.float32)
        self.rl_action_steps = rl_action_steps
        self.max_episode_steps = max_episode_steps
        self.z_min, self.z_max = z_min, z_max
        self.goal_tol_xy, self.goal_tol_z = goal_tol_xy, goal_tol_z
        self.traj_dt = 0.3
        self.pos_gain = np.array(pos_gain)
        self.vel_gain = np.array(vel_gain)
        self.synth_pc = synth_pc
        self.n_obs_steps = 8

        # reward P1
        self.gamma, self.lambda_c = gamma, lambda_c
        self.d_safe, self.beta_avoid = d_safe, beta_avoid

        # obs/act space
        self.observation_space = spaces.Dict({
            'point_cloud': spaces.Box(-50, 50, (self.n_obs_steps, 4096, 3), np.float32),
            'agent_pos': spaces.Box(-np.inf, np.inf, (self.n_obs_steps, 9), np.float32)})
        self.action_space = spaces.Box(-1, 1, (rl_action_steps, 11), np.float32)

        # ── ROS (后台 spin 线程) ──
        if not rospy.core.is_initialized():
            rospy.init_node('rl_sampler', anonymous=True, disable_signals=True)
        self._lock = threading.Lock()
        self._latest_odom: Optional[Odometry] = None
        self._latest_state: Optional[MavrosState] = None
        rospy.Subscriber('/mavros/local_position/odom', Odometry,
                         self._odom_cb, queue_size=1)
        rospy.Subscriber('/mavros/state', MavrosState,
                         self._state_cb, queue_size=1)
        self._cmd_pub = rospy.Publisher('/planning/pos_cmd', PositionCommand, queue_size=10)
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()
        rospy.sleep(0.5)  # 等 subscriber 连上

        # ── 轨迹插值 ──
        self._traj_mgr = TrajectoryManager(self.traj_dt)

        # ── 内部状态 ──
        self.episode_step = 0
        self._first_reset = True    # 首次 reset 复用外层 SITL，后续重启
        self._drone_pos = np.zeros(3, np.float32)
        self._drone_quat = np.array([0., 0., 0., 1.], np.float32)
        self._drone_vel = np.zeros(3, np.float32)
        self._drone_yaw = 0.0
        self._pc_history = np.zeros((self.n_obs_steps, 4096, 3), np.float32)
        self._state_history = np.zeros((self.n_obs_steps, 9), np.float32)
        self._prev_phi = 0.0
        self._cmd_count = 0

    # ── ROS callbacks (后台线程, 只存最新) ──
    def _odom_cb(self, msg):
        with self._lock: self._latest_odom = msg
    def _state_cb(self, msg):
        with self._lock: self._latest_state = msg

    @staticmethod
    def _spin():
        rospy.spin()

    # ── 读状态 (主线程, 线程安全) ──
    def _get_drone_state(self):
        with self._lock:
            odom = self._latest_odom
        if odom is None:
            return None
        p, q, v = odom.pose.pose.position, odom.pose.pose.orientation, odom.twist.twist.linear
        self._drone_pos = np.array([p.x, p.y, p.z], np.float32)
        self._drone_quat = np.array([q.x, q.y, q.z, q.w], np.float32)
        self._drone_vel = np.array([v.x, v.y, v.z], np.float32)
        self._drone_yaw = quat_to_yaw(self._drone_quat)
        return True

    # ── 发指令 ──
    def _publish_cmd(self, pos, vel, acc, yaw, yr):
        cmd = PositionCommand()
        cmd.header.stamp = rospy.Time.now()
        cmd.header.frame_id = "world"
        cmd.position.x, cmd.position.y, cmd.position.z = float(pos[0]), float(pos[1]), float(pos[2])
        cmd.velocity.x, cmd.velocity.y, cmd.velocity.z = float(vel[0]), float(vel[1]), float(vel[2])
        cmd.acceleration.x, cmd.acceleration.y, cmd.acceleration.z = float(acc[0]), float(acc[1]), float(acc[2])
        cmd.yaw, cmd.yaw_dot = float(yaw), float(yr)
        cmd.trajectory_flag = PositionCommand.TRAJECTORY_STATUS_READY
        cmd.trajectory_id = self._cmd_count % 100 + 1
        cmd.kx = self.pos_gain; cmd.kv = self.vel_gain
        self._cmd_pub.publish(cmd)
        self._cmd_count += 1

    # ── 下发轨迹 + 同步等执行 + 稳定等待 ──
    def _execute_trajectory(self, world_wp: np.ndarray, n_steps: int):
        """50Hz 插值下发, 轨迹发完后轮询等位置稳定再返回 (消除采样太早)"""
        self._get_drone_state()
        cur_pos = self._drone_pos.copy()
        self._traj_mgr.set_trajectory(world_wp, self.traj_dt, cur_pos)
        rate = rospy.Rate(50)
        duration = (n_steps - 1) * self.traj_dt
        t_end = time.time() + duration
        # 阶段 1: 插值下发完整轨迹
        while time.time() < t_end and not rospy.is_shutdown():
            r = self._traj_mgr.sample(time.time())
            if r is None: break
            self._publish_cmd(*r[:5])
            rate.sleep()
        # 读最后 waypoint
        wp = self._traj_mgr.waypoints
        if wp is None:
            return
        last = wp[min(n_steps - 1, len(wp) - 1)]
        hold_pos = last[DPOS_W].copy()
        hold_yaw = float(last[YAW])
        # 阶段 2: 持续发 hold + 滑动窗口等位置稳定
        settle_timeout = 3.0           # 最多等 3s
        settle_thresh = 0.05           # 窗口内净位移 < 5cm 算稳定
        settle_window = 25             # 窗口大小 (25帧≈0.5s@50Hz)
        t_settle = time.time()
        pos_ring = collections.deque(maxlen=settle_window)
        pos_ring.append(self._drone_pos.copy())
        while time.time() - t_settle < settle_timeout and not rospy.is_shutdown():
            self._publish_cmd(hold_pos, np.zeros(3), np.zeros(3),
                              hold_yaw, 0.0)
            self._get_drone_state()
            pos_ring.append(self._drone_pos.copy())
            if len(pos_ring) == settle_window:
                net_disp = np.linalg.norm(pos_ring[-1] - pos_ring[0])
                if net_disp < settle_thresh:
                    break
            rate.sleep()

    # ═══════════════════════════════════════════════════════════
    # Gym 接口
    # ═══════════════════════════════════════════════════════════

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_step = 0

        if self._first_reset:
            # ★首次：外层 run_dsrl_train.sh 已起好 SITL + 起飞，直接复用
            self._first_reset = False
            rospy.loginfo("[reset] 首次，复用外层 SITL")
        else:
            # ★后续：完全重启 SITL（干净实例）。只杀 Gazebo+PX4+px4ctrl，留 roscore+mavros
            rospy.loginfo("[reset] 重启 SITL...")
            subprocess.run(['pkill', '-9', '-f', 'px4_sitl_default/bin/px4'],
                           stdout=DEVNULL, stderr=DEVNULL)
            subprocess.run(['pkill', '-9', '-f', 'gzserver'],
                           stdout=DEVNULL, stderr=DEVNULL)
            subprocess.run(['pkill', '-9', '-f', 'gzclient'],
                           stdout=DEVNULL, stderr=DEVNULL)
            subprocess.run(['pkill', '-9', '-f', 'px4ctrl'],
                           stdout=DEVNULL, stderr=DEVNULL)
            time.sleep(5)
            # 清除旧 odom/state/pos 残留（避免 _wait_mavros_online 误判旧位置为"已上线"）
            with self._lock:
                self._latest_odom = None
                self._latest_state = None
            self._drone_pos = np.zeros(3, np.float32)
            world_file = str(WORLDS_DIR / f"{WORLD_ENV_MAP[self.world_id]}.world")
            if not START_SITL_SCRIPT.exists():
                rospy.logerr(f"[reset] start_sitl_rl.sh 不存在: {START_SITL_SCRIPT}")
            if not Path(world_file).exists():
                rospy.logerr(f"[reset] world 文件不存在: {world_file}")
            subprocess.Popen(['bash', str(START_SITL_SCRIPT), world_file],
                             stdout=DEVNULL, stderr=DEVNULL, start_new_session=True)
            self._wait_mavros_online(timeout=90)
            env = os.environ.copy()
            env['GOAL_X'] = str(self.goal_w[0])
            env['GOAL_Y'] = str(self.goal_w[1])
            env['GOAL_Z'] = str(self.goal_w[2])
            proc = subprocess.Popen(['bash', str(TAKEOFF_SCRIPT)], env=env,
                                    stdout=DEVNULL, stderr=DEVNULL)
            proc.wait(timeout=120)

        # ── 共通：等就绪 + 读状态 + 填历史 ──
        self._wait_until_ready(timeout=60)
        self._get_drone_state()
        # P0-2: Episode-local RNG seed for reproducible point cloud
        # 使用 hashlib.blake2b（跨进程稳定的 hash），替代 Python hash()（进程级随机盐）
        _payload_data = {
            "world_id": int(self.world_id),
            "goal_w": [round(float(v), 4) for v in self.goal_w],
            "drone_pos": [round(float(v), 4) for v in self._drone_pos],
        }
        if seed is not None:
            _payload_data["seed"] = int(seed)
        _payload = json.dumps(_payload_data, sort_keys=True).encode("utf-8")
        _digest = hashlib.blake2b(_payload, digest_size=4).digest()
        self._episode_rng_seed = int.from_bytes(_digest, "little")
        z = self._drone_pos[2]
        if z > 4.0 or z < 0.5:
            rospy.logwarn(f"[reset] 高度异常 z={z:.2f} (期望≈2)")
        rospy.loginfo(f"[reset] 就绪 z={z:.2f} pos=[{self._drone_pos[0]:.1f},"
                      f"{self._drone_pos[1]:.1f},{self._drone_pos[2]:.1f}]")

        # ── 填充 8 帧历史 + 初始化 Φ ──
        obs = self._get_obs_frame()
        for _ in range(self.n_obs_steps):
            self._push_obs(obs)
        self._prev_phi = self._compute_phi()
        return self._get_obs_dict(), {}

    def _wait_mavros_online(self, timeout=90):
        """轮询等 MAVROS odom 有数据且为新实例（SITL 重启后）"""
        t0 = time.time()
        while time.time() - t0 < timeout and not rospy.is_shutdown():
            self._get_drone_state()
            z = self._drone_pos[2] if self._drone_pos is not None else 0
            # 新 SITL 实例的无人机在地面（z<0.5），等起飞后 odom 才到 ~2m
            if self._drone_pos is not None and np.any(self._drone_pos != 0):
                return True
            time.sleep(0.5)
        rospy.logwarn("[reset] MAVROS 上线超时")
        return False

    def _wait_until_ready(self, timeout=60):
        """等 px4ctrl TRIGGER + 高度稳定（不要求特定 mode，PX4 不一定报 OFFBOARD）"""
        t0 = time.time()
        while time.time() - t0 < timeout and not rospy.is_shutdown():
            with self._lock:
                st = self._latest_state
            if st is not None and st.armed:
                self._get_drone_state()
                z = self._drone_pos[2]
                if 0.5 < z < 4.0:
                    time.sleep(1.0)
                    return True
            time.sleep(0.5)
        rospy.logwarn("[reset] ready 等待超时")
        return False

    def step(self, action: np.ndarray):
        phi_prev = self._prev_phi
        # 1) action delta → world waypoints → 下发执行
        self._get_drone_state()
        world_wp = action_to_world(action, self._drone_pos, self._drone_vel, self._drone_yaw)
        self._execute_trajectory(world_wp, self.rl_action_steps)
        # 2) 读新状态 + 合成点云
        self._get_drone_state()
        obs_frame = self._get_obs_frame()
        self._push_obs(obs_frame)
        obs_dict = self._get_obs_dict()
        # 3) reward (P1修正)
        reward, terminated = self._compute_reward(phi_prev)
        self.episode_step += 1
        truncated = self.episode_step >= self.max_episode_steps
        early = False
        if not terminated:
            early = self._check_early_death()
            terminated = early
        # ★诊断: done 原因日志
        if terminated or truncated:
            dist = float(np.linalg.norm(self._drone_pos - self.goal_w))
            clr = self._estimate_clearance()
            dpos_z = action[:, 2] if hasattr(action, 'shape') and action.ndim >= 2 else 0
            dpos_z_sum = float(np.sum(dpos_z)) if not isinstance(dpos_z, (int, float)) else dpos_z
            rospy.loginfo(
                f"[DONE] step={self.episode_step}/{self.max_episode_steps} "
                f"pos=[{self._drone_pos[0]:.1f},{self._drone_pos[1]:.1f},{self._drone_pos[2]:.1f}] "
                f"dist={dist:.1f} clr={clr:.3f} dpos_z_sum={dpos_z_sum:.2f} "
                f"terminated={terminated} truncated={truncated} early={early} "
                f"reward={reward:.2f}")
        return obs_dict, reward, terminated, truncated, {
            'step': self.episode_step, 'pos': self._drone_pos.copy()}

    # ═══════════════════════════════════════════════════════════
    # 观测
    # ═══════════════════════════════════════════════════════════

    def _get_obs_frame(self):
        pc = self.synth_pc.synthesize(self._drone_pos + SPAWN_OFFSET,
                                       self._drone_yaw, num_points=4096, add_noise=True,
                                       rng_seed=self._episode_rng_seed)
        s9, _ = _build_state_v65(self._drone_pos, self._drone_quat, self.goal_w)
        return {'point_cloud': pc.astype(np.float32), 'agent_pos': s9.astype(np.float32)}

    def _push_obs(self, obs):
        self._pc_history = np.roll(self._pc_history, -1, axis=0); self._pc_history[-1] = obs['point_cloud']
        self._state_history = np.roll(self._state_history, -1, axis=0); self._state_history[-1] = obs['agent_pos']

    def _get_obs_dict(self):
        return {'point_cloud': self._pc_history.copy(), 'agent_pos': self._state_history.copy()}

    # ═══════════════════════════════════════════════════════════
    # Reward P1 修正
    # ═══════════════════════════════════════════════════════════

    def _compute_phi(self):
        dist = float(np.linalg.norm(self._drone_pos - self.goal_w))
        min_clr = self._estimate_clearance()
        w_avoid = 1.0 / (1.0 + math.exp(5.0 * (min_clr - self.d_safe)))
        g = dist / 1.0 if dist < 1.0 else 1.0
        gdir = self.goal_w[:2] - self._drone_pos[:2]
        gnorm = np.linalg.norm(gdir) + 1e-6
        dh = np.array([np.cos(self._drone_yaw), np.sin(self._drone_yaw)])
        e_yaw = 1.0 - np.dot(dh, gdir / gnorm)
        z_target = (self.z_min + self.z_max) / 2.0
        return -(1 - w_avoid) * dist - (1 - w_avoid) * g * e_yaw - abs(self._drone_pos[2] - z_target)

    def _compute_reward(self, phi_prev):
        phi_next = self._compute_phi()
        self._prev_phi = phi_next
        dist = float(np.linalg.norm(self._drone_pos - self.goal_w))
        reached = (dist < self.goal_tol_xy and abs(self._drone_pos[2] - self.goal_w[2]) < self.goal_tol_z)
        collision = self._check_collision()
        terminated = reached or collision
        if terminated:
            r = -phi_prev + (10.0 if reached else -5.0)
        else:
            r = self.gamma * phi_next - phi_prev
            clr = self._estimate_clearance()
            margin = self.d_safe - clr
            if margin > 0:
                x = self.beta_avoid * margin
                r -= self.lambda_c * (math.log(1 + math.exp(min(x, 20))) / self.beta_avoid)
        return float(r), terminated

    def _estimate_clearance(self):
        """几何 surface-to-surface clearance（复用实验 collision_timeout_monitor.py）"""
        min_clr, _, _ = compute_obstacle_clearance(
            float(self._drone_pos[0]), float(self._drone_pos[1]),
            float(self._drone_pos[2]), self.obstacles, check_3d=True)
        return min_clr   # 表面距离，负=穿透；飞越时返回正值（豁免）

    def _check_collision(self):
        """纯几何碰撞：表面穿透 < 0 = 真撞 + 坠地判据（删 synth_pc + z骤降）"""
        min_clr = self._estimate_clearance()
        if min_clr < 0.0:
            rospy.loginfo(f"[COLLISION] 几何穿透 clr={min_clr:.3f}")
            return True
        if self._drone_pos[2] < 0.2:
            rospy.loginfo(f"[COLLISION] 坠地 z={self._drone_pos[2]:.2f} < 0.2")
            return True
        return False

    def _check_early_death(self):
        p = self._drone_pos
        if abs(p[0]) > 30 or abs(p[1]) > 30:
            rospy.loginfo(f"[EARLY] 出界 x={p[0]:.1f} y={p[1]:.1f}")
            return True
        if p[2] > self.z_max + 1.0:
            rospy.loginfo(f"[EARLY] 高度超限 z={p[2]:.1f} > {self.z_max+1.0}")
            return True
        if self._check_collision():
            return True
        return False

    def close(self):
        pass


# ═══════════════════════════════════════════════════════════════
# 纯函数 (抽自 planner)
# ═══════════════════════════════════════════════════════════════

def _build_state_v65(pos_w, quat_xyzw, goal_w):
    yaw = quat_to_yaw(quat_xyzw)
    dx, dy, dz = goal_w[0]-pos_w[0], goal_w[1]-pos_w[1], goal_w[2]-pos_w[2]
    c, s = np.cos(yaw), np.sin(yaw)
    gbx, gby, gbz = c*dx + s*dy, -s*dx + c*dy, dz
    gd = float(np.sqrt(dx*dx + dy*dy + dz*dz))
    s9 = np.array([pos_w[0], pos_w[1], pos_w[2], gbx, gby, gbz, gd, np.sin(yaw), np.cos(yaw)], np.float32)
    return s9, np.array([gbx, gby, gbz], np.float32)
