#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML Planner v5 — world-frame chunk-wise delta 扩散模型无人机轨迹规划

v4 → v5 核心变更:
  action: body step-wise delta → world chunk-wise delta
    - _body_to_world 前向积分 → _action_to_world 直接加法 (无循环依赖)
  state: 14维 → 15维
    - vel_body → vel_world
    - acc_body → acc_world
    - yaw → sin(yaw) + cos(yaw)

架构: 并行推理流水线 → _action_to_world → TrajectoryManager(50Hz插值) → PX4Ctrl
"""

import os, sys, time, traceback, types
import numpy as np
import rospy
from collections import deque
from threading import Lock, Thread
import csv
from datetime import datetime

from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from quadrotor_msgs.msg import PositionCommand
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import sensor_msgs.point_cloud2 as pc2
from mavros_msgs.msg import State as MavrosState
from mavros_msgs.srv import SetMode

import torch, hydra
import torch._dynamo
torch._dynamo.config.suppress_errors = True

DIFFUSION_POLICY_PATH = os.path.expanduser(
    "~/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy")
if os.path.exists(DIFFUSION_POLICY_PATH):
    sys.path.insert(0, DIFFUSION_POLICY_PATH)
else:
    print("[ML Planner] 致命: 找不到扩散模型路径"); sys.exit(1)

os.environ['WANDB_SILENT'] = "True"
os.environ['WANDB_MODE']   = "disabled"

try:
    from diffusion_policy_3d.workspace.base_workspace import BaseWorkspace
except ImportError:
    traceback.print_exc(); sys.exit(1)


# ═══ v5 索引常量 ═══
# raw action (模型输出): world chunk-wise delta
DPOS_W   = slice(0, 3)   # chunk-wise delta position (world)
VEL_W    = slice(3, 6)   # absolute velocity (world)
ACC_W    = slice(6, 9)   # absolute acceleration (world)
DYAW_CW  = 9             # chunk-wise delta yaw
YAW_RATE = 10            # absolute yaw rate

# world action (转换后)
POS    = slice(0, 3)
VEL    = slice(3, 6)
ACC    = slice(6, 9)
YAW    = 9
YAW_RT = 10

STATE_DIM = 15  # v5: 15维 state

DIM_NAMES_RAW   = ["dx_w","dy_w","dz_w","vx_w","vy_w","vz_w",
                   "ax_w","ay_w","az_w","dyaw_cw","yr"]
DIM_NAMES_WORLD = ["pos_x","pos_y","pos_z","vel_x","vel_y","vel_z",
                   "acc_x","acc_y","acc_z","yaw","yr"]

R_CAM2BODY = np.array([[ 0,0,1],[-1,0,0],[0,-1,0]], dtype=np.float64)


def yaw_to_rotmat(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float64)

def normalize_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi


# ---------- 推理加速补丁（不变） ----------
def _make_fast_conditional_sample(policy, n_steps, device):
    scheduler = policy.noise_scheduler
    scheduler.set_timesteps(n_steps)
    _cached_timesteps = scheduler.timesteps.to(device)

    def fast_conditional_sample(
        self_policy,
        condition_data, condition_mask,
        local_cond=None, global_cond=None,
        generator=None,
        **kwargs
    ):
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        trajectory = torch.where(
            condition_mask,
            condition_data.to(trajectory.dtype),
            trajectory)

        for t in _cached_timesteps:
            model_output = self_policy.model(
                trajectory, t, local_cond=local_cond, global_cond=global_cond)
            trajectory = scheduler.step(
                model_output, t, trajectory,
                **{k: v for k, v in kwargs.items()
                   if k in ('eta', 'use_clipped_model_output', 'variance_noise')}
            ).prev_sample
            trajectory = torch.where(
                condition_mask,
                condition_data.to(trajectory.dtype),
                trajectory)

        return trajectory

    return types.MethodType(fast_conditional_sample, policy)


def _try_compile(module, label):
    backends = [
        ("cudagraphs", dict()),
        ("eager",      dict()),
    ]
    for name, kwargs in backends:
        try:
            compiled = torch.compile(module, backend=name, **kwargs)
            rospy.loginfo(f"[Speed] {label} torch.compile(backend={name}) 已启用")
            return compiled
        except Exception as e:
            rospy.logwarn(f"[Speed] {label} backend={name} 失败: {e}")
    rospy.logwarn(f"[Speed] {label} 所有 compile 后端均失败，回退 eager 模式")
    return module


def apply_inference_patches(policy, n_steps, device, use_compile=False):
    policy.conditional_sample = _make_fast_conditional_sample(policy, n_steps, device)
    rospy.loginfo(f"[Speed] conditional_sample 已 patch（{n_steps}步 DDIM，预缓存 timesteps）")

    if use_compile and device.type == 'cuda':
        policy.model = _try_compile(policy.model, "UNet")
        if hasattr(policy, 'obs_encoder'):
            policy.obs_encoder = _try_compile(policy.obs_encoder, "obs_encoder")

    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
        rospy.loginfo("[Speed] TF32 matmul + cudnn.benchmark 已启用")

    return policy


def warmup_policy(policy, n_obs_steps, num_points, device, n_warmup=3):
    rospy.loginfo(f"[Speed] 开始热身推理（{n_warmup}次）...")
    dummy_pc  = torch.randn(1, n_obs_steps, num_points, 3, device=device)
    dummy_pos = torch.randn(1, n_obs_steps, STATE_DIM,     device=device)  # v5: 15维
    obs = {'point_cloud': dummy_pc, 'agent_pos': dummy_pos}

    t_warmup = []
    with torch.inference_mode():
        for i in range(n_warmup):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.time()
            out = policy.predict_action(obs)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            dt = (time.time() - t0) * 1000
            t_warmup.append(dt)
            rospy.loginfo(f"[Speed] 热身第{i+1}次: {dt:.0f}ms  output={out['action'].shape}")

    rospy.loginfo(f"[Speed] 热身完成，稳定延迟: {t_warmup[-1]:.0f}ms")
    return out['action'].shape


# CSV 日志类
class CSVLogger:
    def __init__(self):
        log_dir = os.path.expanduser("~/ml_planner_log")
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.diff_file = open(os.path.join(log_dir, f"diffusion_{ts}.csv"), 'w', newline='')
        self.diff_w    = csv.writer(self.diff_file)
        self.diff_w.writerow(
            ["plan_id","step","ros_time","inference_ms",
             "cur_x","cur_y","cur_z","cur_yaw",
             "goal_x","goal_y","goal_z","goal_bx","goal_by","goal_bz"]
            + [f"raw_{n}"   for n in DIM_NAMES_RAW]
            + [f"world_{n}" for n in DIM_NAMES_WORLD])

        self.cmd_file = open(os.path.join(log_dir, f"pos_cmd_{ts}.csv"), 'w', newline='')
        self.cmd_w    = csv.writer(self.cmd_file)
        self.cmd_w.writerow(
            ["cmd_id","ros_time","source",
             "cmd_px","cmd_py","cmd_pz","cmd_vx","cmd_vy","cmd_vz",
             "cmd_ax","cmd_ay","cmd_az","cmd_yaw","cmd_yr",
             "cur_px","cur_py","cur_pz","err_m"])

        self.plan_count = 0
        rospy.loginfo(f"[CSV] {self.diff_file.name}")
        rospy.loginfo(f"[CSV] {self.cmd_file.name}")

    def log_diffusion(self, raw_action, world_action, inference_ms,
                      cur_pos, cur_yaw, goal_w, goal_b):
        self.plan_count += 1
        t = rospy.Time.now().to_sec()
        for s in range(len(raw_action)):
            self.diff_w.writerow(
                [self.plan_count, s, f"{t:.3f}", f"{inference_ms:.0f}"]
                + [f"{v:.4f}" for v in cur_pos] + [f"{cur_yaw:.4f}"]
                + [f"{v:.4f}" for v in goal_w]  + [f"{v:.4f}" for v in goal_b]
                + [f"{raw_action[s,d]:.6f}"   for d in range(raw_action.shape[1])]
                + [f"{world_action[s,d]:.6f}" for d in range(world_action.shape[1])])
        self.diff_file.flush()

    def log_cmd(self, cmd_id, source, pos, vel, acc, yaw, yr, cur_pos):
        err = np.linalg.norm(np.array(pos[:3]) - cur_pos)
        self.cmd_w.writerow(
            [cmd_id, f"{rospy.Time.now().to_sec():.3f}", source]
            + [f"{v:.4f}" for v in pos[:3]]
            + [f"{v:.4f}" for v in vel[:3]]
            + [f"{v:.4f}" for v in acc[:3]]
            + [f"{yaw:.4f}", f"{yr:.4f}"]
            + [f"{v:.4f}" for v in cur_pos]
            + [f"{err:.4f}"])
        self.cmd_file.flush()

    def close(self):
        self.diff_file.close()
        self.cmd_file.close()


# 轨迹管理器（不变）
class TrajectoryManager:
    MAX_JUMP_M = 0.5

    def __init__(self):
        self.lock       = Lock()
        self.waypoints  = None
        self.start_time = None
        self.traj_dt    = 0.3
        self.duration   = 0.0

    def set_trajectory(self, world_action, traj_dt, cur_pos=None):
        with self.lock:
            traj = world_action.copy()
            if cur_pos is not None:
                delta = cur_pos - traj[0, POS]
                if np.linalg.norm(delta) > self.MAX_JUMP_M:
                    rospy.logwarn(f"[TrajMgr] 起点偏差{np.linalg.norm(delta):.2f}m，整体平移对齐")
                    traj[:, POS] += delta
            self.waypoints  = traj
            self.start_time = rospy.Time.now()
            self.traj_dt    = traj_dt
            self.duration   = (len(traj) - 1) * traj_dt

    def sample(self, now):
        with self.lock:
            if self.waypoints is None:
                return None
            t        = max(0.0, (now - self.start_time).to_sec())
            progress = t / self.duration if self.duration > 0 else 1.0
            t_idx    = t / self.traj_dt
            i, frac  = int(t_idx), t_idx - int(t_idx)
            N        = len(self.waypoints)
            if i >= N - 1:
                wp = self.waypoints[-1]
                return wp[POS].copy(), np.zeros(3), np.zeros(3), float(wp[YAW]), 0.0, progress
            wp0, wp1 = self.waypoints[i], self.waypoints[i+1]
            return (wp0[POS]*(1-frac) + wp1[POS]*frac,
                    wp0[VEL]*(1-frac) + wp1[VEL]*frac,
                    wp0[ACC]*(1-frac) + wp1[ACC]*frac,
                    float(wp0[YAW]*(1-frac) + wp1[YAW]*frac),
                    float(wp0[YAW_RT]*(1-frac) + wp1[YAW_RT]*frac),
                    progress)

    def clear(self):
        with self.lock:
            self.waypoints  = None
            self.start_time = None

    def is_valid(self):
        with self.lock:
            return self.waypoints is not None


# 主节点
class MLPlannerNode:

    def __init__(self):
        rospy.init_node('ml_planner_node', anonymous=False)
        rospy.loginfo("[ML Planner v5] 初始化 (world chunk-wise delta)")

        self.load_parameters()

        self.data_lock                = Lock()
        self.current_position         = np.array([0.0, 0.0, 2.0])
        self.current_velocity_world   = np.zeros(3)   # v5: 始终维护 world 速度
        self.prev_velocity_world      = np.zeros(3)   # v5: 上一帧 world 速度(用于acc)
        self.current_orientation_xyzw = np.array([0.0, 0.0, 0.0, 1.0])
        self.current_yaw              = 0.0
        self.current_pointcloud       = None
        self.goal_position            = None
        self.has_goal                 = False
        self.waypoint_index           = 0
        self.state                    = "INIT"
        self.cmd_count                = 0

        self.traj_mgr       = TrajectoryManager()
        self.planning_lock  = Lock()
        self.is_planning    = False
        self._replan_logged = False

        self.n_obs_steps    = None
        self.action_horizon = None
        self.num_points     = 4096
        self.obs_history    = None

        self._pc_pinned  = None
        self._pos_pinned = None

        self.csv_logger      = CSVLogger()
        self.set_mode_client = rospy.ServiceProxy('/mavros/set_mode', SetMode)

        self.policy = None
        self.device = None
        self.load_diffusion_policy()
        self.setup_subscribers()
        self.setup_publishers()

        rospy.Timer(rospy.Duration(1.0 / self.control_freq), self.control_loop)
        rospy.Timer(rospy.Duration(0.2), self._vis_tick)
        rospy.Timer(rospy.Duration(0.3), self._sample_obs)

        rospy.on_shutdown(self.csv_logger.close)
        rospy.loginfo("[ML Planner v5] 就绪")

    # 参数加载
    def load_parameters(self):
        self.control_freq    = rospy.get_param("~control_freq",    50.0)
        self.replan_ratio    = rospy.get_param("~replan_ratio",     0.7)
        self.goal_tolerance  = rospy.get_param("~goal_tolerance",   0.5)
        self.default_height  = rospy.get_param("~default_height",   1.5)
        self.flight_mode     = rospy.get_param("~flight_mode",        0)
        self.traj_dt         = rospy.get_param("~traj_dt",          0.3)
        self.drone_id        = rospy.get_param("~drone_id",           0)
        self.pos_gain        = self._parse_gain("~pos_gain", [3.5, 3.5, 3.5])
        self.vel_gain        = self._parse_gain("~vel_gain", [3.5, 3.5, 3.5])
        self.checkpoint_path = rospy.get_param("~checkpoint_path",   "")
        self.num_inference_steps_deploy = rospy.get_param("~num_inference_steps", 5)
        self.use_fp16          = rospy.get_param("~use_fp16",          False)
        self.use_torch_compile = rospy.get_param("~use_torch_compile", True)
        # v5: 始终需要 body→world 转换, 但仅用于 state 构建
        self.odom_vel_is_world = rospy.get_param("~odom_vel_is_world",  False)

        self.waypoints = []
        for i in range(rospy.get_param("~waypoint_num", 0)):
            x = rospy.get_param(f"~waypoint{i}_x", 0.0)
            y = rospy.get_param(f"~waypoint{i}_y", self.default_height)
            z = rospy.get_param(f"~waypoint{i}_z", self.default_height)
            self.waypoints.append(np.array([x, y, z]))
            rospy.loginfo(f"  航点{i}: [{x:.1f},{y:.1f},{z:.1f}]")

    def _parse_gain(self, name, default):
        val = rospy.get_param(name, default)
        if isinstance(val, str):
            return [float(x.strip()) for x in val.strip('[]').split(',')]
        return [float(x) for x in val] if isinstance(val, list) else default

    # 模型加载
    def load_diffusion_policy(self):
        rospy.loginfo("[ML Planner v5] 加载扩散模型...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not os.path.exists(self.checkpoint_path):
            rospy.logerr(f"Checkpoint不存在: {self.checkpoint_path}"); sys.exit(1)
        try:
            payload   = torch.load(self.checkpoint_path, map_location='cpu')
            cfg       = payload['cfg']
            if hasattr(cfg, 'n_obs_steps'):
                self.n_obs_steps = cfg.n_obs_steps
                rospy.loginfo(f"[ML Planner] 从checkpoint读取 n_obs_steps = {self.n_obs_steps}")
            else:
                self.n_obs_steps = 3
            if hasattr(cfg, 'n_action_steps'):
                self.action_horizon = cfg.n_action_steps
                rospy.loginfo(f"[ML Planner] 从checkpoint读取 action_horizon = {self.action_horizon}")
            elif hasattr(cfg, 'horizon') and hasattr(cfg, 'n_obs_steps'):
                self.action_horizon = cfg.horizon - cfg.n_obs_steps + 1
            else:
                self.action_horizon = 7

            self.obs_history = deque(maxlen=self.n_obs_steps)

            cls       = hydra.utils.get_class(cfg._target_)
            workspace = cls(cfg)
            workspace.load_payload(payload, exclude_keys=None, include_keys=None)
            self.policy = workspace.model
            if hasattr(workspace, 'ema_model') and workspace.ema_model is not None:
                self.policy = workspace.ema_model
            self.policy = self.policy.to(self.device).eval()

            if self.use_fp16 and self.device.type != 'cuda':
                rospy.logwarn("[ML Planner] use_fp16 需要CUDA，已忽略")
                self.use_fp16 = False

            self.policy = apply_inference_patches(
                self.policy,
                n_steps      = self.num_inference_steps_deploy,
                device       = self.device,
                use_compile  = self.use_torch_compile)

            if self.device.type == 'cuda':
                self._pc_pinned  = torch.empty(
                    1, self.n_obs_steps, self.num_points, 3,
                    dtype=torch.float32, pin_memory=True)
                self._pos_pinned = torch.empty(
                    1, self.n_obs_steps, STATE_DIM,    # v5: 15维
                    dtype=torch.float32, pin_memory=True)
                rospy.loginfo("[Speed] pinned memory 缓冲区已分配 (state_dim=15)")

            warmup_policy(
                self.policy,
                n_obs_steps = self.n_obs_steps,
                num_points  = self.num_points,
                device      = self.device,
                n_warmup    = 3)

            rospy.loginfo(f"[ML Planner v5] 模型就绪 device={self.device}")

        except Exception as e:
            rospy.logerr(f"模型加载失败: {e}"); traceback.print_exc(); sys.exit(1)

    def _call_policy(self, pc_numpy, pos_numpy):
        if self._pc_pinned is not None:
            self._pc_pinned[0].copy_(torch.from_numpy(pc_numpy))
            self._pos_pinned[0].copy_(torch.from_numpy(pos_numpy))
            pc_gpu  = self._pc_pinned.to(self.device, non_blocking=True)
            pos_gpu = self._pos_pinned.to(self.device, non_blocking=True)
        else:
            pc_gpu  = torch.from_numpy(pc_numpy).unsqueeze(0).to(self.device)
            pos_gpu = torch.from_numpy(pos_numpy).unsqueeze(0).to(self.device)

        if self.use_fp16 and self.device.type == 'cuda':
            pc_gpu  = pc_gpu.half()
            pos_gpu = pos_gpu.half()

        obs = {'point_cloud': pc_gpu, 'agent_pos': pos_gpu}

        with torch.inference_mode():
            r = self.policy.predict_action(obs)

        if isinstance(r, dict):
            r = r.get('action', r.get('action_pred', next(iter(r.values()))))
        elif isinstance(r, tuple):
            r = r[0]
        return r.float().cpu().numpy()[0]   # (N, 11)

    # ROS 回调
    def setup_subscribers(self):
        rospy.Subscriber("~odom",         Odometry,    self.odom_cb,   queue_size=1)
        rospy.Subscriber("~pointcloud",   PointCloud2, self.pc_cb,     queue_size=1)
        rospy.Subscriber("/mavros/state", MavrosState, self.mavros_cb, queue_size=1)

    def setup_publishers(self):
        self.cmd_pub    = rospy.Publisher("/planning/pos_cmd", PositionCommand, queue_size=10)
        ego             = f"/drone_{self.drone_id}_ego_planner_node"
        self.goal_pub   = rospy.Publisher(f"{ego}/goal_point",   Marker,      queue_size=10)
        self.traj_pub   = rospy.Publisher(f"{ego}/optimal_list", Marker,      queue_size=10)
        ml              = f"/drone_{self.drone_id}_ml_planner"
        self.detail_pub = rospy.Publisher(f"{ml}/diffusion_detail", MarkerArray, queue_size=10)
        rospy.sleep(0.5)

    def odom_cb(self, msg):
        with self.data_lock:
            self.current_position = np.array([
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z])
            vel_raw = np.array([
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z])
            self.current_orientation_xyzw = np.array([
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w])

            # v5: 始终计算 world 速度
            if self.odom_vel_is_world:
                self.current_velocity_world = vel_raw
            else:
                # odom twist 是 body frame → 旋转到 world
                w, x, y, z = self._ros_to_wxyz(self.current_orientation_xyzw)
                R_b2w = np.array([
                    [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
                    [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
                    [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
                ], dtype=np.float64)
                self.current_velocity_world = R_b2w @ vel_raw

            self.current_yaw = self._quat_to_yaw(self.current_orientation_xyzw)
            if self.state == "INIT":
                self.state = "WAIT_OFFBOARD"

    def mavros_cb(self, msg):
        if self.state != "WAIT_OFFBOARD":
            return
        if msg.mode != "OFFBOARD":
            try:
                rospy.wait_for_service('/mavros/set_mode', timeout=2.0)
                self.set_mode_client(custom_mode="OFFBOARD")
            except Exception:
                pass
        if msg.mode == "OFFBOARD":
            rospy.loginfo("[ML Planner v5] OFFBOARD 模式")
            rospy.sleep(2.0)
            if self.flight_mode == 0 and len(self.waypoints) > 0:
                self.state = "START_MISSION"
                rospy.Timer(rospy.Duration(2.0), self._start_mission, oneshot=True)

    def _start_mission(self, event):
        rospy.loginfo("[ML Planner v5] 任务开始")
        self._set_next_waypoint()

    def pc_cb(self, msg):
        try:
            pts = np.array(
                list(pc2.read_points(msg, field_names=("x","y","z"), skip_nans=True)),
                dtype=np.float32)
            rospy.loginfo_throttle(5, f"[PC] Received {len(pts)} points")
            if len(pts) == 0:
                return
            n = len(pts)
            if n > self.num_points:
                pts = pts[np.random.choice(n, self.num_points, replace=False)]
            elif n < self.num_points:
                pts = np.tile(pts, ((self.num_points + n - 1)//n, 1))[:self.num_points]
            pts = (R_CAM2BODY @ pts.T).T.astype(np.float32)
            with self.data_lock:
                self.current_pointcloud = pts
        except Exception as e:
            rospy.logwarn_throttle(5, f"点云错误: {e}")

    # ═══ v5: 观测采样 (15维 state) ═══
    def _sample_obs(self, event):
        with self.data_lock:
            if self.current_pointcloud is None or not self.has_goal:
                return
            cur_pos     = self.current_position.copy()
            cur_vel_w   = self.current_velocity_world.copy()   # v5: world velocity
            prev_vel_w  = self.prev_velocity_world.copy()
            cur_ori     = self.current_orientation_xyzw.copy()
            goal_w      = self.goal_position.copy()
            pc          = self.current_pointcloud.copy()
            # 更新 prev
            self.prev_velocity_world = cur_vel_w.copy()

        # v5: world 加速度
        acc_world = (cur_vel_w - prev_vel_w) / self.traj_dt

        agent_pos, _ = self._build_agent_pos(cur_pos, cur_vel_w, acc_world, cur_ori, goal_w)
        self.obs_history.append({'point_cloud': pc, 'agent_pos': agent_pos})

        rospy.logdebug(
            f"[OBS] speed={np.linalg.norm(cur_vel_w):.2f}m/s "
            f"dist={np.linalg.norm(goal_w - cur_pos):.1f}m "
            f"buf={len(self.obs_history)}/{self.n_obs_steps}")

    def _build_agent_pos(self, pos_w, vel_world, acc_world, orient_xyzw, goal_w):
        """v5: 15维 state = [pos_w(3), vel_world(3), acc_world(3),
                              goal_body(3), dist(1), sin(yaw)(1), cos(yaw)(1)]"""
        yaw  = self._quat_to_yaw(orient_xyzw)
        c, s = np.cos(-yaw), np.sin(-yaw)
        R    = np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float64)
        goal_rel = goal_w - pos_w
        agent_pos = np.concatenate([
            pos_w,                       # [0:3]   世界坐标位置
            vel_world,                   # [3:6]   世界速度 (v5: 改为world)
            acc_world,                   # [6:9]   世界加速度 (v5: 改为world)
            R @ goal_rel,                # [9:12]  目标偏移 (body frame, 不变)
            [np.linalg.norm(goal_rel)],  # [12]    距离
            [np.sin(yaw)],               # [13]    sin(yaw) (v5: 替代raw yaw)
            [np.cos(yaw)],               # [14]    cos(yaw) (v5: 新增)
        ]).astype(np.float32)
        return agent_pos, R @ goal_rel

    # ═══ v5 核心: _action_to_world (替代 v4 的 _body_to_world) ═══
    def _action_to_world(self, raw_action, pos_w, vel_world, yaw):
        """
        v5: world step-wise delta → world 坐标航点
        
        与 v4 _body_to_world 的关键区别:
          - v4: pos[k] = pos[k-1] + R(yaw[k-1]) @ dpos_body[k]  (旋转+累积)
          - v5: pos[k] = pos[k-1] + dpos_world[k]                (纯加法累积)
          → yaw 不参与位置积分, 避障偏移不被 yaw 吸收
          
        与前版设计文档（chunk-wise直接加法）的区别:
          - chunk-wise: world_pos = raw[:, 0:3] + pos_w  (无循环, 但normalizer利用率仅4%)
          - step-wise:  world_pos = 累积加法              (有循环, normalizer利用率与v4相同)
        """
        N = len(raw_action)
        positions = np.zeros((N, 3))
        yaws = np.zeros(N)
        
        # 前向积分 (纯向量加法, 无 yaw 旋转!)
        cur_pos, cur_yaw = pos_w.copy(), yaw
        for k in range(N):
            cur_pos = cur_pos + raw_action[k, DPOS_W]   # world delta, 无旋转
            cur_yaw = normalize_angle(cur_yaw + float(raw_action[k, DYAW_CW]))
            positions[k] = cur_pos
            yaws[k] = cur_yaw

        # Z 轴安全 clamp
        Z_MIN, Z_MAX = 1.0, 3.0
        z_before = positions[:, 2].copy()
        positions[:, 2] = np.clip(positions[:, 2], Z_MIN, Z_MAX)
        if np.any(z_before != positions[:, 2]):
            rospy.logwarn(f"[Z-clamp] [{z_before.min():.2f},{z_before.max():.2f}] → "
                          f"[{positions[:,2].min():.2f},{positions[:,2].max():.2f}]")

        # vel/acc 已是 world absolute, 直接复制
        world_action = np.zeros_like(raw_action)
        world_action[:, POS]    = positions
        world_action[:, VEL]    = raw_action[:, VEL_W]
        world_action[:, ACC]    = raw_action[:, ACC_W]
        world_action[:, YAW]    = yaws
        world_action[:, YAW_RT] = raw_action[:, YAW_RATE]

        return world_action

    def control_loop(self, event):
        if self.state in ("WAIT_OFFBOARD", "START_MISSION", "INIT"):
            return
        if not self.has_goal:
            return

        with self.data_lock:
            pos  = self.current_position.copy()
            goal = self.goal_position.copy()
        if np.linalg.norm(pos - goal) < self.goal_tolerance:
            self._on_goal_reached()
            return

        if self.traj_mgr.is_valid():
            r = self.traj_mgr.sample(rospy.Time.now())
            if r is not None:
                pos_cmd, vel_cmd, acc_cmd, yaw_cmd, yr_cmd, progress = r

                if progress >= self.replan_ratio:
                    if not self._replan_logged:
                        rospy.loginfo(f"[ML Planner] 并行推理触发 progress={progress:.2f}")
                        self._replan_logged = True
                    self._try_start_planning()
                else:
                    self._replan_logged = False

                if progress < 1.0:
                    self._publish_cmd(pos_cmd, vel_cmd, acc_cmd, yaw_cmd, yr_cmd, "traj")
                else:
                    self._publish_cmd(pos_cmd, np.zeros(3), np.zeros(3), yaw_cmd, 0.0, "traj_hold")
                return
            else:
                self.traj_mgr.clear()

        self._replan_logged = False
        self._try_start_planning()
        self._publish_hold_cmd()

    def _try_start_planning(self):
        with self.planning_lock:
            if self.is_planning:
                return
            self.is_planning = True
        Thread(target=self._planning_worker, daemon=True).start()

    def _planning_worker(self):
        try:
            self._do_plan()
        except Exception as e:
            rospy.logerr(f"规划异常: {e}"); traceback.print_exc()
        finally:
            with self.planning_lock:
                self.is_planning = False

    def _do_plan(self):
        if len(self.obs_history) < self.n_obs_steps:
            rospy.loginfo_throttle(1, f"积累观测: {len(self.obs_history)}/{self.n_obs_steps}")
            return

        recent_obs = list(self.obs_history)[-self.n_obs_steps:]
        pc_s  = np.stack([o["point_cloud"] for o in recent_obs])
        ag_s  = np.stack([o["agent_pos"]   for o in recent_obs])   # (n_obs, 15)

        with self.data_lock:
            goal_w = self.goal_position.copy()
            cur_pos_now = self.current_position.copy()
            cur_yaw_now = self.current_yaw

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()

        # ========== DEPLOY DUMP (临时诊断) ==========
        if not hasattr(self, '_dump_count'):
            self._dump_count = 0
            self._dump_dir = os.path.expanduser('~/ml_planner_log/deploy_pc_dumps')
            os.makedirs(self._dump_dir, exist_ok=True)

        drone_x = cur_pos_now[0]
        if 5.0 <= drone_x <= 10.0 and self._dump_count < 30:
            pc_latest = pc_s[-1]          # (num_points, 3)
            state_latest = ag_s[-1]       # (15,)
            pc_history = pc_s             # (n_obs, num_points, 3)
            state_history = ag_s          # (n_obs, 15)

            fname = f'{self._dump_dir}/deploy_{self._dump_count:03d}_x{drone_x:.2f}.npz'
            np.savez(fname,
                     pc_latest=pc_latest,
                     state_latest=state_latest,
                     pc_history=pc_history,
                     state_history=state_history,
                     drone_pos=cur_pos_now,
                     drone_yaw=cur_yaw_now)
            self._dump_count += 1
            rospy.loginfo(f'[DUMP] #{self._dump_count} saved: x={drone_x:.2f}')
        # ============================================

        raw_action = self._call_policy(pc_s, ag_s)   # (N, 11)

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        dt_ms = (time.time() - t0) * 1000

        if dt_ms > 500:
            rospy.logwarn(f"[ML Planner] 推理{dt_ms:.0f}ms>500ms")
        else:
            rospy.loginfo(f"[ML Planner] 推理{dt_ms:.0f}ms ✓")

        # v5: 始终用无人机实际odom作为规划起点
        with self.data_lock:
            cur_pos_now = self.current_position.copy()
            cur_vel_now = self.current_velocity_world.copy()
            cur_yaw_now = self.current_yaw

        goal_rel_b   = ag_s[-1][9:12]    # goal_body at [9:12]
        # ═══ v5: _action_to_world (替代 _body_to_world) ═══
        world_action = self._action_to_world(raw_action, cur_pos_now, cur_vel_now, cur_yaw_now)

        self._log_diffusion_output(raw_action, world_action, dt_ms, cur_pos_now, goal_w, goal_rel_b)
        self.csv_logger.log_diffusion(raw_action, world_action, dt_ms,
                                      cur_pos_now, cur_yaw_now, goal_w, goal_rel_b)
        self.traj_mgr.set_trajectory(world_action, self.traj_dt, cur_pos=cur_pos_now)
        self._publish_traj_marker(world_action)
        self._publish_detail_markers(world_action, raw_action)

    def _publish_cmd(self, pos, vel, acc, yaw, yr, source="traj"):
        cmd = PositionCommand()
        cmd.header.stamp    = rospy.Time.now()
        cmd.header.frame_id = "world"
        cmd.position.x, cmd.position.y, cmd.position.z             = float(pos[0]), float(pos[1]), float(pos[2])
        cmd.velocity.x, cmd.velocity.y, cmd.velocity.z             = float(vel[0]), float(vel[1]), float(vel[2])
        cmd.acceleration.x, cmd.acceleration.y, cmd.acceleration.z = float(acc[0]), float(acc[1]), float(acc[2])
        cmd.yaw, cmd.yaw_dot    = float(yaw), float(yr)
        cmd.trajectory_flag     = PositionCommand.TRAJECTORY_STATUS_READY
        cmd.trajectory_id       = self.cmd_count % 100 + 1
        cmd.kx, cmd.kv          = self.pos_gain, self.vel_gain
        self.cmd_pub.publish(cmd)
        self.cmd_count += 1

        with self.data_lock:
            cur = self.current_position.copy()
        self.csv_logger.log_cmd(self.cmd_count, source, pos, vel, acc, yaw, yr, cur)

        if self.cmd_count % 50 == 0:
            err = np.linalg.norm(np.array(pos[:3]) - cur)
            rospy.loginfo(
                f"[CMD#{self.cmd_count}] {source:>9} | "
                f"pos[{pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}] "
                f"vel[{vel[0]:.2f},{vel[1]:.2f},{vel[2]:.2f}] "
                f"yaw={yaw:.2f} | "
                f"实际[{cur[0]:.2f},{cur[1]:.2f},{cur[2]:.2f}] err={err:.2f}m")

    def _publish_hold_cmd(self):
        with self.data_lock:
            hp = self.current_position.copy()
            hy = self.current_yaw
        self._publish_cmd(hp, np.zeros(3), np.zeros(3), hy, 0.0, "hold")

    # 可视化
    def _vis_tick(self, event):
        self._publish_goal_marker()

    def _publish_goal_marker(self):
        if self.goal_position is None:
            return
        m = Marker()
        m.header.stamp = rospy.Time.now(); m.header.frame_id = "world"
        m.ns = "goal_point"; m.id = 0; m.type = Marker.SPHERE; m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = \
            self.goal_position[0], self.goal_position[1], self.goal_position[2]
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.5
        m.color = ColorRGBA(1.0, 0.0, 0.0, 0.8)
        self.goal_pub.publish(m)

    def _publish_traj_marker(self, world_action):
        m = Marker()
        m.header.stamp = rospy.Time.now(); m.header.frame_id = "world"
        m.ns = "optimal_trajectory"; m.id = 0
        m.type = Marker.LINE_STRIP; m.action = Marker.ADD
        m.scale.x = 0.08; m.color = ColorRGBA(0.1, 0.5, 1.0, 0.9)
        m.pose.orientation.w = 1.0
        for i in range(len(world_action)):
            m.points.append(Point(x=float(world_action[i,0]),
                                  y=float(world_action[i,1]),
                                  z=float(world_action[i,2])))
        self.traj_pub.publish(m)

    def _publish_detail_markers(self, world_action, raw_action):
        ma  = MarkerArray()
        now = rospy.Time.now()
        clr = Marker(); clr.action = Marker.DELETEALL
        ma.markers.append(clr)
        for i in range(len(world_action)):
            ratio = i / max(1, len(world_action)-1)
            s = Marker()
            s.header.stamp = now; s.header.frame_id = "world"
            s.ns = "dp"; s.id = i; s.type = Marker.SPHERE; s.action = Marker.ADD
            s.pose.position.x = float(world_action[i,0])
            s.pose.position.y = float(world_action[i,1])
            s.pose.position.z = float(world_action[i,2])
            s.pose.orientation.w = 1.0
            s.scale.x = s.scale.y = s.scale.z = 0.12
            s.color = ColorRGBA(ratio, 1-ratio, 0.2, 0.8)
            ma.markers.append(s)
            if i % 3 == 0:
                v = world_action[i, VEL]
                if np.linalg.norm(v) > 0.01:
                    a = Marker()
                    a.header.stamp = now; a.header.frame_id = "world"
                    a.ns = "dv"; a.id = 100+i; a.type = Marker.ARROW; a.action = Marker.ADD
                    a.points = [
                        Point(x=float(world_action[i,0]),   y=float(world_action[i,1]),   z=float(world_action[i,2])),
                        Point(x=float(world_action[i,0]+v[0]*0.5),
                              y=float(world_action[i,1]+v[1]*0.5),
                              z=float(world_action[i,2]+v[2]*0.5))]
                    a.scale.x = 0.03; a.scale.y = 0.06
                    a.color = ColorRGBA(1.0, 1.0, 0.0, 0.7)
                    ma.markers.append(a)
            if i == 0 or i == len(world_action)-1 or i % 5 == 0:
                t = Marker()
                t.header.stamp = now; t.header.frame_id = "world"
                t.ns = "dt"; t.id = 200+i; t.type = Marker.TEXT_VIEW_FACING; t.action = Marker.ADD
                t.pose.position.x = float(world_action[i,0])
                t.pose.position.y = float(world_action[i,1])
                t.pose.position.z = float(world_action[i,2]) + 0.2
                t.scale.z = 0.15; t.color = ColorRGBA(1,1,1,0.9)
                # v5: 显示 chunk-wise delta 幅值
                t.text = f"t{i} Δw={np.linalg.norm(raw_action[i,DPOS_W]):.2f}"
                ma.markers.append(t)
        self.detail_pub.publish(ma)

    def _log_diffusion_output(self, raw_action, world_action, dt_ms,
                              cur_pos, goal_w, goal_rel_b):
        sep = "=" * 105
        rospy.loginfo(sep)
        rospy.loginfo(
            f"[DIFFUSION v5] {dt_ms:.0f}ms | "
            f"pos[{cur_pos[0]:.2f},{cur_pos[1]:.2f},{cur_pos[2]:.2f}] "
            f"goal[{goal_w[0]:.2f},{goal_w[1]:.2f},{goal_w[2]:.2f}] "
            f"dist={np.linalg.norm(goal_rel_b):.2f}m")
        rospy.loginfo("-" * 105)
        h = f"{'t':>3}" + "".join(f" {n:>8}" for n in DIM_NAMES_RAW[:6]) + f"  {'|Δw|':>6}"
        rospy.loginfo(h)
        for i in range(min(5, len(raw_action))):
            row = f" {i:>2}" + "".join(f" {raw_action[i,d]:>8.4f}" for d in range(6))
            row += f"  {np.linalg.norm(raw_action[i,DPOS_W]):>6.4f}"
            rospy.loginfo(row)
        if len(raw_action) > 5:
            rospy.loginfo(f"  ... ({len(raw_action)} 步)")
        rospy.loginfo("-" * 105)
        h = f"{'t':>3}" + "".join(f" {n:>8}" for n in DIM_NAMES_WORLD[:6]) + \
            f"  {'d_cur':>7} {'d_goal':>7}"
        rospy.loginfo(h)
        for i in range(len(world_action)):
            row = f" {i:>2}" + "".join(f" {world_action[i,d]:>8.3f}" for d in range(6))
            row += f"  {np.linalg.norm(world_action[i,POS]-cur_pos):>6.2f}m"
            row += f" {np.linalg.norm(world_action[i,POS]-goal_w):>6.2f}m"
            rospy.loginfo(row)
        total = sum(np.linalg.norm(world_action[i+1,POS]-world_action[i,POS])
                    for i in range(len(world_action)-1))
        rospy.loginfo(
            f"  轨迹长{total:.2f}m | 终点距目标{np.linalg.norm(world_action[-1,POS]-goal_w):.2f}m")
        rospy.loginfo(sep)

    def _on_goal_reached(self):
        rospy.loginfo("[ML Planner v5] 到达目标")
        if self.flight_mode == 0:
            self._set_next_waypoint()
        else:
            self.has_goal = False

    def _set_next_waypoint(self):
        if self.waypoint_index >= len(self.waypoints):
            rospy.loginfo("[ML Planner v5] 所有航点完成")
            self.has_goal = False
            return
        self.goal_position  = self.waypoints[self.waypoint_index].copy()
        self.has_goal       = True
        self.obs_history.clear()
        self.state          = "EXECUTING"
        rospy.loginfo(f"[ML Planner v5] 航点{self.waypoint_index}: {self.goal_position}")
        self._publish_goal_marker()
        self.waypoint_index += 1

    @staticmethod
    def _ros_to_wxyz(quat_xyzw):
        return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

    @staticmethod
    def _quat_to_yaw(q_xyzw):
        x, y, z, w = q_xyzw
        return np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))


def main():
    try:
        MLPlannerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"异常: {e}"); traceback.print_exc()

if __name__ == '__main__':
    main()