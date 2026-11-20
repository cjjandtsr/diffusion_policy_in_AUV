#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML Planner Node - 扩散模型 + 机体坐标→世界坐标转换 + 轨迹插值 + EGO兼容可视化

核心修改 (适配 body-relative 新模型):
  1. agent_pos 构建: vel→body, quat→wxyz, goal→body_relative
  2. action 后处理: body_relative → world_absolute (前向积分)
  3. TrajectoryManager 接收世界坐标绝对值

架构（对标EGO）：
  planning_thread(异步推理) → body2world转换 → TrajectoryManager(时间插值) → 50Hz pos_cmd → PX4Ctrl
"""

import os, sys, time, traceback
import numpy as np
import rospy
from collections import deque
from threading import Lock, Thread
import csv
from datetime import datetime

# ROS消息
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

DIFFUSION_POLICY_PATH = os.path.expanduser(
    "~/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy")
if os.path.exists(DIFFUSION_POLICY_PATH):
    sys.path.insert(0, DIFFUSION_POLICY_PATH)
else:
    print("[ML Planner] 致命: 找不到扩散模型路径!"); sys.exit(1)

os.environ['WANDB_SILENT'] = "True"
os.environ['WANDB_MODE'] = "disabled"

try:
    from diffusion_policy_3d.workspace.base_workspace import BaseWorkspace
except ImportError:
    traceback.print_exc(); sys.exit(1)


# ═══════════════════════════════════════════════════════════════
# Action维度定义 (新模型: 机体坐标相对值)
# ═══════════════════════════════════════════════════════════════
# 模型输出 action[11] = Δpos_body(3) + Δvel_body(3) + acc_body(3) + Δyaw(1) + yaw_rate(1)
# 转换后世界坐标: pos_w(3) + vel_w(3) + acc_w(3) + yaw(1) + yaw_rate(1)
DPOS_B = slice(0, 3)   # 机体坐标位置增量
DVEL_B = slice(3, 6)   # 机体坐标速度增量
ACC_B  = slice(6, 9)   # 机体坐标加速度
DYAW   = 9             # 偏航角增量
YAW_RATE = 10          # 偏航角速度

# 转换后世界坐标的索引 (与TrajectoryManager一致)
POS = slice(0, 3)
VEL = slice(3, 6)
ACC = slice(6, 9)
YAW = 9
YAW_RT = 10

DIM_NAMES_RAW = ["dp_bx","dp_by","dp_bz","dv_bx","dv_by","dv_bz",
                 "a_bx","a_by","a_bz","dyaw","yr"]
DIM_NAMES_WORLD = ["pos_x","pos_y","pos_z","vel_x","vel_y","vel_z",
                   "acc_x","acc_y","acc_z","yaw","yr"]


# ═══════════════════════════════════════════════════════════════
# 坐标变换工具 (与 bag2zarr_body.py 一致)
# ═══════════════════════════════════════════════════════════════

# 相机光学坐标系 → 机体坐标系 (FLU)
# 与 bag2zarr_body.py 第44-48行完全一致
R_CAM2BODY = np.array([
    [ 0,  0,  1],   # body_x = cam_z  (前方)
    [-1,  0,  0],   # body_y = -cam_x (左方)
    [ 0, -1,  0],   # body_z = -cam_y (上方)
], dtype=np.float64)

def quat_to_rotmat(quat_wxyz):
    """四元数 [w,x,y,z] → 3×3 旋转矩阵 R_body2world"""
    w, x, y, z = [float(v) for v in quat_wxyz]
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
    ], dtype=np.float64)


def yaw_from_quat_wxyz(quat_wxyz):
    """四元数 [w,x,y,z] → 偏航角"""
    w, x, y, z = [float(v) for v in quat_wxyz]
    return np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))


def yaw_to_rotmat(yaw):
    """偏航角 → 3×3 旋转矩阵 (仅绕z轴)"""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1],
    ], dtype=np.float64)


def normalize_angle(a):
    """将角度归一化到 [-π, π]"""
    return (a + np.pi) % (2*np.pi) - np.pi


# ═══════════════════════════════════════════════════════════════
# CSV日志器
# ═══════════════════════════════════════════════════════════════
class CSVLogger:
    def __init__(self):
        log_dir = os.path.expanduser("~/ml_planner_log")
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 扩散模型原始输出 (机体坐标)
        self.diff_path = os.path.join(log_dir, f"diffusion_{ts}.csv")
        self.diff_file = open(self.diff_path, 'w', newline='')
        self.diff_w = csv.writer(self.diff_file)
        self.diff_w.writerow(
            ["plan_id","step","ros_time","inference_ms",
             "cur_x","cur_y","cur_z","cur_yaw",
             "goal_x","goal_y","goal_z",
             "goal_bx","goal_by","goal_bz"]
            + [f"raw_{n}" for n in DIM_NAMES_RAW]
            + [f"world_{n}" for n in DIM_NAMES_WORLD])
        
        self.cmd_path = os.path.join(log_dir, f"pos_cmd_{ts}.csv")
        self.cmd_file = open(self.cmd_path, 'w', newline='')
        self.cmd_w = csv.writer(self.cmd_file)
        self.cmd_w.writerow(
            ["cmd_id","ros_time","source",
             "cmd_px","cmd_py","cmd_pz","cmd_vx","cmd_vy","cmd_vz",
             "cmd_ax","cmd_ay","cmd_az","cmd_yaw","cmd_yr",
             "cur_px","cur_py","cur_pz","err_m"])
        
        self.plan_count = 0
        rospy.loginfo(f"[CSV] {self.diff_path}")
        rospy.loginfo(f"[CSV] {self.cmd_path}")
    
    def log_diffusion(self, raw_action, world_action, inference_ms,
                      cur_pos, cur_yaw, goal_w, goal_b):
        """记录原始输出 + 世界坐标转换结果"""
        self.plan_count += 1
        t = rospy.Time.now().to_sec()
        for s in range(len(raw_action)):
            self.diff_w.writerow(
                [self.plan_count, s, f"{t:.3f}", f"{inference_ms:.0f}"]
                + [f"{v:.4f}" for v in cur_pos]
                + [f"{cur_yaw:.4f}"]
                + [f"{v:.4f}" for v in goal_w]
                + [f"{v:.4f}" for v in goal_b]
                + [f"{raw_action[s,d]:.6f}" for d in range(raw_action.shape[1])]
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
        self.diff_file.close(); self.cmd_file.close()


# ═══════════════════════════════════════════════════════════════
# 轨迹管理器（接收世界坐标绝对值，模仿EGO的traj_server时间采样）
# ═══════════════════════════════════════════════════════════════
class TrajectoryManager:
    """接收已转换为世界坐标的轨迹，按时间插值输出"""
    def __init__(self):
        self.lock = Lock()
        self.waypoints = None     # (N, 11) 世界坐标绝对值
        self.start_time = None
        self.traj_dt = 0.1
        self.duration = 0.0
        self.last_pos = None
        self.last_yaw = 0.0
    
    def set_trajectory(self, world_action, traj_dt):
        """设置世界坐标轨迹"""
        with self.lock:
            self.waypoints = world_action.copy()
            self.start_time = rospy.Time.now()
            self.traj_dt = traj_dt
            self.duration = (len(world_action) - 1) * traj_dt
            self.last_pos = world_action[-1, POS].copy()
            self.last_yaw = float(world_action[-1, YAW])
    
    def sample(self, now):
        with self.lock:
            if self.waypoints is None or self.start_time is None:
                return None
            t = max(0.0, (now - self.start_time).to_sec())
            progress = t / self.duration if self.duration > 0 else 1.0
            t_idx = t / self.traj_dt
            i = int(t_idx); frac = t_idx - i; N = len(self.waypoints)
            if i >= N - 1:
                wp = self.waypoints[-1]
                return (wp[POS], np.zeros(3), np.zeros(3),
                        float(wp[YAW]), 0.0, progress)
            wp0, wp1 = self.waypoints[i], self.waypoints[i+1]
            pos = wp0[POS]*(1-frac) + wp1[POS]*frac
            vel = wp0[VEL]*(1-frac) + wp1[VEL]*frac
            acc = wp0[ACC]*(1-frac) + wp1[ACC]*frac
            yaw = float(wp0[YAW]*(1-frac) + wp1[YAW]*frac)
            yr = float(wp0[YAW_RT]*(1-frac) + wp1[YAW_RT]*frac)
            return pos, vel, acc, yaw, yr, progress
    
    def get_hold_state(self):
        with self.lock:
            return (self.last_pos.copy(), self.last_yaw) if self.last_pos is not None else (None, 0.0)
    
    def get_waypoints(self):
        with self.lock:
            return self.waypoints.copy() if self.waypoints is not None else None
    
    def is_valid(self):
        with self.lock:
            return self.waypoints is not None


# ═══════════════════════════════════════════════════════════════
# ML Planner 主节点
# ═══════════════════════════════════════════════════════════════
class MLPlannerNode:
    
    def __init__(self):
        rospy.init_node('ml_planner_node', anonymous=False)
        rospy.loginfo("=" * 70)
        rospy.loginfo("  ML Planner v2 - Body-Relative → World 坐标转换")
        rospy.loginfo("=" * 70)
        
        self.load_parameters()
        
        self.data_lock = Lock()
        self.current_position = np.array([0.0, 0.0, 2.0])
        self.current_velocity_world = np.zeros(3)  # 世界坐标速度
        self.current_orientation_xyzw = np.array([0, 0, 0, 1.0])  # ROS [x,y,z,w]
        self.current_yaw = 0.0
        self.current_pointcloud = None
        self.goal_position = None  # 世界坐标目标点
        self.has_goal = False
        self.waypoint_index = 0
        self.state = "INIT"
        self.cmd_count = 0
        
        self.traj_mgr = TrajectoryManager()
        self.planning_lock = Lock()
        self.is_planning = False
        
        self.n_obs_steps = 2
        self.action_horizon = 15
        self.num_points = 4096
        self.obs_history = deque(maxlen=self.n_obs_steps)
        
        self.csv_logger = CSVLogger()
        self.set_mode_client = rospy.ServiceProxy('/mavros/set_mode', SetMode)
        
        self.policy = None; self.device = None
        self.load_diffusion_policy()
        self.setup_subscribers()
        self.setup_publishers()
        
        self.control_timer = rospy.Timer(
            rospy.Duration(1.0 / self.control_freq), self.control_loop)
        self.vis_timer = rospy.Timer(rospy.Duration(0.2), self._vis_tick)
        
        rospy.on_shutdown(self.csv_logger.close)
        rospy.loginfo("[ML Planner] 初始化完成")
    
    # ───────── 参数 ─────────
    def load_parameters(self):
        self.control_freq = rospy.get_param("~control_freq", 50.0)
        self.replan_ratio = rospy.get_param("~replan_ratio", 0.7)
        self.goal_tolerance = rospy.get_param("~goal_tolerance", 0.5)
        self.default_height = rospy.get_param("~default_height", 1.5)
        self.flight_mode = rospy.get_param("~flight_mode", 0)
        self.traj_dt = rospy.get_param("~traj_dt", 0.1)
        self.drone_id = rospy.get_param("~drone_id", 0)
        self.pos_gain = self._parse_gain("~pos_gain", [3.5, 3.5, 3.5])
        self.vel_gain = self._parse_gain("~vel_gain", [3.5, 3.5, 3.5])
        self.checkpoint_path = rospy.get_param("~checkpoint_path", "")
        
        # ⭐ 新参数：odom 速度是否为世界坐标 (MAVROS local_position/odom 通常是世界坐标)
        self.odom_vel_is_world = rospy.get_param("~odom_vel_is_world", True)
        
        self.waypoints = []
        for i in range(rospy.get_param("~waypoint_num", 0)):
            x = rospy.get_param(f"~waypoint{i}_x", 0.0)
            y = rospy.get_param(f"~waypoint{i}_y", 0.0)
            z = rospy.get_param(f"~waypoint{i}_z", self.default_height)
            self.waypoints.append(np.array([x, y, z]))
            rospy.loginfo(f"  航点{i}: [{x:.1f}, {y:.1f}, {z:.1f}]")
    
    def _parse_gain(self, name, default):
        val = rospy.get_param(name, default)
        if isinstance(val, str):
            return [float(x.strip()) for x in val.strip('[]').split(',')]
        return [float(x) for x in val] if isinstance(val, list) else default
    
    # ───────── 模型加载 ─────────
    def load_diffusion_policy(self):
        rospy.loginfo("[ML Planner] 加载扩散模型...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not os.path.exists(self.checkpoint_path):
            rospy.logerr(f"Checkpoint不存在: {self.checkpoint_path}"); sys.exit(1)
        try:
            payload = torch.load(self.checkpoint_path, map_location='cpu')
            cfg = payload['cfg']
            if hasattr(cfg, 'n_obs_steps'):
                self.n_obs_steps = cfg.n_obs_steps
                self.obs_history = deque(maxlen=self.n_obs_steps)
            if hasattr(cfg, 'horizon') and hasattr(cfg, 'n_action_steps'):
                self.action_horizon = cfg.horizon - cfg.n_obs_steps + 1
            cls = hydra.utils.get_class(cfg._target_)
            workspace = cls(cfg)
            workspace.load_payload(payload, exclude_keys=None, include_keys=None)
            self.policy = workspace.model
            if hasattr(workspace, 'ema_model') and workspace.ema_model is not None:
                self.policy = workspace.ema_model
            self.policy = self.policy.to(self.device).eval()
            
            dummy = {
                'point_cloud': torch.randn(1, self.n_obs_steps, self.num_points, 3,
                                           device=self.device),
                'agent_pos': torch.randn(1, self.n_obs_steps, 13, device=self.device)}
            with torch.no_grad():
                test_out = self._call_policy(dummy)
            rospy.loginfo(f"[ML Planner] 测试输出: shape={test_out.shape}")
            rospy.loginfo(f"[ML Planner] 模型加载成功! device={self.device}")
        except Exception as e:
            rospy.logerr(f"模型加载失败: {e}"); traceback.print_exc(); sys.exit(1)
    
    def _call_policy(self, obs_tensor):
        if hasattr(self.policy, 'predict_action'):
            r = self.policy.predict_action(obs_tensor)
        else:
            r = self.policy(obs_tensor)
        if isinstance(r, dict):
            r = r.get('action', r.get('action_pred', next(iter(r.values()))))
        elif isinstance(r, tuple):
            r = r[0]
        return r
    
    # ───────── ROS接口 ─────────
    def setup_subscribers(self):
        rospy.Subscriber("~odom", Odometry, self.odom_cb, queue_size=1)
        rospy.Subscriber("~pointcloud", PointCloud2, self.pc_cb, queue_size=1)
        rospy.Subscriber("/mavros/state", MavrosState, self.mavros_cb, queue_size=1)
    
    def setup_publishers(self):
        self.cmd_pub = rospy.Publisher("/planning/pos_cmd", PositionCommand, queue_size=10)
        ego = f"/drone_{self.drone_id}_ego_planner_node"
        self.goal_pub = rospy.Publisher(f"{ego}/goal_point", Marker, queue_size=10)
        self.traj_pub = rospy.Publisher(f"{ego}/optimal_list", Marker, queue_size=10)
        ml = f"/drone_{self.drone_id}_ml_planner"
        self.detail_pub = rospy.Publisher(f"{ml}/diffusion_detail", MarkerArray, queue_size=10)
        rospy.sleep(0.5)
        rospy.loginfo(f"[RViz话题] 目标: {ego}/goal_point  轨迹: {ego}/optimal_list")
        rospy.loginfo(f"[RViz话题] 详细: {ml}/diffusion_detail")
    
    # ═══════════════════════════════════════════════════════════
    # 回调: 获取当前状态
    # ═══════════════════════════════════════════════════════════
    def odom_cb(self, msg):
        with self.data_lock:
            self.current_position = np.array([
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z])
            
            # ROS odom velocity
            vel_raw = np.array([
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z])
            
            # ROS四元数 [x,y,z,w]
            self.current_orientation_xyzw = np.array([
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w])
            
            # 统一存储世界坐标速度
            if self.odom_vel_is_world:
                self.current_velocity_world = vel_raw
            else:
                # odom给的是body速度，需要转为world
                quat_wxyz = self._ros_to_wxyz(self.current_orientation_xyzw)
                R_b2w = quat_to_rotmat(quat_wxyz)
                self.current_velocity_world = R_b2w @ vel_raw
            
            self.current_yaw = self._quat_to_yaw(self.current_orientation_xyzw)
            
            if self.state == "INIT":
                self.state = "WAIT_OFFBOARD"
                rospy.loginfo(f"[ML Planner] 首次位置: {self.current_position}")
    
    def mavros_cb(self, msg):
        if self.state != "WAIT_OFFBOARD":
            return
        if msg.mode != "OFFBOARD":
            try:
                rospy.wait_for_service('/mavros/set_mode', timeout=2.0)
                self.set_mode_client(custom_mode="OFFBOARD")
            except: pass
        if msg.mode == "OFFBOARD":
            rospy.loginfo("[ML Planner] OFFBOARD模式")
            rospy.sleep(2.0)
            if self.flight_mode == 0 and len(self.waypoints) > 0:
                self.state = "START_MISSION"
                rospy.Timer(rospy.Duration(2.0), self._start_mission, oneshot=True)
    
    def _start_mission(self, event):
        rospy.loginfo("[ML Planner] 开始任务!")
        self._set_next_waypoint()
    
    def pc_cb(self, msg):
        try:
            pts = list(pc2.read_points(msg, field_names=("x","y","z"), skip_nans=True))
            if not pts: return
            pts = np.array(pts, dtype=np.float32)
            if len(pts) > self.num_points:
                pts = pts[np.random.choice(len(pts), self.num_points, replace=False)]
            elif len(pts) < self.num_points:
                pts = np.tile(pts, (self.num_points//len(pts)+1, 1))[:self.num_points]
            # 相机光学坐标系 → 机体坐标系, 与 bag2zarr_body.py 一致
            pts = (R_CAM2BODY @ pts.astype(np.float64).T).T.astype(np.float32)
            with self.data_lock:
                self.current_pointcloud = pts
        except Exception as e:
            rospy.logwarn_throttle(5, f"点云错误: {e}")
    
    # ═══════════════════════════════════════════════════════════
    # ⭐ 核心修改1: 构建 agent_pos (匹配训练数据格式)
    # ═══════════════════════════════════════════════════════════
    # 训练数据 state[13] = [pos_w(3), vel_body(3), quat_wxyz(4), goal_rel_body(3)]
    
    def _build_agent_pos(self, pos_w, vel_world, orient_xyzw, goal_w):
        """
        构建与训练数据完全一致的 agent_pos[13]
        
        训练数据格式:
          [0:3]  pos_w       - 世界坐标位置 (不变)
          [3:6]  vel_body    - 机体坐标系速度 = R_w2b @ vel_world
          [6:10] quat_wxyz   - 四元数 [w,x,y,z] (ROS是[x,y,z,w])
          [10:13] goal_rel_b - 机体坐标下目标相对位置 = R_w2b @ (goal_w - pos_w)
        """
        # 四元数: ROS [x,y,z,w] → 训练数据 [w,x,y,z]
        quat_wxyz = self._ros_to_wxyz(orient_xyzw)
        
        # 旋转矩阵
        R_b2w = quat_to_rotmat(quat_wxyz)
        R_w2b = R_b2w.T
        
        # 速度: 世界 → 机体
        vel_body = R_w2b @ vel_world
        
        # 目标: 世界绝对 → 机体相对
        goal_rel_world = goal_w - pos_w
        goal_rel_body = R_w2b @ goal_rel_world
        
        agent_pos = np.concatenate([
            pos_w,          # [0:3]  世界坐标位置
            vel_body,       # [3:6]  机体坐标速度
            quat_wxyz,      # [6:10] 四元数 [w,x,y,z]
            goal_rel_body,  # [10:13] 机体坐标下目标相对位置
        ]).astype(np.float32)
        
        return agent_pos, goal_rel_body
    
    # ═══════════════════════════════════════════════════════════
    # ⭐ 核心修改2: 模型输出后处理 (机体相对 → 世界绝对)
    # ═══════════════════════════════════════════════════════════
    
    def _body_to_world(self, raw_action, pos_w, vel_world, quat_wxyz):
        """
        将模型输出的机体坐标 Δpos 转换为世界坐标轨迹
        
        只使用模型的位置输出:
          位置: 前向积分 Δpos
          速度: 相邻位置差分 / dt
          加速度: 相邻速度差分 / dt
          偏航角变化率: 不使用模型输出, 设为0
        """
        N = len(raw_action)
        world_action = np.zeros_like(raw_action)
        
        cur_pos = pos_w.copy()
        cur_yaw = yaw_from_quat_wxyz(quat_wxyz)
        R_b2w = quat_to_rotmat(quat_wxyz)
        
        # ── 第1步: 前向积分位置和偏航 ──
        positions = np.zeros((N, 3))
        yaws = np.zeros(N)
        
        for k in range(N):
            dpos_b = raw_action[k, DPOS_B]
            dyaw   = float(raw_action[k, DYAW])
            
            cmd_pos_w = cur_pos + R_b2w @ dpos_b
            cmd_yaw = normalize_angle(cur_yaw + dyaw)
            
            positions[k] = cmd_pos_w
            yaws[k] = cmd_yaw
            
            cur_pos = cmd_pos_w
            cur_yaw = cmd_yaw
            R_b2w = yaw_to_rotmat(cur_yaw)
        
        # ── 第2步: 位置差分 → 速度, 速度差分 → 加速度 ──
        dt = self.traj_dt
        velocities = np.zeros((N, 3))
        
        velocities[0] = (positions[0] - pos_w) / dt
        for k in range(1, N):
            velocities[k] = (positions[k] - positions[k-1]) / dt
        
        for k in range(N):
            world_action[k, POS] = positions[k]
            world_action[k, VEL] = velocities[k]
            world_action[k, YAW] = yaws[k]
            world_action[k, YAW_RT] = 0.0
            
            if k == 0:
                world_action[k, ACC] = (velocities[0] - vel_world) / dt
            else:
                world_action[k, ACC] = (velocities[k] - velocities[k-1]) / dt
        
        return world_action
    
    # ═══════════════════════════════════════════════════════════
    # 控制循环 50Hz
    # ═══════════════════════════════════════════════════════════
    def control_loop(self, event):
        if self.state in ("WAIT_OFFBOARD", "START_MISSION", "INIT"):
            return
        if not self.has_goal:
            return
        with self.data_lock:
            pos = self.current_position.copy()
            goal = self.goal_position.copy()
        if np.linalg.norm(pos - goal) < self.goal_tolerance:
            self._on_goal_reached(); return
        
        need_plan = not self.traj_mgr.is_valid()
        if self.traj_mgr.is_valid():
            r = self.traj_mgr.sample(rospy.Time.now())
            if r and r[5] >= self.replan_ratio:
                need_plan = True
        if need_plan:
            self._try_start_planning()
        
        if self.traj_mgr.is_valid():
            r = self.traj_mgr.sample(rospy.Time.now())
            if r:
                self._publish_cmd(r[0], r[1], r[2], r[3], r[4], "traj")
                return
        self._publish_hold_cmd()
    
    # ───────── 规划线程 ─────────
    def _try_start_planning(self):
        with self.planning_lock:
            if self.is_planning: return
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
    
    # ═══════════════════════════════════════════════════════════
    # ⭐ 核心修改3: 规划流程 (构建obs → 推理 → 坐标转换 → 发布)
    # ═══════════════════════════════════════════════════════════
    def _do_plan(self):
        with self.data_lock:
            if self.current_pointcloud is None:
                rospy.logwarn_throttle(2, "等待点云..."); return
            cur_pos = self.current_position.copy()
            cur_vel_w = self.current_velocity_world.copy()
            cur_orient_xyzw = self.current_orientation_xyzw.copy()
            goal_w = self.goal_position.copy()
            pc = self.current_pointcloud.copy()
        
        # ⭐ 构建 agent_pos[13] (匹配训练数据格式)
        agent_pos, goal_rel_b = self._build_agent_pos(
            cur_pos, cur_vel_w, cur_orient_xyzw, goal_w)
        
        obs = {'point_cloud': pc, 'agent_pos': agent_pos}
        self.obs_history.append(obs)
        
        if len(self.obs_history) < self.n_obs_steps:
            rospy.loginfo_throttle(1, f"积累观测: {len(self.obs_history)}/{self.n_obs_steps}")
            return
        
        # 组装 batch
        pc_s = np.stack([o['point_cloud'] for o in self.obs_history])
        ag_s = np.stack([o['agent_pos'] for o in self.obs_history])
        obs_t = {
            'point_cloud': torch.from_numpy(pc_s).unsqueeze(0).to(self.device),
            'agent_pos': torch.from_numpy(ag_s).unsqueeze(0).to(self.device)}
        
        # 推理
        t0 = time.time()
        with torch.no_grad():
            raw_action = self._call_policy(obs_t).cpu().numpy()[0]  # (15, 11) 机体坐标
        dt_ms = (time.time() - t0) * 1000
        
        # ⭐ 核心: 机体坐标 → 世界坐标
        quat_wxyz = self._ros_to_wxyz(cur_orient_xyzw)
        world_action = self._body_to_world(raw_action, cur_pos, cur_vel_w, quat_wxyz)
        
        # 日志
        self._log_diffusion_output(raw_action, world_action, dt_ms, 
                                   cur_pos, goal_w, goal_rel_b)
        self.csv_logger.log_diffusion(raw_action, world_action, dt_ms,
                                      cur_pos, self.current_yaw, goal_w, goal_rel_b)
        
        # ⭐ 设置世界坐标轨迹 (不再是机体相对值!)
        self.traj_mgr.set_trajectory(world_action, self.traj_dt)
        
        # RViz可视化 (世界坐标)
        self._publish_traj_marker(world_action)
        self._publish_detail_markers(world_action, raw_action)
    
    # ═══════════════════════════════════════════════════════════
    # 日志: 同时显示原始输出和转换结果
    # ═══════════════════════════════════════════════════════════
    def _log_diffusion_output(self, raw_action, world_action, dt_ms,
                              cur_pos, goal_w, goal_rel_b):
        rospy.loginfo("=" * 105)
        rospy.loginfo(f"[DIFFUSION] 推理{dt_ms:.0f}ms | raw_shape {raw_action.shape}")
        rospy.loginfo(f"  当前 pos_w: [{cur_pos[0]:.3f}, {cur_pos[1]:.3f}, {cur_pos[2]:.3f}]  "
                     f"yaw: {self.current_yaw:.3f}")
        rospy.loginfo(f"  目标 goal_w: [{goal_w[0]:.3f}, {goal_w[1]:.3f}, {goal_w[2]:.3f}]  "
                     f"goal_b: [{goal_rel_b[0]:.3f}, {goal_rel_b[1]:.3f}, {goal_rel_b[2]:.3f}]  "
                     f"dist: {np.linalg.norm(goal_rel_b):.2f}m")
        
        # 原始机体输出
        rospy.loginfo("-" * 105)
        rospy.loginfo("  [机体坐标 原始输出]")
        h = f"{'t':>3}"
        for n in DIM_NAMES_RAW[:6]:
            h += f" {n:>8}"
        h += f"  {'|dp|':>6}"
        rospy.loginfo(h)
        for i in range(min(5, len(raw_action))):
            row = f" {i:>2}"
            for d in range(6):
                row += f" {raw_action[i,d]:>8.4f}"
            row += f"  {np.linalg.norm(raw_action[i,DPOS_B]):>6.4f}"
            rospy.loginfo(row)
        if len(raw_action) > 5:
            rospy.loginfo(f"  ... ({len(raw_action)} 步)")
        
        # 世界坐标转换结果
        rospy.loginfo("-" * 105)
        rospy.loginfo("  [世界坐标 转换后] (发给PX4Ctrl)")
        h = f"{'t':>3}"
        for n in DIM_NAMES_WORLD[:6]:
            h += f" {n:>8}"
        h += f"  {'d_cur':>7} {'d_goal':>7}"
        rospy.loginfo(h)
        for i in range(len(world_action)):
            row = f" {i:>2}"
            for d in range(6):
                row += f" {world_action[i,d]:>8.3f}"
            d_cur = np.linalg.norm(world_action[i, POS] - cur_pos)
            d_goal = np.linalg.norm(world_action[i, POS] - goal_w)
            row += f"  {d_cur:>6.2f}m {d_goal:>6.2f}m"
            rospy.loginfo(row)
        
        rospy.loginfo("-" * 105)
        total = sum(np.linalg.norm(world_action[i+1,POS]-world_action[i,POS])
                    for i in range(len(world_action)-1))
        rospy.loginfo(f"  轨迹长{total:.2f}m | "
                     f"终点距当前{np.linalg.norm(world_action[-1,POS]-cur_pos):.2f}m | "
                     f"终点距目标{np.linalg.norm(world_action[-1,POS]-goal_w):.2f}m")
        rospy.loginfo("=" * 105)
    
    # ───────── 命令发布+日志 ─────────
    def _publish_cmd(self, pos, vel, acc, yaw, yr, source="traj"):
        cmd = PositionCommand()
        cmd.header.stamp = rospy.Time.now()
        cmd.header.frame_id = "world"
        cmd.position.x, cmd.position.y, cmd.position.z = float(pos[0]), float(pos[1]), float(pos[2])
        cmd.velocity.x, cmd.velocity.y, cmd.velocity.z = float(vel[0]), float(vel[1]), float(vel[2])
        cmd.acceleration.x, cmd.acceleration.y, cmd.acceleration.z = float(acc[0]), float(acc[1]), float(acc[2])
        cmd.yaw, cmd.yaw_dot = float(yaw), float(yr)
        cmd.trajectory_flag = PositionCommand.TRAJECTORY_STATUS_READY
        cmd.trajectory_id = self.cmd_count % 100 + 1
        cmd.kx, cmd.kv = self.pos_gain, self.vel_gain
        self.cmd_pub.publish(cmd)
        self.cmd_count += 1
        
        with self.data_lock: cur = self.current_position.copy()
        self.csv_logger.log_cmd(self.cmd_count, source, pos, vel, acc, yaw, yr, cur)
        
        if self.cmd_count % 50 == 0:
            err = np.linalg.norm(np.array(pos[:3]) - cur)
            rospy.loginfo(
                f"[CMD#{self.cmd_count}] {source:>4} | "
                f"pos[{pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}] "
                f"vel[{vel[0]:.2f},{vel[1]:.2f},{vel[2]:.2f}] "
                f"acc[{acc[0]:.2f},{acc[1]:.2f},{acc[2]:.2f}] "
                f"yaw={yaw:.2f} | 实际[{cur[0]:.2f},{cur[1]:.2f},{cur[2]:.2f}] err={err:.2f}m")
    
    def _publish_hold_cmd(self):
        hp, hy = self.traj_mgr.get_hold_state()
        if hp is None:
            with self.data_lock: hp, hy = self.current_position.copy(), self.current_yaw
        self._publish_cmd(hp, np.zeros(3), np.zeros(3), hy, 0.0, "hold")
    
    # ═══════════════════════════════════════════════════════════
    # RViz可视化（世界坐标）
    # ═══════════════════════════════════════════════════════════
    def _publish_goal_marker(self):
        if self.goal_position is None: return
        m = Marker()
        m.header.stamp = rospy.Time.now(); m.header.frame_id = "world"
        m.ns = "goal_point"; m.id = 0
        m.type = Marker.SPHERE; m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = \
            self.goal_position[0], self.goal_position[1], self.goal_position[2]
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.5
        m.color = ColorRGBA(1.0, 0.0, 0.0, 0.8)
        self.goal_pub.publish(m)
    
    def _publish_traj_marker(self, world_action):
        """世界坐标轨迹线"""
        m = Marker()
        m.header.stamp = rospy.Time.now(); m.header.frame_id = "world"
        m.ns = "optimal_trajectory"; m.id = 0
        m.type = Marker.LINE_STRIP; m.action = Marker.ADD
        m.scale.x = 0.08
        m.color = ColorRGBA(0.1, 0.5, 1.0, 0.9)
        m.pose.orientation.w = 1.0
        for i in range(len(world_action)):
            m.points.append(Point(
                x=float(world_action[i, 0]),
                y=float(world_action[i, 1]),
                z=float(world_action[i, 2])))
        self.traj_pub.publish(m)
    
    def _publish_detail_markers(self, world_action, raw_action):
        """详细可视化：轨迹点 + 速度箭头 + |dpos| 标注"""
        ma = MarkerArray()
        now = rospy.Time.now()
        
        clr = Marker(); clr.action = Marker.DELETEALL
        ma.markers.append(clr)
        
        for i in range(len(world_action)):
            ratio = i / max(1, len(world_action)-1)
            # 轨迹点 (世界坐标)
            s = Marker()
            s.header.stamp = now; s.header.frame_id = "world"
            s.ns = "dp"; s.id = i; s.type = Marker.SPHERE; s.action = Marker.ADD
            s.pose.position.x = float(world_action[i, 0])
            s.pose.position.y = float(world_action[i, 1])
            s.pose.position.z = float(world_action[i, 2])
            s.pose.orientation.w = 1.0
            s.scale.x = s.scale.y = s.scale.z = 0.12
            s.color = ColorRGBA(ratio, 1-ratio, 0.2, 0.8)
            ma.markers.append(s)
            
            # 速度箭头 (世界坐标, 每3步)
            if i % 3 == 0:
                v = world_action[i, VEL]; spd = np.linalg.norm(v)
                if spd > 0.01:
                    a = Marker()
                    a.header.stamp = now; a.header.frame_id = "world"
                    a.ns = "dv"; a.id = 100+i; a.type = Marker.ARROW; a.action = Marker.ADD
                    a.points = [
                        Point(x=float(world_action[i,0]),
                              y=float(world_action[i,1]),
                              z=float(world_action[i,2])),
                        Point(x=float(world_action[i,0]+v[0]*0.5),
                              y=float(world_action[i,1]+v[1]*0.5),
                              z=float(world_action[i,2]+v[2]*0.5))]
                    a.scale.x = 0.03; a.scale.y = 0.06
                    a.color = ColorRGBA(1.0, 1.0, 0.0, 0.7)
                    ma.markers.append(a)
            
            # 标注: 步号 + 原始|dpos|
            if i == 0 or i == len(world_action)-1 or i % 5 == 0:
                t = Marker()
                t.header.stamp = now; t.header.frame_id = "world"
                t.ns = "dt"; t.id = 200+i; t.type = Marker.TEXT_VIEW_FACING; t.action = Marker.ADD
                t.pose.position.x = float(world_action[i, 0])
                t.pose.position.y = float(world_action[i, 1])
                t.pose.position.z = float(world_action[i, 2]) + 0.2
                t.scale.z = 0.15
                t.color = ColorRGBA(1,1,1,0.9)
                dp = np.linalg.norm(raw_action[i, DPOS_B])
                t.text = f"t{i} dp={dp:.2f}"
                ma.markers.append(t)
        
        self.detail_pub.publish(ma)
    
    def _vis_tick(self, event):
        self._publish_goal_marker()
    
    # ───────── 航点 ─────────
    def _on_goal_reached(self):
        rospy.loginfo("[ML Planner] 到达目标!")
        if self.flight_mode == 0: self._set_next_waypoint()
        else: self.has_goal = False
    
    def _set_next_waypoint(self):
        if self.waypoint_index >= len(self.waypoints):
            rospy.loginfo("[ML Planner] 所有航点完成!"); self.has_goal = False; return
        self.goal_position = self.waypoints[self.waypoint_index].copy()
        self.has_goal = True; self.obs_history.clear()
        self.state = "EXECUTING"
        rospy.loginfo(f"[ML Planner] 航点{self.waypoint_index}: {self.goal_position}")
        self._publish_goal_marker()
        self.waypoint_index += 1
    
    # ───────── 工具函数 ─────────
    @staticmethod
    def _ros_to_wxyz(quat_xyzw):
        """ROS [x,y,z,w] → 训练数据 [w,x,y,z]"""
        return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    
    @staticmethod
    def _quat_to_yaw(q_xyzw):
        """ROS四元数 [x,y,z,w] → 偏航角"""
        x, y, z, w = q_xyzw
        return np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))


def main():
    try:
        MLPlannerNode(); rospy.spin()
    except rospy.ROSInterruptException: pass
    except Exception as e:
        rospy.logerr(f"异常: {e}"); traceback.print_exc()

if __name__ == '__main__':
    main()