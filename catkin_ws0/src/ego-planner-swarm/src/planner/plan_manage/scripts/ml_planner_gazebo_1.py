#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML Planner Node - 扩散模型 + 轨迹插值 + EGO兼容可视化 + 详细日志

架构（对标EGO）：
  planning_thread(异步推理) → TrajectoryManager(时间插值) → 50Hz pos_cmd → PX4Ctrl

RViz兼容：发布到EGO相同的话题名，可直接用EGO的default_test.rviz

日志系统：
  - 终端：每次规划打印完整15步×11维数值表
  - 终端：每50条PositionCommand打印pos/vel/acc/yaw及跟踪误差
  - CSV文件：~/ml_planner_log/ 下保存完整记录供离线分析
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
# Action维度定义 (训练数据确认)
# ═══════════════════════════════════════════════════════════════
# action[11] = pos(3) + vel(3) + acc(3) + yaw(1) + yaw_rate(1)
POS = slice(0, 3)
VEL = slice(3, 6)
ACC = slice(6, 9)
YAW = 9
YAW_RATE = 10
DIM_NAMES = ["pos_x","pos_y","pos_z","vel_x","vel_y","vel_z",
             "acc_x","acc_y","acc_z","yaw","yaw_rt"]


# ═══════════════════════════════════════════════════════════════
# CSV日志器 — 离线分析扩散输出和控制命令
# ═══════════════════════════════════════════════════════════════
class CSVLogger:
    def __init__(self):
        log_dir = os.path.expanduser("~/ml_planner_log")
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.diff_path = os.path.join(log_dir, f"diffusion_{ts}.csv")
        self.diff_file = open(self.diff_path, 'w', newline='')
        self.diff_w = csv.writer(self.diff_file)
        self.diff_w.writerow(
            ["plan_id","step","ros_time","inference_ms",
             "cur_x","cur_y","cur_z","goal_x","goal_y","goal_z"]
            + [f"act_{n}" for n in DIM_NAMES])
        
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
    
    def log_diffusion(self, action, inference_ms, cur_pos, goal_pos):
        self.plan_count += 1
        t = rospy.Time.now().to_sec()
        for s in range(len(action)):
            self.diff_w.writerow(
                [self.plan_count, s, f"{t:.3f}", f"{inference_ms:.0f}"]
                + [f"{v:.4f}" for v in cur_pos]
                + [f"{v:.4f}" for v in goal_pos]
                + [f"{action[s,d]:.6f}" for d in range(action.shape[1])])
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
# 轨迹管理器（模仿EGO的traj_server时间采样）
# ═══════════════════════════════════════════════════════════════
class TrajectoryManager:
    def __init__(self):
        self.lock = Lock()
        self.waypoints = None
        self.start_time = None
        self.traj_dt = 0.1
        self.duration = 0.0
        self.last_pos = None
        self.last_yaw = 0.0
    
    def set_trajectory(self, action_array, traj_dt):
        with self.lock:
            self.waypoints = action_array.copy()
            self.start_time = rospy.Time.now()
            self.traj_dt = traj_dt
            self.duration = (len(action_array) - 1) * traj_dt
            self.last_pos = action_array[-1, POS].copy()
            self.last_yaw = float(action_array[-1, YAW])
    
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
            yr = float(wp0[YAW_RATE]*(1-frac) + wp1[YAW_RATE]*frac)
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
        rospy.loginfo("  ML Planner - 扩散模型 + EGO兼容RViz + 日志")
        rospy.loginfo("=" * 70)
        
        self.load_parameters()
        
        self.data_lock = Lock()
        self.current_position = np.array([0.0, 0.0, 2.0])
        self.current_velocity = np.zeros(3)
        self.current_orientation = np.array([0, 0, 0, 1.0])
        self.current_yaw = 0.0
        self.current_pointcloud = None
        self.goal_position = None
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
            rospy.loginfo("[ML Planner] 模型加载成功!")
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
        
        # ═══ 关键修正：使用EGO的话题名 → 直接兼容default_test.rviz ═══
        ego = f"/drone_{self.drone_id}_ego_planner_node"
        self.goal_pub = rospy.Publisher(f"{ego}/goal_point", Marker, queue_size=10)
        self.traj_pub = rospy.Publisher(f"{ego}/optimal_list", Marker, queue_size=10)
        
        # ML专用详细可视化（RViz中手动Add MarkerArray订阅此话题）
        ml = f"/drone_{self.drone_id}_ml_planner"
        self.detail_pub = rospy.Publisher(f"{ml}/diffusion_detail", MarkerArray, queue_size=10)
        
        rospy.sleep(0.5)
        rospy.loginfo(f"[RViz话题] 目标: {ego}/goal_point  轨迹: {ego}/optimal_list")
        rospy.loginfo(f"[RViz话题] 详细: {ml}/diffusion_detail")
        rospy.loginfo(f"[提示] 直接使用EGO的default_test.rviz即可看到目标和轨迹!")
    
    # ───────── 回调 ─────────
    def odom_cb(self, msg):
        with self.data_lock:
            self.current_position = np.array([
                msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
            self.current_velocity = np.array([
                msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])
            self.current_orientation = np.array([
                msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
            self.current_yaw = self._quat_to_yaw(self.current_orientation)
            if self.state == "INIT":
                self.state = "WAIT_OFFBOARD"
                rospy.loginfo(f"[ML Planner] 位置: {self.current_position}")
    
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
            with self.data_lock:
                self.current_pointcloud = pts
        except Exception as e:
            rospy.logwarn_throttle(5, f"点云错误: {e}")
    
    # ═══════════════════════════════════════════════════════════
    # 控制循环 50Hz（模仿traj_server）
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
    
    def _do_plan(self):
        with self.data_lock:
            if self.current_pointcloud is None:
                rospy.logwarn_throttle(2, "等待点云..."); return
            cur_pos = self.current_position.copy()
            goal = self.goal_position.copy()
            agent_pos = np.concatenate([
                cur_pos, self.current_velocity,
                self.current_orientation, goal]).astype(np.float32)
            obs = {'point_cloud': self.current_pointcloud.copy(), 'agent_pos': agent_pos}
        
        self.obs_history.append(obs)
        if len(self.obs_history) < self.n_obs_steps:
            rospy.loginfo_throttle(1, f"积累观测: {len(self.obs_history)}/{self.n_obs_steps}")
            return
        
        pc_s = np.stack([o['point_cloud'] for o in self.obs_history])
        ag_s = np.stack([o['agent_pos'] for o in self.obs_history])
        obs_t = {
            'point_cloud': torch.from_numpy(pc_s).unsqueeze(0).to(self.device),
            'agent_pos': torch.from_numpy(ag_s).unsqueeze(0).to(self.device)}
        
        t0 = time.time()
        with torch.no_grad():
            action = self._call_policy(obs_t).cpu().numpy()[0]  # (15,11)
        dt_ms = (time.time() - t0) * 1000
        
        # ⭐ 详细终端日志（TrajectoryManager之前的原始输出）
        self._log_diffusion_output(action, dt_ms, cur_pos, goal)
        self.csv_logger.log_diffusion(action, dt_ms, cur_pos, goal)
        
        # 设置轨迹
        self.traj_mgr.set_trajectory(action, self.traj_dt)
        
        # RViz可视化
        self._publish_traj_marker(action)
        self._publish_detail_markers(action)
    
    # ═══════════════════════════════════════════════════════════
    # ⭐ 扩散模型输出详细日志
    # ═══════════════════════════════════════════════════════════
    def _log_diffusion_output(self, action, dt_ms, cur_pos, goal):
        rospy.loginfo("=" * 95)
        rospy.loginfo(f"[DIFFUSION] 推理{dt_ms:.0f}ms | shape {action.shape} | "
                     f"轨迹{(len(action)-1)*self.traj_dt:.1f}s")
        rospy.loginfo(f"  当前: [{cur_pos[0]:.3f}, {cur_pos[1]:.3f}, {cur_pos[2]:.3f}]  "
                     f"目标: [{goal[0]:.3f}, {goal[1]:.3f}, {goal[2]:.3f}]  "
                     f"距离: {np.linalg.norm(goal-cur_pos):.2f}m")
        rospy.loginfo("-" * 95)
        h = f"{'t':>3}"
        for n in DIM_NAMES:
            h += f" {n:>8}"
        h += f" {'|d_cur':>7}"
        rospy.loginfo(h)
        rospy.loginfo("-" * 95)
        for i in range(len(action)):
            row = f" {i:>2}"
            for d in range(min(11, action.shape[1])):
                row += f" {action[i,d]:>8.3f}"
            row += f" {np.linalg.norm(action[i,POS]-cur_pos):>7.2f}m"
            rospy.loginfo(row)
        rospy.loginfo("-" * 95)
        total = sum(np.linalg.norm(action[i+1,POS]-action[i,POS]) for i in range(len(action)-1))
        rospy.loginfo(f"  轨迹长{total:.2f}m | "
                     f"速度[{action[:,3:6].min():.2f}~{action[:,3:6].max():.2f}]m/s | "
                     f"加速度[{action[:,6:9].min():.2f}~{action[:,6:9].max():.2f}]m/s²")
        rospy.loginfo("=" * 95)
    
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
    # RViz可视化（兼容EGO的default_test.rviz）
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
    
    def _publish_traj_marker(self, action):
        """单个Marker LINE_STRIP → EGO RViz的optimal_list (rviz/Marker类型)"""
        m = Marker()
        m.header.stamp = rospy.Time.now(); m.header.frame_id = "world"
        m.ns = "optimal_trajectory"; m.id = 0
        m.type = Marker.LINE_STRIP; m.action = Marker.ADD
        m.scale.x = 0.08
        m.color = ColorRGBA(0.1, 0.5, 1.0, 0.9)
        m.pose.orientation.w = 1.0
        for i in range(len(action)):
            m.points.append(Point(x=float(action[i,0]), y=float(action[i,1]), z=float(action[i,2])))
        self.traj_pub.publish(m)
    
    def _publish_detail_markers(self, action):
        """详细可视化：轨迹点(渐变色) + 速度箭头(黄色) + 步号标注"""
        ma = MarkerArray()
        now = rospy.Time.now()
        
        clr = Marker(); clr.action = Marker.DELETEALL
        ma.markers.append(clr)
        
        for i in range(len(action)):
            ratio = i / max(1, len(action)-1)
            # 轨迹点
            s = Marker()
            s.header.stamp = now; s.header.frame_id = "world"
            s.ns = "dp"; s.id = i; s.type = Marker.SPHERE; s.action = Marker.ADD
            s.pose.position.x, s.pose.position.y, s.pose.position.z = \
                float(action[i,0]), float(action[i,1]), float(action[i,2])
            s.pose.orientation.w = 1.0
            s.scale.x = s.scale.y = s.scale.z = 0.12
            s.color = ColorRGBA(ratio, 1-ratio, 0.2, 0.8)
            ma.markers.append(s)
            
            # 速度箭头(每3步)
            if i % 3 == 0:
                v = action[i, VEL]; spd = np.linalg.norm(v)
                if spd > 0.01:
                    a = Marker()
                    a.header.stamp = now; a.header.frame_id = "world"
                    a.ns = "dv"; a.id = 100+i; a.type = Marker.ARROW; a.action = Marker.ADD
                    a.points = [
                        Point(x=float(action[i,0]), y=float(action[i,1]), z=float(action[i,2])),
                        Point(x=float(action[i,0]+v[0]*0.5), y=float(action[i,1]+v[1]*0.5),
                              z=float(action[i,2]+v[2]*0.5))]
                    a.scale.x = 0.03; a.scale.y = 0.06
                    a.color = ColorRGBA(1.0, 1.0, 0.0, 0.7)
                    ma.markers.append(a)
            
            # 标注(首、尾、每5步)
            if i == 0 or i == len(action)-1 or i % 5 == 0:
                t = Marker()
                t.header.stamp = now; t.header.frame_id = "world"
                t.ns = "dt"; t.id = 200+i; t.type = Marker.TEXT_VIEW_FACING; t.action = Marker.ADD
                t.pose.position.x, t.pose.position.y = float(action[i,0]), float(action[i,1])
                t.pose.position.z = float(action[i,2]) + 0.2
                t.scale.z = 0.15
                t.color = ColorRGBA(1,1,1,0.9)
                t.text = f"t{i} v={np.linalg.norm(action[i,VEL]):.1f}"
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
        self.state = "EXECUTING"  # ⭐ 关键！否则control_loop一直return
        rospy.loginfo(f"[ML Planner] 航点{self.waypoint_index}: {self.goal_position}")
        self._publish_goal_marker()
        self.waypoint_index += 1
    
    @staticmethod
    def _quat_to_yaw(q):
        return np.arctan2(2*(q[3]*q[2]+q[0]*q[1]), 1-2*(q[1]**2+q[2]**2))


def main():
    try:
        MLPlannerNode(); rospy.spin()
    except rospy.ROSInterruptException: pass
    except Exception as e:
        rospy.logerr(f"异常: {e}"); traceback.print_exc()

if __name__ == '__main__':
    main()