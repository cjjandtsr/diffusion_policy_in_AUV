#!/usr/bin/env python3
"""
扩散模型推理节点 - 替代 EGO-Planner
=====================================

目录结构:
  模型代码: /home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy/
  模型权重: /home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy/data/outputs/2026.01.30/11.36.24_train_diffusion_unet_hybrid_drone_trajectory/checkpoints/latest.ckpt
  ROS包:    /home/cjj/AUV/catkin_ws0/src/ego-planner-swarm/src/planner/plan_manage/

数据格式 (与训练时完全一致):
  - state:      10维 [pos(3), vel(3), quat(4)]  quat顺序: w,x,y,z
  - goal:       3维  [x, y, z]
  - agent_pos:  13维 = state(10) + goal(3) 拼接
  - action:     11维 [pos(3), vel(3), acc(3), yaw(1), yaw_dot(1)]
  - point_cloud: [4096, 3] 只有xyz

模型配置 (来自idp3.yaml):
  - n_obs_steps: 2
  - horizon: 16  
  - n_action_steps: 15
  - num_inference_steps: 10
"""

import sys
import os

# ══════════════════════════════════════════════════════════════
# 路径配置 - 你的实际目录
# ══════════════════════════════════════════════════════════════
DIFFUSION_POLICY_PATH = "/home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy"
sys.path.insert(0, DIFFUSION_POLICY_PATH)

import rospy
import torch
import numpy as np
import threading
from collections import deque

from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs.point_cloud2 as pc2

# quadrotor_msgs
try:
    from quadrotor_msgs.msg import PositionCommand
    HAS_QUADROTOR_MSGS = True
except ImportError:
    HAS_QUADROTOR_MSGS = False
    rospy.logwarn("[MLPlanner] quadrotor_msgs not found, using PoseStamped fallback")

import hydra
from omegaconf import OmegaConf
OmegaConf.register_new_resolver("eval", eval, replace=True)


# ══════════════════════════════════════════════════════════════
# 参数配置 (必须与训练时完全一致)
# ══════════════════════════════════════════════════════════════

# === ROS话题 (和数据采集时一致) ===
TOPIC_POINTCLOUD = "/iris_D435i/realsense/depth_camera/depth/points"
TOPIC_ODOM       = "/mavros/local_position/odom"
TOPIC_POS_CMD    = "/planning/pos_cmd"
TOPIC_GOAL       = "/move_base_simple/goal"
TOPIC_TRAJ_VIS   = "/planning/optimal_list"

# === 模型参数 (来自idp3.yaml和drone_trajectory.yaml) ===
NUM_POINTS      = 4096    # 点云点数
OBS_HORIZON     = 2       # n_obs_steps
ACTION_HORIZON  = 15      # n_action_steps
STATE_DIM       = 10      # [pos(3), vel(3), quat(4)]
GOAL_DIM        = 3       # [x, y, z]
AGENT_POS_DIM   = 13      # state + goal = 10 + 3
ACTION_DIM      = 11      # [pos(3), vel(3), acc(3), yaw(1), yaw_dot(1)]

# === 执行参数 ===
EXEC_HZ         = 100     # 执行频率 (和traj_server一致)
DT              = 1.0 / EXEC_HZ
REPLAN_RATIO    = 0.7     # 执行到70%时提前推理下一条轨迹
GOAL_TOLERANCE  = 0.5     # 到达目标判定距离(m)
COLLISION_DIST  = 0.5     # 碰撞检测距离(m)


# ══════════════════════════════════════════════════════════════
# ML规划器节点
# ══════════════════════════════════════════════════════════════
class MLPlannerNode:
    def __init__(self, policy, device):
        self.policy = policy
        self.device = device

        # ── 数据缓冲 ──
        self.pc_buffer    = deque(maxlen=OBS_HORIZON)    # 点云历史
        self.state_buffer = deque(maxlen=OBS_HORIZON)    # 状态历史
        self.lock         = threading.Lock()
        self.data_ready   = False

        # ── 目标管理 ──
        self.goal_pos     = None       # 当前目标点 (3,)
        self.have_goal    = False
        self.goal_reached = False
        self.current_pos  = np.zeros(3)

        # ── 轨迹执行 ──
        self.current_traj = None       # 当前执行的轨迹
        self.next_traj    = None       # 预推理的下一条轨迹
        self.exec_step    = 0          # 当前执行步
        self.is_inferring = False

        # ── ROS订阅 ──
        self.sub_pc   = rospy.Subscriber(TOPIC_POINTCLOUD, PointCloud2, self._cb_pointcloud, queue_size=1)
        self.sub_odom = rospy.Subscriber(TOPIC_ODOM,       Odometry,    self._cb_odom,       queue_size=1)
        self.sub_goal = rospy.Subscriber(TOPIC_GOAL,       PoseStamped, self._cb_goal,       queue_size=1)

        # ── ROS发布 ──
        if HAS_QUADROTOR_MSGS:
            self.pub_cmd = rospy.Publisher(TOPIC_POS_CMD, PositionCommand, queue_size=50)
        else:
            self.pub_cmd = rospy.Publisher(TOPIC_POS_CMD, PoseStamped, queue_size=50)
        self.pub_traj_vis = rospy.Publisher(TOPIC_TRAJ_VIS, MarkerArray, queue_size=1)

        # ── 执行定时器 ──
        self.exec_timer = rospy.Timer(rospy.Duration(DT), self._exec_tick)

        rospy.loginfo("=" * 60)
        rospy.loginfo("[MLPlanner] 初始化完成")
        rospy.loginfo("[MLPlanner]   数据格式: state(%d) + goal(%d) = agent_pos(%d)", 
                     STATE_DIM, GOAL_DIM, AGENT_POS_DIM)
        rospy.loginfo("[MLPlanner]   动作维度: %d, 点云点数: %d", ACTION_DIM, NUM_POINTS)
        rospy.loginfo("[MLPlanner]   执行频率: %dHz, 重规划比例: %.0f%%", EXEC_HZ, REPLAN_RATIO*100)
        rospy.loginfo("=" * 60)

    # ════════════════════════════════════════════════════════════
    # 回调函数
    # ════════════════════════════════════════════════════════════
    
    def _cb_pointcloud(self, msg):
        """点云回调 - 提取xyz坐标并采样到固定点数"""
        try:
            # 解析点云
            pts = np.array(
                list(pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z"))),
                dtype=np.float32
            )
            if len(pts) == 0:
                return
            
            # 采样到固定点数
            pts = self._sample_pointcloud(pts, NUM_POINTS)

            with self.lock:
                self.pc_buffer.append(pts)
                self._check_data_ready()
                
        except Exception as e:
            rospy.logerr("[MLPlanner] 点云处理异常: %s", str(e))

    def _cb_odom(self, msg):
        """
        里程计回调 - 提取10维状态
        格式: [pos(3), vel(3), quat(4)]
        四元数顺序: w, x, y, z (与data_collector.py一致)
        """
        try:
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            v = msg.twist.twist.linear
            
            # 更新当前位置 (用于目标到达检测)
            self.current_pos = np.array([p.x, p.y, p.z])
            
            # 构建10维状态 (完全匹配data_collector.py的_extract_ego_state)
            state = np.array([
                p.x, p.y, p.z,           # 位置 [0:3]
                v.x, v.y, v.z,           # 速度 [3:6]
                q.w, q.x, q.y, q.z       # 四元数 [6:10], w在前
            ], dtype=np.float32)

            with self.lock:
                self.state_buffer.append(state)
                self._check_data_ready()
                
        except Exception as e:
            rospy.logerr("[MLPlanner] 里程计处理异常: %s", str(e))

    def _cb_goal(self, msg):
        """目标点回调 - 来自rviz的2D Nav Goal"""
        self.goal_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ], dtype=np.float32)
        
        # 重置状态
        self.have_goal    = True
        self.goal_reached = False
        self.current_traj = None
        self.next_traj    = None
        self.exec_step    = 0
        
        rospy.loginfo("[MLPlanner] ▶ 新目标: (%.2f, %.2f, %.2f)", 
                     self.goal_pos[0], self.goal_pos[1], self.goal_pos[2])

    def _check_data_ready(self):
        """检查数据缓冲是否就绪"""
        if not self.data_ready:
            if len(self.pc_buffer) >= OBS_HORIZON and len(self.state_buffer) >= OBS_HORIZON:
                self.data_ready = True
                rospy.loginfo("[MLPlanner] ✓ 数据缓冲就绪，等待目标点...")

    # ════════════════════════════════════════════════════════════
    # 模型推理
    # ════════════════════════════════════════════════════════════
    
    def _do_inference(self):
        """执行模型推理"""
        if self.is_inferring:
            return
        if self.goal_pos is None:
            return
            
        self.is_inferring = True
        
        try:
            # 获取数据
            with self.lock:
                pc_list    = list(self.pc_buffer)       # List of (4096, 3)
                state_list = list(self.state_buffer)    # List of (10,)

            # ════════════════════════════════════════════════════
            # 关键: 构建agent_pos = state(10) + goal(3) = 13维
            # ════════════════════════════════════════════════════
            agent_pos_list = []
            for state in state_list:
                # 将目标点拼接在状态后面
                agent_pos = np.concatenate([state, self.goal_pos])  # (10,) + (3,) = (13,)
                agent_pos_list.append(agent_pos)
            
            # 转换为数组
            pc        = np.stack(pc_list, axis=0)          # (2, 4096, 3)
            agent_pos = np.stack(agent_pos_list, axis=0)   # (2, 13)

            # 构建观测字典
            obs_dict = {
                'point_cloud': torch.from_numpy(pc).unsqueeze(0).float().to(self.device),        # (1, 2, 4096, 3)
                'agent_pos':   torch.from_numpy(agent_pos).unsqueeze(0).float().to(self.device), # (1, 2, 13)
            }

            rospy.loginfo("[MLPlanner] 推理中... point_cloud: %s, agent_pos: %s", 
                         str(obs_dict['point_cloud'].shape), 
                         str(obs_dict['agent_pos'].shape))

            # 模型推理
            with torch.no_grad():
                result = self.policy.predict_action(obs_dict)
                traj = result['action'][0].cpu().numpy()  # (n_action_steps, 11)

            rospy.loginfo("[MLPlanner] ✓ 推理完成, 轨迹形状: %s", str(traj.shape))

            # 更新轨迹
            if self.current_traj is None:
                self.current_traj = traj
                self.exec_step = 0
                rospy.loginfo("[MLPlanner] ▶ 开始执行轨迹 (%d步)", len(traj))
            else:
                self.next_traj = traj
                rospy.loginfo("[MLPlanner] ✓ 下一条轨迹已准备")

            # 可视化
            self._publish_traj_visualization(traj)

        except Exception as e:
            rospy.logerr("[MLPlanner] 推理异常: %s", str(e))
            import traceback
            traceback.print_exc()
        finally:
            self.is_inferring = False

    # ════════════════════════════════════════════════════════════
    # 执行定时器 (100Hz)
    # ════════════════════════════════════════════════════════════
    
    def _exec_tick(self, event):
        """100Hz执行定时器"""
        # 1. 检查是否有目标
        if not self.have_goal:
            return
        
        # 2. 检查是否到达目标
        if self._check_goal_reached():
            if not self.goal_reached:
                self.goal_reached = True
                rospy.loginfo("[MLPlanner] ✓✓✓ 到达目标!")
            self._publish_hover_cmd()
            return
        
        # 3. 没有轨迹 → 触发推理
        if self.current_traj is None:
            if self.data_ready and not self.is_inferring:
                threading.Thread(target=self._do_inference, daemon=True).start()
            return
        
        # 4. 检查是否需要切换轨迹/触发重规划
        traj_len = len(self.current_traj)
        switch_step = int(traj_len * REPLAN_RATIO)
        
        if self.exec_step >= switch_step:
            # 切换到预推理的轨迹
            if self.next_traj is not None:
                self.current_traj = self.next_traj
                self.next_traj = None
                self.exec_step = 0
                rospy.loginfo("[MLPlanner] ↻ 轨迹切换")
            
            # 触发下一次推理
            if not self.is_inferring:
                threading.Thread(target=self._do_inference, daemon=True).start()
        
        # 5. 获取当前动作 (越界保护)
        idx = min(self.exec_step, len(self.current_traj) - 1)
        action = self.current_traj[idx]
        
        # 6. 碰撞检测
        if not self._collision_check(action):
            rospy.logwarn("[MLPlanner] ⚠ 碰撞风险! 悬停并重规划")
            self._publish_hover_cmd()
            self.current_traj = None  # 触发重规划
            return
        
        # 7. 发布位置命令
        self._publish_position_cmd(action)
        self.exec_step += 1

    # ════════════════════════════════════════════════════════════
    # 安全检查
    # ════════════════════════════════════════════════════════════
    
    def _check_goal_reached(self):
        """检查是否到达目标"""
        if self.goal_pos is None:
            return False
        dist = np.linalg.norm(self.current_pos - self.goal_pos)
        return dist < GOAL_TOLERANCE

    def _collision_check(self, action):
        """简单碰撞检测: 检查目标位置附近是否有障碍物"""
        if len(self.pc_buffer) == 0:
            return True  # 没有点云数据时不阻止
        
        target_pos = action[0:3]  # 位置命令
        latest_pc = self.pc_buffer[-1]
        
        # 计算点云到目标位置的最近距离
        distances = np.linalg.norm(latest_pc - target_pos, axis=1)
        min_dist = distances.min()
        
        return min_dist > COLLISION_DIST

    # ════════════════════════════════════════════════════════════
    # 发布命令
    # ════════════════════════════════════════════════════════════
    
    def _publish_position_cmd(self, action):
        """
        发布位置命令
        action格式: [pos(3), vel(3), acc(3), yaw(1), yaw_dot(1)] = 11维
        完全匹配data_collector.py的_extract_ego_action
        """
        if HAS_QUADROTOR_MSGS:
            msg = PositionCommand()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "world"
            msg.trajectory_flag = PositionCommand.TRAJECTORY_STATUS_READY
            
            # 位置 [0:3]
            msg.position.x = float(action[0])
            msg.position.y = float(action[1])
            msg.position.z = float(action[2])
            
            # 速度 [3:6]
            msg.velocity.x = float(action[3])
            msg.velocity.y = float(action[4])
            msg.velocity.z = float(action[5])
            
            # 加速度 [6:9]
            msg.acceleration.x = float(action[6])
            msg.acceleration.y = float(action[7])
            msg.acceleration.z = float(action[8])
            
            # 偏航 [9:11]
            msg.yaw     = float(action[9])
            msg.yaw_dot = float(action[10])
            
            self.pub_cmd.publish(msg)
        else:
            # 降级方案: 只发位置
            msg = PoseStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "world"
            msg.pose.position.x = float(action[0])
            msg.pose.position.y = float(action[1])
            msg.pose.position.z = float(action[2])
            msg.pose.orientation.w = 1.0
            self.pub_cmd.publish(msg)

    def _publish_hover_cmd(self):
        """悬停命令 - 保持当前位置"""
        if len(self.state_buffer) == 0:
            return
        
        state = self.state_buffer[-1]  # 最新状态
        
        if HAS_QUADROTOR_MSGS:
            msg = PositionCommand()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "world"
            msg.trajectory_flag = PositionCommand.TRAJECTORY_STATUS_READY
            
            # 保持当前位置
            msg.position.x = float(state[0])
            msg.position.y = float(state[1])
            msg.position.z = float(state[2])
            
            # 速度和加速度为0
            msg.velocity.x = msg.velocity.y = msg.velocity.z = 0.0
            msg.acceleration.x = msg.acceleration.y = msg.acceleration.z = 0.0
            
            # 从四元数计算当前yaw
            w, x, y, z = state[6:10]
            msg.yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
            msg.yaw_dot = 0.0
            
            self.pub_cmd.publish(msg)
        else:
            msg = PoseStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "world"
            msg.pose.position.x = float(state[0])
            msg.pose.position.y = float(state[1])
            msg.pose.position.z = float(state[2])
            msg.pose.orientation.w = 1.0
            self.pub_cmd.publish(msg)

    def _publish_traj_visualization(self, traj):
        """发布轨迹可视化"""
        markers = MarkerArray()
        now = rospy.Time.now()
        
        for i, action in enumerate(traj):
            m = Marker()
            m.header.frame_id = "world"
            m.header.stamp = now
            m.ns = "ml_trajectory"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            
            m.pose.position.x = float(action[0])
            m.pose.position.y = float(action[1])
            m.pose.position.z = float(action[2])
            m.pose.orientation.w = 1.0
            
            m.scale.x = m.scale.y = m.scale.z = 0.1
            
            # 颜色渐变: 蓝→红
            ratio = i / max(len(traj) - 1, 1)
            m.color.r = ratio
            m.color.g = 0.2
            m.color.b = 1.0 - ratio
            m.color.a = 0.9
            
            m.lifetime = rospy.Duration(0.5)
            markers.markers.append(m)
        
        self.pub_traj_vis.publish(markers)

    # ════════════════════════════════════════════════════════════
    # 工具函数
    # ════════════════════════════════════════════════════════════
    
    @staticmethod
    def _sample_pointcloud(pts, num_points):
        """采样点云到指定数量"""
        n = len(pts)
        if n >= num_points:
            idx = np.random.choice(n, num_points, replace=False)
        else:
            idx = np.random.choice(n, num_points, replace=True)
        return pts[idx]


# ══════════════════════════════════════════════════════════════
# 模型加载
# ══════════════════════════════════════════════════════════════

def load_model():
    """加载训练好的扩散模型"""
    config_path = os.path.join(DIFFUSION_POLICY_PATH, "diffusion_policy_3d/config")
    
    rospy.loginfo("[MLPlanner] 配置路径: %s", config_path)
    
    # 检查配置文件
    config_file = os.path.join(config_path, "idp3.yaml")
    if not os.path.exists(config_file):
        rospy.logerr("[MLPlanner] 配置文件不存在: %s", config_file)
        raise FileNotFoundError(f"Config not found: {config_file}")
    
    # 清除之前的hydra实例
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    
    # 加载配置
    with hydra.initialize_config_dir(config_dir=config_path, version_base=None):
        cfg = hydra.compose(config_name="idp3")
    
    OmegaConf.resolve(cfg)
    
    # 实例化workspace并加载模型
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    policy = workspace.get_model()
    
    return policy


# ══════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════

def main():
    rospy.init_node("ml_planner_node", anonymous=False)
    
    rospy.loginfo("=" * 60)
    rospy.loginfo("ML Planner - 扩散模型轨迹规划器")
    rospy.loginfo("=" * 60)
    rospy.loginfo("模型代码路径: %s", DIFFUSION_POLICY_PATH)
    
    # 检查路径
    if not os.path.exists(DIFFUSION_POLICY_PATH):
        rospy.logerr("错误: 模型路径不存在: %s", DIFFUSION_POLICY_PATH)
        rospy.logerr("请修改 DIFFUSION_POLICY_PATH 变量")
        return
    
    # 检查checkpoint
    ckpt_base = os.path.join(DIFFUSION_POLICY_PATH, "data/outputs")
    rospy.loginfo("Checkpoint目录: %s", ckpt_base)
    
    # 加载模型
    rospy.loginfo("加载模型...")
    try:
        policy = load_model()
    except Exception as e:
        rospy.logerr("模型加载失败: %s", str(e))
        import traceback
        traceback.print_exc()
        return
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    policy.eval()
    
    rospy.loginfo("=" * 60)
    rospy.loginfo("模型加载成功!")
    rospy.loginfo("  Device: %s", device)
    rospy.loginfo("  Horizon: %d", policy.horizon)
    rospy.loginfo("  n_obs_steps: %d", policy.n_obs_steps)
    rospy.loginfo("  n_action_steps: %d", policy.n_action_steps)
    rospy.loginfo("=" * 60)

    # 启动节点
    try:
        node = MLPlannerNode(policy, device)
        rospy.loginfo("[MLPlanner] 等待目标点...")
        rospy.loginfo("[MLPlanner] 在rviz中使用 '2D Nav Goal' 发送目标")
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("[MLPlanner] 节点关闭")


if __name__ == "__main__":
    main()