#!/usr/bin/env python3
"""
扩散模型推理节点 - 替代 EGO-Planner
=====================================

修复: Hydra与ROS参数冲突问题
解决方案: 手动初始化配置，直接加载checkpoint文件

数据格式:
  - state:      10维 [pos(3), vel(3), quat(4)]
  - goal:       3维  [x, y, z]
  - agent_pos:  13维 = state(10) + goal(3)
  - action:     11维 [pos(3), vel(3), acc(3), yaw(1), yaw_dot(1)]
  - point_cloud: [4096, 3]
"""

import sys
import os

# ══════════════════════════════════════════════════════════════
# 路径配置 - 修改这里适配你的目录
# ══════════════════════════════════════════════════════════════
DIFFUSION_POLICY_PATH = "/home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy"

# 直接指定checkpoint路径
CHECKPOINT_PATH = "/home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy/data/outputs/2026.01.30/11.36.24_train_diffusion_unet_hybrid_drone_trajectory/checkpoints/latest.ckpt"

sys.path.insert(0, DIFFUSION_POLICY_PATH)

# line-buffering
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
OmegaConf.register_new_resolver("eval", eval, replace=True)

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


# ══════════════════════════════════════════════════════════════
# 参数配置
# ══════════════════════════════════════════════════════════════

# ROS话题
TOPIC_POINTCLOUD = "/iris_D435i/realsense/depth_camera/depth/points"
TOPIC_ODOM       = "/mavros/local_position/odom"
TOPIC_POS_CMD    = "/planning/pos_cmd"
TOPIC_GOAL       = "/move_base_simple/goal"
TOPIC_TRAJ_VIS   = "/planning/optimal_list"

# 模型参数
NUM_POINTS      = 4096
OBS_HORIZON     = 2
STATE_DIM       = 10
GOAL_DIM        = 3
AGENT_POS_DIM   = 13
ACTION_DIM      = 11

# 执行参数
EXEC_HZ         = 100
DT              = 1.0 / EXEC_HZ
REPLAN_RATIO    = 0.7
GOAL_TOLERANCE  = 0.5
COLLISION_DIST  = 0.5


# ══════════════════════════════════════════════════════════════
# ML规划器节点
# ══════════════════════════════════════════════════════════════
class MLPlannerNode:
    def __init__(self, policy, device):
        self.policy = policy
        self.device = device

        self.pc_buffer    = deque(maxlen=OBS_HORIZON)
        self.state_buffer = deque(maxlen=OBS_HORIZON)
        self.lock         = threading.Lock()
        self.data_ready   = False

        self.goal_pos     = None
        self.have_goal    = False
        self.goal_reached = False
        self.current_pos  = np.zeros(3)

        self.current_traj = None
        self.next_traj    = None
        self.exec_step    = 0
        self.is_inferring = False

        # ROS订阅
        self.sub_pc   = rospy.Subscriber(TOPIC_POINTCLOUD, PointCloud2, self._cb_pointcloud, queue_size=1)
        self.sub_odom = rospy.Subscriber(TOPIC_ODOM,       Odometry,    self._cb_odom,       queue_size=1)
        self.sub_goal = rospy.Subscriber(TOPIC_GOAL,       PoseStamped, self._cb_goal,       queue_size=1)

        # ROS发布
        if HAS_QUADROTOR_MSGS:
            self.pub_cmd = rospy.Publisher(TOPIC_POS_CMD, PositionCommand, queue_size=50)
        else:
            self.pub_cmd = rospy.Publisher(TOPIC_POS_CMD, PoseStamped, queue_size=50)
        self.pub_traj_vis = rospy.Publisher(TOPIC_TRAJ_VIS, MarkerArray, queue_size=1)

        # 执行定时器
        self.exec_timer = rospy.Timer(rospy.Duration(DT), self._exec_tick)

        rospy.loginfo("=" * 60)
        rospy.loginfo("[MLPlanner] 初始化完成")
        rospy.loginfo("[MLPlanner]   agent_pos: state(%d) + goal(%d) = %d", STATE_DIM, GOAL_DIM, AGENT_POS_DIM)
        rospy.loginfo("[MLPlanner]   action: %d, point_cloud: %d", ACTION_DIM, NUM_POINTS)
        rospy.loginfo("=" * 60)

    def _cb_pointcloud(self, msg):
        try:
            pts = np.array(
                list(pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z"))),
                dtype=np.float32
            )
            if len(pts) == 0:
                return
            pts = self._sample_pointcloud(pts, NUM_POINTS)
            with self.lock:
                self.pc_buffer.append(pts)
                self._check_data_ready()
        except Exception as e:
            rospy.logerr("[MLPlanner] 点云异常: %s", str(e))

    def _cb_odom(self, msg):
        try:
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            v = msg.twist.twist.linear
            
            self.current_pos = np.array([p.x, p.y, p.z])
            
            # 10维状态: [pos(3), vel(3), quat(4)]
            state = np.array([
                p.x, p.y, p.z,
                v.x, v.y, v.z,
                q.w, q.x, q.y, q.z
            ], dtype=np.float32)

            with self.lock:
                self.state_buffer.append(state)
                self._check_data_ready()
        except Exception as e:
            rospy.logerr("[MLPlanner] odom异常: %s", str(e))

    def _cb_goal(self, msg):
        self.goal_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ], dtype=np.float32)
        
        self.have_goal    = True
        self.goal_reached = False
        self.current_traj = None
        self.next_traj    = None
        self.exec_step    = 0
        
        rospy.loginfo("[MLPlanner] ▶ 新目标: (%.2f, %.2f, %.2f)", 
                     self.goal_pos[0], self.goal_pos[1], self.goal_pos[2])

    def _check_data_ready(self):
        if not self.data_ready:
            if len(self.pc_buffer) >= OBS_HORIZON and len(self.state_buffer) >= OBS_HORIZON:
                self.data_ready = True
                rospy.loginfo("[MLPlanner] ✓ 数据就绪，等待目标点...")

    def _do_inference(self):
        if self.is_inferring or self.goal_pos is None:
            return
        self.is_inferring = True
        
        try:
            with self.lock:
                pc_list    = list(self.pc_buffer)
                state_list = list(self.state_buffer)

            # agent_pos = state(10) + goal(3) = 13维
            agent_pos_list = []
            for state in state_list:
                agent_pos = np.concatenate([state, self.goal_pos])
                agent_pos_list.append(agent_pos)
            
            pc        = np.stack(pc_list, axis=0)
            agent_pos = np.stack(agent_pos_list, axis=0)

            obs_dict = {
                'point_cloud': torch.from_numpy(pc).unsqueeze(0).float().to(self.device),
                'agent_pos':   torch.from_numpy(agent_pos).unsqueeze(0).float().to(self.device),
            }

            rospy.loginfo("[MLPlanner] 推理... pc:%s, agent_pos:%s", 
                         str(obs_dict['point_cloud'].shape), 
                         str(obs_dict['agent_pos'].shape))

            # 推理
            with torch.no_grad():
                result = self.policy(obs_dict)
                # 处理不同的返回格式
                if isinstance(result, dict):
                    action = result.get('action', result.get('actions', None))
                    if action is not None:
                        action = action[0]
                elif isinstance(result, tuple):
                    action = result[0]
                else:
                    action = result
                
                if isinstance(action, torch.Tensor):
                    traj = action.cpu().numpy()
                else:
                    traj = np.array([a.cpu().numpy() if isinstance(a, torch.Tensor) else a for a in action])

            rospy.loginfo("[MLPlanner] ✓ 推理完成, 轨迹: %s", str(traj.shape))

            if self.current_traj is None:
                self.current_traj = traj
                self.exec_step = 0
                rospy.loginfo("[MLPlanner] ▶ 执行轨迹 (%d步)", len(traj))
            else:
                self.next_traj = traj

            self._publish_traj_visualization(traj)

        except Exception as e:
            rospy.logerr("[MLPlanner] 推理异常: %s", str(e))
            import traceback
            traceback.print_exc()
        finally:
            self.is_inferring = False

    def _exec_tick(self, event):
        if not self.have_goal:
            return
        
        if self._check_goal_reached():
            if not self.goal_reached:
                self.goal_reached = True
                rospy.loginfo("[MLPlanner] ✓✓✓ 到达目标!")
            self._publish_hover_cmd()
            return
        
        if self.current_traj is None:
            if self.data_ready and not self.is_inferring:
                threading.Thread(target=self._do_inference, daemon=True).start()
            return
        
        traj_len = len(self.current_traj)
        switch_step = int(traj_len * REPLAN_RATIO)
        
        if self.exec_step >= switch_step:
            if self.next_traj is not None:
                self.current_traj = self.next_traj
                self.next_traj = None
                self.exec_step = 0
            if not self.is_inferring:
                threading.Thread(target=self._do_inference, daemon=True).start()
        
        idx = min(self.exec_step, len(self.current_traj) - 1)
        action = self.current_traj[idx]
        
        if not self._collision_check(action):
            rospy.logwarn("[MLPlanner] ⚠ 碰撞风险!")
            self._publish_hover_cmd()
            self.current_traj = None
            return
        
        self._publish_position_cmd(action)
        self.exec_step += 1

    def _check_goal_reached(self):
        if self.goal_pos is None:
            return False
        return np.linalg.norm(self.current_pos - self.goal_pos) < GOAL_TOLERANCE

    def _collision_check(self, action):
        if len(self.pc_buffer) == 0:
            return True
        target_pos = action[0:3]
        latest_pc = self.pc_buffer[-1]
        distances = np.linalg.norm(latest_pc - target_pos, axis=1)
        return distances.min() > COLLISION_DIST

    def _publish_position_cmd(self, action):
        if HAS_QUADROTOR_MSGS:
            msg = PositionCommand()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "world"
            msg.trajectory_flag = PositionCommand.TRAJECTORY_STATUS_READY
            msg.position.x, msg.position.y, msg.position.z = float(action[0]), float(action[1]), float(action[2])
            msg.velocity.x, msg.velocity.y, msg.velocity.z = float(action[3]), float(action[4]), float(action[5])
            msg.acceleration.x, msg.acceleration.y, msg.acceleration.z = float(action[6]), float(action[7]), float(action[8])
            msg.yaw = float(action[9])
            msg.yaw_dot = float(action[10])
            self.pub_cmd.publish(msg)
        else:
            msg = PoseStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "world"
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = float(action[0]), float(action[1]), float(action[2])
            msg.pose.orientation.w = 1.0
            self.pub_cmd.publish(msg)

    def _publish_hover_cmd(self):
        if len(self.state_buffer) == 0:
            return
        state = self.state_buffer[-1]
        if HAS_QUADROTOR_MSGS:
            msg = PositionCommand()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "world"
            msg.trajectory_flag = PositionCommand.TRAJECTORY_STATUS_READY
            msg.position.x, msg.position.y, msg.position.z = float(state[0]), float(state[1]), float(state[2])
            msg.velocity.x = msg.velocity.y = msg.velocity.z = 0.0
            msg.acceleration.x = msg.acceleration.y = msg.acceleration.z = 0.0
            w, x, y, z = state[6:10]
            msg.yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
            msg.yaw_dot = 0.0
            self.pub_cmd.publish(msg)
        else:
            msg = PoseStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "world"
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = float(state[0]), float(state[1]), float(state[2])
            msg.pose.orientation.w = 1.0
            self.pub_cmd.publish(msg)

    def _publish_traj_visualization(self, traj):
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
            m.pose.position.x, m.pose.position.y, m.pose.position.z = float(action[0]), float(action[1]), float(action[2])
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.1
            ratio = i / max(len(traj) - 1, 1)
            m.color.r, m.color.g, m.color.b, m.color.a = ratio, 0.2, 1-ratio, 0.9
            m.lifetime = rospy.Duration(0.5)
            markers.markers.append(m)
        self.pub_traj_vis.publish(markers)

    @staticmethod
    def _sample_pointcloud(pts, num_points):
        n = len(pts)
        if n >= num_points:
            idx = np.random.choice(n, num_points, replace=False)
        else:
            idx = np.random.choice(n, num_points, replace=True)
        return pts[idx]


# ══════════════════════════════════════════════════════════════
# 模型加载 - 直接加载checkpoint，不使用@hydra.main
# ══════════════════════════════════════════════════════════════

def load_policy():
    """
    加载策略模型
    不使用@hydra.main，避免与ROS参数冲突
    """
    config_path = os.path.join(DIFFUSION_POLICY_PATH, "diffusion_policy_3d/config")
    
    print(f"[MLPlanner] 配置路径: {config_path}")
    print(f"[MLPlanner] Checkpoint: {CHECKPOINT_PATH}")
    
    # 检查文件
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    
    # 清除之前的hydra实例
    GlobalHydra.instance().clear()
    
    # 手动初始化hydra配置 (不使用@hydra.main)
    with initialize_config_dir(config_dir=config_path, version_base=None):
        cfg = compose(config_name="idp3")
    
    OmegaConf.resolve(cfg)
    
    # 构建policy
    print("[MLPlanner] 构建模型...")
    policy = hydra.utils.instantiate(cfg.policy)
    
    # 加载checkpoint
    print("[MLPlanner] 加载checkpoint...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    
    # 获取state_dict
    if 'ema_model' in checkpoint:
        print("[MLPlanner] 使用EMA权重")
        state_dict = checkpoint['ema_model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # 处理key前缀 (移除 "model." 前缀)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v
    
    # 加载权重
    missing, unexpected = policy.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"[MLPlanner] Warning - Missing keys: {len(missing)}")
    if unexpected:
        print(f"[MLPlanner] Warning - Unexpected keys: {len(unexpected)}")
    
    print("[MLPlanner] 模型加载成功!")
    return policy


# ══════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════

def main():
    # 过滤ROS参数，避免传给hydra
    # ROS会传递 __name:=xxx __log:=xxx 等参数
    filtered_args = [arg for arg in sys.argv if not arg.startswith('__')]
    sys.argv = filtered_args
    
    # 初始化ROS节点
    rospy.init_node("ml_planner_node", anonymous=False)
    
    rospy.loginfo("=" * 60)
    rospy.loginfo("ML Planner - 扩散模型轨迹规划器")
    rospy.loginfo("=" * 60)
    rospy.loginfo("模型路径: %s", DIFFUSION_POLICY_PATH)
    rospy.loginfo("Checkpoint: %s", CHECKPOINT_PATH)
    
    # 检查路径
    if not os.path.exists(DIFFUSION_POLICY_PATH):
        rospy.logerr("错误: 模型路径不存在!")
        return
    
    if not os.path.exists(CHECKPOINT_PATH):
        rospy.logerr("错误: Checkpoint不存在!")
        return
    
    # 加载模型
    rospy.loginfo("加载模型...")
    try:
        policy = load_policy()
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
    rospy.loginfo("模型就绪!")
    rospy.loginfo("  Device: %s", device)
    rospy.loginfo("  Horizon: %d", policy.horizon)
    rospy.loginfo("  n_obs_steps: %d", policy.n_obs_steps)
    rospy.loginfo("  n_action_steps: %d", policy.n_action_steps)
    rospy.loginfo("=" * 60)

    # 启动规划器节点
    try:
        node = MLPlannerNode(policy, device)
        rospy.loginfo("[MLPlanner] 等待目标点...")
        rospy.loginfo("[MLPlanner] 在rviz中使用 '2D Nav Goal' 发送目标")
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("[MLPlanner] 节点关闭")


if __name__ == "__main__":
    main()