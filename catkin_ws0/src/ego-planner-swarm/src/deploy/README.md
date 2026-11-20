# 扩散模型部署指南 - 替代EGO-Planner

## 📋 概述

用你训练好的扩散模型替代EGO-Planner，在Gazebo仿真环境中进行轨迹规划。

---

## 📁 你的目录结构

```
/home/cjj/AUV/
│
├── Improved-3D-Diffusion-Policy-main/
│   └── Improved-3D-Diffusion-Policy/           # 模型代码根目录
│       ├── diffusion_policy_3d/
│       │   ├── config/
│       │   │   ├── idp3.yaml                   # 模型配置
│       │   │   └── task/
│       │   │       └── drone_trajectory.yaml   # 数据配置
│       │   ├── workspace/
│       │   ├── policy/
│       │   └── dataset/
│       ├── data/outputs/
│       │   └── 2026.01.30/
│       │       └── 11.36.24_.../
│       │           └── checkpoints/
│       │               └── latest.ckpt         # ← 你的模型权重
│       └── train.py
│
└── catkin_ws0/src/ego-planner-swarm/src/planner/
    └── plan_manage/                             # EGO规划器包
        ├── src/
        │   ├── ego_planner_node.cpp
        │   ├── traj_server.cpp
        │   └── ml_planner_node.py              # ← 新增
        └── launch/
            ├── single_run_in_gazebo.launch
            └── ml_planner_deploy.launch        # ← 新增
```

---

## 📊 数据格式 (必须严格匹配)

### 输入格式

| 数据 | 维度 | 格式 | 说明 |
|------|------|------|------|
| **state** | 10 | `[pos(3), vel(3), quat(4)]` | 四元数: w,x,y,z |
| **goal** | 3 | `[x, y, z]` | 目标点 |
| **agent_pos** | 13 | `state + goal` | 拼接后输入模型 |
| **point_cloud** | (4096, 3) | xyz | 无颜色 |

### 输出格式

| 数据 | 维度 | 格式 |
|------|------|------|
| **action** | 11 | `[pos(3), vel(3), acc(3), yaw(1), yaw_dot(1)]` |

### 模型配置 (idp3.yaml)

| 参数 | 值 |
|------|-----|
| n_obs_steps | 2 |
| horizon | 16 |
| n_action_steps | 15 |
| num_points | 4096 |

---

## 🔄 数据流对比

```
EGO原流程:
  点云+odom → ego_planner_node → bspline → traj_server → pos_cmd → px4ctrl

ML新流程 (更简洁):
  点云+odom → ml_planner_node → pos_cmd → px4ctrl
```

---

## 🚀 部署步骤

### 步骤1: 下载文件

将以下4个文件放到同一目录:
- `ml_planner_node.py`
- `ml_planner_deploy.launch`
- `deploy.sh`
- `README.md` (本文件)

### 步骤2: 运行部署脚本

```bash
cd /path/to/downloaded/files
chmod +x deploy.sh
./deploy.sh
```

**或者手动复制:**

```bash
# 复制Python节点
cp ml_planner_node.py /home/cjj/AUV/catkin_ws0/src/ego-planner-swarm/src/planner/plan_manage/src/
chmod +x /home/cjj/AUV/catkin_ws0/src/ego-planner-swarm/src/planner/plan_manage/src/ml_planner_node.py

# 复制launch文件
cp ml_planner_deploy.launch /home/cjj/AUV/catkin_ws0/src/ego-planner-swarm/src/planner/plan_manage/launch/
```

### 步骤3: 启动仿真

```bash
# 终端1: 启动仿真器
roslaunch px4 fast_racing.launch
# 等待约20秒

# 终端2: 起飞
rosrun offboard_pkg offboard_node 30
# 等待约40秒，稳定后按 Ctrl+C

# 终端3: 启动ML规划器
roslaunch ego_planner ml_planner_deploy.launch
```

### 步骤4: 发送目标点

**方法A: RViz**
1. 点击顶部 "2D Nav Goal" 按钮
2. 在地图上点击目标位置

**方法B: 命令行**
```bash
rostopic pub /move_base_simple/goal geometry_msgs/PoseStamped "
header:
  frame_id: 'world'
pose:
  position: {x: 10.0, y: 0.0, z: 1.5}
  orientation: {w: 1.0}"
```

---

## 🔧 配置PX4Ctrl

如果无人机不响应:

```bash
rosrun dynamic_reconfigure dynparam set /px4ctrl mode_bool True
rosrun dynamic_reconfigure dynparam set /px4ctrl cmd_bool True
```

---

## 🔍 调试

### 检查话题

```bash
# 传感器
rostopic hz /iris_D435i/realsense/depth_camera/depth/points
rostopic hz /mavros/local_position/odom

# ML输出
rostopic echo /planning/pos_cmd

# 目标点
rostopic echo /move_base_simple/goal
```

### 正常日志

```
[MLPlanner] ✓ 数据缓冲就绪，等待目标点...
[MLPlanner] ▶ 新目标: (10.00, 0.00, 1.50)
[MLPlanner] 推理中... point_cloud: torch.Size([1, 2, 4096, 3]), agent_pos: torch.Size([1, 2, 13])
[MLPlanner] ✓ 推理完成, 轨迹形状: (15, 11)
[MLPlanner] ▶ 开始执行轨迹 (15步)
```

### 常见问题

| 问题 | 解决方案 |
|------|----------|
| `ModuleNotFoundError: diffusion_policy_3d` | 修改`ml_planner_node.py`第21行的路径 |
| 模型加载失败 | 检查checkpoint路径是否正确 |
| 无人机不动 | 运行上面的dynparam命令 |
| 找不到点云话题 | `rostopic list \| grep points` |

---

## ⚠️ 注意事项

1. **执行频率**: 100Hz (与traj_server一致)
2. **碰撞检测**: 简化版，检查0.5米内障碍物
3. **重规划**: 执行到70%自动推理下一条轨迹
4. **安全测试**: 先用1-2米小目标测试

---

## 📝 文件说明

### ml_planner_node.py

```python
# 核心流程:
# 1. 订阅: 点云、里程计、目标点
# 2. 构建: agent_pos = state(10) + goal(3)
# 3. 推理: 输出15步轨迹
# 4. 执行: 100Hz发布PositionCommand
# 5. 安全: 碰撞检测 + 自动重规划
```

### ml_planner_deploy.launch

```xml
<!-- 启动:
1. ml_planner_node - 模型推理
2. px4ctrl - 底层控制
3. rviz - 可视化
4. odom_visualization - 轨迹显示
-->
```

---

## ✅ 检查清单

- [ ] 模型路径正确 (`DIFFUSION_POLICY_PATH`)
- [ ] checkpoint存在 (`latest.ckpt`)
- [ ] ROS工作空间已source
- [ ] 仿真器启动
- [ ] 无人机已起飞
- [ ] px4ctrl已配置
- [ ] 发送目标点

---

**祝测试顺利！🚀**