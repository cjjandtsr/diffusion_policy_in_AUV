#!/bin/bash
# ══════════════════════════════════════════════════════════════
# takeoff_only.sh — 只起飞, 不启 planner (供 RL 节点接管)
#
# 与 ml_gazebo_fm_auto.sh 的差异:
#   - 去掉最后一步 roslaunch planner
#   - RL 节点自己 pub /planning/pos_cmd
#
# 用法: bash takeoff_only.sh [ckpt_arg...]  (ckpt 参数被忽略, 保留兼容)
# 环境: idp3 conda
# ══════════════════════════════════════════════════════════════
set -u

# 1. 检查仿真器
echo "[1/4] 检查仿真器..."
if ! rostopic list 2>/dev/null | grep -q "/mavros"; then
    echo "  ✗ 未检测到 MAVROS! 先启 SITL。"
    exit 1
fi
echo "  ✓ MAVROS 在线"

# 2. 清理旧的本节点相关进程 (不动 SITL)
echo "[2/4] 清理旧进程..."
pkill -f "ml_planner"       2>/dev/null
pkill -f "rl_sampler"       2>/dev/null
pkill -f "px4ctrl"          2>/dev/null
pkill -f "rviz"             2>/dev/null
pkill -f "offboard_node"    2>/dev/null
sleep 2

# offboard 起飞
echo "  offboard 起飞中 (40s)..."
rosrun offboard_pkg offboard_node 30 &
sleep 40
pkill -f "offboard_node"

# 3. px4ctrl
echo "[3/4] 启动 px4ctrl..."
roslaunch px4ctrl run_node.launch &
PX4CTRL_PID=$!
sleep 12

# 4. 配置 px4ctrl
echo "[4/4] 配置 px4ctrl..."
rosrun dynamic_reconfigure dynparam set /px4ctrl mode_bool False; sleep 2
rosrun dynamic_reconfigure dynparam set /px4ctrl cmd_bool  False; sleep 2
rosrun dynamic_reconfigure dynparam set /px4ctrl mode_bool True;  sleep 3
rosrun dynamic_reconfigure dynparam set /px4ctrl cmd_bool  True;  sleep 3
echo "  ✓ px4ctrl 配置完成, 等待 RL 节点发 pos_cmd"

echo "  ✓ 起飞完成, RL 节点可接管"
exit 0
