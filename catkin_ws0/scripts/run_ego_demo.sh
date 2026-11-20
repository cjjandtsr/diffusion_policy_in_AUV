#!/bin/bash
# ══════════════════════════════════════════════════════════════
# EGO-Planner Demo 启动脚本
# 流程: offboard起飞 → px4ctrl → ego-planner → RViz
# ══════════════════════════════════════════════════════════════

set -e

# ══════════════════════════════════════════════════════════════
# ROS 环境
# ══════════════════════════════════════════════════════════════
source /opt/ros/noetic/setup.bash
source /home/cjj/AUV/catkin_ws0/devel/setup.bash
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:/home/cjj/AUV/PX4-Autopilot:/home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo

ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"
export ROS_MASTER_URI

echo "════════════════════════════════════════════════════════════"
echo "  EGO-Planner Demo"
echo "  ROS_MASTER_URI=$ROS_MASTER_URI"
echo "════════════════════════════════════════════════════════════"
echo ""

# ══════════════════════════════════════════════════════════════
# 0. 环境检查
# ══════════════════════════════════════════════════════════════
echo "[0/5] 检查仿真器状态..."
if ! rostopic list 2>/dev/null | grep -q "/mavros"; then
    echo "  错误: 未检测到 MAVROS 话题!"
    echo "  请先启动仿真器: bash ~/AUV/catkin_ws0/scripts/start_sitl.sh"
    exit 1
fi
echo "  ✓ 仿真器运行中"

# 清理旧进程
echo "  清理旧进程..."
pkill -f "offboard_node" 2>/dev/null || true
pkill -f "px4ctrl" 2>/dev/null || true
pkill -f "single_run_in_gazebo" 2>/dev/null || true
pkill -f "dynparam" 2>/dev/null || true
pkill -f "rviz" 2>/dev/null || true
sleep 2
echo "  ✓ 清理完成"

# ══════════════════════════════════════════════════════════════
# 1. offboard 起飞
# ══════════════════════════════════════════════════════════════
echo "[1/5] offboard 起飞 (30s)..."
rosrun offboard_pkg offboard_node 30 &
OFFBOARD_PID=$!
echo "  offboard PID: $OFFBOARD_PID"
sleep 40
pkill -f "offboard_node" 2>/dev/null || true
echo "  ✓ offboard 完成"

# ══════════════════════════════════════════════════════════════
# 2. 启动 px4ctrl
# ══════════════════════════════════════════════════════════════
echo "[2/5] 启动 px4ctrl..."
roslaunch px4ctrl run_node.launch &
PX4CTRL_PID=$!
echo "  px4ctrl PID: $PX4CTRL_PID"
sleep 12

# 等待 set_parameters 服务就绪
echo "  等待 /px4ctrl/set_parameters 服务..."
for i in $(seq 1 10); do
    if rosservice list 2>/dev/null | grep -q "/px4ctrl/set_parameters"; then
        echo "  ✓ 服务就绪"
        break
    fi
    sleep 2
done

# ══════════════════════════════════════════════════════════════
# 3. 配置 px4ctrl
# ══════════════════════════════════════════════════════════════
echo "[3/5] 配置 px4ctrl 控制参数..."
rosrun dynamic_reconfigure dynparam set /px4ctrl mode_bool False
sleep 2
rosrun dynamic_reconfigure dynparam set /px4ctrl cmd_bool False
sleep 2
rosrun dynamic_reconfigure dynparam set /px4ctrl mode_bool True
sleep 3
rosrun dynamic_reconfigure dynparam set /px4ctrl cmd_bool True
sleep 3
echo "  ✓ px4ctrl 配置完成"

# ══════════════════════════════════════════════════════════════
# 4. 启动 EGO-Planner
# ══════════════════════════════════════════════════════════════
echo "[4/5] 启动 EGO-Planner..."
roslaunch ego_planner single_run_in_gazebo.launch &
EGO_PID=$!
echo "  EGO-Planner PID: $EGO_PID"
sleep 10
echo "  ✓ EGO-Planner 已启动"

# ══════════════════════════════════════════════════════════════
# 5. 启动 RViz
# ══════════════════════════════════════════════════════════════
echo "[5/5] 启动 RViz..."
rosrun rviz rviz -d "$(rospack find ego_planner)/launch/default_test.rviz" &
RVIZ_PID=$!
echo "  RViz PID: $RVIZ_PID"
sleep 5

# ══════════════════════════════════════════════════════════════
# 状态汇总
# ══════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  系统已启动!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  节点状态:"
for name_pid in "offboard(已结束):$OFFBOARD_PID" "px4ctrl:$PX4CTRL_PID" "EGO-Planner:$EGO_PID" "RViz:$RVIZ_PID"; do
    name="${name_pid%%:*}"
    pid="${name_pid##*:}"
    if [ -n "$pid" ] && ps -p "$pid" > /dev/null 2>&1; then
        echo "    ✓ $name (PID $pid)"
    else
        echo "    - $name"
    fi
done
echo ""
echo "  关键话题:"
echo "    /planning/pos_cmd  — EGO 规划输出"
echo "    /px4ctrl/odom     — 里程计"
echo "    /mavros/state      — 飞控状态"
echo ""
echo "  按 Ctrl+C 停止所有节点"
echo ""

# ══════════════════════════════════════════════════════════════
# 清理 trap
# ══════════════════════════════════════════════════════════════
cleanup() {
    echo ""
    echo "正在清理..."
    pkill -f "single_run_in_gazebo" 2>/dev/null || true
    pkill -f "run_node.launch" 2>/dev/null || true
    pkill -f "px4ctrl" 2>/dev/null || true
    pkill -f "dynparam" 2>/dev/null || true
    pkill -f "rviz" 2>/dev/null || true
    pkill -f "ego_planner_node" 2>/dev/null || true
    pkill -f "traj_server" 2>/dev/null || true
    pkill -f "odom_visualization" 2>/dev/null || true
    echo "清理完成"
    exit 0
}
trap cleanup SIGINT SIGTERM

# 保持运行，等待 Ctrl+C
while true; do sleep 1; done
