#!/bin/bash
# start_sitl_rl.sh — RL 专用 SITL 启动（plain iris, 无深度相机, 无 conda）
# 用法: bash start_sitl_rl.sh <world_file>
set -e
WORLD="${1:?Usage: $0 <world_file>}"

AUV_ROOT="$(dirname "$(dirname "$(dirname "$(readlink -f "$0")")")")"
PX4_DIR="$AUV_ROOT/PX4-Autopilot"
CATKIN_WS="$AUV_ROOT/catkin_ws0"

echo "[start_sitl_rl] AUV_ROOT=$AUV_ROOT"
echo "[start_sitl_rl] world=$WORLD"

CLEAN_PATH=$(echo "$PATH" | tr ':' '\n' | grep -ivE "miniconda|conda" | paste -sd ':')
CLEAN_PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -ivE "miniconda|conda" | paste -sd ':')

# 启 SITL 在干净系统 Python 环境（脱离 conda，复刻 start_sitl.sh 的 setsid 隔离）
exec setsid env PATH="$CLEAN_PATH" PYTHONPATH="$CLEAN_PYTHONPATH" \
    bash -c "
        source /opt/ros/noetic/setup.bash
        source $CATKIN_WS/devel/setup.bash
        source $PX4_DIR/Tools/setup_gazebo.bash $PX4_DIR $PX4_DIR/build/px4_sitl_default
        export ROS_PACKAGE_PATH=\$ROS_PACKAGE_PATH:$PX4_DIR:$PX4_DIR/Tools/sitl_gazebo
        exec roslaunch px4 fast_test.launch world:='$WORLD' vehicle:=iris
    "
