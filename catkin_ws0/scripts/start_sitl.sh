#!/bin/bash
WORLD="${1:-/home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds/ood_experiment_worlds/layerA/env_000_obs1.world}"

# 日志落数据盘 (系统盘 / 只有 30G, sitl.log 曾涨到 20.9G)
LOG_DIR="${AUV_LOG_DIR:-/root/autodl-tmp/logs}"
mkdir -p "$LOG_DIR"

GUI="${AUV_GUI:-false}"

echo "[start_sitl] 清理旧进程..."
pkill -9 -f "fast_test.launch"          2>/dev/null
pkill -9 -f "px4_sitl_default/bin/px4"  2>/dev/null
pkill -9 -f "gzserver"                  2>/dev/null
pkill -9 -f "gzclient"                  2>/dev/null
pkill -9 -f "mavros_node"               2>/dev/null
sleep 3
if netstat -tlnp 2>/dev/null | grep -q 11345; then sleep 3; fi

CLEAN_PATH=$(echo "$PATH" | tr ':' '\n' | grep -ivE "miniconda|conda" | paste -sd ':')
CLEAN_PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -ivE "miniconda|conda" | paste -sd ':')

PX4_DIR="/home/cjj/AUV/PX4-Autopilot"
CATKIN_WS="/home/cjj/AUV/catkin_ws0"

echo "[start_sitl] 启动 SITL (系统 python 优先, world=$WORLD)..."
echo "[start_sitl] gui=$GUI"
echo "[start_sitl] 日志: $LOG_DIR/sitl.log"

setsid env PATH="$CLEAN_PATH" PYTHONPATH="$CLEAN_PYTHONPATH" \
    bash -c "
        source /opt/ros/noetic/setup.bash
        source $CATKIN_WS/devel/setup.bash
        source $PX4_DIR/Tools/setup_gazebo.bash $PX4_DIR $PX4_DIR/build/px4_sitl_default
        export ROS_PACKAGE_PATH=\$ROS_PACKAGE_PATH:$PX4_DIR:$PX4_DIR/Tools/sitl_gazebo
        echo '--- ROS_PACKAGE_PATH check ---'
        rospack find px4 || echo 'PX4 STILL NOT FOUND'
        roslaunch px4 fast_test.launch world:='$WORLD' interactive:=false gui:=$GUI
    " > "$LOG_DIR/sitl.log" 2>&1 < /dev/null &

echo "[start_sitl] 已后台启动。看进度: tail -f $LOG_DIR/sitl.log"