#!/bin/bash
# run_one_trial.sh — 复刻手动操作顺序跑一个 trial
# [1]清理 [2]deactivate+起SITL [3]等就绪 [4]activate idp3+source ws+起规划
WORLD="${1:-/home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds/ood_experiment_worlds/layerA/env_000_obs1.world}"
ML_SH="$HOME/AUV/catkin_ws0/scripts/ml_gazebo_fm.sh"
WS_SETUP="$HOME/AUV/catkin_ws0/devel/setup.bash"

# 日志落数据盘 (系统盘 / 只有 30G)
LOG_DIR="${AUV_LOG_DIR:-/root/autodl-tmp/logs}"
mkdir -p "$LOG_DIR"

source ~/.bashrc

echo "═══ [1] 清理旧进程 ═══"
pkill -9 -f "fast_test.launch" 2>/dev/null
pkill -9 -f "px4_sitl_default/bin/px4" 2>/dev/null
pkill -9 -f "gzserver" 2>/dev/null
pkill -9 -f "gzclient" 2>/dev/null
pkill -9 -f "mavros_node" 2>/dev/null
pkill -f "ml_planner" 2>/dev/null
pkill -f "px4ctrl" 2>/dev/null
sleep 3

echo "═══ [2] conda deactivate + 起 SITL (系统 python) ═══"
conda deactivate 2>/dev/null
source "$WS_SETUP"
roslaunch px4 fast_test.launch world:="$WORLD" > "$LOG_DIR/sitl.log" 2>&1 &
echo "  SITL 后台启动 (日志 $LOG_DIR/sitl.log)"

echo "═══ [3] 等 SITL 就绪 ═══"
for i in $(seq 1 40); do
    sleep 1
    CONN=$(timeout 1 rostopic echo -n 1 /mavros/state/connected 2>/dev/null | head -1)
    if [ "$CONN" = "True" ]; then echo "  ✓ SITL 就绪 (${i}s)"; break; fi
done
sleep 3

echo "═══ [4] conda activate idp3 + source ws + 起规划 ═══"
conda activate idp3
source "$WS_SETUP"
bash "$ML_SH" > "$LOG_DIR/planner.log" 2>&1 &
echo "  规划后台启动 (日志 $LOG_DIR/planner.log), 起飞+规划约 60-70s"
echo ""
echo "═══ 监控: tail -f $LOG_DIR/planner.log ═══"
