#!/bin/bash
# ══════════════════════════════════════════════════════════════
# ml_gazebo_fm_auto.sh — ML Planner (FM) 非挂起自动化版
#   - 起完即返回 (不再 trap + while true 挂起)
#   - RViz 受 WITH_RVIZ 控制 (批量跑设 0)
#   - 接收 goal (waypoint0_x/y/z) 透传给 _ood.launch
#   - 清理交给外层 run_trial.sh
# 依赖: 复制 ml_planner_gazebo_fm.launch 为 ml_planner_gazebo_fm_ood.launch,
#       把 waypoint_num / waypoint0_x/y/z 由 value= 改为 default=
# 必须在 idp3 conda 环境调用 (FM 模型需要)
# ══════════════════════════════════════════════════════════════
set -u

# ── 参数 (位置参数沿用原脚本含义, goal 用环境变量传, 避免位置歧义) ──
CKPT_ARG="${1:-}"        # checkpoint 路径 (空=launch 默认 05.22 latest)
ZFF_ARG="${2:-}"         # z_feedforward_mode
FM_STEPS="${3:-}"        # FM 采样步数
CFG_LAMBDA="${4:-}"      # CFG 强度
K_STAB="${5:-}"          # k_stab
LAMBDA_JERK="${6:-}"     # lambda_jerk

# goal / rviz 用环境变量 (run_trial.sh 会 export)
GOAL_X="${GOAL_X:-12.0}"
GOAL_Y="${GOAL_Y:-0.0}"
GOAL_Z="${GOAL_Z:-2.0}"
WITH_RVIZ="${WITH_RVIZ:-0}"

DIFFUSION_ROOT="/root/autodl-tmp/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy"
if [ -d "$DIFFUSION_ROOT" ]; then
    export PYTHONPATH="$DIFFUSION_ROOT:$PYTHONPATH"
    echo "✓ PYTHONPATH: $DIFFUSION_ROOT"
fi

echo "════════════════════════════════════════════════════════════"
echo "  ML Planner (FM) auto | goal=($GOAL_X,$GOAL_Y,$GOAL_Z) rviz=$WITH_RVIZ"
echo "  ckpt: ${CKPT_ARG:-<launch default 05.22 latest>}"
echo "════════════════════════════════════════════════════════════"

# 1. 检查仿真器
echo "[1/5] 检查仿真器..."
if ! rostopic list 2>/dev/null | grep -q "/mavros"; then
    echo "  ✗ 未检测到 MAVROS! 先启 SITL。"
    exit 1
fi
echo "  ✓ MAVROS 在线"

# 2. 清理旧的本节点相关进程 (不动 SITL)
echo "[2/5] 清理旧 planner 进程..."
pkill -f "ml_planner" 2>/dev/null
pkill -f "px4ctrl"    2>/dev/null
pkill -f "rviz"       2>/dev/null
sleep 2

# offboard 起飞 (40s 后 kill, 原逻辑保留)
echo "  offboard 起飞中 (40s)..."
rosrun offboard_pkg offboard_node 30 &
sleep 40
pkill -f "offboard_node"

# 3. px4ctrl
echo "[3/5] 启动 px4ctrl..."
roslaunch px4ctrl run_node.launch &
PX4CTRL_PID=$!
sleep 12

# 4. 配置 px4ctrl
echo "[4/5] 配置 px4ctrl..."
rosrun dynamic_reconfigure dynparam set /px4ctrl mode_bool False; sleep 2
rosrun dynamic_reconfigure dynparam set /px4ctrl cmd_bool  False; sleep 2
rosrun dynamic_reconfigure dynparam set /px4ctrl mode_bool True;  sleep 3
rosrun dynamic_reconfigure dynparam set /px4ctrl cmd_bool  True;  sleep 3
echo "  ✓ px4ctrl 配置完成"

# 5. ML Planner (FM, _ood.launch 支持 goal 透传)
echo "[5/5] 启动 ML Planner (FM)..."
ARGS="waypoint0_x:=$GOAL_X waypoint0_y:=$GOAL_Y waypoint0_z:=$GOAL_Z"
[[ -n "$CKPT_ARG" ]]    && ARGS="$ARGS ckpt:=$CKPT_ARG"
[[ -n "$ZFF_ARG" ]]     && ARGS="$ARGS z_ff_mode:=$ZFF_ARG"
[[ -n "$FM_STEPS" ]]    && ARGS="$ARGS fm_num_steps:=$FM_STEPS"
[[ -n "$CFG_LAMBDA" ]]  && ARGS="$ARGS cfg_lambda:=$CFG_LAMBDA"
[[ -n "$K_STAB" ]]      && ARGS="$ARGS k_stab:=$K_STAB"
[[ -n "$LAMBDA_JERK" ]] && ARGS="$ARGS lambda_jerk:=$LAMBDA_JERK"

# ── Phase β 阶段 0-C: 合成点云开关 (M1 闭环验证) ──
# 用法: USE_SYNTH_PC=true SYNTH_OBS_JSON=xxx.json SYNTH_SPAWN_OFFSET="-8,0,0" bash run_trial.sh ...
# 不定义 = 真实点云 (默认), 定义 USE_SYNTH_PC=true = 合成点云
[[ -n "${USE_SYNTH_PC:-}" ]]       && ARGS="$ARGS use_synth_pc:=${USE_SYNTH_PC:-}"
[[ -n "${SYNTH_OBS_JSON:-}" ]]     && ARGS="$ARGS synth_obstacles_json:=${SYNTH_OBS_JSON:-}"
[[ -n "${SYNTH_SPAWN_OFFSET:-}" ]] && ARGS="$ARGS synth_spawn_offset:=${SYNTH_SPAWN_OFFSET:-}"

roslaunch ego_planner ml_planner_gazebo_fm_ood.launch $ARGS &
ML_PLANNER_PID=$!
echo "  ML Planner PID: $ML_PLANNER_PID (args: $ARGS)"

# RViz (可选)
if [ "$WITH_RVIZ" = "1" ]; then
    sleep 8
    rosrun rviz rviz -d "$(rospack find ego_planner)/launch/default_test.rviz" &
    echo "  RViz PID: $!"
fi

# 把 PID 写文件, 供外层清理参考 (可选)
LOG_DIR="${AUV_LOG_DIR:-/root/autodl-tmp/logs}"
mkdir -p "$LOG_DIR"
echo "$PX4CTRL_PID $ML_PLANNER_PID" > "$LOG_DIR/ml_planner_pids.txt"

echo "  ✓ ML Planner 已后台启动, 脚本返回 (不挂起)"
exit 0