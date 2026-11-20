#!/bin/bash
# ══════════════════════════════════════════════════════════════
# ML Planner 启动脚本 (Phase1a-FM 版) - Flow Matching 部署
# ══════════════════════════════════════════════════════════════

CKPT_ARG="${1:-}"        # checkpoint 路径
ZFF_ARG="${2:-}"         # z_feedforward_mode (full/no_acc/no_ff)
FM_STEPS="${3:-}"        # FM采样步数 (可选, 默认4)
CFG_LAMBDA="${4:-}"      # CFG强度 (可选, 默认1.5)
K_STAB="${5:-}"          # 层次1: k_stab (可选, 默认-1=auto)
LAMBDA_JERK="${6:-}"     # 层次1: lambda_jerk (可选, 默认-1=auto)

# 设置Python路径
DIFFUSION_ROOT="/root/autodl-tmp/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy"
if [ -d "$DIFFUSION_ROOT" ]; then
    export PYTHONPATH="$DIFFUSION_ROOT:$PYTHONPATH"
    echo "✓ 设置 PYTHONPATH: $DIFFUSION_ROOT"
fi

echo "════════════════════════════════════════════════════════════"
echo "  ML Planner - Flow Matching (FM) 模型 - 无RViz模式"
if [[ -n "$CKPT_ARG" ]]; then
    echo "  checkpoint: $CKPT_ARG"
else
    echo "  checkpoint: launch 默认值 (2026.06.01 layer1)"
fi
if [[ -n "$ZFF_ARG" ]]; then
    echo "  z_feedforward_mode: $ZFF_ARG"
else
    echo "  z_feedforward_mode: launch 默认值 (full)"
fi
if [[ -n "$FM_STEPS" ]]; then
    echo "  fm_num_steps: $FM_STEPS"
fi
if [[ -n "$CFG_LAMBDA" ]]; then
    echo "  cfg_lambda: $CFG_LAMBDA"
fi
if [[ -n "$K_STAB" ]]; then
    echo "  k_stab: $K_STAB"
else
    echo "  k_stab: auto (从 ckpt 检测 or 默认 0.0)"
fi
if [[ -n "$LAMBDA_JERK" ]]; then
    echo "  lambda_jerk: $LAMBDA_JERK"
else
    echo "  lambda_jerk: auto (从 ckpt 检测 or 默认 0.0)"
fi
echo "════════════════════════════════════════════════════════════"
echo ""

# 1. 检查仿真器
echo "[1/5] 检查仿真器状态..."
if ! rostopic list 2>/dev/null | grep -q "/mavros"; then
    echo "  错误: 未检测到MAVROS话题!"
    echo "  请先启动仿真器: roslaunch px4 fast_test.launch"
    exit 1
fi
echo "  ✓ 仿真器运行中"

# 2. 清理旧进程
echo "[2/5] 清理旧进程..."
pkill -f "ml_planner" 2>/dev/null
pkill -f "px4ctrl" 2>/dev/null
pkill -f "rviz" 2>/dev/null
sleep 2

# offboard_node
rosrun offboard_pkg offboard_node 30 &
sleep 40
pkill -f "offboard_node"

# 3. 启动px4ctrl
echo "[3/5] 启动px4ctrl..."
roslaunch px4ctrl run_node.launch &
PX4CTRL_PID=$!
echo "  px4ctrl PID: $PX4CTRL_PID"
sleep 12

# 4. 配置px4ctrl
echo "[4/5] 配置px4ctrl控制参数..."
rosrun dynamic_reconfigure dynparam set /px4ctrl mode_bool False
sleep 2
rosrun dynamic_reconfigure dynparam set /px4ctrl cmd_bool False
sleep 2
rosrun dynamic_reconfigure dynparam set /px4ctrl mode_bool True
sleep 3
rosrun dynamic_reconfigure dynparam set /px4ctrl cmd_bool True
sleep 3
echo "  ✓ px4ctrl配置完成"

# 5. 启动ML Planner (FM版本) - 注意使用正确的launch文件
echo "[5/5] 启动ML Planner (Flow Matching)..."
ROSLAUNCH_ARGS=""
[[ -n "$CKPT_ARG" ]]    && ROSLAUNCH_ARGS="$ROSLAUNCH_ARGS ckpt:=$CKPT_ARG"
[[ -n "$ZFF_ARG" ]]     && ROSLAUNCH_ARGS="$ROSLAUNCH_ARGS z_ff_mode:=$ZFF_ARG"
[[ -n "$FM_STEPS" ]]    && ROSLAUNCH_ARGS="$ROSLAUNCH_ARGS fm_num_steps:=$FM_STEPS"
[[ -n "$CFG_LAMBDA" ]]  && ROSLAUNCH_ARGS="$ROSLAUNCH_ARGS cfg_lambda:=$CFG_LAMBDA"
[[ -n "$K_STAB" ]]      && ROSLAUNCH_ARGS="$ROSLAUNCH_ARGS k_stab:=$K_STAB"
[[ -n "$LAMBDA_JERK" ]] && ROSLAUNCH_ARGS="$ROSLAUNCH_ARGS lambda_jerk:=$LAMBDA_JERK"

# ★★★ 关键修改：使用 _fm.launch 而不是普通 launch ★★★
roslaunch ego_planner ml_planner_gazebo_fm.launch $ROSLAUNCH_ARGS &
ML_PLANNER_PID=$!
echo "  ML Planner PID: $ML_PLANNER_PID  (args:$ROSLAUNCH_ARGS)"
sleep 10

echo "[6] RViz启动 "
rosrun rviz rviz -d $(rospack find ego_planner)/launch/default_test.rviz &
RVIZ_PID=$!
echo "  RViz PID: $RVIZ_PID"
sleep 3

# 7. 验证系统
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  系统验证"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "1. 关键话题:"
echo "  /planning/pos_cmd:"
rostopic info /planning/pos_cmd 2>/dev/null | grep -E "Publishers:|Subscribers:" || echo "    未发布"

echo ""
echo "节点状态:"
for name_pid in "px4ctrl:$PX4CTRL_PID" "ML Planner:$ML_PLANNER_PID"; do
    name="${name_pid%%:*}"
    pid="${name_pid##*:}"
    if [ "$pid" != "0" ] && ps -p $pid > /dev/null 2>&1; then
        echo "  ✓ $name 运行中 (PID $pid)"
    else
        echo "  ✗ $name 未运行"
    fi
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Flow Matching 系统已启动!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "按 Ctrl+C 停止系统"
echo ""

cleanup() {
    echo ""
    echo "正在清理..."
    pkill -f "ml_planner" 2>/dev/null
    pkill -f "px4ctrl" 2>/dev/null
    pkill -f "rviz" 2>/dev/null
    pkill -f "roslaunch" 2>/dev/null
    echo "清理完成"
    exit 0
}
trap cleanup SIGINT SIGTERM

while true; do sleep 1; done
