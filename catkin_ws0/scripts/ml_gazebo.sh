#!/bin/bash
# ══════════════════════════════════════════════════════════════
# ML Planner 启动脚本 (Phase ② 版) - 支持 checkpoint 命令行覆盖
# ══════════════════════════════════════════════════════════════
# 变更 (vs Phase ①):
#   - 接受可选参数 $1 = checkpoint 路径
#   - 通过 roslaunch 的 ckpt:= arg 传给 ml_planner_gazebo.launch
#   - 不传参数时用 launch 默认值 (Phase 1a R2)
#
# 用法:
#   bash ml_gazebo.sh                          # 用 launch 默认 ckpt
#   bash ml_gazebo.sh /path/to/phase2/best.ckpt   # 用指定 ckpt
# ══════════════════════════════════════════════════════════════

CKPT_ARG="${1:-}"     # 可选: checkpoint 路径
ZFF_ARG="${2:-}"      # 可选: z_feedforward_mode (full/no_acc/no_ff)

echo "════════════════════════════════════════════════════════════"
echo "  ML Planner - 扩散模型 (EGO兼容) - 无RViz模式"
if [[ -n "$CKPT_ARG" ]]; then
    echo "  checkpoint (覆盖): $CKPT_ARG"
else
    echo "  checkpoint: launch 默认值 (Phase 1a R2)"
fi
if [[ -n "$ZFF_ARG" ]]; then
    echo "  z_feedforward_mode (覆盖): $ZFF_ARG"
else
    echo "  z_feedforward_mode: launch 默认值 (full)"
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

# 4. 配置px4ctrl (与EGO一致: 先False再True, 触发状态转换)
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

# 5. 启动ML Planner (Phase ②: 传 ckpt + z_ff_mode arg)
echo "[5/5] 启动ML Planner (扩散模型)..."
ROSLAUNCH_ARGS=""
[[ -n "$CKPT_ARG" ]] && ROSLAUNCH_ARGS="$ROSLAUNCH_ARGS ckpt:=$CKPT_ARG"
[[ -n "$ZFF_ARG" ]]  && ROSLAUNCH_ARGS="$ROSLAUNCH_ARGS z_ff_mode:=$ZFF_ARG"
roslaunch ego_planner ml_planner_gazebo.launch $ROSLAUNCH_ARGS &
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
echo "5. 节点状态:"
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
echo "  系统已启动! (无RViz，资源占用低)"
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


