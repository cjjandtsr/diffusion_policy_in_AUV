#!/bin/bash
# ══════════════════════════════════════════════════════════════
# ML Planner 启动脚本 (修复版) - 不启动 RViz 版本
# 关键修正：使用EGO的default_test.rviz（话题名已兼容）
# 节省内存/CPU：跳过 RViz 启动
# ══════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════════"
echo "  ML Planner - 扩散模型 (EGO兼容) - 无RViz模式"
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

# 移除 offboard_node 干扰（注释掉）
# rosrun offboard_pkg offboard_node 30 &
# sleep 40
# pkill -f "offboard_node"

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

# 5. 启动ML Planner
echo "[5/5] 启动ML Planner (扩散模型)..."
roslaunch ego_planner ml_planner_gazebo.launch &
ML_PLANNER_PID=$!
echo "  ML Planner PID: $ML_PLANNER_PID"
sleep 10

# 6. 启动RViz（已禁用 - 节省内存/CPU）
echo "[6] 跳过RViz启动 (节省资源模式)"
# 原启动命令已注释：
# rosrun rviz rviz -d $(rospack find ego_planner)/launch/default_test.rviz &
# RVIZ_PID=$!
# echo "  RViz PID: $RVIZ_PID"
# sleep 3
RVIZ_PID="未启动"

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
echo "2. EGO兼容可视化话题 (RViz自动显示):"
for topic in /drone_0_ego_planner_node/goal_point \
             /drone_0_ego_planner_node/optimal_list \
             /drone_0_odom_visualization/robot \
             /drone_0_odom_visualization/path; do
    hz=$(rostopic hz $topic 2>&1 &
         sleep 2; kill $! 2>/dev/null)
    pub=$(rostopic info $topic 2>/dev/null | grep "Publishers:" | head -1)
    if [ -n "$pub" ]; then
        echo "  ✓ $topic"
    else
        echo "  ✗ $topic (未发布)"
    fi
done

echo ""
echo "3. ML专用详细可视化 (RViz手动Add MarkerArray):"
echo "  /drone_0_ml_planner/diffusion_detail"

echo ""
echo "4. 日志文件:"
ls -la ~/ml_planner_log/ 2>/dev/null | tail -3 || echo "  (等待首次规划后生成)"

echo ""
echo "5. 节点状态:"
for name_pid in "px4ctrl:$PX4CTRL_PID" "ML Planner:$ML_PLANNER_PID" "RViz:0"; do
    name="${name_pid%%:*}"
    pid="${name_pid##*:}"
    if [ "$pid" != "0" ] && ps -p $pid > /dev/null 2>&1; then
        echo "  ✓ $name 运行中 (PID $pid)"
    elif [ "$pid" == "0" ]; then
        echo "  - $name 已禁用 (未启动)"
    else
        echo "  ✗ $name 未运行"
    fi
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  系统已启动! (无RViz，资源占用低)"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "可视化提示:"
echo "  由于未启动RViz，当前无法看到3D画面。"
echo "  若需要可视化，可稍后手动运行:"
echo "    rosrun rviz rviz -d \$(rospack find ego_planner)/launch/default_test.rviz"
echo ""
echo "终端日志:"
echo "  每次推理 → 打印完整15步×11维数值表"
echo "  每50条cmd → 打印PositionCommand具体值+跟踪误差"
echo "  CSV文件  → ~/ml_planner_log/ (离线分析)"
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