#!/bin/bash
# ══════════════════════════════════════════════════════════════
# run_trial_ego.sh — 本地 EGO 单次 trial (与服务器 BC 同 monitor 同 goal, 可比)
#   用法:
#     bash run_trial_ego.sh <world> <obstacles_json> <gx> <gy> <gz> <out_json> [tier] [wid] [trial]
#   EGO 启动序列按 ego_gazebo.sh 实际顺序:
#     offboard 40s → ego-planner → sleep15 → px4ctrl → sleep10 → (rviz) → dynparam
#   EGO goal 由 single_run_in_gazebo.launch 的 flight_type=2 + point0 决定(需预先改为 16,0,2),
#     monitor 的 --goal_* 必须与之一致。
# ══════════════════════════════════════════════════════════════
set -u

SCRIPTS_DIR="/home/cjj/AUV/catkin_ws0/scripts"
MONITOR="$SCRIPTS_DIR/collision_timeout_monitor.py"

WORLD="${1:?need world}"
OBS_JSON="${2:?need obstacles_json}"
GX="${3:?gx}"; GY="${4:?gy}"; GZ="${5:?gz}"
OUT_JSON="${6:?out}"
TIER="${7:-ego_layerA}"; WORLD_ID="${8:-0}"; TRIAL="${9:-0}"

WITH_RVIZ="${WITH_RVIZ:-0}"
TIMEOUT="${TIMEOUT:-300}"
DRONE_RADIUS="${DRONE_RADIUS:-0.35}"
SITL_WAIT="${SITL_WAIT:-120}"
PLANNER_WAIT="${PLANNER_WAIT:-90}"
DRONE_NAME="${DRONE_NAME:-iris_D435i}"

PX4_DIR="/home/cjj/AUV/PX4-Autopilot"
CATKIN_WS="/home/cjj/AUV/catkin_ws0"
source /opt/ros/noetic/setup.bash
source "$CATKIN_WS/devel/setup.bash"
source "$PX4_DIR/Tools/setup_gazebo.bash" "$PX4_DIR" "$PX4_DIR/build/px4_sitl_default" 2>/dev/null
export ROS_PACKAGE_PATH="$ROS_PACKAGE_PATH:$PX4_DIR:$PX4_DIR/Tools/sitl_gazebo"

log() { echo "[run_trial_ego] $*"; }

full_cleanup() {
    log "清理..."
    pkill -9 -f "collision_timeout_monitor" 2>/dev/null
    pkill -9 -f "single_run_in_gazebo"      2>/dev/null
    pkill -9 -f "ego_planner_node"          2>/dev/null
    pkill -9 -f "ego_replan_fsm"            2>/dev/null
    pkill -9 -f "traj_server"               2>/dev/null
    pkill -9 -f "planner_manager"           2>/dev/null
    pkill -9 -f "odom_visualization"        2>/dev/null
    pkill -9 -f "px4ctrl"                    2>/dev/null
    pkill -2 -f "offboard_node"              2>/dev/null
    sleep 1
    pkill -9 -f "offboard_node"              2>/dev/null
    pkill -9 -f "rviz"                       2>/dev/null
    pkill -9 -f "fast_test.launch"           2>/dev/null
    pkill -9 -f "px4_sitl_default/bin/px4"   2>/dev/null
    pkill -9 -f "gzserver"                   2>/dev/null
    pkill -9 -f "gzclient"                   2>/dev/null
    pkill -9 -f "mavros_node"                2>/dev/null
    sleep 3
}
trap 'log "中断"; full_cleanup; exit 130' SIGINT SIGTERM

echo "════════════════════════════════════════════════"
echo "  EGO TRIAL  tier=$TIER wid=$WORLD_ID trial=$TRIAL"
echo "  world: $WORLD"
echo "  goal : ($GX,$GY,$GZ)  timeout=${TIMEOUT}s"
echo "════════════════════════════════════════════════"

[ -f "$WORLD" ]    || { log "✗ world 不存在"; exit 2; }
[ -f "$OBS_JSON" ] || { log "✗ obstacles_json 不存在"; exit 2; }
[ -f "$MONITOR" ]  || { log "✗ monitor 不存在"; exit 2; }
mkdir -p "$(dirname "$OUT_JSON")"

full_cleanup

# 1. SITL
log "启动 SITL"
bash "$SCRIPTS_DIR/start_sitl.sh" "$WORLD"

# 2. 等 MAVROS
log "等 MAVROS (上限 ${SITL_WAIT}s)..."
t0=$(date +%s); ok=0
while [ $(( $(date +%s) - t0 )) -lt "$SITL_WAIT" ]; do
    if timeout 3 rostopic echo -n1 /mavros/state 2>/dev/null | grep -m1 "connected:" | grep -q "True"; then ok=1; break; fi
    sleep 3
done
[ "$ok" = "1" ] || { log "✗ MAVROS 超时"; full_cleanup; echo "{\"tier\":\"$TIER\",\"world_id\":$WORLD_ID,\"trial\":$TRIAL,\"setup_failed\":\"mavros_timeout\",\"success_deploy\":false}" > "$OUT_JSON"; exit 3; }
log "✓ MAVROS connected"; sleep 5

# 3. EGO 启动序列 (按 run_ego_demo.sh 实际顺序: px4ctrl先 → 等服务 → dynparam → EGO后)
log "offboard 起飞 (确认 armed+爬升, 失败重试)..."
TAKEOFF_OK=0
for attempt in 1 2 3; do
    log "  起飞尝试 $attempt/3..."
    rosrun offboard_pkg offboard_node 90 &
    OFFB_PID=$!
    # 阶段1: 等 armed=True (上限 30s)
    armed_ok=0; t0=$(date +%s)
    while [ $(( $(date +%s) - t0 )) -lt 30 ]; do
        armed=$(timeout 2 rostopic echo -n1 /mavros/state 2>/dev/null | grep -m1 "armed:" | grep -o "True\|False")
        if [ "$armed" = "True" ]; then armed_ok=1; break; fi
        sleep 2
    done
    if [ "$armed_ok" != "1" ]; then
        log "  尝试 $attempt: 未解锁, 清理重试"
        kill -2 $OFFB_PID 2>/dev/null
        sleep 2
        kill -9 $OFFB_PID 2>/dev/null
        continue
    fi
    # 阶段2: armed 成功 (螺旋桨转了), 让 offboard 继续维持, 固定等 30s 爬升到位
    log "  ✓ armed=True (螺旋桨已启动), 等爬升 30s..."
    TAKEOFF_OK=1
    sleep 30
    # 用 SIGINT 触发 offboard_node 的 signalHandler → AUTO.LOITER → 保持悬停
    log "  发送 SIGINT → offboard_node 切换到 AUTO.LOITER..."
    kill -2 $OFFB_PID 2>/dev/null
    sleep 3
    # 确认已离开 OFFBOARD 模式
    mode=$(timeout 2 rostopic echo -n1 /mavros/state 2>/dev/null | grep -m1 "mode:" | grep -oE "[A-Z.]+" | tail -1)
    log "  当前模式: ${mode:-unknown}"
    break
done

if [ "$TAKEOFF_OK" != "1" ]; then
    log "✗ 3次起飞失败(arm或爬升), 跳过此 trial"
    full_cleanup
    echo "{\"tier\":\"$TIER\",\"world_id\":$WORLD_ID,\"trial\":$TRIAL,\"setup_failed\":\"takeoff_failed\",\"success_deploy\":false}" > "$OUT_JSON"
    exit 6
fi

log "启动 px4ctrl..."
roslaunch px4ctrl run_node.launch &
sleep 12

log "等 /px4ctrl/set_parameters 服务就绪..."
for i in $(seq 1 10); do
    if rosservice list 2>/dev/null | grep -q "/px4ctrl/set_parameters"; then
        log "✓ set_parameters 就绪"; break
    fi
    sleep 2
done

log "配置 px4ctrl..."
rosrun dynamic_reconfigure dynparam set /px4ctrl mode_bool False; sleep 2
rosrun dynamic_reconfigure dynparam set /px4ctrl cmd_bool  False; sleep 2
rosrun dynamic_reconfigure dynparam set /px4ctrl mode_bool True;  sleep 3
rosrun dynamic_reconfigure dynparam set /px4ctrl cmd_bool  True;  sleep 3
log "✓ px4ctrl 配置完成"

log "启动 ego-planner..."
roslaunch ego_planner single_run_in_gazebo.launch &
sleep 10

# 等深度相机话题就绪 (EGO 建图依赖, 批量模式下 SITL 刚起相机可能没稳)
log "等深度相机话题就绪..."
t0=$(date +%s)
while [ $(( $(date +%s) - t0 )) -lt 30 ]; do
    if rostopic info /iris_D435i/realsense/depth_camera/depth/image_raw 2>/dev/null | grep -q "Publishers:"; then
        log "✓ 深度相机话题有 publisher"; break
    fi
    sleep 2
done

# 无论 RViz 开关, 都给 EGO 额外建图缓冲时间
# (单trial RViz=1 时多出的启动时间正是这个缓冲, 批量 RViz=0 必须补上,
#  否则 EGO 在深度图建好前规划, 把无人机周围误判为障碍 → "drone is in obstacle")
log "EGO 建图缓冲 (12s)..."
sleep 12

if [ "$WITH_RVIZ" = "1" ]; then
    rosrun rviz rviz -d "$(rospack find ego_planner)/launch/default_test.rviz" &
    sleep 5
fi

# 4. 等 pos_cmd
log "等 /planning/pos_cmd (上限 ${PLANNER_WAIT}s)..."
t0=$(date +%s); ok=0
while [ $(( $(date +%s) - t0 )) -lt "$PLANNER_WAIT" ]; do
    pc=$(rostopic info /planning/pos_cmd 2>/dev/null | sed -n '/Publishers:/,/Subscribers:/p' | grep -c " \* ")
    if [ "${pc:-0}" -ge 1 ]; then ok=1; break; fi
    sleep 3
done
[ "$ok" = "1" ] || { log "✗ pos_cmd 超时"; full_cleanup; echo "{\"tier\":\"$TIER\",\"world_id\":$WORLD_ID,\"trial\":$TRIAL,\"setup_failed\":\"no_pos_cmd\",\"success_deploy\":false}" > "$OUT_JSON"; exit 5; }
log "✓ pos_cmd 发布, 开始监控"

# 5. monitor (前台 wait)
python3 "$MONITOR" \
    --obstacles_json "$OBS_JSON" \
    --goal_x "$GX" --goal_y "$GY" --goal_z "$GZ" \
    --drone_radius "$DRONE_RADIUS" \
    --timeout "$TIMEOUT" \
    --output "$OUT_JSON" \
    --tier "$TIER" --world_id "$WORLD_ID" --trial "$TRIAL" \
    --filename "$(basename "$WORLD")" \
    --drone_name "$DRONE_NAME"
log "monitor 退出"

full_cleanup

if [ -f "$OUT_JSON" ]; then
    python3 -c "
import json; d=json.load(open('$OUT_JSON'))
print('  success_deploy =', d.get('success_deploy'))
print('  collision      =', d.get('collision'))
print('  timeout        =', d.get('timeout'))
print('  final_dxy      = %.2f' % d.get('final_dist_xy',0))
print('  min_clearance  = %.2f' % d.get('min_clearance',0))
print('  time_to_goal   = %.0fs' % d.get('time_to_goal',0))
"
fi
log "完成: $OUT_JSON"
exit 0