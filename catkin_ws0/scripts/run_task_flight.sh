#!/bin/bash
# ══════════════════════════════════════════════════════════════
# run_task_flight.sh — 逐任务 BC 闭环 trial (出生点 = 任务起点, 起飞高度 = 起点高度)
#   本文件 = run_bc_flight.sh 的副本, 差异:
#     (1) 出生位姿由参数给 (start_x/start_y/start_yaw) → start_sitl_at.sh → fast_test.launch
#     (2) 起飞目标点由 offboard_node_goto 给 (0, 0, start_z), 原 offboard_node 仍写死 z=2
#     (3) 规划器走 ml_planner_gazebo_task.launch, waypoint0_x/y/z 命令行覆盖
#     (4) 目标按 world 系传入, 脚本内部换算 goal_odom = goal_world - spawn_xy
#   原 run_bc_flight.sh 未改。
#
#   用法:
#     bash run_task_flight.sh [world] [start_x] [start_y] [start_yaw] [start_z] \
#                             [gx_world] [gy_world] [gz] \
#                             [out_json] [tier] [world_id] [trial]
#   环境变量:
#     CKPT=<ckpt路径>  WITH_RVIZ=0/1(默认0)  TIMEOUT=1200  SITL_WAIT=120  PLANNER_WAIT=90
#     DRONE_RADIUS=0.35  DRONE_NAME=iris_D435i  SPAWN_Z=<出生高度, 默认0=地面>
#     GOAL_FRAME=world(默认)|odom    OBS_JSON=<障碍json, 默认 ${WORLD}.obstacles.json>
#     TAKEOFF_S=<offboard 起飞段秒数, 默认 30>
#   2026-11-07 P0/P6 三方法对比新增:
#     POLICY_TYPE=auto(默认,旧行为)|baseline|p6_lora
#     LORA_CKPT=<P6 carrier 目录, 默认空>      仅 POLICY_TYPE=p6_lora 需要
#     LORA_ENABLED=true(默认)|false            false→P0 base 分支, true→P6 LoRA 分支
#     ENGINE=bc(默认)|ego                      规划器换成 EGO 时用 ego;
#       起飞/交接/监控/清理时序与 BC 完全一致, 只有规划器那一步不同
#     EGO_LAUNCH=single_run_in_gazebo_task.launch
# ══════════════════════════════════════════════════════════════
set -u

SCRIPTS_DIR="/root/autodl-tmp/AUV/catkin_ws0/scripts"
START_SITL="$SCRIPTS_DIR/start_sitl_at.sh"
MONITOR="$SCRIPTS_DIR/collision_timeout_monitor.py"
LOG_DIR="${AUV_LOG_DIR:-/root/autodl-tmp/logs}"

DEFAULT_WORLD="/root/autodl-tmp/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds/ood_experiment_worlds/layerA/env_000_obs2.world"
WORLD="${1:-$DEFAULT_WORLD}"
SX="${2:-0.0}"; SY="${3:-0.0}"; SYAW="${4:-0.0}"; SZ="${5:-2.0}"
GXW="${6:-8.0}"; GYW="${7:-0.0}"; GZ="${8:-2.0}"
OUT_JSON="${9:-/root/autodl-tmp/AUV/results/bc_$(date +%Y%m%d_%H%M%S).json}"
TIER="${10:-bc_task}"; WORLD_ID="${11:-0}"; TRIAL="${12:-0}"

GOAL_FRAME="${GOAL_FRAME:-world}"
if [ "$GOAL_FRAME" = "world" ]; then
    GX=$(python3 -c "print(f'{$GXW - $SX:.10f}')")
    GY=$(python3 -c "print(f'{$GYW - $SY:.10f}')")
else
    GX="$GXW"; GY="$GYW"
fi

OBS_JSON="${OBS_JSON:-${WORLD}.obstacles.json}"
SPAWN_Z="${SPAWN_Z:-0}"
TAKEOFF_S="${TAKEOFF_S:-30}"

WITH_RVIZ="${WITH_RVIZ:-0}"
TIMEOUT="${TIMEOUT:-1200}"
DRONE_RADIUS="${DRONE_RADIUS:-0.35}"
CKPT="${CKPT:-/root/autodl-tmp/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy/data/outputs/2026.08.16/08.53.26_e5_lambda0p5/checkpoints/latest.ckpt}"
SITL_WAIT="${SITL_WAIT:-120}"
PLANNER_WAIT="${PLANNER_WAIT:-90}"
DRONE_NAME="${DRONE_NAME:-iris_D435i}"
CONDA_ENV="/root/autodl-tmp/conda_envs/idp3"
POLICY_TYPE="${POLICY_TYPE:-auto}"
LORA_CKPT="${LORA_CKPT:-}"
LORA_ENABLED="${LORA_ENABLED:-true}"
ENGINE="${ENGINE:-bc}"
EGO_LAUNCH="${EGO_LAUNCH:-single_run_in_gazebo_task.launch}"

PX4_DIR="/root/autodl-tmp/AUV/PX4-Autopilot"
CATKIN_WS="/root/autodl-tmp/AUV/catkin_ws0"
# ROS 的 setup 链会读未定义变量 (1.ros_distro.sh:3 / 10.roslaunch.sh:3), set -u 下致命
set +u
source /opt/ros/noetic/setup.bash
source "$CATKIN_WS/devel/setup.bash"
source "$PX4_DIR/Tools/setup_gazebo.bash" "$PX4_DIR" "$PX4_DIR/build/px4_sitl_default" 2>/dev/null
export ROS_PACKAGE_PATH="$ROS_PACKAGE_PATH:$PX4_DIR:$PX4_DIR/Tools/sitl_gazebo"
set -u

if [ "${CONDA_PREFIX:-}" != "$CONDA_ENV" ]; then
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate "$CONDA_ENV"
    source "$CATKIN_WS/devel/setup.bash"
fi

log() { echo "[run_task_flight] $*"; }

full_cleanup() {
    log "清理所有进程..."
    pkill -9 -f "collision_timeout_monitor" 2>/dev/null
    pkill -9 -f "ml_planner"                2>/dev/null
    # EGO 臂的残留 (bc 臂下这些 pattern 不匹配任何进程, 无副作用)
    pkill -9 -f "single_run_in_gazebo"      2>/dev/null
    pkill -9 -f "ego_planner_node"          2>/dev/null
    pkill -9 -f "ego_replan_fsm"            2>/dev/null
    pkill -9 -f "traj_server"               2>/dev/null
    pkill -9 -f "planner_manager"           2>/dev/null
    pkill -9 -f "odom_visualization"        2>/dev/null
    pkill -9 -f "px4ctrl"                   2>/dev/null
    pkill -9 -f "offboard_node"             2>/dev/null
    pkill -9 -f "rviz"                      2>/dev/null
    pkill -9 -f "mavproxy"                  2>/dev/null
    pkill -9 -f "fast_test.launch"          2>/dev/null
    pkill -9 -f "px4_sitl_default/bin/px4"  2>/dev/null
    pkill -9 -f "gzserver"                  2>/dev/null
    pkill -9 -f "gzclient"                  2>/dev/null
    pkill -9 -f "mavros_node"               2>/dev/null
    sleep 3
}

trap 'log "中断, 清理后退出"; full_cleanup; exit 130' SIGINT SIGTERM

echo "════════════════════════════════════════════════════════════"
echo "  BC TRIAL  tier=$TIER world_id=$WORLD_ID trial=$TRIAL"
echo "  world : $WORLD"
echo "  spawn : ($SX, $SY, z=$SPAWN_Z 地面) yaw=$SYAW   → odom 原点 = 起点"
echo "  takeoff target: odom (0, 0, $SZ)   [offboard_node_goto]"
echo "  goal  : world ($GXW, $GYW, $GZ)  → odom ($GX, $GY, $GZ)   frame=$GOAL_FRAME"
echo "  obs   : $OBS_JSON"
echo "  timeout=${TIMEOUT}s rviz=$WITH_RVIZ  ckpt=$CKPT"
echo "  engine=$ENGINE  policy_type=$POLICY_TYPE  lora_enabled=$LORA_ENABLED  lora_ckpt=${LORA_CKPT:-<none>}"
echo "  out   : $OUT_JSON"
echo "  python3: $(which python3) $(python3 -V 2>&1)"
echo "════════════════════════════════════════════════════════════"

[ -n "$OBS_JSON" ] && [ ! -f "$OBS_JSON" ] && { log "✗ obstacles_json 不存在: $OBS_JSON"; exit 2; }
[ -f "$WORLD" ]      || { log "✗ world 不存在: $WORLD"; exit 2; }
[ -f "$START_SITL" ] || { log "✗ start_sitl_at.sh 不存在"; exit 2; }
[ -f "$CKPT" ]       || { log "✗ ckpt 不存在: $CKPT"; exit 2; }
[ -f "$MONITOR" ]    || { log "✗ monitor 不存在"; exit 2; }
[ -d "$CATKIN_WS/src/offboard_pkg" ] || { log "✗ offboard_pkg 不存在"; exit 2; }
mkdir -p "$(dirname "$OUT_JSON")" "$LOG_DIR/mavproxy"

# ── 0. 清理 ────────────────────────────────────
full_cleanup

# ── 1. 启 SITL (出生点 = 任务起点 xy) ──────────
log "启动 SITL: $WORLD  spawn=($SX,$SY,$SPAWN_Z) yaw=$SYAW"
AUV_SPAWN_X="$SX" AUV_SPAWN_Y="$SY" AUV_SPAWN_Z="$SPAWN_Z" AUV_SPAWN_YAW="$SYAW" \
    bash "$START_SITL" "$WORLD"

# ── 1b. 地面站 (GCS 心跳; 缺它 AUTO.LOITER 必触发 no_rc_and_no_datalink) ──
log "启动 MAVProxy 地面站 (udpout:127.0.0.1:18570)..."
# 必须是 udpout: —— udp: 会去 bind 18570, 与 PX4 的 GCS 实例撞端口
mavproxy.py --master=udpout:127.0.0.1:18570 --daemon \
    --state-basedir="$LOG_DIR/mavproxy" \
    > "$LOG_DIR/mavproxy.log" 2>&1 < /dev/null &
sleep 5

# ── 2. 等 MAVROS ───────────────────────────────
log "等待 MAVROS connected (上限 ${SITL_WAIT}s)..."
t0=$(date +%s)
ok=0
while [ $(( $(date +%s) - t0 )) -lt "$SITL_WAIT" ]; do
    state=$(timeout 3 rostopic echo -n1 /mavros/state 2>/dev/null | grep -m1 "connected:")
    if echo "$state" | grep -q "True"; then ok=1; break; fi
    sleep 3
done
if [ "$ok" != "1" ]; then
    log "✗ 等 MAVROS 超时, 放弃此 trial"
    full_cleanup
    echo "{\"tier\":\"$TIER\",\"world_id\":\"$WORLD_ID\",\"trial\":$TRIAL,\"setup_failed\":\"mavros_timeout\",\"success_deploy\":false,\"collision\":false,\"timeout\":false}" > "$OUT_JSON"
    exit 3
fi
log "✓ MAVROS connected"
sleep 5

# ── 2b. COM_RCL_EXCEPT=4 ───────────────────────
get_rcl_except() {
    rosservice call /mavros/param/get "param_id: 'COM_RCL_EXCEPT'" 2>/dev/null \
        | grep -m1 "integer:" | grep -oE '[-]?[0-9]+' | head -1
}

log "设置 COM_RCL_EXCEPT=4, 上限 120s..."
ok=0; t0=$(date +%s); rcl=""; tries=0
while [ $(( $(date +%s) - t0 )) -lt 120 ]; do
    rcl=$(get_rcl_except)
    if [ "$rcl" = "4" ]; then ok=1; break; fi
    if [ -n "$rcl" ]; then
        tries=$((tries + 1))
        rosrun mavros mavparam set COM_RCL_EXCEPT 4 >/dev/null 2>&1
    fi
    sleep 3
done
if [ "$ok" != "1" ]; then
    log "✗ COM_RCL_EXCEPT 未能置为 4 (读回 '${rcl:-<空>}', 尝试 ${tries} 次), 放弃此 trial"
    full_cleanup
    echo "{\"tier\":\"$TIER\",\"world_id\":\"$WORLD_ID\",\"trial\":$TRIAL,\"setup_failed\":\"com_rcl_except_failed\",\"success_deploy\":false,\"collision\":false,\"timeout\":false}" > "$OUT_JSON"
    exit 7
fi
log "✓ COM_RCL_EXCEPT = $rcl (写 ${tries} 次)"

# ── 3. offboard 起飞到起点高度 ─────────────────
log "offboard 起飞 (target odom 0,0,$SZ), 飞行 ${TAKEOFF_S}s, 等 $((TAKEOFF_S + 10))s..."
rosrun offboard_pkg offboard_node_goto "$TAKEOFF_S" 0 0 "$SZ" &
OFFB_PID=$!
sleep $((TAKEOFF_S + 10))

if kill -0 $OFFB_PID 2>/dev/null; then
    log "  ⚠ offboard_node_goto 仍存活, 兜底 pkill"
else
    log "  ✓ offboard_node_goto 已自行退出 (定时器 → AUTO.LOITER → shutdown)"
fi
pkill -f "offboard_node" 2>/dev/null
sleep 2

mode=$(timeout 2 rostopic echo -n1 /mavros/state 2>/dev/null | grep -m1 "mode:" | grep -oE "[A-Z.]+" | tail -1)
case "$mode" in
    AUTO.LOITER|LOITER) log "  ✓ 当前模式: $mode" ;;
    *)                  log "  ⚠ 当前模式: ${mode:-unknown} (非 LOITER)" ;;
esac

# 起飞到位自检: odom 应停在 (0, 0, SZ) 附近
tp=$(timeout 3 rostopic echo -n1 /mavros/local_position/odom 2>/dev/null | sed -n '/position:/,/orientation:/p' | grep -oE "[-0-9.]+" | head -3 | tr '\n' ' ')
log "  起飞后 odom 位置: (${tp:-<读不到>})  期望 (0, 0, $SZ)"

# ── 4. px4ctrl → dynparam → 规划器 → rviz (ml_gazebo.sh:55-88 原顺序) ──
log "启动 px4ctrl..."
roslaunch px4ctrl run_node.launch &
PX4CTRL_PID=$!
log "  px4ctrl PID: $PX4CTRL_PID, 等 12s..."
sleep 12

log "配置 px4ctrl..."
rosrun dynamic_reconfigure dynparam set /px4ctrl mode_bool False; sleep 2
rosrun dynamic_reconfigure dynparam set /px4ctrl cmd_bool  False; sleep 2
rosrun dynamic_reconfigure dynparam set /px4ctrl mode_bool True;  sleep 3
rosrun dynamic_reconfigure dynparam set /px4ctrl cmd_bool  True;  sleep 3
log "✓ px4ctrl 配置完成"

# PROF_LAUNCH: 诊断用, 指向带计时的 launch 副本; 留空则用正式 launch
PLANNER_LAUNCH="${PROF_LAUNCH:-ml_planner_gazebo_task.launch}"
# ENGINE 只切换"规划器"这一步; 起飞、交接、监控、清理全链路共用
if [ "$ENGINE" = "ego" ]; then
    # EGO: goal 只能经 point0_x/y/z 传入 (flight_type=2 全局航点),
    # 故必须用参数化副本 single_run_in_gazebo_task.launch —— 原文件是 value= 写死 16/0/2
    log "启动 EGO planner, launch=$EGO_LAUNCH, goal odom=($GX, $GY, $GZ)..."
    roslaunch ego_planner "$EGO_LAUNCH" \
        point0_x:="$GX" point0_y:="$GY" point0_z:="$GZ" &
    PLANNER_PID=$!
    log "  EGO PID: $PLANNER_PID, 等 10s..."
    sleep 10
else
    log "启动 ML Planner (BC/$POLICY_TYPE), launch=$PLANNER_LAUNCH..."
    roslaunch ego_planner "$PLANNER_LAUNCH" \
        ckpt:="$CKPT" waypoint0_x:="$GX" waypoint0_y:="$GY" waypoint0_z:="$GZ" \
        policy_type:="$POLICY_TYPE" lora_enabled:="$LORA_ENABLED" \
        lora_ckpt:="$LORA_CKPT" &
    PLANNER_PID=$!
    log "  ML Planner PID: $PLANNER_PID, 等 10s..."
    sleep 10
fi

if [ "$WITH_RVIZ" = "1" ]; then
    rosrun rviz rviz -d "$(rospack find ego_planner)/launch/default_test.rviz" &
    log "  RViz 已起, 等 3s..."
    sleep 3
fi

# ── 5. 交接自检 ────────────────────────────────
# 注意: 只查 "有 publisher" 不够 —— 无 DISPLAY 时话题照样 advertise 但 0 条消息,
# 规划器会一直 "积累观测: 0/3"。这里必须实际收到一条点云才算通过。
if [ "$ENGINE" = "ego" ]; then
    # EGO 建图消费的是深度图 (image_raw), 不是 ML planner 用的 points
    PC_TOPIC="/iris_D435i/realsense/depth_camera/depth/image_raw"
    PC_DESC="深度图"
else
    PC_TOPIC="/iris_D435i/realsense/depth_camera/depth/points"
    PC_DESC="深度点云"
fi
log "等${PC_DESC}实际出流 (${PC_TOPIC}, 上限 60s)..."
ok=0; t0=$(date +%s)
while [ $(( $(date +%s) - t0 )) -lt 60 ]; do
    if timeout 8 rostopic echo -n1 "$PC_TOPIC" >/dev/null 2>&1; then ok=1; break; fi
    sleep 2
done
if [ "$ok" != "1" ]; then
    log "✗ 深度点云 60s 内无消息。多半是 gzserver 没有 DISPLAY (深度相机渲染不出来)"
    log "  当前 DISPLAY='${DISPLAY:-<unset>}'; 可设 DISPLAY=:1 或 AUV_DISPLAY 后重试"
    full_cleanup
    echo "{\"tier\":\"$TIER\",\"world_id\":\"$WORLD_ID\",\"trial\":$TRIAL,\"setup_failed\":\"no_pointcloud\",\"success_deploy\":false}" > "$OUT_JSON"
    exit 6
fi
log "  ✓ ${PC_DESC}已出流"

if [ "$ENGINE" = "ego" ]; then
    # EGO 必须在深度图建好之后才开始规划, 否则把机周误判为障碍 ("drone is in obstacle").
    # run_trial_ego.sh 里单 trial 靠 RViz 启动时间凑这 12s, 批量 RViz=0 必须显式补
    log "EGO 建图缓冲 (12s)..."
    sleep 12
    # flight_type=1 (MANUAL_TARGET): 目标靠 /move_base_simple/goal 运行时下发.
    # 位姿直接就是 FSM 的 odom 系坐标 (waypointCallback 不做 TF 变换).
    log "发布 EGO 目标 /move_base_simple/goal (odom $GX, $GY, $GZ)..."
    rostopic pub -1 /move_base_simple/goal geometry_msgs/PoseStamped \
        "{header: {frame_id: 'world'}, pose: {position: {x: $GX, y: $GY, z: $GZ}, orientation: {w: 1.0}}}" \
        >/dev/null 2>&1
    sleep 3
fi

mode=$(timeout 2 rostopic echo -n1 /mavros/state 2>/dev/null | grep -m1 "mode:" | grep -oE "[A-Z.]+" | tail -1)
case "$mode" in
    OFFBOARD) log "  ✓ 当前模式 OFFBOARD" ;;
    *)        log "  ⚠ 当前模式 ${mode:-unknown} (非 OFFBOARD)" ;;
esac

# ── 6. 等 /planning/pos_cmd ────────────────────
log "等待 /planning/pos_cmd publisher (上限 ${PLANNER_WAIT}s)..."
t0=$(date +%s)
ok=0
while [ $(( $(date +%s) - t0 )) -lt "$PLANNER_WAIT" ]; do
    if rostopic info /planning/pos_cmd 2>/dev/null | grep -A2 "Publishers:" | grep -q "ml_planner\|drone_"; then
        ok=1; break
    fi
    pubcount=$(rostopic info /planning/pos_cmd 2>/dev/null | sed -n '/Publishers:/,/Subscribers:/p' | grep -c " \* ")
    if [ "${pubcount:-0}" -ge 1 ]; then ok=1; break; fi
    sleep 3
done
if [ "$ok" != "1" ]; then
    log "✗ 等 pos_cmd 超时, 放弃"
    full_cleanup
    echo "{\"tier\":\"$TIER\",\"world_id\":\"$WORLD_ID\",\"trial\":$TRIAL,\"setup_failed\":\"no_pos_cmd\",\"success_deploy\":false}" > "$OUT_JSON"
    exit 5
fi
log "✓ pos_cmd 已发布, 开始监控"

# ── 7. monitor ─────────────────────────────────
log "启动 monitor (timeout=${TIMEOUT}s, goal odom=$GX,$GY,$GZ)..."
python3 "$MONITOR" \
    --obstacles_json "$OBS_JSON" \
    --goal_x "$GX" --goal_y "$GY" --goal_z "$GZ" \
    --drone_radius "$DRONE_RADIUS" \
    --timeout "$TIMEOUT" \
    --output "$OUT_JSON" \
    --tier "$TIER" --world_id "$WORLD_ID" --trial "$TRIAL" \
    --filename "$(basename "$WORLD")" \
    --drone_name "$DRONE_NAME"
MON_RC=$?
log "monitor 退出 (rc=$MON_RC, 以 JSON 为准)"

# ── 8. 清理 ────────────────────────────────────
full_cleanup

# ── 9. 结果 ────────────────────────────────────
if [ -f "$OUT_JSON" ]; then
    log "结果:"
    python3 -c "
import json
d=json.load(open('$OUT_JSON'))
print('  success_deploy =', d.get('success_deploy'))
print('  success_fb     =', d.get('success_flightbench'))
print('  collision      =', d.get('collision'))
print('  timeout        =', d.get('timeout'))
print('  final_dxy      = %.2f m' % d.get('final_dist_xy',0))
print('  min_clearance  = %.2f m' % d.get('min_clearance',0))
print('  time_to_goal   = %.1f s' % d.get('time_to_goal',0))
"
else
    log "✗ 未生成 JSON: $OUT_JSON"
fi

log "TRIAL 完成: $OUT_JSON"
exit 0
