#!/bin/bash
# ══════════════════════════════════════════════════════════════
# run_bc_flight.sh — BC 单次闭环 trial 全自动 (服务器)
#   骨架 = run_trial.sh; 起飞/交接段换成 ego_gazebo.sh:49-71 的顺序
#
#   用法:
#     bash run_bc_flight.sh [world] [obstacles_json] [gx] [gy] [gz] [out_json] [tier] [world_id] [trial]
#   环境变量:
#     CKPT=<ckpt路径>  WITH_RVIZ=0/1(默认1)  TIMEOUT=1200  SITL_WAIT=120  PLANNER_WAIT=90
#     DRONE_RADIUS=0.35  DRONE_NAME=iris_D435i
# ══════════════════════════════════════════════════════════════
set -u

SCRIPTS_DIR="/root/autodl-tmp/AUV/catkin_ws0/scripts"
START_SITL="$SCRIPTS_DIR/start_sitl.sh"
MONITOR="$SCRIPTS_DIR/collision_timeout_monitor.py"
LOG_DIR="${AUV_LOG_DIR:-/root/autodl-tmp/logs}"

DEFAULT_WORLD="/root/autodl-tmp/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds/ood_experiment_worlds/layerA/env_000_obs2.world"
WORLD="${1:-$DEFAULT_WORLD}"
OBS_JSON="${2:-${WORLD%.world}.obstacles.json}"
GX="${3:-8.0}"; GY="${4:-0.0}"; GZ="${5:-2.0}"
OUT_JSON="${6:-/root/autodl-tmp/AUV/results/bc_$(date +%Y%m%d_%H%M%S).json}"
TIER="${7:-bc_layerA}"; WORLD_ID="${8:-0}"; TRIAL="${9:-0}"

WITH_RVIZ="${WITH_RVIZ:-1}"
TIMEOUT="${TIMEOUT:-1200}"
DRONE_RADIUS="${DRONE_RADIUS:-0.35}"
CKPT="${CKPT:-/root/autodl-tmp/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy/data/outputs/2026.08.16/08.53.26_e5_lambda0p5/checkpoints/latest.ckpt}"
SITL_WAIT="${SITL_WAIT:-120}"
PLANNER_WAIT="${PLANNER_WAIT:-90}"
DRONE_NAME="${DRONE_NAME:-iris_D435i}"
CONDA_ENV="/root/autodl-tmp/conda_envs/idp3"

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

log() { echo "[run_bc_flight] $*"; }

full_cleanup() {
    log "清理所有进程..."
    pkill -9 -f "collision_timeout_monitor" 2>/dev/null
    pkill -9 -f "ml_planner"                2>/dev/null
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
echo "  world: $WORLD"
echo "  goal : ($GX, $GY, $GZ)   timeout=${TIMEOUT}s rviz=$WITH_RVIZ"
echo "  ckpt : $CKPT"
echo "  out  : $OUT_JSON"
echo "  python3: $(which python3) $(python3 -V 2>&1)"
echo "════════════════════════════════════════════════════════════"

[ -f "$WORLD" ]      || { log "✗ world 不存在: $WORLD"; exit 2; }
[ -f "$OBS_JSON" ]   || { log "✗ obstacles_json 不存在: $OBS_JSON"; exit 2; }
[ -f "$START_SITL" ] || { log "✗ start_sitl.sh 不存在"; exit 2; }
[ -f "$CKPT" ]       || { log "✗ ckpt 不存在: $CKPT"; exit 2; }
[ -f "$MONITOR" ]    || { log "✗ monitor 不存在"; exit 2; }
[ -d "$CATKIN_WS/src/offboard_pkg" ] || { log "✗ offboard_pkg 不存在"; exit 2; }
mkdir -p "$(dirname "$OUT_JSON")" "$LOG_DIR/mavproxy"

# ── 0. 清理 ────────────────────────────────────
full_cleanup

# ── 1. 启 SITL ─────────────────────────────────
log "启动 SITL: $WORLD"
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
    echo "{\"tier\":\"$TIER\",\"world_id\":$WORLD_ID,\"trial\":$TRIAL,\"setup_failed\":\"mavros_timeout\",\"success_deploy\":false,\"collision\":false,\"timeout\":false}" > "$OUT_JSON"
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
    echo "{\"tier\":\"$TIER\",\"world_id\":$WORLD_ID,\"trial\":$TRIAL,\"setup_failed\":\"com_rcl_except_failed\",\"success_deploy\":false,\"collision\":false,\"timeout\":false}" > "$OUT_JSON"
    exit 7
fi
log "✓ COM_RCL_EXCEPT = $rcl (写 ${tries} 次)"

# ── 3. offboard 起飞 (节点自行收尾) ────────────
log "offboard 起飞, 飞行 30s, 等 40s..."
rosrun offboard_pkg offboard_node 30 &
OFFB_PID=$!
sleep 40

if kill -0 $OFFB_PID 2>/dev/null; then
    log "  ⚠ offboard_node 仍存活, 兜底 pkill"
else
    log "  ✓ offboard_node 已自行退出 (定时器 → AUTO.LOITER → shutdown)"
fi
pkill -f "offboard_node" 2>/dev/null
sleep 2

mode=$(timeout 2 rostopic echo -n1 /mavros/state 2>/dev/null | grep -m1 "mode:" | grep -oE "[A-Z.]+" | tail -1)
case "$mode" in
    AUTO.LOITER|LOITER) log "  ✓ 当前模式: $mode" ;;
    *)                  log "  ⚠ 当前模式: ${mode:-unknown} (非 LOITER)" ;;
esac

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

log "启动 ML Planner (BC)..."
# ml_planner_gazebo.launch:6-8 的 waypoint 用 value= 写死, 命令行不能覆盖, GX/GY/GZ 只喂 monitor
roslaunch ego_planner ml_planner_gazebo.launch ckpt:="$CKPT" &
ML_PLANNER_PID=$!
log "  ML Planner PID: $ML_PLANNER_PID, 等 10s..."
sleep 10

if [ "$WITH_RVIZ" = "1" ]; then
    rosrun rviz rviz -d "$(rospack find ego_planner)/launch/default_test.rviz" &
    log "  RViz 已起, 等 3s..."
    sleep 3
fi

# ── 5. 交接自检 ────────────────────────────────
log "等深度相机 topic..."
t0=$(date +%s)
while [ $(( $(date +%s) - t0 )) -lt 30 ]; do
    if rostopic info /iris_D435i/realsense/depth_camera/depth/image_raw 2>/dev/null | grep -q "Publishers:"; then
        log "  ✓ 深度相机话题有 publisher"; break
    fi
    sleep 2
done

mode=$(timeout 2 rostopic echo -n1 /mavros/state 2>/dev/null | grep -m1 "mode:" | grep -oE "[A-Z.]+" | tail -1)
case "$mode" in
    OFFBOARD) log "  ✓ 已回到 OFFBOARD" ;;
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
    echo "{\"tier\":\"$TIER\",\"world_id\":$WORLD_ID,\"trial\":$TRIAL,\"setup_failed\":\"no_pos_cmd\",\"success_deploy\":false}" > "$OUT_JSON"
    exit 5
fi
log "✓ pos_cmd 已发布, 开始监控"

# ── 7. monitor ─────────────────────────────────
log "启动 monitor (timeout=${TIMEOUT}s)..."
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
import json,sys
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
