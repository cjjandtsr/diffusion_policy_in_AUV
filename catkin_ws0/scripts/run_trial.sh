#!/bin/bash
# ══════════════════════════════════════════════════════════════
# run_trial.sh — 单次闭环 trial 全自动
#   用法:
#     bash run_trial.sh <world> <obstacles_json> <gx> <gy> <gz> <out_json> [tier] [world_id] [trial]
#   示例:
#     bash run_trial.sh \
#       .../layerA/env_000_obs1.world \
#       .../layerA/env_000_obs1.obstacles.json \
#       12.0 0.0 2.0 /root/autodl-tmp/AUV/results/layerA_w0_t0.json \
#       layerA 0 0
#
#   环境: 整脚本在 idp3 conda 环境跑 (monitor 也在 idp3)
#   流程: 清理 → SITL(后台) → 等MAVROS → ml_planner(后台) → 等pos_cmd
#         → monitor(前台 wait, 自己退出) → 清理 → 读 JSON 报结果
#
#   可调环境变量:
#     WITH_RVIZ=0/1     (默认0)
#     TIMEOUT=60        (monitor 每 waypoint 超时秒, 默认 60)
#     DRONE_RADIUS=0.35
#     CKPT=<路径>       (空=launch 默认)
#     SITL_WAIT=120     (等 MAVROS 上限秒)
#     PLANNER_WAIT=90   (等 pos_cmd 上限秒)
#     DRONE_NAME=iris_D435i
# ══════════════════════════════════════════════════════════════
set -u

# ── 路径常量 (服务器) ──
SCRIPTS_DIR="/root/autodl-tmp/AUV/catkin_ws0/scripts"
START_SITL="$SCRIPTS_DIR/start_sitl.sh"
ML_AUTO="$SCRIPTS_DIR/ml_gazebo_fm_auto.sh"
MONITOR="$SCRIPTS_DIR/collision_timeout_monitor.py"

# ── 参数 ──
WORLD="${1:?need world file}"
OBS_JSON="${2:?need obstacles_json}"
GX="${3:?need goal_x}"
GY="${4:?need goal_y}"
GZ="${5:?need goal_z}"
OUT_JSON="${6:?need output json}"
TIER="${7:-unknown}"
WORLD_ID="${8:-0}"
TRIAL="${9:-0}"

# ── 可调 ──
WITH_RVIZ="${WITH_RVIZ:-0}"
TIMEOUT="${TIMEOUT:-1200}"
DRONE_RADIUS="${DRONE_RADIUS:-0.35}"
CKPT="${CKPT:-}"
SITL_WAIT="${SITL_WAIT:-120}"
PLANNER_WAIT="${PLANNER_WAIT:-90}"
DRONE_NAME="${DRONE_NAME:-iris_D435i}"

# ── 统一 source ROS + catkin workspace + PX4 (子步骤继承: ml_planner / monitor) ──
# 注意: 不在脚本里 conda activate, 继承调用者已激活的 idp3 即可
PX4_DIR="/root/autodl-tmp/AUV/PX4-Autopilot"
CATKIN_WS="/root/autodl-tmp/AUV/catkin_ws0"
source /opt/ros/noetic/setup.bash
source "$CATKIN_WS/devel/setup.bash"
source "$PX4_DIR/Tools/setup_gazebo.bash" "$PX4_DIR" "$PX4_DIR/build/px4_sitl_default" 2>/dev/null
export ROS_PACKAGE_PATH="$ROS_PACKAGE_PATH:$PX4_DIR:$PX4_DIR/Tools/sitl_gazebo"

log() { echo "[run_trial] $*"; }

# ── 全面清理函数 ──
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

# 捕获中断, 保证清理
trap 'log "中断, 清理后退出"; full_cleanup; exit 130' SIGINT SIGTERM

echo "════════════════════════════════════════════════════════════"
echo "  TRIAL  tier=$TIER world_id=$WORLD_ID trial=$TRIAL"
echo "  world: $WORLD"
echo "  goal : ($GX, $GY, $GZ)   timeout=${TIMEOUT}s rviz=$WITH_RVIZ"
echo "  out  : $OUT_JSON"
echo "════════════════════════════════════════════════════════════"

# 文件存在性预检 (避免跑到一半才发现路径错)
[ -f "$WORLD" ]    || { log "✗ world 不存在: $WORLD"; exit 2; }
[ -f "$OBS_JSON" ] || { log "✗ obstacles_json 不存在: $OBS_JSON"; exit 2; }
[ -f "$START_SITL" ] || { log "✗ start_sitl.sh 不存在"; exit 2; }
[ -f "$ML_AUTO" ]    || { log "✗ ml_gazebo_fm_auto.sh 不存在"; exit 2; }
[ -f "$MONITOR" ]    || { log "✗ monitor 不存在"; exit 2; }
mkdir -p "$(dirname "$OUT_JSON")"

# ── 0. 清理上一轮 ──
full_cleanup

# ── 1. 启 SITL (后台) ──
log "启动 SITL: $WORLD"
bash "$START_SITL" "$WORLD"

# ── 2. 等 MAVROS connected:True ──
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
    # 写一个失败占位 JSON, 让 batch 不中断
    echo "{\"tier\":\"$TIER\",\"world_id\":$WORLD_ID,\"trial\":$TRIAL,\"setup_failed\":\"mavros_timeout\",\"success_deploy\":false,\"collision\":false,\"timeout\":false}" > "$OUT_JSON"
    exit 3
fi
log "✓ MAVROS connected"
sleep 5  # 给 SITL 多稳一会

# ── 3. 启 ml_planner (后台, 含 offboard 起飞 + px4ctrl) ──
log "启动 ML Planner (FM)... (offboard 起飞约 40s)"
GOAL_X="$GX" GOAL_Y="$GY" GOAL_Z="$GZ" WITH_RVIZ="$WITH_RVIZ" \
    bash "$ML_AUTO" "$CKPT"
if [ $? -ne 0 ]; then
    log "✗ ml_gazebo_fm_auto.sh 返回非零, 放弃"
    full_cleanup
    echo "{\"tier\":\"$TIER\",\"world_id\":$WORLD_ID,\"trial\":$TRIAL,\"setup_failed\":\"planner_launch\",\"success_deploy\":false}" > "$OUT_JSON"
    exit 4
fi

# ── 4. 等 /planning/pos_cmd 有 publisher (确认 planner 真在出规划) ──
log "等待 /planning/pos_cmd publisher (上限 ${PLANNER_WAIT}s)..."
t0=$(date +%s)
ok=0
while [ $(( $(date +%s) - t0 )) -lt "$PLANNER_WAIT" ]; do
    if rostopic info /planning/pos_cmd 2>/dev/null | grep -A2 "Publishers:" | grep -q "ml_planner\|drone_"; then
        ok=1; break
    fi
    # 兜底: 只要有任意 publisher 也算
    pubcount=$(rostopic info /planning/pos_cmd 2>/dev/null | sed -n '/Publishers:/,/Subscribers:/p' | grep -c " \* ")
    if [ "${pubcount:-0}" -ge 1 ]; then ok=1; break; fi
    sleep 3
done
if [ "$ok" != "1" ]; then
    log "✗ 等 pos_cmd 超时 (planner 没出规划), 放弃"
    full_cleanup
    echo "{\"tier\":\"$TIER\",\"world_id\":$WORLD_ID,\"trial\":$TRIAL,\"setup_failed\":\"no_pos_cmd\",\"success_deploy\":false}" > "$OUT_JSON"
    exit 5
fi
log "✓ pos_cmd 已发布, 开始监控"

# ── 5. monitor (前台 wait, 自己退出 = trial 结束) ──
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
log "monitor 退出 (rc=$MON_RC, 注意 rc 恒为0, 以 JSON 为准)"

# ── 6. 清理 ──
full_cleanup

# ── 7. 读 JSON 报结果 ──
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