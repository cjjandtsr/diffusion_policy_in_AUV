#!/bin/bash
# run_e1_judge.sh — E1 单因子判定: per-trial 完整 SITL 重启
# 对齐 run_trial.sh 模式: 每 trial = cleanup → SITL → MAVROS → takeoff → rollout → cleanup
# 用法: bash run_e1_judge.sh [world_id] [n_trials]
set -u

WORLD_ID="${1:-4}"
N_TRIALS="${2:-3}"
AUV_ROOT="/root/autodl-tmp/AUV"
SCRIPTS="$AUV_ROOT/catkin_ws0/scripts"
FM_DIR="$AUV_ROOT/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy"
CKPT="data/outputs/2026.06.17/15.22.50_v9_2_nobs8_bc/checkpoints/best.ckpt"
OUT_DIR="$AUV_ROOT/program/shiyan/results_synthpc_judge"

declare -A WORLD_MAP=(
    [4]="env_004_obs19" [8]="env_008_obs20"
    [11]="env_011_obs23" [12]="env_012_obs22" [18]="env_018_obs25"
)
WORLD_NAME="${WORLD_MAP[$WORLD_ID]}"
WORLD_FILE="$AUV_ROOT/PX4-Autopilot/Tools/sitl_gazebo/worlds/ood_experiment_worlds/layerA/${WORLD_NAME}.world"

PX4_DIR="$AUV_ROOT/PX4-Autopilot"
CATKIN_WS="$AUV_ROOT/catkin_ws0"
source /opt/ros/noetic/setup.bash
source "$CATKIN_WS/devel/setup.bash"
source "$PX4_DIR/Tools/setup_gazebo.bash" "$PX4_DIR" "$PX4_DIR/build/px4_sitl_default" 2>/dev/null
export ROS_PACKAGE_PATH="$ROS_PACKAGE_PATH:$PX4_DIR:$PX4_DIR/Tools/sitl_gazebo"

mkdir -p "$OUT_DIR"

# ── 清理 (对齐 run_trial.sh full_cleanup) ──
full_cleanup() {
    echo "[E1] full_cleanup..."
    pkill -9 -f "rollout_gazebo_eval"      2>/dev/null || true
    pkill -9 -f "px4ctrl"                  2>/dev/null || true
    pkill -9 -f "offboard_node"            2>/dev/null || true
    pkill -9 -f "fast_test.launch"         2>/dev/null || true
    pkill -9 -f "px4_sitl_default/bin/px4" 2>/dev/null || true
    pkill -9 -f "gzserver"                 2>/dev/null || true
    pkill -9 -f "gzclient"                 2>/dev/null || true
    pkill -9 -f "mavros_node"              2>/dev/null || true
    sleep 3
}

trap 'echo "[E1] interrupted"; full_cleanup; exit 130' SIGINT SIGTERM

echo "============================================================"
echo "  E1: GazeboUAVEnv + synth_pc 单因子判定"
echo "  world=$WORLD_ID ($WORLD_NAME)  n_trials=$N_TRIALS"
echo "  mode: per-trial full SITL restart (对齐 run_trial.sh)"
echo "============================================================"

n_ok=0
for t in $(seq 0 $((N_TRIALS - 1))); do
    echo ""
    echo "################################################################"
    echo "# E1 trial $((t+1))/$N_TRIALS"
    echo "################################################################"

    # ── 0. 清理上一轮 ──
    full_cleanup

    # ── 1. 启 SITL ──
    echo "[E1] starting SITL: $WORLD_FILE"
    bash "$SCRIPTS/start_sitl_rl.sh" "$WORLD_FILE" &
    SITL_PID=$!

    # ── 2. 等 MAVROS ──
    echo "[E1] waiting MAVROS (max 90s)..."
    ok=0
    for i in $(seq 1 90); do
        if timeout 2 rostopic list 2>/dev/null | grep -q "/mavros/state"; then
            echo "[E1] MAVROS online (${i}s)"
            ok=1; break
        fi
        sleep 1
    done
    if [ "$ok" != "1" ]; then
        echo "[E1] FAIL: MAVROS timeout"
        full_cleanup
        continue
    fi
    sleep 5

    # ── 3. 起飞 ──
    echo "[E1] takeoff..."
    bash "$SCRIPTS/takeoff_only.sh"

    # ── 4. 单 trial rollout ──
    echo "[E1] rollout trial $t..."
    cd "$FM_DIR"
    python "$SCRIPTS/rollout_gazebo_eval.py" \
        --ckpt "$CKPT" \
        --world_id "$WORLD_ID" \
        --trial_idx "$t" \
        --out_dir "$OUT_DIR" \
        --device cuda:0

    # ── 5. 清理 ──
    full_cleanup

    # ── 6. 读结果 (快速判定) ──
    out_json="$OUT_DIR/E1_gazebo_synthpc_w${WORLD_ID}_t${t}.json"
    if [ -f "$out_json" ]; then
        python3 -c "
import json
d=json.load(open('$out_json'))
print(f'  reached={d.get(\"reached\")}  done={d.get(\"done_reason\")}  steps={d.get(\"n_steps\")}')
print(f'  dpos={d.get(\"per_step_dpos\",[])}')
"
        reached=$(python3 -c "import json;print(json.load(open('$out_json')).get('reached'))")
        [ "$reached" = "True" ] && n_ok=$((n_ok + 1))
    else
        echo "  FAIL: no JSON output"
    fi

    sleep 3
done

# ── 汇总 ──
echo ""
echo "============================================================"
echo "  E1 complete: $n_ok / $N_TRIALS reached"
echo "  results: $OUT_DIR/E1_gazebo_synthpc_w${WORLD_ID}_t*.json"
echo "============================================================"

# 合并为 summary JSON
python3 -c "
import json, glob, os
files = sorted(glob.glob('$OUT_DIR/E1_gazebo_synthpc_w${WORLD_ID}_t*.json'))
trials = [json.load(open(f)) for f in files]
reach_rate = sum(1 for t in trials if t.get('reached')) / len(trials) if trials else 0
coll_rate = sum(1 for t in trials if 'collision' in str(t.get('done_reason',''))) / len(trials) if trials else 0
dpos_s3 = [t['per_step_dpos'][2] for t in trials if len(t.get('per_step_dpos',[])) > 2]
summary = {
    'env': 'GazeboUAVEnv', 'pointcloud': 'synth_pc',
    'world': $WORLD_ID, 'n_trials': len(trials),
    'reach_rate': reach_rate, 'collision_rate': coll_rate,
}
if dpos_s3:
    summary['mean_dpos_step3'] = sum(dpos_s3)/len(dpos_s3)
    summary['dpos_step3_values'] = dpos_s3
summary['verdict'] = 'ROOT_CAUSE_A_synth_pc_OOD' if coll_rate > 0 else 'ROOT_CAUSE_B_history_or_execution'
out = {'summary': summary, 'trials': trials}
with open('$OUT_DIR/E1_gazebo_synthpc_w${WORLD_ID}.json', 'w') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(json.dumps(summary, indent=2))
"
