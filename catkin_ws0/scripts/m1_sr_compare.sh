#!/bin/bash
# ══════════════════════════════════════════════════════════════
# M1 闭环验证 — 薄 wrapper, 复用 run_batch.sh 基础设施
#   1. 生成临时 trials 表
#   2. 跑 N trial 真实渲染
#   3. 跑 N trial 合成点云
#   4. 汇总 SR gap
#
# 用法:
#   USE_SYNTH_PC=0 → 真实渲染, =1 → 合成点云, 由本脚本内设
#   bash m1_sr_compare.sh <world> <obstacles> <ckpt> <goal_x> <goal_y> <goal_z> [N]
# ══════════════════════════════════════════════════════════════
set -u

WORLD="${1:?need world file}"
OBS_JSON="${2:?need obstacles json}"
CKPT="${3:?need ckpt path}"
GX="${4:-12.0}"
GY="${5:-0.0}"
GZ="${6:-2.0}"
N="${7:-5}"

SCRIPTS="/root/autodl-tmp/AUV/catkin_ws0/scripts"
RUN_BATCH="$SCRIPTS/run_batch.sh"
RUN_TRIAL="$SCRIPTS/run_trial.sh"
OUT_BASE="/root/autodl-tmp/AUV/phase_beta/stage0/m1"
WNAME=$(basename "$WORLD" .world)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 生成临时 trials 表
TRIALS_FILE=$(mktemp /tmp/m1_trials_XXXXXX.txt)
for i in $(seq 1 $N); do
    echo "m1 $i $i $WORLD $OBS_JSON $GX $GY $GZ" >> "$TRIALS_FILE"
done

echo "========================================================================"
echo "  M1: $WNAME  (ckpt=$CKPT, N=$N)"
echo "========================================================================"

# ── Phase A: 真实渲染 ──
echo ""
echo "── Phase A: 真实渲染 ──"
RES_REAL="$OUT_BASE/${WNAME}_real"
mkdir -p "$RES_REAL"
CKPT="$CKPT" WITH_RVIZ=0 TIMEOUT=300 \
    bash "$RUN_BATCH" "$TRIALS_FILE" "$RES_REAL"

# ── Phase B: 合成点云 ──
echo ""
echo "── Phase B: 合成点云 ──"
RES_SYNTH="$OUT_BASE/${WNAME}_synth"
mkdir -p "$RES_SYNTH"
CKPT="$CKPT" WITH_RVIZ=0 TIMEOUT=300 \
    USE_SYNTH_PC=true SYNTH_OBS_JSON="$OBS_JSON" SYNTH_SPAWN_OFFSET="-8,0,0" \
    bash "$RUN_BATCH" "$TRIALS_FILE" "$RES_SYNTH"

# ── 汇总 ──
REAL_OK=0; SYNTH_OK=0
for f in "$RES_REAL"/m1_w*_t*.json; do
    [ -f "$f" ] || continue
    succ=$(python3 -c "import json; print(json.load(open('$f')).get('success_deploy',''))")
    [ "$succ" = "True" ] && REAL_OK=$((REAL_OK+1))
done
for f in "$RES_SYNTH"/m1_w*_t*.json; do
    [ -f "$f" ] || continue
    succ=$(python3 -c "import json; print(json.load(open('$f')).get('success_deploy',''))")
    [ "$succ" = "True" ] && SYNTH_OK=$((SYNTH_OK+1))
done

SR_REAL=$(python3 -c "print(round($REAL_OK/$N, 3))")
SR_SYNTH=$(python3 -c "print(round($SYNTH_OK/$N, 3))")
GAP=$(python3 -c "print(round(abs($SR_REAL-$SR_SYNTH), 3))")
VERDICT=$(python3 -c "print('PASS' if $GAP < 0.15 else 'FAIL')")

echo ""
echo "========================================================================"
echo "  M1 结果: $WNAME"
echo "  SR real: $SR_REAL ($REAL_OK/$N)   SR synth: $SR_SYNTH ($SYNTH_OK/$N)"
echo "  gap: $GAP   verdict: $VERDICT"
echo "========================================================================"

RESULT_FILE="$OUT_BASE/m1_${WNAME}_${TIMESTAMP}.json"
cat > "$RESULT_FILE" << EOF
{"world":"$WNAME","timestamp":"$TIMESTAMP","ckpt":"$CKPT","trials":$N,
 "sr_real":$SR_REAL,"sr_synth":$SR_SYNTH,"gap":$GAP,"verdict":"$VERDICT",
 "real_ok":$REAL_OK,"synth_ok":$SYNTH_OK}
EOF
echo "  result: $RESULT_FILE"

rm -f "$TRIALS_FILE"
