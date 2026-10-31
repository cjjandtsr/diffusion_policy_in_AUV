#!/usr/bin/env bash
# 11_val_T: v1.1 §5 val20 T 臂（BC/RL 各 80 条件 = 160 条）
set -uo pipefail
ROOT=/home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy
PY=/home/cjj/miniconda3/envs/idp3/bin/python
OUT=/home/cjj/AUV/program/results/routea_a2_selective_keep_run01
WEIGHTS=$ROOT/data/outputs/dppo_uav_rebuild/routea_a2_selective_keep_run01
VAL_WORLDS=/home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds/val
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export DPPO_VENDOR_ROOT="$ROOT/third_party/dppo_official_cc7234a"
date +'%s' > "$OUT/execution/11_val_T.start_ts"
"$PY" scripts/run_routea_val20_endpoint.py \
  --config "$OUT/arms/T.yaml" --rl-checkpoint "$WEIGHTS/T/checkpoint_update_000256" \
  --eval-config-dir "$ROOT/data/eval_configs" --benchmark routea-val \
  --worlds-val "$VAL_WORLDS" --output-dir "$OUT/val_T" --expected-update 256
rc=$?
printf '%s\n' "$rc" > "$OUT/execution/11_val_T.rc"
date +'%s' > "$OUT/execution/11_val_T.end_ts"
exit "$rc"
