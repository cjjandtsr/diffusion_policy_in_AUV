#!/usr/bin/env bash
# 09_T_u256: v1.1 §4 T 臂续训 u130–256（gate PASS 后，v1.2 修复源码）
set -uo pipefail
ROOT=/home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy
PY=/home/cjj/miniconda3/envs/idp3/bin/python
OUT=/home/cjj/AUV/program/results/routea_a2_selective_keep_run01
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export DPPO_VENDOR_ROOT="$ROOT/third_party/dppo_official_cc7234a"
date +'%s' > "$OUT/execution/09_T_u256.start_ts"
"$PY" scripts/run_routea_a2.py continue-training \
  --repo "$ROOT" --gate "$OUT/first_update_gate.json" --arm T --config "$OUT/arms/T.yaml"
rc=$?
printf '%s\n' "$rc" > "$OUT/execution/09_T_u256.rc"
date +'%s' > "$OUT/execution/09_T_u256.end_ts"
exit "$rc"
