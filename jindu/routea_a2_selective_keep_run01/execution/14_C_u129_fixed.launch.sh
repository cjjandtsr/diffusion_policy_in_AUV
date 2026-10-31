#!/usr/bin/env bash
# 14_C_u129_fixed: v1.2 修复后 C 臂从父 u128 重新分叉 u129（指导 v1.2 §3）
set -uo pipefail
ROOT=/home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy
PY=/home/cjj/miniconda3/envs/idp3/bin/python
OUT=/home/cjj/AUV/program/results/routea_a2_selective_keep_run01
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export DPPO_VENDOR_ROOT="$ROOT/third_party/dppo_official_cc7234a"
date +'%s' > "$OUT/execution/14_C_u129_fixed.start_ts"
"$PY" scripts/train_dppo_uav_rebuild.py --config "$OUT/arms/C.yaml" --stop-after-update 129
rc=$?
printf '%s\n' "$rc" > "$OUT/execution/14_C_u129_fixed.rc"
date +'%s' > "$OUT/execution/14_C_u129_fixed.end_ts"
exit "$rc"
