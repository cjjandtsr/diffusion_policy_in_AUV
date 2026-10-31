#!/usr/bin/env bash
# v1.2 修复恢复：环境变量 + run() 函数（指导 §2，供各 run 步骤 source）
ROOT=/home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy
PY=/home/cjj/miniconda3/envs/idp3/bin/python
OUT=/home/cjj/AUV/program/results/routea_a2_selective_keep_run01
WEIGHTS=$ROOT/data/outputs/dppo_uav_rebuild/routea_a2_selective_keep_run01
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export DPPO_VENDOR_ROOT="$ROOT/third_party/dppo_official_cc7234a"
run() {
  local name="$1"; shift
  test ! -e "$OUT/execution/$name.log"
  printf '%q ' "$@" > "$OUT/execution/$name.command"
  printf '\n' >> "$OUT/execution/$name.command"
  set +e
  "$@" > "$OUT/execution/$name.log" 2>&1
  local rc=$?
  set -e
  printf '%s\n' "$rc" > "$OUT/execution/$name.rc"
  test "$rc" -eq 0
}
