#!/bin/bash
# Route A X/K credit-fix recovery, steps 04-05 (guide 2026-10-28 v1.0).
# 04: X formal 129 -> 256, resuming from the new u129 checkpoint.
# 05: K formal 128 -> 256, fresh fork from the same u128 parent.
# set -e: 05 starts only if 04 exits 0. Per-step rc files in credit_fix_20261028/.
set -euo pipefail
ROOT=/home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy
PY=/home/cjj/miniconda3/envs/idp3/bin/python
OUT=/home/cjj/AUV/program/results/routea_xk_run01
WEIGHTS=$ROOT/data/outputs/dppo_uav_rebuild/routea_xk_run01
FIX=$OUT/execution/credit_fix_20261028
cd "$ROOT"
run_fix() {
  local label=$1
  shift
  printf '%q ' "$@" > "$FIX/$label.command.txt"
  printf '\n' >> "$FIX/$label.command.txt"
  set +e
  "$@" > "$FIX/$label.log" 2>&1
  local rc=$?
  set -e
  printf '%s\n' "$rc" > "$FIX/$label.rc"
  test "$rc" -eq 0
}
run_fix 04_X_u256 "$PY" scripts/run_routea_xk.py --repo "$ROOT" train \
  --config "$OUT/main/X.yaml" --resume "$WEIGHTS/main/X/checkpoint_update_000129"
run_fix 05_K_u256 "$PY" scripts/run_routea_xk.py --repo "$ROOT" train \
  --config "$OUT/main/K.yaml"
