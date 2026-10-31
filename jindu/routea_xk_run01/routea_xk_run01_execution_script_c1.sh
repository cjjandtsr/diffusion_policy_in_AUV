#!/bin/bash
# Route A X/K credit-fix recovery, step 03 (guide 2026-10-28 v1.0).
# Verifies the fixed credit audit at the real crash point: one X update to u129,
# then the guide's two test -f checks (COMMITTED.json + metrics_update_000129.json).
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
run_fix 03_X_u129 "$PY" scripts/run_routea_xk.py --repo "$ROOT" train \
  --config "$OUT/main/X.yaml" --stop 129
test -f "$WEIGHTS/main/X/checkpoint_update_000129/COMMITTED.json"
test -f "$OUT/main/X/metrics_update_000129.json"
