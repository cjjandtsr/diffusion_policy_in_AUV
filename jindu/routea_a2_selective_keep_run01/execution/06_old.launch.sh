#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy
PY=/home/cjj/miniconda3/envs/idp3/bin/python
OUT=/home/cjj/AUV/program/results/routea_a2_selective_keep_run01
WEIGHTS=$ROOT/data/outputs/dppo_uav_rebuild/routea_a2_selective_keep_run01
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
rc=0
"$PY" scripts/run_routea_a2.py old-replay --repo "$ROOT" \
  --receipt "$OUT/parent/receipt.json" --output "$OUT/old_replay" --weights "$WEIGHTS/old_replay" || rc=$?
echo "$rc" > "$OUT/execution/06_old.rc"
exit "$rc"
