#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy
PY=/home/cjj/miniconda3/envs/idp3/bin/python
OUT=/home/cjj/AUV/program/results/routea_a2_selective_keep_run01
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
rc=0
"$PY" scripts/train_dppo_uav_rebuild.py --config "$OUT/arms/T.yaml" --stop-after-update 129 || rc=$?
echo "$rc" > "$OUT/execution/07_T_u129.rc"
exit "$rc"
