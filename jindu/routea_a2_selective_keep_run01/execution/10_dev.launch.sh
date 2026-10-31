#!/usr/bin/env bash
# 10_dev: v1.1 §5 dev 统一评价（BC + 冻结 u128 + C256 + T256，1680 full + smoke）
set -uo pipefail
ROOT=/home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy
PY=/home/cjj/miniconda3/envs/idp3/bin/python
OUT=/home/cjj/AUV/program/results/routea_a2_selective_keep_run01
WEIGHTS=$ROOT/data/outputs/dppo_uav_rebuild/routea_a2_selective_keep_run01
REF=$ROOT/data/outputs/diagnostics/routea_bc_a0_r1_paired_dev_20261010_py38fix_run02
PARENT=$ROOT/data/outputs/dppo_uav_rebuild/routea_common_b2560_seed202609203_v1/checkpoint_update_000128
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export DPPO_VENDOR_ROOT="$ROOT/third_party/dppo_official_cc7234a"
date +'%s' > "$OUT/execution/10_dev.start_ts"
"$PY" scripts/run_routea_paired_comparison.py --reference "$REF" \
  --checkpoint "$PARENT" --checkpoint "$WEIGHTS/C/checkpoint_update_000256" \
  --checkpoint "$WEIGHTS/T/checkpoint_update_000256" --out "$OUT/evaluation_dev"
rc=$?
printf '%s\n' "$rc" > "$OUT/execution/10_dev.rc"
date +'%s' > "$OUT/execution/10_dev.end_ts"
exit "$rc"
