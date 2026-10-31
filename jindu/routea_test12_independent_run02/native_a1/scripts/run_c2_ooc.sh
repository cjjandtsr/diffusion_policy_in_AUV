#!/bin/bash
# C2-OOC: Single-cell online smoke for v1.3 environment
# RTX 4060 Laptop (8GB) optimized
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M)
ROOT="/home/cjj/AUV"
OUTPUT_DIR="${ROOT}/program/shiyan/results_routeA_RL_c2_ooc_ph50_${TIMESTAMP}"

echo "=============================================="
echo "  C2-OOC: Controlled Online Cell Smoke"
echo "  Time: ${TIMESTAMP}"
echo "  Output: ${OUTPUT_DIR}"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "=============================================="

mkdir -p "${OUTPUT_DIR}"

cd "${ROOT}/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy"

PYTHONUNBUFFERED=1 python -u scripts/train_a2_10k_capability.py \
    --routea_rl_formal_train \
    --use_routea_rl_env_v1 \
    --role_balanced_sampler \
    --ft_c2_ooc \
    --ft_c2_ooc_no_twin_stop \
    --ckpt "data/outputs/2026.06.17/15.22.50_v9_2_nobs8_bc/checkpoints/latest.ckpt" \
    --critic_ckpt "${ROOT}/program/shiyan/results_routeA_RL_b0b1_rerun_v13_20260710/b1_critic_rolefix/a2_critic_best.pt" \
    --ft_bootstrap_replay "${ROOT}/program/shiyan/results_routeA_RL_b0b1_rerun_v13_20260710/b0_collect_rolefix/replay.pt" \
    --ft_reward_norm_stats "${ROOT}/program/shiyan/results_routeA_RL_b0b1_rerun_v13_20260710/reward_norm_stats.json" \
    --routea_rl_admission_manifest "${ROOT}/program/reports/routeA_rl_env_v1/calibration/admission_manifest.json" \
    --output_dir "${OUTPUT_DIR}" \
    --seed 42 \
    2>&1 | tee "${OUTPUT_DIR}/train.log"

echo ""
echo "=============================================="
echo "  C2-OOC complete. Output: ${OUTPUT_DIR}"
echo "=============================================="
