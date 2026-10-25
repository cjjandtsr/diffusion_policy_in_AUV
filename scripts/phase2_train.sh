#!/bin/bash
# ════════════════════════════════════════════════════════════════
# Phase ② Constraint-Aware BC — 训练启动脚本
# 版本: 2026-05-17 (参数对齐 v1.2 §5.2: d_safe=0.75, β=20.0, τ_softmin=5.0)
# ════════════════════════════════════════════════════════════════

set -uo pipefail

IDP3_ROOT="/root/autodl-tmp/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy"

# ── 关键路径 (用户按实际改) ──
PHASE1A_R2_CKPT="${IDP3_ROOT}/data/outputs/2026.05.10/16.07.11_v8_phase1a/checkpoints/best.ckpt"
ANCHOR_BANK="/root/autodl-tmp/AUV/anchor_bank.pt"
ZARR_PATH="/root/autodl-tmp/AUV/avoidance_dataset/merged_body_12.zarr"
DEVICE="cuda:0"

MODE="${1:-help}"

cd "$IDP3_ROOT" || { echo "[ERR] 找不到 iDP3 root: $IDP3_ROOT"; exit 1; }

if [[ ! -f "$PHASE1A_R2_CKPT" ]]; then
    echo "[ERR] Phase 1a R2 ckpt 不存在: $PHASE1A_R2_CKPT"
    exit 1
fi

# 公共参数 (v1.2 §5.2 锁定: d_safe=0.75, β=20.0, τ_softmin=5.0)
COMMON_ARGS="--baseline_ckpt $PHASE1A_R2_CKPT \
  --anchor_bank $ANCHOR_BANK \
  --zarr_path $ZARR_PATH \
  --device $DEVICE \
  --batch_size 60 \
  --skip_step2 \
  --collision_d_safe 0.75 \
  --collision_beta 20.0 \
  --collision_tau_softmin 5.0 \
  --collision_k_nn 8 \
  --collision_horizon_decay_tau 8.0"

case "$MODE" in
  pilot)
    echo "════════════════════════════════════════════════════════════"
    echo "  Phase ② Pilot: λ_col=0.05, d_safe=0.75, β=20.0, τ_softmin=5.0"
    echo "════════════════════════════════════════════════════════════"
    python train_phase1a.py \
      $COMMON_ARGS \
      --use_collision_loss \
      --collision_lambda 0.05 \
      --step3_epochs 4 \
      --step3_lr 5e-5 \
      --log_every 1 \
      --exp_name v8_phase2_pilot_lambda0.05
    ;;

  sweep)
    echo "════════════════════════════════════════════════════════════"
    echo "  Phase ② λ_col 扫参 (d_safe=0.75, β=20.0, τ_softmin=5.0)"
    echo "════════════════════════════════════════════════════════════"
    for LAMBDA in 0.3 0.5 1.0; do
      echo "──────────────────────────────────────────────────"
      echo " [Sweep] λ_col = $LAMBDA"
      echo "──────────────────────────────────────────────────"
      python train_phase1a.py \
        $COMMON_ARGS \
        --use_collision_loss \
        --collision_lambda $LAMBDA \
        --step3_epochs 21 \
        --step3_lr 1e-4 \
        --log_every 2 \
        --exp_name v8_phase2_sweep_lambda${LAMBDA}
    done
    ;;

  full)
    LAMBDA="${2:-0.5}"
    echo "════════════════════════════════════════════════════════════"
    echo "  Phase ② 全量训练: λ_col = $LAMBDA, d_safe=0.75, β=20.0, τ_softmin=5.0"
    echo "════════════════════════════════════════════════════════════"
    python train_phase1a.py \
      $COMMON_ARGS \
      --use_collision_loss \
      --collision_lambda $LAMBDA \
      --step3_epochs 301 \
      --step3_lr 1e-4 \
      --log_every 20 \
      --exp_name v8_phase2_bc_collision_lambda${LAMBDA}
    ;;

  sanity)
    echo "════════════════════════════════════════════════════════════"
    echo "  Phase ② sanity check: collision loss 关闭"
    echo "════════════════════════════════════════════════════════════"
    python train_phase1a.py \
      $COMMON_ARGS \
      --step3_epochs 11 \
      --step3_lr 1e-4 \
      --log_every 2 \
      --exp_name v8_phase2_sanity_no_collision
    ;;

  help|*)
    head -25 "$0"
    ;;
esac