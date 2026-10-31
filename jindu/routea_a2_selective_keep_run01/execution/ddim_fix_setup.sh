#!/usr/bin/env bash
# v1.2 修复恢复 §2：环境 + 预检 + 失败证据归档 + 补丁复制（指导第一代码块）
set -euo pipefail
ROOT=/home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy
PY=/home/cjj/miniconda3/envs/idp3/bin/python
OUT=/home/cjj/AUV/program/results/routea_a2_selective_keep_run01
WEIGHTS=$ROOT/data/outputs/dppo_uav_rebuild/routea_a2_selective_keep_run01
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export DPPO_VENDOR_ROOT="$ROOT/third_party/dppo_official_cc7234a"

# --- 指导 §2 预检 ---
test -f "$OUT/parent/receipt.json"
test -f "$OUT/reference/reference.pt"
test -f "$OUT/arms/C.yaml" && test -f "$OUT/arms/T.yaml"
test -f "$WEIGHTS/old_replay/checkpoint_update_000129/state.pt"
test -f "$WEIGHTS/C/checkpoint_update_000129/state.pt"
test ! -e "$OUT/first_update_gate.json"
test ! -e "$OUT/blocked_v1_1"
test ! -e "$WEIGHTS/blocked_v1_1"

# --- 失败证据归档（指导意图：blocked_v1_1/source 保存失败的 v1.1 版本）---
# 适配说明：补丁 2 文件在指导到达前已生效于 WSL（hash 与 Windows 补丁一致），
# 故按指导意图从 pre_existing_a2 存档归档 v1.1 失败版本（hash 已核与 v1.1 清单一致），
# 不把修复版冒充失败证据。归档前逐文件验证 hash == v1.1 清单。
V11_MANIFEST=/mnt/c/Users/cjj/Desktop/IDP3/zhidao/2026-10-24_A2交付文件_sha256.txt
mkdir -p "$OUT/blocked_v1_1/source"
for rel in \
  diffusion_policy_3d/rl/dppo_uav_rebuild/engine/selective_keep.py \
  tests/dppo_uav_rebuild/engine/test_selective_keep.py; do
  mkdir -p "$OUT/blocked_v1_1/source/$(dirname "$rel")"
  expect=$(grep -F "  $rel" "$V11_MANIFEST" | awk '{print $1}')
  actual=$(sha256sum "$OUT/execution/pre_existing_a2/$rel" | awk '{print $1}')
  test -n "$expect" && test "$expect" = "$actual"
  cp -- "$OUT/execution/pre_existing_a2/$rel" "$OUT/blocked_v1_1/source/$rel"
done

# --- 补丁复制（幂等：WSL 已同 hash，仍按指导从 Windows 补丁复制一遍）---
cp -- "/mnt/c/Users/cjj/Desktop/IDP3/chengxu/Improved-3D-Diffusion-Policy/diffusion_policy_3d/rl/dppo_uav_rebuild/engine/selective_keep.py" \
  "$ROOT/diffusion_policy_3d/rl/dppo_uav_rebuild/engine/selective_keep.py"
cp -- "/mnt/c/Users/cjj/Desktop/IDP3/chengxu/Improved-3D-Diffusion-Policy/tests/dppo_uav_rebuild/engine/test_selective_keep.py" \
  "$ROOT/tests/dppo_uav_rebuild/engine/test_selective_keep.py"
cp -- "/mnt/c/Users/cjj/Desktop/IDP3/zhidao/2026-10-24_A2_DDIM修复交付_sha256.txt" \
  "$OUT/execution/ddim_fix_delivery_sha256.txt"
echo 'ddim_fix_setup OK'
