#!/bin/bash
# sync_kinematic.sh — 同步运动学环境文件到服务器部署目录
# 用法: 从本地项目根目录运行 bash sync_kinematic.sh
set -e

LOCAL_DIR="/home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy"
REMOTE_DIR="/root/autodl-tmp/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy"

echo "[sync] 同步运动学闭环环境文件..."

# 1) kinematic_uav_env.py (新文件)
echo "[sync] kinematic_uav_env.py"
cp "$LOCAL_DIR/diffusion_policy_3d/env/kinematic_uav_env.py" \
   "$REMOTE_DIR/diffusion_policy_3d/env/kinematic_uav_env.py"

# 2) collision_geom.py (更新)
echo "[sync] collision_geom.py"
cp "$LOCAL_DIR/diffusion_policy_3d/env/collision_geom.py" \
   "$REMOTE_DIR/diffusion_policy_3d/env/collision_geom.py"

# 3) test_kinematic.py (新文件)
echo "[sync] test_kinematic.py"
cp "$LOCAL_DIR/diffusion_policy_3d/env/test_kinematic.py" \
   "$REMOTE_DIR/diffusion_policy_3d/env/test_kinematic.py"

# 4) train_dsrl_uav.py (已修改: KinematicUAVEnv 接线)
echo "[sync] train_dsrl_uav.py"
cp "$LOCAL_DIR/train_dsrl_uav.py" \
   "$REMOTE_DIR/train_dsrl_uav.py"

echo ""
echo "[sync] 完成. 验证:"
echo "  python -m diffusion_policy_3d.env.test_kinematic"
echo "  python -m diffusion_policy_3d.env.test_kinematic --real-synth"
