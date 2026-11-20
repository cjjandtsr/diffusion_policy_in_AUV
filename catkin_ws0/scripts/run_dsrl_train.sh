#!/bin/bash
# run_dsrl_train.sh — 一键启动 DSRL 训练（SITL + RL，对齐 run_trial.sh 模式）
# 用法: bash run_dsrl_train.sh [world_id] [total_timesteps]
set -e

WORLD_ID="${1:-4}"
TIMESTEPS="${2:-5000}"
AUV_ROOT="/root/autodl-tmp/AUV"
SCRIPTS="$AUV_ROOT/catkin_ws0/scripts"
FM_DIR="$AUV_ROOT/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy"
CKPT="data/outputs/2026.06.17/15.22.50_v9_2_nobs8_bc/checkpoints/best.ckpt"

# world_id → world 文件名
declare -A WORLD_MAP=( [4]="env_004_obs19" [8]="env_008_obs20" [11]="env_011_obs23" [12]="env_012_obs22" [18]="env_018_obs25" )
WORLD_NAME="${WORLD_MAP[$WORLD_ID]}"
WORLD_FILE="$AUV_ROOT/PX4-Autopilot/Tools/sitl_gazebo/worlds/ood_experiment_worlds/layerA/${WORLD_NAME}.world"

# ── 环境：对齐 run_trial.sh，脚本顶部统一 source（继承调用者的 idp3 conda）──
PX4_DIR="$AUV_ROOT/PX4-Autopilot"
CATKIN_WS="$AUV_ROOT/catkin_ws0"
source /opt/ros/noetic/setup.bash
source "$CATKIN_WS/devel/setup.bash"
source "$PX4_DIR/Tools/setup_gazebo.bash" "$PX4_DIR" "$PX4_DIR/build/px4_sitl_default" 2>/dev/null
export ROS_PACKAGE_PATH="$ROS_PACKAGE_PATH:$PX4_DIR:$PX4_DIR/Tools/sitl_gazebo"

# ── 清理函数 ──
cleanup() {
    echo "[run_dsrl] 清理进程..."
    pkill -9 -f "train_dsrl_uav" 2>/dev/null || true
    pkill -9 -f "px4ctrl"         2>/dev/null || true
    pkill -9 -f "gzserver"        2>/dev/null || true
    pkill -9 -f "gzclient"        2>/dev/null || true
    pkill -9 -f "mavros_node"     2>/dev/null || true
    bash "$SCRIPTS/stop_sitl.sh" 2>/dev/null || true
    echo "[run_dsrl] 清理完成"
}
trap cleanup EXIT INT TERM

# ── 1) 首次清理 + 启动 SITL（内部已隔离 conda）──
echo "[run_dsrl] world_id=$WORLD_ID world=$WORLD_NAME timesteps=$TIMESTEPS"
bash "$SCRIPTS/stop_sitl.sh" 2>/dev/null || true
sleep 2
echo "[run_dsrl] 启动 SITL..."
nohup bash "$SCRIPTS/start_sitl_rl.sh" "$WORLD_FILE" > /dev/null 2>&1 &
echo "[run_dsrl] SITL PID=$!"

# ── 2) 等 MAVROS 上线 ──
echo "[run_dsrl] 等 MAVROS..."
for i in $(seq 1 60); do
    if timeout 2 rostopic list 2>/dev/null | grep -q "/mavros/state"; then
        echo "[run_dsrl] MAVROS 上线 (${i}s)"
        break
    fi
    sleep 1
done
sleep 5  # 给 SITL 多稳一会

# ── 3) 起飞 ──
echo "[run_dsrl] 起飞..."
bash "$SCRIPTS/takeoff_only.sh"

# ── 4) 启动 RL 训练（idp3 conda 已激活，继承自调用者）──
echo "[run_dsrl] 启动 RL 训练..."
cd "$FM_DIR"
python train_dsrl_uav.py \
    --ckpt "$CKPT" \
    --world_id "$WORLD_ID" \
    --total_timesteps "$TIMESTEPS" \
    --log_dir "$AUV_ROOT/data/outputs/dsrl_uav" \
    --device cuda:0
