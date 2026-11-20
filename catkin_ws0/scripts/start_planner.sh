#!/bin/bash
# ════════════════════════════════════════════════════════════════
# start_planner.sh — idp3 环境后台启动 ml_gazebo_fm.sh, 不卡终端
# 用法: bash start_planner.sh [ckpt] [zff_mode] [fm_steps] [cfg_lambda]
#   所有参数透传给 ml_gazebo_fm.sh, 不传则用其默认值
# 关键: 内部 conda activate idp3 (torch 需要), 与 SITL 的系统python相反
# 输出 → $LOG_DIR/planner.log (默认数据盘), 脚本立即返回
# 前提: SITL 已就绪 (start_sitl.sh 跑过, /mavros connected)
# ════════════════════════════════════════════════════════════════

LOG_DIR="${AUV_LOG_DIR:-/root/autodl-tmp/logs}"
mkdir -p "$LOG_DIR"

CKPT="${1:-}"
ZFF="${2:-}"
FM_STEPS="${3:-}"
CFG_LAMBDA="${4:-}"

ML_SH="$HOME/AUV/catkin_ws0/scripts/ego_gazebo.sh"   # 改成你的实际路径
if [ ! -f "$ML_SH" ]; then
    echo "[start_planner] ✗ 找不到 $ML_SH, 请改脚本里 ML_SH 路径"
    exit 1
fi

echo "[start_planner] 后台启动 ml_gazebo_fm.sh (idp3 环境)..."
echo "[start_planner] 参数: ckpt='$CKPT' zff='$ZFF' fm_steps='$FM_STEPS' cfg='$CFG_LAMBDA'"
echo "[start_planner] 日志: $LOG_DIR/planner.log"

# setsid 脱离终端; bash -i 读 .bashrc 拿到 conda; activate idp3; 跑 ml_gazebo_fm.sh
# 注意: 这里要 idp3 环境 (与 SITL 相反), 所以走 conda activate
setsid bash -ic "
    conda activate idp3
    bash '$ML_SH' '$CKPT' '$ZFF' '$FM_STEPS' '$CFG_LAMBDA'
" > "$LOG_DIR/planner.log" 2>&1 < /dev/null &

echo "[start_planner] 已后台启动, 终端已释放。"
echo "[start_planner] 看进度: tail -f $LOG_DIR/planner.log"
echo "[start_planner] 起飞+规划约需 60-70s (offboard 40s + px4ctrl 12s + planner 10s)"
