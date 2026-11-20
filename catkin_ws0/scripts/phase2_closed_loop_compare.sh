#!/bin/bash
# ════════════════════════════════════════════════════════════════
# Phase ② Closed-Loop Comparison Runner
# ════════════════════════════════════════════════════════════════
# 目的: 在相同 worlds 上对比多个 checkpoint 的闭环表现
#       (Phase ① baseline vs Phase ② collision-aware)
#
# 与 Phase ① run_honest_baseline.sh 的区别:
#   - 外层多一个 "condition" 循环, 每个 condition = 一个 checkpoint
#   - 通过 ml_gazebo.sh 的 ckpt 参数切换 checkpoint
#   - 日志按 condition 分目录, 方便 analyze 脚本对比
#
# 用法:
#   bash phase2_closed_loop_compare.sh                  # 默认对比 P1aR2 vs Phase②λ0.5
#   bash phase2_closed_loop_compare.sh -w 4 -t 2        # 4 worlds × 2 trials
#   bash phase2_closed_loop_compare.sh --dry-run        # 仅打印 plan
#
# 前提:
#   1. ml_planner_gazebo.launch (Phase ② 版, checkpoint_path 改为 ckpt arg) 已部署
#   2. ml_gazebo.sh (Phase ② 版, 接受 ckpt 参数) 已部署
#   3. launch 里 use_obstacle_projection=false (policy-only, 与 Phase ① 一致)
#   4. PX4 与 ROS 环境已 source
# ════════════════════════════════════════════════════════════════

set -uo pipefail

# ════════════ 对比条件 (condition) 定义 ════════════
# 每行: "<标签>|<checkpoint 绝对路径>"
# 标签用于日志目录命名, 不要含空格
IDP3_DATA="${HOME}/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy/data/outputs"
CONDITIONS=(
    "phase1a_r2|${IDP3_DATA}/2026.05.10/16.07.11_v8_phase1a/checkpoints/best.ckpt"
    "phase2_lambda0.5|${IDP3_DATA}/2026.05.17/13.36.20_v8_phase2_sweep_lambda0.5/checkpoints/latest.ckpt"
)
# 如需对比全部 3 个 sweep λ, 取消下面注释:
# CONDITIONS=(
#     "phase1a_r2|${IDP3_DATA}/2026.05.10/16.07.11_v8_phase1a/checkpoints/best.ckpt"
#     "phase2_lambda0.3|${IDP3_DATA}/2026.05.17/11.50.36_v8_phase2_sweep_lambda0.3/checkpoints/latest.ckpt"
#     "phase2_lambda0.5|${IDP3_DATA}/2026.05.17/13.36.20_v8_phase2_sweep_lambda0.5/checkpoints/latest.ckpt"
#     "phase2_lambda1.0|${IDP3_DATA}/2026.05.17/15.20.34_v8_phase2_sweep_lambda1.0/checkpoints/latest.ckpt"
# )

# ---- 默认参数 ----
NUM_WORLDS=8
NUM_TRIALS=3
FLIGHT_TIMEOUT=180
LOG_ROOT="${HOME}/ml_planner_log/phase2_closed_loop"
DRY_RUN=0

# ---- 8 个代表性 worlds (与 Phase ① 一致, 保证可比) ----
WORLDS=(
    "single_AUV"
    "env_001_obs5"
    "env_002_obs5"
    "env_003_obs8"
    "env_004_obs8"
    "env_005_obs10"
    "env_006_obs10"
    "env_007_obs12"
)

# 解析命令行
while [[ $# -gt 0 ]]; do
    case "$1" in
        -w) NUM_WORLDS="$2"; shift 2 ;;
        -t) NUM_TRIALS="$2"; shift 2 ;;
        --timeout) FLIGHT_TIMEOUT="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) head -22 "$0"; exit 0 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

if [[ ${#WORLDS[@]} -gt $NUM_WORLDS ]]; then
    WORLDS=("${WORLDS[@]:0:$NUM_WORLDS}")
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_ROOT="${LOG_ROOT}/${TIMESTAMP}"
mkdir -p "$RUN_ROOT"

echo "════════════════════════════════════════════════════════════"
echo "  Phase ② Closed-Loop Comparison"
echo "════════════════════════════════════════════════════════════"
echo "  Conditions:   ${#CONDITIONS[@]}"
for cond in "${CONDITIONS[@]}"; do
    echo "    - ${cond%%|*}"
done
echo "  Worlds:       ${WORLDS[*]}"
echo "  Trials/world: $NUM_TRIALS"
echo "  Timeout:      ${FLIGHT_TIMEOUT}s"
echo "  Log root:     $RUN_ROOT"
echo "  总飞行次数:   $(( ${#CONDITIONS[@]} * ${#WORLDS[@]} * NUM_TRIALS ))"
echo "════════════════════════════════════════════════════════════"

# ---- 安全检查: launch 里 use_obstacle_projection=false ----
LAUNCH_FILE=$(rospack find ego_planner 2>/dev/null)/launch/ml_planner_gazebo.launch
if [[ ! -f "$LAUNCH_FILE" ]]; then
    echo "[ERR] 找不到 launch 文件: $LAUNCH_FILE"
    exit 1
fi
if grep -q 'use_obstacle_projection.*value="false"' "$LAUNCH_FILE"; then
    echo "[OK] use_obstacle_projection=false 已确认 (policy-only, 与 Phase ① 可比)"
else
    echo "[ERR] launch 中 use_obstacle_projection 不是 false, 拒绝继续"
    exit 1
fi
# 确认 launch 支持 ckpt arg
if grep -q 'arg name="ckpt"' "$LAUNCH_FILE"; then
    echo "[OK] launch 支持 ckpt arg (Phase ② 版已部署)"
else
    echo "[ERR] launch 不支持 ckpt arg, 请部署 Phase ② 版 ml_planner_gazebo.launch"
    exit 1
fi

# 确认所有 checkpoint 存在
for cond in "${CONDITIONS[@]}"; do
    ckpt="${cond##*|}"
    if [[ ! -f "$ckpt" ]]; then
        echo "[ERR] checkpoint 不存在: $ckpt"
        echo "      (condition: ${cond%%|*})"
        exit 1
    fi
done
echo "[OK] 所有 ${#CONDITIONS[@]} 个 checkpoint 文件已确认存在"

# ════════════ 飞行循环 (condition → world → trial) ════════════
flight_count=0
total_flights=$(( ${#CONDITIONS[@]} * ${#WORLDS[@]} * NUM_TRIALS ))

for cond in "${CONDITIONS[@]}"; do
    cond_label="${cond%%|*}"
    cond_ckpt="${cond##*|}"
    cond_dir="${RUN_ROOT}/${cond_label}"
    mkdir -p "$cond_dir"

    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║ Condition: ${cond_label}"
    echo "║ Checkpoint: ${cond_ckpt}"
    echo "╚══════════════════════════════════════════════════════════╝"

    for world in "${WORLDS[@]}"; do
        for trial in $(seq 1 $NUM_TRIALS); do
            flight_count=$((flight_count + 1))
            flight_id="${world}_trial${trial}"
            echo ""
            echo "──────────────────────────────────────────────────"
            echo " [$flight_count/$total_flights] cond=$cond_label world=$world trial=$trial"
            echo "──────────────────────────────────────────────────"

            if [[ $DRY_RUN -eq 1 ]]; then
                echo "  [DRY-RUN] 跳过实际飞行"
                continue
            fi

            # 1. 启动 PX4 仿真
            echo "  [1/4] 启动 PX4 + Gazebo ($world)..."
            roslaunch px4 ${world}.launch &
            PX4_PID=$!
            sleep 25

            # 2. 启动 ml_planner, 传 checkpoint 路径
            #    ros log 命名为 ros_<flight_id>.log (analyze 脚本的 prefix 约定)
            echo "  [2/4] 启动 ML Planner (ckpt=$cond_label)..."
            bash ${HOME}/AUV/catkin_ws0/scripts/ml_gazebo.sh "$cond_ckpt" \
                > "$cond_dir/ros_${flight_id}.log" 2>&1 &
            ML_PID=$!
            sleep 20

            # 3. 等飞行结束
            echo "  [3/4] 飞行中 (max ${FLIGHT_TIMEOUT}s)..."
            flight_start=$(date +%s)
            while true; do
                elapsed=$(( $(date +%s) - flight_start ))
                if [[ $elapsed -ge $FLIGHT_TIMEOUT ]]; then
                    echo "    [TIMEOUT] 达到 ${FLIGHT_TIMEOUT}s, 终止"
                    break
                fi
                if grep -q "GOAL-REACH\|所有航点完成" \
                    "$cond_dir/ros_${flight_id}.log" 2>/dev/null; then
                    echo "    [GOAL] 检测到到达 (after ${elapsed}s)"
                    sleep 3
                    break
                fi
                sleep 2
            done

            # 4. 清理 + 抢救 CSV
            #    命名: diffusion_<flight_id>.csv / pos_cmd_<flight_id>.csv
            #    (analyze_honest_baseline.py 的 prefix glob 约定)
            echo "  [4/4] 清理 + 保存数据..."
            pkill -INT -f "ml_planner_gazebo" 2>/dev/null
            pkill -INT -f "px4ctrl" 2>/dev/null
            pkill -f "ml_planner" 2>/dev/null
            sleep 3
            pkill -f "gzserver" 2>/dev/null
            pkill -f "gzclient" 2>/dev/null
            pkill -f "px4" 2>/dev/null
            pkill -f "mavros" 2>/dev/null
            sleep 5

            LATEST_DIFF=$(ls -t ${HOME}/ml_planner_log/diffusion_*.csv 2>/dev/null | head -1)
            LATEST_POS=$(ls -t ${HOME}/ml_planner_log/pos_cmd_*.csv 2>/dev/null | head -1)
            LATEST_STATE=$(ls -t ${HOME}/ml_planner_log/state_*.csv 2>/dev/null | head -1)
            [[ -n "$LATEST_DIFF" ]]  && mv "$LATEST_DIFF"  "$cond_dir/diffusion_${flight_id}.csv"
            [[ -n "$LATEST_POS" ]]   && mv "$LATEST_POS"   "$cond_dir/pos_cmd_${flight_id}.csv"
            [[ -n "$LATEST_STATE" ]] && mv "$LATEST_STATE" "$cond_dir/state_${flight_id}.csv"
            echo "    数据已保存到 $cond_dir/"

            if [[ -f "$cond_dir/pos_cmd_${flight_id}.csv" ]]; then
                tail -1 "$cond_dir/pos_cmd_${flight_id}.csv" | \
                    awk -F',' '{printf "    final: cur=(%.2f,%.2f,%.2f)\n", $(NF-7),$(NF-6),$(NF-5)}' 2>/dev/null
            fi
        done
    done
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  全部飞行完成 (${flight_count} flights)"
echo "  日志根目录: $RUN_ROOT"
echo ""
echo "  下一步: 逐 condition 分析 + 对比"
for cond in "${CONDITIONS[@]}"; do
    cl="${cond%%|*}"
    echo "    python3 analyze_honest_baseline.py --dir $RUN_ROOT/$cl"
done
echo ""
echo "  或用对比脚本一次性汇总:"
echo "    python3 phase2_compare_results.py --run-root $RUN_ROOT"
echo "════════════════════════════════════════════════════════════"