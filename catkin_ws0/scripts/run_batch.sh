#!/bin/bash
# ══════════════════════════════════════════════════════════════
# run_batch.sh — 批量跑 OOD 闭环实验
#   读一张实验表 (trials.txt), 每行一个 trial, 循环调 run_trial.sh,
#   全部跑完后汇总成 CSV。
#
#   用法:
#     bash run_batch.sh <trials.txt> [results_dir]
#
#   trials.txt 每行 (空格分隔, # 开头为注释):
#     <tier> <world_id> <trial> <world_path> <obstacles_json> <gx> <gy> <gz>
#   示例行:
#     layerA 0 0 /root/autodl-tmp/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds/ood_experiment_worlds/layerA/env_000_obs1.world /root/autodl-tmp/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds/ood_experiment_worlds/layerA/env_000_obs1.obstacles.json 12.0 0.0 2.0
#
#   可调环境变量 (透传给 run_trial.sh):
#     WITH_RVIZ TIMEOUT DRONE_RADIUS CKPT DRONE_NAME
# ══════════════════════════════════════════════════════════════
set -u

SCRIPTS_DIR="/root/autodl-tmp/AUV/catkin_ws0/scripts"
RUN_TRIAL="${RUN_TRIAL_SCRIPT:-$SCRIPTS_DIR/run_trial.sh}"

TRIALS_FILE="${1:?need trials.txt}"
RESULTS_DIR="${2:-/root/autodl-tmp/AUV/results}"
SUMMARY_CSV="$RESULTS_DIR/summary_$(date +%Y%m%d_%H%M%S).csv"

[ -f "$TRIALS_FILE" ]  || { echo "✗ trials 表不存在: $TRIALS_FILE"; exit 1; }
[ -f "$RUN_TRIAL" ]    || { echo "✗ run_trial.sh 不存在"; exit 1; }
mkdir -p "$RESULTS_DIR"

echo "tier,world_id,trial,filename,success_deploy,success_fb,collision,timeout,final_dxy,min_clearance,time_to_goal,setup_failed" > "$SUMMARY_CSV"

n_total=0; n_ok=0
while read -r tier wid tno world obs gx gy gz _rest; do
    # 跳过空行/注释
    [ -z "${tier:-}" ] && continue
    case "$tier" in \#*) continue;; esac

    n_total=$((n_total+1))
    out="$RESULTS_DIR/${tier}_w${wid}_t${tno}.json"

    echo ""
    echo "################################################################"
    echo "# [$n_total] $tier w$wid t$tno  →  $out"
    echo "################################################################"

    bash "$RUN_TRIAL" "$world" "$obs" "$gx" "$gy" "$gz" "$out" "$tier" "$wid" "$tno"

    # 汇总到 CSV
    if [ -f "$out" ]; then
        python3 -c "
import json
d=json.load(open('$out'))
def g(k,dv=''):
    v=d.get(k,dv); return '' if v is None else v
row=[d.get('tier',''),d.get('world_id',''),d.get('trial',''),d.get('filename',''),
     g('success_deploy'),g('success_flightbench'),g('collision'),g('timeout'),
     '%.3f'%d.get('final_dist_xy',0),'%.3f'%d.get('min_clearance',0),
     '%.2f'%d.get('time_to_goal',0),g('setup_failed')]
print(','.join(str(x) for x in row))
" >> "$SUMMARY_CSV"
        # 统计成功
        succ=$(python3 -c "import json;print(json.load(open('$out')).get('success_deploy'))")
        [ "$succ" = "True" ] && n_ok=$((n_ok+1))
    else
        echo "$tier,$wid,$tno,,ERROR_NO_JSON,,,,,,," >> "$SUMMARY_CSV"
    fi

    # trial 间隔, 让端口/进程彻底释放
    sleep 5
done < "$TRIALS_FILE"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  批量完成: $n_ok / $n_total deploy-success"
echo "  汇总 CSV: $SUMMARY_CSV"
echo "════════════════════════════════════════════════════════════"
column -t -s, "$SUMMARY_CSV" 2>/dev/null || cat "$SUMMARY_CSV"