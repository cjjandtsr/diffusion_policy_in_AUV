#!/bin/bash
# ══════════════════════════════════════════════════════════════
# run_batch_ego.sh — 本地 EGO 批量 (20 world × N轮), 与 BC 同 monitor 可比
#   用法: bash run_batch_ego.sh <trials.txt> [results_dir]
#   trials 行: <tier> <wid> <trial> <world> <obstacles_json> <gx> <gy> <gz>
# ══════════════════════════════════════════════════════════════
set -u
SCRIPTS_DIR="/home/cjj/AUV/catkin_ws0/scripts"
RUN_TRIAL="$SCRIPTS_DIR/run_trial_ego.sh"

TRIALS_FILE="${1:?need trials.txt}"
RESULTS_DIR="${2:-/home/cjj/AUV/results_ego}"
SUMMARY_CSV="$RESULTS_DIR/summary_ego_$(date +%Y%m%d_%H%M%S).csv"

[ -f "$TRIALS_FILE" ] || { echo "✗ trials 不存在"; exit 1; }
[ -f "$RUN_TRIAL" ]   || { echo "✗ run_trial_ego.sh 不存在"; exit 1; }
mkdir -p "$RESULTS_DIR"

echo "tier,world_id,trial,filename,success_deploy,success_fb,collision,timeout,final_dxy,min_clearance,time_to_goal,setup_failed" > "$SUMMARY_CSV"

n=0; ok=0
while read -r tier wid tno world obs gx gy gz _r; do
    [ -z "${tier:-}" ] && continue
    case "$tier" in \#*) continue;; esac
    n=$((n+1))
    out="$RESULTS_DIR/${tier}_w${wid}_t${tno}.json"
    echo ""; echo "### [$n] $tier w$wid t$tno → $out"
    bash "$RUN_TRIAL" "$world" "$obs" "$gx" "$gy" "$gz" "$out" "$tier" "$wid" "$tno"
    if [ -f "$out" ]; then
        python3 -c "
import json; d=json.load(open('$out'))
def g(k):
    v=d.get(k,''); return '' if v is None else v
print(','.join(str(x) for x in [d.get('tier',''),d.get('world_id',''),d.get('trial',''),d.get('filename',''),g('success_deploy'),g('success_flightbench'),g('collision'),g('timeout'),'%.3f'%d.get('final_dist_xy',0),'%.3f'%d.get('min_clearance',0),'%.2f'%d.get('time_to_goal',0),g('setup_failed')]))
" >> "$SUMMARY_CSV"
        s=$(python3 -c "import json;print(json.load(open('$out')).get('success_deploy'))")
        [ "$s" = "True" ] && ok=$((ok+1))
    fi
    sleep 5
done < "$TRIALS_FILE"

echo ""; echo "═══ EGO 批量完成: $ok/$n deploy-success ═══"
echo "CSV: $SUMMARY_CSV"
column -t -s, "$SUMMARY_CSV" 2>/dev/null || cat "$SUMMARY_CSV"