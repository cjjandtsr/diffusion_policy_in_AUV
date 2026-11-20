#!/bin/bash
# ══════════════════════════════════════════════════════════════
# run_bank_sample.sh — 按任务清单逐条跑 run_task_flight.sh 并汇总
#
#   清单 TSV 每行 (制表符分隔, # 开头为注释):
#     world_name  category  start_x start_y start_z start_yaw  goal_x goal_y goal_z
#   坐标全部是**世界系**; start_z 同时作为 offboard 起飞目标高度。
#
#   用法: bash run_bank_sample.sh <tasks.tsv> <out_dir>
#   环境: TIMEOUT(默认150) SPAWN_Z(默认0) TAKEOFF_S(默认30) CKPT
#
#   产出: <out_dir>/<序号>_<world>__<category>.json      每个 trial 的 monitor JSON
#         <out_dir>/<序号>_..._state.csv                  轨迹
#         <out_dir>/summary.tsv                           逐条汇总
# ══════════════════════════════════════════════════════════════
set -u

SCRIPTS_DIR="/root/autodl-tmp/AUV/catkin_ws0/scripts"
RUN_ONE="$SCRIPTS_DIR/run_task_flight.sh"
WORLDS_DIR="/root/autodl-tmp/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds/test"

LIST="${1:?用法: run_bank_sample.sh <tasks.tsv> <out_dir>}"
OUT_DIR="${2:?用法: run_bank_sample.sh <tasks.tsv> <out_dir>}"
TIMEOUT="${TIMEOUT:-150}"
export SPAWN_Z="${SPAWN_Z:-0}"
export TAKEOFF_S="${TAKEOFF_S:-30}"
export TIMEOUT

[ -f "$LIST" ] || { echo "✗ 清单不存在: $LIST"; exit 2; }
[ -f "$RUN_ONE" ] || { echo "✗ run_task_flight.sh 不存在"; exit 2; }
mkdir -p "$OUT_DIR"

SUMMARY="$OUT_DIR/summary.tsv"
printf 'idx\tworld\tcategory\tsuccess_deploy\tsuccess_fb\tcollision\ttimeout\tlost\tfinal_dxy\tmin_clearance\ttime_to_goal\tsetup_failed\n' > "$SUMMARY"

n_total=$(grep -vc '^\s*\(#\|$\)' "$LIST")
i=0
while IFS=$'\t' read -r world cat sx sy sz syaw gx gy gz; do
    [ -z "${world:-}" ] && continue
    case "$world" in \#*) continue;; esac
    # 容错: 字段可能多带空白
    world=$(echo "$world" | tr -d ' \r')

    i=$((i + 1))
    wfile="$WORLDS_DIR/${world}.world"
    out="$OUT_DIR/$(printf '%02d' "$i")_${world}__${cat}.json"

    echo ""
    echo "############################################################"
    echo "# [$i/$n_total] $cat  ${world}"
    echo "#   start=(${sx}, ${sy}, ${sz}) yaw=${syaw}"
    echo "#   goal =(${gx}, ${gy}, ${gz})"
    echo "#   out  =$out"
    echo "############################################################"

    if [ ! -f "$wfile" ]; then
        echo "  ✗ world 文件缺失: $wfile"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$i" "$world" "$cat" "" "" "" "" "" "" "" "" "world_missing" >> "$SUMMARY"
        continue
    fi

    bash "$RUN_ONE" "$wfile" "$sx" "$sy" "$syaw" "$sz" "$gx" "$gy" "$gz" \
        "$out" "$cat" "$i" 0
    rc=$?
    echo "  run_task_flight rc=$rc"

    if [ -f "$out" ]; then
        python3 - "$i" "$world" "$cat" "$out" <<'PY' >> "$SUMMARY"
import json, sys
idx, world, cat, path = sys.argv[1:5]
d = json.load(open(path))
def g(k):
    v = d.get(k, '')
    if v is None: return ''
    if isinstance(v, bool): return 'True' if v else 'False'
    if isinstance(v, float): return f'{v:.3f}' if v == v else 'inf'
    return str(v)
print('\t'.join([idx, world, cat, g('success_deploy'), g('success_flightbench'),
                 g('collision'), g('timeout'), g('lost'), g('final_dist_xy'),
                 g('min_clearance'), g('time_to_goal'), g('setup_failed')]))
PY
    else
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$i" "$world" "$cat" "" "" "" "" "" "" "" "" "no_json" >> "$SUMMARY"
    fi

    sleep 5
done < "$LIST"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  样本完成: $i / $n_total"
echo "  汇总: $SUMMARY"
echo "════════════════════════════════════════════════════════════"
column -t -s$'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
