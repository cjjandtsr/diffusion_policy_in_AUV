#!/usr/bin/env bash
# Only the missing frozen paper matrix. Preserve every failed attempt.
set -eo pipefail
POLICY=/home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy
ROS_WS=/home/cjj/AUV/catkin_ws0
PY=/home/cjj/miniconda3/envs/idp3/bin/python
RUN=/home/cjj/AUV/program/results/routea_paired_evaluation/paper_ego_common_h4_v1_1_new120_run01
BASE=/home/cjj/AUV/program/paper_p3_limited_v1_run01/results
source /opt/ros/noetic/setup.bash
source "$ROS_WS/devel/setup.bash"
set -u
export ROS_MASTER_URI=http://127.0.0.1:11329
export PYTHONDONTWRITEBYTECODE=1
cd "$POLICY"
mkdir -p "$RUN/logs"
exec 9>"$RUN/.runner.lock"
flock -n 9 || { echo 'Another runner holds this output'; exit 1; }
export POLICY RUN
"$PY" - <<'PY'
import os,json,hashlib
from pathlib import Path
p=Path(os.environ['POLICY'])
m=json.loads((p/'scripts/routea_ego_fix_v1_1.json').read_text())
for item in m['files']:
    assert hashlib.sha256((p/item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
print('FIX_HASH_OK',len(m['files']))
PY
# There is no service-level retry: restart the entire uncommitted episode.
for attempt in 1 2 3; do
  log="$RUN/logs/new.$(date +%Y%m%dT%H%M%S).attempt$attempt.log"
  resume=()
  [ ! -d "$RUN/new" ] || resume=(--resume)
  set +e
  "$PY" scripts/run_routea_paired_comparison.py \
    --policy ego --task-bank "$BASE/bank/bank.json" \
    --ego-launch "$ROS_WS/src/ego-planner-swarm/src/planner/plan_manage/launch/paper_ego_eval.launch" \
    --compare-existing "$BASE/full/bc" \
    --compare-existing "$BASE/full/rl_2f3f9914db78_u32" \
    --out "$RUN/new" "${resume[@]}" 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  set -e
  printf '%s\n' "$rc" > "$log.rc"
  if [ "$rc" -eq 0 ]; then
    "$PY" - <<'PY'
import os,json
from pathlib import Path
p=Path(os.environ['RUN'])/'new'
d=json.loads((p/'complete.json').read_text())
assert d['status']=='complete' and d['episodes']==120,d
assert len(list((p/'full/ego/commits').glob('*.json')))==120
print('NEW120_COMPLETE')
PY
    exit 0
  fi
  # Match the terminal exception, not arbitrary words in an earlier log line.
  if ! tail -n 1 "$log" | tr -d '\r' | grep -Fxq 'rospy.service.ServiceException: unable to connect to service: [Errno 104] Connection reset by peer'; then
    echo 'STOP: not the observed connection-handshake failure'
    exit "$rc"
  fi
  [ "$attempt" -lt 3 ] || { echo 'STOP: retry budget exhausted'; exit "$rc"; }
  echo 'Retrying uncommitted episode after connection reset; previous evidence retained'
  sleep 15
done
