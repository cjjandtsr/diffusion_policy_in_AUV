#!/usr/bin/env bash
# F10 step 3: formal evaluation of S1 (P6 g1) and S4 (BC g1) on dev420 and val20.
set -u
ROOT=/home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy
RUN=/home/cjj/AUV/program/results/routea_f10_run01
PY=/home/cjj/miniconda3/envs/idp3/bin/python
cd "$ROOT"
run_arm() {
  local name="$1"; shift
  echo "$PY scripts/run_routea_f10_eval.py worker --output $RUN $*" > "$RUN/execution/${name}.command.txt"
  date +%s > "$RUN/execution/${name}.start"
  "$PY" scripts/run_routea_f10_eval.py worker --output "$RUN" "$@" > "$RUN/execution/${name}.log" 2>&1
  echo $? > "$RUN/execution/${name}.rc"
  date +%s > "$RUN/execution/${name}.end"
  echo "ARM_DONE ${name} rc=$(cat "$RUN/execution/${name}.rc")"
}
run_arm S1_P6_g1_dev --policy P6 --guidance g1 --split dev --full-run-approved
run_arm S4_BC_g1_dev --policy BC --guidance g1 --split dev --full-run-approved
run_arm S1_P6_g1_val --policy P6 --guidance g1 --split val --full-run-approved
run_arm S4_BC_g1_val --policy BC --guidance g1 --split val --full-run-approved
echo ALL_ARMS_DONE
