#!/usr/bin/env bash
# F10 step-2 probe driver: probes T1, T4, T5, T6 timing; comparison and chain probe after.
set -u
ROOT=/home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy
RUN=/home/cjj/AUV/program/results/routea_f10_run01
PY=/home/cjj/miniconda3/envs/idp3/bin/python
cd "$ROOT"
run_one() {
  local name="$1"; shift
  echo "$PY scripts/run_routea_f10_eval.py worker --output $RUN $*" > "$RUN/execution/${name}.command.txt"
  "$PY" scripts/run_routea_f10_eval.py worker --output "$RUN" "$@" > "$RUN/execution/${name}.log" 2>&1
  echo $? > "$RUN/execution/${name}.rc"
  echo "PROBE_DONE ${name} rc=$(cat "$RUN/execution/${name}.rc")"
}
run_one T1_none      --policy P6 --guidance none --split dev --limit 10 --tag T1_none
run_one T1_g1_eta0   --policy P6 --guidance g1 --eta 0 --split dev --limit 10 --tag T1_g1_eta0
run_one T4_BC_none   --policy BC --guidance none --split dev --limit 10 --tag T4_BC_none
run_one T5_P6_G0     --policy P6 --guidance G0 --split dev --limit 10 --tag T5_P6_G0
run_one T6_P6_g1     --policy P6 --guidance g1 --split dev --limit 10 --tag T6_P6_g1
echo PROBE_SEQUENCE_DONE
