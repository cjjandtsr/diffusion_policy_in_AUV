#!/usr/bin/env bash
# F5 evaluation resume after the v3 gate fix (decision B, guide "F5评价v3门禁修复与接续 v1.0").
# Runs dev for the four trained arms (bc/P6 dev kept from the first pass), then val for all six,
# then strata freeze + paired analysis. Every step records command/log/rc under OUT/evaluation_gate_fix_v1.
set -euo pipefail
source /home/cjj/AUV/program/task_distribution_implementation_20261031/env.sh
: "${ROOT:?}" "${OUT:?}" "${WEIGHTS:?}" "${DEV_BANK:?}" "${VAL_BANK:?}" "${P6_NATIVE:?}" "${P6_CHECKPOINT:?}"
PY=/home/cjj/miniconda3/envs/idp3/bin/python
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
printf '%s  %s\n' f86538e135184f5b16ecda87400e7636d6564e723617fefeb73750690d6158f6 \
  scripts/routea_paired_support/runtime.py | sha256sum -c -

FIX="$OUT/evaluation_gate_fix_v1"
run_fix() {
  local name="$1"; shift
  local prefix="$FIX/$name"
  test ! -e "$prefix.rc"
  printf '%q ' "$@" > "$prefix.command.txt"
  set +e
  "$@" 2>&1 | tee "$prefix.log"
  local rc=${PIPESTATUS[0]}
  set -e
  printf '%s\n' "$rc" > "$prefix.rc"
  return "$rc"
}

run_fix 00_tests "$PY" -m pytest tests/routea_paired/test_common_environment_versions.py \
  tests/routea_paired/test_f5_analysis.py tests/routea_paired/test_xk_evaluation.py \
  tests/dppo_uav_rebuild/rewards/test_reward_v3.py -q

for SPLIT in dev val; do
  if [ "$SPLIT" = dev ]; then
    BANK="$DEV_BANK"; REP=5; ESPLIT=rl_train; POLICIES='P8 P9 P10 P11'
  else
    BANK="$VAL_BANK"; REP=1; ESPLIT=rl_val; POLICIES='bc P6 P8 P9 P10 P11'
  fi
  for P in $POLICIES; do
    if [ "$P" = bc ] || [ "$P" = P6 ]; then
      SOURCE="$P6_NATIVE"; CKPT="$P6_CHECKPOINT"; U=256
    else
      SOURCE="$ROOT"; CKPT="$WEIGHTS/$P/checkpoint_update_000192"; U=192
    fi
    test ! -e "$OUT/eval_${SPLIT}/full/$P"
    run_fix "${SPLIT}_${P}" "$PY" scripts/run_routea_xk_evaluation.py worker \
      --repo "$ROOT" --current-root "$ROOT" --source-root "$SOURCE" \
      --checkpoint "$CKPT" --expected-update "$U" --policy "$P" \
      --bank "$BANK" --split "$ESPLIT" \
      --worlds /home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds/train \
      --output "$OUT/eval_${SPLIT}/full/$P" --repeats "$REP" --clip 2.0 \
      --min-std 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1
  done
done

test ! -e "$OUT/dev_strata.json"
run_fix strata_dev "$PY" scripts/analyze_routea_f5.py freeze-strata \
  --split dev420 --bank "$DEV_BANK" --output "$OUT/dev_strata.json"
test ! -e "$OUT/val_strata.json"
run_fix strata_val "$PY" scripts/analyze_routea_f5.py freeze-strata \
  --split val20 --bank "$VAL_BANK" --output "$OUT/val_strata.json"
test ! -e "$OUT/analysis.json"
run_fix analyze "$PY" scripts/analyze_routea_f5.py evaluate \
  --dev "$OUT/eval_dev" --dev-bank "$DEV_BANK" --dev-strata "$OUT/dev_strata.json" \
  --val "$OUT/eval_val" --val-bank "$VAL_BANK" --val-strata "$OUT/val_strata.json" \
  --output "$OUT/analysis.json"
echo "F5_EVAL_RESUME_DONE"
