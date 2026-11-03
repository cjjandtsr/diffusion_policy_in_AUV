#!/usr/bin/env bash
# F5 unified evaluation: six policies x (dev420 repeats=5, val20 repeats=1) = 3000 condition records.
set -euo pipefail
source /home/cjj/AUV/program/task_distribution_implementation_20261031/env.sh
cd "$ROOT"
for SPLIT in dev val; do
  if [ "$SPLIT" = dev ]; then
    BANK="$DEV_BANK"; REP=5; ESPLIT=rl_train
  else
    BANK="$VAL_BANK"; REP=1; ESPLIT=rl_val
  fi
  for P in bc P6 P8 P9 P10 P11; do
    if [ "$P" = bc ] || [ "$P" = P6 ]; then
      SOURCE="$P6_NATIVE"; CKPT="$P6_CHECKPOINT"; U=256
    else
      SOURCE="$ROOT"; CKPT="$WEIGHTS/$P/checkpoint_update_000192"; U=192
    fi
    run_logged "eval_${SPLIT}_${P}" "$PY" scripts/run_routea_xk_evaluation.py worker \
      --repo "$ROOT" --current-root "$ROOT" --source-root "$SOURCE" \
      --checkpoint "$CKPT" --expected-update "$U" --policy "$P" \
      --bank "$BANK" --split "$ESPLIT" \
      --worlds "$WORLDS" \
      --output "$OUT/eval_${SPLIT}/full/$P" --repeats "$REP" --clip 2.0 \
      --min-std 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1
    echo "F5_EVAL_DONE ${SPLIT}/${P}"
  done
done
echo "F5_EVAL_MATRIX_DONE"
