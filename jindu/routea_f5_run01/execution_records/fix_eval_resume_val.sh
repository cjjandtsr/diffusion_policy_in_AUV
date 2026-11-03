#!/usr/bin/env bash
# F5 evaluation, part 2: val stage only, after the frozen val_bank absolute-path repair.
#
# Part 1 (fix_eval_resume.sh) completed: 00_tests rc=0 (24 passed) and dev for bc/P6/P8/P9/P10/P11
# (six summaries x 420 records = 2520). It then aborted at val/bc with
# FileNotFoundError on an obsolete absolute obstacle_path inside the frozen val bank.
# Remedy: symlink /home/cjj/AUV/program/results/routea_xk_run01/val_bank -> jindu/.../val_bank
# (bytes unchanged; 20/20 obstacle sha256 equal the recorded obstacle_sha256).
# Failed scene archived at evaluation_gate_fix_v1/val_bc_failed_before_episode/.
#
# This script reruns nothing that already succeeded: val for six policies, then strata + analysis.
set -euo pipefail
source /home/cjj/AUV/program/task_distribution_implementation_20261031/env.sh
: "${ROOT:?}" "${OUT:?}" "${WEIGHTS:?}" "${DEV_BANK:?}" "${VAL_BANK:?}" "${P6_NATIVE:?}" "${P6_CHECKPOINT:?}"
PY=/home/cjj/miniconda3/envs/idp3/bin/python
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# The evaluation helper fix from decision B must still be the delivered bytes.
printf '%s  %s\n' f86538e135184f5b16ecda87400e7636d6564e723617fefeb73750690d6158f6 \
  scripts/routea_paired_support/runtime.py | sha256sum -c -

# Precondition: the frozen val bank must resolve, and every obstacle must hash-match.
export VAL_BANK
"$PY" - <<'PYEOF'
import hashlib, json, os, sys
bank = os.environ["VAL_BANK"]
b = json.load(open(bank))
bad = []
for uid, w in b["worlds"].items():
    p = w["obstacle_path"]
    if not os.path.exists(p):
        bad.append((uid, p, "missing")); continue
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    if h != w["obstacle_sha256"]:
        bad.append((uid, p, "sha mismatch"))
print("val_bank=%s worlds=%d obstacles_bad=%d" % (bank, len(b["worlds"]), len(bad)))
if bad:
    for row in bad[:5]:
        print("BAD", row)
    sys.exit(3)
PYEOF

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

BANK="$VAL_BANK"; REP=1; ESPLIT=rl_val
for P in bc P6 P8 P9 P10 P11; do
  if [ "$P" = bc ] || [ "$P" = P6 ]; then
    SOURCE="$P6_NATIVE"; CKPT="$P6_CHECKPOINT"; U=256
  else
    SOURCE="$ROOT"; CKPT="$WEIGHTS/$P/checkpoint_update_000192"; U=192
  fi
  test ! -e "$OUT/eval_val/full/$P"
  run_fix "val_${P}" "$PY" scripts/run_routea_xk_evaluation.py worker \
    --repo "$ROOT" --current-root "$ROOT" --source-root "$SOURCE" \
    --checkpoint "$CKPT" --expected-update "$U" --policy "$P" \
    --bank "$BANK" --split "$ESPLIT" \
    --worlds /home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds/train \
    --output "$OUT/eval_val/full/$P" --repeats "$REP" --clip 2.0 \
    --min-std 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1
  echo "F5_EVAL_DONE val/${P}"
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
echo "F5_EVAL_RESUME_VAL_DONE"
