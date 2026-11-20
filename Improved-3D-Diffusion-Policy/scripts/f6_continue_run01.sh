#!/usr/bin/env bash
# Continue the preserved F6 run01 with explicit repeatability-policy receipts.
set -euo pipefail
PY=/home/cjj/miniconda3/envs/idp3/bin/python
BASE=/home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy
ROOT=/home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy-F6-run01
OUT=/home/cjj/AUV/program/results/routea_f6_run01
OLD=/home/cjj/AUV/program/results/routea_f5_run01
WEIGHTS="$BASE/data/outputs/dppo_uav_rebuild/routea_f6_run01"
OLD_W="$BASE/data/outputs/dppo_uav_rebuild/routea_f5_run01"
P9_DIAG=/home/cjj/AUV/program/results/routea_f6_P9_repeatability_run01
TOOLS=/mnt/c/Users/cjj/Desktop/IDP3/program/f6_continue_20261108
CONT="$OUT/continuation_policy_20261108"
DEV_BANK="$BASE/data/outputs/diagnostics/routea_bc_a0_r1_paired_dev_20261010_py38fix_run02/bank/bank.json"
VAL_BANK=/home/cjj/AUV/program/results/routea_xk_run01/val_bank/bank.json
WORLDS=/home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds/train
test ! -e "$CONT"
mkdir -p "$CONT/execution"
run_logged() {
  local name="$1"; shift
  printf '%q ' "$@" > "$CONT/execution/$name.command.txt"
  printf '\n' >> "$CONT/execution/$name.command.txt"
  set +e
  "$@" 2>&1 | tee "$CONT/execution/$name.log"
  local rc=${PIPESTATUS[0]}
  set -e
  printf '%s\n' "$rc" > "$CONT/execution/$name.rc"
  return "$rc"
}
# Preserve the execution-side Repair 5: isolated helper imports plus user-site deps.
HELPER_PY=("$PY" -I -c 'import sys,runpy; sys.path.append("/home/cjj/.local/lib/python3.8/site-packages"); p=sys.argv[1]; del sys.argv[1]; runpy.run_path(p,run_name="__main__")')
DIAG="$TOOLS/f6_repeatability_diagnostics.py"
ACCEPT="$TOOLS/f6_training_acceptance.py"
run_logged closure "$PY" -I "$TOOLS/f6_base_closure_check.py" \
  "$OLD/prepared/prepared.json" "$OUT/delivery_sha256.json" "$BASE" "$ROOT"
run_logged P9_accept "${HELPER_PY[@]}" "$ACCEPT" --repo "$ROOT" --prepared "$OUT/prepared" \
  --baseline "$P9_DIAG/compare_baseline" --d1 "$P9_DIAG/compare_D1" --d2 "$P9_DIAG/compare_D2" \
  --output "$CONT/P9_training_acceptance.json"
cd "$ROOT"
run_logged P11_u129 "$PY" scripts/run_routea_f6.py run --repo "$ROOT" \
  --prepared "$OUT/prepared" --run-id P11 --stop-after-update 129
run_logged P11_compare "${HELPER_PY[@]}" "$DIAG" --repo "$ROOT" compare \
  --prepared "$OUT/prepared" --control P11 \
  --old-metrics "$OLD/prepared/training/P11/metrics_update_000129.json" \
  --new-metrics "$OUT/prepared/reproduction/P11/metrics_update_000129.json" \
  --old-checkpoint "$OLD_W/P11/checkpoint_update_000129" \
  --new-checkpoint "$WEIGHTS/reproduction/P11/checkpoint_update_000129" --output "$CONT/P11_baseline"
EXTRA=()
if "$PY" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["formal_passed"] else 1)' "$CONT/P11_baseline/diagnostic_summary.json"; then
  : # Exact repetition needs no extra GPU diagnostics.
else
  DOUT="$CONT/P11_repeatability"
  DW="$BASE/data/outputs/dppo_uav_rebuild/routea_f6_P11_diagnostic_only_run01"
  run_logged P11_stage "${HELPER_PY[@]}" "$DIAG" --repo "$ROOT" stage \
    --f5-prepared "$OLD/prepared" --f6-prepared "$OUT/prepared" --control P11 \
    --output "$DOUT/prepared" --weights "$DW"
  cd "$BASE"
  run_logged P11_D1 "$PY" scripts/run_routea_f5.py run --repo "$BASE" \
    --prepared "$DOUT/prepared/D1/run_prepared" --run-id P11 --stop-after-update 129
  cd "$ROOT"
  run_logged P11_D2 "$PY" scripts/run_routea_f6.py run --repo "$ROOT" \
    --prepared "$DOUT/prepared/D2/run_prepared" --run-id P11 --stop-after-update 129
  for D in D1 D2; do
    if [ "$D" = D1 ]; then REF_M="$OLD/prepared/training/P11"; REF_W="$OLD_W/P11";
    else REF_M="$OUT/prepared/reproduction/P11"; REF_W="$WEIGHTS/reproduction/P11"; fi
    run_logged "P11_${D}_compare" "${HELPER_PY[@]}" "$DIAG" --repo "$ROOT" compare \
      --prepared "$DOUT/prepared/$D/compare_prepared" --control P11 \
      --old-metrics "$REF_M/metrics_update_000129.json" --new-metrics "$DOUT/prepared/$D/metrics/metrics_update_000129.json" \
      --old-checkpoint "$REF_W/checkpoint_update_000129" --new-checkpoint "$DW/$D/P11/checkpoint_update_000129" \
      --output "$DOUT/compare_$D"
  done
  EXTRA=(--d1 "$DOUT/compare_D1" --d2 "$DOUT/compare_D2")
fi
run_logged P11_accept "${HELPER_PY[@]}" "$ACCEPT" --repo "$ROOT" --prepared "$OUT/prepared" \
  --baseline "$CONT/P11_baseline" "${EXTRA[@]}" --output "$CONT/P11_training_acceptance.json"
run_logged migrate_eval "$PY" scripts/routea_f6_support.py rebind-evaluation \
  --prepared "$OUT/prepared" --repo "$ROOT" --source-root "$OUT/native_f5" \
  --checkpoint "$OLD_W/P9/checkpoint_update_000192" \
  --output "$WEIGHTS/evaluation_reproduction/F6_G_A6_copy_u192" --report "$CONT/evaluation_identity_migration.json"
run_logged eval_reproduce "$PY" scripts/run_routea_xk_evaluation.py worker \
  --repo "$ROOT" --current-root "$ROOT" --source-root "$ROOT" \
  --checkpoint "$WEIGHTS/evaluation_reproduction/F6_G_A6_copy_u192" --expected-update 192 \
  --policy F6_G_A6_evaluation_copy --evaluation-reproduction --bank "$DEV_BANK" --split rl_train --worlds "$WORLDS" \
  --output "$OUT/reproduction_eval/full/F6_G_A6_evaluation_copy" --repeats 5 --clip 2.0 \
  --min-std 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1
run_logged compare_eval "$PY" scripts/routea_f6_support.py compare-evaluation \
  --prepared "$OUT/prepared" --migration "$CONT/evaluation_identity_migration.json" \
  --old "$OLD/eval_dev/full/P9" --new "$OUT/reproduction_eval/full/F6_G_A6_evaluation_copy" \
  --bank "$DEV_BANK" --output "$CONT/evaluation_reproduction.json"
run_logged seal "$PY" scripts/routea_f6_support.py seal-gates --prepared "$OUT/prepared" \
  --p9 "$CONT/P9_training_acceptance.json" --p11 "$CONT/P11_training_acceptance.json" \
  --evaluation "$CONT/evaluation_reproduction.json" --output "$CONT/reproduction_gates.json"
for P in P13 P12; do
  run_logged "${P}_u129" "$PY" scripts/run_routea_f6.py run --repo "$ROOT" \
    --prepared "$OUT/prepared" --run-id "$P" --stop-after-update 129 --gates "$CONT/reproduction_gates.json"
  run_logged "${P}_u192" "$PY" scripts/run_routea_f6.py run --repo "$ROOT" \
    --prepared "$OUT/prepared" --run-id "$P" --stop-after-update 192 \
    --resume "$WEIGHTS/$P/checkpoint_update_000129" --gates "$CONT/reproduction_gates.json"
  run_logged "${P}_monitor" "$PY" scripts/analyze_routea_f6.py monitor \
    --metrics "$OUT/prepared/training/$P" --output "$CONT/${P}_monitor.json"
done
for SPLIT in dev val; do
  if [ "$SPLIT" = dev ]; then BANK="$DEV_BANK"; REP=5; ESPLIT=rl_train;
  else BANK="$VAL_BANK"; REP=1; ESPLIT=rl_val; fi
  for P in P12 P13; do
    run_logged "eval_${SPLIT}_${P}" "$PY" scripts/run_routea_xk_evaluation.py worker \
      --repo "$ROOT" --current-root "$ROOT" --source-root "$ROOT" \
      --checkpoint "$WEIGHTS/$P/checkpoint_update_000192" --expected-update 192 --policy "$P" \
      --bank "$BANK" --split "$ESPLIT" --worlds "$WORLDS" --output "$OUT/eval_${SPLIT}/full/$P" \
      --repeats "$REP" --clip 2.0 --min-std 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1
  done
done
run_logged analyze "$PY" scripts/analyze_routea_f6.py evaluate --prepared "$OUT/prepared" \
  --gates "$CONT/reproduction_gates.json" --dev "$OUT/eval_dev" --val "$OUT/eval_val" \
  --old-dev "$OLD/eval_dev" --old-val "$OLD/eval_val" --dev-bank "$DEV_BANK" --val-bank "$VAL_BANK" \
  --dev-strata "$OLD/dev_strata.json" --val-strata "$OLD/val_strata.json" --output "$CONT/analysis.json"
