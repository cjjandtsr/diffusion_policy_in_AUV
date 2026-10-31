#!/bin/bash
# Route A X/K E1 evaluation (C2/E1 guide v1.0, steps 12-17), after both arms u256.
# 12/13: dev 1680 (BC/u128/K256/X256) + analysis; 14/15: val 192/256 (K/X) + trend;
# 16/17: common clip2 1260 (BC/K/X, clip2 selected) + trend analysis.
set -euo pipefail
ROOT=/home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy
PY=/home/cjj/miniconda3/envs/idp3/bin/python
A2=/home/cjj/AUV/program/results/routea_a2_selective_keep_run01
TEST12=/home/cjj/AUV/program/results/routea_test12_independent_run02
OUT=/home/cjj/AUV/program/results/routea_xk_run01
WEIGHTS=$ROOT/data/outputs/dppo_uav_rebuild/routea_xk_run01
cd "$ROOT"
run_logged() {
  local label=$1
  shift
  printf '%q ' "$@" > "$OUT/execution/$label.command.txt"
  printf '\n' >> "$OUT/execution/$label.command.txt"
  set +e
  "$@" > "$OUT/execution/$label.log" 2>&1
  local rc=$?
  set -e
  printf '%s\n' "$rc" > "$OUT/execution/$label.rc"
  test "$rc" -eq 0
}
REF=$(cat "$OUT/execution/dev_bank_path.txt")
test -n "$REF"
mkdir -p "$OUT/e1_plans"
"$PY" - "$ROOT" "$OUT" "$WEIGHTS" "$TEST12" <<'PY'
import json,pathlib,sys
root,out,weights,test12=map(pathlib.Path,sys.argv[1:])
receipt=json.loads((out/'inputs/inputs.json').read_text())
parent=json.loads(pathlib.Path(receipt['receipt']).read_text())['parent_checkpoint']
control=root/'data/outputs/dppo_uav_rebuild/routea_a2_selective_keep_run01/C/checkpoint_update_000256'
def model(name,path,update,source):
    return dict(policy=name,checkpoint=str(path),update=update,source_root=str(source))
bc=model('bc',control,256,test12/'native_a2')
u128=model('u128',parent,128,test12/'native_a1')
def arms(update):
    return [model(a,weights/'main'/a/('checkpoint_update_%06d'%update),update,root) for a in ['K','X']]
def save(name,models,clip):
    with (out/'e1_plans'/name).open('x') as f:
        json.dump(dict(models=models,denoised_clip_value=clip,min_std_vector=[.1]*11),f,indent=2)
save('dev_original.json',[bc,u128]+arms(256),1.)
for update in [192,256]:save('val_%d.json'%update,arms(update),1.)
selection=json.loads((out/'c2/selection.json').read_text())['selected_protocol']
if selection and selection['denoised_clip_value']==2.:
    save('dev_common_clip2.json',[bc]+arms(256),2.)
PY
VAL_WORLDS=/home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds/val
run_logged 12_dev "$PY" scripts/run_routea_xk_evaluation.py matrix --repo "$ROOT" \
  --plan "$OUT/e1_plans/dev_original.json" --bank "$REF/bank/bank.json" \
  --worlds "$VAL_WORLDS" --output "$OUT/e1_dev_original"
run_logged 13_dev_analysis "$PY" scripts/run_routea_xk_evaluation.py analyze-e1 --repo "$ROOT" \
  --evaluation "$OUT/e1_dev_original" --bank "$REF/bank/bank.json" \
  --x-metrics "$OUT/main/X/metrics_update_000256.json" --k-metrics "$OUT/main/K/metrics_update_000256.json" \
  --historical-c "$A2/evaluation_dev/full/rl_910559bb5de2_u256" \
  --output "$OUT/e1_dev_original/analysis.json"
for update in 192 256; do
  run_logged "14_val_$update" "$PY" scripts/run_routea_xk_evaluation.py matrix --repo "$ROOT" \
    --plan "$OUT/e1_plans/val_$update.json" --bank "$OUT/val_bank/bank.json" \
    --worlds "$VAL_WORLDS" --split rl_val --output "$OUT/e1_val_$update"
  run_logged "15_val_analysis_$update" "$PY" scripts/run_routea_xk_evaluation.py analyze-e1 --repo "$ROOT" \
    --evaluation "$OUT/e1_val_$update" --bank "$OUT/val_bank/bank.json" --trend-only \
    --x-metrics "$OUT/main/X/metrics_update_$(printf '%06d' "$update").json" \
    --k-metrics "$OUT/main/K/metrics_update_$(printf '%06d' "$update").json" \
    --output "$OUT/e1_val_$update/analysis.json"
done
if test -f "$OUT/e1_plans/dev_common_clip2.json"; then
  run_logged 16_clip2 "$PY" scripts/run_routea_xk_evaluation.py matrix --repo "$ROOT" \
    --plan "$OUT/e1_plans/dev_common_clip2.json" --bank "$REF/bank/bank.json" \
    --worlds "$VAL_WORLDS" --output "$OUT/e1_dev_common_clip2"
  run_logged 17_clip2_analysis "$PY" scripts/run_routea_xk_evaluation.py analyze-e1 --repo "$ROOT" \
    --evaluation "$OUT/e1_dev_common_clip2" --bank "$REF/bank/bank.json" --trend-only \
    --output "$OUT/e1_dev_common_clip2/analysis.json"
fi
