set -euo pipefail
ROOT=/home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy
PY=/home/cjj/miniconda3/envs/idp3/bin/python
PROGRAM=/home/cjj/AUV/program
A2=$PROGRAM/results/routea_a2_selective_keep_run01
OUT=$PROGRAM/results/routea_test12_independent_run02
WEIGHTS=$ROOT/data/outputs/dppo_uav_rebuild
PARENT=$WEIGHTS/routea_common_b2560_seed202609203_v1/checkpoint_update_000128
CONTROL=$WEIGHTS/routea_a2_selective_keep_run01/C/checkpoint_update_000256
TREATMENT=$WEIGHTS/routea_a2_selective_keep_run01/T/checkpoint_update_000256
TEST_WORLDS=/home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds/test
cd "$ROOT"
test "$($PY -c 'import sys; print(".".join(map(str,sys.version_info[:3])))')" = 3.8.20
test ! -e "$OUT"
test -f "$A2/parent/receipt.json"
test -f "$ROOT/data/eval_configs/episode_configs_rl_test.json"
test -d "$TEST_WORLDS"
for ckpt in "$PARENT" "$CONTROL" "$TREATMENT"; do
  test -f "$ckpt/state.pt" && test -f "$ckpt/header.json" && test -f "$ckpt/SHA256.json"
done
test "$(sha256sum scripts/evaluate_dppo_uav_formal_endpoint.py | cut -d' ' -f1)" = 13455ae9d910d8426a91e5c5a55da16790430aa2f0f7fd0470c2dbfd672b5f38
LOG=$PROGRAM/results/routea_test12_independent_run02_execution
mkdir -p "$LOG"
test "$(sha256sum scripts/run_routea_xk_evaluation.py | cut -d' ' -f1)" = 15bca3863d426f881236c2dcca76bcab7cc5fa6ed3d9011e27ebf8f05d3d39e7
sha256sum scripts/run_routea_xk_evaluation.py > "$LOG/delivery_sha256.txt"
"$PY" --version > "$LOG/python_version.txt" 2>&1
set +e
"$PY" scripts/run_routea_xk_evaluation.py test12 --repo "$ROOT" \
  --parent-receipt "$A2/parent/receipt.json" --u128 "$PARENT" \
  --control "$CONTROL" --treatment "$TREATMENT" --worlds "$TEST_WORLDS" \
  --output "$OUT" > "$LOG/test12.log" 2>&1
RC=$?
set -e
printf '%s\n' "$RC" > "$LOG/test12.rc"
test "$RC" -eq 0
cp -a "$LOG" "$OUT/execution/controller"
"$PY" - "$OUT" <<'PY'
import collections,json,pathlib,sys
p=pathlib.Path(sys.argv[1]); b=json.loads((p/'bank/bank.json').read_text())
c=json.loads((p/'completion.json').read_text())
assert len(b['worlds'])==12 and len(b['tasks'])==48 and len(b['episodes'])==240
assert b['split']=='rl_test' and c['episodes']==960
expected={e['key'] for e in b['episodes']}
for name in ['bc','u128','C','T']:
    job=p/'full'/name; s=json.loads((job/'summary.json').read_text())
    provenance=json.loads((job/'policy_provenance.json').read_text())
    assert s['episodes']==240 and len(s['records'])==240
    assert provenance['actual_use_base_policy']==(name=='bc')
    assert {f.stem for f in (job/'commits').glob('*.json')}==expected
    assert (p/'execution'/(name+'.rc')).read_text().strip()=='0'
    print(name,collections.Counter(r['done_reason'] for r in s['records']))
print('TEST12_COMPLETE',c['episodes'],c['bank_hash'])
PY
mkdir -p "$OUT/weight_identity"
for pair in "u128:$PARENT" "C:$CONTROL" "T:$TREATMENT"; do
  name=${pair%%:*}; ckpt=${pair#*:}
  mkdir -p "$OUT/weight_identity/$name"
  for file in header.json SHA256.json COMMITTED.json; do
    if test -f "$ckpt/$file"; then cp -- "$ckpt/$file" "$OUT/weight_identity/$name/$file"; fi
  done
  sha256sum "$ckpt/state.pt" > "$OUT/weight_identity/$name/state_sha256.txt"
done
cd "$OUT"
find . -type f ! -name return_sha256.txt ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum > return_sha256.txt
sha256sum -c return_sha256.txt > "$LOG/return_verify.log"
