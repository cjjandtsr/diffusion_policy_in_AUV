#!/bin/bash
# Route A A1 val20 与同状态机制验证执行指导 v1.2 —— 四段命令逐字转录，同一 bash 会话按序执行
# 段失败即整体中止（set -euo pipefail + 各段 test rc）
set -euo pipefail
trap 'echo "=== MASTER_EXIT rc=$? $(date -Is) ==="' EXIT

ROOT=/home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy
PY=/home/cjj/miniconda3/envs/idp3/bin/python
PROGRAM=/home/cjj/AUV/program
OUT=$PROGRAM/results/routea_a1_mechanism_val20_run03
WEIGHTS=$ROOT/data/outputs/dppo_uav_rebuild
W1280=$WEIGHTS/routea_common_b1280_seed202609203_v1
W2560=$WEIGHTS/routea_common_b2560_seed202609203_v1
CFG2560=$ROOT/diffusion_policy_3d/rl/dppo_uav_rebuild/config/train_e5_dppo_common_b2560_seed202609203_v1.yaml
VAL_WORLDS=/home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds/val
RECOVERY_BANK=$ROOT/scripts/routea_paired_support/a1_mechanism_bank_v1.json

echo "=== SEG1_PREP_START $(date -Is) ==="
cd "$ROOT"
test "$($PY -c 'import sys; print(".".join(map(str,sys.version_info[:3])))')" = 3.8.20
test -f "$CFG2560" && test -f "$RECOVERY_BANK"
test -d "$VAL_WORLDS"
REF=
for candidate in \
  "$PROGRAM/routea_outputs_migration_v2_1_run02/migrated_outputs/diagnostics/routea_bc_a0_r1_paired_dev_20261010_py38fix_run02" \
  "$PROGRAM/results/routea_paired_evaluation/routea_bc_a0_r1_paired_dev_20261010_py38fix_run02" \
  "$ROOT/data/outputs/diagnostics/routea_bc_a0_r1_paired_dev_20261010_py38fix_run02"; do
  if test -f "$candidate/bank/bank.json"; then REF="$candidate"; break; fi
done
test -n "$REF"
mkdir -p "$OUT/execution"
"$PY" --version > "$OUT/execution/python_version.txt" 2>&1
sha256sum scripts/evaluate_dppo_uav_formal_endpoint.py \
  scripts/run_routea_val20_endpoint.py \
  scripts/routea_paired_support/recovery.py \
  scripts/routea_paired_support/a1_mechanism_bank_v1.json \
  tests/routea_paired/test_recovery.py \
  tests/routea_paired/test_val_only_protocol.py > "$OUT/execution/delivery_sha256.txt"
test "$(sha256sum scripts/evaluate_dppo_uav_formal_endpoint.py | cut -d' ' -f1)" = 13455ae9d910d8426a91e5c5a55da16790430aa2f0f7fd0470c2dbfd672b5f38
test "$(sha256sum scripts/run_routea_val20_endpoint.py | cut -d' ' -f1)" = 2f2ade59e115a6b8366814dc6d1898a3d4f294939e612db20126445a9549b680
test "$(sha256sum scripts/routea_paired_support/recovery.py | cut -d' ' -f1)" = cffcbead33c715c6965de2e1abe832bbfa3be655b8a8cec495192265a541057e
test "$(sha256sum scripts/routea_paired_support/a1_mechanism_bank_v1.json | cut -d' ' -f1)" = 9527f70ba0ad695f4693dac6aec758a0022882f237546240b898b8fea00666f0
test "$(sha256sum tests/routea_paired/test_recovery.py | cut -d' ' -f1)" = 76acf9684983f9b86e62677b5e531bfcd9691d4ff2f01749ff3e1721ca7a889b
test "$(sha256sum tests/routea_paired/test_val_only_protocol.py | cut -d' ' -f1)" = 1dbfe9086a096ce9929528b4c0c3f05732f4c9f8eceeed04058417213a823723
"$PY" - "$ROOT/data/eval_configs/episode_configs_rl_val.json" "$VAL_WORLDS" <<'PY' > "$OUT/execution/val_assets.txt"
import json, pathlib, sys
bank=json.load(open(sys.argv[1],encoding='utf-8'))
root=pathlib.Path(sys.argv[2])
assert len(bank)==20 and all(len(v)==4 for v in bank.values())
paths=[root/(uid+'.world.obstacles.json') for uid in bank]
assert all(p.is_file() for p in paths), [str(p) for p in paths if not p.is_file()]
print('VAL_ASSETS_PASS',len(paths),root)
PY
for u in 0 1 8 16 24 32; do test -f "$W1280/checkpoint_update_$(printf '%06d' "$u")/state.pt"; done
for u in 0 8 16 24 32 128 256; do test -f "$W2560/checkpoint_update_$(printf '%06d' "$u")/state.pt"; done
"$PY" -c 'import json,sys,pathlib; p=pathlib.Path(sys.argv[1]); b=json.loads(p.read_text(encoding="utf-8")); assert len(b["states"])==18 and b["draws"]==20 and b["continuation_draws"]==[0,1,2]; print("A1_BANK",b["content_hash"])' "$RECOVERY_BANK"
set +e
"$PY" -m pytest tests/routea_paired/test_recovery.py tests/routea_paired/test_val_only_protocol.py -q \
  > "$OUT/execution/pytest.log" 2>&1
RC=$?
set -e
printf '%s\n' "$RC" > "$OUT/execution/pytest.rc"
test "$RC" -eq 0
echo "REF=$REF" > "$OUT/execution/ref_location.txt"
echo "=== SEG1_PREP_DONE $(date -Is) ==="

echo "=== SEG2_VAL20_START $(date -Is) ==="
cd "$ROOT"
set +e
"$PY" scripts/run_routea_val20_endpoint.py \
  --config "$CFG2560" --rl-checkpoint "$W2560/checkpoint_update_000128" \
  --eval-config-dir "$ROOT/data/eval_configs" --benchmark routea-val \
  --worlds-val "$VAL_WORLDS" --output-dir "$OUT/val_u128" --expected-update 128 \
  > "$OUT/execution/val_u128.log" 2>&1
RC=$?
set -e
printf '%s\n' "$RC" > "$OUT/execution/val_u128.rc"
test "$RC" -eq 0
set +e
"$PY" scripts/run_routea_val20_endpoint.py \
  --config "$CFG2560" --rl-checkpoint "$W2560/checkpoint_update_000256" \
  --eval-config-dir "$ROOT/data/eval_configs" --benchmark routea-val \
  --worlds-val "$VAL_WORLDS" --output-dir "$OUT/val_u256" --expected-update 256 \
  > "$OUT/execution/val_u256.log" 2>&1
RC=$?
set -e
printf '%s\n' "$RC" > "$OUT/execution/val_u256.rc"
test "$RC" -eq 0
set +e
"$PY" - "$OUT/val_u128" "$OUT/val_u256" > "$OUT/execution/val_pair_check.log" 2>&1 <<'PY'
import collections, json, pathlib, sys
results=[]
for folder in map(pathlib.Path,sys.argv[1:]):
    provenance=json.load(open(folder/'evaluation_provenance.json',encoding='utf-8'))
    boundary=json.load(open(folder/'arrival_boundary_audit.json',encoding='utf-8'))
    assert provenance['benchmark']=='routea-val'
    assert provenance['world_count']==20 and provenance['episodes_per_policy']==80
    assert boundary['radius_m']==0.5 and boundary['minimum_rounding_tolerance_m']==1e-6
    assert boundary['accepted_boundary_count']==len(boundary['accepted_boundary_events'])
    assert all(0 < e['excess_m'] <= e['rounding_allowance_m']
               for e in boundary['accepted_boundary_events'])
    rows=[json.loads(line) for line in (folder/'paired_episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    counts=collections.Counter(row['policy'] for row in rows)
    assert len(rows)==160 and counts=={'bc':80,'rl':80}
    assert all(row['split']=='rl_val' for row in rows)
    results.append({kind:{row['config_hash']:row for row in rows if row['policy']==kind}
                    for kind in ('bc','rl')})
assert set(results[0]['rl'])==set(results[1]['rl'])==set(results[0]['bc'])
bc_diff=[key for key in results[0]['bc'] if results[0]['bc'][key]!=results[1]['bc'][key]]
bc_outcome_diff=[key for key in bc_diff if results[0]['bc'][key]['done_reason']!=results[1]['bc'][key]['done_reason']]
print('BC_REPEAT_DIFFERENT_RECORDS',len(bc_diff),'OUTCOME_DIFFERENCES',len(bc_outcome_diff))
for key in bc_outcome_diff:
    a,b=results[0]['bc'][key],results[1]['bc'][key]
    print('BC_OUTCOME_DIFFERENCE',key,a['done_reason'],b['done_reason'],
          a['final_distance_3d'],b['final_distance_3d'])
for label,rows in zip(('u128','u256'),results):
    print(label,'BC',dict(collections.Counter(x['done_reason'] for x in rows['bc'].values())),
          'RL',dict(collections.Counter(x['done_reason'] for x in rows['rl'].values())))
print('VAL_PAIR_PASS',len(results[0]['bc']))
PY
RC=$?
set -e
printf '%s\n' "$RC" > "$OUT/execution/val_pair_check.rc"
test "$RC" -eq 0
echo "=== SEG2_VAL20_DONE $(date -Is) ==="

echo "=== SEG3_SAME_STATE_MAIN_START $(date -Is) ==="
cd "$ROOT"
set +e
"$PY" scripts/run_routea_paired_comparison.py \
  --checkpoint "$W2560/checkpoint_update_000128" \
  --checkpoint "$W2560/checkpoint_update_000256" \
  --reference "$REF" \
  --recovery-bank "$RECOVERY_BANK" --recovery-only \
  --out "$OUT/same_state_main" \
  > "$OUT/execution/same_state_main.log" 2>&1
RC=$?
set -e
printf '%s\n' "$RC" > "$OUT/execution/same_state_main.rc"
test "$RC" -eq 0
echo "=== SEG3_SAME_STATE_MAIN_DONE $(date -Is) ==="

echo "=== SEG4_SAME_STATE_EARLY_START $(date -Is) ==="
cd "$ROOT"
ARGS=()
for u in 0 1 8 16 24 32; do ARGS+=(--checkpoint "$W1280/checkpoint_update_$(printf '%06d' "$u")"); done
for u in 0 8 16 24 32; do ARGS+=(--checkpoint "$W2560/checkpoint_update_$(printf '%06d' "$u")"); done
set +e
"$PY" scripts/run_routea_paired_comparison.py "${ARGS[@]}" \
  --reference "$REF" \
  --recovery-bank "$RECOVERY_BANK" --recovery-only \
  --out "$OUT/same_state_early" \
  > "$OUT/execution/same_state_early.log" 2>&1
RC=$?
set -e
printf '%s\n' "$RC" > "$OUT/execution/same_state_early.rc"
test "$RC" -eq 0
echo "=== SEG4_SAME_STATE_EARLY_DONE $(date -Is) ==="
echo "=== MASTER_ALL_DONE $(date -Is) ==="
