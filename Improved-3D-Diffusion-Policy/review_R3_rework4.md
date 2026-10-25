# Independent Review — R3 rework #4 (external-review #4 P1 closure)

- Reviewer: fresh independent code-reviewer agent (zero prior project context; all claims re-verified from code, git, and command output)
- Date: 2026-08-13
- Repo: /home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy (branch a-official-core-rework)
- HEAD under review: 8cfee02 (tree 8202f468e0e85645539afd3fe3746e512e8b3584)
- Chain: 4b947a0 (RED test) → 1f0e5bb (fix) → 8cfee02 (docs interim status); prior baseline 6f2028d..3054eee + docs a85bc8e
- Python: /home/cjj/miniconda3/envs/idp3/bin/python3 (torch 2.4.1+cu121), Python 3.8.20, pytest 8.3.5

## Verdict summary

The external-review #4 P1 (live LoRA parameter-tensor-count gap) is **fixed and proven**: the three-way equality `len(lora_state) == saved n_lora_param_tensors == len(live lora_params)` is now enforced before any shape walk and before any state load; the adversarial probe now reports CheckpointError in both directions with zero state loads; all checkpoint batteries and both archived evidence logs are green and hash-authentic; the rework #4 commits themselves touched only the test file, checkpoint.py, and docs.

One integrity finding remains (P1, documentation-only, inherited from the rework #3 chain within the reviewed range): a frozen-file (kinematic_uav_env_v2.py) modification mandated by RULING-7 P0-2 was made without the project's own freeze-lift hash accounting convention, and a stale "frozen 5 zero-diff" claim in ENTRYPOINT_CLOSURE_REPORT.md was never reconciled.

## Findings

### P1-1 [FACT] Frozen-set accounting gap: kinematic_uav_env_v2.py modified in range without freeze-lift accounting

- The frozen set (5 blobs, per ENTRYPOINT_CLOSURE_REPORT.md §P8-e, anchored at git freeze commit 85ac9f0) is: `diffusion_policy_3d/env/kinematic_uav_env_v2.py` (baseline sha256 d1168142…), `diffusion_policy_3d/env/collision_geom.py`, `diffusion_policy_3d/model/common/synth_pointcloud.py`, `diffusion_policy_3d/policy/phase1a_policy.py`, `scripts/train_ddpm_ppo_meanlogp_3_e040.py`. Note: the task brief's parenthetical ("the other 4 E040-prefixed legacy scripts") misdescribes this set — there are no 5 E040-prefixed files in the tree (`git ls-tree -r --name-only HEAD | grep -i e040` yields 4, of which only the train script is frozen); the 4 non-script members are env/model/policy files.
- `git diff 42d652d..8cfee02` shows TWO of the five frozen blobs changed:
  - `scripts/train_ddpm_ppo_meanlogp_3_e040.py` (8 insertions, commit c05d131, 2026-08-10 R3.3): **documented** freeze-lift with pre/post sha256 accounting (7868d1be… → c548efc2…, ENTRYPOINT_CLOSURE_REPORT.md §R3.3). Verified: current file sha256 == c548efc2b21bd2740f067585c1d526eb25591fc461035e614fba149b12dcdab4 (exact match). Accounted exception — acceptable.
  - `diffusion_policy_3d/env/kinematic_uav_env_v2.py` (7+/4-, commit 6f2028d, 2026-08-11 rework #3): the RULING-7 P0-2 fix (removal of `hash(str(world_id))` from the episode-seed payload). This change is **mandated** by the binding ruling (RL_REBUILD_TRACEABILITY.md:1178-1183 P0-2 lists kinematic_uav_env_v2.py:238/266 as defect sites) and is listed in the rework #3 plan's allowed-modification set (`/home/cjj/AUV/plan/task_plan_r3_rework3_execution_20260813.md`), yet the same plan's forbidden list still names "E040 冻结 5 文件" — an internal contradiction. No pre/post sha256 accounting was recorded anywhere (grep for the baseline hash d1168142… finds only the P8-e listing itself); current sha256 is 5b7b16516da460776c9c5340e52a2f96a33ffaa0fe10532ba9e2dd0ae8625658.
  - The P8 evidence claim in ENTRYPOINT_CLOSURE_REPORT.md ("git diff 85ac9f0 empty for all 5 frozen files", appended 2026-08-07) is now stale and unreconciled.
- Impact: documentation/audit-trail integrity only. The functional change is correct, ruling-mandated, and covered by the cross-process RNG test and the GPU gate. It does not affect the rework #4 fix.
- Fix: append an R3-rework evidence note to ENTRYPOINT_CLOSURE_REPORT.md recording the env_v2 lift: baseline d1168142… → 5b7b1651…, mandate RULING-7 P0-2, mirroring the E040-script precedent (line ~235-241); correct the stale "empty diff for all 5" sentence; resolve the rework #3 plan's allowed/forbidden contradiction.

### P2-1 [FACT] Task brief composition error for the frozen set
The brief said "ZERO changes to the 5 E040 frozen files (scripts/train_ddpm_ppo_meanlogp_3_e040.py + the other 4 E040-prefixed legacy scripts)". The authoritative frozen-5 list (above) does not match; future checks should cite ENTRYPOINT_CLOSURE_REPORT.md §P8-e + anchor 85ac9f0. Verified via `git ls-tree -r --name-only 85ac9f0 | grep -i e040` (4 files) and the report listing.

### P2-2 [FACT] Banner reference to plan file not resolvable in-repo
review_R3_rework3.md SUPERSEDED banner cites `plan/task_plan_r3_rework4_execution_20260813.md`; that path does not exist in the repo (it lives at /home/cjj/AUV/plan/, outside the git tree). Minor; the review body is unaffected.

No P0 findings. No other P1 findings.

## Required verifications — exact commands and output

### 1. Code-level (checkpoint.py / train_loop.py) — ALL PASS [FACT]

Read `diffusion_policy_3d/rl/dppo_uav/checkpoint.py` `validate_checkpoint_lora_usable` (lines 390-506):
- (a) `if len(state_keys) != declared_tensors: raise CheckpointError` at line 467 (declared_tensors = saved `n_lora_param_tensors`, None-refused at 461-466; key-naming contract at 455). PASS
- (b) `if live_params is None: raise CheckpointError` at lines 473-480. PASS
- (c) `live_tensor_count = len(live_params); if live_tensor_count != declared_tensors: raise CheckpointError` at lines 489-497 — located BEFORE the per-tensor shape walk (lines 498-506). PASS
- (d) Ordering in `restore_training_state` (train_loop.py:393-531): `validate_resume_identity` (451) → `validate_lora_identity` (452) → empty-policy gate (457) → `validate_checkpoint_lora_usable` (462) → ... → `policy.load_state_dict` (468) → `restore_lora_state` (470). The live-count check therefore runs before any state load. PASS
- (e) `grep -rn "2 \* n_adaptors\|2\*n_adaptors\|len(lora_params).*2\|n_params\s*=\s*2" diffusion_policy_3d/rl/dppo_uav/*.py` → 0 matches (comments at 443-444 and 487-488 describe 2n as a production fact, not a restore formula; the count logic uses only len(state_keys)/declared_tensors/len(live_params)). PASS
- Save-side symmetry: `build_lora_identity` hard-fails when `len(lora_state) != n_lora_param_tensors` (checkpoint.py:359-364). PASS

### 2. Independent adversarial reproduction — PASS [FACT]

`/home/cjj/miniconda3/envs/idp3/bin/python3 temp/r3_rework4_probe.py` output (exact):
```
=== A) saved=4 / live=2, shapes match -> validator ===
RESULT: CheckpointError raised: live n_lora_param_tensors 2 != checkpoint declared 4 — the live wiring exposes a different parameter
=== B) saved=2 / live=4, shapes match -> validator ===
RESULT: CheckpointError: live n_lora_param_tensors 4 != checkpoint declared 2 — the live wiring exposes a different parameter
=== C) full restore path, saved=4 / live=2, state-load spy ===
RESULT: CheckpointError before load: live n_lora_param_tensors 2 != checkpoint declared 4 — the live wiring exposes a different parameter
STATE_LOAD_CALLS_BEFORE_FAILURE: []
```
Probe A: BAD_LIVE_PARAM_COUNT_ACCEPTED eliminated. Probe B: KeyError (previous behavior) replaced by CheckpointError. Probe C: failure before any load (validation-first atomic restore contract upheld). Probe temp file r3_rework4_probe_ckpt.pt was removed by the probe itself.

`/home/cjj/miniconda3/envs/idp3/bin/python3 -m pytest tests/dppo_parity/test_r3_rework4_negative.py -v` → **4 passed in 4.03s** (test_validate_lora_usable_rejects_live_count_mismatch[4-2], [2-4]; test_restore_rejects_live_count_mismatch_before_any_load[4-2], [2-4]).

### 3. Targeted checkpoint battery — PASS [FACT]

`/home/cjj/miniconda3/envs/idp3/bin/python3 -m pytest -q tests/dppo_parity/test_r3_rework_negative.py tests/dppo_parity/test_r3_rework2_negative.py tests/dppo_parity/test_r3_rework3_negative.py tests/dppo_parity/test_r3_rework3_positive.py tests/dppo_parity/test_checkpoint.py` → **94 passed in 10.47s** (matches the expected 94).

### 4. Evidence integrity — PASS [FACT]

`sha256sum` of the three logs:
- cpu_regression_r3_rework4_20260813.log → `6a57a1fe04a445af89bba8616d32ee02904f8e46db3cda1d003a0d445e060207` — exact match. Session header: `HEAD=8cfee02861839ab00ff1472a3b33ceed2b90c216 tree=8202f468e0e85645539afd3fe3746e512e8b3584`. Tail: `536 passed, 9 deselected in 171.04s (0:02:51)` and `PYTEST_EXIT=0`. Exact match.
- gpu_gate_rework4_20260813.log → `515f7bbe9346418892d44830dc0935865b0d8e0415e9d59a3753c00d9fc98a64` — exact match. Header HEAD=8cfee02…; both gate tests PASSED (`test_gpu_fresh_process_resume_chain PASSED`, `test_gpu_canonical_paired_eval_real_artifact PASSED`); tail `2 passed in 4380.17s (1:13:00)` and `PYTEST_EXIT=0`. Exact match.
- gpu_gate_rework3_20260813_run2.log (prior evidence) → `fdff7d9e4114da25fe375a58eb2cc5c6f1546364b35e4d911fa0cb76dd3f5f59`, tail `2 passed in 4405.12s (1:13:25)` — retained history, not re-run. Exact match.

### 5. Test-coverage claim — PASS [FACT]

Read tests/dppo_parity/test_r3_rework4_negative.py: `@pytest.mark.parametrize("n_saved, n_live", [(4, 2), (2, 4)])` on BOTH `test_validate_lora_usable_rejects_live_count_mismatch` (line 57) and `test_restore_rejects_live_count_mismatch_before_any_load` (line 65). The restore tests assert `calls == []` (line 84) with the message "no state load may run on a live-count mismatch". Both directions × both paths = 4 tests, all green (item 2).

### 6. RED evidence — PARTIAL PASS [FACT]

- `git show 4b947a0 --stat` → `tests/dppo_parity/test_r3_rework4_negative.py | 87 +++++++++++++++++++++++++++` — test-file-only. PASS
- `git show 1f0e5bb --stat` → `diffusion_policy_3d/rl/dppo_uav/checkpoint.py | 23 ++++++++++++++++++++++-` — exactly 22 insertions, 1 deletion, single file, allowed scope. PASS (exact match)
- `git diff 42d652d..8cfee02 --stat -- third_party/dppo_official_cc7234a` → empty, exit 0. Vendor (401 files) untouched. PASS
- E040 frozen 5 in range `42d652d..8cfee02`: two changed blobs — the E040 script (accounted freeze-lift, current sha256 matches recorded snapshot c548efc2…) and kinematic_uav_env_v2.py (unaccounted, see P1-1). The other three (collision_geom, synth_pointcloud, phase1a_policy) zero-diff vs 85ac9f0. The rework #4 commits themselves (4b947a0/1f0e5bb/8cfee02) touched no frozen or vendor file (8cfee02 stat: RL_REBUILD_TRACEABILITY.md + review_R3_rework3.md only). PARTIAL — P1-1.

### 7. Annotation integrity — PASS [FACT]

`git diff 3054eee..8cfee02 -- review_R3_rework3.md` shows the file as new (it was first committed in a85bc8e, after 3054eee), so the correct annotation comparison is `git diff a85bc8e..8cfee02 -- review_R3_rework3.md`: 19 lines, exactly (i) SUPERSEDED banner added at the top and (ii) the verdict line struck (`Verdict: ~~PASS~~ … SUPERSEDED by external-review #4`). Body content otherwise untouched (only diff hunks are the two banner/verdict insertions). PASS

## Evidence / verification performed

Commands executed (all from repo root, cd /home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy):
1. git rev-parse HEAD / git log / git status (HEAD=8cfee02, untracked temp/r3_rework4_probe.py)
2. Read checkpoint.py:390-506 and checkpoint.py:300-366 (validator, build side); train_loop.py:393-531 (restore ordering)
3. grep 2*n_adaptors family + n_lora_param_tensors usage in dppo_uav/*.py (0 production derivations)
4. python3 temp/r3_rework4_probe.py (output above)
5. pytest -v tests/dppo_parity/test_r3_rework4_negative.py (4 passed)
6. pytest -q <5-file battery> (94 passed)
7. sha256sum of 3 regression logs + head/tail reads
8. Read test_r3_rework4_negative.py (both directions, spy assert)
9. git show 4b947a0/1f0e5bb/8cfee02 --stat; git diff 42d652d..8cfee02 --stat (full range, vendor, scripts/, E040 files, env/)
10. git hash-object/sha256sum of E040 + env_v2 files vs anchor 85ac9f0 and recorded baselines; ENTRYPOINT_CLOSURE_REPORT.md §§1/P8/R3.3/P8-e read
11. git diff a85bc8e..8cfee02 -- review_R3_rework3.md; RL_REBUILD_TRACEABILITY.md:403/879/1171-1183; rework #3/#4 plan files under /home/cjj/AUV/plan/

## What I did not verify

- The GPU gate was not re-run (73 min); the archived log + sha256 + headers/tails are the evidence (item 4). Log markers (2 tests PASSED, PYTEST_EXIT=0) confirmed.
- No code or doc was modified except writing this review file.

## Conditions

1. [P1-1] Record the kinematic_uav_env_v2.py freeze-lift with pre/post sha256 accounting in ENTRYPOINT_CLOSURE_REPORT.md (d1168142… → 5b7b1651…, mandate RULING-7 P0-2), and correct the stale "git diff 85ac9f0 empty for all 5 frozen files" sentence (P8 evidence section).
2. [P1-1] Resolve the rework #3 plan's internal contradiction (kinematic_uav_env_v2.py listed in both the allowed and the forbidden sets).

## Verdict

PASS-WITH-CONDITIONS

## Addendum (post-check observation)

### P2-3 [FACT] Uncommitted working-tree change at review time
`git status` shows `M ENTRYPOINT_CLOSURE_REPORT.md` (14 insertions, appended "R3 rework #3 evidence" + "R3 rework #4 evidence" sections) in the working tree at HEAD=8cfee02. This change is NOT part of any reviewed commit (it was not created by this reviewer — only review_R3_rework4.md was written here) and contains a placeholder line "[backfilled after review completion]". It is consistent with the plan's post-review RTM/evidence backfill step; the closure workflow should commit it after the verdict. Its RED/fix/suite/GPU claims match my independent measurements (battery 94 + 4 new negatives = 98; CPU 536/9; GPU 2 passed 4380.17s; sha256s match item 4).
