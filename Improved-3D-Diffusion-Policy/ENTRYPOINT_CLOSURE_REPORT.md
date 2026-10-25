# Entry-Point Closure Report — A-Official-Core, repository-wide (P7 rewrite)

Date: 2026-08-07. Design: RouteA_RL_Training_System_A_OfficialCore_Design_v1_0 §8.2
(SHA256 `4ede81feecbde9a1580110b431e36ff8bcd01d7f2303320a5e2284bff4049407`, 1622 lines).

## 0. Status

**This is the repository-wide entry inventory, not a closure claim.** T5
COMPLETE/PASS remains revoked (ASSUMPTION-9, RL_REBUILD_TRACEABILITY.md:137).
The previous version of this file was revoked because it scoped the audit to
the `dppo_uav` package and mislabeled TaskSpecRegistry / RoleSampler /
AdmissionChecker as closed (their producers are tests/mocks only — train_loop
has 0 references). This rewrite covers every entry category the adjudication
enumerated, classifies each entry, and marks what is still deferred:

- **canonical** — the A-Official-Core path (dppo_uav), formal train/eval only.
- **guarded** — legacy E-series training scripts: formal mode rejected.
- **historical** — frozen/deprecated legacy pipeline scripts, unmodified.
- **T2-ext (executed, not deferred)** — the real env wiring items (B3
  adjudication) were the P8 work package; completed 2026-08-07 and closed
  by the R2 gate series (see the closure note below and §10).
- **frozen** — E040 baseline files, byte-untouched (0-modification constraint).
  R3.3 amendment (2026-08-10): the 0-modification constraint was lifted for the
  E040 script only, by design, with hash accounting (pre/post sha256 in §1);
  the remaining 4 frozen files stay byte-untouched.

Real-env wiring and A* (T2 extension work package) were P8 of the rework:
executed 2026-08-07 (adapter `5bdae7e`, integration `4418ad4`), verified by
the R2 gate series (G3 v8 11/11 gates, `ROLE_METRICS_EVIDENCE_v6`,
`LIFECYCLE_EVIDENCE_v3` 21 tests) and closed with R2 = PASS (final, rework
#6). History: `task_plan_t5_rework.md` and RTM ASSUMPTION-9 (B3).

## 1. 全部 train_ddpm_ppo*.py（legacy E-series training entry points）

All 7 scripts on disk are classified `HISTORICAL_FORMAL_ENTRYPOINTS`
(`diffusion_policy_3d/rl/dppo_uav/entrypoint_guard.py:23-31`) and carry the
`HISTORICAL_INVALID_FOR_FORMAL_TRAINING` symbol (entrypoint_guard.py:18-20):

| Script | Status | Protection |
|--------|--------|------------|
| `scripts/train_ddpm_ppo.py` | historical, guarded | wired P7: `guard_formal_mode(…, args.mode)` after parse_args |
| `scripts/train_ddpm_ppo_meanlogp.py` | historical, guarded | wired P7 |
| `scripts/train_ddpm_ppo_meanlogp_1.py` | historical, guarded | wired P7 |
| `scripts/train_ddpm_ppo_meanlogp_2.py` | historical, guarded | wired P7 |
| `scripts/train_ddpm_ppo_meanlogp_3.py` | historical, guarded | wired P7 |
| `scripts/train_ddpm_ppo_meanlogp_3_e039.py` | historical, guarded | wired P7 |
| `scripts/train_ddpm_ppo_meanlogp_3_e040.py` | historical, guarded | wired R3.3: pre-modification (historical baseline) sha256 `7868d1be8ec9a957b3f9e352bd0185dc3634875bf06a1dbeab04f6e44f1768cf` → guarded snapshot sha256 `c548efc2b21bd2740f067585c1d526eb25591fc461035e614fba149b12dcdab4` |

Wiring proof (tests): `test_legacy_script_rejects_formal_mode_subprocess`
(parametrized over all 7 scripts — `--mode formal --dry_run` exits non-zero
with the HISTORICAL message) and `test_all_legacy_scripts_wire_the_guard`
(every legacy script carries the call, the E040 script included since R3.3).

## 2. 全部 paired/single eval

**Canonical (A-Official-Core):**
- `run_paired_eval` (`train_loop.py`) — BC vs RL on shared seeds, one
  `EpisodePairedRecord` per seed; runs its **own point-cloud validity gate**
  (P7): an invalid reset/step frame voids that paired seed and is counted;
  `run_e2e_batch` hard-fails on any voided seed (never persists a
  PPO==BC-polluted eval).
- Paired-eval JSON persistence (P7; R3.4 filename + atomic-write):
  `paired_eval_envstep_<9d>_ppoupd_<6d>_<ckpt12>.json` (cumulative
  counters) written next to the checkpoint — full §4.17 artifact dict,
  tmp + os.replace atomic write (existing file with a different hash →
  RuntimeError, no overwrite; identical → no-op), round-trip tested
  (`test_eval_json_persisted_roundtrip`).
- `build_runtime_eval_artifact` / `validate_eval_artifact` (§4.17) — runtime
  contract hash at the call site (ASSUMPTION-8 item 6); A2
  use_obstacle_projection=False hard-fail; A5 history_indices required.

**Legacy pipeline eval scripts (historical — NOT part of A-Official-Core,**
**unmodified, neither fed by nor feeding the canonical artifact):**
- active legacy: `eval_layera.py` (layerA paired), `eval_ddpm_paired.py`
  (rl_test paired, loads LoRA), `eval_ddpm_bc_baseline.py`, `eval_bc_baseline.py`.
- historical probes/utilities: `eval_paired.py`, `eval_v13_metrics.py`,
  `eval_v14_metrics.py`, `run_c2_ooc_expanded_eval.py`,
  `gen_eval_configs.py` (episode-config generator).

## 3. 全部 sampling/predict callers

- **Canonical sampling callers** of `sample_ddpm_with_logprob`
  (frozen `phase1a_policy.py`):
  - `sample_dppo_rollout` (actor_adapter) — training rollout (P6: B2
    five-field `chains_policy`/`old_logprobs` region window);
  - `_eval_episode` (train_loop) — eval rollout (P7 gate);
  - `make_get_new_logprobs` — likelihood evaluation only (no sampling; A4
    single logprob std).
- **Historical sampling callers**: the 7 legacy scripts (guarded) and
  `scripts/probe_e040_diag.py` (diagnostic probe).
- **`predict()`**: defined by `phase1a_policy.py`, `anchor_head.py`,
  `base_policy.py`, `diffusion_pointcloud_policy.py`, `phase1a_policy_fm*.py` —
  **no caller in the A-Official-Core** (the core samples via
  `sample_ddpm_with_logprob` only). `predict` is the BC-inference path of the
  legacy eval scripts; classified historical.

## 4. 全部 observation/action slicing

- **Single observation adapter**: `select_latest_observations`
  (`observation_adapter.py`) — sole canonical caller `train_loop._selected`
  (A5: returns `ObservationSelection` metadata; history_indices enters the
  result dict and the eval artifact). `PolicyObsEncoder` encodes the
  already-selected batch (no internal slicing).
- **Action slicing**: `split_trajectory` + `slice_executed_logprobs`
  (`phase1a_actor_adapter.py`) — sole canonical caller `sample_dppo_rollout`;
  executed action = policy action `[:exec_steps]` (ASSUMPTION-4).
- **Normalizer unnormalize**: train_loop training rollout + `_eval_episode`
  (gaussian mode per hard constraint).
- Legacy scripts carry their own oldest-3/8 slicing (附录 A row 1) — frozen
  historical, unmodified.

## 5. reward/normalization sources

- **Single source**: the env reward module (`kinematic_uav_env_v2.py`
  `_compute_reward` / `_compute_reward_v14` / `_compute_reward_v16`) is the
  ONLY normalization source. `RewardNormState(version, source="env",
  method="none", params)` (`reward_norm.py`, frozen dataclass) is the
  checkpoint carrier; the loaded state is never overwritten by a live one
  (附录 H anti-pattern closed; round-trip test
  `test_reward_norm_state_preserved_through_checkpoint_roundtrip`).
- Legacy scripts expose a `--reward_norm` CLI flag (the historical
  anti-pattern) — historical only, never in the canonical core.

## 6. projection sources

`use_obstacle_projection` read points:
- `policy_loader.py:109` (default False), `phase1a_policy.py:60` (default
  False), `train_fm_sde_arfm.py:102` (FM training script — separate path,
  historical).

A2 (ASSUMPTION-9) §7.4 closure in the canonical core:
- `contracts.py`: field + `assert not use_obstacle_projection` (True → build
  failure);
- `train_loop.run_e2e_batch`: policy-vs-contract mismatch hard-fail;
- eval artifact: field + `validate_eval_artifact` hard-fails on True;
- checkpoint/eval contract serialization carries False.
Projection is policy-external (KD-Tree repulsion, policy-external per 策略边
界判定); the formal core rejects it at every boundary.

## 7. E-series 状态

All E-series training entry points are classified historical and wired with
formal-mode rejection: 6 since P7, the E040 script since R3.3 (see §1).
`probe_e040_diag.py` is a diagnostic probe (historical). No E-series script
can run formal training (behavioral subprocess proof, §1). The E040 BC
baseline itself remains the frozen reference for BC comparisons.

## 8. 唯一 canonical formal train/eval 绝对路径

- **Formal train**: `diffusion_policy_3d/rl/dppo_uav/train_loop.py::run_e2e_batch`
  — the only entry that may run formal training; legacy entries are rejected
  by the guard (no second canonical path exists).
- **Formal eval**: `run_paired_eval` → `build_runtime_eval_artifact` →
  persisted `paired_eval_envstep_<9d>_ppoupd_<6d>_<ckpt12>.json`
  (R3.4 atomic-write, §2).
- Legacy active eval (`eval_layera.py` / `eval_ddpm_paired.py`) remains the
  legacy pipeline's evaluation surface, outside and unaffected by the
  A-Official-Core.

## 9. Evaluation 完整性（裁决清单逐项核验）

| 裁决项 | 状态 | 证据 |
|--------|------|------|
| paired-eval JSON 落盘 | ✅ P7 | `test_eval_json_persisted_roundtrip` |
| eval rollout 独立 PC validity gate | ✅ P7 | `test_run_paired_eval_gates_invalid_*` + run_e2e_batch hard-fail |
| capability metrics | ✅ R3.5 | `capability_metrics.py`（d_soft^H / onset / trap / side_flip / support cell + `OfflineDensity`）；offline density 缺失 hard-fail（eval 前置 RuntimeError / CLI exit 2）— commits `77e2fa2` + `7c9a9af` |
| history_indices 进 trace/eval | ✅ P5 | `test_eval_artifact_carries_history_indices` |
| obstacle-projection contract | ✅ P5 | §6 |
| dual std (sampling/logprob) | ✅ P5 | `test_dual_std_*` |
| role/sample counts | ✅ P5 | `test_role_stats_in_train_loop_result` |
| reward-norm 单一来源 | ✅ P5 | §5 |
| T5 shiyan evidence | ✅ existing log (20260806) kept as history — predates P7 and has no paired JSON; the new evidence standard is R3.4+ paired eval (frozen bank + episode traces + atomic-write artifact) | `program/shiyan/dppo_parity_t5_20260806.log` |

## 10. RTM / gates

- RTM rows with "PASS (…pending T7)" were reverted to OPEN/BLOCKED (P1) —
  OBS-RECENCY / PC-VALID / ACT-OFFSET / ROLE-PROP / ADMISSION. No logical
  PASS-on-pending remains (see RL_REBUILD_TRACEABILITY.md).
- R3.6 closure (2026-08-11): OBS-RECENCY and ACT-OFFSET closed PASS
  (sentinel tests + G5 real-env assertions, see RL_REBUILD_TRACEABILITY.md
  rows); PC-VALID / ROLE-PROP / ADMISSION already PASS via the R2 gate
  series (evidence v6/_v4).
- G4 stays BLOCKED with the adjudicated semantics: Gazebo—kinematics
  equivalence is [FACT]; BLOCKED means only "the frozen fixture is not
  found" — never "not yet proven". Fixture search continues; a replacement
  non-regression fixture requires a separate user decision.
- Gate order (binding): G3 (real 10k calibration) → G4 (equivalence
  non-regression) → G5 (real-env no-update) → G6 (real one-update + base
  freeze + live LoRA checkpoint + optimizer resume) → T6/G7 micro-train →
  T7/G8 multi-seed. **No micro-train and no formal training before the R4
  re-gates (G5/G6) pass** — MasterPlan v1.0 (2026-08-07) ruled G5 FAIL /
  G6 REOPEN (the pre-R0 "GREEN" closeouts were accepted under semantics
  that do not satisfy the design); R1 has re-closed the identity axis
  (`G5_ACTUAL_BATCH_IDENTITY_v2.json`), G5/G6 re-run at R4. G4 BLOCKED
  does not block G3/G5/G6 execution (ASSUMPTION-4: EQUIV-REG is a state,
  not a gate input) — this is not "phase skipping": every gate that runs
  does so on the production path with structured evidence.
- T2-ext (B3) executed, not deferred: `UAVVectorEnvAdapter.reset/step`
  real wiring, `TaskSpecRegistry` / `RoleSampler` / `AdmissionChecker`
  into train_loop, A* port (ASSUMPTION-5) — completed 2026-08-07 (P8:
  adapter `5bdae7e`, integration `4418ad4`) and closed by the R2 gate
  series (G3 v8 11/11, ROLE_METRICS_EVIDENCE_v6, LIFECYCLE_EVIDENCE_v3
  21 tests; R2 = PASS final after rework #6). The "integrated in T7"
  comments in environment_adapter.py were implementer-written, not design
  text, and were removed by the R3.0c comment cleanup (c681a1b).

## Evidence (2026-08-07)

- Full suite: **191 passed** (188 CPU + 3 GPU), `python -m pytest tests/ -q`.
- Vendor `third_party/dppo_official_cc7234a/`: 401/401 files verified
  (`sha256sum -c SHA256SUMS`).
- E040 frozen set: 5 files anchored by git freeze commit `85ac9f0`
  ("chore: freeze pre-A-Official-Core baseline"). Baseline sha256s:
  - `diffusion_policy_3d/env/kinematic_uav_env_v2.py`  `d1168142401ff26714cd20ebe6d72d868275ffe8841ca21e7054493d008384b3`
  - `diffusion_policy_3d/env/collision_geom.py`          `3673e1b27f01863e873449f1e2801f76faf4871090fe742bb737cc3379384652`
  - `diffusion_policy_3d/model/common/synth_pointcloud.py` `c5a164bb9b6c7b32213c8f0180ddfb2a5bc0302e2327790b8098256307fb6dab`
  - `diffusion_policy_3d/policy/phase1a_policy.py`       `3871b8debd92e30c06d9647989cf6d1ff78d94d216f40c058e03699cdf3a914a`
  - `scripts/train_ddpm_ppo_meanlogp_3_e040.py`          `7868d1be8ec9a957b3f9e352bd0185dc3634875bf06a1dbeab04f6e44f1768cf` (previously recorded in §1 table; re-verified)
- Freeze-lift accounting (2026-08-13, review_R3_rework4.md P1-1 reconciliation):
  of the 5 frozen blobs, 2 were subsequently modified under explicit
  sanctions — the earlier claim "`git diff 85ac9f0` is empty for each" is
  stale and superseded by this block:
  - `diffusion_policy_3d/env/kinematic_uav_env_v2.py`: freeze lifted by
    RULING-7 P0-2 (env RNG: remove `hash(str(world_id))`, blake2b over
    canonical JSON), executed in `6f2028d` (11 lines, 7+/4-). Pre (85ac9f0):
    `d1168142…` (baseline above) → post (HEAD 8cfee02):
    `5b7b16516da460776c9c5340e52a2f96a33ffaa0fe10532ba9e2dd0ae8625658`.
  - `scripts/train_ddpm_ppo_meanlogp_3_e040.py`: freeze lifted by the design
    R3.3 amendment (2026-08-10, §1 table): pre `7868d1be…` → guarded
    snapshot `c548efc2b21bd2740f067585c1d526eb25591fc461035e614fba149b12dcdab4`.
  - The other 3 blobs (collision_geom.py, synth_pointcloud.py,
    phase1a_policy.py) remain byte-identical to 85ac9f0 (`git diff
    85ac9f0 -- <file>` empty, verified 2026-08-13).
- New tests: `tests/dppo_parity/test_p7_eval_persistence.py` (5),
  `tests/dppo_parity/test_p5_contracts.py` (+2 wiring tests).
- Git: `a-official-core` branch; P6 `1cbe88c`, P7 (this report + wiring + eval
  persistence) head commit.

### P8 evidence (2026-08-07, appended)

- Full suite now **221 passed** (218 CPU + 3 GPU) — P8-a adapter `5bdae7e`,
  P8-b integration `4418ad4`.
- Frozen-set re-verification: E040 script sha256 matches §1 recorded baseline;
  `git diff --stat 85ac9f0` empty for all 5 frozen files; working tree clean;
  vendor still 401/401.
  **(2026-08-07 历史快照——此后经批准解除冻结：E040 脚本 2026-08-10 R3.3
  修正案、env_v2 2026-08-11 RULING-7 P0-2；两处 pre/post hash 记账见
  §Evidence "Freeze-lift accounting"。)**

### R3.3 evidence (2026-08-10, appended)

- **E040 freeze lift with hash accounting** (guide-sanctioned minimal guard):
  pre-modification sha256 `7868d1be8ec9a957b3f9e352bd0185dc3634875bf06a1dbeab04f6e44f1768cf`
  (the P7-era frozen baseline, still recorded in §1) → guarded snapshot
  sha256 `c548efc2b21bd2740f067585c1d526eb25591fc461035e614fba149b12dcdab4`.
  Guard = `guard_formal_mode("scripts/train_ddpm_ppo_meanlogp_3_e040.py",
  args.mode)` right after `parse_args`, mirroring the sibling scripts.
- The remaining 4 frozen baseline files stay byte-untouched:
  `git diff 85ac9f0` still empty for each (env/collision_geom/synth_pointcloud/
  phase1a_policy).
  **(2026-08-10 历史快照——env_v2 随后于 rework #3（`6f2028d`）经 RULING-7
  P0-2 批准解除冻结并记账（§Evidence "Freeze-lift accounting"）；当前剩余
  3 个冻结文件与 85ac9f0 byte-identical（2026-08-13 复核）。)**
- New subprocess proof: `tests/dppo_parity/test_canonical_entrypoints.py`
  parametrized over all 7 legacy scripts (`--mode formal --dry_run` must exit
  non-zero with the HISTORICAL message). RED confirmed first (E040 exited 0),
  then GREEN 8/8 after the guard.
- p5 wiring test re-scoped: `test_all_legacy_scripts_wire_the_guard` (was
  `test_all_nonfrozen_legacy_scripts_wire_the_guard`) — no freeze exception.
- Code-review round 2 on the canonical entries: Block (2 HIGH, 3 WARNING,
  3 LOW) → **PASS**, all findings fixed: H1 the eval bank's
  manifest_entry_hash is recomputed from the manifest itself at CLI level
  (a bank-vs-itself gate would be a tautology); H2 every bank world gated
  against the production pool at CLI level + AdmissionExhausted/ValueError
  wrap around the eval loop; W1 per-batch train timestamps + independent
  eval timestamp (timestamp is excluded from the config-hash layer); W2
  role/split enum validation in load_paired_bank + policy/value-state and
  exact_resolved_config access wrapped; W3 repo-root-relative path
  resolution (preflight and entries); L1 unique per-PID dirty-tree marker
  + atexit cleanup; L2 exact exit codes asserted (1 preflight/identity,
  2 CLI/bank); L3 new coverage: eval dirty-tree preflight, bank pool/hash/
  role rejections, identity mismatch (structurally valid fake checkpoint),
  positive preflight (both clean-tree tests skip on a dirty tree and go
  live post-commit). Closure pair 45 passed / 2 skipped; full CPU suite
  406 passed / 2 skipped / 7 deselected (GPU).

### R3.4 evidence (2026-08-10, appended)

- **EVAL-STD** (commit `e30d835`): SpyPolicy RED proved the eval path
  silently used the policy default sampling std (0.01); `_eval_episode`
  now receives both floors (min_sampling/min_logprob) from LoopConfig —
  `test_eval_std_passthrough`, 2 tests RED→GREEN. The std is driven by the
  checkpoint's own exact_resolved_config.
- **Frozen paired bank v2**: `PairedBankEntry` (frozen, 14 fields) +
  `freeze_paired_bank` — episode_config_hash frozen by the deterministic
  (spec, seed) contract; `read_paired_bank_file` is the single schema
  source shared by both train and eval entries; bank_version="2".
- **Episode traces**: every eval step records pos/z/yaw/goal_dist/
  clearance/hit_count/action norm/action max_abs/done_reason/terminated/
  truncated; `EpisodePairedRecord.trace = {"bc", "rl"}` with non-empty +
  full-key validation.
- **Atomic unique artifact name**:
  `paired_eval_envstep_<9d>_ppoupd_<6d>_<ckpt12>.json` (cumulative
  counters) — tmp + os.replace; existing file with a different hash →
  RuntimeError (no overwrite); identical → no-op.
- code-reviewer 7M/5L all fixed (mock s9 layout mirrors the real frozen
  geometry; 28 phase labels stripped; expected_contract_hash now the real
  runtime hash). CPU suite 431 passed / 0 failed / 7 deselected.

### R3.5 evidence (2026-08-10, appended)

- **CAPABILITY-AUDIT** (commits `77e2fa2` + `7c9a9af`): new
  `capability_metrics.py` — d_soft^H (H=8, β=8.0, ω decay), onset, trap,
  side_flip, support-cell metrics + `OfflineDensity` (frozen dataclass:
  counts, bin_size, tau_D=p20 with p20=0→p50; empty/all-zero counts →
  ValueError). `compute_capability_metrics` raises on empty records /
  density None / any of the three capability roles missing.
- `build_runtime_eval_artifact` uses the real computation (placeholder
  `_CAPABILITY_PLACEHOLDER` deleted); offline density missing → hard-fail
  (eval preflight RuntimeError / CLI exit 2). Validator upgraded: required
  keys per group, finite non-bool numbers, n_episodes role alignment,
  sr/cr ∈ [0,1].
- RED first: `test_capability_metrics.py` 14+2 tests (hand-computed onset
  lead/lag, low-density cell coverage, trap escape); code-reviewer final
  round 2W+6L all fixed (8 new RED tests first); CPU suite 460 passed /
  2 skipped / 7 deselected; source-file-hash axis live.

### R3.6 evidence (2026-08-11, appended)

- **Step 1**: repository-wide bypass search over 9 axes (legacy DPPO
  loss, observation slicing, action slicing, formal mode, checkpoint
  writer, paired evaluator, RoleSampler caller, sigma_clip_min,
  projection flags) — no bypass; FM scripts are a separate stack (no
  dppo_uav import).
- **Step 2**: full CPU suite `-m "not gpu"` → **462 passed / 7 deselected /
  174.08s** (`temp/r3_cpu_suite.log`; the 460+2-skipped delta = two
  clean-tree live tests (identity mismatch + positive preflight)
  flipping skip/pass between dirty and clean tree: R3.5's run was on
  the pre-commit dirty tree, R3.6 Step 2 ran on the clean tree).
- **Step 3**: fresh-process checkpoint negative/positive tests → **24
  passed / 14.60s** (`temp/r3_checkpoint_fresh_process.log`): six
  mismatch classes + source_file_hashes all hard-fail before any policy
  state load (spy proof: 0 calls), matching identity restores, schema v1
  rejected, empty policy_state rejected.
- **Step 4**: stale-state removal (§0/§2/§8/§9/§10 above +
  RL_REBUILD_TRACEABILITY.md rows + task_plan_t5_rework.md) +
  `implementation_manifest_R3.json` + `review_R3.md` +
  `program/reports/routeA_rl_env_v1/official_core_r3/R3_AUDIT_REPORT.json`.
- **Step 5**: commit `d88ed73` `docs(dppo): close checkpoint entrypoint
  and evaluation audit` (+ RTM backfill `9b9e332`). RTM completion gate met: 7 R3 rows PASS; RTM OPEN
  only EQUIV-REG / GATE-ORDER and the training gates (G5/G6/G7/G8).

### R3 rework #3 evidence (2026-08-13, appended; PASS verdict later superseded by external-review #4)

- schema v4 (`n_lora_param_tensors` parameter-tensor axis, v3 audit-only), real-chain positive tests (real insert_lora → save → fresh → restore, bitwise), env RNG (hash(str(world_id)) removed from both kinematic envs, blake2b over canonical JSON, `require_fixed_python_hashseed` preflight at both canonical entries, restorable=True semantics), GPU gate run1 RED (2 failed in 3099.49s, `gpu_gate_rework3_20260813.log` sha256 6638a77c…) → fixes `3a367d7` (test harness) + `3054eee` (production: `_eval_episode` unnormalize CUDA→host) → run2 2 passed in 4405.12s (`gpu_gate_rework3_20260813_run2.log` sha256 fdff7d9e…), clean-tree CPU 532 passed / 9 deselected / 175.96s EXIT=0 (`cpu_regression_r3_rework3_20260813_final.log` sha256 515329a2…), fresh review `review_R3_rework3.md` PASS (0 P0 / 0 P1 / 5 P2). That PASS verdict was voided by external-review #4 (banner added 2026-08-13; body kept as-is).

### R3 rework #4 evidence (2026-08-13, appended)

- Ruling: external-review #4 P1 — only two legs of the RULING-7 three-way equality were implemented; adversarial probe saved=4/live=2 → validator ACCEPTED, full restore loaded policy state before `restore_lora_state` failed (STATE_LOAD_CALLS_BEFORE_FAILURE=['policy']); saved=2/live=4 → KeyError. Ruling independently reproduced before fixing.
- RED first: `4b947a0` `tests/dppo_parity/test_r3_rework4_negative.py` — 4 tests (direct validator + full-restore state-load spy, both directions) → 4 failed in 3.45s with exactly the ruling's failure modes.
- Fix: `1f0e5bb` `checkpoint.py` (+22/−1) — live parameter-tensor count checked against declared `n_lora_param_tensors` BEFORE the per-tensor shape walk and before any state load; 2n stated as current production fact only. Probe re-run: both directions CheckpointError, spy `[]`.
- Targeted checkpoint battery: 98 passed in 11.58s (new negatives + rework #2/#3 checkpoint tests + test_checkpoint.py).
- clean-tree full CPU suite: 536 passed / 9 deselected in 171.04s PYTEST_EXIT=0 (`cpu_regression_r3_rework4_20260813.log` sha256 6a57a1fe…, session header HEAD=8cfee02).
- GPU double gate (checkpoint core modified → rerun mandated by the ruling): 2 passed in 4380.17s (1:13:00) PYTEST_EXIT=0 (`gpu_gate_rework4_20260813.log` sha256 515f7bbe…, session header HEAD=8cfee02).
- Fresh independent review: `review_R3_rework4.md` — [backfilled after review completion].
