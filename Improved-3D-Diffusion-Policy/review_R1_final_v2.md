# R1 Rework #2 Closure — Independent Final Review

- Reviewer: independent, no-context (no prior project knowledge; all claims verified from git objects and disk)
- Date: 2026-08-08
- Frozen commit reviewed: 6680cd775273d13d4afa658ec95ea0036241af7a (tree 17e4e6b6b7590336426cf56b57bf02b201d089ad)
- Verdict: **PASS** (with one informational note on item G schema location)

## Per-item results

| # | Result | Evidence (command + result) |
|---|--------|------------------------------|
| A | PASS | `git show 6680cd7:.../phase1a_actor_adapter.py`: sample_dppo_rollout(policy,obs_b,contract,eta,min_sampling...,min_logprob...,sigma_clip_max), make_get_new_logprobs(policy,contract,eta,...), _obs_conditions(policy,obs_b) — no cfg_lambda param; all resolve via `_require_policy_cfg_lambda`. Helper: missing attr -> AttributeError; non-finite -> ValueError; `math.isclose(lam,1.5,rel_tol=0.0,abs_tol=0.0)` fail -> ValueError. Low-level per_dim_denoising_logprobs / per_step_denoising_logp exist in chain_logprobs.py. Diff vs 7141934 confirms `cfg_lambda` param removed (3 sites). |
| B | PASS | policy_loader.py: `cfg_lambda_cfg = policy_cfg.get("cfg_lambda", 1.5)`; `!= 1.5` -> ValueError pre-construction; post-construction `if policy.cfg_lambda != 1.5: raise ValueError`. |
| C | PASS | train_loop.py `_build_checkpoint_data`: "policy_contract" = {policy_cfg_lambda (1.5), rollout_policy_batch_size: 1, likelihood_microbatch_size: 1} + trainer next_value_batch_size_configured (from config) / next_value_batch_size_resolved (from metrics). TrainerConfig.__post_init__ rejects batch_size<=1 and next_value_batch_size<=0. Round-trip test exists and asserts exact dicts + resolved==metrics (test_train_loop.py:270-302). |
| D | PASS | official_loss_adapter.py: `ratio_std = 0.0 if ratio.numel() == 1 else float(ratio.std().item())` inside `with torch.no_grad()` (loss/grads untouched). test_ratio_std_diagnostic.py: B=1 -> 0.0/finite/grads flow/warnings-as-errors; B=2 parity with manual std. |
| E | PASS | G5 v1 vs v2 `identity_thresholds` identical: {"max_abs_diff": 1e-05, "ratio_mean": [0.99999, 1.00001]}. Test constants IDENTITY_TOLERANCE=1e-5, RATIO_MIN/MAX=0.99999/1.00001. v2: B=1 max_abs_diff=0.0, 1_randperm=0.0, ratio_mean 1.0; B=128 3.662109375e-4 informational. |
| F | PASS | /tmp/r1_rework2_authority.log (exists): "275 passed, 8 warnings in 3474.00s (0:57:53)", EXIT_CODE=0, no FAILED/ERROR lines. All 8 warning entries are "synth: no ray hit" RuntimeWarning (synth_pointcloud.py:495). Zero ddof/std warnings (sealed run had 21; now none), zero CFG-related warnings. Log sha256 a34cea96... matches manifest log_sha256. |
| G | PASS* | G5_ACTUAL_BATCH_IDENTITY_v2.json holds 22 source_file_hashes — all 22 equal `git show 6680cd7:<file> | sha256sum` (0 mismatches). Manifest v2 commit_binding: code_result_commit 6680cd7, git_tree_hash 17e4e6b6b7590336426cf56b57bf02b201d089ad == real `git rev-parse 6680cd7^{tree}`, worktree_status_at_run clean. G5 v2 code_result_commit == 6680cd7, git_tree_hash matches. *Note: claim said hashes live in implementation_manifest_R1_v2.json — that file has NO source_file_hashes key; the binding lives in the G5 v2 JSON instead. Binding substance fully satisfied. |
| H | PASS | G5_ACTUAL_BATCH_IDENTITY.json: code_result_commit "61d2b772e2a642c93bb2506cbc9cd6d89fb7a9d3", B=128 4.57763671875e-4, v1 thresholds unchanged. implementation_manifest_R1.json on-disk sha256 b19ecef4... == `git show ba81ac5` and `git show 7141934` (byte-identical); git log last touch ba81ac5. R1_CORRECTNESS_REPORT.json references r1_rework_final_code 61d2b77 and authority run "259 passed / 26 warnings" (v1 closure). v1 file mtimes 16:47 predate all v2 files (19:09). |
| I | PASS | `git diff --name-only 7141934 6680cd7` = exactly the 15 claimed files (6 product + 8 tests + task_plan_r1_rework2.md). `git diff 7141934 6680cd7 -- third_party/dppo_official_cc7234a` empty. No reward/environment_adapter/Gazebo/dynamics files in the diff. |
| J | PASS | `git show --stat 4f71798`: 7 files — all tests/ + task_plan_r1_rework2.md, +497/-88; zero diffusion_policy_3d product files. RED (tests) precedes GREEN 6680cd7 in history. |
| K | PASS | Independent CPU re-run: `python -m pytest -q tests/dppo_parity/test_cfg_contract.py test_ratio_std_diagnostic.py test_next_value_chunking.py test_train_loop.py -p no:cacheprovider` -> "31 passed in 8.55s", exit 0 (env idp3, python 3.8.20, torch 2.4.1+cu121). Matches claimed 19+2+4+6=31 (test_cfg_contract's 13 defs collect 19 via parametrization). |
| L | PASS | Guide_A_OfficialCore_R1_PPO_GAE_Correctness_v1_1.md exists (2026-08-08); §"v1.1 取代 v1.0" explicitly supersedes the B=128 identity clause and pins B=1 behavior-batch semantics (B=128 informational). RL_REBUILD_TRACEABILITY.md: zero occurrences of "4.27e-4"; LOGP-BATCH-ID row (line 18) and R1.2 ledger line (252) use 4.57763671875e-4 (v1 run value) with rework #2 value 3.662109375e-4 noted. R1_阶段总结.md rollback section: only git revert path (newest-first 6680cd7 4f71798), explicitly "禁止 git reset --hard". |

## Overall verdict

**PASS.** All 12 items A–L verified independently with evidence; no FAILs.

Conditions / notes:
1. Informational (item G): the review prompt's description of the v2 manifest schema is inaccurate — implementation_manifest_R1_v2.json does not contain a source_file_hashes field. The file-level hash binding exists and is fully correct in G5_ACTUAL_BATCH_IDENTITY_v2.json (22/22 hashes match the frozen commit). No action required.
2. Informational: _build_checkpoint_data persists policy_cfg_lambda via `getattr(policy, "cfg_lambda", 1.5)` (default fallback in the persistence layer only); the formal resolution path (_require_policy_cfg_lambda) still hard-fails, and the round-trip test pins the persisted value to exactly 1.5. Not a contract violation.
3. Not re-verified (out of scope, CPU-only environment): GPU gate tests (G5/G6 real-env) — their results are taken from the authority log (item F), which is bound by sha256 to the manifest. The 31-test CPU re-run (item K) independently confirms all rework #2 product gates.
