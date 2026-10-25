# Review: R3 Evidence Patch (dc3c239) — External-Review #5 Follow-up

**Reviewer**: fresh independent reviewer, zero prior project knowledge; all claims verified from files, git, and command output.
**Repo**: /home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy (branch a-official-core-rework)
**HEAD under review**: dc3c2396e732b7dd2e5ad376b3a67b0eee36cd73 (`docs(dppo): R3 evidence patch per external-review #5 — audit v3 citation, freeze-set time scope, stale comments`)
**Parent (prior closure)**: db8ae06
**Scope**: documentation/evidence patch only. Verdict applies to the 5 required fixes + 7 mandated verifications.

---

## P0 — Critical

None found.

## P1 — Should fix

None found.

## P2 — Suggestions

1. **[JUDGMENT] Per-requirement `analysis[].evidence` strings name test files** (e.g. `test_checkpoint.py key-set tests + RTM row 28`), while the log artifacts with SHA256s live in `evidence_files`. The mapping closes via the `evidence_files` section (test command → log path → sha256, e.g. `targeted_r3_rework4` entry records the exact 6-file pytest command, `98 passed in 11.08s`, head `6bd3e91`, sha256 `d64e6c45…`), so there is no evidential gap — but an explicit cross-reference (analysis.evidence → evidence_files key) would make the chain self-documenting.
2. **[JUDGMENT] Time-scope qualifier style**: the two new qualifier blocks in ENTRYPOINT_CLOSURE_REPORT.md are written in Chinese while the surrounding narrative is English. Content is correct and unambiguous; a style nit only.
3. **[JUDGMENT] Recorded runtime variance**: JSON records the targeted battery as `98 passed in 11.08s` (run at HEAD 6bd3e91); this review's rerun gave `98 passed in 9.48s`. Test-count identity holds; runtime spread is normal. No action needed.

---

## Verification 1 — JSON parse and required fields

Command:
```
python3 -c "import json; d=json.load(open('/home/cjj/AUV/program/reports/routeA_rl_env_v1/official_core_r3/R3_AUDIT_REPORT_v3.json')); print(len(d['analysis']), d['requirement_ids'])"
```
Output:
```
7 ['CRITIC-STATE-ONLY', 'CKPT-AUDIT', 'RESUME-HASH-GATE', 'LEGACY-FORMAL-GUARD', 'EVAL-STD', 'EVAL-AUDIT', 'CAPABILITY-AUDIT']
```
Verified fields (JSON lines):
- `schema_version`: `"4"` (line 2 of `version_note`-adjacent `schema_version` key, exact value `"4"`) [FACT]
- `commits.code_result_commit`: `"1f0e5bb (rework #4 fix: live LoRA param-tensor count check before shape walk / any state load)"` [FACT]
- `commits.final_closure_head`: `"db8ae06"` [FACT]
- `commits.final_closure_tree`: `"233bb8e08eb82877ebb31febb069495fbe7c3edf"` [FACT]
- Explicit non-completeness statement (version_note): "This report covers the R3 checkpoint/entrypoint/eval work package ONLY — it is NOT a claim that the whole RL training system is complete; R4 gates G4-G10 remain (G4 EQUIV-REG fixture, G5/G6 full rerun with microbatch=1 semantics, G7-G10), and formal training must not start before them." [FACT]
- Superseded history: "supersedes v1 (R3.6/462-test era, 2026-08-11) and v2 (rework #1/499-test era, 2026-08-11) — both retained as history" [FACT]
- review_R3_rework4 conditions disposition: `rework4_chain` includes `6bd3e91 (review conditions)`; a condition entry carries `"disposed_by": "6bd3e91"` [FACT]
- 7 requirement verdicts (all PASS): CRITIC-STATE-ONLY PASS / CKPT-AUDIT PASS (rework #4 后) / RESUME-HASH-GATE PASS (rework #4 后) / LEGACY-FORMAL-GUARD PASS / EVAL-STD PASS / EVAL-AUDIT PASS / CAPABILITY-AUDIT PASS [FACT]
- Three-tier results with full SHA256s: targeted 98 passed in 11.08s (`d64e6c453a6a8350cdf57e95bed52fc767e3bb51cf172b6abb8c95a341785276`), CPU 536 passed / 9 deselected in 171.04s (`6a57a1fe04a445af89bba8616d32ee02904f8e46db3cda1d003a0d445e060207`), GPU 2 passed in 4380.17s (`515f7bbe9346418892d44830dc0935865b0d8e0415e9d59a3753c00d9fc98a64`), history run2 (`fdff7d9e4114da25fe375a58eb2cc5c6f1546364b35e4d911fa0cb76dd3f5f59`) [FACT]

## Verification 2 — RTM mapping

Command: `grep -n "R3_AUDIT_REPORT" RL_REBUILD_TRACEABILITY.md`
All 7 R3 rows cite `R3_AUDIT_REPORT_v3.json`（v1/v2 历史）in their evidence (6th) column: line 16 CKPT-AUDIT, line 17 EVAL-AUDIT, line 28 CRITIC-STATE-ONLY, line 29 RESUME-HASH-GATE, line 30 LEGACY-FORMAL-GUARD, line 31 EVAL-STD, line 32 CAPABILITY-AUDIT. [FACT]
The bare `R3_AUDIT_REPORT.json` appears only at lines 876 and 1003 — both inside history/log sections ("R3.6 ... Step 4 文档去陈旧 + implementation_manifest_R3.json + review_R3.md + .../R3_AUDIT_REPORT.json" and "Rework 证据文件：...（v1 保留历史）、R3_AUDIT_REPORT_v2.json"), NOT in any R3 row's evidence cell. [FACT]
The commit diff confirms the exact change: each of the 7 rows' evidence cell was rewritten `JSON`/`R3_AUDIT_REPORT.json` → `R3_AUDIT_REPORT_v3.json`（v1/v2 历史）, with no other row content touched. [FACT]

## Verification 3 — SHA256 recomputation

Command: `sha256sum .../regression/targeted_r3_rework4_20260813.log .../cpu_regression_r3_rework4_20260813.log .../gpu_gate_rework4_20260813.log .../gpu_gate_rework3_20260813_run2.log`
Output (all four computed fresh):
```
d64e6c453a6a8350cdf57e95bed52fc767e3bb51cf172b6abb8c95a341785276  targeted_r3_rework4_20260813.log
6a57a1fe04a445af89bba8616d32ee02904f8e46db3cda1d003a0d445e060207  cpu_regression_r3_rework4_20260813.log
515f7bbe9346418892d44830dc0935865b0d8e0415e9d59a3753c00d9fc98a64  gpu_gate_rework4_20260813.log
fdff7d9e4114da25fe375a58eb2cc5c6f1546364b35e4d911fa0cb76dd3f5f59  gpu_gate_rework3_20260813_run2.log
```
All four match the values recorded in R3_AUDIT_REPORT_v3.json (`evidence_files.targeted_checkpoint_battery`, `.cpu_suite_full`, `.gpu_gate`, `.history_gpu_run2`) exactly. [FACT]

## Verification 4 — No production logic change

`git diff db8ae06..dc3c239 --stat`:
```
ENTRYPOINT_CLOSURE_REPORT.md                  |  6 ++++++
RL_REBUILD_TRACEABILITY.md                    | 14 +++++++-------
diffusion_policy_3d/rl/dppo_uav/train_loop.py | 12 ++++++------
3 files changed, 19 insertions(+), 13 deletions(-)
```
Only the three allowed files. [FACT]
`git diff db8ae06..dc3c239 -- diffusion_policy_3d/rl/dppo_uav/train_loop.py`: single hunk at ~line 1122-1132 rewriting the `environment_rng_state` comment block. Every removed line and every added line is a `#` comment line (schema v3/restorable=False wording → schema v4/restorable=True wording). Zero executable statements changed. [FACT]
`git diff db8ae06..dc3c239 -- diffusion_policy_3d/rl/dppo_uav/checkpoint.py`: 0 bytes — empty. No production code touched. [FACT]

## Verification 5 — Time-scope qualifiers

ENTRYPOINT_CLOSURE_REPORT.md:
- Lines 243-247 (P8 evidence section): frozen-set re-verification sentence now carries `**(2026-08-07 历史快照——此后经批准解除冻结：E040 脚本 2026-08-10 R3.3 修正案、env_v2 2026-08-11 RULING-7 P0-2；两处 pre/post hash 记账见 §Evidence "Freeze-lift accounting"。)**` [FACT]
- Lines 255-262 (R3.3 evidence section): remaining-4-frozen-files sentence now carries `**(2026-08-10 历史快照——env_v2 随后于 rework #3（6f2028d）经 RULING-7 P0-2 批准解除冻结并记账（§Evidence "Freeze-lift accounting"）；当前剩余 3 个冻结文件与 85ac9f0 byte-identical（2026-08-13 复核）。)**` [FACT]
- Target block exists: `Freeze-lift accounting (2026-08-13, review_R3_rework4.md P1-1 reconciliation)` at line 219. [FACT]

## Verification 6 — Worktree clean

`git status --porcelain` → empty output, exit 0. [FACT]

## Verification 7 — Quick regression (mandatory)

Command:
```
python3 -m pytest -q tests/dppo_parity/test_r3_rework4_negative.py tests/dppo_parity/test_r3_rework_negative.py tests/dppo_parity/test_r3_rework2_negative.py tests/dppo_parity/test_r3_rework3_negative.py tests/dppo_parity/test_r3_rework3_positive.py tests/dppo_parity/test_checkpoint.py
```
Output:
```
........................................................................ [ 73%]
..........................                                               [100%]
98 passed in 9.48s
```
98 passed — matches the JSON-recorded targeted battery count (98 passed; recorded 11.08s at HEAD 6bd3e91 vs 9.48s now, runtime variance only). No full CPU suite or GPU tests run, per instructions. [FACT]

---

## Findings summary

- P0: 0
- P1: 0
- P2: 3 (documentation cross-reference style; language-consistency nit; recorded runtime variance — none actionable)

## Verdict

PASS
