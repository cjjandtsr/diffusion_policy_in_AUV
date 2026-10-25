# dppo_official_cc7234a — Vendored Official DPPO Source

## Upstream

- Repo: https://github.com/irom-princeton/dppo
- Commit: `cc7234ad7ff39a8f32de3af903606723a16f0648`
- Paper: arXiv:2409.00588
- License: MIT (see `LICENSE`; copy also at repo root `LICENSE_dppo_cc7234a_MIT.txt`)
- Downloaded: 2026-08-06 via codeload.github.com tarball
  (github.com main domain unreachable from this host; see
  `program/zhidao/` notes and audit report
  `IDP3/review/RouteA_RL_Training_System_A_OfficialCore_Design_v1_0_ReadinessAudit_20260806.md`)

## Freeze policy (A-Official-Core design §5.2 G0)

- This directory is a **frozen snapshot**. Do NOT edit upstream files.
- File hashes recorded in `SHA256SUMS` (one line per file, `sha256sum` format).
- For upstream parity work, reference these files by path; the implementation
  package `diffusion_policy_3d/rl/dppo_uav/` must never import from here at
  runtime — parity tests may read them for comparison only.
- Key files for parity:
  - `model/diffusion/diffusion_ppo.py` — PPO loss reference
  - `model/diffusion/diffusion_vpg.py` — get_logprobs / get_logprobs_subsample
  - `agent/finetune/train_ppo_diffusion_agent.py` — rollout/GAE/update loop
