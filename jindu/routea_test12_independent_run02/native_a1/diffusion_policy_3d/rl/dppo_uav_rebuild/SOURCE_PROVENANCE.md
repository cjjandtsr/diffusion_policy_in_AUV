# Source Provenance

## 1. DPPO official program

The complete local official snapshot is under
`third_party/dppo_official_cc7234a`.

- Upstream commit: `cc7234ad7ff39a8f32de3af903606723a16f0648`
- Upstream manifest: `SHA256SUMS`
- Manifest SHA256: `5594f7d9a68d27d5b8e1bbeb4317faf8334cbe0d6202e3c8a51b103db16fee91`
- Manifest entries: `401`
- Network used in this rebuild: `false`
- Transfer source: the user-provided, P1-verified local WSL snapshot

All 401 paths are official DPPO files.  The authoritative per-file list and
original hashes are `third_party/dppo_official_cc7234a/SHA256SUMS`.

### Official algorithm files used directly

- `model/diffusion/diffusion.py`
- `model/diffusion/diffusion_vpg.py`
- `model/diffusion/diffusion_ppo.py`
- `model/diffusion/sampling.py`

`E5PPODiffusion` inherits official `forward`, `p_mean_var`, `get_logprobs`,
`get_logprobs_subsample`, and `loss`.  No project-side PPO ratio, clipping,
denoising discount, Normal log-probability, or value-loss copy exists.

### Official files minimally modified

| Official file | Original SHA256 | Modified SHA256 | Reason | Patch |
|---|---|---|---|---|
| `model/diffusion/diffusion_ppo.py` | `e6071bb23b426b0c75e347ccde71bb91e3a572de0b17f0a28fa294dc3f3ca563` | `cd770a61980a2bbd415725fb67eaddf17ae911905b5bf5530ad7684f26bff9db` | Optional `reward_start=0`; E5 uses `[2:6]`, upstream default unchanged | `uav_idp3/patches/0001-e5-reward-offset.patch` |
| `model/diffusion/diffusion_vpg.py` | `fa697cf69d38adedbd3fd4afd06ee96f37133b480b61505326c3cc0ce833fc54` | `7ab594d7c3f22a2127219b1b62c665449427d552bedb92f3b3b4378e0dc8d70c` | Complete direct-x0 DDIM support and add opt-in deterministic diffusers CPU-scalar parity; upstream epsilon/default behavior remains unchanged | `uav_idp3/patches/0002-native-ddim-clip-mode.patch` |

### New files placed inside the official directory

- `uav_idp3/__init__.py`
- `uav_idp3/e5_runtime.py`
- `uav_idp3/patches/0001-e5-reward-offset.patch`
- `uav_idp3/patches/0002-native-ddim-clip-mode.patch`

These four are iDP3-UAV additions, not upstream DPPO files.

## 2. Reused existing project programs

These files are imported and used in place; this delivery does not replace or
duplicate their implementation:

- `diffusion_policy_3d/policy/diffusion_pointcloud_policy.py` — native E5 model and DDIM inference
- `diffusion_policy_3d/common/normalizer.py` — native action/observation normalization
- `diffusion_policy_3d/env/reward_v13.py` — all eight formal Reward v1.3 terms and history
- `diffusion_policy_3d/env/kinematic_uav_env.py` — lightweight transition dynamics
- `diffusion_policy_3d/env/traj_utils.py` — 11D delta action conversion
- `diffusion_policy_3d/env/collision_geom.py` — collision/clearance geometry
- `diffusion_policy_3d/model/common/synth_pointcloud.py` — synthetic point cloud generation
- `diffusion_policy_3d/rl/dppo_uav/integration.py` — admitted `rl_train` manifest parsing

The historical `diffusion_policy_3d/rl/routea_replay_buffer.py` is deliberately
not reused: it hard-checks legacy 1536-D observations and represents the old
off-policy replay route, whereas official DPPO requires on-policy chains.

## 3. Files written from scratch for this rebuild

Every source/config under `diffusion_policy_3d/rl/dppo_uav_rebuild/` is new,
including:

- `bc_runtime/`: E5 identity, native loader, inference bridge, fingerprint
- `official_core/`: verified import boundary and dual-hash vendor identity
- `actor_ddpm/`: conditioning, LoRA wrappers, official-chain rollout sampler
- `environments/`: immutable tasks/facts, point-cloud gate, factory, old-env adapter
- `rewards/`: typed interface and thin Reward v1.3 facade
- `engine/`: contracts, ordered critic, GAE, on-policy buffer, transaction, trainer
- `checkpointing/`: identity-first atomic save/restore
- `audit/`: read-only rollout, learning-target, parameter-delta, and post-update evidence
- `evaluation/`: frozen paired bank, bootstrap metrics, admission labels
- `deployment/`: shared checkpoint/deployment loader
- `entrypoints/`: system builder, runtime identity, canonical pipeline
- `config/e5_bc_identity.yaml`, `config/train_e5_dppo_v1.yaml`, and the
  isolated `config/train_e5_dppo_pilot10_v1.yaml`

The following new scripts and tests were also written from scratch:

- `scripts/train_dppo_uav_rebuild.py`
- `scripts/verify_dppo_uav_rebuild.py`
- `scripts/verify_dppo_checkpoint_resume.py`
- `scripts/verify_dppo_vendor_snapshot.py`
- `scripts/verify_e5_identity_config.py`
- `scripts/verify_e5_native_load.py`
- `scripts/verify_e5_lora_inventory.py`
- `scripts/verify_e5_official_parity.py`
- `scripts/verify_e5_system_build.py`
- `scripts/diagnose_e5_first_transition.py`
- every file under `tests/dppo_uav_rebuild/`

## 4. LoRA scope clarification

P5 counted the complete E5 policy as `16 Linear + 43 Conv1d`.  That total
contains the observation encoder (`2 Linear + 10 Conv1d`).  The formal actor
keeps the encoder frozen and installs LoRA only into `policy.model`, therefore
the trainable registry is `14 Linear + 33 Conv1d = 47` denoiser modules.  This
is consistent with the design requirement that encoder updates remain zero.

## 5. Verification evidence

- Historical Audit-v1 full regression before the RNG restore fix: `59 passed`,
  `0 failed`, `0 errors`, `0 skipped`; JUnit: `FINAL_TEST_RESULTS.xml`
- Current default aggregate fast contracts: `51 passed`, `1 deselected`,
  `0 failed`, `10.62s`
- CUDA checkpoint roundtrip regression: `2 passed`; covers both default CPU
  restore and `map_location=cuda:0` RNG restore
- Real E5 zero-LoRA: exact output, max absolute difference `0.0`
- Real E5 native/official 10-step parity: max/mean absolute difference `0.0`
- Real full-system build: PASS
- Current staged aggregate: vendor, fast contracts, E5 identity, LoRA,
  system build, and native/official parity PASS; RL cold-resume awaits copying
  the RNG fix into WSL
- Real pre-audit WSL smoke: exit `0`; `128` transitions; one attempted and
  committed update; zero rejected updates; no OOM, NaN/Inf, or traceback
- Network downloads or installs: none

The audit-v1 source has not yet produced a WSL pilot artifact.  These checks
verify implementation, integration, and one-update executability.  They do not
constitute evidence of positive RL benefit, Reward v1.3 task-level correctness,
long-run training stability, or Gazebo/PX4 capability.
