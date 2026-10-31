# E5 DDPM + Official DPPO RL Rebuild

This package is the canonical DDPM reinforcement-learning branch for the E5
checkpoint.  It preserves native E5 behavior and delegates sampling,
denoising likelihood, and PPO mathematics to the vendored DPPO implementation.

## Fixed contracts

- BC checkpoint: `data/outputs/2026.08.16/08.53.26_e5_lambda0p5/checkpoints/latest.ckpt`
- E5 input: latest 3 chronological frames, `3 x 192 = 576`
- Diffusion chain: `10` DDIM transitions over a full `(16, 11)` trajectory
- Policy action view: full trajectory `[2:10]` (`8 x 11`)
- Lightweight rollout execution: policy prefix 4 = full trajectory `[2:6]`
- Deployment execution: policy prefix 2 = full trajectory `[2:4]`
- PPO reward credit: `reward_start=2`, `reward_horizon=4`
- LoRA scope: denoiser only (`14 Linear + 33 Conv1d = 47`); obs encoder frozen
- Reward: existing `diffusion_policy_3d.env.reward_v13` (delegated, not copied)
- Training data: manifest `rl_train` only; `w4/w8/w11/w12/w18` remain eval-only

## Offline verification (one command)

Run after merging this overlay into the WSL repository:

```bash
cd /home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy
/home/cjj/miniconda3/envs/idp3/bin/python scripts/verify_dppo_uav_rebuild.py \
  --checkpoint data/outputs/2026.08.16/08.53.26_e5_lambda0p5/checkpoints/latest.ckpt \
  --device cuda:0 \
  --rl-config diffusion_policy_3d/rl/dppo_uav_rebuild/config/train_e5_dppo_v1.yaml \
  --rl-checkpoint data/outputs/dppo_uav_rebuild/e5_dppo_v1/checkpoint_update_000001
```

The verifier does not download or install anything.  It must report separate
states for fast local contracts, real E5 parity, real RL checkpoint resume,
real GPU training, and Gazebo/PX4.  The default contract suite excludes eight
heavy real-vendor tests whose behavior is already covered by the real LoRA,
system-build, and full-parity subprocesses.  Those tests remain available for
developer-only full regression, but are not repeated in the normal gate.

## One-update run

The provided YAML is intentionally one update (`128` transitions) so the first
execution is an auditable smoke/update gate:

```bash
/home/cjj/miniconda3/envs/idp3/bin/python scripts/train_dppo_uav_rebuild.py \
  --config diffusion_policy_3d/rl/dppo_uav_rebuild/config/train_e5_dppo_v1.yaml
```

Increase `run.updates` only after the real parity and one-update artifact pass.
The trainer never auto-reduces learning rates, exploration, advantages, or
reward signal after a rejected update; it rolls back the whole update.

The first real WSL smoke completed on 2026-08-27 with exit code `0`, one
committed actor/critic update, `128` environment transitions, no rejected
update, no OOM, and no non-finite value.  This proves executable training and
atomic artifact publication only; it is not positive-benefit evidence.

## Audit v1 and cold resume

Audit v1 is independent of the official DPPO math, Reward v1.3, and the
environment.  Each update JSON now records:

- all eight Reward v1.3 term distributions and reward/term-sum consistency;
- completed and partial segment returns, terminal reasons, and task roles;
- value, next-value, GAE advantage, and return distributions;
- pre-update official metrics, post-update official metrics, and exact
  actor/critic parameter deltas.

The aggregate command above cold-loads the real one-update artifact without
performing another update.  The standalone command below remains available
only for focused diagnosis:

```bash
/home/cjj/miniconda3/envs/idp3/bin/python \
  scripts/verify_dppo_checkpoint_resume.py \
  --config diffusion_policy_3d/rl/dppo_uav_rebuild/config/train_e5_dppo_v1.yaml \
  --checkpoint data/outputs/dppo_uav_rebuild/e5_dppo_v1/checkpoint_update_000001
```

The verifier refuses to run if the config update budget is greater than the
checkpoint update index, and verifies that all checkpoint hashes remain
unchanged.  CPU and CUDA RNG byte states are normalized back to contiguous CPU
`uint8` tensors before PyTorch RNG restoration, so `map_location=cuda:0` cannot
turn the CPU RNG state into an invalid CUDA ByteTensor.

## Ten-update pilot

The pilot starts again from the frozen E5 BC checkpoint; it does not continue
the unaudited smoke update:

```bash
/home/cjj/miniconda3/envs/idp3/bin/python scripts/train_dppo_uav_rebuild.py \
  --config diffusion_policy_3d/rl/dppo_uav_rebuild/config/train_e5_dppo_pilot10_v1.yaml
```

Its separate output is
`data/outputs/dppo_uav_rebuild/e5_dppo_pilot10_v1_seed20260827`.  The pilot is
for update/audit time-series diagnosis.  It is still not a positive-benefit,
multi-seed, Gazebo/PX4, or formal Reward v1.3 validation claim.

## Copy boundary

Merge these paths into the WSL repository root without replacing unrelated
files:

- `diffusion_policy_3d/rl/dppo_uav_rebuild/`
- `scripts/*dppo_uav_rebuild.py`
- `scripts/verify_dppo_checkpoint_resume.py`
- `scripts/verify_e5_*.py` and `scripts/verify_dppo_vendor_snapshot.py`
- `tests/dppo_uav_rebuild/`
- `third_party/dppo_official_cc7234a/` (complete tree; see `SOURCE_PROVENANCE.md`)

Do not copy a staged top-level `diffusion_policy_3d/__init__.py` or
`diffusion_policy_3d/rl/__init__.py`; this delivery intentionally contains
neither.
