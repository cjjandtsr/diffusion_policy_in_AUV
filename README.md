# Diffusion Policy in AUV

End-to-end autonomous obstacle-avoidance trajectory planning for UAVs, built on
[Improved 3D Diffusion Policy (iDP3)](https://github.com/YanjieZe/Improved-3D-Diffusion-Policy).
The policy observes point-cloud measurements and UAV state, and outputs 11-dim trajectory chunks
(position + velocity + acceleration) directly — no global planner or external map in the loop.

The repo name keeps the project codename "AUV"; the target vehicle in the code and experiments is a UAV.

## What is inside

- **3D diffusion / flow-matching policies** over point-cloud observations
  (`diffusion_policy_3d/policy/`): DDPM point-cloud policy, flow-matching (FM) policy family
  (`fm_base_policy.py`, `phase1a_policy_fm*.py`), SDE/ODE samplers, LoRA adapters for RL fine-tuning.
- **DPPO reinforcement learning** on top of the diffusion policy
  (`diffusion_policy_3d/rl/`): training loop, replay buffer, environment adapter, and
  `dppo_uav_rebuild/` — the canonical RL branch that delegates sampling, denoising likelihood, and
  PPO mathematics to a vendored official DPPO implementation (snapshot `cc7234a`, MIT).
  See `diffusion_policy_3d/rl/dppo_uav_rebuild/README.md` for fixed contracts and gates.
- **Kinematic UAV environment** (`diffusion_policy_3d/env/`): fast closed-loop simulator with
  synthetic point-cloud sensing, collision geometry, and shaped rewards
  (`reward_v13` … `reward_v17`: progress, collision, arrival). A Gazebo / PX4 SITL interface
  (`gazebo_uav_env.py`, `dppo_uav_rebuild/deployment/`) targets real-flight-style rollout.
- **Formal evaluation endpoint** (`scripts/evaluate_dppo_uav_formal_endpoint.py`): audited benchmark
  splits (`fixed32`, `routea-val-test`), paired comparison against the BC baseline, and
  checkpoint-identity checks.
- **Safety guidance module** (`policy/safety_guide.py`): in-denoiser safety guidance during sampling.
- **Tests**: parity gates, checkpoint resume, environment and reward contracts under `tests/`.

Several research paths coexist in this codebase (DDPM BC, FM BC, DPPO variants, ReinFlow
experiments). The current main line is the FM policy + official-semantics DPPO rebuild.

## Repository layout

```
.
├── Improved-3D-Diffusion-Policy/          # main project
│   ├── diffusion_policy_3d/               # core package
│   │   ├── policy/                        # diffusion/FM policies, DPPO kernel, safety guide
│   │   ├── rl/                            # DPPO trainers, official rebuild, ReinFlow, replay buffer
│   │   ├── env/                           # kinematic UAV env, Gazebo env, rewards
│   │   ├── model/                         # point-cloud/vision encoders, diffusion nets
│   │   ├── config/                        # Hydra configs
│   │   └── dataset/ data/ training/ workspace/ common/
│   ├── scripts/                           # training / evaluation entry points and diagnostics
│   ├── tests/                             # pytest suites (parity, gates, contracts)
│   ├── third_party/                       # vendored official DPPO reference (MIT)
│   ├── program/  zhidao/                  # internal reports and execution guides (Chinese)
│   └── RL_REBUILD_TRACEABILITY.md         # RL rebuild traceability notes
├── scripts/                               # upstream iDP3 wrappers (kept for reference)
└── third_party/                           # upstream iDP3 third-party components
```

The scripts under the repo-root `scripts/` are upstream iDP3 wrappers; all AUV-specific entry
points live under `Improved-3D-Diffusion-Policy/scripts/`.

## Installation

The code is developed and tested in a conda environment with the following versions
(recorded in `Improved-3D-Diffusion-Policy/BASELINE_MANIFEST.json`):

- Python 3.8.20
- PyTorch 2.4.1 + CUDA 12.1
- NumPy 1.23.5
- conda env name: `idp3`

```bash
# 1. create the environment (example)
conda create -n idp3 python=3.8
conda activate idp3
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# 2. install the package in editable mode
cd Improved-3D-Diffusion-Policy
pip install -e .
```

Datasets and checkpoints live under `data/` and are excluded from this repository
(see `.gitignore`). The RL trainer loads BC checkpoints from `data/outputs/...` —
see `diffusion_policy_3d/rl/dppo_uav_rebuild/README.md` for the fixed checkpoint contracts.

## Quick start

All commands are run from `Improved-3D-Diffusion-Policy/`.

**BC baseline evaluation**

```bash
python scripts/eval_bc_baseline.py   # loads the FM BC checkpoint, runs rl_val worlds
```

**RL training (official-semantics DPPO, one-update smoke gate)**

```bash
python scripts/train_dppo_uav_rebuild.py \
  --config diffusion_policy_3d/rl/dppo_uav_rebuild/config/train_e5_dppo_v1.yaml
```

Increase `run.updates` only after the one-update gate passes. The trainer never silently
reduces learning rate, exploration, advantages, or reward signal after a rejected update —
it rolls back the whole update.

**Offline verification of the rebuild**

```bash
python scripts/verify_dppo_uav_rebuild.py \
  --checkpoint data/outputs/2026.08.16/08.53.26_e5_lambda0p5/checkpoints/latest.ckpt \
  --device cuda:0 \
  --rl-config diffusion_policy_3d/rl/dppo_uav_rebuild/config/train_e5_dppo_v1.yaml \
  --rl-checkpoint data/outputs/dppo_uav_rebuild/e5_dppo_v1/checkpoint_update_000001
```

**Formal evaluation**

```bash
python scripts/evaluate_dppo_uav_formal_endpoint.py \
  --config <policy.yaml> \
  --rl-checkpoint <path-to-rl-checkpoint> \
  --eval-config-dir <eval-configs> \
  --benchmark fixed32 \
  --worlds-test <worlds-dir> \
  --worlds-layera <layera-dir> \
  --output-dir <results-dir> \
  [--expected-update 256]
```

Other training entry points in `scripts/` (DDPM PPO, FM-SDE PPO / ARFM, ReinFlow):

- `train_dppo_uav_official.py`
- `train_fm_sde_ppo.py`, `train_fm_sde_arfm.py`, `train_reinflow_fm.py`
- `train_a2_10k_capability.py`

## Testing

```bash
pytest tests/
```

The suites cover DPPO parity with the vendored official implementation, checkpoint resume,
environment/reward contracts, and gate instrumentation.

## Documentation

Internal documentation (Chinese) is kept in the repository:

- `Improved-3D-Diffusion-Policy/RL_REBUILD_TRACEABILITY.md` — RL rebuild traceability
- `Improved-3D-Diffusion-Policy/ENTRYPOINT_CLOSURE_REPORT.md` — entry-point closure report
- `Improved-3D-Diffusion-Policy/diffusion_policy_3d/rl/dppo_uav_rebuild/README.md` — rebuild contracts and gates
- `Improved-3D-Diffusion-Policy/program/` — experiment reports and analysis
- `Improved-3D-Diffusion-Policy/zhidao/` — execution guides for training/evaluation runs

## License

MIT. Based on [Improved 3D Diffusion Policy](https://github.com/YanjieZe/Improved-3D-Diffusion-Policy)
(© 2024 Yanjie Ze, MIT). The vendored official DPPO reference (`third_party/dppo_official_cc7234a`)
is © 2024 Intelligent Robot Motion Lab, MIT — see `Improved-3D-Diffusion-Policy/LICENSE_dppo_cc7234a_MIT.txt`.

## Acknowledgments

- [iDP3: Generalizable Humanoid Manipulation with 3D Diffusion Policies](https://github.com/YanjieZe/Improved-3D-Diffusion-Policy)
- Official DPPO implementation (vendored snapshot `cc7234a`)
