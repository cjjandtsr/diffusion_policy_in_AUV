# Migration Map

| Old component | New canonical component | Decision |
|---|---|---|
| E5 checkpoint loading paths | `bc_runtime/native_loader.py` | Reuse Hydra/native policy; add exact identity and device gates |
| Legacy 8-frame actor/replay condition (1536) | `bc_runtime/inference.py` + `actor_ddpm/conditioning.py` | Environment remains 8 frames; actor uses chronological latest 3 = 576 |
| Historical project DPPO transition/loss adapters | official `VPGDiffusion` / `PPODiffusion` | Replaced by direct inheritance; no duplicate math |
| `routea_replay_buffer.py` | `engine/rollout_buffer.py` | Replaced: formal DPPO is on-policy and stores denoising chains |
| Mean-pooled critic input | `engine/value.py` | Replaced with ordered 3x192 temporal input |
| Reward copies/variants | `rewards/v13/facade.py` | Delegates existing `env/reward_v13.py`; v1.6 excluded from formal training |
| Ambiguous action horizon | `ActionSpec` + `ActionBundle` | Full16, policy2:10, rollout2:6, deployment2:4 |
| Non-transactional update | `engine/update_transaction.py` | Actor/critic/optimizer/RNG rollback on hard failure |
| Monolithic checkpoint | `checkpointing/` | Header identity first, state hash, atomic directory publish |
| Separate eval/deploy loaders | `deployment/loader.py` | One BC+LoRA loader for both |

