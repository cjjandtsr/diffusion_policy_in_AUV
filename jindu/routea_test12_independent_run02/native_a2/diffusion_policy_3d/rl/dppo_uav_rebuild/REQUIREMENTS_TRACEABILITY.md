# Requirements Traceability

| Requirement | Implementation | Verification |
|---|---|---|
| Preserve native E5 DDPM behavior | `bc_runtime/`, full 16-step `E5PPODiffusion` | E5 checkpoint identity + real parity script |
| Freeze observation encoder | LoRA only under `policy.model` (`14+33` targets) | real E5 inventory + zero-LoRA exact output |
| Use official DPPO algorithm | inherited official VPG/PPO methods | direct-symbol/inheritance tests, vendor dual hash |
| Correct reward-to-action credit | official loss optional `reward_start`; fixed `[2:6]` | gradient-location test |
| Keep Reward v1.3 | thin `RewardV13Facade` | delegation/eight-term test and source hash |
| Collision precedence | `TransitionFacts`, `KinematicAdapter` | table-driven precedence test |
| No eval-only training worlds | `manifest_train_entries(rl_train)` + task contract | eval-world rejection test |
| Preserve exploration/learning signal | no auto damping; rollback hard failures | non-zero actor/critic gradient gates |
| No policy-external trajectory rewrite | direct normalized diffusion output, only named slicing | ActionBundle slice tests |
| Exact resume boundary | atomic checkpoint; rollout boundary truncation/bootstrap | corruption/identity/resume tests |
| Real checkpoint cold load | read-only `resume_probe`, update-budget equality guard | real artifact resume command + before/after SHA256 |
| Reward and learning audit | independent `audit/` module; no reward/math mutation | eight-term, segment, target-distribution tests |
| Effective optimizer update | pre/post official metrics + actor/critic parameter delta | audited-update unit test and pilot JSON |
| Positive-benefit evidence | paired bank, bootstrap CI, three capability metrics | admission remains false until evidence supplied |
| Gazebo/PX4 claim separation | independent admission flag | lightweight result cannot set Gazebo claim |
