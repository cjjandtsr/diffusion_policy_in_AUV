"""EVAL-STD: the eval rollout must receive the RESOLVED sampling/logprob
denoising floors — never the policy default.

RED contract: a spy policy records every sample_ddpm_with_logprob call's
min_sampling_denoising_std (the official kernel's sampling floor —
R5.1; the old sigma_clip_min clamp semantics are gone). The LoopConfig
values are set to distinct 0.07/0.11; every eval sampling call must
receive exactly 0.07 (relying on the policy default 0.1 is the
config-silence defect this test proves), and BOTH floors must arrive at
_eval_episode. The eval replays the FROZEN paired bank —
never runtime task re-selection.
"""

import pytest

from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.evaluator import freeze_paired_bank
from diffusion_policy_3d.rl.dppo_uav.train_loop import (
    LoopConfig,
    run_paired_eval,
)

from fake_policy import FakePolicy
from mock_uav_env import MockUAVEnv


def _task_specs() -> list:
    """One spec per role — the shared EpisodeTaskSampler (R2.1) hard-fails
    on an empty role bucket."""
    from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
        EpisodeTaskSpec,
    )

    return [
        EpisodeTaskSpec(
            world_uid=f"mock_{role}",
            obstacle_json="{}",
            role=role,
            split="rl_train",
            admission_id=f"mock__{role}",
            generator_version="mock",
            manifest_entry_hash=f"hash_{role}",
        )
        for role in ("anticipatory", "dense_recovery",
                     "support_expansion", "yaw_lateral_recovery")
    ]


def _bank(contract: RLSystemContract, n: int = 2):
    """Freeze n entries on a clean single-slot mock — (spec, seed) fixes
    the sampled geometry, so the eval-time replay reproduces the same
    episode_config_hash and the freeze check passes."""
    env = MockUAVEnv(contract)
    return freeze_paired_bank(
        env, _task_specs()[:n], tuple(1001 + i for i in range(n))
    )


class _SpyPolicy(FakePolicy):
    """FakePolicy recording min_sampling_denoising_std per sample call."""

    def __init__(self, contract: RLSystemContract) -> None:
        super().__init__(contract)
        self.sampling_std_seen: list = []

    def sample_ddpm_with_logprob(
        self,
        obs_dict,
        eta: float = 1.0,
        min_sampling_denoising_std: float = 0.1,
        min_logprob_denoising_std: float = 0.1,
        num_inference_steps=None,
        ft_denoising_steps=None,
    ):
        self.sampling_std_seen.append(min_sampling_denoising_std)
        return super().sample_ddpm_with_logprob(
            obs_dict, eta, min_sampling_denoising_std,
            min_logprob_denoising_std, num_inference_steps,
            ft_denoising_steps,
        )


def _cfg() -> LoopConfig:
    return LoopConfig(
        mode="one_update", n_episodes=2, max_ep_len=12, device="cpu",
        output_dir="/tmp/none", eval_n_episodes=2,
        min_sampling_denoising_std=0.07,
        min_logprob_denoising_std=0.11,
    )


def test_eval_rollout_receives_resolved_sampling_std():
    """Every eval sampling call must run at the resolved floor 0.07 —
    never the policy default 0.1 (the official kernel's
    min_sampling_denoising_std floor IS the sampling noise floor)."""
    contract = RLSystemContract()
    bc_policy = _SpyPolicy(contract)
    rl_policy = _SpyPolicy(contract)
    cfg = _cfg()
    env = MockUAVEnv(contract)
    records, invalid = run_paired_eval(
        env, bc_policy, rl_policy, contract, cfg, _bank(contract),
    bank_id="test-bank",
        policy_checkpoint_hash="ckpt-eval1",
    )
    assert invalid == 0
    assert len(records) == cfg.eval_n_episodes
    seen = bc_policy.sampling_std_seen + rl_policy.sampling_std_seen
    assert seen, "the spy must see every eval sampling call"
    assert all(s == 0.07 for s in seen), (
        "every eval rollout must sample at the resolved floor 0.07, got "
        f"{set(seen)} — the policy default 0.1 is forbidden (EVAL-STD)"
    )
    assert 0.1 not in seen, (
        "no eval sampling call may fall back to the policy default 0.1"
    )


def test_eval_episode_receives_both_resolved_stds(monkeypatch):
    """Both floors reach _eval_episode (interface contract). The logprob
    floor is carried through the eval interface even though the eval path
    computes no likelihood — it binds in the PPO path only; a silent
    policy-default reliance on EITHER floor is the defect."""
    import diffusion_policy_3d.rl.dppo_uav.train_loop as tl

    contract = RLSystemContract()
    cfg = _cfg()
    received: list = []
    real = tl._eval_episode

    def recorder(env, policy, contract, spec, device, max_ep_len, seed,
                 gate, cam_far, min_sampling_denoising_std,
                 min_logprob_denoising_std):
        received.append(
            (min_sampling_denoising_std, min_logprob_denoising_std)
        )
        return real(env, policy, contract, spec, device, max_ep_len, seed,
                    gate, cam_far, min_sampling_denoising_std,
                    min_logprob_denoising_std)

    monkeypatch.setattr(tl, "_eval_episode", recorder)
    policy = FakePolicy(contract)
    env = MockUAVEnv(contract)
    records, invalid = run_paired_eval(
        env, policy, policy, contract, cfg, _bank(contract),
    bank_id="test-bank",
        policy_checkpoint_hash="ckpt-eval1",
    )
    assert invalid == 0
    assert len(records) == cfg.eval_n_episodes
    assert received, "the recorder must see every eval episode"
    assert all(a == 0.07 and b == 0.11 for a, b in received), (
        f"both resolved floors must reach _eval_episode, got {received}"
    )
