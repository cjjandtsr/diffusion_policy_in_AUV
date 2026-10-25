from types import SimpleNamespace

import torch

from diffusion_policy_3d.rl.dppo_uav_rebuild.actor_ddpm.rollout_sampler import (
    DDPMRolloutSampler,
)


class _Runtime:
    ft_denoising_steps = 10
    ddim_t = torch.tensor([45, 40, 35, 30, 25, 20, 15, 10, 5, 0])

    def __call__(self, cond, deterministic, return_chain, use_base_policy):
        batch = cond["state"].shape[0]
        chain = torch.arange(batch * 11 * 16 * 11, dtype=torch.float32).reshape(
            batch, 11, 16, 11
        )
        return SimpleNamespace(trajectories=chain[:, -1], chains=chain)

    def get_logprobs(self, cond, chains):
        batch = chains.shape[0]
        return torch.zeros(batch * 10, 16, 11)


class _ActionNormalizer:
    def unnormalize(self, value):
        return value + 1000.0


class _Normalizer:
    def __getitem__(self, key):
        assert key == "action"
        return _ActionNormalizer()


def test_rollout_preserves_full_chain_and_named_action_views():
    sampler = DDPMRolloutSampler(_Runtime(), _Normalizer())
    cond = {"state": torch.zeros(2, 576), "global_cond": torch.zeros(2, 576)}

    result = sampler.sample(cond, deterministic=False)

    assert result.normalized_chain.shape == (2, 11, 16, 11)
    assert result.old_logprobs.shape == (2, 10, 16, 11)
    assert result.action.full.shape == (2, 16, 11)
    assert result.action.policy.shape == (2, 8, 11)
    assert result.action.deployment_executed.shape == (2, 2, 11)
    assert result.action.rollout_executed.shape == (2, 4, 11)
    torch.testing.assert_close(result.action.policy, result.action.full[:, 2:10])

