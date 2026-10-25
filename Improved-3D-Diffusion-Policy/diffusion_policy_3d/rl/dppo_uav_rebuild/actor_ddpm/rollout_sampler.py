"""Official-chain DDPM rollout sampling with explicit E5 action views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.contracts import (
    ActionBundle,
    ActionSpec,
)


class RolloutContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class DDPMRollout:
    condition: Dict[str, torch.Tensor]
    normalized_chain: torch.Tensor
    old_logprobs: torch.Tensor
    timesteps: torch.Tensor
    action: ActionBundle


class DDPMRolloutSampler:
    def __init__(self, runtime, normalizer, action_spec: ActionSpec = None):
        self.runtime = runtime
        self.normalizer = normalizer
        self.action_spec = action_spec or ActionSpec.e5()

    @torch.no_grad()
    def sample(self, condition, *, deterministic: bool) -> DDPMRollout:
        samples = self.runtime(
            cond=condition,
            deterministic=deterministic,
            return_chain=True,
            use_base_policy=False,
        )
        chain = samples.chains
        expected = (
            condition["state"].shape[0],
            self.runtime.ft_denoising_steps + 1,
            self.action_spec.full_horizon,
            self.action_spec.action_dim,
        )
        if tuple(chain.shape) != expected:
            raise RolloutContractError(
                "official chain mismatch: expected {}, got {}".format(
                    expected, tuple(chain.shape)
                )
            )
        flat_logprobs = self.runtime.get_logprobs(condition, chain)
        logprob_shape = (
            expected[0],
            self.runtime.ft_denoising_steps,
            self.action_spec.full_horizon,
            self.action_spec.action_dim,
        )
        old_logprobs = flat_logprobs.reshape(logprob_shape)
        normalized_full = samples.trajectories
        full = self.normalizer["action"].unnormalize(normalized_full)
        policy, deployment, rollout = self.action_spec.split(full)
        action = ActionBundle(
            spec=self.action_spec,
            full=full,
            policy=policy,
            deployment_executed=deployment,
            rollout_executed=rollout,
        )
        return DDPMRollout(
            condition={key: value.detach() for key, value in condition.items()},
            normalized_chain=chain.detach(),
            old_logprobs=old_logprobs.detach(),
            timesteps=self.runtime.ddim_t[-self.runtime.ft_denoising_steps :]
            .detach()
            .clone(),
            action=action,
        )

