"""Microbatch accumulation must equal one direct logical reduction."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.trainer import DDPMTrainer


class Bundle:
    def __init__(self, parameter): self.parameter = parameter
    def named_adapter_parameters(self): yield "actor", self.parameter


class Runtime:
    def __init__(self):
        self.actor_parameter = nn.Parameter(torch.tensor(0.02, dtype=torch.float64))
        class Critic(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(576, 1)
            def forward(self, observation):
                return self.linear(observation["state"])
        self.critic = Critic()
        self.critic.double()
        nn.init.constant_(self.critic.linear.weight, 0.001)
        nn.init.constant_(self.critic.linear.bias, 0.0)
        self.gamma_denoising = 0.9; self.ft_denoising_steps = 10
        self.clip_ploss_coef_base = 0.001; self.clip_ploss_coef = 0.1
        self.clip_ploss_coef_rate = 3.0; self.clip_vloss_coef = 0.2
        self.clip_advantage_lower_quantile = 0.05
        self.clip_advantage_upper_quantile = 0.95

    def get_logprobs_subsample(self, obs, prev, nxt, inds, get_ent=True):
        return self.actor_parameter.expand(len(inds), 16, 11), torch.ones(len(inds), 11, dtype=torch.float64)


def make_trainer():
    runtime = Runtime()
    return DDPMTrainer(
        runtime=runtime, lora_bundle=Bundle(runtime.actor_parameter),
        actor_optimizer=torch.optim.Adam([runtime.actor_parameter], lr=1e-3),
        critic_optimizer=torch.optim.Adam(runtime.critic.parameters(), lr=1e-3),
    ), runtime


def batch(n=257):
    state = torch.linspace(-1, 1, n * 576, dtype=torch.float64).reshape(n, 576)
    return SimpleNamespace(
        condition={"state": state, "global_cond": state},
        chains_prev=torch.zeros(n, 16, 11, dtype=torch.float64), chains_next=torch.zeros(n, 16, 11, dtype=torch.float64),
        denoising_inds=torch.arange(n).remainder(10), returns=torch.linspace(-1, 1, n, dtype=torch.float64),
        oldvalues=torch.zeros(n, dtype=torch.float64), advantages=torch.linspace(-2, 2, n, dtype=torch.float64),
        oldlogprobs=torch.zeros(n, 16, 11, dtype=torch.float64),
    )


@pytest.mark.parametrize("size", [257, 1280, 2560])
def test_direct_and_uneven_microbatch_updates_match(size):
    direct, direct_runtime = make_trainer()
    accum, accum_runtime = make_trainer()
    accum_runtime.critic.load_state_dict(direct_runtime.critic.state_dict())
    direct_metrics = direct.update_logical_batch(batch(size), microbatch_pairs=size)
    accum_metrics = accum.update_logical_batch(batch(size), microbatch_pairs=128)
    torch.testing.assert_close(direct_runtime.actor_parameter, accum_runtime.actor_parameter, rtol=1e-5, atol=1e-7)
    for left, right in zip(direct_runtime.critic.parameters(), accum_runtime.critic.parameters()):
        torch.testing.assert_close(left, right, rtol=1e-8, atol=1e-8)
    assert abs(direct_metrics.total_loss - accum_metrics.total_loss) <= 1e-6
    assert direct.counters.committed_updates == accum.counters.committed_updates == 1
