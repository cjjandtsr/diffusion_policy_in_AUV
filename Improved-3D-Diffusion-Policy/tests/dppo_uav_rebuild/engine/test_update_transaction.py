from types import SimpleNamespace

import pytest
import torch
from torch import nn

from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.trainer import DDPMTrainer
from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.update_transaction import (
    UpdateRejected,
)


class _Bundle:
    def __init__(self, parameter):
        self.parameter = parameter

    def named_adapter_parameters(self):
        yield "lora_b", self.parameter


class _Runtime:
    def __init__(self):
        self.actor_parameter = nn.Parameter(torch.tensor(0.25))
        self.critic = nn.Linear(2, 1)
        nn.init.constant_(self.critic.weight, 0.2)
        nn.init.constant_(self.critic.bias, 0.1)
        self.last_kwargs = None

    def loss(self, *args, **kwargs):
        self.last_kwargs = kwargs
        policy = (self.actor_parameter - 1.0).pow(2)
        value = sum(parameter.pow(2).sum() for parameter in self.critic.parameters())
        zero = policy.new_zeros(())
        return policy, zero, value, 0.0, 0.0, 1.0, 0.0, 1.0


def _batch():
    return SimpleNamespace(
        condition={"state": torch.zeros(128, 576), "global_cond": torch.zeros(128, 576)},
        chains_prev=torch.zeros(128, 16, 11),
        chains_next=torch.zeros(128, 16, 11),
        denoising_inds=torch.zeros(128, dtype=torch.long),
        returns=torch.zeros(128),
        oldvalues=torch.zeros(128),
        advantages=torch.ones(128),
        oldlogprobs=torch.zeros(128, 16, 11),
    )


def _trainer():
    runtime = _Runtime()
    actor_optimizer = torch.optim.Adam([runtime.actor_parameter], lr=1e-2)
    critic_optimizer = torch.optim.Adam(runtime.critic.parameters(), lr=1e-2)
    trainer = DDPMTrainer(
        runtime=runtime,
        lora_bundle=_Bundle(runtime.actor_parameter),
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
    )
    return trainer, runtime


def test_trainer_calls_official_loss_with_e5_reward_slice():
    trainer, runtime = _trainer()
    metrics = trainer.update(_batch())
    assert runtime.last_kwargs == {
        "use_bc_loss": False,
        "reward_horizon": 4,
        "reward_start": 2,
    }
    assert metrics.actor_grad_norm > 0 and metrics.critic_grad_norm > 0
    assert metrics.actor_updated is True
    assert metrics.actor_learning_rate == pytest.approx(1e-2)
    assert metrics.critic_learning_rate == pytest.approx(1e-2)
    assert trainer.counters.committed_updates == 1


def test_critic_only_warmup_changes_critic_without_consuming_actor_optimizer_state():
    trainer, runtime = _trainer()
    before_actor = runtime.actor_parameter.detach().clone()
    before_critic = {
        key: value.clone() for key, value in runtime.critic.state_dict().items()
    }

    metrics = trainer.update(_batch(), update_actor=False)

    torch.testing.assert_close(runtime.actor_parameter, before_actor)
    assert any(
        not torch.equal(value, before_critic[key])
        for key, value in runtime.critic.state_dict().items()
    )
    assert trainer.actor_optimizer.state == {}
    assert metrics.actor_updated is False
    assert metrics.actor_grad_norm == pytest.approx(0.0)
    assert metrics.critic_grad_norm > 0
    assert metrics.total_loss == pytest.approx(0.5 * metrics.value_loss)
    assert trainer.counters.committed_updates == 1


def test_zero_actor_gradient_is_recoverable_without_committing_parameters():
    trainer, runtime = _trainer()
    with torch.no_grad():
        runtime.actor_parameter.fill_(1.0)
    before_critic = {
        key: value.clone() for key, value in runtime.critic.state_dict().items()
    }

    with pytest.raises(UpdateRejected, match="actor gradient is zero") as captured:
        trainer.update(_batch())

    assert captured.value.recoverable is True
    assert captured.value.reason == "zero_actor_gradient"
    for key, value in runtime.critic.state_dict().items():
        torch.testing.assert_close(value, before_critic[key])
    assert trainer.counters.committed_updates == 0
    assert trainer.counters.rejected_updates == 1


def test_late_nonfinite_failure_restores_parameters_and_optimizer_state():
    trainer, runtime = _trainer()
    before_actor = runtime.actor_parameter.detach().clone()
    before_critic = {key: value.clone() for key, value in runtime.critic.state_dict().items()}
    original_step = trainer.actor_optimizer.step

    def poisoned_step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        with torch.no_grad():
            runtime.actor_parameter.fill_(float("nan"))
        return result

    trainer.actor_optimizer.step = poisoned_step
    with pytest.raises(UpdateRejected, match="non-finite parameters"):
        trainer.update(_batch())

    torch.testing.assert_close(runtime.actor_parameter, before_actor)
    for key, value in runtime.critic.state_dict().items():
        torch.testing.assert_close(value, before_critic[key])
    assert trainer.counters.committed_updates == 0
    assert trainer.counters.rejected_updates == 1
