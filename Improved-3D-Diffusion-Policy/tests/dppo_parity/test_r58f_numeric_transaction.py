"""R5.8-F (N8/H-3/C-1): transactional numeric safety negative tests.

The three fault classes the update must catch BEFORE anything is
published: an Inf trainable parameter after the optimizer step, a NaN
kernel table, and an actor parameter that lost its gradient path (K5).

Run: pytest -q tests/dppo_parity/test_r58f_numeric_transaction.py
"""

import numpy as np
import pytest
import torch

from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.policy.dppo_transition_kernel import (
    DPPOTransitionKernel,
    build_ddim_tables,
)
from diffusion_policy_3d.rl.dppo_uav.trainer import (
    TrainerConfig,
    ppo_update,
)

from facade_testutil import make_test_context
from test_randperm_epoch_coverage import _marker_buffer  # shared fixture


class _CorruptingAdam(torch.optim.Adam):
    """Steps normally but first corrupts a target parameter with +Inf —
    the post-step transaction check must catch it."""

    def __init__(self, params, corrupt_param, lr=1e-3):
        super().__init__(params, lr=lr)
        self.corrupt_param = corrupt_param

    def step(self, *args, **kwargs):
        with torch.no_grad():
            self.corrupt_param.add_(float("inf"))
        super().step(*args, **kwargs)


def test_inf_param_after_step_hard_fails():
    """N8: an Inf trainable parameter after optimizer.step must raise
    BEFORE anything can be persisted (the exception is the transaction
    boundary — no checkpoint is published by the caller)."""
    contract = RLSystemContract()
    buf = _marker_buffer(contract)
    policy, facade, encode_cond, critic = make_test_context(contract)
    opt = _CorruptingAdam(
        list(policy.model.parameters()) + list(critic.parameters()),
        corrupt_param=next(iter(policy.model.parameters())),
    )
    with pytest.raises(FloatingPointError, match="after"):
        ppo_update(
            buffer=buf, contract=contract, facade=facade,
            encode_cond=encode_cond,
            actor_optimizer=opt,
            critic_optimizer=torch.optim.Adam(
                critic.parameters(), lr=1e-3),
            config=TrainerConfig(
                update_epochs=1, batch_size=16, max_grad_norm=None,
                target_kl=None, ent_coef=0.0, vf_coef=1.0,
            ),
        )


def test_nan_kernel_table_hard_fails():
    """H-3: a NaN alpha entry must be rejected by the kernel guard — the
    old alpha==1.0-only check let NaN through (NaN comparisons are
    False) and NaN actions reached the env."""
    ac = torch.linspace(0.9999, 0.9, 10)
    ac[3] = float("nan")
    timesteps = torch.arange(9, -1, -1)
    tables = build_ddim_tables(ac, timesteps)
    kernel = DPPOTransitionKernel(tables, ft_denoising_steps=5)
    x = torch.randn(2, 8, 11)
    eps = torch.randn(2, 8, 11)
    # timesteps = [9..0] -> table position 6 reads ac[3] (the NaN entry)
    index = torch.full((2,), 6, dtype=torch.long)
    with pytest.raises(ValueError, match="alpha"):
        kernel.transition_stats(
            x, eps, index,
            min_sampling_denoising_std=0.1,
            min_logprob_denoising_std=0.1,
        )


def test_gradless_actor_param_hard_fails():
    """C-1/K5: an actor parameter that accumulated no gradient over the
    epoch (a dropped LoRA hook / a detached path) must hard-fail — the
    old logprob-flow check could not see it."""
    contract = RLSystemContract()
    buf = _marker_buffer(contract)
    policy, facade, encode_cond, critic = make_test_context(contract)
    # a live, trainable parameter that is NOT in the actor's forward —
    # exactly the "hook dropped" shape
    dummy = torch.nn.Parameter(torch.zeros(1, requires_grad=True))
    opt = torch.optim.Adam(
        list(policy.model.parameters()) + [dummy], lr=1e-3)
    with pytest.raises(FloatingPointError, match="without gradient"):
        ppo_update(
            buffer=buf, contract=contract, facade=facade,
            encode_cond=encode_cond,
            actor_optimizer=opt,
            critic_optimizer=torch.optim.Adam(
                critic.parameters(), lr=1e-3),
            config=TrainerConfig(
                update_epochs=1, batch_size=16, max_grad_norm=None,
                target_kl=None, ent_coef=0.0, vf_coef=1.0,
            ),
        )
