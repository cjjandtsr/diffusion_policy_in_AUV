"""ACT-OFFSET: policy action starts at full-trajectory index action_start=2.

Design §2.3 / §4.5: policy_action = full[:, a0:a0+H_a] (a0=2, H_a=8),
executed_action = policy_action[:, :A_e] with A_e = exec_steps = 4
(ASSUMPTION-4, EQUIV-REG OPEN pending T9 re-freeze).
"""

import torch
import pytest

from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.phase1a_actor_adapter import (
    split_trajectory,
    slice_executed_logprobs,

    DPPORolloutSample,
)

B, H_F, D_A = 2, 16, 11


def _sentinel_trajectory():
    """Channel 0 of each frame == its step index, so slicing is verifiable."""
    t = torch.arange(H_F).reshape(1, H_F, 1).float()
    return t.expand(B, H_F, D_A)  # (B, 16, 11)


def test_policy_action_region():
    """policy_action == full_trajectory[:, 2:10] (index sentinel)."""
    full = _sentinel_trajectory()
    contract = RLSystemContract()
    policy_action, executed_action = split_trajectory(full, contract)
    assert policy_action.shape == (B, contract.action_horizon, D_A)
    # sentinel: first policy step must be original index 2, not 0
    assert torch.all(policy_action[:, 0] == 2.0)
    assert torch.all(policy_action[:, -1] == 9.0)
    assert not torch.all(policy_action[:, 0] == 0.0)  # old offset bug


def test_executed_action_region():
    """executed_action == policy_action[:, :4] == full[:, 2:6]."""
    full = _sentinel_trajectory()
    contract = RLSystemContract()
    _, executed_action = split_trajectory(full, contract)
    assert executed_action.shape == (B, contract.exec_steps, D_A)
    assert torch.all(executed_action[:, 0] == 2.0)
    assert torch.all(executed_action[:, -1] == 5.0)


def test_executed_logprob_slice():
    """Design §4.6 option 2: policy-horizon logprobs sliced to exec region."""
    contract = RLSystemContract()
    K_f = contract.ft_denoising_steps
    policy_logprobs = torch.randn(B, K_f, contract.action_horizon, D_A)
    executed = slice_executed_logprobs(policy_logprobs, contract)
    assert executed.shape == (B, K_f, contract.exec_steps, D_A)
    assert torch.equal(executed, policy_logprobs[..., :4, :])


def test_denoising_indices_official_semantics():
    """R5.8-G (N10): the ambiguous descending denoising_indices() is
    deleted — the OFFICIAL optimizer axis is the ASCENDING
    optimizer_denoise_indices 0..K_f-1 (i=0 = noisiest fine-tune step;
    the loss adapter discounts it by gamma^(K_f-1-i))."""
    contract = RLSystemContract()
    idx = torch.arange(contract.ft_denoising_steps, dtype=torch.long)
    assert idx.shape == (contract.ft_denoising_steps,)
    assert torch.equal(idx, torch.arange(5, dtype=torch.long))


def test_denoising_indices_discount_direction():
    """γ^(K_f-1-i): i=0 (noisiest) gets the smallest discount —
    audit-corrected direction on the ASCENDING optimizer axis."""
    contract = RLSystemContract()
    idx = torch.arange(contract.ft_denoising_steps, dtype=torch.long)
    discounts = contract.gamma_denoising ** (contract.ft_denoising_steps - 1 - idx)
    assert float(discounts[0]) == pytest.approx(
        contract.gamma_denoising ** 4, rel=1e-6
    )
    assert float(discounts[-1]) == pytest.approx(1.0, rel=1e-6)


def test_dppo_rollout_sample_field_shapes():
    """Design §4.5 dataclass shape contract (B2 five-field taxonomy)."""
    contract = RLSystemContract()
    full = _sentinel_trajectory()
    policy_action, executed_action = split_trajectory(full, contract)
    sample = DPPORolloutSample(
        full_trajectory=full,
        policy_action=policy_action,
        executed_action=executed_action,
        chains_policy=torch.randn(
            B, contract.ft_denoising_steps + 1, contract.action_horizon, D_A
        ),
        old_logprobs=torch.randn(
            B, contract.ft_denoising_steps, contract.action_horizon, D_A
        ),
        obs_condition=torch.randn(B, contract.obs_steps * 192),
        # R5.8-G (N10): the three explicit index axes
        optimizer_denoise_indices=torch.arange(
            contract.ft_denoising_steps, dtype=torch.long),
        scheduler_timesteps=torch.arange(
            contract.denoising_steps - 1,
            contract.denoising_steps - 1 - contract.ft_denoising_steps,
            -1, dtype=torch.long,
        ),
        full_chain_indices=torch.arange(
            contract.denoising_steps - contract.ft_denoising_steps,
            contract.denoising_steps, dtype=torch.long,
        ),
    )
    assert sample.executed_action.shape[1] == contract.exec_steps
    # The mandatory pair is the region window (H_a), not the full horizon
    # — chains_policy/old_logprobs carry H_a.
    assert sample.chains_policy.shape[1] == contract.ft_denoising_steps + 1
    assert sample.chains_policy.shape[2] == contract.action_horizon
    assert sample.old_logprobs.shape[1] == contract.ft_denoising_steps
    assert sample.obs_condition.shape[1] == contract.obs_steps * 192
    assert (sample.optimizer_denoise_indices.shape[0]
            == contract.ft_denoising_steps)
    assert (sample.scheduler_timesteps.shape[0]
            == contract.ft_denoising_steps)
    assert (sample.full_chain_indices.shape[0]
            == contract.ft_denoising_steps)
