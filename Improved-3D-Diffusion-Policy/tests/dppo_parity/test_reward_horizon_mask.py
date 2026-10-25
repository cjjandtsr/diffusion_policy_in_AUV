"""DPPO-PARITY F10 ghost action (§4.12 step 2 + §4.15 hard failure):
only the executed region [0:exec_steps) enters the PPO ratio.

§4.15 lists "unexecuted action 产生 PPO gradient" as a hard failure —
the adapter must make the unexecuted region literally absent from the
computation (zero gradient, exact output invariance).
"""

import torch

from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.official_loss_adapter import dppo_policy_loss

B = 8
H_A = 8
D_A = 11


def _inputs(contract, requires_grad=False):
    torch.manual_seed(3)
    new = torch.randn(B, H_A, D_A) * 1.5
    if requires_grad:
        new.requires_grad_(True)
    old = torch.randn(B, H_A, D_A) * 1.5
    adv = torch.randn(B)
    inds = torch.randint(0, contract.ft_denoising_steps, (B,))
    return new, old, adv, inds


def test_unexecuted_region_exact_invariance():
    """Scrambling new_logprobs[:, exec_steps:, :] changes NOTHING — the
    unexecuted region never enters the loss (exact equality, not tolerance)."""
    contract = RLSystemContract()
    new, old, adv, inds = _inputs(contract)
    base = dppo_policy_loss(
        new_logprobs=new, old_logprobs=old, advantages=adv,
        denoising_indices=inds, exec_steps=contract.exec_steps,
        ft_steps=contract.ft_denoising_steps,
        gamma_denoising=contract.gamma_denoising,
    )
    scrambled = new.clone()
    scrambled[:, contract.exec_steps:, :] = torch.randn(
        B, H_A - contract.exec_steps, D_A
    )
    alt = dppo_policy_loss(
        new_logprobs=scrambled, old_logprobs=old, advantages=adv,
        denoising_indices=inds, exec_steps=contract.exec_steps,
        ft_steps=contract.ft_denoising_steps,
        gamma_denoising=contract.gamma_denoising,
    )
    assert base.pg_loss.item() == alt.pg_loss.item()
    assert base.approx_kl == alt.approx_kl
    assert base.clipfrac == alt.clipfrac


def test_unexecuted_region_zero_gradient():
    """F10: gradient wrt new_logprobs[:, exec_steps:, :] is exactly zero;
    the executed region carries the learning signal."""
    contract = RLSystemContract()
    new, old, adv, inds = _inputs(contract, requires_grad=True)
    out = dppo_policy_loss(
        new_logprobs=new, old_logprobs=old, advantages=adv,
        denoising_indices=inds, exec_steps=contract.exec_steps,
        ft_steps=contract.ft_denoising_steps,
        gamma_denoising=contract.gamma_denoising,
    )
    out.pg_loss.backward()
    grad = new.grad
    assert grad is not None
    executed = grad[:, : contract.exec_steps, :]
    unexecuted = grad[:, contract.exec_steps :, :]
    assert executed.abs().sum() > 0.0, "executed region must carry gradient"
    assert unexecuted.abs().sum() == 0.0, (
        "unexecuted action produced PPO gradient — §4.15 hard failure"
    )


def test_old_logprobs_also_horizon_masked():
    """The executed-only mask applies symmetrically to old_logprobs."""
    contract = RLSystemContract()
    new, old, adv, inds = _inputs(contract)
    base = dppo_policy_loss(
        new_logprobs=new, old_logprobs=old, advantages=adv,
        denoising_indices=inds, exec_steps=contract.exec_steps,
        ft_steps=contract.ft_denoising_steps,
        gamma_denoising=contract.gamma_denoising,
    )
    scrambled_old = old.clone()
    scrambled_old[:, contract.exec_steps:, :] = torch.randn(
        B, H_A - contract.exec_steps, D_A
    )
    alt = dppo_policy_loss(
        new_logprobs=new, old_logprobs=scrambled_old, advantages=adv,
        denoising_indices=inds, exec_steps=contract.exec_steps,
        ft_steps=contract.ft_denoising_steps,
        gamma_denoising=contract.gamma_denoising,
    )
    assert base.pg_loss.item() == alt.pg_loss.item()


def test_exec_steps_parameter_widens_mask():
    """exec_steps=8 (no mask) vs exec_steps=4: with 8 the whole chunk is in
    the ratio, so a scramble in [4:8) MUST change the loss."""
    contract = RLSystemContract()
    new, old, adv, inds = _inputs(contract)
    full = dppo_policy_loss(
        new_logprobs=new, old_logprobs=old, advantages=adv,
        denoising_indices=inds, exec_steps=contract.action_horizon,
        ft_steps=contract.ft_denoising_steps,
        gamma_denoising=contract.gamma_denoising,
    )
    scrambled = new.clone()
    scrambled[:, contract.exec_steps:, :] = torch.randn(
        B, H_A - contract.exec_steps, D_A
    )
    full_scrambled = dppo_policy_loss(
        new_logprobs=scrambled, old_logprobs=old, advantages=adv,
        denoising_indices=inds, exec_steps=contract.action_horizon,
        ft_steps=contract.ft_denoising_steps,
        gamma_denoising=contract.gamma_denoising,
    )
    assert full.pg_loss.item() != full_scrambled.pg_loss.item()
