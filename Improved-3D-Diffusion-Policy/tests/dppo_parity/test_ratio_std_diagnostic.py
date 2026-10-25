"""R1 rework #2 (audit 2026-08-08): B=1 ratio_std diagnostic fix.

The official-parity loss computes ``ratio_std`` with ``torch.std`` on the
ratio vector; at the B=1 likelihood recompute (microbatch=1 semantics) a
single element produces NaN + a UserWarning (degrees of freedom <= 0) —
the 21 std warnings in the sealed authoritative run. The fix: a
single-element ratio reports std = 0.0 (informational diagnostic) with NO
warning; loss, gradients, clip and KL are untouched; B>1 semantics match
the manual torch.std exactly.

Run: pytest -q tests/dppo_parity/test_ratio_std_diagnostic.py
"""

import math
import warnings

import pytest
import torch

from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.official_loss_adapter import dppo_policy_loss


def _loss_args(B: int, contract: RLSystemContract):
    """Deterministic per-batch logprobs/advantages — the tests assert
    semantics (zero / manual parity), not golden values."""
    torch.manual_seed(B)
    new = torch.randn(B, contract.exec_steps, contract.action_dim)
    old = torch.randn(B, contract.exec_steps, contract.action_dim)
    advantages = torch.randn(B)
    denoising_inds = torch.zeros(B, dtype=torch.long)
    return new, old, advantages, denoising_inds


def _loss(B: int, contract: RLSystemContract):
    new, old, adv, inds = _loss_args(B, contract)
    return dppo_policy_loss(
        new_logprobs=new,
        old_logprobs=old,
        advantages=adv,
        denoising_indices=inds,
        exec_steps=contract.exec_steps,
        ft_steps=contract.ft_denoising_steps,
        gamma_denoising=contract.gamma_denoising,
        clip_base=1e-3,
        clip_final=0.2,
        clip_rate=3.0,
        norm_adv=False,  # production B=1 recompute path (adv normalized at
        # the minibatch level once, then per-element recompute runs
        # norm_adv=False — the only NaN in the sealed run was ratio_std)
    )


def test_b1_ratio_std_is_zero_finite_no_warning():
    """B=1 single-element ratio std must be 0.0, finite, and warning-free —
    the audited diagnostic NaN (official_loss_adapter.py:120)."""
    contract = RLSystemContract()
    new, old, adv, inds = _loss_args(1, contract)
    # gradients must flow through the loss (fix must not touch the graph):
    # leaves need requires_grad BEFORE the forward builds it
    new.requires_grad_(True)
    old.requires_grad_(True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the gate
        out = dppo_policy_loss(
            new_logprobs=new, old_logprobs=old, advantages=adv,
            denoising_indices=inds, exec_steps=contract.exec_steps,
            ft_steps=contract.ft_denoising_steps,
            gamma_denoising=contract.gamma_denoising,
            clip_base=1e-3, clip_final=0.2, clip_rate=3.0,
            norm_adv=False,
        )
    assert out.ratio_std == 0.0
    assert math.isfinite(out.ratio_std)
    # loss / diagnostics stay finite at B=1
    assert torch.isfinite(out.pg_loss)
    assert math.isfinite(out.approx_kl)
    assert math.isfinite(out.clipfrac)
    out.pg_loss.backward()
    assert torch.isfinite(new.grad).all()
    assert torch.isfinite(old.grad).all()


def test_b2_ratio_std_matches_manual_std():
    """B>1 semantics unchanged: ratio_std equals the manual torch.std of
    the (clamped, executed-region) ratio — the diagnostic must not change
    the B>1 path the official-parity oracle compares against."""
    contract = RLSystemContract()
    new, old, adv, inds = _loss_args(2, contract)
    out = _loss(2, contract)
    ratio = (
        new.clamp(min=-5, max=2)[:, : contract.exec_steps, :]
        .mean(dim=(-1, -2))
        - old.clamp(min=-5, max=2)[:, : contract.exec_steps, :]
        .mean(dim=(-1, -2))
    ).exp()
    assert out.ratio_std == float(ratio.std().item())
    assert math.isfinite(out.ratio_std)
