"""DPPO-PARITY (§4.12 + G2 §5.4): value-loss branch of the official
PPODiffusion.loss() (diffusion_ppo.py:177-189) as a pure function.

Oracle: unmodified official snapshot, object.__new__ + stubbed
get_logprobs_subsample / critic (critic returns a fixed vector so the
v_loss path is exercised against oracle-computed values).

G2 gates: v_loss rtol<=1e-5 atol<=1e-6; clipped branch exact structure
(unclipped vs clipped max) verified against oracle.
"""

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "third_party" / "dppo_official_cc7234a"))

from model.diffusion.diffusion_ppo import PPODiffusion as OfficialPPO  # noqa: E402

from diffusion_policy_3d.rl.dppo_uav.official_loss_adapter import (  # noqa: E402
    dppo_value_loss,
)

B = 32
CLIP_VLOSS = 0.2


def _oracle_values(fx, clip_vloss_coef):
    """Run the official v_loss branch with a stubbed critic.

    Returns (v_loss, newvalues) computed by the unmodified official code.
    """
    new, old, ret = fx
    official = object.__new__(OfficialPPO)
    official.norm_adv = True
    official.clip_advantage_lower_quantile = 0.0
    official.clip_advantage_upper_quantile = 1.0
    official.gamma_denoising = 0.8
    official.ft_denoising_steps = 5
    official.clip_ploss_coef = 0.2
    official.clip_ploss_coef_base = 1e-3
    official.clip_ploss_coef_rate = 3.0
    official.device = torch.device("cpu")
    official.clip_vloss_coef = clip_vloss_coef
    official.entropy_fixed = False
    official.get_logprobs_subsample = (
        lambda obs, cp, cn, di, get_ent=True: (torch.zeros(B, 8, 11), torch.zeros(B))
    )
    official.critic = lambda obs: fx[0]  # newvalues = fx[0]
    chains_prev = torch.zeros(B, 8, 11)
    chains_next = torch.zeros(B, 8, 11)
    returns = ret
    oldvalues = old
    args = dict(
        obs={"state": torch.zeros(B, 3, 9)},
        chains_prev=chains_prev,
        chains_next=chains_next,
        denoising_inds=torch.zeros(B, dtype=torch.long),
        returns=returns,
        oldvalues=oldvalues,
        advantages=torch.zeros(B),
        # official call site selects per denoising index already
        # (train_ppo_diffusion_agent.py:314 logprobs_b = logprobs_k[
        #  batch_inds_b, denoising_inds_b]) -> (B, Ta, Da)
        oldlogprobs=torch.zeros(B, 8, 11),
        use_bc_loss=False,
        reward_horizon=4,
    )
    _pg, _ent, v_loss, _cf, _kl, _rm, _bc, _eta = official.loss(**args)
    return v_loss, fx[0]


def test_value_loss_parity_unclipped():
    torch.manual_seed(0)
    newvalues = torch.randn(B) * 2.0  # critic output
    oldvalues = torch.randn(B)
    returns = torch.randn(B) * 2.0
    fx = (newvalues, oldvalues, returns)
    oracle_v, _nv = _oracle_values(fx, None)  # official: no clipping
    out = dppo_value_loss(
        newvalues=fx[0], oldvalues=fx[1], returns=fx[2],
        clip_vloss_coef=None,
    )
    # official: v_loss = 0.5 * mean((newvalues - returns)^2)
    assert torch.allclose(out.v_loss, oracle_v, rtol=1e-5, atol=1e-6)


def test_value_loss_parity_clipped():
    torch.manual_seed(1)
    newvalues = torch.randn(B) * 3.0  # wide spread -> clipping engaged
    oldvalues = torch.randn(B) * 0.5
    returns = torch.randn(B) * 3.0
    fx = (newvalues, oldvalues, returns)
    oracle_v, _nv = _oracle_values(fx, CLIP_VLOSS)
    out = dppo_value_loss(
        newvalues=fx[0], oldvalues=fx[1], returns=fx[2],
        clip_vloss_coef=CLIP_VLOSS,
    )
    assert torch.allclose(out.v_loss, oracle_v, rtol=1e-5, atol=1e-6)


def test_value_loss_clip_branch_engaged():
    """Official clip-vloss is an ELEMENTWISE max(unclipped, clipped), not a
    min. With |newvalues - oldvalues| beyond the window AND returns beyond
    newvalues, the clipped term is the LARGER one and drives the loss —
    clipping bounds value predictions against overshooting returns."""
    newvalues = torch.full((B,), 3.0)
    oldvalues = torch.zeros(B)
    returns = torch.full((B,), 3.5)
    out = dppo_value_loss(
        newvalues=newvalues, oldvalues=oldvalues, returns=returns,
        clip_vloss_coef=0.2,
    )
    unclipped_elem = (newvalues - returns) ** 2  # 0.25 per element
    clipped_elem = (
        torch.clamp(newvalues - oldvalues, -0.2, 0.2) - returns
    ) ** 2  # (0.2 - 3.5)^2 = 10.89 per element
    expected = 0.5 * torch.max(unclipped_elem, clipped_elem).mean()
    assert torch.allclose(out.v_loss, expected, rtol=1e-6)
    assert out.v_loss.item() > (0.5 * unclipped_elem.mean()).item()


def test_value_loss_frozen_output():
    out = dppo_value_loss(
        newvalues=torch.zeros(B), oldvalues=torch.zeros(B),
        returns=torch.zeros(B), clip_vloss_coef=None,
    )
    with pytest.raises(Exception):
        out.v_loss = torch.ones(())  # type: ignore[misc]


def test_value_loss_shape_mismatch_rejected():
    with pytest.raises(AssertionError):
        dppo_value_loss(
            newvalues=torch.zeros(B), oldvalues=torch.zeros(B - 1),
            returns=torch.zeros(B), clip_vloss_coef=None,
        )
