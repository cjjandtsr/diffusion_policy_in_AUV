"""DPPO-PARITY (§4.12 + G2 §5.4): synthetic fixture vs unmodified official
snapshot as oracle.

Oracle: PPODiffusion.loss() from third_party/dppo_official_cc7234a (commit
cc7234a), instantiated via object.__new__ (no __init__) with the numeric
config attributes set and model calls stubbed — the numeric core is
identical to the frozen official code. Expected values never come from the
project function itself.

G2 gates: clamped mean logprob rtol<=1e-5 atol<=1e-6; denoising discount
atol<=1e-7; epsilon_k atol<=1e-7; ratio rtol<=1e-5 atol<=1e-6; pg_loss
rtol<=1e-5 atol<=1e-6; approx_kl rtol<=1e-5 atol<=1e-6; clipfrac exact.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "third_party" / "dppo_official_cc7234a"))

from model.diffusion.diffusion_ppo import PPODiffusion as OfficialPPO  # noqa: E402

from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract  # noqa: E402
from diffusion_policy_3d.rl.dppo_uav.official_loss_adapter import (  # noqa: E402
    DPPOLossOutput,
    dppo_policy_loss,
)

B = 32
K_F = 5
H_A = 8
D_A = 11
CLIP_BASE = 1e-3
CLIP_FINAL = 0.2
CLIP_RATE = 3.0


@pytest.fixture(scope="module")
def fixture():
    """G2 synthetic fixture: fixed seed, boundary + clamp-out logprobs,
    advantage pos/neg/zero/outlier, denoising indices covering 0..4."""
    torch.manual_seed(0)
    new_logprobs = torch.randn(B, H_A, D_A) * 1.5
    old_logprobs = torch.randn(B, H_A, D_A) * 1.5
    # boundary / clamp-out values
    new_logprobs[0, 0, 0] = 5.0  # above clamp max 2
    old_logprobs[1] = -6.0  # below clamp min -5
    new_logprobs[2] = 2.0  # exactly at max
    old_logprobs[3] = -5.0  # exactly at min
    advantages = torch.randn(B)
    advantages[4] = 0.0
    advantages[5] = 50.0
    advantages[6] = -50.0
    denoising_inds = torch.cat(
        [torch.tensor([0, 1, 2, 3, 4]), torch.randint(0, K_F, (B - 5,))]
    )
    return new_logprobs, old_logprobs, advantages, denoising_inds


def _official_oracle(fx, contract):
    """Bare official instance with model calls stubbed; returns the same
    numeric pipeline the frozen code runs."""
    official = object.__new__(OfficialPPO)
    official.norm_adv = True
    official.clip_advantage_lower_quantile = 0.0
    official.clip_advantage_upper_quantile = 1.0
    official.gamma_denoising = contract.gamma_denoising
    official.ft_denoising_steps = contract.ft_denoising_steps
    official.clip_ploss_coef = CLIP_FINAL
    official.clip_ploss_coef_base = CLIP_BASE
    official.clip_ploss_coef_rate = CLIP_RATE
    official.device = torch.device("cpu")
    official.clip_vloss_coef = None
    official.entropy_fixed = False
    official.get_logprobs_subsample = (
        lambda obs, cp, cn, di, get_ent=True: (fx[0], torch.zeros(B))
    )
    official.critic = lambda obs: torch.zeros(B)
    # chain fixtures must satisfy the call; contents unused by the stubs
    chains_prev = torch.zeros(B, H_A, D_A)
    chains_next = torch.zeros(B, H_A, D_A)
    returns = torch.zeros(B)
    oldvalues = torch.zeros(B)
    args = dict(
        obs={"state": torch.zeros(B, 3, 9)},
        chains_prev=chains_prev,
        chains_next=chains_next,
        denoising_inds=fx[3],
        returns=returns,
        oldvalues=oldvalues,
        advantages=fx[2].clone(),  # official mutates advantages in place
        oldlogprobs=fx[1],
        use_bc_loss=False,
        reward_horizon=contract.exec_steps,
    )
    pg_loss, _ent, _v, clipfrac, approx_kl, ratio_mean, _bc, _eta = (
        official.loss(**args)
    )
    return pg_loss, clipfrac, approx_kl, ratio_mean


def _project(fx, contract):
    new, old, adv, inds = fx
    return dppo_policy_loss(
        new_logprobs=new,
        old_logprobs=old,
        advantages=adv,
        denoising_indices=inds,
        exec_steps=contract.exec_steps,
        ft_steps=contract.ft_denoising_steps,
        gamma_denoising=contract.gamma_denoising,
        clip_base=CLIP_BASE,
        clip_final=CLIP_FINAL,
        clip_rate=CLIP_RATE,
    )


# ---------- G2 gates ----------

def test_clamped_mean_logprob_parity(fixture):
    contract = RLSystemContract()
    out = _project(fixture, contract)
    new, old = fixture[0], fixture[1]
    # official lines: clamp(min=-5, max=2) -> [:, :exec_steps, :] -> mean((-1,-2))
    expected_new = new.clamp(min=-5, max=2)[:, : contract.exec_steps, :].mean(
        dim=(-1, -2)
    )
    expected_old = old.clamp(min=-5, max=2)[:, : contract.exec_steps, :].mean(
        dim=(-1, -2)
    )
    assert torch.allclose(
        out.clamped_new_mean, expected_new, rtol=1e-5, atol=1e-6
    )
    assert torch.allclose(
        out.clamped_old_mean, expected_old, rtol=1e-5, atol=1e-6
    )


def test_denoising_discount_parity(fixture):
    contract = RLSystemContract()
    out = _project(fixture, contract)
    inds = fixture[3]
    # official lines (diffusion_ppo.py): gamma ** (ft_steps - i - 1) per i
    expected = torch.tensor(
        [
            contract.gamma_denoising ** (contract.ft_denoising_steps - i - 1)
            for i in inds.tolist()
        ]
    )
    assert torch.allclose(out.discount, expected, rtol=0, atol=1e-7)


def test_epsilon_k_parity(fixture):
    contract = RLSystemContract()
    out = _project(fixture, contract)
    inds = fixture[3].float()
    # official lines (diffusion_ppo.py:145-152): exponential interpolation
    t = inds / (contract.ft_denoising_steps - 1)
    expected = CLIP_BASE + (CLIP_FINAL - CLIP_BASE) * (
        torch.exp(CLIP_RATE * t) - 1
    ) / (math.exp(CLIP_RATE) - 1)
    assert torch.allclose(out.epsilon, expected, rtol=0, atol=1e-7)


def test_pg_loss_parity(fixture):
    contract = RLSystemContract()
    oracle_pg, _cf, _kl, _rm = _official_oracle(fixture, contract)
    out = _project(fixture, contract)
    assert torch.allclose(out.pg_loss, oracle_pg, rtol=1e-5, atol=1e-6)


def test_approx_kl_parity(fixture):
    contract = RLSystemContract()
    _pg, _cf, oracle_kl, _rm = _official_oracle(fixture, contract)
    out = _project(fixture, contract)
    assert abs(out.approx_kl - oracle_kl) <= 1e-6 + 1e-5 * abs(oracle_kl)


def test_ratio_mean_parity(fixture):
    contract = RLSystemContract()
    _pg, _cf, _kl, oracle_ratio = _official_oracle(fixture, contract)
    out = _project(fixture, contract)
    assert abs(out.ratio_mean - oracle_ratio) <= 1e-6 + 1e-5 * abs(oracle_ratio)


def test_clipfrac_exact_parity(fixture):
    contract = RLSystemContract()
    _pg, oracle_clipfrac, _kl, _rm = _official_oracle(fixture, contract)
    out = _project(fixture, contract)
    assert out.clipfrac == oracle_clipfrac


# ---------- shape guards ----------

def test_logprob_shape_mismatch_rejected():
    contract = RLSystemContract()
    new = torch.zeros(B, H_A, D_A)
    old = torch.zeros(B, H_A, D_A - 1)  # 10 dims
    adv = torch.zeros(B)
    inds = torch.zeros(B, dtype=torch.long)
    with pytest.raises(AssertionError):
        dppo_policy_loss(
            new_logprobs=new, old_logprobs=old, advantages=adv,
            denoising_indices=inds, exec_steps=contract.exec_steps,
            ft_steps=contract.ft_denoising_steps,
            gamma_denoising=contract.gamma_denoising,
        )


def test_denoising_index_range_rejected():
    contract = RLSystemContract()
    new = torch.zeros(B, H_A, D_A)
    old = torch.zeros(B, H_A, D_A)
    adv = torch.zeros(B)
    inds = torch.full((B,), 5, dtype=torch.long)  # == ft_steps, out of range
    with pytest.raises(AssertionError):
        dppo_policy_loss(
            new_logprobs=new, old_logprobs=old, advantages=adv,
            denoising_indices=inds, exec_steps=contract.exec_steps,
            ft_steps=contract.ft_denoising_steps,
            gamma_denoising=contract.gamma_denoising,
        )


def test_batch_size_mismatch_rejected():
    contract = RLSystemContract()
    new = torch.zeros(B, H_A, D_A)
    old = torch.zeros(B, H_A, D_A)
    adv = torch.zeros(B - 1)  # advantages shorter than logprobs batch
    inds = torch.zeros(B, dtype=torch.long)
    with pytest.raises(AssertionError):
        dppo_policy_loss(
            new_logprobs=new, old_logprobs=old, advantages=adv,
            denoising_indices=inds, exec_steps=contract.exec_steps,
            ft_steps=contract.ft_denoising_steps,
            gamma_denoising=contract.gamma_denoising,
        )


def test_dppoloss_output_frozen():
    out = DPPOLossOutput(
        pg_loss=torch.zeros(()), approx_kl=0.0, clipfrac=0.0,
        ratio_mean=0.0, ratio_std=0.0, logratio_mean=0.0,
        clamped_new_mean=torch.zeros(B), clamped_old_mean=torch.zeros(B),
        discount=torch.zeros(B), epsilon=torch.zeros(B),
    )
    with pytest.raises(Exception):
        out.pg_loss = torch.ones(())  # type: ignore[misc]
