"""Official DPPO policy loss as a pure function (design §4.12).

The 10-step order mirrors the frozen official snapshot
(third_party/dppo_official_cc7234a/model/diffusion/diffusion_ppo.py:93-162,
upstream commit cc7234a) line by line:

1. new/old.clamp(-5, 2)
2. [:, :exec_steps, :]              (F10 ghost-action mask)
3. .mean((-1, -2))
4. full-batch advantage normalization
5. advantage quantile clipping (defaults = official config: 0.0/1.0)
6. denoising discount gamma_denoising ** (ft_steps - 1 - i)
7. logratio / ratio
8. denoising-specific epsilon (exponential interpolation)
9. PPO clipped surrogate
10. approx KL (k3), clipfrac, ratio diagnostics

Numerical parity vs the unmodified official snapshot is enforced by
tests/dppo_parity/test_official_loss_parity.py (G2 gate, §5.4).
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class DPPOLossOutput:
    """Diagnostics returned by dppo_policy_loss (§4.12 step 10)."""

    pg_loss: torch.Tensor  # scalar
    approx_kl: float
    clipfrac: float
    ratio_mean: float
    ratio_std: float
    logratio_mean: float
    clamped_new_mean: torch.Tensor  # (B,)
    clamped_old_mean: torch.Tensor  # (B,)
    discount: torch.Tensor  # (B,)
    epsilon: torch.Tensor  # (B,)


def dppo_policy_loss(
    new_logprobs: torch.Tensor,  # (B, H_a, D_a), one selected denoise k per B
    old_logprobs: torch.Tensor,  # (B, H_a, D_a)
    advantages: torch.Tensor,  # (B,)
    denoising_indices: torch.Tensor,  # (B,) int in [0, ft_steps)
    exec_steps: int,
    ft_steps: int,
    gamma_denoising: float,
    clip_base: float = 1e-3,  # official clip_ploss_coef_base
    clip_final: float = 0.2,  # official clip_ploss_coef
    clip_rate: float = 3.0,  # official clip_ploss_coef_rate
    norm_adv: bool = True,
    clip_advantage_lower_quantile: float = 0.0,
    clip_advantage_upper_quantile: float = 1.0,
) -> DPPOLossOutput:
    """Official 10-step PPO loss (see module docstring)."""
    batch = new_logprobs.shape[0]
    assert old_logprobs.shape == new_logprobs.shape, (
        f"old_logprobs {old_logprobs.shape} != new_logprobs {new_logprobs.shape}"
    )
    assert advantages.shape == (batch,), (
        f"advantages must be ({batch},), got {advantages.shape}"
    )
    assert denoising_indices.shape == (batch,), (
        f"denoising_indices must be ({batch},), got {denoising_indices.shape}"
    )
    assert 1 <= exec_steps <= new_logprobs.shape[1], (
        f"exec_steps {exec_steps} must satisfy 1 <= exec_steps <="
        f" {new_logprobs.shape[1]}"
    )
    assert ft_steps >= 1, f"ft_steps must be >= 1, got {ft_steps}"
    assert denoising_indices.min() >= 0 and denoising_indices.max() < ft_steps, (
        f"denoising_indices must be in [0, {ft_steps}), got"
        f" [{denoising_indices.min()}, {denoising_indices.max()}]"
    )

    # 1. clamp
    new = new_logprobs.clamp(min=-5, max=2)
    old = old_logprobs.clamp(min=-5, max=2)
    # 2. executed-region mask (F10 ghost action)
    new = new[:, :exec_steps, :]
    old = old[:, :exec_steps, :]
    # 3. mean over (h, d)
    new_mean = new.mean(dim=(-1, -2))
    old_mean = old.mean(dim=(-1, -2))
    # 4. full-batch advantage normalization
    if norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    # 5. advantage quantile clipping (official config defaults: identity)
    adv_min = torch.quantile(advantages, clip_advantage_lower_quantile)
    adv_max = torch.quantile(advantages, clip_advantage_upper_quantile)
    advantages = advantages.clamp(min=adv_min, max=adv_max)
    # 6. denoising discount (official: gamma ** (ft_steps - i - 1) per i)
    discount = gamma_denoising ** (ft_steps - 1 - denoising_indices)
    advantages = advantages * discount
    # 7. logratio / ratio
    logratio = new_mean - old_mean
    ratio = logratio.exp()
    # 8. denoising-specific epsilon (official exponential interpolation)
    t = denoising_indices.float() / (ft_steps - 1)
    if ft_steps > 1:
        epsilon = clip_base + (clip_final - clip_base) * (
            torch.exp(clip_rate * t) - 1
        ) / (math.exp(clip_rate) - 1)
    else:
        epsilon = t  # official quirk for ft_steps == 1, kept for parity
    # 9. PPO clipped surrogate
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
    # 10. diagnostics
    with torch.no_grad():
        approx_kl = ((ratio - 1) - logratio).mean().item()  # k3 estimator
        clipfrac = ((ratio - 1.0).abs() > epsilon).float().mean().item()
        ratio_mean = float(ratio.mean().item())
        # Rework #2 (audit 2026-08-08): the B=1 microbatch recompute makes
        # ratio a single element — torch.std then emits a ddof warning and
        # returns NaN (the 21 warnings in the sealed authority run). The
        # std is an informational diagnostic; a single element has zero
        # spread by definition, so report 0.0 without the warning. Loss,
        # gradients, clip and KL are untouched (this block is no_grad).
        ratio_std = 0.0 if ratio.numel() == 1 else float(ratio.std().item())
        logratio_mean = float(logratio.mean().item())
    return DPPOLossOutput(
        pg_loss=pg_loss,
        approx_kl=approx_kl,
        clipfrac=clipfrac,
        ratio_mean=ratio_mean,
        ratio_std=ratio_std,
        logratio_mean=logratio_mean,
        clamped_new_mean=new_mean.detach(),
        clamped_old_mean=old_mean.detach(),
        discount=discount.detach(),
        epsilon=epsilon.detach(),
    )


@dataclass(frozen=True)
class DPPOValueLossOutput:
    """Value-loss branch of the official loss (§4.12, diffusion_ppo.py:177-189)."""

    v_loss: torch.Tensor  # scalar


def dppo_value_loss(
    newvalues: torch.Tensor,  # (B,) critic output
    oldvalues: torch.Tensor,  # (B,)
    returns: torch.Tensor,  # (B,)
    clip_vloss_coef: Optional[float],  # None = no clipping (official default)
) -> DPPOValueLossOutput:
    """Official value-loss branch mirrored line by line.

    Official (diffusion_ppo.py:177-189):
      newvalues = critic(obs).view(-1)
      if clip_vloss_coef is not None:
          v_loss_unclipped = (newvalues - returns)**2
          v_clipped = oldvalues + clamp(newvalues-oldvalues, -c, c)
          v_loss_clipped = (v_clipped - returns)**2
          v_loss = 0.5 * max(v_loss_unclipped, v_loss_clipped).mean()
      else:
          v_loss = 0.5 * ((newvalues - returns)**2).mean()
    """
    batch = newvalues.shape[0]
    assert oldvalues.shape == (batch,), (
        f"oldvalues {oldvalues.shape} != newvalues {newvalues.shape}"
    )
    assert returns.shape == (batch,), (
        f"returns {returns.shape} != newvalues {newvalues.shape}"
    )
    if clip_vloss_coef is None:
        v_loss = 0.5 * ((newvalues - returns) ** 2).mean()
    else:
        v_loss_unclipped = (newvalues - returns) ** 2
        v_clipped = oldvalues + torch.clamp(
            newvalues - oldvalues, -clip_vloss_coef, clip_vloss_coef
        )
        v_loss_clipped = (v_clipped - returns) ** 2
        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
    return DPPOValueLossOutput(v_loss=v_loss)
