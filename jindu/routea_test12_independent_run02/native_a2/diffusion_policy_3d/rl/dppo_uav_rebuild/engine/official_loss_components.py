"""Unreduced official-DPPO loss components for global logical reduction."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class AdvantageStats:
    mean: torch.Tensor
    std: torch.Tensor
    lower_quantile: torch.Tensor
    upper_quantile: torch.Tensor

    @classmethod
    def from_tensor(cls, advantages, lower=0.0, upper=1.0):
        values = advantages.detach()
        mean = values.mean()
        std = values.std()
        normalized = (values - mean) / (std + 1e-8)
        return cls(
            mean=mean,
            std=std,
            lower_quantile=torch.quantile(normalized, float(lower)),
            upper_quantile=torch.quantile(normalized, float(upper)),
        )


@dataclass(frozen=True)
class OfficialLossComponents:
    policy: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor
    bc: torch.Tensor
    logratio: torch.Tensor
    ratio: torch.Tensor
    approx_kl: torch.Tensor
    clip_mask: torch.Tensor
    slice_diagnostics: dict = None


def _per_pair_eta(eta, pair_count, reference):
    value = torch.as_tensor(eta, device=reference.device, dtype=reference.dtype)
    if value.numel() == 1:
        return value.reshape(1).expand(pair_count)
    if value.shape[0] != pair_count:
        raise ValueError("eta must be scalar or have one row per pair")
    return value.reshape(pair_count, -1).mean(dim=1)


def compute_official_components(
    runtime,
    batch,
    stats: AdvantageStats,
    *,
    reward_start=2,
    reward_horizon=4,
    use_bc_loss=False,
):
    if use_bc_loss:
        raise NotImplementedError("logical BC component is disabled for Route A v1.7")
    actor_condition = {
        key: batch.condition[key]
        for key in ("state", "global_cond") if key in batch.condition
    }
    newlogprobs, eta = runtime.get_logprobs_subsample(
        actor_condition,
        batch.chains_prev,
        batch.chains_next,
        batch.denoising_inds,
        get_ent=True,
    )
    reward_end = reward_start + reward_horizon
    if reward_start != 2 or reward_horizon not in (4, 8) or reward_end > newlogprobs.shape[1]:
        raise ValueError("E5 logical loss requires A0 [2:6] or R1 [2:10]")
    if newlogprobs.shape != batch.oldlogprobs.shape:
        raise ValueError("new/old logprob tensors must pair every action index")
    new = newlogprobs.clamp(-5, 2)[:, reward_start:reward_end, :].mean(dim=(-1, -2))
    old = batch.oldlogprobs.clamp(-5, 2)[:, reward_start:reward_end, :].mean(dim=(-1, -2))
    advantages = (batch.advantages - stats.mean) / (stats.std + 1e-8)
    advantages = advantages.clamp(stats.lower_quantile, stats.upper_quantile)
    discount = torch.stack([
        torch.as_tensor(
            runtime.gamma_denoising ** (runtime.ft_denoising_steps - int(index) - 1),
            device=advantages.device,
            dtype=advantages.dtype,
        )
        for index in batch.denoising_inds
    ])
    advantages = advantages * discount
    logratio = new - old
    ratio = logratio.exp()
    t = batch.denoising_inds.float() / (runtime.ft_denoising_steps - 1)
    if runtime.ft_denoising_steps > 1:
        clip = runtime.clip_ploss_coef_base + (
            runtime.clip_ploss_coef - runtime.clip_ploss_coef_base
        ) * (torch.exp(runtime.clip_ploss_coef_rate * t) - 1) / (
            math.exp(runtime.clip_ploss_coef_rate) - 1
        )
    else:
        clip = t
    slice_diagnostics = None
    if reward_horizon == 8:
        with torch.no_grad():
            slice_diagnostics = {}
            for label, start, end in (("prefix", 2, 6), ("tail", 6, 10), ("full", 2, 10)):
                new_slice = newlogprobs.detach().clamp(-5, 2)[:, start:end, :]
                old_slice = batch.oldlogprobs.detach().clamp(-5, 2)[:, start:end, :]
                lr = new_slice.mean(dim=(-1, -2)) - old_slice.mean(dim=(-1, -2))
                rr = lr.exp()
                selected_actions = batch.chains_next.detach()[:, start:end, :]
                slice_diagnostics[label] = {
                    "logratio": lr,
                    "approx_kl": (rr - 1) - lr,
                    "clip_mask": ((rr - 1).abs() > clip),
                    "logprob_drift": (new_slice - old_slice).abs().mean(dim=(-1, -2)),
                    "action_variance": selected_actions.var(dim=0, unbiased=False).mean(),
                }
    policy = torch.maximum(
        -advantages * ratio,
        -advantages * torch.clamp(ratio, 1 - clip, 1 + clip),
    )
    newvalues = runtime.critic(batch.condition).reshape(-1)
    if runtime.clip_vloss_coef is not None:
        unclipped = (newvalues - batch.returns) ** 2
        clipped_values = batch.oldvalues + torch.clamp(
            newvalues - batch.oldvalues,
            -runtime.clip_vloss_coef,
            runtime.clip_vloss_coef,
        )
        value = torch.maximum(unclipped, (clipped_values - batch.returns) ** 2)
    else:
        value = (newvalues - batch.returns) ** 2
    entropy = -_per_pair_eta(eta, len(policy), policy)
    return OfficialLossComponents(
        policy=policy,
        value=value,
        entropy=entropy,
        bc=torch.zeros_like(policy),
        logratio=logratio,
        ratio=ratio,
        approx_kl=(ratio - 1) - logratio,
        clip_mask=((ratio - 1).abs() > clip),
        slice_diagnostics=slice_diagnostics,
    )


__all__ = (
    "AdvantageStats",
    "OfficialLossComponents",
    "compute_official_components",
)
