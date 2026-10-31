"""Per-dimension per-step denoising logprobs (design §4.12; buffer invariant 5).

Thin chain-window / batch adapter over the FROZEN official DPPO runtime
(guide R7-2B Step 7): per-dim and per-step logprobs delegate to the
``OfficialPhase1aFacade``'s vendor methods — this module owns NO
Gaussian/DDIM math of its own (the project transition math that used to
live here, ``dppo_transition_kernel``, remains importable for parity and
historical scripts but is no longer on the canonical path). The
rollout's stored old logprobs and the trainer's new-logprob recompute
both come through here, so old/new share the identical vendor runtime.

Window semantics: a chain window (B, ft+1, R, Da) holds the last ft+1
steps of the full K-step chain; window step i maps to full step
k = K - ft + i.
"""

from typing import Optional

import torch
import torch.nn as nn

from diffusion_policy_3d.rl.dppo_uav.official_runtime import (
    OfficialPhase1aFacade,
    build_official_phase1a_facade,
    make_official_cond,
)
from diffusion_policy_3d.rl.dppo_uav.vendor_loader import (  # noqa: F401
    load_official_dppo_symbols as _load_official_dppo_symbols,
)


def _official_facade(
    *,
    model: nn.Module,
    scheduler,
    ft_denoising_steps: int,
    cfg_lambda: float,
    device,
    horizon_steps: int,
    action_dim: int,
    eta: float,
    min_sampling_denoising_std: float,
    min_logprob_denoising_std: float,
) -> OfficialPhase1aFacade:
    """Build the official facade for one logprob call (symbols cached)."""
    return build_official_phase1a_facade(
        model=model,
        scheduler=scheduler,
        ft_denoising_steps=ft_denoising_steps,
        cfg_lambda=cfg_lambda,
        device=device,
        horizon_steps=horizon_steps,
        action_dim=action_dim,
        eta=eta,
        min_sampling_denoising_std=min_sampling_denoising_std,
        min_logprob_denoising_std=min_logprob_denoising_std,
    )


def _check_timesteps_match(timesteps: torch.Tensor, scheduler) -> None:
    """The official facade derives its DDIM tables from the scheduler's
    timesteps; a caller-supplied timesteps that disagrees would silently
    mis-map steps — fail-closed instead."""
    if not torch.equal(timesteps.cpu(), scheduler.timesteps.cpu()):
        raise RuntimeError(
            "timesteps must equal scheduler.timesteps — the official "
            "facade derives its DDIM tables from the scheduler"
        )


def per_step_denoising_logp(
    x_k: torch.Tensor,  # (B, R, Da)
    x_next: torch.Tensor,  # (B, R, Da)
    k_batch: torch.Tensor,  # (B,) long — full-chain step per element
    timesteps: torch.Tensor,  # (K,) scheduler timesteps
    model: nn.Module,
    global_cond: torch.Tensor,  # (B, D) encoded condition
    global_cond_uncond: Optional[torch.Tensor],
    cfg_lambda: float,
    scheduler,
    eta: float,
    min_sampling_denoising_std: float,
    min_logprob_denoising_std: float,
    ft_denoising_steps: int,
) -> torch.Tensor:
    """Per-dim per-step logp for one denoising transition -> (B, R, Da),
    computed by the official vendor runtime under the LOGPROB std floor.

    ``k_batch`` holds FULL-chain step indices; the official likelihood
    window covers the last ``ft`` transitions, so each index maps to
    ``rel = k - (K - ft)`` in ``[0, ft)`` — out-of-range fails closed.
    EVERY transition — including the last one into valid timestep 0 —
    is a real stochastic transition (no terminal branch)."""
    _check_timesteps_match(timesteps, scheduler)
    device = x_k.device
    K = int(timesteps.shape[0])
    ft = int(ft_denoising_steps)
    rel = k_batch.long().to(device) - (K - ft)
    if rel.min().item() < 0 or rel.max().item() >= ft:
        raise RuntimeError(
            f"k_batch outside the official fine-tune window [K-ft, K): "
            f"indices in [{k_batch.min().item()}, {k_batch.max().item()}] "
            f"with K={K}, ft={ft} — the official likelihood covers only "
            "the last ft transitions"
        )
    facade = _official_facade(
        model=model, scheduler=scheduler, ft_denoising_steps=ft,
        cfg_lambda=cfg_lambda, device=device,
        horizon_steps=x_k.shape[1], action_dim=x_k.shape[2], eta=eta,
        min_sampling_denoising_std=min_sampling_denoising_std,
        min_logprob_denoising_std=min_logprob_denoising_std,
    )
    cond = make_official_cond(global_cond, global_cond_uncond, device)
    return facade.get_logprobs_subsample(cond, x_k, x_next, rel)


def selected_step_logprobs(
    logps: torch.Tensor,  # (B, K_f, H_a, 11)
    denoising_inds: torch.Tensor,  # (B,) long — optimizer denoise index
) -> torch.Tensor:
    """Gather the per-element selected denoising step -> (B, H_a, 11)."""
    return logps[
        torch.arange(logps.shape[0], device=logps.device), denoising_inds
    ]


def per_dim_denoising_logprobs(
    chain: torch.Tensor,  # (B, ft+1, R, Da) region window
    timesteps: torch.Tensor,  # (K,) scheduler timesteps
    model: nn.Module,
    global_cond: torch.Tensor,
    global_cond_uncond: Optional[torch.Tensor],
    cfg_lambda: float,
    scheduler,
    eta: float,
    min_sampling_denoising_std: float,
    min_logprob_denoising_std: float,
    ft_denoising_steps: int,
) -> torch.Tensor:
    """Per-dim logps over the fine-tune window -> (B, ft, R, Da).

    Window step i -> full-chain step k = K - ft + i -> official denoise
    index i. Loops the window calling the SAME official
    ``get_logprobs_subsample`` the recompute path (``per_step_denoising_logp``)
    uses per element, so rollout-stored and recomputed per-(b, k) values
    are bitwise identical at the behavior batch (B=1 identity gate)."""
    _check_timesteps_match(timesteps, scheduler)
    device = chain.device
    ft = int(ft_denoising_steps)
    facade = _official_facade(
        model=model, scheduler=scheduler, ft_denoising_steps=ft,
        cfg_lambda=cfg_lambda, device=device,
        horizon_steps=chain.shape[2], action_dim=chain.shape[3], eta=eta,
        min_sampling_denoising_std=min_sampling_denoising_std,
        min_logprob_denoising_std=min_logprob_denoising_std,
    )
    cond = make_official_cond(global_cond, global_cond_uncond, device)
    outs = []
    for i in range(ft):
        rel = torch.full(
            (chain.shape[0],), i, dtype=torch.long, device=device
        )
        outs.append(
            facade.get_logprobs_subsample(
                cond, chain[:, i], chain[:, i + 1], rel
            )
        )
    return torch.stack(outs, dim=1)  # (B, ft, R, Da)
