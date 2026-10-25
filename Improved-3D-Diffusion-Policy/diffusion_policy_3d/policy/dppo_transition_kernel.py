"""Official DPPO DDIM transition kernel (review R5.1 / §4.1.5).

The ONLY place in the project that implements the DDIM stochastic
transition math. Rollout sampling, stored old logprobs and the PPO
update's new-logprob recompute must all go through this kernel — no
second copy of the Gaussian/DDIM formulas anywhere (K6 single-kernel).

R7-2B Step 7-8: the CANONICAL train→rollout→recompute→loss path now
executes the frozen official vendor runtime via
``official_runtime.OfficialPhase1aFacade`` — this module is the
parity/historical home of the project kernel: it stays importable for
the parity tests and the legacy E-series shims below, but the canonical
caller count is 0 (guide Step 8). The legacy
``ddim_step_with_logprob`` / ``compute_per_step_logp_from_chain`` /
``compute_joint_logp_from_chain`` shims moved here verbatim from
dp_sampler (R7-2B) so the frozen historical E-series scripts keep a
home for their math.

Semantics follow the frozen official DPPO vendor
(``third_party/dppo_official_cc7234a/model/diffusion/diffusion.py``
``p_mean_var`` DDIM branch + ``diffusion_vpg.py`` sampling/logprob std
floors), for eta=1:

- tables are indexed by DDIM index (NOT by scheduler timestep): a
  transition whose target is the VALID timestep 0 is a real transition —
  ``alpha_prev=1.0`` makes its raw sigma ~1e-10, which the sampling and
  logprob std floors lift to the configured minima, exactly like the
  vendor. There is NO ``t_prev > 0`` terminal branch: loop termination is
  decided by the reverse-loop orchestration (dp_sampler), never by the
  distribution.
- mean / raw_std follow vendor p_mean_var: x_recon from epsilon,
  sigma = eta * sqrt((1-a_prev)/(1-a) * (1 - a/a_prev)).clamp(min=1e-10),
  mu = sqrt(a_prev) * x_recon + sqrt(1-a_prev-sigma^2).clamp(min=0) * eps.
- sampling_std = max(raw_std, min_sampling_denoising_std);
  logprob_std = max(raw_std, min_logprob_denoising_std) — the two floors
  are independent (vendor diffusion_vpg.py:276-297 / 380-396).
- sample() takes an EXPLICIT noise tensor (vendor parity + paired-RNG
  requirements) and clamps it to +-randn_clip_value (vendor default 10).
- log_prob() returns per-dimension logprob of shape (B, R, Da) — never
  summed (the PPO loss needs per-element values).
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal


@dataclass(frozen=True)
class DDIMTables:
    """Per-index DDIM tables (vendor diffusion.py:164-193 construction).

    ``ddim_t`` is the DESCENDING scheduler-timestep schedule (including
    timestep 0 as the last element); ``alpha_prev[i]`` is
    alphas_cumprod[ddim_t[i+1]] with the last index set to exactly 1.0.
    """

    ddim_t: torch.Tensor  # (S,) long, descending
    alpha: torch.Tensor  # (S,) float32
    alpha_prev: torch.Tensor  # (S,) float32
    sqrt_one_minus_alpha: torch.Tensor  # (S,) float32

    @property
    def ddim_steps(self) -> int:
        return int(self.ddim_t.shape[0])


def build_ddim_tables(
    alphas_cumprod: torch.Tensor, ddim_t: torch.Tensor,
) -> DDIMTables:
    """Build the vendor-rule per-index tables from a scheduler's
    cumulative alphas and the descending timestep schedule."""
    ddim_t = ddim_t.long()
    alpha = alphas_cumprod[ddim_t].clone().to(torch.float32)
    alpha_prev = torch.cat(
        [
            alphas_cumprod[ddim_t[1:]].clone().to(torch.float32),
            torch.tensor([1.0], dtype=torch.float32,
                         device=alphas_cumprod.device),
        ]
    )
    sqrt_one_minus_alpha = (1.0 - alpha) ** 0.5
    return DDIMTables(
        ddim_t=ddim_t,
        alpha=alpha,
        alpha_prev=alpha_prev,
        sqrt_one_minus_alpha=sqrt_one_minus_alpha,
    )


@dataclass(frozen=True)
class TransitionStats:
    """One DDIM transition's distribution (vendor p_mean_var semantics)."""

    mean: torch.Tensor  # (B, R, Da)
    raw_std: torch.Tensor  # (B, 1, 1) per-dim sigma (eta=1)
    sampling_std: torch.Tensor  # (B, 1, 1)
    logprob_std: torch.Tensor  # (B, 1, 1)
    scheduler_timestep: torch.Tensor  # (B,) long
    optimizer_denoise_index: torch.Tensor  # (B,) long


class DPPOTransitionKernel(nn.Module):
    """Official DPPO DDIM stochastic transition distribution (eta=1).

    ``ft_denoising_steps`` maps full-chain ddim indices to optimizer
    denoise indices (0..K_f-1 over the LAST K_f transitions); non-FT
    indices get negative optimizer indices (sampling-only steps).
    """

    def __init__(
        self,
        tables: DDIMTables,
        ft_denoising_steps: int,
        eta: float = 1.0,
        denoised_clip_value: float = None,
        eps_clip_value: float = None,
        randn_clip_value: float = 10.0,
    ) -> None:
        super().__init__()
        self.register_buffer("ddim_t", tables.ddim_t, persistent=False)
        self.register_buffer("alpha", tables.alpha, persistent=False)
        self.register_buffer("alpha_prev", tables.alpha_prev,
                             persistent=False)
        self.register_buffer("sqrt_one_minus_alpha",
                             tables.sqrt_one_minus_alpha, persistent=False)
        self.ft_denoising_steps = int(ft_denoising_steps)
        self.eta = float(eta)
        self.denoised_clip_value = denoised_clip_value
        self.eps_clip_value = eps_clip_value
        self.randn_clip_value = float(randn_clip_value)

    @property
    def ddim_steps(self) -> int:
        return int(self.ddim_t.shape[0])

    def _gather(self, index: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
        """(B,) long index -> (B, 1, 1) broadcastable table row. The
        tables live on the scheduler's device (CPU) — index tensors may
        be CUDA — so the table is moved to the index device BEFORE the
        gather (a CUDA index on a CPU table would otherwise raise)."""
        return table.to(index.device)[index][:, None, None]

    def transition_stats(
        self,
        x_t: torch.Tensor,  # (B, R, Da)
        eps: torch.Tensor,  # (B, R, Da) model epsilon output
        ddim_index: torch.Tensor,  # (B,) long full-chain index
        min_sampling_denoising_std: float,
        min_logprob_denoising_std: float,
    ) -> TransitionStats:
        """Vendor p_mean_var (DDIM branch) + the two independent std
        floors. No terminal branch: every index, including the last one
        into timestep 0, is a real stochastic transition."""
        index = ddim_index.long()
        alpha = self._gather(index, self.alpha)
        # review partA P2-3: alpha==1.0 would make the sigma radicand
        # 0/0 -> NaN that the 1e-10 clamp cannot rescue. Production is
        # safe only because the loader drops set_alpha_to_one — make the
        # invariant explicit instead of accidental.
        if not torch.isfinite(alpha).all() or (
            float((1.0 - alpha).min().item()) <= 0.0
        ):
            raise ValueError(
                "DDIM kernel: alpha == 1.0 encountered — the transition "
                "radicand (1-alpha_prev)/(1-alpha) is undefined; check "
                "the scheduler alphas_cumprod configuration"
            )
        alpha_prev = self._gather(index, self.alpha_prev)
        sqrt_one_minus_alpha = self._gather(index,
                                            self.sqrt_one_minus_alpha)

        x_recon = (x_t - sqrt_one_minus_alpha * eps) / (alpha ** 0.5)
        if self.denoised_clip_value is not None:
            x_recon = x_recon.clamp(-self.denoised_clip_value,
                                    self.denoised_clip_value)
            # vendor: re-derive eps from the clamped x_recon (DDIM branch)
            eps = (x_t - alpha ** 0.5 * x_recon) / sqrt_one_minus_alpha
        if self.eps_clip_value is not None:
            eps = eps.clamp(-self.eps_clip_value, self.eps_clip_value)

        # eta=1 per-dim sigma: (1-a_prev)/(1-a) * (1-a/a_prev), >= 1e-10
        sigma = (
            self.eta
            * ((1.0 - alpha_prev) / (1.0 - alpha) * (1.0 - alpha / alpha_prev))
            ** 0.5
        ).clamp(min=1e-10)
        dir_xt_coef = (1.0 - alpha_prev - sigma ** 2).clamp(min=0).sqrt()
        mean = (alpha_prev ** 0.5) * x_recon + dir_xt_coef * eps

        raw_std = sigma  # (B, 1, 1)
        sampling_std = torch.maximum(
            raw_std, torch.tensor(min_sampling_denoising_std,
                                  device=raw_std.device)
        )
        logprob_std = torch.maximum(
            raw_std, torch.tensor(min_logprob_denoising_std,
                                  device=raw_std.device)
        )
        ft_start = self.ddim_steps - self.ft_denoising_steps
        return TransitionStats(
            mean=mean,
            raw_std=raw_std,
            sampling_std=sampling_std,
            logprob_std=logprob_std,
            scheduler_timestep=self.ddim_t.to(raw_std.device)[index],
            optimizer_denoise_index=index - ft_start,
        )

    def sample(self, stats: TransitionStats, noise: torch.Tensor) -> torch.Tensor:
        """x_next = mean + sampling_std * clamp(noise, +-randn_clip)."""
        noise = noise.clamp(-self.randn_clip_value, self.randn_clip_value)
        return stats.mean + stats.sampling_std * noise

    def log_prob(self, stats: TransitionStats, x_next: torch.Tensor) -> torch.Tensor:
        """Per-dimension logprob (B, R, Da) under the logprob std floor —
        NEVER summed (the official loss needs per-element values)."""
        return Normal(stats.mean, stats.logprob_std).log_prob(x_next)


# ═══════════════════════════════════════════════════════════════════
# Legacy shims (moved verbatim from dp_sampler, R7-2B)
# ═══════════════════════════════════════════════════════════════════

def ddim_step_with_logprob(
    x_t: torch.Tensor,
    model_output: torch.Tensor,
    t: int,
    t_prev: int,
    scheduler,
    eta: float = 1.0,
    sigma_clip_min: float = 0.01,
    sigma_clip_max: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """LEGACY shim for the frozen historical E-series scripts — one DDIM
    transition with summed logprob, computed by the project kernel.
    ``sigma_clip_min/max`` are deprecated and ignored (the kernel owns
    the std floors). Never used by the canonical RL path (R7-2B Step 8:
    the canonical path executes the frozen official vendor runtime).

    The lazy official_runtime imports keep this module's import free of
    vendor verification (the shims only run inside the historical
    scripts).
    """
    from diffusion_policy_3d.rl.dppo_uav.official_runtime import (
        _model_output_to_epsilon,
        _require_eta_one,
    )

    _require_eta_one(eta)
    timesteps = scheduler.timesteps
    t_list = [int(ts.item()) for ts in timesteps]
    if t not in t_list:
        raise ValueError(
            f"legacy ddim_step_with_logprob: t={t} not in scheduler "
            f"timesteps {t_list} — the legacy shim only supports steps on "
            "the scheduler schedule"
        )
    step_index = t_list.index(t)
    device = x_t.device
    kernel = DPPOTransitionKernel(
        build_ddim_tables(
            scheduler.alphas_cumprod.to(device), timesteps.to(device)
        ),
        ft_denoising_steps=5,
    )
    eps = _model_output_to_epsilon(
        x_t, model_output, int(timesteps[step_index]), scheduler, device
    )
    index_b = torch.full((x_t.shape[0],), step_index, dtype=torch.long,
                         device=device)
    stats = kernel.transition_stats(x_t, eps, index_b, 0.1, 0.1)
    noise = torch.randn_like(x_t)
    x_next = kernel.sample(stats, noise)
    step_logp = kernel.log_prob(stats, x_next).view(
        x_next.shape[0], -1
    ).sum(dim=1)  # (B,) summed — the joint-computation contract
    return x_next, step_logp


def compute_per_step_logp_from_chain(
    chain: List[torch.Tensor],
    timesteps: torch.Tensor,
    model: nn.Module,
    global_cond: torch.Tensor,
    global_cond_uncond: torch.Tensor,
    cfg_lambda: float,
    scheduler,
    eta: float,
    sigma_clip_min: float,
    sigma_clip_max: float,
    ft_denoising_steps: int = None,
    logp_clamp_min: float = -5.0,
    logp_clamp_max: float = 2.0,
) -> torch.Tensor:
    """LEGACY shim for the frozen historical E-series scripts — per-step
    (summed) logp recompute over a stored chain via the project kernel.
    Not used by the canonical RL path (R7-2B: the recompute runs through
    the official facade via chain_logprobs)."""
    from diffusion_policy_3d.rl.dppo_uav.official_runtime import (
        _model_output_to_epsilon,
        _require_eta_one,
    )

    _require_eta_one(eta)
    device = next(model.parameters()).device
    B = chain[0].shape[0]
    K = len(timesteps)
    if ft_denoising_steps is not None and ft_denoising_steps < K:
        start_k = K - ft_denoising_steps
    else:
        start_k = 0
        ft_denoising_steps = K
    kernel = DPPOTransitionKernel(
        build_ddim_tables(scheduler.alphas_cumprod, timesteps),
        ft_denoising_steps=ft_denoising_steps,
    )
    use_cfg = cfg_lambda > 0 and global_cond_uncond is not None
    logp_steps = []
    for k in range(start_k, K):
        x_k = chain[k].to(device)
        x_next = chain[k + 1].to(device)
        t_batch = torch.full((B,), int(timesteps[k]), device=device,
                             dtype=torch.long)
        if use_cfg:
            gc = torch.cat([global_cond.to(device),
                            global_cond_uncond.to(device)], dim=0)
            xb = torch.cat([x_k, x_k], dim=0)
            tb = torch.cat([t_batch, t_batch], dim=0)
            mo = model(sample=xb, timestep=tb, local_cond=None, global_cond=gc)
            mo_cond, mo_uncond = mo.chunk(2, dim=0)
            model_output = mo_uncond + cfg_lambda * (mo_cond - mo_uncond)
        else:
            model_output = model(
                sample=x_k, timestep=t_batch, local_cond=None,
                global_cond=global_cond.to(device),
            )
        index_b = torch.full((B,), k, dtype=torch.long, device=device)
        # review partA P1-2: the kernel consumes EPSILON — sample-mode
        # model outputs (x̂₀) must go through the Tweedie conversion
        # first, exactly like every other kernel call site.
        eps = _model_output_to_epsilon(
            x_k, model_output, int(timesteps[k]), scheduler, device
        )
        stats = kernel.transition_stats(
            x_k, eps, index_b, 0.1, 0.1
        )
        logp_steps.append(
            kernel.log_prob(stats, x_next).view(B, -1).sum(dim=1)
        )
    return torch.stack(logp_steps, dim=1)  # (B, ft_steps)


def compute_joint_logp_from_chain(
    chain: List[torch.Tensor],
    timesteps: torch.Tensor,
    model: nn.Module,
    global_cond: torch.Tensor,
    global_cond_uncond: torch.Tensor,
    cfg_lambda: float,
    scheduler,
    eta: float,
    sigma_clip_min: float,
    sigma_clip_max: float,
    ft_denoising_steps: int = None,
    gamma_denoise: float = 1.0,
) -> torch.Tensor:
    """LEGACY shim: γ-weighted joint logp via the project kernel."""
    logp_per_step = compute_per_step_logp_from_chain(
        chain=chain, timesteps=timesteps, model=model,
        global_cond=global_cond, global_cond_uncond=global_cond_uncond,
        cfg_lambda=cfg_lambda, scheduler=scheduler, eta=eta,
        sigma_clip_min=sigma_clip_min, sigma_clip_max=sigma_clip_max,
        ft_denoising_steps=ft_denoising_steps,
    )
    ft = logp_per_step.shape[1]
    if gamma_denoise < 1.0 and ft > 1:
        ks = torch.arange(ft, device=logp_per_step.device, dtype=torch.float32)
        weights = gamma_denoise ** (ft - 1 - ks)
        return (logp_per_step * weights.unsqueeze(0)).sum(dim=1)
    return logp_per_step.sum(dim=1)
