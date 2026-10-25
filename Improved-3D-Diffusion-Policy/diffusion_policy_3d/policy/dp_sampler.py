"""
dp_sampler.py — reverse-loop orchestration for official DPPO DDIM.

The transition MATH executes in the FROZEN official vendor runtime
through ``official_runtime.OfficialPhase1aFacade`` (guide R7-2B): this
module only maps loop steps to facade indices, injects the shell noise
and runs the reverse loop. The project transition kernel
(``DPPOTransitionKernel``, dppo_transition_kernel.py) remains the
parity/historical home of the kernel math and the legacy
``ddim_step_with_logprob`` / ``compute_per_step_logp_from_chain`` /
``compute_joint_logp_from_chain`` shims (moved there in R7-2B) — the
canonical caller count of the project kernel is 0 (guide Step 8).

The canonical RL path uses
``ddim_step_official(x_t, model, cond, step_index, scheduler,
cfg_lambda, ...)`` with explicit full-chain index semantics; the
official likelihood window covers only the last ``ft`` transitions.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from diffusion_policy_3d.rl.dppo_uav.official_runtime import (
    OfficialPhase1aFacade,
    _model_output_to_epsilon,  # noqa: F401 — re-exported (tests/E-series)
    build_official_phase1a_facade,
)
from diffusion_policy_3d.rl.dppo_uav.vendor_loader import (  # noqa: F401
    load_official_dppo_symbols as _load_official_dppo_symbols,
)


def ddim_step_official(
    x_t: torch.Tensor,
    model: nn.Module,
    cond: Dict[str, torch.Tensor],
    step_index: int,
    scheduler,
    cfg_lambda: float,
    eta: float,
    min_sampling_denoising_std: float,
    min_logprob_denoising_std: float,
    ft_denoising_steps: int,
    noise: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, OfficialPhase1aFacade]:
    """One official DDIM transition executed by the frozen vendor
    runtime: returns (x_next, step_logp, facade).

    ``step_index`` is the FULL-chain index (0 = noisiest) — loop
    termination is the caller's job, never the distribution's. The last
    step into valid timestep 0 is a REAL stochastic transition whose
    std is floored (vendor p_mean_var). The official likelihood window
    covers the last ``ft`` transitions only: base steps return
    step_logp = 0 (the vendor ``get_logprobs_subsample`` maps relative
    indices in [0, ft)). ``noise`` may be injected for paired-RNG /
    vendor-parity use; None draws randn_like(x_t) — the
    mean/logvar/Normal logprob always come from the vendor, only the
    shell noise is injected (guide Step 7).
    """
    device = x_t.device
    timesteps = scheduler.timesteps.to(device)
    K = int(timesteps.shape[0])
    ft = int(ft_denoising_steps)
    facade = build_official_phase1a_facade(
        model=model,
        scheduler=scheduler,
        ft_denoising_steps=ft,
        cfg_lambda=cfg_lambda,
        device=device,
        horizon_steps=x_t.shape[1],
        action_dim=x_t.shape[2],
        eta=eta,
        min_sampling_denoising_std=min_sampling_denoising_std,
        min_logprob_denoising_std=min_logprob_denoising_std,
    )
    index_b = torch.full((x_t.shape[0],), step_index, dtype=torch.long,
                         device=device)
    t_b = torch.full((x_t.shape[0],), int(timesteps[step_index]),
                     dtype=torch.long, device=device)
    mean, logvar, _ = facade.p_mean_var(x_t, t_b, cond, index=index_b)
    # sampling std floor — exactly the vendor forward() post-processing
    std = torch.exp(0.5 * logvar)
    std = torch.clip(std, min=min_sampling_denoising_std)
    if noise is None:
        noise = torch.randn_like(x_t)
    # shell noise clamped to the vendor forward()'s randn_clip_value
    # (±10, diffusion.py) — the project kernel historically clamped at
    # dppo_transition_kernel.py:215; the facade carries the value so the
    # draw and any injected paired-RNG noise share one clamp.
    noise = noise.clamp(-facade.randn_clip_value, facade.randn_clip_value)
    x_next = mean + std * noise
    if step_index >= K - ft:
        rel = torch.full((x_t.shape[0],), step_index - (K - ft),
                         dtype=torch.long, device=device)
        logp = facade.get_logprobs_subsample(cond, x_t, x_next, rel)
        # (B,) summed — the legacy joint-computation contract
        step_logp = logp.view(x_t.shape[0], -1).sum(dim=1)
    else:
        step_logp = torch.zeros(x_t.shape[0], device=device)
    return x_next, step_logp, facade
