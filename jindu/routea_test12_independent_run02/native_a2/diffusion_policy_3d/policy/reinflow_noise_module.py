"""
ReinFlowNoiseModule — Per-step scalar noise for ReinFlow-style stochastic path (§4.1.2 A1)

Design doc: RouteA_ReinFlow_FM_RL_Design_v1_2.md §4.1.2

对 4-step Euler:

    sigma_k = clip(softplus(rho_k) + sigma_min, sigma_min, sigma_max), k=0..3
    x_{k+1} = x_k + dt * v_theta(x_k, t_k, c) + sigma_k * xi_k

用法:
    noise_module = ReinFlowNoiseModule(num_fm_steps=4, sigma_min=1e-4, sigma_max=0.10)
    noise = noise_module()  # (4,) — per-step sigma
    # Inject into Euler step:
    x_k1 = x_k + dt * v_k + noise[k] * torch.randn_like(x_k)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReinFlowNoiseModule(nn.Module):
    """Per-step learnable scalar noise for ReinFlow stochastic path.

    A1 default: 4 scalar rhos → softplus → clip → per-step sigma.
    """

    def __init__(self, num_fm_steps: int = 4,
                 sigma_min: float = 1e-4,
                 sigma_max: float = 0.10,
                 init_sigma: float = 0.01):
        super().__init__()
        self.num_fm_steps = num_fm_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # rho_k = log(exp(init_sigma - sigma_min) - 1), so softplus(rho_k) + sigma_min ≈ init_sigma
        rho_init = torch.tensor(math.log(math.exp(init_sigma - sigma_min) - 1)
                                if init_sigma > sigma_min else 0.0)
        self.rho = nn.Parameter(rho_init.expand(num_fm_steps))

    def forward(self) -> torch.Tensor:
        """Return per-step sigma, shape (num_fm_steps,).

        sigma_k = clip(softplus(rho_k) + sigma_min, sigma_min, sigma_max)
        """
        raw = F.softplus(self.rho) + self.sigma_min
        return torch.clamp(raw, self.sigma_min, self.sigma_max)

    def get_sigma(self) -> torch.Tensor:
        """Alias for forward()."""
        return self.forward()

    def get_log_sigma(self) -> torch.Tensor:
        """Return log(sigma) for numerical stability in log-prob computation."""
        return torch.log(self.forward() + 1e-10)

    def extra_repr(self) -> str:
        return (f"num_fm_steps={self.num_fm_steps}, "
                f"sigma_min={self.sigma_min}, sigma_max={self.sigma_max}, "
                f"init_rho={self.rho.data.tolist()}")


# ═══════════════════════════════════════════════════════════════════
# Stochastic path helpers
# ═══════════════════════════════════════════════════════════════════

def compute_stochastic_path_log_prob(
    noise_sigma: torch.Tensor,   # (num_fm_steps,) per-step sigma
    flow_velocity: torch.Tensor, # (B, num_fm_steps, T, Da) model outputs
    noise_xi: torch.Tensor,      # (B, num_fm_steps, T, Da) noise injected
    dt: float,
    reduce: str = "sum",
) -> torch.Tensor:
    """Compute log-probability of a stochastic FM path.

    For the ReinFlow stochastic Euler path:
        x_{k+1} = x_k + dt * v_k + sigma_k * xi_k, xi_k ~ N(0, I)

    The transition probability is:
        p(x_{k+1} | x_k) = N(x_k + dt * v_k, sigma_k^2 * I)

    The log-prob of the full path:
        log p(x_1..N | x_0) = sum_k log p(x_{k+1} | x_k)

    Args:
        noise_sigma: Per-step sigma values
        flow_velocity: Model velocity outputs v_k for each step
        noise_xi: The noise that was injected
        dt: Integration timestep
        reduce: 'sum' or 'mean' over batch

    Returns:
        log_prob: scalar if reduce='sum', else (B,) tensor
    """
    # residual = x_{k+1} - (x_k + dt * v_k) = sigma_k * xi_k
    # log p = -0.5 * sum_k ( ||residual||^2 / sigma_k^2 + T*Da * log(2*pi*sigma_k^2) )
    B, n_steps, T, Da = noise_xi.shape
    device = noise_xi.device

    # sigma: (n_steps,) → (1, n_steps, 1, 1) for broadcasting
    sigma = noise_sigma.view(1, -1, 1, 1).to(device)

    # Squared normalized residual
    squared_norm = (noise_xi ** 2).sum(dim=(2, 3))  # (B, n_steps)

    # Log-prob per step: -0.5 * (squared_norm / sigma^2 + D * log(2*pi*sigma^2))
    log_det = Da * T * torch.log(2 * math.pi * sigma.squeeze())  # (n_steps,)
    log_prob_per_step = -0.5 * (squared_norm / sigma.squeeze() ** 2 + log_det.unsqueeze(0))  # (B, n_steps)
    log_prob_sum = log_prob_per_step.sum(dim=1)  # (B,)

    if reduce == "sum":
        return log_prob_sum.sum()
    elif reduce == "mean":
        return log_prob_sum.mean()
    return log_prob_sum


import math
