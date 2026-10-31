"""
fm_sde_sampler.py — SDE sampling with state-conditioned noise network and exact log-prob.

Design: RouteA_ReinFlow_FM_SDE_PPO_Design_v1_0.md §2.2.4, §2.2.5

[UPGRADE from A1 scalar noise (archived at program/backup/2026-07-16-sde-ppo/)]:
Per-step scalar learnable rho -> per-dim state-conditioned MLP sigma(t, a, o).
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class NoiseNetwork(nn.Module):
    """State-conditioned noise network sigma(t, a, o).

    Input:  concat[obs_enc(192d), t_scaled(1d), a_k(11d)] = 204d
    Output: sigma_k(11d), per-dim noise scale in [sigma_min, sigma_max]

    Architecture: Linear(204, 128) -> ReLU -> Linear(128, 11) -> tanh -> affine
    Params: ~0.03M
    Trainable during RL (LR=1e-4), discarded after training.
    """

    def __init__(
        self,
        obs_dim: int = 192,
        action_dim: int = 11,
        hidden_dim: int = 128,
        sigma_min: float = 0.01,
        sigma_max: float = 0.1,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        input_dim = obs_dim + 1 + action_dim  # 192 + 1 + 11 = 204

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Init: push initial sigma near sigma_min
        # After tanh: y in [-1, 1]; sigma = sigma_min + (sigma_max - sigma_min)*(y+1)/2
        # For initial sigma approx sigma_min: y approx -1 -> tanh input << 0
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, -2.0)  # tanh(-2.0) approx -0.964

    def forward(
        self, obs_enc: torch.Tensor, t_scaled: torch.Tensor, a_k: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            obs_enc: (B, 192) or (B, T, 192) -- encoded observation
            t_scaled: (B,) or (B, 1) -- FM time scaled by fm_t_scale
            a_k: (B, T, 11) -- current noisy action

        Returns:
            sigma: (B, T, 11) -- per-dim noise scale
        """
        B, T, Da = a_k.shape

        # Broadcast obs_enc to match a_k shape
        if obs_enc.dim() == 2:
            obs_enc = obs_enc.unsqueeze(1).expand(-1, T, -1)  # (B, T, 192)

        # Broadcast t_scaled to (B, T, 1)
        if t_scaled.dim() == 1:
            t_scaled = t_scaled.view(B, 1, 1).expand(-1, T, 1).to(dtype=a_k.dtype)
        elif t_scaled.dim() == 2:
            t_scaled = t_scaled.unsqueeze(-1).expand(-1, T, 1).to(dtype=a_k.dtype)

        # Concatenate: (B, T, 204)
        net_input = torch.cat([obs_enc, t_scaled, a_k], dim=-1)

        # MLP forward
        raw = self.net(net_input)  # (B, T, 11)

        # tanh -> [-1, 1] -> affine -> [sigma_min, sigma_max]
        y = torch.tanh(raw)
        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * (y + 1.0) / 2.0

        return sigma


def log_gaussian_density(
    x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    """Compute log N(x | mu, sigma^2 * I) for per-dim sigma.

    Sums over the action dimension and all other non-batch dimensions,
    returning one log-probability per batch element.

    log N(x|mu,sigma^2*I) = -0.5 * sum_d[log(2*pi) + 2*log(sigma_d) + (x_d-mu_d)^2/sigma_d^2]

    For inputs with shape (B, T, Da), this computes the joint log-density
    of the entire trajectory by summing over T and Da.

    Args:
        x: (B, ..., D) -- sample
        mu: (B, ..., D) -- mean
        sigma: (B, ..., D) -- per-dim standard deviation

    Returns:
        logp: (B,) -- log probability per batch element (summed over all non-batch dims)
    """
    eps = 1e-10
    # Compute per-dim contributions, sum over last dim (D)
    D = x.shape[-1]
    log_det = 2.0 * torch.log(sigma + eps).sum(dim=-1)  # (B, ...)
    sq_error = ((x - mu) ** 2 / (sigma ** 2 + eps)).sum(dim=-1)  # (B, ...)
    logp = -0.5 * (D * math.log(2.0 * math.pi) + log_det + sq_error)  # (B, ...)
    # Sum over all remaining non-batch dimensions (e.g., T)
    if logp.dim() > 1:
        logp = logp.flatten(1).sum(dim=1)
    return logp


def sde_sample_with_logprob(
    obs_enc: torch.Tensor,
    velocity_head,
    noise_net: NoiseNetwork,
    horizon: int,
    action_dim: int,
    n_action_steps: int,
    n_obs_steps: int,
    num_fm_steps: int = 4,
    sigma_min: float = 0.005,
    fm_t_scale: float = 50.0,
    fm_anchor_t_start: float = 0.95,
    cfg_lambda: float = 1.5,
    global_cond: Optional[torch.Tensor] = None,
    global_cond_uncond: Optional[torch.Tensor] = None,
    local_cond: Optional[torch.Tensor] = None,
    condition_mask: Optional[torch.Tensor] = None,
    condition_data: Optional[torch.Tensor] = None,
    anchor_idx: Optional[torch.Tensor] = None,
    anchor_bank_full: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    """SDE sampling with exact joint log-probability.

    Design doc section 2.2.5 pseudocode (lines 191-256).

    Per-step k:
      1. Enforce anchor condition_mask (before velocity)
      2. v = velocity_head(a_k, t_k, obs) with optional CFG
      3. sigma = noise_net(t_k, a_k, obs_enc)  -- no CFG (P0-4)
      4. mu = a_k + v * dt
      5. a_{k+1} ~ N(mu, sigma^2 * I)
      6. Enforce anchor condition_mask again (after noise)
      7. step_logp = log N(a_{k+1} | mu, sigma^2), anchor dims -> 0
      8. a_chain.append(a_{k+1})

    Args:
        obs_enc: (B, 192) -- encoded observation from frozen obs_encoder
        velocity_head: UNet with forward(sample, timestep, local_cond, global_cond)
        noise_net: NoiseNetwork instance
        horizon: T (e.g. 16) -- full trajectory horizon
        action_dim: Da (11) -- action dimension
        n_action_steps: number of action steps to extract
        n_obs_steps: number of observation steps
        num_fm_steps: K (4) -- number of integration steps
        sigma_min: FM sigma_min (0.005)
        fm_t_scale: time scaling factor (50)
        fm_anchor_t_start: anchor initialization time (0.95)
        cfg_lambda: CFG strength (1.5)
        global_cond: (B, global_cond_dim) -- conditional encoding
        global_cond_uncond: (B, global_cond_dim) -- unconditional encoding
        local_cond: optional local conditioning
        condition_mask: (B, T, Da) bool -- anchor condition mask
        condition_data: (B, T, Da) -- anchor condition data
        anchor_idx: (B,) -- anchor indices
        anchor_bank_full: (K_anchors, T, Da) -- anchor bank

    Returns:
        action: (B, n_action_steps, action_dim) -- unnormalized action chunk
        logp: (B,) -- joint log-probability over K steps
        a_chain: list of (B, T, Da) -- [a_0, a_1, ..., a_K]
    """
    B = obs_enc.shape[0]
    T, Da = horizon, action_dim
    device = obs_enc.device
    dtype = obs_enc.dtype

    # ---- Initialize a_0 ----
    if anchor_idx is not None and anchor_bank_full is not None:
        anchor_traj = anchor_bank_full[anchor_idx]  # (B, T, Da)
        noise_init = torch.randn_like(anchor_traj)
        t_start = fm_anchor_t_start
        a = (1.0 - t_start) * noise_init + t_start * anchor_traj
    else:
        a = torch.randn(B, T, Da, device=device, dtype=dtype)
        t_start = 0.0

    logp = torch.zeros(B, device=device, dtype=torch.float32)
    a_chain = [a.clone()]

    # ---- Timestep grid (linspace, K+1 points) ----
    t_end = 1.0 - sigma_min
    t_start_clamped = min(t_start, t_end - 1e-6)
    timesteps = torch.linspace(
        t_start_clamped, t_end, num_fm_steps + 1, device=device
    )

    use_cfg = cfg_lambda > 0 and global_cond_uncond is not None

    for k in range(num_fm_steps):
        t_k = timesteps[k]
        dt = timesteps[k + 1] - t_k

        # ---- P0-1: Anchor enforcement before velocity ----
        if condition_mask is not None:
            a[condition_mask] = condition_data[condition_mask]

        # ---- Velocity prediction (with CFG) ----
        t_input = (t_k * fm_t_scale).float().expand(B)

        if use_cfg:
            a_batched = torch.cat([a, a], dim=0)
            t_batched = torch.cat([t_input, t_input], dim=0)
            global_batched = torch.cat([global_cond, global_cond_uncond], dim=0)
            local_batched = None
            if local_cond is not None:
                local_batched = torch.cat([local_cond, local_cond], dim=0)
            v_batched = velocity_head(
                sample=a_batched,
                timestep=t_batched,
                local_cond=local_batched,
                global_cond=global_batched,
            )
            v_cond, v_uncond = v_batched.chunk(2, dim=0)
            v = v_uncond + cfg_lambda * (v_cond - v_uncond)
        else:
            v = velocity_head(
                sample=a,
                timestep=t_input,
                local_cond=local_cond,
                global_cond=global_cond,
            )

        # ---- P0-4: Noise prediction (no CFG, conditional path only) ----
        sigma = noise_net(obs_enc, t_input, a)

        # ---- Gaussian transition: a_{k+1} ~ N(a_k + v*dt, sigma^2*I) ----
        mu = a + v * dt
        noise_step = torch.randn_like(a)
        a_next = mu + sigma * noise_step

        # ---- P0-1: Anchor enforcement after noise injection ----
        if condition_mask is not None:
            a_next[condition_mask] = condition_data[condition_mask]

        # ---- Exact log-prob (P0-2: anchor dims -> 0 contribution) ----
        step_logp = log_gaussian_density(a_next, mu, sigma)  # (B,) -- summed over T x Da
        if condition_mask is not None:
            # Zero out logp for anchor-conditioned samples
            any_cond = condition_mask.any(dim=-1).any(dim=-1)  # (B,) -- any dim conditioned
            step_logp = step_logp * (~any_cond).float()

        logp = logp + step_logp

        a = a_next
        a_chain.append(a.clone())

    # ---- Extract action chunk ----
    start = n_obs_steps - 1
    end = start + n_action_steps
    action = a[:, start:end, :]  # (B, n_action_steps, Da)

    return action, logp, a_chain
