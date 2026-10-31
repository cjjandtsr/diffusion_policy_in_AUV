"""
ppo_trainer.py — PPO training loop, GAE, rollout buffer for FM-SDE policy.

Design: RouteA_ReinFlow_FM_SDE_PPO_Design_v1_0.md §3.1, §3.2

Provides:
  - PPORolloutBuffer: stores (obs, a_chain, logp_old, reward, done, V_old, next_obs)
  - compute_gae: GAE advantage with running reward normalization
  - PPOTrainer: PPO clipped surrogate update loop
"""

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy_3d.policy.fm_sde_sampler import log_gaussian_density


# ═══════════════════════════════════════════════════════════════════
# PPO Rollout Buffer
# ═══════════════════════════════════════════════════════════════════

class PPORolloutBuffer:
    """Flat buffer for PPO rollout storage.

    Stores per-timestep transitions, flattens across episodes for PPO update.
    """

    def __init__(self):
        self.obs_list: list = []          # global_cond (1536,)
        self.obs_uncond_list: list = []   # global_cond_uncond for CFG (1536,) or None
        self.a_chain_list: list = []      # list of list[Tensor] — [a_0,...,a_K] per transition
        self.logp_old_list: list = []     # per-step tensors (ft_steps,) for DPPO, or scalars for FM-SDE
        self.bc_chain_list: list = []     # BC reference chain (optional, for BC regularization)
        self.reward_list: list = []
        self.done_list: list = []
        self.V_old_list: list = []
        self.next_obs_list: list = []     # next global_cond (1536,)

    def add(
        self,
        obs: torch.Tensor,
        obs_uncond: torch.Tensor,
        a_chain: List[torch.Tensor],
        logp_old: torch.Tensor,
        reward: float,
        done: bool,
        V_old: torch.Tensor,
        next_obs: torch.Tensor,
        bc_chain: List[torch.Tensor] = None,
    ):
        """Add one transition. bc_chain is optional BC reference chain for regularization."""
        self.obs_list.append(obs)
        self.obs_uncond_list.append(obs_uncond if obs_uncond is not None else obs)
        self.a_chain_list.append(a_chain)
        self.logp_old_list.append(logp_old)
        self.bc_chain_list.append(bc_chain)
        self.reward_list.append(reward)
        self.done_list.append(done)
        self.V_old_list.append(V_old)
        self.next_obs_list.append(next_obs)

    def clear(self):
        """Reset buffer."""
        self.obs_list.clear()
        self.obs_uncond_list.clear()
        self.a_chain_list.clear()
        self.logp_old_list.clear()
        self.bc_chain_list.clear()
        self.reward_list.clear()
        self.done_list.clear()
        self.V_old_list.clear()
        self.next_obs_list.clear()

    def __len__(self) -> int:
        return len(self.reward_list)


# ═══════════════════════════════════════════════════════════════════
# GAE Computation
# ═══════════════════════════════════════════════════════════════════

def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    reward_mean: float = 0.0,
    reward_std: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages and returns.

    Design §3.1 P0-3: reward normalization before GAE.

    Args:
        rewards: (T,) — raw rewards
        values: (T+1,) — V(s) for each state, last is V(s_T) or 0
        dones: (T,) — float (0 or 1) for episode termination
        gamma: discount factor
        gae_lambda: GAE lambda
        reward_mean: running reward mean
        reward_std: running reward std

    Returns:
        advantages: (T,) — GAE advantages
        returns: (T,) — R_target = A + V_old
    """
    # P0-3: Reward normalization
    r_norm = (rewards - reward_mean) / (reward_std + 1e-8)

    T = len(rewards)
    advantages = torch.zeros(T, device=rewards.device, dtype=torch.float32)
    gae = 0.0

    for t in reversed(range(T)):
        delta = r_norm[t] + gamma * (1.0 - dones[t]) * values[t + 1] - values[t]
        gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * gae
        advantages[t] = gae

    returns = advantages + values[:T]

    return advantages, returns


# ═══════════════════════════════════════════════════════════════════
# PPO Trainer
# ═══════════════════════════════════════════════════════════════════

class PPOTrainer:
    """PPO clipped surrogate update loop.

    Design §3.1 algorithm.
    """

    def __init__(
        self,
        clip_epsilon: float = 0.2,
        target_kl: float = 0.015,
        entropy_alpha: float = 0.0,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        max_grad_norm: float = 0.5,
        beta_w2: float = 0.0,
        ref_velocity_head: nn.Module = None,
    ):
        self.clip_epsilon = clip_epsilon
        self.target_kl = target_kl
        self.entropy_alpha = entropy_alpha
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.max_grad_norm = max_grad_norm
        self.beta_w2 = beta_w2
        self.ref_velocity_head = ref_velocity_head

        # Running reward normalization stats
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self._reward_buffer: List[float] = []

    def update_reward_stats(self, rewards: List[float]):
        """Update running reward statistics from collected batch."""
        self._reward_buffer.extend(rewards)
        # Keep buffer bounded
        if len(self._reward_buffer) > 10000:
            self._reward_buffer = self._reward_buffer[-10000:]
        if len(self._reward_buffer) > 1:
            r_tensor = torch.tensor(self._reward_buffer, dtype=torch.float32)
            self.reward_mean = r_tensor.mean().item()
            self.reward_std = max(r_tensor.std().item(), 1.0)

    def _recompute_joint_logp_batched(
        self,
        a_chains_batch: List[torch.Tensor],
        obs_encs: torch.Tensor,
        velocity_head: nn.Module,
        noise_net: nn.Module,
        horizon: int,
        action_dim: int,
        num_fm_steps: int,
        sigma_min: float,
        fm_t_scale: float,
        global_conds: torch.Tensor,
        global_conds_uncond: torch.Tensor,
        cfg_lambda: float,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Recompute joint log-prob for a BATCH of stored a_chains, WITH CFG.

        Mathematically identical to per-sample version — GroupNorm ensures
        batch-independence. Speedup: ~B× fewer UNet forwards.

        Args:
            a_chains_batch: [a_0, ..., a_K] each (B, T, Da) — stacked across batch
            obs_encs: (B, 1536) — conditional global encodings
            velocity_head, noise_net: current policy params
            horizon, action_dim, num_fm_steps, sigma_min, fm_t_scale: config
            global_conds: (B, 1536) — conditional encodings for UNet
            global_conds_uncond: (B, 1536) — unconditional encodings for CFG
            cfg_lambda: CFG strength
            timesteps: (K+1,) — FM timesteps

        Returns:
            logp: (B,) — joint log-prob per batch element (summed over K steps)
        """
        device = next(velocity_head.parameters()).device
        B = a_chains_batch[0].shape[0]

        # Derive 192d obs_encs from 1536d global_conds for NoiseNet
        obs_encs_192 = obs_encs.to(device)
        if obs_encs_192.shape[-1] != 192:
            n_frames = obs_encs_192.shape[-1] // 192
            obs_encs_192 = obs_encs_192.reshape(B, n_frames, 192).mean(dim=1)  # (B, 192)

        use_cfg = cfg_lambda > 0 and global_conds_uncond is not None
        logp_total = torch.zeros(B, device=device, dtype=torch.float32)

        for k in range(num_fm_steps):
            t_k = timesteps[k]
            dt = timesteps[k + 1] - t_k
            a_k = a_chains_batch[k].to(device)       # (B, T, Da)
            a_next = a_chains_batch[k + 1].to(device)  # (B, T, Da)

            t_input = (t_k * fm_t_scale).float().expand(B).to(device)  # (B,)

            if use_cfg:
                # CFG: cond + uncond batch → (2B, T, Da), (2B,), (2B, 1536)
                a_cfg = torch.cat([a_k, a_k], dim=0)           # (2B, T, Da)
                t_cfg = torch.cat([t_input, t_input], dim=0)    # (2B,)
                gc_cfg = torch.cat([global_conds.to(device), global_conds_uncond.to(device)], dim=0)  # (2B, 1536)
                v_cfg = velocity_head(
                    sample=a_cfg, timestep=t_cfg,
                    local_cond=None, global_cond=gc_cfg,
                )  # (2B, T, Da)
                v_cond, v_uncond = v_cfg.chunk(2, dim=0)  # each (B, T, Da)
                v = v_uncond + cfg_lambda * (v_cond - v_uncond)  # (B, T, Da)
            else:
                v = velocity_head(
                    sample=a_k, timestep=t_input,
                    local_cond=None, global_cond=global_conds.to(device),
                )  # (B, T, Da)

            # Noise prediction (no CFG, conditional path only)
            sigma = noise_net(obs_encs_192, t_input, a_k)  # (B, T, Da)

            # Reconstructed mean with CFG velocity
            mu = a_k + v * dt  # (B, T, Da)

            # Log-prob per batch element
            step_logp = log_gaussian_density(a_next, mu, sigma)  # (B,) — summed over T x Da
            logp_total = logp_total + step_logp

        return logp_total  # (B,)

    def update(
        self,
        buffer: PPORolloutBuffer,
        velocity_head: nn.Module,
        noise_net: nn.Module,
        value_head: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        noise_optimizer: torch.optim.Optimizer,
        value_optimizer: torch.optim.Optimizer,
        horizon: int,
        action_dim: int,
        num_fm_steps: int = 4,
        sigma_min: float = 0.005,
        fm_t_scale: float = 50.0,
        cfg_lambda: float = 0.0,
        K_epochs: int = 4,
        mini_batch_size: int = 64,
    ) -> Dict[str, float]:
        """PPO update loop.

        Args:
            buffer: filled PPORolloutBuffer
            velocity_head: UNet (trainable)
            noise_net: NoiseNetwork (trainable)
            value_head: V(s) MLP (trainable)
            actor_optimizer: optimizer for velocity_head params
            noise_optimizer: optimizer for noise_net params
            value_optimizer: optimizer for value_head params
            horizon, action_dim, num_fm_steps, sigma_min, fm_t_scale: config
            K_epochs: PPO epochs per batch
            mini_batch_size: mini-batch size

        Returns:
            metrics: dict with actor_loss, value_loss, entropy, approx_kl
        """
        device = next(velocity_head.parameters()).device
        N = len(buffer)

        if N == 0:
            return {"actor_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}

        # ── Stack buffer data ──
        rewards = torch.tensor(buffer.reward_list, dtype=torch.float32, device=device)
        dones = torch.tensor(buffer.done_list, dtype=torch.float32, device=device)

        # Helper: 1536d global_cond → 192d obs_enc for ValueHead/NoiseNet
        def to_obs_enc(gc: torch.Tensor) -> torch.Tensor:
            if gc.dim() == 1:
                gc = gc.unsqueeze(0)
            if gc.shape[-1] == 192:
                return gc
            n_frames = gc.shape[-1] // 192
            return gc.reshape(-1, n_frames, 192).mean(dim=1)

        # Values: V(s) for all states + V(s_next) for last state (needed for GAE)
        V_old_all = torch.stack([v.detach().to(device).reshape(()) for v in buffer.V_old_list])  # (N,)
        last_next_obs = torch.stack([o for o in buffer.next_obs_list])  # (N, 1536)
        with torch.no_grad():
            V_next = value_head(to_obs_enc(last_next_obs).to(device)).reshape(-1)  # (N,)
        values_for_gae = torch.cat([V_old_all, V_next[-1:]], dim=0)  # (N+1,)

        # ── Update reward stats ──
        self.update_reward_stats(buffer.reward_list)

        # ── Compute GAE ──
        advantages, returns = compute_gae(
            rewards=rewards,
            values=values_for_gae,
            dones=dones,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            reward_mean=self.reward_mean,
            reward_std=self.reward_std,
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── Prepare obs and old logp ──
        obs_batch = torch.stack([o.clone().detach() for o in buffer.obs_list]).to(device)  # (N, 1536)
        obs_uncond_batch = torch.stack([o.clone().detach() for o in buffer.obs_uncond_list]).to(device)  # (N, 1536)
        logp_old = torch.stack([lp.clone().detach() for lp in buffer.logp_old_list]).to(device)  # (N,)

        # ── PPO epochs ──
        indices = torch.randperm(N, device=device)
        total_samples = 0
        metrics = {"actor_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}

        # Precompute timesteps (same for all samples)
        t_end = 1.0 - sigma_min
        fm_timesteps = torch.linspace(0.0, t_end, num_fm_steps + 1, device=device)

        for epoch in range(K_epochs):
            for start in range(0, N, mini_batch_size):
                end = min(start + mini_batch_size, N)
                mb_indices = indices[start:end]
                mb_size = len(mb_indices)

                # ── Recompute log-prob BATCHED for mini-batch ──
                # Stack a_chains: list of (K+1) tensors, each (mb_size, T, Da)
                mb_a_chains = []
                for k in range(num_fm_steps + 1):
                    mb_a_chains.append(torch.stack([
                        buffer.a_chain_list[idx.item()][k] for idx in mb_indices
                    ]).to(device))
                logp_new = self._recompute_joint_logp_batched(
                    a_chains_batch=mb_a_chains,
                    obs_encs=obs_batch[mb_indices],
                    velocity_head=velocity_head,
                    noise_net=noise_net,
                    horizon=horizon,
                    action_dim=action_dim,
                    num_fm_steps=num_fm_steps,
                    sigma_min=sigma_min,
                    fm_t_scale=fm_t_scale,
                    global_conds=obs_batch[mb_indices],
                    global_conds_uncond=obs_uncond_batch[mb_indices],
                    cfg_lambda=cfg_lambda,
                    timesteps=fm_timesteps,
                )  # (mb_size,)

                logp_old_mb = logp_old[mb_indices]  # (mb_size,)

                # ── Ratio ──
                ratio = torch.exp(logp_new - logp_old_mb)

                # ── Actor loss (clipped surrogate) ──
                adv_mb = advantages[mb_indices]
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * adv_mb
                actor_loss = -torch.min(surr1, surr2).mean()

                # ── Entropy bonus: +α·E[logp] — minimizing pushes σ↑ → exploration↑ ──
                # Design §3.1 L31: correct sign (+ not -)
                entropy_term = self.entropy_alpha * logp_new.mean()

                # ── W2 BC regularization (Phase 1c): β_w2·‖v_θ−v_ref‖² ──
                w2_term = torch.tensor(0.0, device=device)
                if self.beta_w2 > 0 and self.ref_velocity_head is not None:
                    # Sample random FM time for W2 computation
                    t_w2 = torch.rand(mb_size, device=device)  # (mb_size,) U[0,1]
                    t_w2_input = (t_w2 * fm_t_scale).float()  # (mb_size,)
                    # x_t along FM path: a_0 + t*(a_K - a_0)
                    a_0_mb = mb_a_chains[0]   # (mb_size, T, Da)
                    a_K_mb = mb_a_chains[-1]  # (mb_size, T, Da)
                    t_w2_exp = t_w2.view(mb_size, 1, 1)  # (mb_size, 1, 1)
                    x_t_w2 = (1.0 - t_w2_exp) * a_0_mb + t_w2_exp * a_K_mb  # (mb_size, T, Da)
                    # Current velocity
                    v_cur = velocity_head(
                        sample=x_t_w2, timestep=t_w2_input,
                        local_cond=None, global_cond=obs_batch[mb_indices],
                    )
                    # Reference velocity (frozen BC, no grad)
                    with torch.no_grad():
                        v_ref = self.ref_velocity_head.to(device)(
                            sample=x_t_w2, timestep=t_w2_input,
                            local_cond=None, global_cond=obs_batch[mb_indices],
                        )
                    w2_term = self.beta_w2 * ((v_cur - v_ref) ** 2).mean()

                # ── Total actor loss ──
                total_actor_loss = actor_loss + entropy_term + w2_term

                # ── Value loss ──
                V_new_mb = value_head(to_obs_enc(obs_batch[mb_indices]).to(device)).squeeze(-1)
                returns_mb = returns[mb_indices].to(device)
                value_loss = F.mse_loss(V_new_mb, returns_mb)

                # ── Gradient steps (separate LRs, with clipping) ──
                actor_optimizer.zero_grad()
                noise_optimizer.zero_grad()
                value_optimizer.zero_grad()

                total_actor_loss.backward()
                # Gradient clipping: prevent single-step logp jumps
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(velocity_head.parameters(), self.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(noise_net.parameters(), self.max_grad_norm)
                actor_optimizer.step()
                noise_optimizer.step()

                value_loss.backward()
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(value_head.parameters(), self.max_grad_norm)
                value_optimizer.step()

                # ── Accumulate metrics ──
                metrics["actor_loss"] += actor_loss.item() * mb_size
                metrics["value_loss"] += value_loss.item() * mb_size
                metrics["entropy"] += entropy_term.item() * mb_size
                total_samples += mb_size

            # ── Early stop on KL (batched, single forward pass) ──
            with torch.no_grad():
                # Stack all a_chains: list of (K+1) tensors, each (N, T, Da)
                full_a_chains = []
                for k in range(num_fm_steps + 1):
                    full_a_chains.append(torch.stack([
                        buffer.a_chain_list[i][k] for i in range(N)
                    ]).to(device))
                logp_full_new = self._recompute_joint_logp_batched(
                    a_chains_batch=full_a_chains,
                    obs_encs=obs_batch,
                    velocity_head=velocity_head,
                    noise_net=noise_net,
                    horizon=horizon,
                    action_dim=action_dim,
                    num_fm_steps=num_fm_steps,
                    sigma_min=sigma_min,
                    fm_t_scale=fm_t_scale,
                    global_conds=obs_batch,
                    global_conds_uncond=obs_uncond_batch,
                    cfg_lambda=cfg_lambda,
                    timesteps=fm_timesteps,
                )  # (N,)
                ratio_full = torch.exp(logp_full_new - logp_old.to(device))
                # Normalize kl to per-dimension (logp sums over horizon*Da*num_fm_steps dims)
                total_dims = horizon * action_dim * num_fm_steps
                approx_kl = ((ratio_full - 1.0) - torch.log(ratio_full + 1e-10)).mean().item() / total_dims

            metrics["approx_kl"] = approx_kl
            if approx_kl > 1.5 * self.target_kl:
                break

        # ── Average metrics ──
        if total_samples > 0:
            metrics["actor_loss"] /= total_samples
            metrics["value_loss"] /= total_samples
            metrics["entropy"] /= total_samples

        return metrics
