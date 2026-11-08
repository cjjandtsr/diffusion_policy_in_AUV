"""
arfm_trainer.py — ARFM (Adaptive Reinforced Flow Matching) trainer.

Design: RouteA_ARFM_Trainer_Design.md

Replaces PPO's clipped surrogate objective with reward-weighted FM loss + W2
regularization. Key simplifications:
  - No ValueHead, GAE, advantage estimation
  - No logp recomputation (saves ~800 UNet forwards per update)
  - Standard FM pretraining loss, weighted by normalized episode reward
  - W2 regularization (||v_θ - v_ref||²) prevents BC forgetting

References:
  - ORW-CFM-W2 (Fan et al., ICLR 2025, arXiv:2502.06061)
  - AWM (Xue et al., 2025, arXiv:2509.25050)
"""

import copy
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from diffusion_policy_3d.rl.ppo_trainer import PPORolloutBuffer  # reuse buffer


class ARFMTrainer:
    """Reward-weighted FM loss + W2 regularization for online RL fine-tuning.

    Design doc §3.

    The W2 regularization uses a FROZEN reference copy of the velocity head
    (BC pretrained weights). This prevents BC forgetting by penalizing large
    deviations from the original policy.
    """

    def __init__(
        self,
        alpha: float = 0.5,          # reward weight scale
        beta_w2: float = 0.1,        # W2 regularization strength
        max_grad_norm: float = 0.5,  # gradient clipping
        ref_velocity_head: Optional[nn.Module] = None,  # frozen BC reference
    ):
        self.alpha = alpha
        self.beta_w2 = beta_w2
        self.max_grad_norm = max_grad_norm

        # Frozen reference model for W2 regularization
        if ref_velocity_head is not None:
            self.ref_velocity_head = copy.deepcopy(ref_velocity_head)
            self.ref_velocity_head.eval()
            for p in self.ref_velocity_head.parameters():
                p.requires_grad_(False)
        else:
            self.ref_velocity_head = None

        # Running reward stats for normalization
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self._reward_buffer: List[float] = []

    def update_reward_stats(self, rewards: List[float]):
        """Update running reward statistics."""
        self._reward_buffer.extend(rewards)
        if len(self._reward_buffer) > 10000:
            self._reward_buffer = self._reward_buffer[-10000:]
        if len(self._reward_buffer) > 1:
            r_tensor = torch.tensor(self._reward_buffer, dtype=torch.float32)
            self.reward_mean = r_tensor.mean().item()
            self.reward_std = max(r_tensor.std().item(), 1.0)

    def update(
        self,
        buffer: PPORolloutBuffer,
        velocity_head: nn.Module,
        noise_net: nn.Module,
        optimizer: torch.optim.Optimizer,
        horizon: int,
        action_dim: int,
        num_fm_steps: int = 4,
        sigma_min: float = 0.005,
        fm_t_scale: float = 50.0,
        cfg_lambda: float = 0.0,
    ) -> Dict[str, float]:
        """ARFM update: reward-weighted FM loss + W2 regularization.

        Args:
            buffer: filled rollout buffer (uses obs, a_chain, reward fields)
            velocity_head: UNet (trainable, LoRA)
            noise_net: NoiseNetwork (frozen during ARFM update)
            optimizer: single optimizer for velocity_head (and noise_net if desired)
            horizon, action_dim, num_fm_steps, sigma_min, fm_t_scale: config
            cfg_lambda: CFG strength (for W2 ref computation; CFG is NOT applied
                        to the training loss — only conditional path is trained)

        Returns:
            metrics: dict with arfm_loss, fm_loss, w2_reg
        """
        device = next(velocity_head.parameters()).device
        N = len(buffer)

        if N == 0:
            return {"arfm_loss": 0.0, "fm_loss": 0.0, "w2_reg": 0.0}

        # ── 1. Stack buffer data ──
        rewards = torch.tensor(buffer.reward_list, dtype=torch.float32, device=device)
        self.update_reward_stats(buffer.reward_list)

        # 1536d → 192d obs_enc helper
        def to_obs_enc(gc: torch.Tensor) -> torch.Tensor:
            if gc.dim() == 1:
                gc = gc.unsqueeze(0)
            if gc.shape[-1] == 192:
                return gc
            n_frames = gc.shape[-1] // 192
            return gc.reshape(-1, n_frames, 192).mean(dim=1)

        obs_batch = torch.stack([
            o.clone().detach() for o in buffer.obs_list
        ]).to(device)  # (N, 1536) — conditional encoding for UNet

        obs_enc_192 = to_obs_enc(obs_batch)  # (N, 192) — for NoiseNet

        # ── 2. Reward → normalized weights ──
        r_norm = (rewards - self.reward_mean) / (self.reward_std + 1e-8)
        weights = torch.softmax(self.alpha * r_norm, dim=0)  # (N,) — batch-normalized

        # ── 3. Stack a_chains: a_0 (noise) and a_K (clean action) ──
        a_0_batch = torch.stack([
            buffer.a_chain_list[i][0] for i in range(N)
        ]).to(device)  # (N, T, Da)
        a_K_batch = torch.stack([
            buffer.a_chain_list[i][-1] for i in range(N)
        ]).to(device)  # (N, T, Da)

        # ── 4. FM velocity loss (batched, conditional path only) ──
        # Sample random FM time
        t = torch.rand(1, device=device).expand(N)  # (N,)  U[0,1]
        t_expanded = t.view(N, 1, 1)  # (N, 1, 1)
        x_t = (1.0 - t_expanded) * a_0_batch + t_expanded * a_K_batch  # (N, T, Da)
        v_target = a_K_batch - a_0_batch  # (N, T, Da) — true velocity

        t_input = (t * fm_t_scale).float()  # (N,)

        v_pred = velocity_head(
            sample=x_t, timestep=t_input,
            local_cond=None, global_cond=obs_batch,
        )  # (N, T, Da)

        fm_loss_per_sample = ((v_pred - v_target) ** 2).mean(dim=[1, 2])  # (N,)

        # ── 5. W2 regularization: ||v_θ(x_t) - v_ref(x_t)||² ──
        if self.ref_velocity_head is not None and self.beta_w2 > 0:
            with torch.no_grad():
                v_ref = self.ref_velocity_head.to(device)(
                    sample=x_t, timestep=t_input,
                    local_cond=None, global_cond=obs_batch,
                )
            w2_per_sample = ((v_pred - v_ref) ** 2).mean(dim=[1, 2])  # (N,)
        else:
            w2_per_sample = torch.zeros(N, device=device)

        # ── 6. Total loss ──
        weighted_fm = (weights * fm_loss_per_sample).mean()
        w2_reg = w2_per_sample.mean()
        loss = weighted_fm + self.beta_w2 * w2_reg

        # ── 7. Backward + step ──
        optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm > 0:
            params = list(velocity_head.parameters()) + list(noise_net.parameters())
            torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        optimizer.step()

        return {
            "arfm_loss": loss.item(),
            "fm_loss": weighted_fm.item(),
            "w2_reg": w2_reg.item(),
        }
