#!/usr/bin/env python3
"""
train_ddpm_ppo.py — DPPO (DDPM Diffusion Policy + PPO) training entry point.

Design: program/sheji/DPPO_DDPM_RL_PostTraining_Design_v1_0.md §3, §4, §6

Loads DDPM BC checkpoint (Phase1aPolicy), inserts LoRA, adds ValueHead,
runs PPO online fine-tuning with DDIM η-noise sampling + exact log-prob.

Usage:
    python scripts/train_ddpm_ppo.py --mode smoke       # 200 upd, local
    python scripts/train_ddpm_ppo.py --mode formal      # 1000 upd
    python scripts/train_ddpm_ppo.py --dry_run          # imports only
"""
import argparse, json, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.eval_ddpm_bc_baseline import load_ddpm_policy
from diffusion_policy_3d.rl.dppo_uav.entrypoint_guard import (
    HISTORICAL_INVALID_FOR_FORMAL_TRAINING,
    guard_formal_mode,
)


# ── Default ckpt: Phase② λ=0.5 (best DDPM baseline, E011) ──
DEFAULT_BC_CKPT = "data/outputs/2026.05.17/13.36.20_v8_phase2_sweep_lambda0.5/checkpoints/latest.ckpt"

# ═══════════════════════════════════════════════════════════════════
# E038★ F6 修复副本（2026-08-03，E037 失败定案后，设计 v2.0 §7 G1）
#   继承 E037（Fix-1 去二次 meritocracy / Add-1 分量日志 / Add-2 中间 ckpt）
#   F6 (G1): 删除 r_safety 全量（reward_v16.py 绝对式 r_safety 已删）
#     —— r_safety 家族三连败（E031 绝对式空旷巡航 / E036 delta 逃离陷阱 /
#        E037 gate-delta 绕圈不完成）：C2 目标=障碍密集区，任何 clearance 型
#        shaping 都惩罚任务行为 → reward = progress + smooth + collision + reach
#     —— 本副本：delta hook 移除，改纯 terms 收集 hook（G1 日志替代源，审查 P0-2）
#   E038 单一变量：相对 E037 仅 reward 结构（删 r_safety）+ G8/G11 基础设施；
#   seed 归 E039（维持单变量）。设计：program/sheji/DPPO_Structural_Refactoring_v2_0.md
# ═══════════════════════════════════════════════════════════════════

_TERMS_COLLECTOR = []  # 模块级收集器：每步 terms/metrics 快照（hook 写入，每 update 后清空）


def _install_terms_collector() -> None:
    """E038 G1 日志替代源：r_safety 已删、delta reward hook 已移除（原 _DELTA_TERMS
    无写入源）→ 分量日志与 d_soft monitor 改由本 hook 供给。纯收集，不修改 reward。"""
    from diffusion_policy_3d.env.kinematic_uav_env_v2 import KinematicUAVEnv

    _orig_compute = KinematicUAVEnv._compute_reward_v16

    def hooked_compute(self, action, pos_t, yaw_t, vel_t):
        reward, terminated = _orig_compute(self, action, pos_t, yaw_t, vel_t)
        _TERMS_COLLECTOR.append((
            dict(self._last_v14_terms or {}),
            dict(self._last_v14_metrics or {}),
        ))
        return reward, terminated

    KinematicUAVEnv._compute_reward_v16 = hooked_compute
DEFAULT_ADMISSION_MANIFEST = "../../program/reports/routeA_rl_env_v1/calibration/admission_manifest_unified_292worlds.json"
DEFAULT_REWARD_NORM = "program/shiyan/results_routeA_RL_b0b1_rerun_v13_20260710/reward_norm_stats.json"
DEFAULT_OUTPUT_DIR = "outputs/ddpm_ppo"


# ═══════════════════════════════════════════════════════════════════
# Value Head (same as FM-SDE)
# ═══════════════════════════════════════════════════════════════════

class ValueHead(nn.Module):
    """V(s): 192 -> 128 -> 64 -> 1. P1-6: LayerNorm for stable training."""

    def __init__(self, input_dim=192, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x): return self.net(x)


# ═══════════════════════════════════════════════════════════════════
# Observation Encoding (same as FM-SDE, adapted for Phase1aPolicy)
# ═══════════════════════════════════════════════════════════════════

def encode_obs(obs_dict, policy, device):
    """Encode raw obs dict -> global_cond (1, obs_dim * n_obs_steps)."""
    from diffusion_policy_3d.common.pytorch_util import dict_apply
    nobs = policy.normalizer.normalize(obs_dict)
    nobs = dict_apply(nobs, lambda x: torch.from_numpy(x).to(device).unsqueeze(0)
                      if isinstance(x, np.ndarray) else x.to(device).unsqueeze(0))
    if not policy.use_pc_color:
        nobs['point_cloud'] = nobs['point_cloud'][..., :3]
    this_nobs = dict_apply(
        nobs, lambda x: x[:, :policy.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
    return policy.obs_encoder(this_nobs).reshape(1, -1)


def encode_obs_uncond(obs_dict, policy, device):
    """Encode raw obs -> unconditional global_cond for CFG."""
    from diffusion_policy_3d.common.pytorch_util import dict_apply
    nobs = policy.normalizer.normalize(obs_dict)
    nobs = dict_apply(nobs, lambda x: torch.from_numpy(x).to(device).unsqueeze(0)
                      if isinstance(x, np.ndarray) else x.to(device).unsqueeze(0))
    if not policy.use_pc_color:
        nobs['point_cloud'] = nobs['point_cloud'][..., :3]
    nobs['agent_pos'][:, :, 3:7] = 0.0
    this_nobs = dict_apply(
        nobs, lambda x: x[:, :policy.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
    return policy.obs_encoder(this_nobs).reshape(1, -1)


def global_to_obs_enc(global_cond, n_obs_steps=3, obs_dim=192):
    """Extract 192d obs_enc from global_cond by averaging over frames."""
    if global_cond.dim() == 1:
        global_cond = global_cond.unsqueeze(0)
    return global_cond.reshape(-1, n_obs_steps, obs_dim).mean(dim=1)


def _stack_obs_batch(obs_list, device):
    """Stack raw obs dicts (list of np-array dicts) -> tensor dict (B, T, ...)."""
    keys = obs_list[0].keys()
    return {
        k: torch.stack([
            torch.from_numpy(o[k]).to(device) if isinstance(o[k], np.ndarray)
            else o[k].to(device) for o in obs_list])
        for k in keys
    }


def encode_obs_batch(obs_b, policy):
    """Encode batched raw obs (B, T, ...) -> global_cond (B, -1)."""
    from diffusion_policy_3d.common.pytorch_util import dict_apply
    nobs = policy.normalizer.normalize(obs_b)
    if not policy.use_pc_color:
        nobs['point_cloud'] = nobs['point_cloud'][..., :3]
    this_nobs = dict_apply(
        nobs, lambda x: x[:, :policy.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
    B = obs_b['agent_pos'].shape[0]
    return policy.obs_encoder(this_nobs).reshape(B, -1)


def encode_obs_uncond_batch(obs_b, policy):
    """Encode batched raw obs -> unconditional global_cond (B, -1) for CFG."""
    from diffusion_policy_3d.common.pytorch_util import dict_apply
    nobs = policy.normalizer.normalize(obs_b)
    if not policy.use_pc_color:
        nobs['point_cloud'] = nobs['point_cloud'][..., :3]
    nobs['agent_pos'][:, :, 3:7] = 0.0
    this_nobs = dict_apply(
        nobs, lambda x: x[:, :policy.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
    B = obs_b['agent_pos'].shape[0]
    return policy.obs_encoder(this_nobs).reshape(B, -1)


# ═══════════════════════════════════════════════════════════════════
# PPO Rollout Buffer (reuse from FM-SDE)
# ═══════════════════════════════════════════════════════════════════

from diffusion_policy_3d.rl.ppo_trainer import PPORolloutBuffer, compute_gae


# ═══════════════════════════════════════════════════════════════════
# DPPO Trainer (DDPM-specific PPO update)
# ═══════════════════════════════════════════════════════════════════

class DPPOTrainer:
    """DPPO (Ren et al., ICLR 2025) — per-step PPO on DDPM denoising chain.

    Key DPPO design elements:
      - γ_denoise: per-step advantage discount (early steps → less signal)
      - ε_k exponential schedule: tighter clip for early (noisy) steps
      - Per-step logp + per-step ratio (not joint-chain)
    """

    def __init__(self, clip_epsilon=0.2, target_kl=0.015, entropy_alpha=0.0,
                 gamma=0.99, gae_lambda=0.95, max_grad_norm=0.5,
                 gamma_denoise=0.8, eps_base=1e-3, eps_final=0.2, eps_beta=3.0,
                 bc_model=None, w_bc=0.0, logp_mean_norm=False):
        self.clip_epsilon = clip_epsilon
        self.target_kl = target_kl
        self.entropy_alpha = entropy_alpha
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.max_grad_norm = max_grad_norm
        # P0 (2026-08-01): logp 维度均值化开关。采样/重算两侧同公式(176 维求和)，
        # 此处对 logp_diff 除以 horizon×action_dim = 两侧同时缩放 = 官方 per-dim mean，
        # ratio 与 KL 尺度恢复正常（Δlogp std 2.75 → 0.016, clip 窗口 ±0.18 生效）。
        self.logp_mean_norm = logp_mean_norm
        # DPPO per-step
        self.gamma_denoise = gamma_denoise
        self.eps_base = eps_base
        self.eps_final = eps_final
        self.eps_beta = eps_beta
        # BC regularization (DPPO Phase 3)
        self.bc_model = bc_model
        self.w_bc = w_bc
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self._reward_buffer = []

    def update_reward_stats(self, rewards):
        self._reward_buffer.extend(rewards)
        if len(self._reward_buffer) > 10000:
            self._reward_buffer = self._reward_buffer[-10000:]
        if len(self._reward_buffer) > 1:
            r = torch.tensor(self._reward_buffer, dtype=torch.float32)
            self.reward_mean = r.mean().item()
            self.reward_std = max(r.std().item(), 1.0)

    def _generate_deterministic_chain(self, x_T, model, scheduler, global_conds,
                                        global_conds_uncond, cfg_lambda,
                                        num_inference_steps):
        """Generate deterministic DDIM chain (η=0) from x_T using given model.

        Used for BC regularization: generate BC reference chain during PPO update
        (batched, efficient) instead of during rollout (per-step, expensive).

        Args:
            x_T: (B, H, Da) initial noise
            model: UNet (BC model, frozen, no LoRA)
            global_conds: (B, D) encoded observations
            global_conds_uncond: (B, D) or None
            cfg_lambda: CFG strength
            num_inference_steps: K

        Returns:
            chain: list of (B, H, Da) length K+1, chain[0]=x_T, chain[-1]=x_0
        """
        from diffusion_policy_3d.policy.dp_sampler import ddim_step_with_logprob, _model_output_to_epsilon

        device = next(model.parameters()).device
        B = x_T.shape[0]
        scheduler.set_timesteps(num_inference_steps)
        timesteps = scheduler.timesteps.to(device)
        K = len(timesteps)

        use_cfg = cfg_lambda > 0 and global_conds_uncond is not None
        x_t = x_T.to(device)
        chain = [x_t.clone()]

        for i in range(K):
            t = int(timesteps[i].item())
            t_prev = int(timesteps[i + 1].item()) if i + 1 < K else 0
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            if use_cfg:
                x_batched = torch.cat([x_t, x_t], dim=0)
                t_batched = torch.cat([t_batch, t_batch], dim=0)
                gc_batched = torch.cat([global_conds, global_conds_uncond], dim=0)
                eps_batched = model(sample=x_batched, timestep=t_batched,
                                    local_cond=None, global_cond=gc_batched)
                eps_cond, eps_uncond = eps_batched.chunk(2, dim=0)
                eps = eps_uncond + cfg_lambda * (eps_cond - eps_uncond)
            else:
                eps = model(sample=x_t, timestep=t_batch,
                            local_cond=None, global_cond=global_conds)

            x_next, _ = ddim_step_with_logprob(
                x_t=x_t, model_output=eps, t=t, t_prev=t_prev, scheduler=scheduler,
                eta=0.0)  # η=0 → deterministic, logp not needed
            x_t = x_next
            chain.append(x_t.clone())

        return chain

    def _recompute_joint_logp_batched(self, a_chains_batch, obs_encs_192,
                                       model, scheduler, global_conds,
                                       global_conds_uncond, cfg_lambda,
                                       num_inference_steps, eta,
                                       sigma_clip_min, sigma_clip_max,
                                       ft_denoising_steps=None):
        """Recompute joint (γ-weighted sum) logp for DDPM chain.

        Returns (B,) scalar per sample — one PPO ratio per transition.
        """
        from diffusion_policy_3d.policy.dp_sampler import compute_joint_logp_from_chain

        device = next(model.parameters()).device
        scheduler.set_timesteps(num_inference_steps)
        timesteps = scheduler.timesteps.to(device)

        return compute_joint_logp_from_chain(
            chain=a_chains_batch, timesteps=timesteps,
            model=model, global_cond=global_conds,
            global_cond_uncond=global_conds_uncond,
            cfg_lambda=cfg_lambda, scheduler=scheduler,
            eta=eta, sigma_clip_min=sigma_clip_min,
            sigma_clip_max=sigma_clip_max,
            ft_denoising_steps=ft_denoising_steps,
            gamma_denoise=self.gamma_denoise,
        )

    def _compute_eps_schedule(self, ft_steps: int, device) -> torch.Tensor:
        """DPPO §4.2: per-step ε_k exponential schedule.

        k=0 (final, near x_0) → loose ε≈ε_final=0.2
        k=ft-1 (early, noisy) → tight ε≈ε_base=1e-3
        """
        if ft_steps <= 1:
            return torch.tensor([self.eps_final], device=device)
        denom = max(ft_steps - 1, 1)
        ks = torch.arange(ft_steps, device=device, dtype=torch.float32)
        # Invert: k=0 (final step near x_0) → t=1 → ε_final; k=ft-1 (early, noisy) → t=0 → ε_base
        t = 1.0 - ks / denom  # normalized [0, 1], 1=final step, 0=earliest step
        exp_vals = (torch.exp(self.eps_beta * t) - 1.0) / (torch.exp(torch.tensor(self.eps_beta)) - 1.0)
        eps_k = self.eps_base + (self.eps_final - self.eps_base) * exp_vals
        return eps_k

    def update(self, buffer, model, scheduler, value_head,
               actor_optimizer, value_optimizer,
               horizon, action_dim, num_inference_steps,
               n_obs_steps, cfg_lambda, eta,
               sigma_clip_min, sigma_clip_max,
               ft_denoising_steps=None,
               K_epochs=4, mini_batch_size=64,
               advantage_weight=None,
               skip_actor=False):
        """Standard PPO on γ-weighted joint logp (scalar ratio, constant ε=0.2).

        γ_denoise weights early denoising steps less in the logp sum → the PPO
        ratio naturally focuses credit on later (more important) steps. BC reg
        uses per-step logp internally to anchor each step independently.

        E039 (F5, 设计 v1.1 §4.3): skip_actor=True 时跳过 actor 全部计算
        （logp 重算 / actor_loss / bc_loss），只做 critic 更新——失败主导批次
        （ep_sr==0）的 advantage 全为失败经验，actor 更新学"怎么不亏"
        （E037/E038 尾态吸收路径）；critic 照常更新保拟合。
        """
        device = next(model.parameters()).device
        n_transitions = len(buffer)

        # Stack buffer
        obs_all = torch.stack([b.to(device) for b in buffer.obs_list])
        obs_uncond_all = torch.stack([b.to(device) for b in buffer.obs_uncond_list])
        rewards = torch.tensor(buffer.reward_list, device=device)
        dones = torch.tensor(buffer.done_list, device=device, dtype=torch.float32)
        V_old_all = torch.stack([b.to(device) for b in buffer.V_old_list])
        next_obs_all = torch.stack([b.to(device) for b in buffer.next_obs_list])

        # GAE
        with torch.no_grad():
            V_last = value_head(global_to_obs_enc(next_obs_all[-1:],
                                     n_obs_steps=n_obs_steps, obs_dim=192)).squeeze()
            if V_last.dim() == 0:
                V_last = V_last.unsqueeze(0)
            values = torch.cat([V_old_all, V_last], dim=0)
            advantages, returns = compute_gae(
                rewards, values, dones, self.gamma, self.gae_lambda,
                self.reward_mean, self.reward_std)

        obs_encs_192 = global_to_obs_enc(obs_all, n_obs_steps=n_obs_steps, obs_dim=192)

        # Joint logp: scalar per transition (γ-weighted sum of per-step logps)
        logp_old = torch.stack([b.to(device) for b in buffer.logp_old_list])  # (N,) scalars

        # FIX P1-2: Normalize advantages over FULL batch (REINFORCE++: per-minibatch norm is biased).
        # Move normalization BEFORE the mini-batch loop.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        if advantage_weight is not None:
            advantages = advantages * torch.tensor(advantage_weight, device=device, dtype=torch.float32)

        actor_loss_total = 0.0
        value_loss_total = 0.0
        approx_kl_total = 0.0
        clipfrac_total = 0.0
        n_updates = 0

        for _ in range(K_epochs):
            kl_triggered = False  # FIX P1-4: epoch-level KL early-stop
            indices = torch.randperm(n_transitions, device=device)
            for start in range(0, n_transitions, mini_batch_size):
                idx = indices[start:start + mini_batch_size]
                if len(idx) < 2:
                    continue

                if skip_actor:
                    # E039 (F5, 设计 v1.1 §4.3): ep_sr==0 失败主导批次 → 只训 critic
                    # （保拟合），actor 不动。跳过 logp 重算/actor_loss/bc_loss
                    # （最贵的计算也在 skip 范围，失效批次不白烧 GPU）。
                    V_pred = value_head(obs_encs_192[idx]).squeeze(-1)
                    returns_batch = returns[idx]
                    if returns_batch.dim() > 1:
                        returns_batch = returns_batch.squeeze(-1)
                    value_loss = nn.functional.mse_loss(V_pred, returns_batch)
                    value_optimizer.zero_grad()
                    value_loss.backward()
                    if self.max_grad_norm > 0:
                        nn.utils.clip_grad_norm_(value_head.parameters(), self.max_grad_norm)
                    value_optimizer.step()
                    value_loss_total += value_loss.item()
                    n_updates += 1
                    continue

                chains_batch = []
                for k in range(num_inference_steps + 1):
                    chains_batch.append(torch.stack(
                        [buffer.a_chain_list[i][k].to(device) for i in idx]))

                # Joint (γ-weighted) logp → (M,) scalar
                logp_new = self._recompute_joint_logp_batched(
                    chains_batch, obs_encs_192[idx],
                    model, scheduler,
                    obs_all[idx], obs_uncond_all[idx],
                    cfg_lambda, num_inference_steps,
                    eta, sigma_clip_min, sigma_clip_max,
                    ft_denoising_steps=ft_denoising_steps)  # (M,)

                logp_old_mb = logp_old[idx]  # (M,)

                # FIX P1-2: Advantage already normalized over full batch above
                adv_mb = advantages[idx]
                # E036 Fix-1 (R-B, probe3 定案 2026-08-03):
                #   移除 mini-batch 二次 meritocracy（原文件此处将 advantage 再乘
                #   w²∈[0.01,1.0]）—— 探针 single_merit 变体证明它是 KL 单调衰减
                #   的直接机制；第一次应用（full-batch z-score 后）保留。

                logp_diff = torch.clamp(logp_new - logp_old_mb, -20.0, 20.0)
                if self.logp_mean_norm:
                    # P0: 176 维求和 → 维度均值（官方 DDPO/DPPO 结构），KL/ratio 尺度正常化
                    logp_diff = logp_diff / (horizon * action_dim)
                ratio = torch.exp(logp_diff)  # (M,)
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * adv_mb
                actor_loss = -torch.min(surr1, surr2).mean()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - torch.log(ratio + 1e-8)).mean()
                    clipfrac = ((ratio < 1.0 - self.clip_epsilon) |
                                (ratio > 1.0 + self.clip_epsilon)).float().mean()

                # FIX P1-3/4: KL threshold 1.5× (standard PPO), epoch-level early-stop
                if approx_kl > 1.5 * self.target_kl:
                    kl_triggered = True
                    break

                V_pred = value_head(obs_encs_192[idx]).squeeze(-1)
                returns_batch = returns[idx]
                if returns_batch.dim() > 1:
                    returns_batch = returns_batch.squeeze(-1)
                value_loss = nn.functional.mse_loss(V_pred, returns_batch)

                # BC regularization: joint logp on BC chain
                bc_loss = torch.tensor(0.0, device=device)
                if self.w_bc > 0 and self.bc_model is not None:
                    x_T_batch = chains_batch[0].to(device)
                    with torch.no_grad():
                        bc_chains_batch = self._generate_deterministic_chain(
                            x_T_batch, self.bc_model.model, scheduler,
                            obs_all[idx], obs_uncond_all[idx],
                            cfg_lambda, num_inference_steps)
                    bc_logp = self._recompute_joint_logp_batched(
                        bc_chains_batch, obs_encs_192[idx],
                        model, scheduler,
                        obs_all[idx], obs_uncond_all[idx],
                        cfg_lambda, num_inference_steps,
                        eta, sigma_clip_min, sigma_clip_max,
                        ft_denoising_steps=ft_denoising_steps)  # (M,)
                    bc_loss = -bc_logp.mean()

                total_loss = actor_loss + 0.5 * value_loss + self.w_bc * bc_loss

                actor_optimizer.zero_grad()
                value_optimizer.zero_grad()
                total_loss.backward()
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                    nn.utils.clip_grad_norm_(value_head.parameters(), self.max_grad_norm)
                actor_optimizer.step()
                value_optimizer.step()

                actor_loss_total += actor_loss.item()
                value_loss_total += value_loss.item()
                approx_kl_total += approx_kl.item()
                clipfrac_total += clipfrac.item()
                n_updates += 1

            # FIX P1-4: epoch-level KL early-stop — if any mini-batch triggered KL, skip remaining epochs
            if kl_triggered:
                break

        if n_updates == 0:
            n_updates = 1

        return {
            'actor_loss': actor_loss_total / n_updates,
            'value_loss': value_loss_total / n_updates,
            'approx_kl': approx_kl_total / n_updates,
            'clipfrac': clipfrac_total / n_updates,
            'entropy': 0.0,
            'n_updates': n_updates,
        }


# ═══════════════════════════════════════════════════════════════════
# Eval helper
# ═══════════════════════════════════════════════════════════════════

def run_eval(policy, env, step, device):
    """Deterministic eval (η=0 DDIM)."""
    policy.eval()
    with torch.no_grad():
        try:
            obs_dict, _ = env.reset()
            done, total_r, steps = False, 0.0, 0
            while not done and steps < 200:
                obs_b = {}
                for k, v in obs_dict.items():
                    t = torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else v.to(device)
                    obs_b[k] = t.unsqueeze(0)
                out = policy.predict_action(obs_b)
                act = out['action_pred'] if isinstance(out, dict) else out
                obs_dict, reward, terminated, truncated, _ = env.step(
                    act.squeeze(0).cpu().numpy())
                done = terminated or truncated
                total_r += float(reward)
                steps += 1
            collision = getattr(env, '_collision_detected', False)
            reached = terminated and not collision
            print(f"[EVAL step={step}] r={total_r:.1f} steps={steps} "
                  f"collision={collision} reached={reached}")
        except Exception as e:
            import traceback
            print(f"[EVAL step={step}] ERROR: {e}")
            traceback.print_exc()
    policy.train()


# ═══════════════════════════════════════════════════════════════════
# Parallel episode runner (E031-accel: 4 envs, CPU overlap)
# ═══════════════════════════════════════════════════════════════════

def run_episode(env, policy, value_head, device, args):
    """Run one episode on `env` (thread-parallel). Returns (transitions, stats).

    Transitions: list of tuples matching PPORolloutBuffer.add() args.
    GPU inference stays per-env (batch=1) but overlaps with other envs'
    CPU work (SynthPointCloud / physics) across threads.
    """
    transitions = []
    with torch.no_grad():
        obs_dict, _ = env.reset()
        ep_rewards, ep_logps = [], []
        ep_return = 0.0
        ep_reached = False

        for step_i in range(args.max_ep_len):
            global_cond = encode_obs(obs_dict, policy, device)
            global_cond_uncond = encode_obs_uncond(obs_dict, policy, device)
            obs_enc_192 = global_to_obs_enc(global_cond, policy.n_obs_steps, 192)
            V_old = value_head(obs_enc_192)

            obs_b = {}
            for k, v in obs_dict.items():
                t = torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else v.to(device)
                obs_b[k] = t.unsqueeze(0)

            action, logp, chain = policy.sample_ddpm_with_logprob(
                obs_b, eta=args.eta,
                sigma_clip_min=args.sigma_clip_min,
                sigma_clip_max=args.sigma_clip_max,
                ft_denoising_steps=args.ft_denoising_steps)

            # FIX P0-1: unnormalize before env.step() (physical action space)
            action = policy.normalizer['action'].unnormalize(action)
            action_np = action.squeeze(0).cpu().numpy()
            next_obs_dict, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            next_global_cond = encode_obs(next_obs_dict, policy, device)

            transitions.append((
                global_cond.squeeze(0).cpu(),
                global_cond_uncond.squeeze(0).cpu(),
                [ac.squeeze(0).cpu() for ac in chain],
                logp.squeeze(0).detach().cpu(),
                float(reward), bool(done),
                V_old.squeeze().detach().cpu(),
                next_global_cond.squeeze(0).cpu(),
            ))

            ep_rewards.append(float(reward))
            ep_logps.append(logp.mean().item())
            ep_return += float(reward)
            if terminated and not getattr(env, '_collision_detected', False):
                ep_reached = True
            obs_dict = next_obs_dict
            if done:
                break

    stats = (
        np.mean(ep_rewards) if ep_rewards else 0.0,
        np.mean(ep_logps) if ep_logps else 0.0,
        ep_return,
        ep_reached,
    )
    return transitions, stats


def run_episodes_batched(envs, policy, value_head, device, args):
    """Run one episode per env, synchronously, with batch=N GPU inference.

    E031-accel: replaces N×batch=1 GPU calls (which serialize on the CUDA
    context lock even across threads) with ONE batched forward per step
    (encode + value + sample_ddpm_with_logprob). Env stepping stays on CPU
    threads. Per-env transition/stats semantics identical to run_episode.
    Returns (transitions_list, stats_list) — one episode result per env.
    """
    n = len(envs)
    transitions = [[] for _ in range(n)]
    ep_rewards = [[] for _ in range(n)]
    ep_logps = [[] for _ in range(n)]
    ep_returns = [0.0] * n
    ep_reached = [False] * n
    done = [False] * n
    obs_list = [e.reset()[0] for e in envs]
    prev_enc = [None] * n  # pending next_obs encoding (filled next iteration)

    with torch.no_grad():
        for _step_i in range(args.max_ep_len):
            active = [i for i in range(n) if not done[i]]
            if not active:
                break

            # ── one batched GPU pass over all active envs ──
            obs_b = _stack_obs_batch([obs_list[i] for i in active], device)
            global_cond = encode_obs_batch(obs_b, policy)             # (A, -1)
            global_cond_uncond = encode_obs_uncond_batch(obs_b, policy)
            obs_enc_192 = global_to_obs_enc(global_cond, policy.n_obs_steps, 192)
            V_old = value_head(obs_enc_192)                           # (A, 1)
            action, logp, chain = policy.sample_ddpm_with_logprob(
                obs_b, eta=args.eta,
                sigma_clip_min=args.sigma_clip_min,
                sigma_clip_max=args.sigma_clip_max,
                ft_denoising_steps=args.ft_denoising_steps)           # (A,H,11),(A,),[(A,H,11)]
            action = policy.normalizer['action'].unnormalize(action)  # (A,H,11)

            # ── CPU env stepping in parallel (physics/synth overlap) ──
            with ThreadPoolExecutor(max_workers=len(active)) as ex:
                futs = {ex.submit(envs[i].step, action[k].cpu().numpy()): (k, i)
                        for k, i in enumerate(active)}
                for f in as_completed(futs):
                    k, i = futs[f]
                    next_obs, reward, terminated, truncated, _ = f.result()
                    is_done = terminated or truncated
                    if prev_enc[i] is not None:
                        # this step's encoding = previous transition's next_obs
                        transitions[i][-1][7] = prev_enc[i].cpu()
                    if is_done:
                        next_gc = encode_obs(next_obs, policy, device).squeeze(0).cpu()
                        prev_enc[i] = None
                    else:
                        next_gc = None
                        prev_enc[i] = global_cond[k]
                    transitions[i].append([
                        global_cond[k].cpu(),
                        global_cond_uncond[k].cpu(),
                        [c[k].cpu() for c in chain],
                        logp[k].detach().cpu(),
                        float(reward), bool(is_done),
                        V_old[k].squeeze().detach().cpu(),
                        next_gc,
                    ])
                    ep_rewards[i].append(float(reward))
                    ep_logps[i].append(logp[k].item())
                    ep_returns[i] += float(reward)
                    if terminated and not getattr(envs[i], '_collision_detected', False):
                        ep_reached[i] = True
                    obs_list[i] = next_obs
                    if is_done:
                        done[i] = True

    stats_list = []
    for i in range(n):
        stats_list.append((
            np.mean(ep_rewards[i]) if ep_rewards[i] else 0.0,
            np.mean(ep_logps[i]) if ep_logps[i] else 0.0,
            ep_returns[i],
            ep_reached[i],
        ))
    return [[tuple(t) for t in tr] for tr in transitions], stats_list


def build_envs(args, _PROJECT_ROOT):
    """Build env pool + N parallel env instances (shared world/synth pools)."""
    from diffusion_policy_3d.env.kinematic_uav_env_v2 import KinematicUAVEnv
    from diffusion_policy_3d.rl.routea_rl_env_adapter_v1 import (
        load_routea_rl_admission_manifest, build_role_buckets)
    from diffusion_policy_3d.model.common.synth_pointcloud import SynthPointCloud

    manifest_path = os.path.join(_PROJECT_ROOT, args.admission_manifest)
    manifest = load_routea_rl_admission_manifest(manifest_path)
    auv_root = os.path.expanduser("~/AUV")
    worlds_dir = os.path.join(auv_root, "PX4-Autopilot/Tools/sitl_gazebo/worlds/train")

    train_buckets = build_role_buckets(manifest, "rl_train")

    role_worlds = {}
    synth_pc_pool = {}

    for role in ['anticipatory', 'dense_recovery', 'support_expansion', 'yaw_lateral_recovery']:
        role_worlds[role] = []
        seen = set()

        for entry in train_buckets.get(role, []):
            name = entry.world_uid
            json_path = os.path.join(worlds_dir, f"{name}.world.obstacles.json")
            if os.path.exists(json_path) and name not in seen:
                role_worlds[role].append((name, json_path))
                synth_pc_pool[name] = SynthPointCloud(obstacles_json=json_path)
                seen.add(name)

    # Build balanced pool (P4a): equal probability per role via oversampling
    world_pool = []
    target_per_role = max(len(v) for v in role_worlds.values())
    for role, worlds in role_worlds.items():
        if not worlds:
            continue
        repeats = target_per_role // len(worlds) + 1
        world_pool.extend((worlds * repeats)[:target_per_role])

    total_unique = sum(len(v) for v in role_worlds.values())
    print(f"[ENV] Expanded pool: {total_unique} unique worlds ({len(world_pool)} balanced, "
          f"target ~{target_per_role}/role)")
    for role, worlds in role_worlds.items():
        print(f"  {role}: {len(worlds)} worlds")

    n_envs = max(1, args.n_env_parallel)
    envs = []
    for i in range(n_envs):
        e = KinematicUAVEnv(
            world_pool=world_pool, synth_pc_pool=synth_pc_pool,
            curriculum_stage='C2', max_episode_steps=args.max_ep_len,
            reward_version=args.reward_version)
        envs.append(e)
    print(f"[ENV] {len(envs)} KinematicUAVEnv instances created, reward={args.reward_version}")
    return envs


# ═══════════════════════════════════════════════════════════════════
# Main Training Loop
# ═══════════════════════════════════════════════════════════════════

def _save_checkpoint(policy, value_head, actor_optimizer, value_optimizer,
                     lora_info, args, out_dir: str, step: int) -> None:
    """E036 Add-2: 中间/最终 ckpt 保存（LoRA 独立 state_dict，原 final 块复用）。

    step=0 表示最终保存（命名 ddpm_ppo_final.pt），否则 ddpm_ppo_step{step}.pt。
    """
    name = "ddpm_ppo_final.pt" if step == 0 else f"ddpm_ppo_step{step}.pt"
    save_path = os.path.join(out_dir, name)
    save_dict = {
        "step": step,
        "model_state_dict": policy.model.state_dict(),
        "value_head_state_dict": value_head.state_dict(),
        "actor_optimizer": actor_optimizer.state_dict(),
        "value_optimizer": value_optimizer.state_dict(),
        "config": {
            "horizon": policy.horizon, "action_dim": policy.action_dim,
            "n_action_steps": policy.n_action_steps,
            "n_obs_steps": policy.n_obs_steps,
            "num_inference_steps": args.num_inference_steps,
            "cfg_lambda": policy.cfg_lambda,
        },
    }
    if lora_info and lora_info.get("lora_modules") is not None:
        save_dict["lora_state_dict"] = lora_info["lora_modules"].state_dict()
        save_dict["lora_rank"] = args.lora_rank
        save_dict["lora_alpha"] = args.lora_alpha
    # G8 (E038): RNG 状态（Rescue 001 教训；无 seed → 重启随机重放，RNG 供诊断）
    save_dict["rng_state"] = {
        "torch": torch.get_rng_state().tolist(),
        "torch_cuda": [s.tolist() for s in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available() else [],
        "numpy": np.random.get_state(),
        "random": random.getstate(),
    }
    torch.save(save_dict, save_path)
    print(f"[SAVE] {save_path}")


def run_training(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    # 1. Load DDPM BC policy
    ckpt_path = os.path.join(_PROJECT_ROOT, args.bc_ckpt)
    policy = load_ddpm_policy(ckpt_path, device=str(device))
    if args.cfg_lambda is not None:
        policy.cfg_lambda = args.cfg_lambda
        print(f"[CFG] cfg_lambda = {args.cfg_lambda}")
    policy.obs_encoder.requires_grad_(False)
    print("[FREEZE] obs_encoder frozen")

    # Print σ_t sequence (design §6.2 step 0a)
    scheduler = policy.noise_scheduler
    scheduler.set_timesteps(args.num_inference_steps)
    print(f"[SCHEDULER] timesteps: {scheduler.timesteps.tolist()}")
    print(f"[SCHEDULER] num_train_timesteps={scheduler.config.num_train_timesteps}, "
          f"prediction_type={scheduler.config.prediction_type}")

    # LoRA
    lora_info = None
    if args.lora:
        from diffusion_policy_3d.policy.lora_adapters import insert_lora
        lora_info = insert_lora(policy.model, r=args.lora_rank, alpha=args.lora_alpha)
        policy.model.requires_grad_(False)
        for p in lora_info["lora_params"]:
            p.requires_grad_(True)
        print(f"[FREEZE] UNet frozen, {len(lora_info['lora_params'])} LoRA params trainable")

    # 2. Value Head (No NoiseNetwork needed — DDPM scheduler gives σ_t)
    value_head = ValueHead(input_dim=192).to(device).train()
    print(f"[INIT] ValueHead: {sum(p.numel() for p in value_head.parameters()):,} params")

    # 3. Optimizers
    actor_params = lora_info["lora_params"] if lora_info else [p for p in policy.model.parameters() if p.requires_grad]
    actor_optimizer = torch.optim.Adam(actor_params, lr=args.lr_actor)
    value_optimizer = torch.optim.Adam(value_head.parameters(), lr=args.lr_value)
    print(f"[OPTIM] Actor: {sum(p.numel() for p in actor_params):,} params, LR={args.lr_actor}")

    # 4. PPO Trainer (DPPO per-step)
    # 4a. BC regularization: load frozen BC model (no LoRA) for reference actions
    bc_model = None
    if args.w_bc > 0:
        bc_model = load_ddpm_policy(ckpt_path, device=str(device))
        bc_model.obs_encoder.requires_grad_(False)
        bc_model.model.requires_grad_(False)
        bc_model.eval()
        # Override cfg_lambda to match training policy
        if args.cfg_lambda is not None:
            bc_model.cfg_lambda = args.cfg_lambda
        print(f"[BC_REG] Frozen BC model loaded, w_bc={args.w_bc}")
        print(f"[BC_REG] BC model params: {sum(p.numel() for p in bc_model.parameters()):,} (all frozen)")

    ppo_trainer = DPPOTrainer(
        clip_epsilon=args.clip_epsilon, target_kl=args.target_kl,
        entropy_alpha=args.entropy_alpha, max_grad_norm=args.max_grad_norm,
        gamma_denoise=args.gamma_denoise,
        eps_base=args.eps_base, eps_final=args.eps_final, eps_beta=args.eps_beta,
        bc_model=bc_model, w_bc=args.w_bc,
        logp_mean_norm=args.logp_mean_norm)

    # 5. Load reward norm stats
    reward_norm_path = os.path.join(_PROJECT_ROOT, args.reward_norm)
    if os.path.exists(reward_norm_path):
        with open(reward_norm_path) as f:
            rn = json.load(f)
        ppo_trainer.reward_mean = rn.get("total_mean", 0.0)
        ppo_trainer.reward_std = max(rn.get("total_std", 1.0), 1.0)
        print(f"[REWARD_NORM] mean={ppo_trainer.reward_mean:.2f} std={ppo_trainer.reward_std:.2f}")

    # 6. Build env(s) — E031-accel: N parallel instances sharing world/synth pools
    env = None
    if not args.dry_run:
        # E038 G1: 安装 terms 收集 hook（必须在 build_envs 前，类级 patch）
        _install_terms_collector()
        envs = build_envs(args, _PROJECT_ROOT)
        env = envs[0]

    # E036 Add-2: 输出目录提前创建（中间 ckpt 复用）
    out_dir = os.path.join(_PROJECT_ROOT, args.output_dir)
    if not args.dry_run:
        os.makedirs(out_dir, exist_ok=True)

    # 7. Training loop
    print(f"\n{'='*60}\nTRAINING START: mode={args.mode}, max_updates={args.max_updates}\n{'='*60}\n")

    buffer = PPORolloutBuffer()
    # Batch accumulation: persist across update_step until target_transitions reached.
    # P0-1 fix: increases effective N from ~50 to target (2048+) → reliable KL, stable advantage.
    ep_return_list = []     # accumulates across rounds
    ep_success_list = []    # accumulates across rounds
    collection_steps = 0
    # E039 (F5, 设计 v1.1 §4.3): 条件变量状态机——默认关，激活即 E039b 子变量
    f5_active = False
    f5_hist = []  # (update_step, ep_sr, r_prog) 最近 5 updates，供激活判定

    _t_step = time.time()  # E031-accel: per-step wall clock
    for update_step in range(args.max_updates):
        # E039 P1-1 (code-reviewer): 循环开头清空 terms collector——eval 回合
        # （fixed-step 触发，在上一轮 update 块之后）也会经 hooked 环境写入
        # _TERMS_COLLECTOR；不清空则 F5 激活读取/[comp] 打印混入 eval 分量，
        # r_prog 符号可能被翻转（连败区间典型近 0）。收集前清空保证只含本轮训练。
        _TERMS_COLLECTOR.clear()
        if args.dry_run:
            print(f"[{update_step+1:4d}/{args.max_updates}] DRY RUN")
            continue

        # Collect episodes (data accumulation only — PPO fires separately)
        all_rewards, all_logps = [], []
        total_steps = 0

        # E031-accel: parallel episode collection over N envs (CPU overlap)
        n_par = max(1, args.n_env_parallel)
        if args.batched_rollout:
            # E031-accel v2: sync waves of n_par envs, batch=n_par GPU inference
            # (device utilization: one forward replaces n_par serialized batch=1)
            for batch_start in range(0, args.n_eps_per_batch, n_par):
                batch_n = min(n_par, args.n_eps_per_batch - batch_start)
                trs_list, stats_list = run_episodes_batched(
                    envs[:batch_n], policy, value_head, device, args)
                for transitions, stats in zip(trs_list, stats_list):
                    for tr in transitions:
                        buffer.add(*tr)
                    all_rewards.append(stats[0])
                    all_logps.append(stats[1])
                    ep_return_list.append(stats[2])
                    ep_success_list.append(stats[3])
                    total_steps += len(transitions)
        else:
            with ThreadPoolExecutor(max_workers=n_par) as ex:
                for batch_start in range(0, args.n_eps_per_batch, n_par):
                    batch_n = min(n_par, args.n_eps_per_batch - batch_start)
                    futs = [ex.submit(run_episode, envs[i], policy, value_head, device, args)
                            for i in range(batch_n)]
                    for f in as_completed(futs):
                        transitions, stats = f.result()
                        for tr in transitions:
                            buffer.add(*tr)
                        all_rewards.append(stats[0])
                        all_logps.append(stats[1])
                        ep_return_list.append(stats[2])
                        ep_success_list.append(stats[3])
                        total_steps += len(transitions)

        collection_steps += total_steps

        # PPO Update: only when enough transitions accumulated (batch accumulation)
        if len(buffer) >= args.target_transitions:
            ppo_trainer.update_reward_stats(
                [float(r) for r in buffer.reward_list])

            # Continuous meritocracy across ALL accumulated episodes
            ep_returns_arr = np.array(ep_return_list, dtype=np.float32)
            r_min, r_max = ep_returns_arr.min(), ep_returns_arr.max()
            if r_max > r_min + 0.01:
                ep_weights = 0.1 + 0.9 * (ep_returns_arr - r_min) / (r_max - r_min)
            else:
                ep_weights = np.ones_like(ep_returns_arr)

            advantage_weight = []
            ep_idx = 0
            for i in range(len(buffer.reward_list)):
                if i > 0 and buffer.done_list[i-1]:
                    ep_idx += 1
                w = float(ep_weights[min(ep_idx, len(ep_weights) - 1)])
                advantage_weight.append(w)

            # E039（设计 v1.1 §4.1）: w_bc 衰减放手——w(t) = w0 · max(0, 1 − t/T_decay)。
            # 锚定期（探针门 60 + E038 阶段 A 窗口 70 + 余量）内策略在 BC 附近安全尝试；
            # 放手后由 KL 窗 + PPO clip 维持稳定，不加回（避免调参，§4.1 放手期风险）。
            w_bc_t = args.w_bc * max(0.0, 1.0 - update_step / max(1, args.w_bc_decay_updates))
            ppo_trainer.w_bc = w_bc_t

            # E039（F5）: ep_sr 提前计算（ep_success_list 已在收集阶段累积，
            # 移到 update() 之前供 skip 判定；打印复用在原位置）
            ep_sr = np.mean(ep_success_list) if ep_success_list else 0.0

            # E039（F5）: 条件变量状态机（默认关，--f5_enabled 开启；设计 v1.1 §4.3）
            # - 激活条件: ep_sr 连 5 < 0.05 且 r_prog 负 → 启用并记为 E039b 子变量
            # - 执行条件: 每 batch ep_sr == 0（严格零成功）→ 跳过 actor 更新
            skip_actor = False
            if args.f5_enabled:
                _rp_now = float(np.mean([t.get("progress", 0.0) for t, m in _TERMS_COLLECTOR])) \
                    if _TERMS_COLLECTOR else None
                if _rp_now is not None:
                    f5_hist.append((update_step, ep_sr, _rp_now))
                    f5_hist[:] = f5_hist[-5:]  # 只保留最近 5 updates（激活判定窗）
                    # P2-1 (code-reviewer): 激活判定限定在衰减窗内（设计 §4.3
                    # "探针门/衰减窗期间"）；激活后执行（ep_sr==0 跳过）不受限
                    if (not f5_active
                            and update_step <= args.w_bc_decay_updates
                            and len(f5_hist) >= 5
                            and all(e < 0.05 for _, e, _ in f5_hist)
                            and f5_hist[-1][2] < 0.0):
                        f5_active = True
                        print(f"    [F5] ACTIVATED (E039b): ep_sr 连 5 <0.05 且 r_prog 负 "
                              f"→ actor 跳过机制接管（ep_sr 探针判据移交 F5）", flush=True)
                if f5_active and ep_sr == 0.0:
                    skip_actor = True

            metrics = ppo_trainer.update(
                buffer=buffer, model=policy.model,
                scheduler=policy.noise_scheduler,
                value_head=value_head,
                actor_optimizer=actor_optimizer,
                value_optimizer=value_optimizer,
                horizon=policy.horizon, action_dim=policy.action_dim,
                num_inference_steps=args.num_inference_steps,
                n_obs_steps=policy.n_obs_steps,
                cfg_lambda=policy.cfg_lambda,
                eta=args.eta,
                sigma_clip_min=args.sigma_clip_min,
                sigma_clip_max=args.sigma_clip_max,
                ft_denoising_steps=args.ft_denoising_steps,
                K_epochs=args.K_epochs,
                mini_batch_size=args.mini_batch_size,
                advantage_weight=advantage_weight,
                skip_actor=skip_actor,
            )
            # FIX P2-1: Compute SR BEFORE clearing ep_success_list（E039 已前移，
            # 此处不再重复计算，保留变量供日志）
            buffer.clear()
            ep_return_list.clear()
            ep_success_list.clear()

            mean_r = np.mean(all_rewards)
            mean_lp = np.mean(all_logps)
            dt = time.time() - _t_step
            _t_step = time.time()
            print(f"[{update_step+1:4d}/{args.max_updates}] "
                  f"a_loss={metrics['actor_loss']:.4f} "
                  f"v_loss={metrics['value_loss']:.4f} "
                  f"kl={metrics['approx_kl']:.6f} "
                  f"cf={metrics['clipfrac']:.3f} nu={metrics['n_updates']} "
                  f"r={mean_r:.2f} logp={mean_lp:.2f} ep_sr={ep_sr:.2f} "
                  f"N={len(buffer.reward_list)} steps={collection_steps} "
                  f"T={dt:.0f}s w_bc={w_bc_t:.4f} f5={'skip' if skip_actor else ('on' if f5_active else 'off')}")
            # E038 Add-1（G1 日志替代源）: reward 分量逐轮日志——_TERMS_COLLECTOR
            # 由纯收集 hook 写入；r_safety 已删（F6）→ r_safe 列移除；d_soft 供 monitor
            if _TERMS_COLLECTOR:
                _rp = [t.get("progress", 0.0) for t, m in _TERMS_COLLECTOR]
                _rm = [t.get("smooth", 0.0) for t, m in _TERMS_COLLECTOR]
                _ds = [m.get("d_soft_h", -1.0) for t, m in _TERMS_COLLECTOR]
                print(f"    [comp] r_prog={np.mean(_rp):+.3f} "
                      f"r_smoo={np.mean(_rm):+.3f} d_soft={np.mean(_ds):.2f}")
                _TERMS_COLLECTOR.clear()
            collection_steps = 0

            # E036 Add-2: 中间 ckpt（E031 教训：无中间 ckpt 无法加载训练中策略）
            if (update_step + 1) % args.save_freq == 0:
                _save_checkpoint(policy, value_head, actor_optimizer, value_optimizer,
                                 lora_info, args, out_dir, update_step + 1)
        else:
            # Collection progress only (no PPO yet)
            if (update_step + 1) % 10 == 0:
                print(f"[{update_step+1:4d}/{args.max_updates}] collecting... "
                      f"buffer={len(buffer)}/{args.target_transitions} "
                      f"eps={len(ep_return_list)} steps={collection_steps}")

        # G11 (E038): eval 与 update 解耦——固定步数触发，不依赖 buffer 分支
        # （E036/E037 教训：eval 原绑 update 分支内，collecting 轮时 eval_freq
        #  错过 → 日志 0 行 [EVAL]；现每轮检查，fixed-step 触发）
        if (update_step + 1) % args.eval_freq == 0:
            run_eval(policy, env, update_step + 1, device)

    # 8. Save final (save state_dicts, including LoRA which is NOT in model.state_dict())
    if not args.dry_run:
        _save_checkpoint(policy, value_head, actor_optimizer, value_optimizer,
                         lora_info, args, out_dir, 0)

    print(f"\n{'='*60}\nTRAINING COMPLETE\n{'='*60}")


# ═══════════════════════════════════════════════════════════════════
# Args
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="DPPO: DDPM Diffusion Policy + PPO")
    p.add_argument("--mode", default="smoke", choices=["smoke", "formal", "dry_run"])
    p.add_argument("--bc_ckpt", default=DEFAULT_BC_CKPT)
    p.add_argument("--admission_manifest", default=DEFAULT_ADMISSION_MANIFEST)
    p.add_argument("--reward_norm", default=DEFAULT_REWARD_NORM)
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--max_updates", type=int, default=None)
    p.add_argument("--max_ep_len", type=int, default=100,
                   help="Max episode steps (100 = 2s@50Hz; shorter = less survival bias, cleaner goal signal)")
    p.add_argument("--n_eps_per_batch", type=int, default=None)
    p.add_argument("--n_env_parallel", type=int, default=4,
                   help="E031-accel: parallel env instances for episode collection (CPU overlap)")
    p.add_argument("--K_epochs", type=int, default=None)
    p.add_argument("--mini_batch_size", type=int, default=None)
    p.add_argument("--eval_freq", type=int, default=None)
    p.add_argument("--save_freq", type=int, default=100,
                   help="E038 G8: 中间 ckpt 保存间隔（0=不存中间；500→100，Rescue 001 教训）")
    p.add_argument("--lr_actor", type=float, default=3e-4)  # E027c confirmed: 3e-4 > 1e-4
    p.add_argument("--lr_value", type=float, default=1e-3)
    p.add_argument("--clip_epsilon", type=float, default=0.2)
    p.add_argument("--target_kl", type=float, default=0.015)
    p.add_argument("--entropy_alpha", type=float, default=0.005)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--sigma_clip_min", type=float, default=0.01)
    p.add_argument("--sigma_clip_max", type=float, default=10.0,
                   help="DDIM noise sigma max (10.0 for per-step logp diversity)")
    p.add_argument("--ft_denoising_steps", type=int, default=5,
                   help="Number of last denoising steps to fine-tune (0=all). DPPO/RL-100 recommend 5.")
    p.add_argument("--cfg_lambda", type=float, default=None)
    p.add_argument("--reward_version", default="v16")
    p.add_argument("--lora", action="store_true", default=True)
    p.add_argument("--lora_rank", type=int, default=16)   # DPPO Phase 2: 4→16
    p.add_argument("--lora_alpha", type=int, default=16)  # DPPO Phase 2: 4→16
    # P0 (2026-08-01): logp 维度均值化（176 维求和 → 除以 horizon×action_dim）
    p.add_argument("--logp_mean_norm", action="store_true", default=False)
    p.add_argument("--batched_rollout", action="store_true", default=False,
                   help="E031-accel v2: sync N-env waves, batch=N GPU inference "
                        "(device utilization; default off keeps thread rollout)")
    # DPPO per-step (Ren et al., ICLR 2025)
    p.add_argument("--gamma_denoise", type=float, default=0.8)
    p.add_argument("--eps_base", type=float, default=1e-3)
    p.add_argument("--eps_final", type=float, default=0.2)
    p.add_argument("--eps_beta", type=float, default=3.0)
    p.add_argument("--w_bc", type=float, default=0.0,
                   help="BC regularization weight (0.0=disabled, 0.01=DPPO recommended)")
    # E039 (设计 v1.1 §4.1): w_bc 衰减放手时间表——w(t) = w0·max(0, 1−t/T_decay)
    p.add_argument("--w_bc_decay_updates", type=int, default=200,
                   help="E039: w_bc 线性衰减至 0 的 update 数（T_decay 预注册=200）")
    # E039 (设计 v1.1 §4.3): F5 条件变量——默认关；激活 = E039b 子变量
    p.add_argument("--f5_enabled", action="store_true", default=False,
                   help="E039: F5 条件变量开关（ep_sr==0 跳过 actor update；"
                        "激活条件 ep_sr 连 5 <0.05 且 r_prog 负）")
    p.add_argument("--target_transitions", type=int, default=2048,
                   help="Batch accumulation target (P0-1 fix: N≥2048 for reliable PPO stats)")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    # [ASSUMPTION-9] formal-mode rejection protection: legacy
    # E-series entry points may not run formal training (only the
    # A-Official-Core dppo_uav path may).
    guard_formal_mode("scripts/train_ddpm_ppo_meanlogp_3_e039.py", args.mode)


    if args.mode == "smoke":
        args.max_updates = args.max_updates or 200
        args.n_eps_per_batch = args.n_eps_per_batch or 64   # DPPO Phase 2: 16→64
        args.K_epochs = args.K_epochs or 10
        args.mini_batch_size = args.mini_batch_size or 256  # DPPO Phase 2: 64→256
        args.eval_freq = args.eval_freq or 50
    elif args.mode == "formal":
        args.max_updates = args.max_updates or 1000
        args.n_eps_per_batch = args.n_eps_per_batch or 128  # DPPO Phase 2: 32→128
        args.K_epochs = args.K_epochs or 10
        args.mini_batch_size = args.mini_batch_size or 256
        args.eval_freq = args.eval_freq or 200
    else:
        args.max_updates = args.max_updates or 1
        args.n_eps_per_batch = 1
        args.K_epochs = 1
        args.mini_batch_size = 64
        args.eval_freq = 1
        args.dry_run = True
    return args


if __name__ == "__main__":
    args = parse_args()
    print("[CONFIG]")
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")
    run_training(args)
