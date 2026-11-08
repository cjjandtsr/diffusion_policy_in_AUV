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
import argparse, json, os, sys
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
                 bc_model=None, w_bc=0.0):
        self.clip_epsilon = clip_epsilon
        self.target_kl = target_kl
        self.entropy_alpha = entropy_alpha
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.max_grad_norm = max_grad_norm
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
               advantage_weight=None):
        """Standard PPO on γ-weighted joint logp (scalar ratio, constant ε=0.2).

        γ_denoise weights early denoising steps less in the logp sum → the PPO
        ratio naturally focuses credit on later (more important) steps. BC reg
        uses per-step logp internally to anchor each step independently.
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
        n_updates = 0

        for _ in range(K_epochs):
            kl_triggered = False  # FIX P1-4: epoch-level KL early-stop
            indices = torch.randperm(n_transitions, device=device)
            for start in range(0, n_transitions, mini_batch_size):
                idx = indices[start:start + mini_batch_size]
                if len(idx) < 2:
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
                if advantage_weight is not None:
                    w_mb = torch.tensor([advantage_weight[i] for i in idx.tolist()],
                                        device=device, dtype=torch.float32)
                    adv_mb = adv_mb * w_mb

                logp_diff = torch.clamp(logp_new - logp_old_mb, -20.0, 20.0)
                ratio = torch.exp(logp_diff)  # (M,)
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * adv_mb
                actor_loss = -torch.min(surr1, surr2).mean()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - torch.log(ratio + 1e-8)).mean()

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
# Main Training Loop
# ═══════════════════════════════════════════════════════════════════

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
        bc_model=bc_model, w_bc=args.w_bc)

    # 5. Load reward norm stats
    reward_norm_path = os.path.join(_PROJECT_ROOT, args.reward_norm)
    if os.path.exists(reward_norm_path):
        with open(reward_norm_path) as f:
            rn = json.load(f)
        ppo_trainer.reward_mean = rn.get("total_mean", 0.0)
        ppo_trainer.reward_std = max(rn.get("total_std", 1.0), 1.0)
        print(f"[REWARD_NORM] mean={ppo_trainer.reward_mean:.2f} std={ppo_trainer.reward_std:.2f}")

    # 6. Build env
    env = None
    if not args.dry_run:
        from diffusion_policy_3d.env.kinematic_uav_env_v2 import KinematicUAVEnv
        from diffusion_policy_3d.rl.routea_rl_env_adapter_v1 import (
            load_routea_rl_admission_manifest, build_role_buckets)
        from diffusion_policy_3d.model.common.synth_pointcloud import SynthPointCloud

        manifest_path = os.path.join(_PROJECT_ROOT, args.admission_manifest)
        manifest = load_routea_rl_admission_manifest(manifest_path)
        auv_root = os.path.expanduser("~/AUV")
        worlds_dir = os.path.join(auv_root, "PX4-Autopilot/Tools/sitl_gazebo/worlds/train")

        # Unified world pool: all worlds in manifest are rl_train (292 total)
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

        env = KinematicUAVEnv(
            world_pool=world_pool, synth_pc_pool=synth_pc_pool,
            curriculum_stage='C2', max_episode_steps=args.max_ep_len,
            reward_version=args.reward_version)
        print(f"[ENV] KinematicUAVEnv created, reward={args.reward_version}")

    # 7. Training loop
    print(f"\n{'='*60}\nTRAINING START: mode={args.mode}, max_updates={args.max_updates}\n{'='*60}\n")

    buffer = PPORolloutBuffer()
    # Batch accumulation: persist across update_step until target_transitions reached.
    # P0-1 fix: increases effective N from ~50 to target (2048+) → reliable KL, stable advantage.
    ep_return_list = []     # accumulates across rounds
    ep_success_list = []    # accumulates across rounds
    collection_steps = 0

    for update_step in range(args.max_updates):
        if args.dry_run:
            print(f"[{update_step+1:4d}/{args.max_updates}] DRY RUN")
            continue

        # Collect episodes (data accumulation only — PPO fires separately)
        all_rewards, all_logps = [], []
        total_steps = 0

        for ep in range(args.n_eps_per_batch):
            obs_dict, _ = env.reset()
            ep_rewards, ep_logps = [], []
            ep_return = 0.0
            ep_reached = False

            for step_i in range(args.max_ep_len):
                with torch.no_grad():
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

                # FIX P0-1: DPPO sample_ddpm_with_logprob returns normalized action.
                # Unnormalize before env.step() to match physical action space.
                # Without this, dpos_z suffers 27× vertical distortion (std=0.037).
                action = policy.normalizer['action'].unnormalize(action)
                action_np = action.squeeze(0).cpu().numpy()
                next_obs_dict, reward, terminated, truncated, _ = env.step(action_np)
                done = terminated or truncated

                with torch.no_grad():
                    next_global_cond = encode_obs(next_obs_dict, policy, device)

                buffer.add(
                    obs=global_cond.squeeze(0).cpu(),
                    obs_uncond=global_cond_uncond.squeeze(0).cpu(),
                    a_chain=[ac.squeeze(0).cpu() for ac in chain],
                    logp_old=logp.squeeze(0).detach().cpu(),
                    reward=float(reward), done=bool(done),
                    V_old=V_old.squeeze().detach().cpu(),
                    next_obs=next_global_cond.squeeze(0).cpu(),
                )

                ep_rewards.append(float(reward))
                ep_logps.append(logp.mean().item())
                ep_return += float(reward)
                total_steps += 1
                if terminated and not getattr(env, '_collision_detected', False):
                    ep_reached = True
                obs_dict = next_obs_dict
                if done:
                    break

            all_rewards.append(np.mean(ep_rewards) if ep_rewards else 0.0)
            all_logps.append(np.mean(ep_logps) if ep_logps else 0.0)
            ep_return_list.append(ep_return)
            ep_success_list.append(ep_reached)

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
            )
            # FIX P2-1: Compute SR BEFORE clearing ep_success_list
            ep_sr = np.mean(ep_success_list) if ep_success_list else 0.0
            buffer.clear()
            ep_return_list.clear()
            ep_success_list.clear()

            mean_r = np.mean(all_rewards)
            mean_lp = np.mean(all_logps)
            print(f"[{update_step+1:4d}/{args.max_updates}] "
                  f"a_loss={metrics['actor_loss']:.4f} "
                  f"v_loss={metrics['value_loss']:.4f} "
                  f"kl={metrics['approx_kl']:.6f} "
                  f"r={mean_r:.2f} logp={mean_lp:.2f} ep_sr={ep_sr:.2f} "
                  f"N={len(buffer.reward_list)} steps={collection_steps}")
            collection_steps = 0

            if (update_step + 1) % args.eval_freq == 0:
                run_eval(policy, env, update_step + 1, device)
        else:
            # Collection progress only (no PPO yet)
            if (update_step + 1) % 10 == 0:
                print(f"[{update_step+1:4d}/{args.max_updates}] collecting... "
                      f"buffer={len(buffer)}/{args.target_transitions} "
                      f"eps={len(ep_return_list)} steps={collection_steps}")

    # 8. Save final (save state_dicts, including LoRA which is NOT in model.state_dict())
    if not args.dry_run:
        out_dir = os.path.join(_PROJECT_ROOT, args.output_dir)
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, "ddpm_ppo_final.pt")
        save_dict = {
            "step": args.max_updates,
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
        # ★ Fix: LoRA modules are standalone (not in model state_dict) — save separately
        if lora_info and lora_info.get("lora_modules") is not None:
            save_dict["lora_state_dict"] = lora_info["lora_modules"].state_dict()
            save_dict["lora_rank"] = args.lora_rank
            save_dict["lora_alpha"] = args.lora_alpha
            print(f"[SAVE] LoRA state_dict: {len(save_dict['lora_state_dict'])} keys, "
                  f"{sum(v.numel() for v in lora_info['lora_modules'].parameters()):,} params")
        torch.save(save_dict, save_path)
        print(f"[SAVE] {save_path}")

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
    p.add_argument("--K_epochs", type=int, default=None)
    p.add_argument("--mini_batch_size", type=int, default=None)
    p.add_argument("--eval_freq", type=int, default=None)
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
    # DPPO per-step (Ren et al., ICLR 2025)
    p.add_argument("--gamma_denoise", type=float, default=0.8)
    p.add_argument("--eps_base", type=float, default=1e-3)
    p.add_argument("--eps_final", type=float, default=0.2)
    p.add_argument("--eps_beta", type=float, default=3.0)
    p.add_argument("--w_bc", type=float, default=0.0,
                   help="BC regularization weight (0.0=disabled, 0.01=DPPO recommended)")
    p.add_argument("--target_transitions", type=int, default=2048,
                   help="Batch accumulation target (P0-1 fix: N≥2048 for reliable PPO stats)")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    # [ASSUMPTION-9] formal-mode rejection protection: legacy
    # E-series entry points may not run formal training (only the
    # A-Official-Core dppo_uav path may).
    guard_formal_mode("scripts/train_ddpm_ppo.py", args.mode)


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
