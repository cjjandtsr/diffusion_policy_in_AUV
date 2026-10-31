#!/usr/bin/env python3
"""
train_fm_sde_ppo.py — FM-SDE + PPO RL post-training entry point.

Design: RouteA_ReinFlow_FM_SDE_PPO_Design_v1_0.md §3, §4

Loads BC checkpoint, builds SDE sampler + PPO trainer + env,
runs online RL fine-tuning.

Usage:
    python scripts/train_fm_sde_ppo.py --mode smoke       # 200 upd, local RTX 4060
    python scripts/train_fm_sde_ppo.py --mode formal      # 1000 upd, server
    python scripts/train_fm_sde_ppo.py --dry_run          # imports only, no env
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from diffusion_policy_3d.policy.phase1a_policy_fm import Phase1aPolicyFM
from diffusion_policy_3d.policy.fm_sde_sampler import NoiseNetwork
from diffusion_policy_3d.rl.ppo_trainer import PPORolloutBuffer, PPOTrainer
from diffusion_policy_3d.common.pytorch_util import dict_apply

DEFAULT_BC_CKPT = "data/outputs/2026.06.17/15.22.50_v9_2_nobs8_bc/checkpoints/latest.ckpt"
DEFAULT_ADMISSION_MANIFEST = "../../program/reports/routeA_rl_env_v1/calibration/admission_manifest_kinematic_v1.json"
DEFAULT_REWARD_NORM = "program/shiyan/results_routeA_RL_b0b1_rerun_v13_20260710/reward_norm_stats.json"
DEFAULT_OUTPUT_DIR = "outputs/fm_sde_ppo"


# ═══════════════════════════════════════════════════════════════════
# Value Head
# ═══════════════════════════════════════════════════════════════════

class ValueHead(nn.Module):
    """V(s) MLP: 192 -> 128 -> 64 -> 1. ~0.05M params. Design doc §2.2.3."""

    def __init__(self, input_dim: int = 192, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════
# BC Checkpoint Loading
# ═══════════════════════════════════════════════════════════════════

def load_bc_policy(ckpt_path: str, device: str = "cuda:0") -> Phase1aPolicyFM:
    """Load BC checkpoint and return policy in eval mode.

    The BC checkpoint structure (from idp3 training):
      ckpt['cfg']['policy']  → policy constructor args
      ckpt['state_dicts']['model'] → model weights
      ckpt['state_dicts']['ema_model'] → EMA weights (fallback)
      Top-level: num_fm_steps, sigma_min, fm_t_scale, etc.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"BC checkpoint not found: {ckpt_path}")

    print(f"[LOAD] BC checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Policy config is nested under cfg.policy
    cfg = ckpt.get("cfg", {})
    policy_cfg = cfg.get("policy", cfg)  # fallback to cfg itself
    if not policy_cfg:
        raise ValueError("Checkpoint missing 'cfg.policy' key")

    # FM params: from ckpt top-level (prefer) or policy_cfg (fallback)
    fm_kwargs = dict(
        num_fm_steps=ckpt.get("num_fm_steps", policy_cfg.get("num_fm_steps", 4)),
        sigma_min=ckpt.get("sigma_min", policy_cfg.get("sigma_min", 0.005)),
        fm_t_scale=ckpt.get("fm_t_scale", policy_cfg.get("fm_t_scale", 50)),
        solver=ckpt.get("fm_solver", policy_cfg.get("solver", "euler")),
        fm_anchor_t_start=ckpt.get("fm_anchor_t_start", policy_cfg.get("fm_anchor_t_start", 0.95)),
    )

    policy = Phase1aPolicyFM(
        shape_meta=policy_cfg["shape_meta"],
        noise_scheduler=policy_cfg.get("noise_scheduler"),
        horizon=policy_cfg.get("horizon", 16),
        n_action_steps=policy_cfg.get("n_action_steps", 8),
        n_obs_steps=ckpt.get("n_obs_steps", policy_cfg.get("n_obs_steps", 3)),
        num_inference_steps=policy_cfg.get("num_inference_steps"),
        obs_as_global_cond=policy_cfg.get("obs_as_global_cond", True),
        diffusion_step_embed_dim=policy_cfg.get("diffusion_step_embed_dim", 256),
        down_dims=policy_cfg.get("down_dims", (256, 512, 1024)),
        kernel_size=policy_cfg.get("kernel_size", 5),
        n_groups=policy_cfg.get("n_groups", 8),
        condition_type=policy_cfg.get("condition_type") or "film",
        use_down_condition=policy_cfg.get("use_down_condition", True),
        use_mid_condition=policy_cfg.get("use_mid_condition", True),
        use_up_condition=policy_cfg.get("use_up_condition", True),
        use_pc_color=policy_cfg.get("use_pc_color", False),
        pointnet_type=policy_cfg.get("pointnet_type", "pointnet"),
        pointcloud_encoder_cfg=policy_cfg.get("pointcloud_encoder_cfg"),
        point_downsample=policy_cfg.get("point_downsample", False),
        anchor_bank_path=policy_cfg.get("anchor_bank_path"),
        anchor_loss_weight=policy_cfg.get("anchor_loss_weight", 0.5),
        cfg_dropout_prob=policy_cfg.get("cfg_dropout_prob", 0.15),
        cfg_lambda=policy_cfg.get("cfg_lambda", 1.5),
        use_projection=policy_cfg.get("use_projection", True),
        z_min=policy_cfg.get("z_min", 1.0),
        z_max=policy_cfg.get("z_max", 5.0),
        yaw_max_per_step=policy_cfg.get("yaw_max_per_step", 0.524),
        yaw_max_accumulated=policy_cfg.get("yaw_max_accumulated", 3.14159),
        use_obstacle_projection=policy_cfg.get("use_obstacle_projection", False),
        use_collision_loss=policy_cfg.get("use_collision_loss", False),
        training_phase=policy_cfg.get("training_phase", "joint"),
        **fm_kwargs,
    )

    # State dict: from state_dicts.model (prefer) or state_dicts.ema_model (fallback)
    state_dicts = ckpt.get("state_dicts", {})
    state_dict = state_dicts.get("model", state_dicts.get("ema_model", {}))
    if not state_dict:
        raise ValueError("Checkpoint missing state_dicts.model key")

    missing, unexpected = policy.load_state_dict(state_dict, strict=False)
    if missing:
        # Filter out non-trainable / expected missing keys
        expected_missing = [k for k in missing
                           if not any(p in k for p in ('_dummy_variable', 'k_stab', 'lambda_jerk'))]
        if expected_missing:
            print(f"[WARN] Missing keys: {len(expected_missing)} (first 3: {expected_missing[:3]})")
    if unexpected:
        print(f"[WARN] Unexpected keys: {len(unexpected)}")

    policy.to(device)
    policy.eval()
    print("[LOAD] BC policy loaded successfully")
    return policy


# ═══════════════════════════════════════════════════════════════════
# Observation Encoding
# ═══════════════════════════════════════════════════════════════════

def encode_obs(obs_dict: Dict, policy: Phase1aPolicyFM, device: torch.device
               ) -> torch.Tensor:
    """Encode raw observation dict -> global_cond (1, 1536) tensor.

    Returns the full concatenated encoding (obs_dim * n_obs_steps) for the UNet.
    For NoiseNetwork/ValueHead (192d), use: global_cond.reshape(n_obs_steps, 192).mean(0)
    """
    nobs = policy.normalizer.normalize(obs_dict)
    nobs = dict_apply(nobs, lambda x: torch.from_numpy(x).to(device).unsqueeze(0)
                      if isinstance(x, np.ndarray)
                      else x.to(device).unsqueeze(0))

    if not policy.use_pc_color:
        nobs['point_cloud'] = nobs['point_cloud'][..., :3]

    this_nobs = dict_apply(
        nobs,
        lambda x: x[:, :policy.n_obs_steps, ...].reshape(-1, *x.shape[2:])
    )
    global_cond = policy.obs_encoder(this_nobs).reshape(1, -1)  # (1, 1536)
    return global_cond


def global_to_obs_enc(global_cond: torch.Tensor, n_obs_steps: int = 8,
                      obs_dim: int = 192) -> torch.Tensor:
    """Extract 192d obs_enc from 1536d global_cond by averaging over frames."""
    if global_cond.dim() == 1:
        global_cond = global_cond.unsqueeze(0)
    return global_cond.reshape(-1, n_obs_steps, obs_dim).mean(dim=1)  # (B, 192)


def encode_obs_uncond(obs_dict: Dict, policy: Phase1aPolicyFM, device: torch.device
                      ) -> torch.Tensor:
    """Encode raw obs -> unconditional global_cond (1, 1536) for CFG replay.

    Zeros out agent_pos[:,:,3:7] like predict_action() does for CFG unconditional.
    """
    nobs = policy.normalizer.normalize(obs_dict)
    nobs = dict_apply(nobs, lambda x: torch.from_numpy(x).to(device).unsqueeze(0)
                      if isinstance(x, np.ndarray)
                      else x.to(device).unsqueeze(0))

    if not policy.use_pc_color:
        nobs['point_cloud'] = nobs['point_cloud'][..., :3]

    # Zero out conditioning features for unconditional encoding
    nobs['agent_pos'][:, :, 3:7] = 0.0

    this_nobs = dict_apply(
        nobs,
        lambda x: x[:, :policy.n_obs_steps, ...].reshape(-1, *x.shape[2:])
    )
    # Note: must use item assignment after the lambda — handle separately
    # Already done above with nobs['agent_pos'][:, :, 3:7] = 0.0
    return policy.obs_encoder(this_nobs).reshape(1, -1)  # (1, 1536)


# ═══════════════════════════════════════════════════════════════════
# Env Helpers
# ═══════════════════════════════════════════════════════════════════

def _env_obs_to_dict(obs) -> Dict:
    """Convert env obs to dict format expected by policy."""
    if isinstance(obs, dict):
        return {k: np.asarray(v) if not isinstance(v, np.ndarray) else v
                for k, v in obs.items()}
    if isinstance(obs, (tuple, list)) and len(obs) == 2:
        return {"agent_pos": np.asarray(obs[0]), "point_cloud": np.asarray(obs[1])}
    raise TypeError(f"Unexpected obs type: {type(obs)}")


def run_eval(policy, env, step: int, device: torch.device):
    """Run ODE eval (no SDE noise) at checkpoint step."""
    policy.eval()
    with torch.no_grad():
        try:
            obs_dict, _ = env.reset()  # Gymnasium: (obs, info)
            done = False
            total_r = 0.0
            steps = 0
            while not done and steps < 200:
                # Convert numpy to tensor + add batch dim (policy expects tensor with batch)
                obs_b = {}
                for k, v in obs_dict.items():
                    if isinstance(v, np.ndarray):
                        v = torch.from_numpy(v).to(device)
                    else:
                        v = v.to(device)
                    obs_b[k] = v.unsqueeze(0)
                action_out = policy.predict_action(obs_b)
                action = action_out['action_pred'] if isinstance(action_out, dict) else action_out
                action_np = action.squeeze(0).cpu().numpy()
                obs_dict, reward, terminated, truncated, _ = env.step(action_np)
                done = terminated or truncated
                total_r += reward
                steps += 1
            collision = getattr(env, '_collision_detected', False)
            reached = terminated and not collision
            print(f"[EVAL step={step}] total_r={total_r:.2f}, steps={steps}, collision={collision}, reached={reached}")
        except Exception as e:
            import traceback
            print(f"[EVAL step={step}] ERROR: {e}")
            traceback.print_exc()
    policy.train()


def save_checkpoint(policy, value_head, noise_net, actor_opt, noise_opt, value_opt,
                    step, args, final=False):
    """Save training checkpoint including noise_net for resume capability.

    noise_net is saved for training resume; discarded at deployment (design §2.2.4).
    """
    out_dir = os.path.join(_PROJECT_ROOT, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    tag = "final" if final else f"step_{step}"
    path = os.path.join(out_dir, f"ppo_{tag}.pt")
    torch.save({
        "step": step,
        "velocity_head": policy.model.state_dict(),
        "value_head": value_head.state_dict(),
        "noise_net": noise_net.state_dict(),
        "actor_optimizer": actor_opt.state_dict(),
        "noise_optimizer": noise_opt.state_dict(),
        "value_optimizer": value_opt.state_dict(),
        "config": {
            "horizon": policy.horizon,
            "action_dim": policy.action_dim,
            "n_action_steps": policy.n_action_steps,
            "n_obs_steps": policy.n_obs_steps,
            "num_fm_steps": policy.num_fm_steps,
            "sigma_min": policy.sigma_min,
            "fm_t_scale": policy.fm_t_scale,
            "cfg_lambda": policy.cfg_lambda,
        },
    }, path)
    print(f"[SAVE] {path}")


# ═══════════════════════════════════════════════════════════════════
# Main Training Loop
# ═══════════════════════════════════════════════════════════════════

def run_training(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    # 1. Load BC policy
    ckpt_path = os.path.join(_PROJECT_ROOT, args.bc_ckpt)
    policy = load_bc_policy(ckpt_path, device=str(device))
    # Override cfg_lambda if specified via CLI (Phase 5d sweep)
    if args.cfg_lambda is not None:
        policy.cfg_lambda = args.cfg_lambda
        print(f"[CFG] cfg_lambda overridden to {args.cfg_lambda}")
    policy.obs_encoder.requires_grad_(False)
    print("[FREEZE] obs_encoder frozen")

    # LoRA: insert adapters, freeze UNet, only train LoRA params
    lora_info = None
    if args.lora:
        from diffusion_policy_3d.policy.lora_adapters import insert_lora
        lora_info = insert_lora(policy.model, r=args.lora_rank, alpha=args.lora_alpha)
        # Freeze all UNet params — only LoRA params are trainable (added to optimizer below)
        policy.model.requires_grad_(False)
        for p in lora_info["lora_params"]:
            p.requires_grad_(True)
        print(f"[FREEZE] UNet frozen, only LoRA params trainable")

    # 2. Build new modules
    noise_net = NoiseNetwork(
        obs_dim=192, action_dim=11, hidden_dim=128,
        sigma_min=args.sigma_min, sigma_max=args.sigma_max,
    ).to(device).train()
    value_head = ValueHead(input_dim=192, hidden_dim=128).to(device).train()
    print(f"[INIT] NoiseNetwork: {sum(p.numel() for p in noise_net.parameters()):,} params")
    print(f"[INIT] ValueHead: {sum(p.numel() for p in value_head.parameters()):,} params")

    # 3. Optimizers (separate LR, design §3.3)
    if lora_info is not None:
        actor_params = lora_info["lora_params"]
    else:
        actor_params = [p for p in policy.model.parameters() if p.requires_grad]
    actor_optimizer = torch.optim.Adam(actor_params, lr=args.lr_actor)
    noise_optimizer = torch.optim.Adam(noise_net.parameters(), lr=args.lr_noise)
    value_optimizer = torch.optim.Adam(value_head.parameters(), lr=args.lr_value)
    print(f"[OPTIM] Actor: {sum(p.numel() for p in actor_params):,} params, LR={args.lr_actor}")

    # 4. PPO trainer (with optional W2 BC regularization)
    ref_velocity_head = None
    if args.beta_w2 > 0:
        import copy
        ref_velocity_head = copy.deepcopy(policy.model)
        ref_velocity_head.eval()
        for p in ref_velocity_head.parameters():
            p.requires_grad_(False)
        print(f"[W2] Frozen BC reference model created, beta_w2={args.beta_w2}")

    ppo_trainer = PPOTrainer(
        clip_epsilon=args.clip_epsilon, target_kl=args.target_kl,
        entropy_alpha=args.entropy_alpha,
        max_grad_norm=args.max_grad_norm,
        beta_w2=args.beta_w2,
        ref_velocity_head=ref_velocity_head,
    )

    # 5. Load reward norm stats
    reward_norm_path = os.path.join(_PROJECT_ROOT, args.reward_norm)
    if os.path.exists(reward_norm_path):
        with open(reward_norm_path, "r") as f:
            rn_stats = json.load(f)
        ppo_trainer.reward_mean = rn_stats.get("total_mean", 0.0)
        ppo_trainer.reward_std = max(rn_stats.get("total_std", 1.0), 1.0)
        print(f"[REWARD_NORM] mean={ppo_trainer.reward_mean:.2f}, std={ppo_trainer.reward_std:.2f}")

    # 6. Build env
    if args.dry_run:
        env = None
        print("[DRY_RUN] Skipping env setup")
    else:
        from diffusion_policy_3d.env.kinematic_uav_env_v2 import KinematicUAVEnv
        from diffusion_policy_3d.rl.routea_rl_env_adapter_v1 import (
            load_routea_rl_admission_manifest, build_role_buckets,
        )
        from diffusion_policy_3d.model.common.synth_pointcloud import SynthPointCloud
        import json as _json

        # Load admission manifest → get all admitted train worlds
        manifest_path = os.path.join(_PROJECT_ROOT, args.admission_manifest)
        manifest = load_routea_rl_admission_manifest(manifest_path)
        buckets = build_role_buckets(manifest, "rl_train")

        # Build world pool: [(name, json_path), ...] for all trainable roles
        # Paths: worlds are under ~/AUV/PX4-Autopilot/..., obstacles.json alongside .world files
        auv_root = os.path.expanduser("~/AUV")
        worlds_dir = os.path.join(auv_root, "PX4-Autopilot/Tools/sitl_gazebo/worlds")
        world_pool = []
        synth_pc_pool = {}
        for role, entries in buckets.items():
            for entry in entries:
                name = entry.world_uid
                # world_uid matches the filename stem (e.g. routeA_rl_v1__anticipatory__...)
                json_path = os.path.join(
                    worlds_dir, f"{name}.world.obstacles.json"
                )
                if not os.path.exists(json_path):
                    print(f"[WARN] Missing obstacles JSON: {json_path}")
                    continue
                world_pool.append((name, json_path))
                synth_pc_pool[name] = SynthPointCloud(obstacles_json=json_path)

        # Count per role
        role_counts = {role: len(entries) for role, entries in buckets.items()}
        print(f"[ENV] World pool: {len(world_pool)} worlds across {list(role_counts.keys())}")
        print(f"[ENV] Role counts: {role_counts}")

        # Phase 4: Balanced role sampling (P4a)
        if args.balanced_roles:
            # Group world_pool by role
            role_worlds = {role: [] for role in buckets}
            for role, entries in buckets.items():
                for entry in entries:
                    name = entry.world_uid
                    json_path = os.path.join(worlds_dir, f"{name}.world.obstacles.json")
                    if os.path.exists(json_path):
                        role_worlds[role].append((name, json_path))
            # Target: equal probability per role → balanced world count
            target_per_role = max(len(v) for v in role_worlds.values())
            world_pool = []
            for role, worlds in role_worlds.items():
                if len(worlds) == 0:
                    continue
                # Duplicate to target count (with shuffling for variety)
                import random as _random
                repeats = target_per_role // len(worlds) + 1
                expanded = (worlds * repeats)[:target_per_role]
                world_pool.extend(expanded)
            print(f"[ENV] Balanced: {len(world_pool)} worlds, ~{target_per_role} per role")

        # Phase 4b: Weak role boost (2x on antic/support/yaw)
        if args.weak_role_boost:
            # Group by role
            role_worlds_p4b = {role: [] for role in buckets}
            for role, entries in buckets.items():
                for entry in entries:
                    name = entry.world_uid
                    json_path = os.path.join(worlds_dir, f"{name}.world.obstacles.json")
                    if os.path.exists(json_path):
                        role_worlds_p4b[role].append((name, json_path))
            # Boost non-dense roles: 2x sampling weight
            weak_roles = {'anticipatory', 'support_expansion', 'yaw_lateral_recovery'}
            world_pool = []
            for role, worlds in role_worlds_p4b.items():
                if role in weak_roles:
                    world_pool.extend(worlds * 2)  # double weight
                else:
                    world_pool.extend(worlds)
            print(f"[ENV] Weak-role boost: {len(world_pool)} worlds "
                  f"(weak roles 2x: {weak_roles})")

        # Phase 4c: Hard example pool (BC SR < 0.2 + all antic/support/yaw)
        if args.hard_pool:
            world_pool = []
            weak_roles = {'anticipatory', 'support_expansion', 'yaw_lateral_recovery'}
            for role, entries in buckets.items():
                for entry in entries:
                    name = entry.world_uid
                    json_path = os.path.join(worlds_dir, f"{name}.world.obstacles.json")
                    if not os.path.exists(json_path):
                        continue
                    bc_sr = getattr(entry, 'bc_fm_sr', 0.5)
                    if bc_sr < 0.2 or role in weak_roles:
                        world_pool.append((name, json_path))
                        synth_pc_pool[name] = SynthPointCloud(obstacles_json=json_path)
            print(f"[ENV] Hard pool: {len(world_pool)} worlds "
                  f"(BC SR<0.2 or weak role)")

        # Single env with world_pool (resets pick random world)
        env_kwargs = dict(
            world_pool=world_pool,
            synth_pc_pool=synth_pc_pool,
            curriculum_stage='C2',
            max_episode_steps=args.max_ep_len,
            reward_version=args.reward_version,
        )
        # Phase 2: pass reward params if specified
        for rp in ['lambda_c', 'd_safe', 'beta_avoid', 'lambda_stuck', 'lambda_escape']:
            if getattr(args, rp, None) is not None:
                env_kwargs[rp] = getattr(args, rp)
        if args.use_stuck_penalty:
            env_kwargs['use_stuck_penalty'] = True
        if args.use_escape_reward:
            env_kwargs['use_escape_reward'] = True
        env = KinematicUAVEnv(**env_kwargs)
        print(f"[ENV] KinematicUAVEnv created (reward_version={args.reward_version})")

    # 7. Training loop
    print(f"\n{'='*60}")
    print(f"TRAINING START: mode={args.mode}, max_updates={args.max_updates}")
    print(f"{'='*60}\n")

    buffer = PPORolloutBuffer()

    for update_step in range(args.max_updates):
        if args.dry_run:
            print(f"[{update_step+1:4d}/{args.max_updates}] DRY RUN — skipped")
            continue

        # ── Collect N_eps_per_batch episodes ──
        all_rewards = []
        all_logps = []
        total_steps = 0

        for ep in range(args.n_eps_per_batch):
            obs_dict, _ = env.reset()  # random world from pool
            ep_rewards = []
            ep_logps = []

            for _ in range(args.max_ep_len):
                with torch.no_grad():
                    global_cond = encode_obs(obs_dict, policy, device)
                    global_cond_uncond = encode_obs_uncond(obs_dict, policy, device)
                    obs_enc_192 = global_to_obs_enc(global_cond, policy.n_obs_steps, 192)
                    V_old = value_head(obs_enc_192)
                    # Convert to tensor + batch dim
                    _obs_batched = {}
                    for k, v in obs_dict.items():
                        t = torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else v.to(device)
                        _obs_batched[k] = t.unsqueeze(0)
                    # SDE action
                    action, logp, a_chain = policy.sample_sde_with_logprob(
                        _obs_batched, noise_net)

                action_np = action.squeeze(0).cpu().numpy()
                next_obs_dict, reward, done, truncated, _ = env.step(action_np)

                with torch.no_grad():
                    next_global_cond = encode_obs(next_obs_dict, policy, device)

                buffer.add(
                    obs=global_cond.squeeze(0).cpu(),
                    obs_uncond=global_cond_uncond.squeeze(0).cpu(),
                    a_chain=[ac.squeeze(0).cpu() for ac in a_chain],
                    logp_old=logp.squeeze(0).detach().cpu(),
                    reward=float(reward), done=bool(done),
                    V_old=V_old.squeeze(0).detach().cpu(),
                    next_obs=next_global_cond.squeeze(0).cpu(),
                )

                ep_rewards.append(float(reward))
                ep_logps.append(logp.item())
                total_steps += 1
                obs_dict = next_obs_dict
                if done:
                    break

            all_rewards.append(np.mean(ep_rewards) if ep_rewards else 0.0)
            all_logps.append(np.mean(ep_logps) if ep_logps else 0.0)

        # ── PPO Update (on accumulated buffer) ──
        if len(buffer) >= args.mini_batch_size:
            metrics = ppo_trainer.update(
                buffer=buffer, velocity_head=policy.model,
                noise_net=noise_net, value_head=value_head,
                actor_optimizer=actor_optimizer,
                noise_optimizer=noise_optimizer,
                value_optimizer=value_optimizer,
                horizon=policy.horizon, action_dim=policy.action_dim,
                num_fm_steps=policy.num_fm_steps,
                sigma_min=policy.sigma_min,
                fm_t_scale=policy.fm_t_scale,
                cfg_lambda=policy.cfg_lambda,
                K_epochs=args.K_epochs,
                mini_batch_size=args.mini_batch_size,
            )
            buffer.clear()

            mean_r = np.mean(all_rewards)
            mean_lp = np.mean(all_logps)
            print(
                f"[{update_step+1:4d}/{args.max_updates}] "
                f"a_loss={metrics['actor_loss']:.4f} "
                f"v_loss={metrics['value_loss']:.4f} "
                f"ent={metrics['entropy']:.4f} "
                f"kl={metrics['approx_kl']:.4f} "
                f"r={mean_r:.2f} logp={mean_lp:.2f} steps={total_steps}"
            )

            if (update_step + 1) % args.eval_freq == 0:
                run_eval(policy, env, update_step + 1, device)
                save_checkpoint(policy, value_head, noise_net, actor_optimizer,
                                noise_optimizer, value_optimizer,
                                update_step + 1, args)

    # 8. Final save
    if not args.dry_run:
        save_checkpoint(policy, value_head, noise_net, actor_optimizer,
                        noise_optimizer, value_optimizer,
                        args.max_updates, args, final=True)
    print(f"\n{'='*60}\nTRAINING COMPLETE\n{'='*60}")


# ═══════════════════════════════════════════════════════════════════
# Args
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="FM-SDE + PPO RL Training")
    p.add_argument("--mode", type=str, default="smoke",
                   choices=["smoke", "formal", "dry_run"])
    p.add_argument("--bc_ckpt", type=str, default=DEFAULT_BC_CKPT)
    p.add_argument("--admission_manifest", type=str, default=DEFAULT_ADMISSION_MANIFEST)
    p.add_argument("--reward_norm", type=str, default=DEFAULT_REWARD_NORM)
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--max_updates", type=int, default=None)
    p.add_argument("--max_ep_len", type=int, default=200)
    p.add_argument("--n_eps_per_batch", type=int, default=None,
                   help="Episodes to collect per PPO update (smoke=4, formal=16)")
    p.add_argument("--K_epochs", type=int, default=None)
    p.add_argument("--mini_batch_size", type=int, default=None)
    p.add_argument("--eval_freq", type=int, default=None)
    p.add_argument("--lr_actor", type=float, default=1e-4)  # v1.1: 3e-4→1e-4 for kl stability
    p.add_argument("--lr_value", type=float, default=1e-3)
    p.add_argument("--lr_noise", type=float, default=1e-4)
    p.add_argument("--clip_epsilon", type=float, default=0.2)
    p.add_argument("--target_kl", type=float, default=0.015)
    p.add_argument("--entropy_alpha", type=float, default=None)
    p.add_argument("--sigma_min", type=float, default=0.01)
    p.add_argument("--sigma_max", type=float, default=0.1)
    p.add_argument("--cfg_lambda", type=float, default=None,
                   help="CFG strength (default: from policy config, typically 1.5). Phase 5d sweep: {1.0, 2.0, 2.5}")
    p.add_argument("--max_grad_norm", type=float, default=0.5,
                   help="Max gradient norm for clipping (0=disabled)")
    p.add_argument("--reward_version", type=str, default="legacy",
                   choices=["legacy", "v14"],
                   help="Reward version: 'legacy' (PBRS) or 'v14' (adaptive-threshold)")
    p.add_argument("--lora", action="store_true", help="Use LoRA fine-tuning (freeze UNet)")
    p.add_argument("--lora_rank", type=int, default=4)
    p.add_argument("--lora_alpha", type=int, default=4)
    p.add_argument("--beta_w2", type=float, default=0.0,
                   help="W2 BC regularization strength (0=disabled). Phase 1c sweep: {0.01, 0.05, 0.1}")
    # Phase 2: Reward parameters
    p.add_argument("--lambda_c", type=float, default=None,
                   help="Collision penalty weight (default: 0.1)")
    p.add_argument("--d_safe", type=float, default=None,
                   help="Safety distance threshold (default: 0.75)")
    p.add_argument("--beta_avoid", type=float, default=None,
                   help="Softplus sharpness for collision (default: 20.0)")
    p.add_argument("--use_stuck_penalty", action="store_true",
                   help="Enable stuck penalty")
    p.add_argument("--lambda_stuck", type=float, default=None,
                   help="Stuck penalty weight (default: 0.5)")
    p.add_argument("--use_escape_reward", action="store_true",
                   help="Enable escape reward")
    p.add_argument("--lambda_escape", type=float, default=None,
                   help="Escape reward weight (default: 3.0)")
    # Phase 4: Environment diversity
    p.add_argument("--balanced_roles", action="store_true",
                   help="Balance role sampling to equal probability per role")
    p.add_argument("--weak_role_boost", action="store_true",
                   help="2x weight on antic/support/yaw roles (fewer training worlds)")
    p.add_argument("--hard_pool", action="store_true",
                   help="Use only BC SR<0.2 worlds + all antic/support/yaw")
    p.add_argument("--dry_run", action="store_true")

    args = p.parse_args()

    if args.mode == "smoke":
        args.max_updates = args.max_updates or 200
        args.n_eps_per_batch = args.n_eps_per_batch or 4
        args.K_epochs = args.K_epochs or 4
        args.mini_batch_size = args.mini_batch_size or 64
        if args.entropy_alpha is None: args.entropy_alpha = 0.0
        args.eval_freq = args.eval_freq or 50
    elif args.mode == "formal":
        args.max_updates = args.max_updates or 1000
        args.n_eps_per_batch = args.n_eps_per_batch or 16
        args.K_epochs = args.K_epochs or 4
        args.mini_batch_size = args.mini_batch_size or 128
        if args.entropy_alpha is None: args.entropy_alpha = 0.005
        args.eval_freq = args.eval_freq or 200
    else:
        args.max_updates = args.max_updates or 1
        args.K_epochs = args.K_epochs or 1
        args.mini_batch_size = args.mini_batch_size or 64
        if args.entropy_alpha is None: args.entropy_alpha = 0.0
        args.eval_freq = args.eval_freq or 1
        args.dry_run = True

    return args


if __name__ == "__main__":
    args = parse_args()
    print("[CONFIG]")
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")
    run_training(args)
