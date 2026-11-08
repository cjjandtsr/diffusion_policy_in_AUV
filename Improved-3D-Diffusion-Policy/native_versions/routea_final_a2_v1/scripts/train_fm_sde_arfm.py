#!/usr/bin/env python3
"""
train_fm_sde_arfm.py — FM-SDE + ARFM (Reward-Weighted FM) RL post-training.

Design: RouteA_ARFM_Trainer_Design.md

Replaces PPO with ARFM: reward-weighted FM loss + W2 regularization.
No ValueHead, GAE, advantage, or logp recomputation needed.

Usage:
    python scripts/train_fm_sde_arfm.py --mode smoke       # 200 upd, local
    python scripts/train_fm_sde_arfm.py --mode formal      # 1000 upd, server
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from diffusion_policy_3d.policy.phase1a_policy_fm import Phase1aPolicyFM
from diffusion_policy_3d.policy.fm_sde_sampler import NoiseNetwork
from diffusion_policy_3d.rl.ppo_trainer import PPORolloutBuffer  # reuse buffer
from diffusion_policy_3d.rl.arfm_trainer import ARFMTrainer
from diffusion_policy_3d.common.pytorch_util import dict_apply

DEFAULT_BC_CKPT = "data/outputs/2026.06.17/15.22.50_v9_2_nobs8_bc/checkpoints/latest.ckpt"
DEFAULT_ADMISSION_MANIFEST = "../../program/reports/routeA_rl_env_v1/calibration/admission_manifest_unified_292worlds.json"
DEFAULT_OUTPUT_DIR = "outputs/fm_sde_arfm"



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


def save_checkpoint(policy, noise_net, optimizer, step, args, final=False):
    """Save ARFM training checkpoint."""
    out_dir = os.path.join(_PROJECT_ROOT, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    tag = "final" if final else f"step_{step}"
    path = os.path.join(out_dir, f"arfm_{tag}.pt")
    torch.save({
        "step": step,
        "velocity_head": policy.model.state_dict(),
        "noise_net": noise_net.state_dict(),
        "optimizer": optimizer.state_dict(),
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
        "arfm_alpha": args.arfm_alpha,
        "arfm_beta_w2": args.arfm_beta_w2,
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
    policy.obs_encoder.requires_grad_(False)
    print("[FREEZE] obs_encoder frozen")

    # LoRA: insert adapters, freeze UNet, only train LoRA params
    lora_info = None
    if args.lora:
        from diffusion_policy_3d.policy.lora_adapters import insert_lora
        lora_info = insert_lora(policy.model, r=args.lora_rank, alpha=args.lora_alpha)
        policy.model.requires_grad_(False)
        for p in lora_info["lora_params"]:
            p.requires_grad_(True)
        print(f"[FREEZE] UNet frozen, only LoRA params trainable")

    # 2. Build NoiseNetwork (for SDE exploration; frozen during ARFM update)
    noise_net = NoiseNetwork(
        obs_dim=192, action_dim=11, hidden_dim=128,
        sigma_min=args.sigma_min, sigma_max=args.sigma_max,
    ).to(device).train()
    print(f"[INIT] NoiseNetwork: {sum(p.numel() for p in noise_net.parameters()):,} params")

    # 3. Optimizer (single optimizer for all trainable params)
    if lora_info is not None:
        trainable_params = list(lora_info["lora_params"]) + list(noise_net.parameters())
    else:
        trainable_params = [p for p in policy.model.parameters() if p.requires_grad] + list(noise_net.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr_actor)
    print(f"[OPTIM] Trainable params: {sum(p.numel() for p in trainable_params):,}, LR={args.lr_actor}")

    # 4. ARFM trainer (with frozen BC reference for W2 regularization)
    arfm_trainer = ARFMTrainer(
        alpha=args.arfm_alpha,
        beta_w2=args.arfm_beta_w2,
        max_grad_norm=args.max_grad_norm,
        ref_velocity_head=policy.model,  # frozen copy of BC weights
    )
    print(f"[ARFM] alpha={args.arfm_alpha}, beta_w2={args.arfm_beta_w2}")

    # 5. Build env
    if args.dry_run:
        env = None
        print("[DRY_RUN] Skipping env setup")
    else:
        from diffusion_policy_3d.env.kinematic_uav_env_v2 import KinematicUAVEnv
        from diffusion_policy_3d.rl.routea_rl_env_adapter_v1 import (
            load_routea_rl_admission_manifest, build_role_buckets,
        )
        from diffusion_policy_3d.model.common.synth_pointcloud import SynthPointCloud

        manifest_path = os.path.join(_PROJECT_ROOT, args.admission_manifest)
        manifest = load_routea_rl_admission_manifest(manifest_path)
        buckets = build_role_buckets(manifest, "rl_train")

        auv_root = os.path.expanduser("~/AUV")
        worlds_dir = os.path.join(auv_root, "PX4-Autopilot/Tools/sitl_gazebo/worlds/train")
        world_pool = []
        synth_pc_pool = {}
        for role, entries in buckets.items():
            for entry in entries:
                name = entry.world_uid
                json_path = os.path.join(worlds_dir, f"{name}.world.obstacles.json")
                if not os.path.exists(json_path):
                    print(f"[WARN] Missing obstacles JSON: {json_path}")
                    continue
                world_pool.append((name, json_path))
                synth_pc_pool[name] = SynthPointCloud(obstacles_json=json_path)

        role_counts = {role: len(entries) for role, entries in buckets.items()}
        print(f"[ENV] World pool: {len(world_pool)} worlds across {list(role_counts.keys())}")
        print(f"[ENV] Role counts: {role_counts}")

        env = KinematicUAVEnv(
            world_pool=world_pool,
            synth_pc_pool=synth_pc_pool,
            curriculum_stage='C2',
            max_episode_steps=args.max_ep_len,
            reward_version=args.reward_version,
        )
        print(f"[ENV] KinematicUAVEnv created (reward_version={args.reward_version})")

    # 6. Training loop
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
        total_steps = 0

        for ep in range(args.n_eps_per_batch):
            obs_dict, _ = env.reset()
            ep_rewards = []

            for _ in range(args.max_ep_len):
                with torch.no_grad():
                    global_cond = encode_obs(obs_dict, policy, device)
                    _obs_batched = {}
                    for k, v in obs_dict.items():
                        t = torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else v.to(device)
                        _obs_batched[k] = t.unsqueeze(0)
                    action, _logp, a_chain = policy.sample_sde_with_logprob(
                        _obs_batched, noise_net)

                action_np = action.squeeze(0).cpu().numpy()
                next_obs_dict, reward, done, truncated, _ = env.step(action_np)

                # ARFM: only need obs + a_chain + reward (no logp, no V)
                buffer.add(
                    obs=global_cond.squeeze(0).cpu(),
                    obs_uncond=global_cond.squeeze(0).cpu(),
                    a_chain=[ac.squeeze(0).cpu() for ac in a_chain],
                    logp_old=torch.tensor(0.0),   # unused by ARFM
                    reward=float(reward), done=bool(done),
                    V_old=torch.tensor(0.0),       # unused by ARFM
                    next_obs=global_cond.squeeze(0).cpu(),
                )

                ep_rewards.append(float(reward))
                total_steps += 1
                obs_dict = next_obs_dict
                if done:
                    break

            all_rewards.append(np.mean(ep_rewards) if ep_rewards else 0.0)

        # ── ARFM Update ──
        if len(buffer) >= args.mini_batch_size:
            metrics = arfm_trainer.update(
                buffer=buffer,
                velocity_head=policy.model,
                noise_net=noise_net,
                optimizer=optimizer,
                horizon=policy.horizon,
                action_dim=policy.action_dim,
                num_fm_steps=policy.num_fm_steps,
                sigma_min=policy.sigma_min,
                fm_t_scale=policy.fm_t_scale,
                cfg_lambda=policy.cfg_lambda,
            )
            buffer.clear()

            mean_r = np.mean(all_rewards)
            print(
                f"[{update_step+1:4d}/{args.max_updates}] "
                f"arfm={metrics['arfm_loss']:.4f} "
                f"fm={metrics['fm_loss']:.4f} "
                f"w2={metrics['w2_reg']:.4f} "
                f"r={mean_r:.2f} steps={total_steps}"
            )

            if (update_step + 1) % args.eval_freq == 0:
                run_eval(policy, env, update_step + 1, device)
                save_checkpoint(policy, noise_net, optimizer,
                                update_step + 1, args)

    # 7. Final save
    if not args.dry_run:
        save_checkpoint(policy, noise_net, optimizer,
                        args.max_updates, args, final=True)
    print(f"\n{'='*60}\nTRAINING COMPLETE\n{'='*60}")


# ═══════════════════════════════════════════════════════════════════
# Args
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="FM-SDE + ARFM RL Training")
    p.add_argument("--mode", type=str, default="smoke",
                   choices=["smoke", "formal", "dry_run"])
    p.add_argument("--bc_ckpt", type=str, default=DEFAULT_BC_CKPT)
    p.add_argument("--admission_manifest", type=str, default=DEFAULT_ADMISSION_MANIFEST)
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--max_updates", type=int, default=None)
    p.add_argument("--max_ep_len", type=int, default=200)
    p.add_argument("--n_eps_per_batch", type=int, default=None)
    p.add_argument("--mini_batch_size", type=int, default=None)
    p.add_argument("--eval_freq", type=int, default=None)
    p.add_argument("--lr_actor", type=float, default=1e-4)
    p.add_argument("--sigma_min", type=float, default=0.01)
    p.add_argument("--sigma_max", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--reward_version", type=str, default="legacy",
                   choices=["legacy", "v14"])
    # ARFM-specific
    p.add_argument("--arfm_alpha", type=float, default=0.5,
                   help="Reward weight scale for ARFM")
    p.add_argument("--arfm_beta_w2", type=float, default=0.1,
                   help="W2 regularization strength")
    # LoRA
    p.add_argument("--lora", action="store_true", help="Use LoRA fine-tuning")
    p.add_argument("--lora_rank", type=int, default=4)
    p.add_argument("--lora_alpha", type=int, default=4)
    p.add_argument("--dry_run", action="store_true")

    args = p.parse_args()

    if args.mode == "smoke":
        args.max_updates = args.max_updates or 200
        args.n_eps_per_batch = args.n_eps_per_batch or 16
        args.mini_batch_size = args.mini_batch_size or 64
        args.eval_freq = args.eval_freq or 50
    elif args.mode == "formal":
        args.max_updates = args.max_updates or 1000
        args.n_eps_per_batch = args.n_eps_per_batch or 32
        args.mini_batch_size = args.mini_batch_size or 128
        args.eval_freq = args.eval_freq or 200
    else:
        args.max_updates = args.max_updates or 1
        args.mini_batch_size = args.mini_batch_size or 64
        args.eval_freq = args.eval_freq or 1
        args.dry_run = True

    return args


if __name__ == "__main__":
    args = parse_args()
    print("[CONFIG]")
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")
    run_training(args)
