#!/usr/bin/env python3
"""
p0_5_stochastic_path_toy_test.py — P0-5: Stochastic path log-prob toy test (§8.5)

验证 4-step stochastic path 数值稳定性。对固定单 obs，从 frozen FM policy 采样
N=128 stochastic paths，检查 log-prob 的数值范围、MSE vs deterministic teacher、
sigma grad 是否可反传、reproducibility。

用法:
  python tools/p0_5_stochastic_path_toy_test.py [--device cuda:0]
"""

import json, os, sys, math, time, argparse
from collections import OrderedDict
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

import torch
import dill
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from diffusion_policy_3d.env.kinematic_uav_env import KinematicUAVEnv
from diffusion_policy_3d.env.kinematic_uav_env_v2 import WORLDS_DIR, WORLD_ENV_MAP
from diffusion_policy_3d.model.common.synth_pointcloud import SynthPointCloud
from diffusion_policy_3d.policy.phase1a_policy_fm import Phase1aPolicyFM
from diffusion_policy_3d.policy.fm_base_policy import FMBasePolicy
from diffusion_policy_3d.policy.reinflow_noise_module import (
    ReinFlowNoiseModule, compute_stochastic_path_log_prob,
)

# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════

N_PATHS = 128
# Action normalizer stats (from v9_2 BC training dataset)
ACTION_STD = np.array([
    0.0660, 0.0360, 0.0300,    # dpos_world (m)
    0.1300, 0.0700, 0.0600,    # dvel_world (m/s)
    0.2600, 0.1400, 0.1200,    # acc_world (m/s^2)
    0.1600,                     # dyaw (rad)
    0.2000,                     # yaw_rate (rad/s)
], dtype=np.float32)
BC_ACTION_VARIANCE = float(np.mean(ACTION_STD ** 2))


def load_policy_and_obs(ckpt_path, device):
    """Load policy and collect one observation from OOD world 11."""
    ckpt = torch.load(ckpt_path, map_location=device, pickle_module=dill)
    cfg = ckpt["cfg"]
    sd = ckpt["state_dicts"]["ema_model"]
    pc = cfg["policy"]

    fm = Phase1aPolicyFM(
        shape_meta=cfg["shape_meta"],
        noise_scheduler=DDIMScheduler(
            num_train_timesteps=50, beta_start=0.0001, beta_end=0.02,
            beta_schedule="squaredcos_cap_v2", clip_sample=True,
            set_alpha_to_one=True, steps_offset=0, prediction_type="sample",
        ),
        horizon=cfg["horizon"],
        n_action_steps=cfg["n_action_steps"],
        n_obs_steps=cfg["n_obs_steps"],
        obs_as_global_cond=pc.get("obs_as_global_cond", True),
        diffusion_step_embed_dim=pc.get("diffusion_step_embed_dim", 128),
        down_dims=pc.get("down_dims", (256, 512, 1024)),
        kernel_size=pc.get("kernel_size", 5),
        n_groups=pc.get("n_groups", 8),
        anchor_bank_path=None,
        cfg_lambda=1.5, num_fm_steps=4, sigma_min=0.005, fm_t_scale=50,
        solver="euler", fm_anchor_t_start=0.95, use_pc_color=False,
        pointnet_type="multi_stage_pointnet",
        pointcloud_encoder_cfg=pc.get("pointcloud_encoder_cfg", None),
        point_downsample=pc.get("point_downsample", True),
    )
    fm.load_state_dict(sd, strict=False)
    fm.to(device).eval()
    for p in fm.parameters():
        p.requires_grad_(False)

    # Collect one obs from OOD world 11
    wname = WORLD_ENV_MAP[11]
    obs_json = str(WORLDS_DIR / f"{wname}.obstacles.json")
    synth_pc = SynthPointCloud(obs_json, device="cpu")
    env = KinematicUAVEnv(
        world_id=11, synth_pc=synth_pc,
        rl_action_steps=8, h_min=0.3, z_min=1.0, z_max=5.0,
        disturb_enable=False, mid_state_init_ratio=0.0,
        seed=42, reward_version="v13",
    )
    obs, _ = env.reset()
    env.close()

    # Normalize obs (returns tensors directly)
    nobs = fm.normalizer.normalize(obs)
    nobs = {k: v.unsqueeze(0).to(device) for k, v in nobs.items()}
    if not fm.use_pc_color:
        nobs['point_cloud'] = nobs['point_cloud'][..., :3]

    # Encode obs → global_cond
    this_nobs = {k: v[:, :fm.n_obs_steps].reshape(-1, *v.shape[2:])
                 for k, v in nobs.items()}
    global_cond = fm.obs_encoder(this_nobs).reshape(1, -1)

    # Uncond for CFG
    nobs_u = {k: v.clone() for k, v in nobs.items()}
    nobs_u['agent_pos'][:, :, 3:7] = 0.0
    this_u = {k: v[:, :fm.n_obs_steps].reshape(-1, *v.shape[2:])
              for k, v in nobs_u.items()}
    global_cond_uncond = fm.obs_encoder(this_u).reshape(1, -1)

    return fm, global_cond, global_cond_uncond


def run_deterministic(fm, global_cond, global_cond_uncond, device):
    """One deterministic FM sample."""
    B = global_cond.shape[0]
    T = fm.horizon
    Da = fm.action_dim
    cond_data = torch.zeros(B, T, Da, device=device)
    cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
    with torch.no_grad():
        sample = fm.conditional_sample_fm(
            cond_data, cond_mask,
            local_cond=None, global_cond=global_cond,
            global_cond_uncond=global_cond_uncond,
            anchor_idx=None)
    return fm.normalizer['action'].unnormalize(sample[..., :Da])


def run_stochastic_path(fm, global_cond, global_cond_uncond,
                        noise_module, n_paths, device):
    """Run N stochastic paths with per-step noise injection.

    Returns:
        actions: (N, T, Da) unnormalized actions
        log_probs: (N,) log-probability of each path
        noise_xi: (N, num_fm_steps, T, Da) injected noise
    """
    B = global_cond.shape[0]
    T = fm.horizon
    Da = fm.action_dim
    n_steps = fm.num_fm_steps

    with torch.enable_grad():
        sigma = noise_module()  # (n_steps,)

        # Repeat global_cond for N paths
        global_cond_n = global_cond.expand(n_paths, -1)
        global_cond_uncond_n = global_cond_uncond.expand(n_paths, -1)

        # Noise init
        x_t = torch.randn(n_paths, T, Da, device=device)
        cond_data = torch.zeros(n_paths, T, Da, device=device)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        t_end = 1.0 - fm.sigma_min
        timesteps = torch.linspace(0.0, t_end, n_steps + 1, device=device)
        noise_xi = torch.zeros(n_paths, n_steps, T, Da, device=device)

        for i in range(n_steps):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t

            t_input = (t * fm.fm_t_scale).float().expand(n_paths)

            # CFG dual forward
            x_t_batched = torch.cat([x_t, x_t], dim=0)
            t_batched = torch.cat([t_input, t_input], dim=0)
            global_batched = torch.cat([global_cond_n, global_cond_uncond_n], dim=0)

            with torch.no_grad():  # frozen FM
                v_batched = fm.model(sample=x_t_batched, timestep=t_batched,
                                     global_cond=global_batched)
            v_cond, v_uncond = v_batched.chunk(2, dim=0)
            v = v_uncond + fm.cfg_lambda * (v_cond - v_uncond)

            # Inject noise
            xi_k = torch.randn(n_paths, T, Da, device=device)
            noise_xi[:, i] = xi_k
            sigma_k = sigma[i]
            x_t = x_t + dt * v + sigma_k * xi_k

        # Unnormalize
        actions = fm.normalizer['action'].unnormalize(x_t[..., :Da])

    return actions, sigma, noise_xi


def compute_log_prob(sigma, noise_xi, T, Da):
    """Compute log-prob from injected noise.

    log p = -0.5 * sum_k ( ||xi_k||^2 / sigma_k^2 + T*Da * log(2*pi*sigma_k^2) )
    """
    n_paths, n_steps = noise_xi.shape[:2]
    sigma_flat = sigma.view(1, -1, 1, 1)  # (1, n_steps, 1, 1)

    squared_norm = (noise_xi ** 2).sum(dim=(2, 3))  # (N, n_steps)
    log_det = (T * Da) * torch.log(2 * math.pi * sigma_flat ** 2)  # (1, n_steps)

    log_prob_per_step = -0.5 * (squared_norm / sigma_flat.squeeze() ** 2
                                + log_det.squeeze().unsqueeze(0))  # (N, n_steps)
    return log_prob_per_step.sum(dim=1)  # (N,)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--n_paths", type=int, default=N_PATHS)
    parser.add_argument("--seed", type=int, default=20260629)
    args = parser.parse_args()

    if args.ckpt is None:
        args.ckpt = str(PROJECT_DIR / "data/outputs/2026.06.17/15.22.50_v9_2_nobs8_bc/checkpoints/latest.ckpt")
    if args.output_dir is None:
        ts = time.strftime("%Y%m%d_%H%M")
        args.output_dir = str(PROJECT_DIR / "program" / "shiyan" /
                              f"stochastic_path_toy_test_{ts}")
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)

    def log(msg):
        print(msg)
        with open(os.path.join(args.output_dir, "toy_test.log"), "a") as f:
            f.write(msg + "\n")
    # Clear log
    with open(os.path.join(args.output_dir, "toy_test.log"), "w") as _:
        pass

    log("=" * 60)
    log("P0-5: Stochastic Path Log-Prob Toy Test")
    log(f"Checkpoint: {args.ckpt}")
    log(f"Device: {device}")
    log(f"N paths: {args.n_paths}")
    log(f"Seed: {args.seed}")
    log("=" * 60)

    # ── Load ──
    log("\n[1/5] Loading frozen BC policy + obs...")
    fm, global_cond, global_cond_uncond = load_policy_and_obs(args.ckpt, device)
    log(f"  global_cond shape: {global_cond.shape}")
    T, Da = fm.horizon, fm.action_dim
    log(f"  horizon={T}, action_dim={Da}, num_fm_steps={fm.num_fm_steps}")

    # ── Noise module ──
    log("\n[2/5] Creating ReinFlowNoiseModule...")
    noise_module = ReinFlowNoiseModule(
        num_fm_steps=fm.num_fm_steps,
        sigma_min=1e-4, sigma_max=0.10, init_sigma=0.01,
    ).to(device)
    log(f"  {noise_module}")

    # ── Deterministic teacher ──
    log("\n[3/5] Deterministic teacher...")
    teacher_action = run_deterministic(fm, global_cond, global_cond_uncond, device)
    teacher_norm = float(teacher_action.norm(dim=(1, 2)).mean())
    log(f"  Teacher action norm: {teacher_norm:.4f}")

    # ── Stochastic paths (×128) ──
    log(f"\n[4/5] Running {args.n_paths} stochastic paths...")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    t0 = time.time()
    actions, sigma, noise_xi = run_stochastic_path(
        fm, global_cond, global_cond_uncond, noise_module, args.n_paths, device)
    elapsed = time.time() - t0

    log(f"  Done in {elapsed:.2f}s ({elapsed/args.n_paths*1000:.1f}ms/path)")
    log(f"  actions shape: {actions.shape}")
    log(f"  sigma values: {sigma.detach().cpu().tolist()}")

    # ── Compute statistics ──
    log("\n[5/5] Analysis...")
    actions_np = actions.detach().cpu().numpy()

    # Log-prob
    log_prob = compute_log_prob(sigma.detach(), noise_xi, T, Da)
    lp_np = log_prob.detach().cpu().numpy()

    # Action norms
    action_norms = np.linalg.norm(actions_np.reshape(args.n_paths, -1), axis=1)

    # MSE vs teacher
    teacher_np = teacher_action.cpu().numpy()
    mse = np.mean((actions_np - teacher_np) ** 2, axis=(1, 2))

    # World-frame stats
    world_std = actions_np.std(axis=(0, 1))  # (11,) per-dim std in world frame
    world_rms = float(np.sqrt(np.mean(world_std ** 2)))

    # Checks
    n_nan_lp = int(np.isnan(lp_np).sum() + np.isinf(lp_np).sum())
    n_nan_act = int(np.isnan(action_norms).sum() + np.isinf(action_norms).sum())
    lp_std = float(np.std(lp_np))
    lp_mean = float(np.mean(lp_np))
    mse_mean = float(np.mean(mse))
    mse_ratio = mse_mean / BC_ACTION_VARIANCE if BC_ACTION_VARIANCE > 0 else float("inf")

    # Sigma gradient
    sigma_grad_norm = 0.0
    if sigma.grad is not None:
        sigma_grad_norm = sigma.grad.norm().item()

    # Reproducibility: two runs with same seed must give identical results
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    actions_rep_a, sigma_rep_a, noise_xi_rep_a = run_stochastic_path(
        fm, global_cond, global_cond_uncond, noise_module, args.n_paths, device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    actions_rep_b, sigma_rep_b, noise_xi_rep_b = run_stochastic_path(
        fm, global_cond, global_cond_uncond, noise_module, args.n_paths, device)
    max_diff = float(torch.abs(actions_rep_a - actions_rep_b).max().item())
    reproducibility_pass = max_diff < 1e-6

    results = OrderedDict([
        ("meta", OrderedDict([
            ("task", "P0-5 stochastic path log-prob toy test"),
            ("checkpoint", args.ckpt),
            ("n_paths", args.n_paths),
            ("device", str(device)),
            ("eval_time", time.strftime("%Y-%m-%dT%H:%M:%S")),
        ])),
        ("noise_module", OrderedDict([
            ("sigma_min", noise_module.sigma_min),
            ("sigma_max", noise_module.sigma_max),
            ("sigma_actual", sigma.detach().cpu().tolist()),
            ("rho_values", noise_module.rho.detach().cpu().tolist()),
        ])),
        ("deterministic", OrderedDict([
            ("teacher_action_norm", round(teacher_norm, 4),
        )])),
        ("log_prob", OrderedDict([
            ("mean", round(lp_mean, 2)),
            ("std", round(lp_std, 4)),
            ("min", round(float(lp_np.min()), 2)),
            ("max", round(float(lp_np.max()), 2)),
            ("nan_count", n_nan_lp),
            ("inf_count", n_nan_act),
        ])),
        ("action", OrderedDict([
            ("norm_mean", round(float(np.mean(action_norms)), 4)),
            ("norm_std", round(float(np.std(action_norms)), 4)),
            ("teacher_norm", round(teacher_norm, 4)),
        ])),
        ("mse_vs_teacher", OrderedDict([
            ("mean", round(mse_mean, 6)),
            ("std", round(float(np.std(mse)), 6)),
            ("bc_action_variance", round(BC_ACTION_VARIANCE, 6)),
            ("ratio_to_bc_var", round(mse_ratio, 6)),
        ])),
        ("world_frame", OrderedDict([
            ("per_dim_std", [round(float(s), 4) for s in world_std]),
            ("rms_mean", round(world_rms, 4)),
            ("recommendation", (
                f"{'< 0.02m → 建议 sigma_max=0.20' if world_rms < 0.02 else 'OK'}"
            )),
        ])),
        ("reproducibility", OrderedDict([
            ("max_diff", round(max_diff, 8)),
            ("pass", reproducibility_pass),
        ])),
        ("gradient", OrderedDict([
            ("sigma_grad_norm", round(sigma_grad_norm, 6)),
            ("has_grad", sigma.grad is not None),
        ])),
        ("gates", OrderedDict([
            ("no_nan_inf", n_nan_lp == 0 and n_nan_act == 0),
            ("log_prob_std_positive", lp_std > 0),
            ("log_prob_std_not_exploding", lp_std < 1e6),
            ("mse_under_10pct_bc_var", mse_ratio < 0.10),
            ("reproducible", reproducibility_pass),
            ("pass", False),  # set below
        ])),
        ("sigma_max_recommendation", (
            f"sigma_max=0.10, world RMS={world_rms:.4f}m "
            f"{'→ 建议上调到 sigma_max=0.20' if world_rms < 0.02 else '→ OK'}"
        )),
    ])

    # Final gate
    results["gates"]["pass"] = all([
        results["gates"]["no_nan_inf"],
        results["gates"]["log_prob_std_positive"],
        results["gates"]["log_prob_std_not_exploding"],
        results["gates"]["mse_under_10pct_bc_var"],
        results["gates"]["reproducible"],
    ])

    # ── Save ──
    with open(os.path.join(args.output_dir, "stochastic_path_toy_test.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    # ── Report ──
    log("\n" + "=" * 60)
    log("RESULTS")
    log("=" * 60)
    log(f"\n📊 Log-prob: mean={lp_mean:.2f}, std={lp_std:.4f}, "
        f"range=[{float(lp_np.min()):.2f}, {float(lp_np.max()):.2f}]")
    log(f"   NaN: {n_nan_lp}, Inf: {n_nan_act}")
    log(f"   Gate: {'✅' if results['gates']['no_nan_inf'] else '❌'} no_nan_inf")

    log(f"\n📏 Action: norm mean={float(np.mean(action_norms)):.4f} "
        f"(teacher={teacher_norm:.4f})")
    log(f"   MSE vs teacher: {mse_mean:.6f} (BC var={BC_ACTION_VARIANCE:.6f}, "
        f"ratio={mse_ratio:.4%})")
    log(f"   Gate: {'✅' if results['gates']['mse_under_10pct_bc_var'] else '❌'} MSE < 10% BC var")

    log(f"\n🌍 World-frame: RMS={world_rms:.4f}m")
    log(f"   {results['sigma_max_recommendation']}")

    log(f"\n🔁 Reproducibility: max_diff={max_diff:.8f} "
        f"{'✅' if reproducibility_pass else '❌'}")

    log(f"\n📈 Grad: {'✅' if sigma.grad is not None else '⚠️'} sigma grad exists, "
        f"norm={sigma_grad_norm:.6f}")

    verdict = "✅ PASS" if results["gates"]["pass"] else "❌ FAIL"
    log(f"\nOverall: {verdict}")

    result_file = os.path.join(args.output_dir, "stochastic_path_toy_test.json")
    print(f"\nResults: {result_file}")


if __name__ == "__main__":
    main()
