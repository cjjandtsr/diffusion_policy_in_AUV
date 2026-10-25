#!/usr/bin/env python3
"""
p0_6_bc_variance_calibration.py — P0-6: BC baseline variance calibration (§8.6)

测量 BC policy 在 Reward v1.3 下的自然方差，为 A2 gate 提供置信区间阈值。

设置:
  policy = frozen v9_2 BC/FM
  reward/env = Reward v1.3 eval mode
  scenarios = canonical, support, default, dense
  runs = 10
  episodes_per_run = 20
  paired seed bank = fixed per run; run_id changes seed bank group

输出:
  bc_variance_summary.json  — 每 run 指标 + 跨 run 95% CI
  bc_variance_table.md      — 可读报告

用法:
  python tools/p0_6_bc_variance_calibration.py [--device cuda:0]
"""

import json, os, sys, math, time, argparse
from collections import OrderedDict
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

import torch, dill
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from diffusion_policy_3d.env.kinematic_uav_env import KinematicUAVEnv
from diffusion_policy_3d.env.kinematic_uav_env_v2 import WORLDS_DIR, WORLD_ENV_MAP
from diffusion_policy_3d.model.common.synth_pointcloud import SynthPointCloud
from diffusion_policy_3d.policy.phase1a_policy_fm_rl import Phase1aPolicyFM
from diffusion_policy_3d.policy.fm_base_policy import FMBasePolicy

OOD_WORLDS = [4, 8, 11, 12, 18]
MAX_STEPS = 60
REWARD_TERM_KEYS = ["base", "pbrs", "onset", "future_clearance",
                    "trap", "flip", "support", "guard"]
SCENARIOS = ["canonical", "support", "default", "dense"]
SCENARIO_CFG = {
    "canonical":  {"z_min": 1.0, "z_max": 5.0, "disturb": True,  "n": 20},
    "support":    {"z_min": 1.0, "z_max": 2.5, "disturb": False, "n": 20},
    "default":    {"z_min": 1.0, "z_max": 5.0, "disturb": True,  "n": 20},
    "dense":      {"z_min": 1.0, "z_max": 5.0, "disturb": True,  "n": 20},
}

# ═══════════════════════════════════════════════════════════════════
# Policy loading (same as eval_reward_v13_short.py)
# ═══════════════════════════════════════════════════════════════════

def load_bc_policy(ckpt_path, device="cuda:0"):
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
        horizon=cfg["horizon"], n_action_steps=cfg["n_action_steps"],
        n_obs_steps=cfg["n_obs_steps"],
        obs_as_global_cond=pc.get("obs_as_global_cond", True),
        diffusion_step_embed_dim=pc.get("diffusion_step_embed_dim", 128),
        down_dims=pc.get("down_dims", (256, 512, 1024)),
        kernel_size=pc.get("kernel_size", 5), n_groups=pc.get("n_groups", 8),
        anchor_bank_path=None, cfg_lambda=1.5, num_fm_steps=4,
        sigma_min=0.005, fm_t_scale=50, solver="euler",
        fm_anchor_t_start=0.95, use_pc_color=False,
        pointnet_type="multi_stage_pointnet",
        pointcloud_encoder_cfg=pc.get("pointcloud_encoder_cfg", None),
        point_downsample=pc.get("point_downsample", True),
    )
    fm.load_state_dict(sd, strict=False)
    fm.to(device).eval()
    base_policy = FMBasePolicy(fm, device)
    dummy_z = torch.zeros(1, 176, device=device)
    dummy_obs = {"point_cloud": np.zeros((8, 4096, 3), dtype=np.float32),
                 "agent_pos": np.zeros((8, 9), dtype=np.float32)}
    with torch.no_grad():
        _ = base_policy(dummy_obs, dummy_z)
    return base_policy


def create_env(world_id, z_min, z_max, disturb, seed):
    wname = WORLD_ENV_MAP[world_id]
    obs_json = str(WORLDS_DIR / f"{wname}.obstacles.json")
    synth_pc = SynthPointCloud(obs_json, device="cpu")
    return KinematicUAVEnv(
        world_id=world_id, synth_pc=synth_pc,
        rl_action_steps=8, h_min=0.3,
        z_min=z_min, z_max=z_max,
        disturb_enable=disturb, disturb_prob=0.2, disturb_pos_scale=0.3,
        mid_state_init_ratio=0.0, seed=seed, reward_version="v13",
    )


def run_episode(env, base_policy, device, max_steps=MAX_STEPS):
    obs, info = env.reset()
    z = np.zeros(176, dtype=np.float32)
    z_tensor = torch.from_numpy(z).unsqueeze(0).to(device)
    ep_reward = 0.0
    done_reason = "truncated"
    for step_i in range(max_steps):
        raw_obs = env._get_obs_dict()
        raw_obs["point_cloud"] = raw_obs["point_cloud"][..., :3]
        with torch.no_grad():
            real_action = base_policy(raw_obs, z_tensor)
        obs, reward, terminated, truncated, info = env.step(real_action[0])
        ep_reward += float(reward)
        if terminated or truncated:
            done_reason = info.get("done_reason", "unknown")
            break
    reached = (done_reason == "reached")
    collided = ("collision" in done_reason or "penetrat" in done_reason
                or "crash" in done_reason)
    return {"reached": reached, "collided": collided,
            "done_reason": done_reason, "reward_total": round(ep_reward, 4)}


def run_scenario(name, cfg, base_policy, device, run_id, run_seed):
    n_ep = cfg["n"]
    results = []
    worlds = OOD_WORLDS
    eps_per_world = max(1, n_ep // len(worlds))
    remaining = n_ep
    for world_id in worlds:
        for ep in range(eps_per_world):
            if remaining <= 0:
                break
            ep_seed = run_seed + hash(f"{name}_run{run_id}_{world_id}_{ep}") % (2**31 - 1)
            env = create_env(world_id, cfg["z_min"], cfg["z_max"],
                             cfg["disturb"], ep_seed)
            ep_data = run_episode(env, base_policy, device)
            results.append(ep_data)
            env.close()
            remaining -= 1
    sr = float(np.mean([r["reached"] for r in results]))
    cr = float(np.mean([r["collided"] for r in results]))
    reward = float(np.mean([r["reward_total"] for r in results]))
    return {"n": len(results), "SR": round(sr, 4), "CR": round(cr, 4),
            "reward_mean": round(reward, 4)}


def compute_ci(values, confidence=0.95):
    """95% CI from a list of values."""
    arr = np.array(values)
    n = len(arr)
    if n < 2:
        return {"mean": float(np.mean(arr)) if n > 0 else 0,
                "std": 0, "ci_low": 0, "ci_high": 0, "n": n}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    se = std / math.sqrt(n)
    from scipy.stats import t as t_dist
    try:
        t_val = t_dist.ppf((1 + confidence) / 2, n - 1)
    except Exception:
        t_val = 1.96  # fallback to normal
    return {"mean": round(mean, 4), "std": round(std, 4),
            "ci_low": round(mean - t_val * se, 4),
            "ci_high": round(mean + t_val * se, 4),
            "ci_width_pp": round(2 * t_val * se * 100, 2), "n": n}


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--base_seed", type=int, default=20260629)
    args = parser.parse_args()

    if args.ckpt is None:
        args.ckpt = str(PROJECT_DIR / "data/outputs/2026.06.17/15.22.50_v9_2_nobs8_bc/checkpoints/latest.ckpt")
    if args.output_dir is None:
        ts = time.strftime("%Y%m%d_%H%M")
        args.output_dir = str(PROJECT_DIR / "program" / "shiyan" /
                              f"bc_variance_calibration_{ts}")
    os.makedirs(args.output_dir, exist_ok=True)

    log_path = os.path.join(args.output_dir, "bc_variance.log")
    log_file = open(log_path, "w")
    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log("=" * 60)
    log("P0-6: BC Baseline Variance Calibration")
    log(f"Checkpoint: {args.ckpt}")
    log(f"Runs: {args.runs}, Base seed: {args.base_seed}")
    log(f"Scenarios: {', '.join(SCENARIOS)}")
    log("=" * 60)

    log("\nLoading BC policy...")
    base_policy = load_bc_policy(args.ckpt, device=args.device)
    log("  OK")

    # Per-scenario: collect {run_id: {SR, CR, reward_mean}}
    all_results = {s: [] for s in SCENARIOS}

    for run_id in range(args.runs):
        run_seed = args.base_seed + run_id * 1000
        log(f"\nRun {run_id + 1}/{args.runs} (seed={run_seed})...")
        for sname in SCENARIOS:
            cfg = SCENARIO_CFG[sname]
            r = run_scenario(sname, cfg, base_policy, args.device,
                             run_id, run_seed)
            r["run_id"] = run_id
            all_results[sname].append(r)
            log(f"  {sname:12s}: SR={r['SR']:.3f}, CR={r['CR']:.3f}, "
                f"reward={r['reward_mean']:.1f} (n={r['n']})")

    # Compute CI per scenario
    log("\n" + "=" * 60)
    log("VARIANCE SUMMARY")
    log("=" * 60)

    summary = OrderedDict()
    for sname in SCENARIOS:
        srs = [r["SR"] for r in all_results[sname]]
        crs = [r["CR"] for r in all_results[sname]]
        rws = [r["reward_mean"] for r in all_results[sname]]

        sr_ci = compute_ci(srs)
        cr_ci = compute_ci(crs)
        rw_ci = compute_ci(rws)

        entry = OrderedDict([
            ("SR", sr_ci),
            ("CR", cr_ci),
            ("reward_mean", rw_ci),
            ("raw_runs", {
                "SR": [round(v, 4) for v in srs],
                "CR": [round(v, 4) for v in crs],
                "reward_mean": [round(v, 4) for v in rws],
            }),
        ])
        summary[sname] = entry

        log(f"\n{sname}:")
        log(f"  SR: {sr_ci['mean']:.4f} ± {sr_ci['std']:.4f}, "
            f"95%CI [{sr_ci['ci_low']:.4f}, {sr_ci['ci_high']:.4f}], "
            f"width={sr_ci['ci_width_pp']:.2f}pp")
        log(f"  CR: {cr_ci['mean']:.4f} ± {cr_ci['std']:.4f}, "
            f"95%CI [{cr_ci['ci_low']:.4f}, {cr_ci['ci_high']:.4f}], "
            f"width={cr_ci['ci_width_pp']:.2f}pp")
        log(f"  Reward: {rw_ci['mean']:.1f} ± {rw_ci['std']:.1f}, "
            f"95%CI [{rw_ci['ci_low']:.1f}, {rw_ci['ci_high']:.1f}]")

    # Save JSON
    with open(os.path.join(args.output_dir, "bc_variance_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Generate markdown table
    md = [
        "## P0-6 BC Baseline Variance",
        "",
        f"**Checkpoint**: {args.ckpt}",
        f"**Runs**: {args.runs}, **Base seed**: {args.base_seed}",
        "",
        "| Scenario | SR mean | SR 95% CI | SR CI width (pp) | CR mean | CR 95% CI | CR CI width (pp) |",
        "|----------|---------|-----------|-----------------|---------|-----------|-----------------|",
    ]
    for sname in SCENARIOS:
        s = summary[sname]
        sr = s["SR"]
        cr = s["CR"]
        md.append(
            f"| {sname} | {sr['mean']:.4f} | [{sr['ci_low']:.4f}, {sr['ci_high']:.4f}] "
            f"| {sr['ci_width_pp']:.2f}pp | {cr['mean']:.4f} | [{cr['ci_low']:.4f}, {cr['ci_high']:.4f}] "
            f"| {cr['ci_width_pp']:.2f}pp |"
        )

    md.extend([
        "",
        "### Raw per-run SR values",
        "",
    ])
    for sname in SCENARIOS:
        raw = summary[sname]["raw_runs"]["SR"]
        md.append(f"- {sname}: {[round(v, 3) for v in raw]}")

    md_path = os.path.join(args.output_dir, "bc_variance_table.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md))

    log(f"\nResults: {args.output_dir}")
    log_file.close()
    print(f"\nTable: {md_path}")


if __name__ == "__main__":
    main()
