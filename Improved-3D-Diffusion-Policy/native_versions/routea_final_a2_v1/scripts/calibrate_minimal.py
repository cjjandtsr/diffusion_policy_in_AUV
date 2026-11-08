#!/usr/bin/env python3
"""
Minimal calibration: evaluate DDPM BC on new worlds for admission decisions.
Replaces the broken calibrate_routea_rl_worlds_kinematic_v1.py dependency chain.
"""
import argparse, json, os, sys, time, numpy as np
from pathlib import Path
from collections import defaultdict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from diffusion_policy_3d.env.kinematic_uav_env_v2 import KinematicUAVEnv
from diffusion_policy_3d.model.common.synth_pointcloud import SynthPointCloud
from scripts.eval_ddpm_bc_baseline import load_ddpm_policy

BC_CKPT = "data/outputs/2026.05.17/13.36.20_v8_phase2_sweep_lambda0.5/checkpoints/latest.ckpt"
WORLDS_DIR = os.path.expanduser("~/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds")
MANIFEST_PATH = os.path.expanduser("~/AUV/program/reports/routeA_rl_env_v1/worldgen/routeA_rl_v1_world_manifest.json")
N_EPS = 4
MAX_STEPS = 200
CURRICULUM = "C2"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=str, required=True, help="Output admission manifest JSON path")
    p.add_argument("--n_eps", type=int, default=N_EPS)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--splits", type=str, nargs="+", default=["rl_train"])
    p.add_argument("--min_sr_for_admission", type=float, default=0.0,
                   help="Min BC SR to admit world (0=admit all geometrically feasible)")
    args = p.parse_args()
    device = args.device

    # Load manifest
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    all_worlds = manifest["worlds"]
    print(f"Loaded {len(all_worlds)} worlds from manifest")

    # Filter by split
    worlds = [w for w in all_worlds if w["split"] in args.splits]
    print(f"Filtered to {len(worlds)} worlds (splits={args.splits})")

    # Load BC policy
    policy = load_ddpm_policy(BC_CKPT, device)
    policy.eval()
    print(f"BC policy loaded")

    results = []
    n_admitted = 0
    t0 = time.time()

    for i, w in enumerate(worlds):
        uid = w["world_uid"]
        json_path = os.path.join(WORLDS_DIR, f"{uid}.world.obstacles.json")
        if not os.path.exists(json_path):
            print(f"  [{i+1:3d}/{len(worlds)}] SKIP (no obstacles.json): {uid}")
            continue

        role = w.get("role_validated") or w["role_generated"]
        synth_pc = SynthPointCloud(obstacles_json=json_path)

        successes, collisions, total_r, total_steps = 0, 0, 0.0, 0
        with torch.no_grad():
            for ep in range(args.n_eps):
                try:
                    env = KinematicUAVEnv(
                        world_pool=[(uid, json_path)],
                        synth_pc_pool={uid: synth_pc},
                        curriculum_stage=CURRICULUM,
                        max_episode_steps=MAX_STEPS,
                        episode_role=role,
                        disturb_enable=False,
                        mid_state_init_ratio=0.0,
                        seed=42 + i * 100 + ep)
                    obs_dict, _ = env.reset(seed=42 + i * 100 + ep)
                    done, ep_r, steps = False, 0.0, 0
                    while not done and steps < MAX_STEPS:
                        obs_b = {}
                        for k, v in obs_dict.items():
                            t = torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else v.to(device)
                            obs_b[k] = t.unsqueeze(0)
                        out = policy.predict_action(obs_b)
                        act = out['action_pred'] if isinstance(out, dict) else out
                        obs_dict, reward, terminated, truncated, _ = env.step(act.squeeze(0).cpu().numpy())
                        done = terminated or truncated
                        ep_r += float(reward)
                        steps += 1
                    collision = getattr(env, '_collision_detected', False)
                    success = terminated and not collision
                    if success:
                        successes += 1
                    if collision:
                        collisions += 1
                    total_r += ep_r
                    total_steps += steps
                except Exception as e:
                    print(f"  [{i+1:3d}/{len(worlds)}] ERROR ep={ep}: {e}")
                    continue

        sr = successes / args.n_eps if args.n_eps > 0 else 0.0
        cr = collisions / args.n_eps if args.n_eps > 0 else 0.0
        mean_r = total_r / args.n_eps if args.n_eps > 0 else 0.0
        mean_steps = total_steps / args.n_eps if args.n_eps > 0 else 0.0

        admitted = sr >= args.min_sr_for_admission or args.min_sr_for_admission == 0.0
        if admitted:
            n_admitted += 1

        result = {
            "world_uid": uid, "split": w["split"],
            "role_generated": w["role_generated"],
            "role_active": role, "admission_status": "admitted" if admitted else "rejected",
            "bc_sr": sr, "bc_cr": cr, "bc_mean_r": mean_r, "bc_mean_steps": mean_steps,
            "n_episodes": args.n_eps,
        }
        results.append(result)

        elapsed = time.time() - t0
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (len(worlds) - i - 1) / rate if rate > 0 else 0
        print(f"  [{i+1:3d}/{len(worlds)}] {role:25s} SR={sr:.2f} CR={cr:.2f} "
              f"admitted={admitted} | {rate:.1f}w/s ETA={eta:.0f}s | {uid[:60]}")

    # Build admission manifest
    out = {
        "schema_version": "routeA_rl_env_v1_minimal_calib",
        "calibration_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {"n_eps": args.n_eps, "bc_ckpt": BC_CKPT, "min_sr": args.min_sr_for_admission},
        "stats": {"total": len(results), "admitted": n_admitted, "rejected": len(results) - n_admitted},
        "results": results,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nDone: {n_admitted}/{len(results)} admitted → {args.output}")


if __name__ == "__main__":
    main()
