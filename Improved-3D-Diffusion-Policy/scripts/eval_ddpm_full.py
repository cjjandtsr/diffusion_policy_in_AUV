#!/usr/bin/env python3
"""
DDPM BC Baseline — Full Evaluation with per-episode data recording.

Evaluates all DDPM checkpoints + FM v9_2 on layerA (20 worlds × 4 eps).
Saves complete per-episode results to JSON for reproducibility.

Usage:
  python scripts/eval_ddpm_full.py
"""
import json, os, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.eval_ddpm_bc_baseline import load_ddpm_policy
from scripts.train_fm_sde_ppo import load_bc_policy as load_fm_policy
from diffusion_policy_3d.env.kinematic_uav_env_v2 import KinematicUAVEnv
from diffusion_policy_3d.model.common.synth_pointcloud import SynthPointCloud

LAYER_A_DIR = os.path.expanduser('~/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds/ood_experiment_worlds/layerA')
N_EP = 4
MAX_STEPS = 200
DEVICE = 'cuda:0'
OUTPUT_DIR = Path('outputs/ddpm_bc_eval')

# ── Checkpoints ──
CKPTS = {
    'DDPM_0522_no_collision': {
        'path': 'data/outputs/2026.05.22/21.16.19_v8_phase1a/checkpoints/latest.ckpt',
        'type': 'ddpm',
    },
    'Phase2_lambda0.05': {
        'path': 'data/outputs/2026.05.17/11.25.34_v8_phase2_pilot_lambda0.05/checkpoints/latest.ckpt',
        'type': 'ddpm',
    },
    'Phase2_lambda0.3': {
        'path': 'data/outputs/2026.05.17/11.50.36_v8_phase2_sweep_lambda0.3/checkpoints/latest.ckpt',
        'type': 'ddpm',
    },
    'Phase2_lambda0.5': {
        'path': 'data/outputs/2026.05.17/13.36.20_v8_phase2_sweep_lambda0.5/checkpoints/latest.ckpt',
        'type': 'ddpm',
    },
    'Phase2_lambda1.0': {
        'path': 'data/outputs/2026.05.17/15.20.34_v8_phase2_sweep_lambda1.0/checkpoints/latest.ckpt',
        'type': 'ddpm',
    },
    'FM_v9_2': {
        'path': 'data/outputs/2026.06.17/15.22.50_v9_2_nobs8_bc/checkpoints/latest.ckpt',
        'type': 'fm',
    },
}


def wilson_ci(successes, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return p, max(0.0, center - margin), min(1.0, center + margin)


def get_layerA_worlds():
    """Return sorted (uid, json_path) for all layerA worlds."""
    worlds = []
    for f in sorted(os.listdir(LAYER_A_DIR)):
        if f.endswith('.obstacles.json'):
            uid = f.replace('.obstacles.json', '')
            worlds.append((uid, os.path.join(LAYER_A_DIR, f)))
    return worlds


def load_policy(ckpt_cfg):
    """Load policy by type."""
    ckpt_path = ckpt_cfg['path']
    if ckpt_cfg['type'] == 'ddpm':
        return load_ddpm_policy(ckpt_path, DEVICE)
    else:
        return load_fm_policy(ckpt_path, DEVICE)


def eval_one_policy(policy, worlds):
    """Evaluate policy on all worlds, returning per-episode data and aggregates."""
    all_eps = []
    policy.eval()

    for world_uid, json_path in worlds:
        synth_pc = SynthPointCloud(obstacles_json=json_path)
        env = KinematicUAVEnv(
            world_pool=[(world_uid, json_path)],
            synth_pc_pool={world_uid: synth_pc},
            curriculum_stage='C2',
            max_episode_steps=MAX_STEPS,
        )

        with torch.no_grad():
            for ep in range(N_EP):
                obs_dict, _ = env.reset()
                done, total_r, steps = False, 0.0, 0
                traj_positions = []

                while not done and steps < MAX_STEPS:
                    obs_b = {}
                    for k, v in obs_dict.items():
                        t = torch.from_numpy(v).to(DEVICE) if isinstance(v, np.ndarray) else v.to(DEVICE)
                        obs_b[k] = t.unsqueeze(0)
                    out = policy.predict_action(obs_b)
                    act = out['action_pred'] if isinstance(out, dict) else out
                    obs_dict, reward, terminated, truncated, _ = env.step(
                        act.squeeze(0).cpu().numpy())
                    done = terminated or truncated
                    total_r += float(reward)
                    steps += 1
                    # Record position each step
                    if 'agent_pos' in obs_dict:
                        ap = obs_dict['agent_pos']
                        if isinstance(ap, np.ndarray):
                            traj_positions.append(ap[-1, :3].tolist())

                collision = getattr(env, '_collision_detected', False)
                success = terminated and not collision

                ep_data = {
                    'world': world_uid,
                    'episode': ep,
                    'success': bool(success),
                    'collision': bool(collision),
                    'terminated': bool(terminated),
                    'total_reward': total_r,
                    'steps': steps,
                    'trajectory_length': len(traj_positions),
                }
                all_eps.append(ep_data)
        env.close()

    # Aggregate
    n = len(all_eps)
    n_success = sum(e['success'] for e in all_eps)
    n_collision = sum(e['collision'] for e in all_eps)
    sr, sr_lo, sr_hi = wilson_ci(n_success, n)
    cr, cr_lo, cr_hi = wilson_ci(n_collision, n)
    mean_r = np.mean([e['total_reward'] for e in all_eps])
    mean_steps = np.mean([e['steps'] for e in all_eps])

    # Per-world aggregation
    per_world = defaultdict(list)
    for e in all_eps:
        per_world[e['world']].append(e)

    world_summaries = {}
    for w, eps in sorted(per_world.items()):
        ns = sum(e['success'] for e in eps)
        nc = sum(e['collision'] for e in eps)
        world_summaries[w] = {
            'SR': ns / len(eps),
            'CR': nc / len(eps),
            'mean_r': np.mean([e['total_reward'] for e in eps]),
            'n_eps': len(eps),
        }

    return {
        'summary': {
            'n_episodes': n,
            'n_success': n_success,
            'n_collision': n_collision,
            'SR': float(sr),
            'SR_CI95_lo': float(sr_lo),
            'SR_CI95_hi': float(sr_hi),
            'CR': float(cr),
            'CR_CI95_lo': float(cr_lo),
            'CR_CI95_hi': float(cr_hi),
            'mean_reward': float(mean_r),
            'mean_steps': float(mean_steps),
        },
        'per_world': world_summaries,
        'episodes': all_eps,
    }


def main():
    worlds = get_layerA_worlds()
    print(f"[INIT] layerA: {len(worlds)} worlds × {N_EP} eps = {len(worlds) * N_EP} total episodes")
    print(f"[INIT] {len(CKPTS)} checkpoints to evaluate\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    all_summaries = {}
    for label, ckpt_cfg in CKPTS.items():
        if not os.path.exists(ckpt_cfg['path']):
            print(f"[SKIP] {label}: checkpoint not found at {ckpt_cfg['path']}")
            continue

        t0 = time.time()
        print(f"[EVAL] {label} ...")
        policy = load_policy(ckpt_cfg)
        result = eval_one_policy(policy, worlds)
        elapsed = time.time() - t0

        s = result['summary']
        print(f"  SR={s['SR']:.4f} [{s['SR_CI95_lo']:.4f}, {s['SR_CI95_hi']:.4f}]  "
              f"CR={s['CR']:.4f}  mean_r={s['mean_reward']:.2f}  "
              f"time={elapsed:.0f}s\n")

        all_summaries[label] = {
            'label': label,
            'ckpt_path': ckpt_cfg['path'],
            'ckpt_type': ckpt_cfg['type'],
            **result['summary'],
            'eval_time_s': elapsed,
        }

        # Save individual result
        out_path = OUTPUT_DIR / f'{label}_{timestamp}.json'
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"  [SAVE] {out_path}")

    # ── Comparison table ──
    print(f"\n{'='*80}")
    print(f"{'FINAL COMPARISON — layerA (20 worlds × 4 eps)':^80}")
    print(f"{'='*80}")
    header = f"{'Label':30s} {'SR':>8s} {'95% CI':>22s} {'CR':>8s} {'mean_r':>8s} {'Time':>7s}"
    print(header)
    print('-' * len(header))
    for label in CKPTS:
        if label not in all_summaries:
            continue
        s = all_summaries[label]
        print(f"{label:30s} {s['SR']:8.4f} "
              f"[{s['SR_CI95_lo']:.4f},{s['SR_CI95_hi']:.4f}]   "
              f"{s['CR']:8.4f} {s['mean_reward']:8.2f} {s.get('eval_time_s',0):6.0f}s")
    print(f"{'='*80}")

    # Save master summary
    master_path = OUTPUT_DIR / f'master_summary_{timestamp}.json'
    with open(master_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'eval_config': {
                'split': 'layerA',
                'n_worlds': len(worlds),
                'n_ep_per_world': N_EP,
                'max_steps': MAX_STEPS,
                'device': DEVICE,
                'world_dir': LAYER_A_DIR,
            },
            'results': all_summaries,
        }, f, indent=2, default=str)
    print(f"\n[SAVE] Master summary: {master_path}")


if __name__ == '__main__':
    main()
