#!/usr/bin/env python3
"""
quantify_sampling_v13.py — 量化 v1.3 start/goal 采样质量

采样 5k-10k episode config，统计:
  - 每 role 的 rejection 分布 (distance, clearance, A*, path_stretch)
  - Fallback 率 (validated vs invalid)
  - A* path_stretch 分布
  - 每 role 激活频率

纯 env 采样，不加载 policy / GPU。
用法: python scripts/quantify_sampling_v13.py [--n_total 8000] [--seed 42]
"""

import argparse, json, math, sys, time, os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffusion_policy_3d.env.kinematic_uav_env import KinematicUAVEnv
from diffusion_policy_3d.env.collision_geom import load_obstacles

AUV_ROOT = Path(os.path.expanduser("~/AUV"))
WORLD_DIR = AUV_ROOT / "PX4-Autopilot/Tools/sitl_gazebo/worlds"
MANIFEST_PATH = AUV_ROOT / "program/reports/routeA_rl_env_v1/calibration/admission_manifest.json"

ROLES = ['anticipatory', 'dense_recovery', 'support_expansion', 'yaw_lateral_recovery']
ROLE_LABELS = {
    'anticipatory': 'Anticipatory',
    'dense_recovery': 'Dense Recovery',
    'support_expansion': 'Support Expansion',
    'yaw_lateral_recovery': 'Yaw Lateral',
}


def load_dense_worlds(manifest_path: Path, split: str = 'rl_train') -> List[Tuple[str, str]]:
    """Load admitted dense_recovery worlds from manifest."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    pool = []
    for entry in manifest.get('results', []):
        wsplit = entry.get('split', '')
        admission = entry.get('admission_status', '')
        wname = entry['world_uid']
        if wname.endswith('.world'):
            wname = wname[:-6]

        # Only dense train worlds admitted for dense_recovery
        if wsplit != split:
            continue
        if 'dense_recovery' not in admission:
            continue

        obs_json = WORLD_DIR / f"{wname}.world.obstacles.json"
        if obs_json.exists():
            pool.append((wname, str(obs_json)))

    return pool


def sample_one_world(wname: str, obs_json: str, n_per_role: int,
                     curriculum_stage: str = 'C2', z_max: float = 5.0,
                     seed: int = 42) -> dict:
    """Sample episodes from one world, collecting per-role stats."""
    # Create a bare env (no synth_pc needed for sampling only)
    env = KinematicUAVEnv(
        world_id=4,
        curriculum_stage=curriculum_stage,
        z_max=z_max,
        seed=seed,
    )
    # Manually set obstacles for this world (bypass synth_pc)
    env.obstacles = load_obstacles(obs_json)
    env.obstacles_json = obs_json
    env.world_id = 0  # dummy
    # Invalidate occ_grid cache
    if hasattr(env, '_occ_grid_cache'):
        env._occ_grid_cache = None

    world_stats = {
        'world': wname,
        'n_obstacles': len(env.obstacles),
        'roles': {},
    }

    for role in ROLES:
        role_rejections = defaultdict(int)
        role_fallback = {'fallback_validated': 0, 'fallback_invalid': 0}
        fallback_reasons = defaultdict(int)
        path_stretches = []
        a_star_path_lens = []
        n_accepted = 0
        total_attempts = 0

        for i in range(n_per_role):
            # Reset episode tracking (env.reset() does more than we need here)
            env._episode_role = role
            env._episode_fallback_type = None
            env._episode_fallback_reasons = []
            env._last_rejection_histogram = {}

            # Call the actual sampling method
            start_pos, start_yaw, goal_pos = env._sample_start_goal(role=role)

            # Collect rejection histogram
            rh = env._last_rejection_histogram
            for k, v in rh.items():
                if k != 'success':
                    role_rejections[k] += v
            total_attempts += sum(v for v in rh.values())

            # Track fallback
            fb_type = env._episode_fallback_type
            if fb_type:
                role_fallback[fb_type] += 1
                for reason in env._episode_fallback_reasons:
                    fallback_reasons[reason] += 1
            else:
                n_accepted += 1

        # Calculate rates
        total = n_per_role
        world_stats['roles'][role] = {
            'n_total': total,
            'n_accepted': n_accepted,
            'accept_rate': n_accepted / total if total > 0 else 0,
            'total_attempts': total_attempts,
            'avg_attempts_per_success': total_attempts / max(n_accepted, 1),
            'rejection_histogram': dict(role_rejections),
            'fallback': role_fallback,
            'fallback_reasons': dict(fallback_reasons),
        }

    return world_stats


def main():
    parser = argparse.ArgumentParser(description='Quantify v1.3 sampling quality')
    parser.add_argument('--n_total', type=int, default=8000,
                        help='Total episodes across all worlds+roles (default 8000)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--curriculum', type=str, default='C2',
                        choices=['C0', 'C1', 'C2'])
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path (default: program/shiyan/sampling_quant_v13_<date>.json)')
    args = parser.parse_args()

    # Load dense worlds
    print(f"Loading dense worlds from manifest: {MANIFEST_PATH}")
    dense_worlds = load_dense_worlds(MANIFEST_PATH, 'rl_train')
    if not dense_worlds:
        print("ERROR: No dense train worlds found in manifest. Trying fallback...")
        # Fallback: scan directory for worlds with >=25 obstacles
        for f in sorted(WORLD_DIR.glob("*.obstacles.json")):
            obs = load_obstacles(str(f))
            if len(obs) >= 25:
                wname = f.stem.replace('.obstacles', '')
                dense_worlds.append((wname, str(f)))
        print(f"Fallback: found {len(dense_worlds)} dense worlds by obstacle count")

    n_worlds = len(dense_worlds)
    n_per_role_per_world = max(1, args.n_total // (n_worlds * len(ROLES)))
    n_total_actual = n_worlds * len(ROLES) * n_per_role_per_world

    print(f"Worlds: {n_worlds} | Roles: {len(ROLES)} | "
          f"Episodes per role per world: {n_per_role_per_world} | Total: {n_total_actual}")
    print(f"Curriculum: {args.curriculum}")
    print(f"Dense worlds: {[w[0] for w in dense_worlds]}")
    print()

    # Sample each world
    all_results = {
        'config': {
            'n_total': n_total_actual,
            'n_worlds': n_worlds,
            'n_roles': len(ROLES),
            'n_per_role_per_world': n_per_role_per_world,
            'curriculum_stage': args.curriculum,
            'seed': args.seed,
        },
        'worlds': [],
        'aggregate': {},
    }

    t_start = time.time()
    for wi, (wname, obs_json) in enumerate(dense_worlds):
        wseed = args.seed + wi * 1000
        print(f"[{wi+1}/{n_worlds}] {wname} ({len(load_obstacles(obs_json))} obstacles)...",
              end=' ', flush=True)
        t_world = time.time()
        wstats = sample_one_world(wname, obs_json, n_per_role_per_world,
                                  curriculum_stage=args.curriculum, seed=wseed)
        elapsed = time.time() - t_world
        # Quick summary
        for role in ROLES:
            rs = wstats['roles'][role]
            print(f"{role[:4]}:{rs['accept_rate']:.1%}/{rs['fallback']['fallback_validated']}/{rs['fallback']['fallback_invalid']}",
                  end=' ', flush=True)
        print(f"({elapsed:.1f}s)")
        all_results['worlds'].append(wstats)

    total_elapsed = time.time() - t_start

    # ── Aggregate across worlds ──
    agg = {
        'n_total_episodes': 0,
        'n_accepted': 0,
        'n_fallback_validated': 0,
        'n_fallback_invalid': 0,
        'total_rejection_histogram': defaultdict(int),
        'total_fallback_reasons': defaultdict(int),
        'per_role': {},
    }

    for role in ROLES:
        role_agg = {
            'n_total': 0,
            'n_accepted': 0,
            'n_fallback_validated': 0,
            'n_fallback_invalid': 0,
            'rejection_histogram': defaultdict(int),
            'fallback_reasons': defaultdict(int),
        }
        for ws in all_results['worlds']:
            rs = ws['roles'][role]
            role_agg['n_total'] += rs['n_total']
            role_agg['n_accepted'] += rs['n_accepted']
            role_agg['n_fallback_validated'] += rs['fallback']['fallback_validated']
            role_agg['n_fallback_invalid'] += rs['fallback']['fallback_invalid']
            for k, v in rs['rejection_histogram'].items():
                role_agg['rejection_histogram'][k] += v
            for k, v in rs['fallback_reasons'].items():
                role_agg['fallback_reasons'][k] += v

        agg['n_accepted'] += role_agg['n_accepted']
        agg['n_fallback_validated'] += role_agg['n_fallback_validated']
        agg['n_fallback_invalid'] += role_agg['n_fallback_invalid']
        agg['n_total_episodes'] += role_agg['n_total']

        total_accept = role_agg['n_accepted'] + role_agg['n_fallback_validated'] + role_agg['n_fallback_invalid']
        role_agg['accept_rate'] = role_agg['n_accepted'] / max(total_accept, 1)
        role_agg['fallback_rate'] = (role_agg['n_fallback_validated'] + role_agg['n_fallback_invalid']) / max(total_accept, 1)
        role_agg['fallback_valid_rate'] = role_agg['n_fallback_validated'] / max(role_agg['n_fallback_validated'] + role_agg['n_fallback_invalid'], 1)
        # Convert defaultdict to plain dict for JSON
        role_agg['rejection_histogram'] = dict(role_agg['rejection_histogram'])
        role_agg['fallback_reasons'] = dict(role_agg['fallback_reasons'])
        agg['per_role'][role] = role_agg

    agg['overall_accept_rate'] = agg['n_accepted'] / max(agg['n_total_episodes'], 1)
    agg['overall_fallback_rate'] = (agg['n_fallback_validated'] + agg['n_fallback_invalid']) / max(agg['n_total_episodes'], 1)
    agg['total_rejection_histogram'] = dict(agg['total_rejection_histogram'])
    agg['total_fallback_reasons'] = dict(agg['total_fallback_reasons'])
    agg['elapsed_seconds'] = total_elapsed

    all_results['aggregate'] = agg

    # ── Print summary ──
    print(f"\n{'='*70}")
    print(f"Sampling Quality Report — {args.curriculum} Curriculum")
    print(f"{'='*70}")
    print(f"Worlds: {n_worlds}  |  Episodes: {agg['n_total_episodes']}  |  Time: {total_elapsed:.1f}s")
    print(f"\n{'Role':<22} {'Accept':>8} {'Fallback':>10} {'Fb-Valid':>10} {'Fb-Invalid':>10}")
    print(f"{'-'*22} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
    for role in ROLES:
        ra = agg['per_role'][role]
        print(f"{ROLE_LABELS[role]:<22} {ra['accept_rate']:>7.1%} {ra['fallback_rate']:>9.1%} "
              f"{ra['n_fallback_validated']:>9}  {ra['n_fallback_invalid']:>9}")

    print(f"\nRejection Histogram (total across all attempts):")
    for role in ROLES:
        ra = agg['per_role'][role]
        rh = ra['rejection_histogram']
        total_rej = sum(rh.values())
        parts = [f"{k}={v}" for k, v in sorted(rh.items())]
        print(f"  {ROLE_LABELS[role]:<22} total_rej={total_rej} | {', '.join(parts)}")

    if agg['total_fallback_reasons']:
        print(f"\nFallback Reasons (invalid fallbacks only):")
        for k, v in sorted(agg['total_fallback_reasons'].items()):
            print(f"  {k}: {v}")

    # ── Save ──
    if args.output:
        out_path = Path(args.output)
    else:
        from datetime import datetime
        date_str = datetime.now().strftime('%Y%m%d_%H%M')
        out_dir = AUV_ROOT / "program/shiyan"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"sampling_quant_v13_{args.curriculum}_{date_str}.json"

    # Serialize (convert any remaining non-serializable types)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to: {out_path}")

    return all_results


if __name__ == '__main__':
    main()
