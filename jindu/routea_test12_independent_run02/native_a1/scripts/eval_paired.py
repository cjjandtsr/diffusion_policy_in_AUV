#!/usr/bin/env python3
"""
Paired evaluation: BC baseline vs PPO-trained policy using fixed episode_configs.

Design ref: program/review/GPT_Review_StartGoal_Sampling_Design_20260709.md §2.3
- Each world has 4 pre-generated (start, goal, yaw) configs (master seed=42)
- pre/post eval reuse same configs → SR delta cleanly attributable (MC-5)
- mid_state_init_ratio=0.0, disturb_enable=False

Usage:
  python scripts/eval_paired.py --ckpt outputs/fm_sde_ppo/ppo_final.pt
  python scripts/eval_paired.py --bc_only
  python scripts/eval_paired.py --split rl_test  # default
  python scripts/eval_paired.py --split rl_val
  python scripts/eval_paired.py --split layerA
"""
import argparse, json, os, sys, torch, numpy as np
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffusion_policy_3d.env.kinematic_uav_env_v2 import KinematicUAVEnv
from diffusion_policy_3d.model.common.synth_pointcloud import SynthPointCloud

MAX_STEPS = 200
CURRICULUM = 'C2'
CONFIG_DIR = Path(__file__).resolve().parent.parent / 'data' / 'eval_configs'
WORLDS_DIR = os.path.expanduser('~/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds')
LAYER_A_DIR = os.path.join(WORLDS_DIR, 'ood_experiment_worlds', 'layerA')


def load_episode_configs(split):
    """Load pre-generated episode configs. Returns {world_uid: [cfg, ...]}."""
    config_map = {
        'rl_val': CONFIG_DIR / 'episode_configs_rl_val.json',
        'rl_test': CONFIG_DIR / 'episode_configs_rl_test.json',
        'layerA': CONFIG_DIR / 'episode_configs_layerA.json',
    }
    path = config_map.get(split)
    if path is None or not path.exists():
        raise FileNotFoundError(f'Config not found for split={split} at {path}')
    with open(path) as f:
        return json.load(f)


def get_json_path(split, world_uid):
    """Resolve world JSON path based on split (layerA uses different dir)."""
    if split == 'layerA':
        return os.path.join(LAYER_A_DIR, f'{world_uid}.obstacles.json')
    return os.path.join(WORLDS_DIR, f'{world_uid}.world.obstacles.json')


def load_policy(ckpt_path, device):
    """Load BC or PPO checkpoint."""
    from scripts.train_fm_sde_ppo import load_bc_policy as _load_bc
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'velocity_head' in ckpt:
        bc_path = 'data/outputs/2026.06.17/15.22.50_v9_2_nobs8_bc/checkpoints/latest.ckpt'
        policy = _load_bc(bc_path, device)
        vh_sd = ckpt['velocity_head']
        policy.model.load_state_dict(vh_sd, strict=False)
        print(f'[LOAD] PPO ckpt step={ckpt.get("step","?")}')
    else:
        policy = _load_bc(ckpt_path, device)
        print('[LOAD] BC checkpoint')
    policy.eval()
    return policy


def eval_policy(policy, device, label, split):
    """Evaluate policy using fixed episode_configs. Returns per-role and overall metrics."""
    configs = load_episode_configs(split)
    n_worlds = len(configs)
    total_eps = sum(len(cfgs) for cfgs in configs.values())

    results = defaultdict(list)
    done_count = 0

    for world_uid, world_configs in configs.items():
        json_path = get_json_path(split, world_uid)
        synth_pc = SynthPointCloud(obstacles_json=json_path)
        env = KinematicUAVEnv(
            world_pool=[(world_uid, json_path)],
            synth_pc_pool={world_uid: synth_pc},
            curriculum_stage=CURRICULUM,
            max_episode_steps=MAX_STEPS,
            disturb_enable=False,        # ★ eval: off
            mid_state_init_ratio=0.0)    # ★ eval: off

        ep_results = []
        with torch.no_grad():
            for cfg in world_configs:
                obs_dict, _ = env.reset(options={'episode_config': cfg})
                done, total_r, steps = False, 0.0, 0
                while not done and steps < MAX_STEPS:
                    obs_b = {}
                    for k, v in obs_dict.items():
                        t = torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else v.to(device)
                        obs_b[k] = t.unsqueeze(0)
                    out = policy.predict_action(obs_b)
                    act = out['action_pred'] if isinstance(out, dict) else out
                    obs_dict, reward, terminated, truncated, _ = env.step(
                        act.squeeze(0).cpu().numpy())
                    done = terminated or truncated
                    total_r += reward
                    steps += 1
                collision = getattr(env, '_collision_detected', False)
                success = terminated and not collision
                ep_results.append({
                    'r': total_r, 'steps': steps,
                    'collision': collision, 'success': success,
                    'episode_id': cfg.get('episode_id', '?'),
                })

        role = world_configs[0].get('role', 'unknown')
        results[role].append({'world': world_uid, 'eps': ep_results})
        done_count += 1
        mean_r = np.mean([e['r'] for e in ep_results])
        print(f'  [{done_count}/{n_worlds}] {role:25s} {world_uid[:55]:55s} mean_r={mean_r:7.2f}')

    # Summary
    print(f'\n=== {label} ({n_worlds} worlds × {total_eps // n_worlds} eps) ===')
    all_eps = []
    role_order = ['anticipatory', 'dense_recovery', 'support_expansion', 'yaw_lateral_recovery', 'unknown']
    for role in role_order:
        rs = results.get(role, [])
        if not rs:
            continue
        eps = [e for w in rs for e in w['eps']]
        all_eps.extend(eps)
        mean_r = np.mean([e['r'] for e in eps])
        sr = np.mean([1.0 if e['success'] else 0.0 for e in eps])
        cr = np.mean([1.0 if e['collision'] else 0.0 for e in eps])
        mean_steps = np.mean([e['steps'] for e in eps])
        print(f'  {role:25s}: mean_r={mean_r:7.2f}, SR={sr:.3f}, CR={cr:.3f}, steps={mean_steps:.1f}')

    overall_r = np.mean([e['r'] for e in all_eps])
    overall_sr = np.mean([1.0 if e['success'] else 0.0 for e in all_eps])
    overall_cr = np.mean([1.0 if e['collision'] else 0.0 for e in all_eps])
    print(f'  {"OVERALL":25s}: mean_r={overall_r:7.2f}, SR={overall_sr:.3f}, CR={overall_cr:.3f}')

    return {
        'per_role': {role: {
            'mean_r': np.mean([e['r'] for w in rs for e in w['eps']]),
            'SR': np.mean([1.0 if e['success'] else 0.0 for e in [e for w in rs for e in w['eps']]]),
            'CR': np.mean([1.0 if e['collision'] else 0.0 for e in [e for w in rs for e in w['eps']]]),
        } for role, rs in results.items()},
        'overall': {'mean_r': overall_r, 'SR': overall_sr, 'CR': overall_cr},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, default=None, help='PPO checkpoint path')
    p.add_argument('--bc_only', action='store_true')
    p.add_argument('--split', type=str, default='rl_test', help='rl_val | rl_test | layerA')
    args = p.parse_args()
    device = 'cuda:0'

    # --- BC baseline ---
    bc_ckpt = 'data/outputs/2026.06.17/15.22.50_v9_2_nobs8_bc/checkpoints/latest.ckpt'
    bc_policy = load_policy(bc_ckpt, device)
    bc_results = eval_policy(bc_policy, device, 'BC Baseline', args.split)

    # --- PPO (if provided) ---
    if args.ckpt and not args.bc_only:
        ppo_policy = load_policy(args.ckpt, device)
        ppo_results = eval_policy(ppo_policy, device, f'PPO ({args.ckpt})', args.split)

        # --- Delta (paired: same configs, different policies) ---
        print(f'\n=== Δ (PPO - BC) [paired, deterministic] ===')
        role_order = ['anticipatory', 'dense_recovery', 'support_expansion', 'yaw_lateral_recovery', 'unknown']
        for role in role_order:
            bc = bc_results['per_role'].get(role)
            pp = ppo_results['per_role'].get(role, bc)
            if bc:
                dsr = pp['SR'] - bc['SR']
                dcr = pp['CR'] - bc['CR']
                dr = pp['mean_r'] - bc['mean_r']
                print(f'  {role:25s}: ΔSR={dsr:+.3f}, ΔCR={dcr:+.3f}, Δmean_r={dr:+.2f}')
        do_sr = ppo_results['overall']['SR'] - bc_results['overall']['SR']
        do_r = ppo_results['overall']['mean_r'] - bc_results['overall']['mean_r']
        print(f'  {"OVERALL":25s}: ΔSR={do_sr:+.3f}, Δmean_r={do_r:+.2f}')


if __name__ == '__main__':
    main()
