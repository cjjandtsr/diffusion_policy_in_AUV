#!/usr/bin/env python3
"""
v1.4 完整指标评估 — paired BC vs PPO with v1.4 reward components.

与 eval_v13_metrics.py 的区别:
  - 使用 env 内置的 v1.4 reward (env._last_v14_terms / _last_v14_metrics)
  - 4 分量替代 8 分量: r_goal, r_safety, r_explore, r_recovery

Usage:
  python scripts/eval_v14_metrics.py --ckpt outputs/fm_sde_ppo/ppo_final.pt
  python scripts/eval_v14_metrics.py --bc_only --split rl_test
"""
import argparse, json, os, sys, time
import numpy as np
import torch
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
    if split == 'layerA':
        return os.path.join(LAYER_A_DIR, f'{world_uid}.obstacles.json')
    return os.path.join(WORLDS_DIR, f'{world_uid}.world.obstacles.json')


def load_policy(ckpt_path, device):
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


# ═══════════════════════════════════════════════════════════════
# Main eval
# ═══════════════════════════════════════════════════════════════

def eval_policy_v14(policy, device, label, split):
    """Evaluate policy with v1.4 metrics from env."""
    configs = load_episode_configs(split)
    n_worlds = len(configs)
    per_role_eps = defaultdict(list)
    done_count = 0

    for world_uid, world_configs in configs.items():
        json_path = get_json_path(split, world_uid)
        synth_pc = SynthPointCloud(obstacles_json=json_path)
        env = KinematicUAVEnv(
            world_pool=[(world_uid, json_path)],
            synth_pc_pool={world_uid: synth_pc},
            curriculum_stage=CURRICULUM,
            max_episode_steps=MAX_STEPS,
            disturb_enable=False,
            mid_state_init_ratio=0.0,
            reward_version='v14')

        role = world_configs[0].get('role', 'unknown')
        ep_results = []

        with torch.no_grad():
            for cfg in world_configs:
                obs_dict, _ = env.reset(options={'episode_config': cfg})
                done_flag = False
                steps = 0
                total_r = 0.0

                # Accumulate v1.4 components
                terms_sum = {"goal": 0.0, "safety": 0.0, "explore": 0.0, "recovery": 0.0}
                d_soft_h_vals = []
                d_ref_vals = []
                stall_count = 0
                stall_recovered_count = 0
                novel_cell_count = 0
                positions = [env._drone_pos.copy()]

                while not done_flag and steps < MAX_STEPS:
                    obs_b = {}
                    for k, v in obs_dict.items():
                        t = torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else v.to(device)
                        obs_b[k] = t.unsqueeze(0)
                    out = policy.predict_action(obs_b)
                    act = out['action_pred'].squeeze(0).cpu().numpy()

                    obs_dict, reward, terminated, truncated, _ = env.step(act)
                    done_flag = terminated or truncated
                    total_r += reward
                    steps += 1

                    # Collect v1.4 metrics from env
                    last_terms = getattr(env, '_last_v14_terms', {})
                    last_metrics = getattr(env, '_last_v14_metrics', {})
                    for k in terms_sum:
                        terms_sum[k] += last_terms.get(k, 0.0)
                    d_sh = last_metrics.get('d_soft_h', -1.0)
                    if d_sh >= 0:
                        d_soft_h_vals.append(d_sh)
                    d_ref = last_metrics.get('d_ref', -1.0)
                    if d_ref >= 0:
                        d_ref_vals.append(d_ref)
                    if last_metrics.get('stall_active', False):
                        stall_count += 1
                    if last_metrics.get('stall_recovered', False):
                        stall_recovered_count += 1
                    if last_metrics.get('novel_cell', False):
                        novel_cell_count += 1
                    positions.append(env._drone_pos.copy())

                collision = getattr(env, '_collision_detected', False)
                success = terminated and not collision

                # Path length
                path_len = 0.0
                for i in range(1, len(positions)):
                    path_len += float(np.linalg.norm(positions[i] - positions[i-1]))

                ep = {
                    'SR': 1.0 if success else 0.0,
                    'CR': 1.0 if collision else 0.0,
                    'mean_r': total_r,
                    'steps': steps,
                    'path_length': path_len,
                    'reward_total': sum(terms_sum.values()),
                    'r_goal': terms_sum['goal'],
                    'r_safety': terms_sum['safety'],
                    'r_explore': terms_sum['explore'],
                    'r_recovery': terms_sum['recovery'],
                    'd_soft_h_mean': float(np.mean(d_soft_h_vals)) if d_soft_h_vals else -1.0,
                    'd_soft_h_p10': float(np.percentile(d_soft_h_vals, 10)) if d_soft_h_vals else -1.0,
                    'd_ref_mean': float(np.mean(d_ref_vals)) if d_ref_vals else -1.0,
                    'stall_active_steps': stall_count,
                    'stall_recovered': stall_recovered_count,
                    'novel_cell_steps': novel_cell_count,
                }
                ep_results.append(ep)
                per_role_eps[role].append(ep)

        done_count += 1
        sr_world = np.mean([e['SR'] for e in ep_results])
        mean_r_world = np.mean([e['mean_r'] for e in ep_results])
        print(f'  [{done_count}/{n_worlds}] {role:25s} {world_uid[:55]:55s} '
              f'SR={sr_world:.3f} mean_r={mean_r_world:7.2f}')

    # ═══════════════════════════════════════════════════════════
    # Print tables
    # ═══════════════════════════════════════════════════════════
    print(f'\n{"="*90}')
    print(f'{label} — v1.4 Metrics ({n_worlds} worlds)')
    print(f'{"="*90}')

    role_order = ['anticipatory', 'dense_recovery', 'support_expansion', 'yaw_lateral_recovery', 'unknown']
    all_eps = []

    metric_keys = [
        ('SR', 'SR'), ('CR', 'CR'), ('mean_r', 'mean_r'), ('steps', 'steps'),
        ('path_length', 'path_length (m)'),
        ('reward_total', '--- r_total ---'),
        ('r_goal', '  r_goal'), ('r_safety', '  r_safety'),
        ('r_explore', '  r_explore'), ('r_recovery', '  r_recovery'),
        ('d_soft_h_mean', 'd_soft_h_mean'), ('d_soft_h_p10', 'd_soft_h_p10'),
        ('d_ref_mean', 'd_ref_mean (EMA)'),
        ('stall_active_steps', 'stall_active_steps'),
        ('stall_recovered', 'stall_recovered'),
        ('novel_cell_steps', 'novel_cell_steps'),
    ]

    for role in role_order:
        eps_list = per_role_eps.get(role, [])
        if not eps_list:
            continue
        all_eps.extend(eps_list)
        print(f'\n  {role} ({len(eps_list)} eps)')
        for key, label in metric_keys:
            vals = [e[key] for e in eps_list if key in e]
            if vals:
                print(f'    {label:35s} {np.mean(vals):10.4f}')

    print(f'\n  OVERALL ({len(all_eps)} eps)')
    for key, label in metric_keys:
        vals = [e[key] for e in all_eps if key in e]
        if vals:
            print(f'    {label:35s} {np.mean(vals):10.4f}')

    return {'per_role': per_role_eps, 'all_eps': all_eps}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, default=None)
    p.add_argument('--bc_only', action='store_true')
    p.add_argument('--split', type=str, default='rl_test')
    args = p.parse_args()
    device = 'cuda:0'

    t_start = time.time()

    bc_ckpt = 'data/outputs/2026.06.17/15.22.50_v9_2_nobs8_bc/checkpoints/latest.ckpt'
    bc_policy = load_policy(bc_ckpt, device)
    bc_results = eval_policy_v14(bc_policy, device, 'BC Baseline', args.split)

    if args.ckpt and not args.bc_only:
        ppo_policy = load_policy(args.ckpt, device)
        ppo_results = eval_policy_v14(ppo_policy, device, f'PPO', args.split)

        print(f'\n{"="*90}')
        print('Δ (PPO - BC) — Key v1.4 Metrics')
        print(f'{"="*90}')

        delta_keys = ['SR', 'CR', 'mean_r', 'reward_total', 'r_goal', 'r_safety',
                      'r_explore', 'r_recovery', 'd_soft_h_mean', 'd_soft_h_p10',
                      'stall_active_steps', 'stall_recovered', 'novel_cell_steps',
                      'path_length']

        for role in ['anticipatory', 'dense_recovery', 'support_expansion',
                      'yaw_lateral_recovery', 'OVERALL']:
            if role == 'OVERALL':
                bc_list = bc_results['all_eps']
                pp_list = ppo_results['all_eps']
            else:
                bc_list = bc_results['per_role'].get(role, [])
                pp_list = ppo_results['per_role'].get(role, [])
            if not bc_list:
                continue
            print(f'\n  {role}:')
            for k in delta_keys:
                bv = np.mean([e[k] for e in bc_list if k in e])
                pv = np.mean([e[k] for e in pp_list if k in e])
                if np.isnan(bv) or np.isnan(pv):
                    continue
                print(f'    {k:30s}  BC={bv:8.4f}  PPO={pv:8.4f}  Δ={pv-bv:+.4f}')

    print(f'\n[DONE] Total time: {time.time()-t_start:.1f}s')


if __name__ == '__main__':
    main()
