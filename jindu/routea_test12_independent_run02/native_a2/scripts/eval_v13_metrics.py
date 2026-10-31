#!/usr/bin/env python3
"""
v1.3 完整指标评估脚本 — paired BC vs PPO with all 14 capability metrics.

与 eval_paired.py 的区别:
  - 在每 step 调用 reward_v13.compute_reward_v13() 记录所有 reward 分量
  - 汇总 14 项能力指标 (Reward v1.3 设计 §5.2)
  - 输出每个 role + overall 的全指标表格

Usage:
  python scripts/eval_v13_metrics.py --ckpt outputs/fm_sde_ppo/ppo_final.pt
  python scripts/eval_v13_metrics.py --bc_only
  python scripts/eval_v13_metrics.py --split rl_test       # default
  python scripts/eval_v13_metrics.py --split rl_val
"""
import argparse, json, os, sys, math, time
import numpy as np
import torch
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffusion_policy_3d.env.kinematic_uav_env_v2 import KinematicUAVEnv
from diffusion_policy_3d.env.reward_v13 import (
    compute_reward_v13, create_reward_history, REWARD_V13_DEFAULTS,
    reconstruct_future_positions, temporal_softmin_clearance,
)
from diffusion_policy_3d.model.common.synth_pointcloud import SynthPointCloud

MAX_STEPS = 200
CURRICULUM = 'C2'
CONFIG_DIR = Path(__file__).resolve().parent.parent / 'data' / 'eval_configs'
WORLDS_DIR = os.path.expanduser('~/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds')
LAYER_A_DIR = os.path.join(WORLDS_DIR, 'ood_experiment_worlds', 'layerA')

# ═══════════════════════════════════════════════════════════════
# Config / policy loading (same as eval_paired.py)
# ═══════════════════════════════════════════════════════════════

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
# v1.3 metric collection helpers
# ═══════════════════════════════════════════════════════════════

def _compute_episode_v13_metrics(episode_data):
    """从单 episode 的 per-step 数据计算汇总指标 (设计 §5.2).

    episode_data: list of per-step dicts with keys:
      - 'reward_terms': dict of 8 reward components
      - 'metrics': dict from compute_reward_v13
      - 'pos_world': (3,) current position
      - 'collision': bool
      - 'reached': bool
      - 'action_chunk': (H,11) or None

    Returns: dict of episode-level v1.3 metrics.
    """
    N = len(episode_data)
    if N == 0:
        return {}

    # Accumulators
    reward_terms_sum = {k: 0.0 for k in ['base', 'pbrs', 'onset', 'future_clearance',
                                          'trap', 'flip', 'support', 'guard']}
    reward_total = 0.0

    d_soft_h_vals = []
    risk_gate_vals = []
    onset_active_count = 0
    onset_active_steps = []
    trap_active_count = 0
    trap_recovered_count = 0
    side_flip_vals = []
    loop_detected_count = 0
    loop_rate_vals = []
    low_density_count = 0
    support_density_vals = []
    free_cone_min_vals = []
    free_cone_p20_vals = []

    # Closest clearance across episode
    min_d_soft_h = float('inf')

    # Path length
    positions = []

    for step_data in episode_data:
        rt = step_data.get('reward_terms', {})
        for k in reward_terms_sum:
            reward_terms_sum[k] += rt.get(k, 0.0)

        m = step_data.get('metrics', {})
        d_sh = m.get('d_soft_h', -1.0)
        if d_sh >= 0:
            d_soft_h_vals.append(d_sh)
            if d_sh < min_d_soft_h:
                min_d_soft_h = d_sh

        rg = m.get('risk_gate', 0.0)
        risk_gate_vals.append(rg)

        if m.get('onset_active', False):
            onset_active_count += 1
            onset_active_steps.append(step_data)

        if m.get('trap_active', False):
            trap_active_count += 1
        if m.get('trap_recovered', False):
            trap_recovered_count += 1

        side_flip_vals.append(m.get('side_flip_count', 0))

        if m.get('loop_detected', False):
            loop_detected_count += 1
        loop_rate_vals.append(m.get('loop_detected_rate', 0.0))

        if m.get('low_density_cell', False):
            low_density_count += 1
        sd = m.get('support_cell_density', -1.0)
        if sd >= 0:
            support_density_vals.append(sd)

        fcm = m.get('l_free_cone_min', -1.0)
        if fcm >= 0:
            free_cone_min_vals.append(fcm)
        fcp = m.get('l_free_cone_p20', -1.0)
        if fcp >= 0:
            free_cone_p20_vals.append(fcp)

        pw = step_data.get('pos_world')
        if pw is not None:
            positions.append(pw)

    # ── Compute design §5.2 metrics ──
    out = {
        # §5.2 row 1
        'reward_total': sum(reward_terms_sum.values()),
        # §5.2 row 2: reward components
        'reward_base': reward_terms_sum['base'],
        'reward_pbrs': reward_terms_sum['pbrs'],
        'reward_onset': reward_terms_sum['onset'],
        'reward_future_clearance': reward_terms_sum['future_clearance'],
        'reward_trap': reward_terms_sum['trap'],
        'reward_flip': reward_terms_sum['flip'],
        'reward_support': reward_terms_sum['support'],
        'reward_guard': reward_terms_sum['guard'],
        # §5.2 row 3: future clearance softmin
        'future_clearance_softmin_mean': float(np.mean(d_soft_h_vals)) if d_soft_h_vals else -1.0,
        'future_clearance_softmin_p10': float(np.percentile(d_soft_h_vals, 10)) if d_soft_h_vals else -1.0,
        'future_clearance_softmin_min': float(min_d_soft_h) if min_d_soft_h < float('inf') else -1.0,
        # §5.2 row 4: onset
        'onset_active_steps': onset_active_count,
        'onset_active_rate': onset_active_count / max(N, 1),
        # §5.2 row 5: trap recovery
        'trap_active_steps': trap_active_count,
        'trap_recovered_count': trap_recovered_count,
        'trap_recovery_rate': trap_recovered_count / max(trap_active_count, 1) if trap_active_count > 0 else 0.0,
        # §5.2 row 6: side flip
        'side_flip_total': sum(side_flip_vals),
        'side_flip_per_step': float(np.mean(side_flip_vals)) if side_flip_vals else 0.0,
        # §5.2 row 7: loop
        'loop_detected_steps': loop_detected_count,
        'loop_detected_rate': float(np.mean(loop_rate_vals)) if loop_rate_vals else 0.0,
        # §5.2 row 8: support
        'low_density_cell_steps': low_density_count,
        'low_density_rate': low_density_count / max(N, 1),
        'support_cell_density_mean': float(np.mean(support_density_vals)) if support_density_vals else -1.0,
        # §5.2 row 9: free cone (PBRS input)
        'free_cone_min_mean': float(np.mean(free_cone_min_vals)) if free_cone_min_vals else -1.0,
        'free_cone_p20_mean': float(np.mean(free_cone_p20_vals)) if free_cone_p20_vals else -1.0,
        # §5.2 row 10: path length
        'path_length': float(_compute_path_length(positions)),
        # §5.2 row 10b: deadend dwell (trap steps)
        'deadend_dwell_steps': trap_active_count,
        # extra
        'risk_gate_mean': float(np.mean(risk_gate_vals)) if risk_gate_vals else 0.0,
        'n_steps': N,
    }
    return out


def _compute_path_length(positions):
    """Cumulative 3D path length from position sequence."""
    if len(positions) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(positions)):
        total += float(np.linalg.norm(positions[i] - positions[i-1]))
    return total


# ═══════════════════════════════════════════════════════════════
# Main eval
# ═══════════════════════════════════════════════════════════════

def eval_policy_v13(policy, device, label, split):
    """Evaluate policy with full v1.3 metrics."""
    configs = load_episode_configs(split)
    n_worlds = len(configs)

    # Per-world results for detailed reporting
    world_results = []  # list of {world_uid, role, eps: [{v13_metrics, sr, cr, ...}]}
    per_role_eps = defaultdict(list)  # role -> list of episode v13 metric dicts

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
            mid_state_init_ratio=0.0)

        role = world_configs[0].get('role', 'unknown')
        ep_results = []

        with torch.no_grad():
            for cfg in world_configs:
                obs_dict, _ = env.reset(options={'episode_config': cfg})
                done_flag = False
                steps = 0
                total_r_env = 0.0  # env's internal reward (legacy)

                # v1.3 history (per-episode)
                v13_history = create_reward_history(
                    trap_window=REWARD_V13_DEFAULTS['trap_window'],
                    loop_window=REWARD_V13_DEFAULTS['loop_window'],
                    n_base=REWARD_V13_DEFAULTS['n_base'],
                )
                v13_step_data = []  # list of per-step v1.3 data

                # Track position for path length
                prev_pos = env._drone_pos.copy()

                while not done_flag and steps < MAX_STEPS:
                    # ── Prepare observation batch ──
                    obs_b = {}
                    for k, v in obs_dict.items():
                        t = torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else v.to(device)
                        obs_b[k] = t.unsqueeze(0)

                    out = policy.predict_action(obs_b)
                    action_pred = out['action_pred']  # (1, n_action_steps, 11)
                    action_np = action_pred.squeeze(0).cpu().numpy()  # (n_action_steps, 11)

                    # ── Capture s_t (before step) ──
                    s_t = {
                        'pos_world': env._drone_pos.copy(),
                        'yaw': float(env._drone_yaw),
                        'vel_world': env._drone_vel.copy(),
                        'z': float(env._drone_pos[2]),
                    }

                    # ── Step env ──
                    obs_dict, reward_env, terminated, truncated, _ = env.step(action_np)
                    done_flag = terminated or truncated
                    total_r_env += reward_env
                    steps += 1

                    # ── Capture s_tp1 (after step) ──
                    collision = getattr(env, '_collision_detected', False)
                    s_tp1 = {
                        'pos_world': env._drone_pos.copy(),
                        'yaw': float(env._drone_yaw),
                        'vel_world': env._drone_vel.copy(),
                        'z': float(env._drone_pos[2]),
                        'collided': collision,
                    }

                    # ── Compute v1.3 reward ──
                    try:
                        _, v13_terms, v13_metrics, v13_history = compute_reward_v13(
                            s_t=s_t,
                            action_chunk=action_np,
                            s_tp1=s_tp1,
                            pointcloud=None,
                            obstacle_json=env.obstacles,
                            goal_world=env.goal_w,
                            history=v13_history,
                            offline_density=None,  # support metrics will be 0
                            cfg=None,              # use defaults
                            clearance_fn=None,     # use collision_geom default
                        )
                    except Exception as e:
                        # Failsafe: if v13 reward crashes, record zeros
                        v13_terms = {k: 0.0 for k in ['base', 'pbrs', 'onset', 'future_clearance',
                                                       'trap', 'flip', 'support', 'guard']}
                        v13_metrics = {}

                    v13_step_data.append({
                        'reward_terms': v13_terms,
                        'metrics': v13_metrics,
                        'pos_world': s_tp1['pos_world'],
                        'collision': collision,
                        'reached': terminated and not collision,
                        'action_chunk': action_np,
                    })

                    prev_pos = s_tp1['pos_world']

                # ── Episode summary ──
                reached = terminated and not collision
                success = reached  # SR = terminated and not collision
                ep_v13 = _compute_episode_v13_metrics(v13_step_data)
                ep_v13['SR'] = 1.0 if success else 0.0
                ep_v13['CR'] = 1.0 if collision else 0.0
                ep_v13['mean_r_env'] = total_r_env
                ep_v13['steps'] = steps
                ep_v13['reached'] = reached
                ep_v13['collision'] = collision
                ep_v13['episode_id'] = cfg.get('episode_id', '?')

                ep_results.append(ep_v13)
                per_role_eps[role].append(ep_v13)

        world_results.append({'world': world_uid, 'role': role, 'eps': ep_results})
        done_count += 1

        # Progress line
        sr_world = np.mean([e['SR'] for e in ep_results])
        mean_r_world = np.mean([e['mean_r_env'] for e in ep_results])
        print(f'  [{done_count}/{n_worlds}] {role:25s} {world_uid[:55]:55s} '
              f'SR={sr_world:.3f} mean_r_env={mean_r_world:7.2f}')

    # ═══════════════════════════════════════════════════════════
    # Print full v1.3 metrics table
    # ═══════════════════════════════════════════════════════════
    print(f'\n{"="*100}')
    print(f'{label} — v1.3 Full Metrics ({n_worlds} worlds)')
    print(f'{"="*100}')

    role_order = ['anticipatory', 'dense_recovery', 'support_expansion', 'yaw_lateral_recovery', 'unknown']
    all_eps = []

    for role in role_order:
        eps_list = per_role_eps.get(role, [])
        if not eps_list:
            continue
        all_eps.extend(eps_list)

        _print_role_metrics(role, eps_list)

    # Overall
    print(f'\n{"─"*100}')
    print('OVERALL')
    print(f'{"─"*100}')
    _print_role_metrics('OVERALL', all_eps)

    return {
        'per_role': {role: _agg_eps(eps) for role, eps in per_role_eps.items()},
        'overall': _agg_eps(all_eps),
        'world_results': world_results,
    }


def _agg_eps(eps_list):
    """Aggregate list of episode metric dicts into summary dict."""
    if not eps_list:
        return {}
    keys = eps_list[0].keys()
    out = {}
    for k in keys:
        vals = [e[k] for e in eps_list if isinstance(e[k], (int, float, np.floating, np.integer))]
        if vals:
            out[k] = float(np.mean(vals))
    out['n_eps'] = len(eps_list)
    return out


def _print_role_metrics(role, eps_list):
    """Print one row (or role block) of v1.3 metrics."""
    if not eps_list:
        return
    a = _agg_eps(eps_list)
    n = a.get('n_eps', 0)

    print(f'\n  {role} ({n} eps)')
    print(f'  {"Metric":45s} {"Value":>10s}')
    print(f'  {"-"*55}')

    # Core eval metrics
    for key, label in [
        ('SR', 'SR (Success Rate)'),
        ('CR', 'CR (Collision Rate)'),
        ('mean_r_env', 'mean_r (env legacy)'),
        ('steps', 'mean steps'),
        ('path_length', 'path_length (m)'),
    ]:
        print(f'  {label:45s} {a.get(key, -1):10.4f}')

    # v1.3 reward components
    print(f'  {"--- v1.3 Reward Components ---":45s}')
    for key, label in [
        ('reward_total', 'reward_total'),
        ('reward_base', '  r_base'),
        ('reward_pbrs', '  r_pbrs'),
        ('reward_onset', '  r_onset'),
        ('reward_future_clearance', '  r_future_clearance'),
        ('reward_trap', '  r_trap'),
        ('reward_flip', '  r_flip'),
        ('reward_support', '  r_support'),
        ('reward_guard', '  r_guard'),
    ]:
        print(f'  {label:45s} {a.get(key, -1):10.4f}')

    # Capability metrics (§5.2)
    print(f'  {"--- Capability Metrics (§5.2) ---":45s}')
    for key, label in [
        ('future_clearance_softmin_mean', 'future_clearance_softmin_mean'),
        ('future_clearance_softmin_p10', 'future_clearance_softmin_p10'),
        ('future_clearance_softmin_min', 'future_clearance_softmin_min'),
        ('onset_active_rate', 'onset_active_rate'),
        ('onset_active_steps', 'onset_active_steps (total)'),
        ('trap_active_steps', 'trap_active_steps'),
        ('trap_recovery_rate', 'trap_recovery_rate'),
        ('side_flip_total', 'side_flip_total'),
        ('side_flip_per_step', 'side_flip_per_step'),
        ('loop_detected_rate', 'loop_detected_rate'),
        ('deadend_dwell_steps', 'deadend_dwell_steps'),
        ('low_density_rate', 'low_density_rate'),
        ('support_cell_density_mean', 'support_cell_density_mean'),
        ('free_cone_min_mean', 'free_cone_min_mean'),
        ('free_cone_p20_mean', 'free_cone_p20_mean'),
        ('risk_gate_mean', 'risk_gate_mean'),
    ]:
        print(f'  {label:45s} {a.get(key, -1):10.4f}')


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, default=None, help='PPO checkpoint path')
    p.add_argument('--bc_only', action='store_true')
    p.add_argument('--split', type=str, default='rl_test', help='rl_val | rl_test | layerA')
    args = p.parse_args()
    device = 'cuda:0'

    t_start = time.time()

    # --- BC baseline ---
    bc_ckpt = 'data/outputs/2026.06.17/15.22.50_v9_2_nobs8_bc/checkpoints/latest.ckpt'
    bc_policy = load_policy(bc_ckpt, device)
    bc_results = eval_policy_v13(bc_policy, device, 'BC Baseline', args.split)

    # --- PPO (if provided) ---
    if args.ckpt and not args.bc_only:
        ppo_policy = load_policy(args.ckpt, device)
        ppo_results = eval_policy_v13(ppo_policy, device, f'PPO ({args.ckpt})', args.split)

        # --- Delta ---
        print(f'\n{"="*100}')
        print('Δ (PPO - BC) — Key v1.3 Metrics')
        print(f'{"="*100}')

        role_order = ['anticipatory', 'dense_recovery', 'support_expansion', 'yaw_lateral_recovery', 'unknown']
        delta_keys = [
            'SR', 'CR', 'mean_r_env', 'reward_total',
            'reward_base', 'reward_pbrs', 'reward_onset', 'reward_future_clearance',
            'reward_trap', 'reward_flip', 'reward_support', 'reward_guard',
            'future_clearance_softmin_mean', 'future_clearance_softmin_p10',
            'onset_active_rate', 'trap_recovery_rate',
            'side_flip_per_step', 'loop_detected_rate', 'low_density_rate',
            'path_length', 'deadend_dwell_steps',
        ]

        for role in role_order:
            bc_role = bc_results['per_role'].get(role, {})
            pp_role = ppo_results['per_role'].get(role, {})
            if not bc_role:
                continue
            print(f'\n  {role}:')
            for k in delta_keys:
                bv = bc_role.get(k, float('nan'))
                pv = pp_role.get(k, float('nan'))
                if np.isnan(bv) or np.isnan(pv):
                    continue
                delta = pv - bv
                print(f'    {k:45s}  BC={bv:8.4f}  PPO={pv:8.4f}  Δ={delta:+.4f}')

        # Overall delta
        bc_ov = bc_results['overall']
        pp_ov = ppo_results['overall']
        print(f'\n  OVERALL:')
        for k in delta_keys:
            bv = bc_ov.get(k, float('nan'))
            pv = pp_ov.get(k, float('nan'))
            if np.isnan(bv) or np.isnan(pv):
                continue
            delta = pv - bv
            print(f'    {k:45s}  BC={bv:8.4f}  PPO={pv:8.4f}  Δ={delta:+.4f}')

    elapsed = time.time() - t_start
    print(f'\n[DONE] Total time: {elapsed:.1f}s')


if __name__ == '__main__':
    main()
