#!/usr/bin/env python3
"""BC baseline eval on rl_val worlds (20 worlds x 4 eps)."""
import torch, sys, numpy as np, os
from collections import defaultdict
sys.path.insert(0, '.')

from diffusion_policy_3d.rl.routea_rl_env_adapter_v1 import (
    load_routea_rl_admission_manifest, build_role_buckets)
from diffusion_policy_3d.env.kinematic_uav_env_v2 import KinematicUAVEnv
from diffusion_policy_3d.model.common.synth_pointcloud import SynthPointCloud
from scripts.train_fm_sde_ppo import load_bc_policy

device = 'cuda:0'
policy = load_bc_policy(
    'data/outputs/2026.06.17/15.22.50_v9_2_nobs8_bc/checkpoints/latest.ckpt', device)
policy.eval()

manifest = load_routea_rl_admission_manifest(
    os.path.expanduser('~/AUV/program/reports/routeA_rl_env_v1/calibration/admission_manifest_kinematic_v1.json'))
val_buckets = build_role_buckets(manifest, 'rl_val')
worlds_dir = os.path.expanduser('~/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds')

results = defaultdict(list)
n_total = sum(len(v) for v in val_buckets.values())

for role, entries in val_buckets.items():
    for entry in entries:
        name = entry.world_uid
        json_path = os.path.join(worlds_dir, f'{name}.world.obstacles.json')
        synth_pc = SynthPointCloud(obstacles_json=json_path)
        env = KinematicUAVEnv(
            world_pool=[(name, json_path)],
            synth_pc_pool={name: synth_pc},
            curriculum_stage='C2', max_episode_steps=200)

        ep_rewards = []
        for ep in range(4):
            with torch.no_grad():
                obs_dict, _ = env.reset()
                done, total_r, steps = False, 0.0, 0
                while not done and steps < 200:
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
                ep_rewards.append({'r': total_r, 'steps': steps, 'collision': collision})

        mean_r = np.mean([e['r'] for e in ep_rewards])
        results[role].append({'world': name, 'mean_r': mean_r, 'eps': ep_rewards})
        done_count = sum(1 for v in results.values() for _ in v)
        print(f'[{done_count}/{n_total}] {role:25s} {name[:60]:60s} mean_r={mean_r:7.2f}')

print(f'\n=== BC Baseline (rl_val, {n_total} worlds x 4 eps) ===')
all_eps = []
for role in ['anticipatory', 'dense_recovery', 'support_expansion', 'yaw_lateral_recovery']:
    rs = results.get(role, [])
    if rs:
        eps = [e for w in rs for e in w['eps']]
        all_eps.extend(eps)
        mean_r = np.mean([e['r'] for e in eps])
        sr = np.mean([1.0 if e['r'] > 0 else 0.0 for e in eps])
        cr = np.mean([1.0 if e['collision'] else 0.0 for e in eps])
        print(f'  {role:25s}: mean_r={mean_r:7.2f}, SR={sr:.3f}, CR={cr:.3f}, steps={np.mean([e["steps"] for e in eps]):.1f}')

overall_r = np.mean([e['r'] for e in all_eps])
overall_sr = np.mean([1.0 if e['r'] > 0 else 0.0 for e in all_eps])
overall_cr = np.mean([1.0 if e['collision'] else 0.0 for e in all_eps])
print(f'  {"OVERALL":25s}: mean_r={overall_r:7.2f}, SR={overall_sr:.3f}, CR={overall_cr:.3f}')
