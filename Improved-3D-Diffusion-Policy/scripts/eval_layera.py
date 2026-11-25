#!/usr/bin/env python3
"""
Evaluate BC baseline vs PPO on layerA OOD worlds (20 worlds from old pipeline).

Usage:
  python scripts/eval_layera.py --ckpt outputs/fm_sde_ppo/ppo_final.pt
  python scripts/eval_layera.py --bc_only
"""
import argparse, json, os, sys, torch, numpy as np
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffusion_policy_3d.env.kinematic_uav_env_v2 import KinematicUAVEnv
from diffusion_policy_3d.model.common.synth_pointcloud import SynthPointCloud

N_EPISODES_PER_WORLD = 4
MAX_STEPS = 200
CURRICULUM = 'C2'
LAYER_A_DIR = os.path.expanduser('~/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds/ood_experiment_worlds/layerA')


def list_layera_worlds():
    """Return sorted list of (uid, json_path) for all layerA worlds."""
    worlds = []
    for f in sorted(os.listdir(LAYER_A_DIR)):
        if f.endswith('.obstacles.json'):
            uid = f.replace('.obstacles.json', '')
            worlds.append((uid, os.path.join(LAYER_A_DIR, f)))
    return worlds


def load_policy(ckpt_path, device):
    """Load policy checkpoint. Auto-detects DDPM vs FM vs PPO."""
    from scripts.train_fm_sde_ppo import load_bc_policy as _load_bc
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Simple detection: FM checkpoints have num_fm_steps key at top level
    is_fm = 'num_fm_steps' in ckpt
    is_ppo = 'velocity_head' in ckpt

    if is_fm or is_ppo:
        if is_ppo:
            bc_path = 'data/outputs/2026.06.17/15.22.50_v9_2_nobs8_bc/checkpoints/latest.ckpt'
            policy = _load_bc(bc_path, device)
            vh_sd = ckpt['velocity_head']
            policy.model.load_state_dict(vh_sd, strict=False)
            print(f'[LOAD] PPO ckpt step={ckpt.get("step","?")}')
        else:
            policy = _load_bc(ckpt_path, device)
            print('[LOAD] FM BC checkpoint')
    else:
        # DDPM Phase1aPolicy
        from scripts.eval_ddpm_bc_baseline import load_ddpm_policy
        policy = load_ddpm_policy(ckpt_path, device)
        print('[LOAD] DDPM BC checkpoint')
    policy.eval()
    return policy


def eval_layera(policy, device, label):
    worlds = list_layera_worlds()
    print(f'\n=== {label} — layerA ({len(worlds)} worlds × {N_EPISODES_PER_WORLD} eps) ===')

    all_results = []
    for idx, (uid, json_path) in enumerate(worlds):
        synth_pc = SynthPointCloud(obstacles_json=json_path)
        env = KinematicUAVEnv(
            world_pool=[(uid, json_path)],
            synth_pc_pool={uid: synth_pc},
            curriculum_stage=CURRICULUM, max_episode_steps=MAX_STEPS)

        ep_results = []
        with torch.no_grad():
            for ep in range(N_EPISODES_PER_WORLD):
                obs_dict, _ = env.reset()
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
                ep_results.append({'r': total_r, 'steps': steps, 'collision': collision, 'success': success})

        mean_r = np.mean([e['r'] for e in ep_results])
        sr = np.mean([e['success'] for e in ep_results])
        cr = np.mean([e['collision'] for e in ep_results])
        all_results.extend(ep_results)
        print(f'  [{idx+1:2d}/{len(worlds)}] {uid:25s} mean_r={mean_r:7.2f} SR={sr:.3f} CR={cr:.3f}')

    overall_r = np.mean([e['r'] for e in all_results])
    overall_sr = np.mean([e['success'] for e in all_results])
    overall_cr = np.mean([e['collision'] for e in all_results])
    print(f'  {"OVERALL":25s} mean_r={overall_r:7.2f} SR={overall_sr:.3f} CR={overall_cr:.3f}')
    return {'mean_r': overall_r, 'SR': overall_sr, 'CR': overall_cr}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, default=None, help='PPO checkpoint path')
    p.add_argument('--bc_only', action='store_true')
    args = p.parse_args()
    device = 'cuda:0'

    bc_ckpt = 'data/outputs/2026.06.17/15.22.50_v9_2_nobs8_bc/checkpoints/latest.ckpt'
    bc_policy = load_policy(bc_ckpt, device)
    bc_results = eval_layera(bc_policy, device, 'BC Baseline')

    if args.ckpt and not args.bc_only:
        ppo_policy = load_policy(args.ckpt, device)
        ppo_results = eval_layera(ppo_policy, device, f'PPO ({args.ckpt})')
        print(f'\n=== Δ (PPO - BC) ===')
        print(f'  Δmean_r = {ppo_results["mean_r"] - bc_results["mean_r"]:+.2f}')
        print(f'  ΔSR     = {ppo_results["SR"] - bc_results["SR"]:+.3f}')
        print(f'  ΔCR     = {ppo_results["CR"] - bc_results["CR"]:+.3f}')


if __name__ == '__main__':
    main()
