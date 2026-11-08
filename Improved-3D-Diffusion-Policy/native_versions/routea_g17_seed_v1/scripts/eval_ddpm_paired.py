#!/usr/bin/env python3
"""
Paired evaluation for DDPM BC vs DPPO policy on a specified split.

Properly loads LoRA weights (which are NOT in model.state_dict()) from DPPO ckpt.
Uses fixed episode_configs for deterministic pre/post comparison (MC-5).

Usage:
  python scripts/eval_ddpm_paired.py --split rl_train
  python scripts/eval_ddpm_paired.py --split rl_test --ckpt outputs/ddpm_ppo_200ep_v2/ddpm_ppo_final.pt
  python scripts/eval_ddpm_paired.py --split rl_train --bc_only
"""
import argparse, json, os, sys, time, numpy as np
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from diffusion_policy_3d.env.kinematic_uav_env_v2 import KinematicUAVEnv
from diffusion_policy_3d.model.common.synth_pointcloud import SynthPointCloud
from diffusion_policy_3d.policy.lora_adapters import insert_lora
from scripts.eval_ddpm_bc_baseline import load_ddpm_policy

import torch
import torch.nn as nn

MAX_STEPS = 200
CURRICULUM = 'C2'
CONFIG_DIR = _PROJECT_ROOT / 'data' / 'eval_configs'
WORLDS_DIR = os.path.expanduser('~/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds')
LAYER_A_DIR = os.path.join(WORLDS_DIR, 'ood_experiment_worlds', 'layerA')

# Default checkpoints
DEFAULT_BC_CKPT = 'data/outputs/2026.05.17/13.36.20_v8_phase2_sweep_lambda0.5/checkpoints/latest.ckpt'
DEFAULT_DPPO_CKPT = 'outputs/ddpm_ppo_200ep_v2/ddpm_ppo_final.pt'


# ═══════════════════════════════════════════════════════════════════
# Value Head (same as train_ddpm_ppo.py)
# ═══════════════════════════════════════════════════════════════════

class ValueHead(nn.Module):
    """V(s): 192 -> 128 -> 64 -> 1"""
    def __init__(self, input_dim=192, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x): return self.net(x)


# ═══════════════════════════════════════════════════════════════════
# Policy loaders
# ═══════════════════════════════════════════════════════════════════

def load_bc_policy(device: str):
    """Load DDPM BC policy (no LoRA)."""
    print(f"[LOAD] BC: {DEFAULT_BC_CKPT}")
    policy = load_ddpm_policy(DEFAULT_BC_CKPT, device)
    policy.eval()
    return policy


def load_ppo_policy(device: str, ckpt_path: str):
    """Load DPPO PPO policy: BC + LoRA insert + load PPO state_dict.

    Key fix: LoRA modules are standalone (not in model.state_dict()),
    saved separately as 'lora_state_dict'. We must insert LoRA first,
    then load lora_state_dict into the nn.ModuleList.
    """
    print(f"[LOAD] PPO: {ckpt_path}")
    ckpt = torch.load(os.path.join(_PROJECT_ROOT, ckpt_path),
                      map_location=device, weights_only=False)

    # 1. Load base BC policy
    policy = load_ddpm_policy(DEFAULT_BC_CKPT, device)

    # 2. Insert LoRA (same config as training)
    lora_rank = ckpt.get('lora_rank', 4)
    lora_alpha = ckpt.get('lora_alpha', 4)
    lora_info = insert_lora(policy.model, r=lora_rank, alpha=lora_alpha, verbose=False)
    policy.model.requires_grad_(False)
    for p in lora_info["lora_params"]:
        p.requires_grad_(True)

    # 3. Load UNet weights from PPO ckpt (includes frozen UNet)
    #    NOTE: UNet should be identical to BC (frozen), this is a no-op check
    ppo_model_sd = ckpt['model_state_dict']
    missing, unexpected = policy.model.load_state_dict(ppo_model_sd, strict=False)
    real_missing = [k for k in missing if 'anchor' not in k and 'dummy' not in k
                    and 'mask_generator' not in k and 'lora' not in k.lower()]
    if real_missing:
        print(f"  [WARN] UNet missing keys: {len(real_missing)}")

    # 4. ★ Load LoRA weights (the actual trained parameters)
    if 'lora_state_dict' in ckpt and ckpt['lora_state_dict']:
        lora_info["lora_modules"].load_state_dict(ckpt['lora_state_dict'])
        lora_params_count = sum(v.numel() for v in ckpt['lora_state_dict'].values())
        print(f"[LOAD] LoRA state_dict: {len(ckpt['lora_state_dict'])} keys, "
              f"{lora_params_count:,} params")
    else:
        print("[WARN] No lora_state_dict in checkpoint — PPO = BC (no trained weights)")

    step = ckpt.get('step', '?')
    print(f"[LOAD] PPO step={step}, LoRA rank={lora_rank}, alpha={lora_alpha}")

    policy.eval()
    return policy


# ═══════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════

SPLIT_subDIR_map = {
    'rl_train': 'train',
    'rl_val': 'val',
    'rl_test': 'test',
    'layerA': 'ood_experiment_worlds/layerA',
}

def get_json_path(split, world_uid):
    """Resolve world JSON path based on split → subdirectory."""
    subdir = SPLIT_subDIR_map.get(split, 'train')
    if split == 'layerA':
        return os.path.join(WORLDS_DIR, subdir, f'{world_uid}.obstacles.json')
    return os.path.join(WORLDS_DIR, subdir, f'{world_uid}.world.obstacles.json')


def eval_policy(policy, device, label, split):
    """Evaluate policy using fixed episode_configs. Returns per-role and overall metrics."""
    config_path = CONFIG_DIR / f'episode_configs_{split}.json'
    if not config_path.exists():
        raise FileNotFoundError(f'Config not found: {config_path}')
    with open(config_path) as f:
        configs = json.load(f)

    n_worlds = len(configs)
    total_eps = sum(len(cfgs) for cfgs in configs.values())
    print(f'\n=== {label} — {split} ({n_worlds} worlds × {total_eps // max(n_worlds, 1)} eps) ===')

    results = defaultdict(list)
    all_eps = []
    done_count = 0

    for world_uid, world_configs in configs.items():
        json_path = get_json_path(split, world_uid)
        if not os.path.exists(json_path):
            print(f'  [SKIP] Missing JSON: {json_path}')
            continue

        synth_pc = SynthPointCloud(obstacles_json=json_path)
        env = KinematicUAVEnv(
            world_pool=[(world_uid, json_path)],
            synth_pc_pool={world_uid: synth_pc},
            curriculum_stage=CURRICULUM,
            max_episode_steps=MAX_STEPS,
            disturb_enable=False,
            mid_state_init_ratio=0.0)

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
                    total_r += float(reward)
                    steps += 1
                collision = getattr(env, '_collision_detected', False)
                success = terminated and not collision
                ep_results.append({
                    'r': total_r, 'steps': steps,
                    'collision': collision, 'success': success,
                    'episode_id': cfg.get('episode_id', '?'),
                })

        # Infer role from world_uid
        role = 'unknown'
        for r in ['anticipatory', 'dense_recovery', 'support_expansion',
                  'yaw_lateral_recovery', 'canonical', 'default']:
            if r in world_uid:
                role = r
                break
        results[role].append({'world': world_uid, 'eps': ep_results})
        all_eps.extend(ep_results)
        done_count += 1
        mean_r = np.mean([e['r'] for e in ep_results])
        sr = np.mean([e['success'] for e in ep_results])
        print(f'  [{done_count:2d}/{n_worlds}] {role:25s} {world_uid[:50]:50s} '
              f'mean_r={mean_r:7.2f} SR={sr:.3f}')

    # ── Summary ──
    role_order = ['anticipatory', 'dense_recovery', 'support_expansion',
                  'yaw_lateral_recovery', 'canonical', 'default', 'unknown']
    for role in role_order:
        rs = results.get(role, [])
        if not rs:
            continue
        eps = [e for w in rs for e in w['eps']]
        mean_r = np.mean([e['r'] for e in eps])
        sr = np.mean([e['success'] for e in eps])
        cr = np.mean([e['collision'] for e in eps])
        mean_steps = np.mean([e['steps'] for e in eps])
        print(f'  {role:25s}: mean_r={mean_r:7.2f} SR={sr:.3f} CR={cr:.3f} steps={mean_steps:.1f}')

    overall_r = np.mean([e['r'] for e in all_eps])
    overall_sr = np.mean([e['success'] for e in all_eps])
    overall_cr = np.mean([e['collision'] for e in all_eps])
    print(f'  {"OVERALL":25s}: mean_r={overall_r:7.2f} SR={overall_sr:.3f} CR={overall_cr:.3f}')

    per_role = {}
    for role, rs in results.items():
        eps = [e for w in rs for e in w['eps']]
        per_role[role] = {
            'mean_r': np.mean([e['r'] for e in eps]),
            'SR': np.mean([e['success'] for e in eps]),
            'CR': np.mean([e['collision'] for e in eps]),
            'n': len(eps),
        }

    return {
        'per_role': per_role,
        'overall': {'mean_r': overall_r, 'SR': overall_sr, 'CR': overall_cr},
    }


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="DDPM Paired Eval: BC vs DPPO")
    p.add_argument('--split', type=str, default='rl_train',
                   help='rl_train | rl_val | rl_test | layerA')
    p.add_argument('--ckpt', type=str, default=None,
                   help='DPPO checkpoint path (relative to project root)')
    p.add_argument('--bc_only', action='store_true',
                   help='BC baseline only, skip PPO')
    p.add_argument('--device', type=str, default='cuda:0')
    args = p.parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f'[DEVICE] {device}')

    # --- BC baseline ---
    bc_policy = load_bc_policy(device)
    t0 = time.time()
    bc_results = eval_policy(bc_policy, device, 'BC Baseline', args.split)
    bc_elapsed = time.time() - t0

    # --- PPO ---
    if args.ckpt and not args.bc_only:
        ppo_policy = load_ppo_policy(device, args.ckpt)
        t0 = time.time()
        ppo_results = eval_policy(ppo_policy, device, f'PPO ({args.ckpt})', args.split)
        ppo_elapsed = time.time() - t0

        # --- Δ (paired: same configs, different policies) ---
        print(f'\n{"="*70}')
        print(f'  Δ (PPO − BC) [paired, deterministic, {args.split}]')
        print(f'  Eval time: BC={bc_elapsed:.0f}s, PPO={ppo_elapsed:.0f}s')
        print(f'{"="*70}')
        role_order = ['anticipatory', 'dense_recovery', 'support_expansion',
                      'yaw_lateral_recovery', 'canonical', 'default', 'unknown']
        all_bc_sr, all_ppo_sr = [], []
        for role in role_order:
            bc = bc_results['per_role'].get(role)
            pp = ppo_results['per_role'].get(role, bc)
            if bc:
                dsr = pp['SR'] - bc['SR']
                dcr = pp['CR'] - bc['CR']
                dr = pp['mean_r'] - bc['mean_r']
                n = bc.get('n', 0)
                print(f'  {role:25s}: n={n:3d}  ΔSR={dsr:+.4f}  ΔCR={dcr:+.4f}  Δmean_r={dr:+.2f}')
                all_bc_sr.extend([bc['SR']] * n)
                all_ppo_sr.extend([pp['SR']] * n)
        do_sr = ppo_results['overall']['SR'] - bc_results['overall']['SR']
        do_r = ppo_results['overall']['mean_r'] - bc_results['overall']['mean_r']
        do_cr = ppo_results['overall']['CR'] - bc_results['overall']['CR']
        print(f'  {"OVERALL":25s}:        ΔSR={do_sr:+.4f}  ΔCR={do_cr:+.4f}  Δmean_r={do_r:+.2f}')
        print(f'{"="*70}\n')

        # ── Save results ──
        out = {
            'split': args.split,
            'bc_ckpt': DEFAULT_BC_CKPT,
            'ppo_ckpt': args.ckpt,
            'bc': bc_results,
            'ppo': ppo_results,
            'delta': {
                'overall': {'ΔSR': do_sr, 'ΔCR': do_cr, 'Δmean_r': do_r},
                'per_role': {},
            },
        }
        for role in role_order:
            bc = bc_results['per_role'].get(role)
            pp = ppo_results['per_role'].get(role, bc)
            if bc and pp:
                out['delta']['per_role'][role] = {
                    'ΔSR': pp['SR'] - bc['SR'],
                    'ΔCR': pp['CR'] - bc['CR'],
                    'Δmean_r': pp['mean_r'] - bc['mean_r'],
                }

        out_path = _PROJECT_ROOT / 'outputs' / f'eval_ddpm_paired_{args.split}_{time.strftime("%Y%m%d_%H%M%S")}.json'
        os.makedirs(out_path.parent, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(out, f, indent=2, default=str)
        print(f'[SAVE] {out_path}')


if __name__ == '__main__':
    main()
