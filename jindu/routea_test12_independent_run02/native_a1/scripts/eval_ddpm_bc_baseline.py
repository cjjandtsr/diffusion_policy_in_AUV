#!/usr/bin/env python3
"""
DDPM BC Baseline Evaluation — Phase1aPolicy (DDPM epsilon-pred UNet) on rl_test.

Evaluates DDPM checkpoints (Phase ② λ sweep, v8_phase1a) using fixed episode_configs.
Follows zhidao/02_RL_EVAL_RULES.md: E-01 through E-10.

Usage:
  # Single checkpoint
  python scripts/eval_ddpm_bc_baseline.py --ckpt data/outputs/2026.05.17/13.36.20_v8_phase2_sweep_lambda0.5/checkpoints/latest.ckpt
  # Sweep all Phase ② checkpoints
  python scripts/eval_ddpm_bc_baseline.py --sweep_phase2
  # 05.22 vs 05.17 comparison
  python scripts/eval_ddpm_bc_baseline.py --compare
"""
import argparse, json, os, sys, time, warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from diffusion_policy_3d.policy.phase1a_policy import Phase1aPolicy
from diffusion_policy_3d.env.kinematic_uav_env_v2 import KinematicUAVEnv
from diffusion_policy_3d.model.common.synth_pointcloud import SynthPointCloud

MAX_STEPS = 200
CURRICULUM = 'C2'
SPLIT = 'rl_test'
CONFIG_PATH = _PROJECT_ROOT / 'data' / 'eval_configs' / 'episode_configs_rl_test.json'
WORLDS_DIR = os.path.expanduser('~/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds/test')

# ── Phase ② checkpoints on server ──
PHASE2_CKPTS = {
    'λ=0.05': 'data/outputs/2026.05.17/11.25.34_v8_phase2_pilot_lambda0.05/checkpoints/latest.ckpt',
    'λ=0.3':  'data/outputs/2026.05.17/11.50.36_v8_phase2_sweep_lambda0.3/checkpoints/latest.ckpt',
    'λ=0.5':  'data/outputs/2026.05.17/13.36.20_v8_phase2_sweep_lambda0.5/checkpoints/latest.ckpt',
    'λ=1.0':  'data/outputs/2026.05.17/15.20.34_v8_phase2_sweep_lambda1.0/checkpoints/latest.ckpt',
}
V8_PHASE1A = 'data/outputs/2026.05.22/21.16.19_v8_phase1a/checkpoints/latest.ckpt'


# ═══════════════════════════════════════════════════════════════════
# Wilson 95% CI
# ═══════════════════════════════════════════════════════════════════

def wilson_ci(successes: int, n: int, z: float = 1.96):
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return p, max(0.0, center - margin), min(1.0, center + margin)


# ═══════════════════════════════════════════════════════════════════
# DDPM Policy Loader
# ═══════════════════════════════════════════════════════════════════

def load_ddpm_policy(ckpt_path: str, device: str = "cuda:0") -> Phase1aPolicy:
    """Load Phase1aPolicy (DDPM epsilon-pred UNet) from checkpoint.

    Manually constructs policy + scheduler from config, then loads state_dict.
    Hydra workspace cannot be used because cfg.policy._target_ may point to
    DiffusionPointcloudPolicy (base class) while state_dict has Phase1aPolicy keys.
    """
    ckpt_path = os.path.join(_PROJECT_ROOT, ckpt_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"[LOAD] {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    cfg = ckpt.get('cfg', {})
    policy_cfg = cfg.get('policy', cfg)
    if not policy_cfg:
        raise ValueError("Checkpoint missing 'cfg.policy' key")

    n_obs = ckpt.get('n_obs_steps', policy_cfg.get('n_obs_steps', 3))
    print(f"  n_obs_steps={n_obs}, horizon={policy_cfg.get('horizon')}, "
          f"num_inference_steps={policy_cfg.get('num_inference_steps')}")

    shape_meta = policy_cfg.get('shape_meta')
    if shape_meta is None:
        raise ValueError("Checkpoint missing shape_meta")

    # Reconstruct DDPMScheduler from stored config dict
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    ns_cfg = policy_cfg.get('noise_scheduler')
    if ns_cfg is None:
        raise ValueError("Checkpoint missing noise_scheduler config")
    VALID_SCHEDULER_KEYS = {
        'num_train_timesteps', 'beta_start', 'beta_end', 'beta_schedule',
        'trained_betas', 'variance_type', 'clip_sample', 'prediction_type',
        'thresholding', 'dynamic_thresholding_ratio', 'clip_sample_range',
        'sample_max_value', 'timestep_spacing', 'steps_offset', 'rescale_betas_zero_snr',
    }
    ns_dict = {k: v for k, v in ns_cfg.items()
               if not k.startswith('_') and k in VALID_SCHEDULER_KEYS}
    noise_scheduler = DDPMScheduler(**ns_dict)
    print(f"  scheduler: num_train_timesteps={ns_dict.get('num_train_timesteps')}, "
          f"prediction_type={ns_dict.get('prediction_type')}, "
          f"beta_schedule={ns_dict.get('beta_schedule')}")

    policy = Phase1aPolicy(
        shape_meta=shape_meta,
        noise_scheduler=noise_scheduler,
        horizon=policy_cfg.get('horizon', 16),
        n_action_steps=policy_cfg.get('n_action_steps', 8),
        n_obs_steps=n_obs,
        num_inference_steps=policy_cfg.get('num_inference_steps', 10),
        obs_as_global_cond=policy_cfg.get('obs_as_global_cond', True),
        diffusion_step_embed_dim=policy_cfg.get('diffusion_step_embed_dim', 256),
        down_dims=policy_cfg.get('down_dims', (256, 512, 1024)),
        kernel_size=policy_cfg.get('kernel_size', 5),
        n_groups=policy_cfg.get('n_groups', 8),
        condition_type=policy_cfg.get('condition_type') or 'film',
        use_down_condition=policy_cfg.get('use_down_condition', True),
        use_mid_condition=policy_cfg.get('use_mid_condition', True),
        use_up_condition=policy_cfg.get('use_up_condition', True),
        use_pc_color=policy_cfg.get('use_pc_color', False),
        pointnet_type=policy_cfg.get('pointnet_type', 'pointnet'),
        pointcloud_encoder_cfg=policy_cfg.get('pointcloud_encoder_cfg'),
        point_downsample=policy_cfg.get('point_downsample', False),
        anchor_bank_path=policy_cfg.get('anchor_bank_path'),
        anchor_loss_weight=policy_cfg.get('anchor_loss_weight', 0.5),
        cfg_dropout_prob=policy_cfg.get('cfg_dropout_prob', 0.15),
        cfg_lambda=policy_cfg.get('cfg_lambda', 1.5),
        truncated_steps=policy_cfg.get('truncated_steps', 50),
        truncation_ratio=policy_cfg.get('truncation_ratio', 1.0),
        use_projection=policy_cfg.get('use_projection', True),
        z_min=policy_cfg.get('z_min', 1.0),
        z_max=policy_cfg.get('z_max', 5.0),
        yaw_max_per_step=policy_cfg.get('yaw_max_per_step', 0.524),
        yaw_max_accumulated=policy_cfg.get('yaw_max_accumulated', 3.14159),
        use_obstacle_projection=policy_cfg.get('use_obstacle_projection', False),
        use_collision_loss=policy_cfg.get('use_collision_loss', False),
        training_phase=policy_cfg.get('training_phase', 'joint'),
    )

    state_dicts = ckpt.get('state_dicts', {})
    state_dict = state_dicts.get('model', state_dicts.get('ema_model', {}))
    if not state_dict:
        raise ValueError("Checkpoint missing state_dicts.model")

    missing, unexpected = policy.load_state_dict(state_dict, strict=False)
    real_missing = [k for k in missing if '_dummy_variable' not in k
                    and 'anchor_head' not in k and 'k_stab' not in k
                    and 'lambda_jerk' not in k]
    if real_missing:
        print(f"  [WARN] Missing keys: {len(real_missing)} (first 3: {real_missing[:3]})")

    policy.to(device)
    policy.eval()
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"  Loaded: {n_params:,} params, type={type(policy).__name__}, device={device}")
    return policy


# ═══════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════

def eval_ddpm_policy(policy: Phase1aPolicy, device: str, label: str) -> dict:
    """Evaluate DDPM policy on rl_test using fixed episode_configs.

    Returns dict with per-role metrics and overall summary.
    """
    # Load episode configs
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Episode configs not found: {CONFIG_PATH}")
    with open(CONFIG_PATH) as f:
        configs = json.load(f)

    n_worlds = len(configs)
    total_eps = sum(len(cfgs) for cfgs in configs.values())
    print(f"\n{'='*60}")
    print(f"[EVAL] {label}")
    print(f"[EVAL] Split={SPLIT}, worlds={n_worlds}, episodes={total_eps}")
    print(f"{'='*60}")

    role_results = defaultdict(list)  # role → list of per-ep dicts
    all_episodes = []
    done_count = 0

    for world_uid, world_configs in configs.items():
        # Resolve world path
        json_path = os.path.join(WORLDS_DIR, f'{world_uid}.world.obstacles.json')
        if not os.path.exists(json_path):
            print(f"  [SKIP] Missing obstacles JSON: {json_path}")
            continue

        synth_pc = SynthPointCloud(obstacles_json=json_path)
        env = KinematicUAVEnv(
            world_pool=[(world_uid, json_path)],
            synth_pc_pool={world_uid: synth_pc},
            curriculum_stage=CURRICULUM,
            max_episode_steps=MAX_STEPS,
            reward_version='legacy',  # doesn't matter for BC eval
        )

        for cfg in world_configs:
            with torch.no_grad():
                obs_dict, _ = env.reset(options={'episode_config': cfg})
                done = False
                total_r = 0.0
                steps = 0
                collision = False

                while not done and steps < MAX_STEPS:
                    # Build obs batch
                    obs_b = {}
                    for k, v in obs_dict.items():
                        if isinstance(v, np.ndarray):
                            t = torch.from_numpy(v).to(device)
                        else:
                            t = v.to(device)
                        obs_b[k] = t.unsqueeze(0)

                    # Deterministic DDIM inference (η=0)
                    action_out = policy.predict_action(obs_b)
                    action = action_out['action_pred'] if isinstance(action_out, dict) else action_out
                    action_np = action.squeeze(0).cpu().numpy()

                    obs_dict, reward, terminated, truncated, _ = env.step(action_np)
                    done = terminated or truncated
                    total_r += float(reward)
                    steps += 1

                collision = getattr(env, '_collision_detected', False)
                success = terminated and not collision  # E-03
                episode = {
                    'world': world_uid,
                    'episode_id': cfg.get('episode_id', '?'),
                    'total_r': total_r,
                    'steps': steps,
                    'collision': collision,
                    'success': success,
                }
                all_episodes.append(episode)

                # Infer role from world_uid
                role = 'unknown'
                for r in ['anticipatory', 'dense_recovery', 'support_expansion',
                          'yaw_lateral_recovery', 'canonical', 'default']:
                    if r in world_uid:
                        role = r
                        break
                role_results[role].append(episode)

            done_count += 1
            if done_count % 20 == 0 or done_count == total_eps:
                sr_sofar = np.mean([e['success'] for e in all_episodes])
                cr_sofar = np.mean([e['collision'] for e in all_episodes])
                print(f"  [{done_count}/{total_eps}] SR={sr_sofar:.3f} CR={cr_sofar:.3f}")

    env.close()

    # ── Aggregate ──
    n_total = len(all_episodes)
    n_success = sum(e['success'] for e in all_episodes)
    n_collision = sum(e['collision'] for e in all_episodes)
    p_sr, sr_lo, sr_hi = wilson_ci(n_success, n_total)
    p_cr, cr_lo, cr_hi = wilson_ci(n_collision, n_total)
    mean_r = np.mean([e['total_r'] for e in all_episodes])
    mean_steps = np.mean([e['steps'] for e in all_episodes])

    summary = {
        'label': label,
        'split': SPLIT,
        'n_worlds': n_worlds,
        'n_episodes': n_total,
        'n_success': n_success,
        'n_collision': n_collision,
        'SR': p_sr, 'SR_CI95': (sr_lo, sr_hi),
        'CR': p_cr, 'CR_CI95': (cr_lo, cr_hi),
        'mean_r': mean_r, 'mean_steps': mean_steps,
    }

    # Per-role
    role_summaries = {}
    for role, eps in sorted(role_results.items()):
        n = len(eps)
        s = sum(e['success'] for e in eps)
        c = sum(e['collision'] for e in eps)
        sr, sr_l, sr_h = wilson_ci(s, n)
        cr, cr_l, cr_h = wilson_ci(c, n)
        role_summaries[role] = {
            'n': n, 'SR': sr, 'SR_CI95': (sr_l, sr_h),
            'CR': cr, 'CR_CI95': (cr_l, cr_h),
            'mean_r': np.mean([e['total_r'] for e in eps]),
        }

    return {'summary': summary, 'roles': role_summaries, 'episodes': all_episodes}


# ═══════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════

def print_report(results: dict):
    """Pretty-print evaluation results."""
    s = results['summary']
    print(f"\n{'='*60}")
    print(f"  {s['label']}")
    print(f"  Split={s['split']}, Worlds={s['n_worlds']}, Episodes={s['n_episodes']}")
    print(f"  SR={s['SR']:.4f}  [95% CI: {s['SR_CI95'][0]:.4f}, {s['SR_CI95'][1]:.4f}]")
    print(f"  CR={s['CR']:.4f}  [95% CI: {s['CR_CI95'][0]:.4f}, {s['CR_CI95'][1]:.4f}]")
    print(f"  mean_r={s['mean_r']:.2f}, mean_steps={s['mean_steps']:.1f}")
    print(f"  Success={s['n_success']}/{s['n_episodes']}, Collision={s['n_collision']}/{s['n_episodes']}")

    print(f"\n  Per-Role:")
    header = f"  {'Role':25s} {'N':>4s} {'SR':>8s} {'SR_CI95':>20s} {'CR':>8s} {'CR_CI95':>20s} {'mean_r':>8s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for role, r in sorted(results['roles'].items()):
        print(f"  {role:25s} {r['n']:4d} {r['SR']:8.4f} "
              f"[{r['SR_CI95'][0]:.4f}, {r['SR_CI95'][1]:.4f}]   "
              f"{r['CR']:8.4f} [{r['CR_CI95'][0]:.4f}, {r['CR_CI95'][1]:.4f}]   "
              f"{r['mean_r']:8.2f}")
    print(f"{'='*60}\n")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="DDPM BC Baseline Evaluation")
    p.add_argument('--ckpt', type=str, default=None,
                   help='Single checkpoint path (relative to project root)')
    p.add_argument('--sweep_phase2', action='store_true',
                   help='Evaluate all 4 Phase ② λ sweep checkpoints')
    p.add_argument('--compare', action='store_true',
                   help='Evaluate Phase ② λ=0.5 vs v8_phase1a (05.22)')
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--output', type=str, default=None,
                   help='Save results JSON to this path')
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"[DEVICE] {device}")

    ckpts_to_eval = []

    if args.sweep_phase2:
        ckpts_to_eval = list(PHASE2_CKPTS.items())
    elif args.compare:
        ckpts_to_eval = [('Phase②_λ=0.5', PHASE2_CKPTS['λ=0.5']),
                         ('v8_phase1a_05.22', V8_PHASE1A)]
    elif args.ckpt:
        ckpts_to_eval = [(os.path.basename(args.ckpt), args.ckpt)]
    else:
        print("Specify --ckpt, --sweep_phase2, or --compare")
        sys.exit(1)

    all_results = []
    for label, ckpt_path in ckpts_to_eval:
        t_start = time.time()
        try:
            policy = load_ddpm_policy(ckpt_path, device)
        except FileNotFoundError as e:
            print(f"[SKIP] {label}: {e}")
            continue

        result = eval_ddpm_policy(policy, device, label)
        elapsed = time.time() - t_start
        result['summary']['eval_time_s'] = elapsed
        print_report(result)
        all_results.append(result)

    # ── Comparison table ──
    if len(all_results) >= 2:
        print(f"\n{'='*80}")
        print(f"  COMPARISON")
        print(f"  {'Label':30s} {'SR':>8s} {'SR_CI95':>22s} {'CR':>8s} {'mean_r':>8s} {'Time':>8s}")
        print(f"  {'-'*30} {'-'*8} {'-'*22} {'-'*8} {'-'*8} {'-'*8}")
        for r in all_results:
            s = r['summary']
            print(f"  {s['label']:30s} {s['SR']:8.4f} "
                  f"[{s['SR_CI95'][0]:.4f},{s['SR_CI95'][1]:.4f}]   "
                  f"{s['CR']:8.4f} {s['mean_r']:8.2f} {s.get('eval_time_s',0):7.0f}s")
        print(f"{'='*80}\n")

    # ── Save ──
    if args.output:
        out_path = os.path.join(_PROJECT_ROOT, args.output) if not os.path.isabs(args.output) else args.output
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        # Convert to serializable format
        serializable = []
        for r in all_results:
            sr = {k: v for k, v in r.items() if k != 'episodes'}
            serializable.append(sr)
        with open(out_path, 'w') as f:
            json.dump(serializable, f, indent=2, default=str)
        print(f"[SAVE] {out_path}")


if __name__ == '__main__':
    main()
