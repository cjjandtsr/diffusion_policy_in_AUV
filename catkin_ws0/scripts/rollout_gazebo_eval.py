#!/usr/bin/env python3
"""
rollout_gazebo_eval.py — E1 单因子判定: 单 trial rollout
=========================================================
按 Implementation Guide §3 实现。
每 trial 由外层 shell 完整重启 SITL 后调用本脚本 (n_trials=1),
GazeboUAVEnv 始终走首次 reset (复用外层已起飞的 SITL), 不触发内部重启。

用法 (外层已启 SITL+起飞):
  cd ~/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy
  python ~/AUV/catkin_ws0/scripts/rollout_gazebo_eval.py \
    --ckpt data/outputs/2026.06.17/15.22.50_v9_2_nobs8_bc/checkpoints/best.ckpt \
    --world_id 4 --trial_idx 0 --device cuda:0
"""

import os, sys, json, time, hashlib, argparse
import numpy as np
import torch
import dill

AUV_ROOT = os.path.expanduser("~/AUV")
CATKIN_WS = os.path.join(AUV_ROOT, "catkin_ws0")
FM_DIR = os.path.join(AUV_ROOT,
    "Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy")
SCRIPTS = os.path.join(CATKIN_WS, "scripts")

sys.path.insert(0, FM_DIR)
sys.path.insert(0, SCRIPTS)

from diffusion_policy_3d.env.gazebo_uav_env import GazeboUAVEnv
from diffusion_policy_3d.model.common.synth_pointcloud import SynthPointCloud
from train_dsrl_uav import load_frozen_fm

WORLDS_DIR = os.path.join(AUV_ROOT,
    "PX4-Autopilot/Tools/sitl_gazebo/worlds/ood_experiment_worlds/layerA")
WORLD_ENV_MAP = {4: "env_004_obs19", 8: "env_008_obs20",
                 11: "env_011_obs23", 12: "env_012_obs22", 18: "env_018_obs25"}


def md5(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()


def rollout_one(env, policy, max_steps=60):
    """单 trial: 确定性 FM forward (predict_action, CFG λ=1.5, 不经 DSRL).
    按 §3 代码框架 + per_step_dpos/pos/clr 记录."""
    obs, info = env.reset()
    per_step_dpos = []
    per_step_world_pos = []
    per_step_clr = []
    start_pos = env._drone_pos.copy()

    for t in range(max_steps):
        obs_batched = {
            k: np.expand_dims(v, 0) if isinstance(v, np.ndarray)
               else v.unsqueeze(0)
            for k, v in obs.items()
        }
        result = policy.predict_action(obs_batched)
        action = result['action_pred']  # (1, 8, 11)

        if isinstance(action, torch.Tensor):
            action_np = action[0].cpu().numpy()
        else:
            action_np = action[0]

        dpos_sum = float(np.linalg.norm(action_np[:, 0:3].sum(axis=0)))

        obs, reward, term, trunc, info = env.step(action_np)

        current_pos = env._drone_pos.copy()
        per_step_dpos.append(dpos_sum)
        per_step_world_pos.append(current_pos.tolist())

        try:
            clr = env._estimate_clearance()
        except Exception:
            clr = float('nan')
        per_step_clr.append(clr)

        if term or trunc:
            dist_to_goal = float(np.linalg.norm(current_pos - env.goal_w))
            if dist_to_goal < env.goal_tol_xy:
                done_reason = "reached"
            elif current_pos[2] < 0.2:
                done_reason = "crash_ground"
            elif clr < 0:
                done_reason = f"collision(clr={clr:.3f})"
            else:
                done_reason = f"early(z={current_pos[2]:.2f},clr={clr:.3f})"
            return dict(
                reached=(done_reason == "reached"),
                done_reason=done_reason, n_steps=t + 1,
                final_dist=dist_to_goal, final_clr=clr,
                final_z=float(current_pos[2]),
                per_step_dpos=per_step_dpos,
                per_step_pos=per_step_world_pos,
                per_step_clr=per_step_clr,
                start_pos=start_pos.tolist(),
            )

    clr = float(env._estimate_clearance())
    return dict(
        reached=False, done_reason="timeout", n_steps=max_steps,
        final_dist=float(np.linalg.norm(env._drone_pos - env.goal_w)),
        final_clr=clr, final_z=float(env._drone_pos[2]),
        per_step_dpos=per_step_dpos,
        per_step_pos=per_step_world_pos,
        per_step_clr=per_step_clr,
        start_pos=start_pos.tolist(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--world_id', type=int, default=4)
    parser.add_argument('--trial_idx', type=int, default=0)
    parser.add_argument('--max_steps', type=int, default=60)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out_dir', required=True)
    args = parser.parse_args()

    # ── ckpt ──
    ckpt_full = os.path.join(FM_DIR, args.ckpt) if not os.path.isabs(args.ckpt) else args.ckpt
    assert os.path.exists(ckpt_full), f"ckpt not found: {ckpt_full}"
    ckpt_md5 = md5(ckpt_full)

    # ── FM ──
    policy = load_frozen_fm(ckpt_full, args.device)
    policy.cfg_lambda = 1.5
    policy.eval()

    # ── synth_pc + env ──
    world_name = WORLD_ENV_MAP[args.world_id]
    obstacles_json = os.path.join(WORLDS_DIR, f"{world_name}.obstacles.json")
    synth_pc = SynthPointCloud(obstacles_json, device=args.device)

    env = GazeboUAVEnv(
        world_id=args.world_id,
        synth_pc=synth_pc,
        rl_action_steps=8,
        max_episode_steps=args.max_steps,
    )

    # ── 单 trial ──
    t0 = time.time()
    try:
        result = rollout_one(env, policy, args.max_steps)
    except Exception as e:
        import traceback
        traceback.print_exc()
        result = dict(reached=False, done_reason=f"exception:{e}",
                      n_steps=0, per_step_dpos=[], per_step_pos=[],
                      per_step_clr=[], final_dist=-1, final_clr=-1,
                      final_z=-1, start_pos=[0,0,0])
    elapsed = time.time() - t0

    result['trial'] = args.trial_idx
    result['elapsed_s'] = elapsed
    result['ckpt_md5'] = ckpt_md5

    out_path = os.path.join(args.out_dir,
        f"E1_gazebo_synthpc_w{args.world_id}_t{args.trial_idx}.json")
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[E1] t{args.trial_idx}: done={result['done_reason']}, "
          f"steps={result['n_steps']}, "
          f"dpos={[f'{d:.2f}' for d in result.get('per_step_dpos', [])]}, "
          f"elapsed={elapsed:.1f}s")
    print(f"[E1] saved: {out_path}")


if __name__ == '__main__':
    main()
