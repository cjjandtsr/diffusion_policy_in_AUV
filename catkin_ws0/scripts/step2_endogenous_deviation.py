#!/usr/bin/env python3
"""
step2_endogenous_deviation.py — 步骤2: 内生偏离涌现验证 (设计 §8 步骤2)

目的 (一份 rollout 三处复用):
  ①验证内生偏离存在: 关人为扰动, 完美执行+滚动重规划, 看 e_yaw/dist/高度
    是否非单调 (有波动/漂移) → 证明内生偏离 (完美执行也有偏离, 整个环境前提)
  ②标定扰动幅度: 内生偏离的 P95/P99 → 设 disturb_pos_scale / mid_state_pos_scale
  ③5.5 基准: 记录 FM action 分布 (dpos_norm/dyaw/dz) → 步骤5.5 运动学vs Gazebo对照基准

方法: 多 world 池, 关扰动(disturb_enable=False), 冻结 FM 用标准高斯噪声 (FM默认采样,
      不掺 RL), 跑 N episode, 每 8 步记录偏离量 + action 分布.

用法:
  python step2_endogenous_deviation.py \
    --ckpt data/outputs/2026.06.17/15.22.50_v9_2_nobs8_bc/checkpoints/best.ckpt \
    --n_episodes 50 --device cuda:0
"""

import os, sys, glob, argparse, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffusion_policy_3d.env.kinematic_uav_env import KinematicUAVEnv
from diffusion_policy_3d.model.common.synth_pointcloud import SynthPointCloud
# 复用 train_dsrl_uav.py 的 FM 加载
from train_dsrl_uav import load_frozen_fm
from diffusion_policy_3d.policy.fm_base_policy import FMBasePolicy

TRAIN_DIR = "/root/autodl-tmp/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds/worlds_v8"
NOISE_DIM = 176  # 16 × 11


def scan_world_pool(worlds_dir):
    paths = sorted(glob.glob(os.path.join(worlds_dir, "env_*.obstacles.json")))
    return [(os.path.basename(p).replace(".obstacles.json", ""), p) for p in paths]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--n_episodes', type=int, default=50)
    ap.add_argument('--max_steps', type=int, default=60)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out', default='step2_deviation_stats.json')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    device = args.device

    # 1) 加载冻结 FM
    print('[1/4] 加载冻结 FM...')
    fm_policy = load_frozen_fm(args.ckpt, device)
    base_policy = FMBasePolicy(fm_policy, device)

    # 2) 训练 world 池 + 预加载 synth_pc (0.57GB, 可忽略)
    print('[2/4] 扫描 world 池 + 预加载 synth_pc...')
    train_pool = scan_world_pool(TRAIN_DIR)
    print(f'  训练 world: {len(train_pool)} 个')
    synth_pc_pool = {name: SynthPointCloud(path, device=device)
                     for name, path in train_pool}

    # 3) 环境: 关人为扰动 (测原生内生偏离), 关中间态初始化 (测顺利起点的内生偏离)
    print('[3/4] 构造环境 (扰动关闭)...')
    env = KinematicUAVEnv(
        world_pool=train_pool,
        synth_pc_pool=synth_pc_pool,
        rl_action_steps=8,
        max_episode_steps=args.max_steps,
        h_min=0.3,
        disturb_enable=False,        # ★关人为扰动: 只看内生偏离
        mid_state_init_ratio=0.0,    # ★从顺利起点开始 (内生偏离不靠人造中间态)
        use_stuck_penalty=False, use_escape_reward=False,
        seed=args.seed,
    )

    rng = np.random.default_rng(args.seed)

    # 累积统计
    all_e_yaw = []          # 每步 e_yaw (航向误差)
    all_dist = []           # 每步 ‖pos-goal‖
    all_z_dev = []          # 每步 |z - z_target|
    # 偏离的"步间变化" (内生偏离的核心: 相邻规划段之间的跳变)
    all_d_eyaw = []         # |Δe_yaw| 步间
    all_d_dist = []         # |Δdist| 步间 (注: dist 单调减是正常, 看非单调部分)
    all_d_z = []            # |Δz_dev| 步间
    # action 分布 (5.5 基准)
    all_dpos_norm = []      # ‖dpos‖ per step
    all_dyaw = []
    all_dz = []
    # 每 episode 的偏离轨迹 (验证非单调)
    episode_traces = []

    z_target = env.z_target
    print(f'[4/4] 跑 {args.n_episodes} episode (标准高斯噪声)...')
    for ep in range(args.n_episodes):
        obs_dict, info = env.reset()
        wid = info['world_id']
        trace = {'world': wid, 'e_yaw': [], 'dist': [], 'z_dev': []}
        prev = {'e_yaw': None, 'dist': None, 'z_dev': None}

        for step in range(args.max_steps):
            # 标准高斯噪声 (FM 默认采样, 不掺 RL)
            z = rng.standard_normal((1, NOISE_DIM)).astype(np.float32)
            action = base_policy(obs_dict, z)[0]   # (8, 11)

            # action 分布 (5.5 基准): 每步 dpos/dyaw/dz
            dpos = action[:, 0:3]
            all_dpos_norm.extend(np.linalg.norm(dpos, axis=1).tolist())
            all_dyaw.extend(action[:, 9].tolist())
            all_dz.extend(action[:, 2].tolist())

            obs_dict, reward, term, trunc, info = env.step(action)

            # 偏离量 (从 9D state 读, 或直接算)
            pos = env._drone_pos
            yaw = env._drone_yaw
            gdir = env.goal_w[:2] - pos[:2]
            gn = np.linalg.norm(gdir) + 1e-6
            dh = np.array([np.cos(yaw), np.sin(yaw)])
            e_yaw = float(1.0 - np.dot(dh, gdir / gn))
            dist = float(np.linalg.norm(pos - env.goal_w))
            z_dev = float(abs(pos[2] - z_target))

            all_e_yaw.append(e_yaw); all_dist.append(dist); all_z_dev.append(z_dev)
            trace['e_yaw'].append(e_yaw); trace['dist'].append(dist); trace['z_dev'].append(z_dev)

            # 步间变化 (内生偏离核心)
            if prev['e_yaw'] is not None:
                all_d_eyaw.append(abs(e_yaw - prev['e_yaw']))
                all_d_dist.append(dist - prev['dist'])   # 带符号: >0 = 远离(偏离)
                all_d_z.append(abs(z_dev - prev['z_dev']))
            prev = {'e_yaw': e_yaw, 'dist': dist, 'z_dev': z_dev}

            if term or trunc:
                break
        episode_traces.append(trace)
        if (ep + 1) % 10 == 0:
            print(f'  episode {ep+1}/{args.n_episodes}, world={wid}, '
                  f'len={len(trace["dist"])}, done={info.get("done_reason","trunc")}')

    # ── 统计 ──
    def stats(arr):
        a = np.asarray(arr)
        if len(a) == 0:
            return {}
        return {'mean': float(a.mean()), 'std': float(a.std()),
                'p50': float(np.percentile(a, 50)),
                'p95': float(np.percentile(a, 95)),
                'p99': float(np.percentile(a, 99)),
                'max': float(a.max()), 'n': len(a)}

    # ①内生偏离存在性: 步间变化非零 (尤其 d_dist 有正值 = 远离目标的偏离)
    d_dist_arr = np.asarray(all_d_dist)
    frac_away = float((d_dist_arr > 0).mean()) if len(d_dist_arr) else 0.0  # dist增大的步占比

    result = {
        'n_episodes': args.n_episodes,
        'n_worlds_seen': len({t['world'] for t in episode_traces}),
        # ①内生偏离存在性
        'endogenous_deviation': {
            'frac_steps_dist_increased': frac_away,   # >0 即有"远离目标"的内生偏离
            'd_eyaw_step': stats(all_d_eyaw),         # 航向步间跳变
            'd_dist_step': stats(all_d_dist),
            'd_z_step': stats(all_d_z),               # 高度步间跳变
        },
        # ②扰动幅度标定 (用偏离量的 P95/P99)
        'disturb_calibration': {
            'e_yaw': stats(all_e_yaw),
            'dist': stats(all_dist),
            'z_dev': stats(all_z_dev),
            'note': 'disturb_pos_scale 参考 z_dev/位置偏离 P95 的 1.0-1.5 倍; '
                    'mid_state_pos_scale 同源',
        },
        # ③5.5 FM action 基准
        'fm_action_baseline_kinematic': {
            'dpos_norm': stats(all_dpos_norm),
            'dyaw': stats(all_dyaw),
            'dz': stats(all_dz),
            'note': '步骤5.5: Gazebo 跑同样统计, 逐分量比较, 差异>20% 需调试观测管线',
        },
    }

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # ── 打印判读 ──
    print('\n' + '=' * 60)
    print('步骤2 结果判读')
    print('=' * 60)
    print(f"见过 world: {result['n_worlds_seen']} 个 (应接近 {min(args.n_episodes, len(train_pool))})")
    print(f"\n①内生偏离存在性:")
    print(f"  dist 增大(远离目标)的步占比 = {frac_away:.1%}")
    print(f"    (>0 即证明内生偏离: 完美执行下仍有'飞过头/偏离'需下段纠正)")
    print(f"  航向步间跳变 |Δe_yaw|: mean={stats(all_d_eyaw).get('mean',0):.4f}, "
          f"p95={stats(all_d_eyaw).get('p95',0):.4f}")
    print(f"  高度步间跳变 |Δz_dev|: mean={stats(all_d_z).get('mean',0):.4f}, "
          f"p95={stats(all_d_z).get('p95',0):.4f}")
    judge = "✓ 内生偏离存在" if frac_away > 0.05 else "✗ 偏离过小, 检查滚动机制"
    print(f"  判读: {judge}")
    print(f"\n②扰动幅度标定:")
    zd = stats(all_z_dev)
    print(f"  z_dev P95={zd.get('p95',0):.3f} → disturb_pos_scale ≈ {zd.get('p95',0)*1.2:.3f} (P95×1.2)")
    print(f"\n③5.5 FM action 基准 (运动学):")
    dp = stats(all_dpos_norm)
    print(f"  dpos_norm: mean={dp.get('mean',0):.4f}, p95={dp.get('p95',0):.4f}")
    print(f"\n详细统计 → {args.out}")
    print('=' * 60)


if __name__ == '__main__':
    main()