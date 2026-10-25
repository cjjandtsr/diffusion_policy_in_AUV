#!/usr/bin/env python3
"""
build_offline_density.py — P0-1: 从 frozen dataset 构建 offline density 并保存。

从 merged_body_12.zarr 读取所有 trajectory 位置和目标位置，
调用 reward_v13.build_offline_density() 构建 cell visit count 直方图，
保存到 JSON 供 Reward v1.3 support term 使用。

用法:
    python tools/build_offline_density.py [--zarr_path <path>] [--output <path>]

输出:
    JSON 文件，含 cell_counts, total, p20/p50 threshold, bin_size
"""

import argparse, json, math, os, sys
from pathlib import Path

import numpy as np
import torch
import zarr

# 确保能找到 diffusion_policy_3d
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from diffusion_policy_3d.env.reward_v13 import build_offline_density, compute_support_cell


def reconstruct_goal_from_state(pos_world, goal_body, sin_yaw, cos_yaw):
    """从 state 的 pos_world(3) + goal_body(3) + sin_yaw + cos_yaw 重建 goal_world。

    goal_body = R(-yaw) · (goal_world - pos_world)
    → goal_world = R(yaw) · goal_body + pos_world
    """
    yaw = math.atan2(sin_yaw, cos_yaw)
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s, 0],
                  [s,  c, 0],
                  [0,  0, 1]], dtype=np.float64)
    return R @ goal_body + pos_world


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zarr_path', type=str,
                        default='/home/cjj/AUV/avoidance_dataset/merged_body_12.zarr')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path (default: project_root/data/offline_density.json)')
    parser.add_argument('--bin_size', type=float, nargs=3, default=[1.0, 0.5, 0.5],
                        help='support_bin (x_parallel, x_perp, z)')
    args = parser.parse_args()

    if args.output is None:
        args.output = str(PROJECT_DIR / 'data' / 'offline_density.json')

    print(f"[build_offline_density] Loading zarr from {args.zarr_path}...")
    z = zarr.open(args.zarr_path, mode='r')
    state = z['data']['state'][:]       # (N, 9)
    episode_ends = z['meta']['episode_ends'][:]  # (988,)
    N = len(state)
    print(f"  Total frames: {N}, Total episodes: {len(episode_ends)}")

    # 为每个 frame 分配对应的 episode goal
    # 从每个 episode 的第一帧 reconstruct goal_world
    print("  Reconstructing per-episode goals...")
    goals = np.zeros((N, 3), dtype=np.float64)

    prev_end = 0
    for ep_idx, end in enumerate(episode_ends):
        start_idx = prev_end
        end_idx = int(end)
        if start_idx >= end_idx:
            prev_end = end_idx
            continue

        # 用 episode 第一帧重建 goal
        s0 = state[start_idx]
        pos_world = s0[:3].copy()
        goal_body = s0[3:6].copy()
        sin_yaw = float(s0[7])
        cos_yaw = float(s0[8])
        goal_w = reconstruct_goal_from_state(pos_world, goal_body, sin_yaw, cos_yaw)

        # 该 episode 所有帧用同一个 goal
        goals[start_idx:end_idx] = goal_w.reshape(1, 3)
        prev_end = end_idx

    positions = state[:, :3].astype(np.float64)
    bin_size = tuple(args.bin_size)

    print(f"  Building density grid (bin_size={bin_size})...")
    density = build_offline_density(positions, goals, bin_size=bin_size)

    # 序列化（cell_counts 的 key 从 tuple 转 str）
    serializable_counts = {str(k): v for k, v in density['cell_counts'].items()}
    density_out = {
        'cell_counts': serializable_counts,
        'total': density['total'],
        'p20_threshold': density['p20_threshold'],
        'p50_threshold': density['p50_threshold'],
        'bin_size': list(density['bin_size']),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(density_out, f, indent=2)

    print(f"\n  ✅ Density saved to {args.output}")
    print(f"  Total cells: {len(density['cell_counts'])}")
    print(f"  p20_threshold: {density['p20_threshold']:.4f}")
    print(f"  p50_threshold: {density['p50_threshold']:.4f}")

    # Quick sanity: check in-support density vs low-density for canonical positions
    canonical_pos = np.array([0.0, 0.0, 2.0])  # mid-point
    canonical_goal = np.array([8.0, 0.0, 2.0])
    cell = compute_support_cell(canonical_pos, canonical_goal, bin_size)
    cell_key = str(cell)
    canon_density = serializable_counts.get(cell_key, 0) / max(density['total'], 1)
    print(f"  Canonical mid-point cell {cell}: density={canon_density:.6f}")

    # Low-density spot (far from path)
    low_pos = np.array([5.0, 5.0, 4.0])
    cell_low = compute_support_cell(low_pos, canonical_goal, bin_size)
    low_key = str(cell_low)
    low_density = serializable_counts.get(low_key, 0) / max(density['total'], 1)
    print(f"  Low-density cell {cell_low}: density={low_density:.6f}")

    if canon_density > low_density:
        print("  ✅ in-support density > low-density density: PASS")
    else:
        print("  ⚠️  in-support density <= low-density density: need review")

    return density_out


if __name__ == '__main__':
    main()
