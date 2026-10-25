#!/usr/bin/env python3
"""
build_routea_offline_density.py — Build offline_density.json from frozen EGO dataset

Input:  frozen EGO zarr dataset (merged_body_12.zarr)
Output: diffusion_policy_3d/data/offline_density.json

Per Reward v1.3 design: cell_counts, total, p20_threshold, p50_threshold, bin_size.
Goal extraction: episode final position from state[:, 0:3] at episode_ends indices.
Bin size: from REWARD_V13_DEFAULTS["support_bin"], default (1.0, 0.5, 0.5).

Usage:
  python tools/build_routea_offline_density.py \
    --dataset /home/cjj/AUV/avoidance_dataset/merged_body_12.zarr \
    --output diffusion_policy_3d/data/offline_density.json
"""

import argparse
import json
import os
import sys
import numpy as np
import zarr

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPT_DIR)

from diffusion_policy_3d.env.reward_v13 import (
    build_offline_density,
    REWARD_V13_DEFAULTS,
)


def main():
    parser = argparse.ArgumentParser(description="Build offline density from EGO dataset")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to frozen EGO zarr dataset")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for offline_density.json")
    parser.add_argument("--bin_size", type=str, default=None,
                        help="Bin size as 'x,y,z' (default: from REWARD_V13_DEFAULTS)")
    args = parser.parse_args()

    # Resolve bin_size
    if args.bin_size:
        parts = [float(x.strip()) for x in args.bin_size.split(",")]
        bin_size = tuple(parts)
    else:
        bin_size = tuple(REWARD_V13_DEFAULTS["support_bin"])

    print(f"[build_offline_density] Dataset: {args.dataset}")
    print(f"[build_offline_density] Output: {args.output}")
    print(f"[build_offline_density] Bin size: {bin_size}")

    # ── Open dataset ──
    if not os.path.exists(args.dataset):
        print(f"ERROR: Dataset not found: {args.dataset}")
        sys.exit(1)

    root = zarr.open(args.dataset, mode="r")
    print(f"[build_offline_density] Available keys:")
    for k in root.keys():
        obj = root[k]
        if hasattr(obj, "shape"):
            print(f"  {k}: shape={obj.shape}, dtype={obj.dtype}")

    # ── Extract positions and goals ──
    state = root["data/state"][:]
    episode_ends = root["meta/episode_ends"][:]

    # State columns per EGO dataset convention:
    #   col 0-2: pos_world (x, y, z)
    #   col 3-5: vel_world
    #   col 6: speed
    #   col 7-8: orientation (cos, sin of yaw)
    positions = state[:, 0:3].copy()  # (N, 3)

    # Goal: final position of each episode
    goals_list = []
    prev_end = 0
    for ep_end in episode_ends:
        ep_end = int(ep_end)
        if ep_end > prev_end and ep_end <= len(positions):
            goal_pos = positions[ep_end - 1].copy()  # last state of episode
            goals_list.append(goal_pos)
        prev_end = ep_end

    if not goals_list:
        print("ERROR: No episode goals extracted from dataset")
        print(f"  episode_ends: {episode_ends[:10]}")
        print(f"  state shape: {state.shape}")
        sys.exit(1)

    # Broadcast goals to each timestep in the episode
    goals = np.zeros((len(positions), 3), dtype=np.float32)
    prev_end = 0
    goal_idx = 0
    for ep_end in episode_ends:
        ep_end = int(ep_end)
        if ep_end > prev_end and ep_end <= len(positions):
            goals[prev_end:ep_end] = goals_list[goal_idx]
            goal_idx += 1
        prev_end = ep_end

    print(f"[build_offline_density] Extracted {len(positions)} positions, "
          f"{len(goals_list)} episodes, {goal_idx} goal assignments")
    print(f"[build_offline_density] Goal sample (first 3):")
    for i in range(min(3, len(goals_list))):
        print(f"  ep {i}: goal={goals_list[i]}")

    # ── Build density ──
    density = build_offline_density(positions, goals, bin_size=bin_size)

    n_cells = len(density["cell_counts"])
    print(f"[build_offline_density] Built density: {n_cells} cells, "
          f"total={density['total']}, p20={density['p20_threshold']:.3f}, "
          f"p50={density['p50_threshold']:.3f}")

    if n_cells == 0:
        print("ERROR: cell_counts is empty — density build failed")
        sys.exit(1)
    if density["total"] == 0:
        print("ERROR: total=0 — density build failed")
        sys.exit(1)

    # ── Serialize (cell_counts keys as strings for JSON) ──
    cell_counts_str = {}
    for cell, count in density["cell_counts"].items():
        key = str(cell)  # "(x, y, z)"
        cell_counts_str[key] = count

    output_data = {
        "cell_counts": cell_counts_str,
        "total": density["total"],
        "p20_threshold": density["p20_threshold"],
        "p50_threshold": density["p50_threshold"],
        "bin_size": list(density["bin_size"]),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"[build_offline_density] ✅ Written to {args.output}")
    print(f"[build_offline_density] File size: {os.path.getsize(args.output)} bytes")


if __name__ == "__main__":
    main()
