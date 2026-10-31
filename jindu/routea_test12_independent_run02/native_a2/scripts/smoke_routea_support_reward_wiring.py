#!/usr/bin/env python3
"""
smoke_routea_support_reward_wiring.py — Prove support reward wiring is connected

Design: Task 2 of Guide Route A RL P0 Wiring Health Fix v1
Output: support_reward_smoke.json with PASS/BLOCKED status

Usage:
  python scripts/smoke_routea_support_reward_wiring.py \
    --output /path/to/support_reward_smoke.json
"""

import argparse
import json
import math
import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPT_DIR)

from diffusion_policy_3d.env.reward_v13 import (
    REWARD_V13_DEFAULTS, create_reward_history, compute_reward_v13,
    load_offline_density, compute_support_cell,
    compute_episode_terminal_support_bonus,
)


def _cell_to_world(cell, goal, bin_size):
    """Convert cell index back to approximate world position (bin center)."""
    xp_bin, xpe_bin, z_bin = cell
    x_parallel = xp_bin * bin_size[0]
    x_perp_signed = xpe_bin * bin_size[1]
    z = z_bin * bin_size[2]

    goal_norm = np.linalg.norm(goal)
    if goal_norm < 1e-6:
        return np.zeros(3)
    goal_unit = goal / goal_norm
    goal_unit_xy = goal_unit[:2]
    goal_unit_xy_norm = np.linalg.norm(goal_unit_xy)
    if goal_unit_xy_norm < 1e-6:
        return np.array([goal[0], goal[1], z])

    # Perp unit in horizontal plane: rotate goal_unit_xy by +90 deg
    perp_unit_xy = np.array([-goal_unit_xy[1], goal_unit_xy[0]])
    perp_unit_xy = perp_unit_xy / goal_unit_xy_norm

    pos_xy = goal[:2] + x_parallel * goal_unit_xy + x_perp_signed * perp_unit_xy
    return np.array([pos_xy[0], pos_xy[1], z], dtype=np.float64)


def _find_low_density_cell_and_position(density, goal, bin_size):
    """Find a low-density cell and a world position that maps to it.

    Uses iterative search: picks low-density cells, computes bin center,
    verifies the mapping, perturbs if needed.
    """
    cell_counts = density["cell_counts"]
    tau_d = density.get("p20_threshold", 2)

    # Sort low-density cells by count (prefer count=0 or 1)
    low_cells = sorted(
        [(c, cnt) for c, cnt in cell_counts.items() if cnt < tau_d],
        key=lambda x: x[1],
    )

    for cell, count in low_cells:
        # Skip cells with unreasonable z
        if cell[2] < 2 or cell[2] > 10:
            continue

        # Start from bin center
        pos = _cell_to_world(cell, goal, bin_size)

        # Verify or search locally
        for dx in np.linspace(-0.4, 0.4, 5):
            for dy in np.linspace(-0.2, 0.2, 3):
                for dz in np.linspace(-0.2, 0.2, 3):
                    test_pos = pos + np.array([dx, dy, dz])
                    test_cell = compute_support_cell(test_pos, goal, bin_size)
                    if test_cell == cell:
                        # Verify it really is low-density in the actual lookup
                        actual_count = cell_counts.get(test_cell, 999)
                        if actual_count < tau_d:
                            return cell, test_pos, actual_count

    return None, None, None


def main():
    parser = argparse.ArgumentParser(description="Support reward wiring smoke test")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for smoke result JSON")
    args = parser.parse_args()

    result = {
        "status": "PENDING",
        "density_loaded": False,
        "candidate_cell": None,
        "cell_count": None,
        "tau_d": None,
        "low_density_cell": False,
        "support_term": 0.0,
        "support_cell_density": 0.0,
        "reward_total": 0.0,
        "terminal_bonus": 0.0,
        "blocked_gate": None,
        "gate_details": {},
        "error": None,
    }

    # ── Step 1: Load density ──
    density = load_offline_density()
    if density is None:
        result["status"] = "BLOCKED_OFFLINE_DENSITY_NOT_FOUND"
        result["error"] = "load_offline_density() returned None"
        _save(result, args.output)
        return

    result["density_loaded"] = True
    result["tau_d"] = density.get("p20_threshold", 0)
    n_cells = len(density.get("cell_counts", {}))
    print(f"[smoke] Density loaded: {n_cells} cells, tau_d={result['tau_d']}")

    # ── Step 2: Find low-density candidate ──
    cfg = dict(REWARD_V13_DEFAULTS)
    goal = np.array([8.0, 0.0, 2.0], dtype=np.float64)
    bin_size = density.get("bin_size", (1.0, 0.5, 0.5))

    cell, pos_tp1, cell_count = _find_low_density_cell_and_position(density, goal, bin_size)

    if cell is None:
        result["status"] = "BLOCKED_NO_LOW_DENSITY_CELL"
        result["error"] = "No low-density cell with valid world position found"
        _save(result, args.output)
        return

    result["candidate_cell"] = list(cell)
    result["cell_count"] = cell_count
    print(f"[smoke] Candidate: cell={cell}, count={cell_count}, pos_tp1={pos_tp1}")

    # Verify cell mapping
    verify_cell = compute_support_cell(pos_tp1, goal, bin_size)
    assert verify_cell == cell, f"Cell mismatch: {verify_cell} != {cell}"
    print(f"[smoke] Cell mapping verified: {cell}")

    # ── Step 3: Construct transition ──
    # pos_t farther from goal than pos_tp1 → positive progress
    goal_dir = goal - pos_tp1
    goal_dir_unit = goal_dir / max(np.linalg.norm(goal_dir), 1e-6)
    pos_t = pos_tp1 - goal_dir_unit * 0.5  # 0.5m behind pos_tp1

    # Ensure pos_t is also farther from goal
    assert np.linalg.norm(pos_t - goal) > np.linalg.norm(pos_tp1 - goal), \
        "pos_t must be farther from goal than pos_tp1"

    # Action chunk: 8 steps moving from pos_t toward pos_tp1
    H = 8
    action_chunk = np.zeros((H, 11), dtype=np.float64)
    step_dpos = (pos_tp1 - pos_t) / H
    for i in range(H):
        action_chunk[i, 0:3] = step_dpos
    action_chunk[:, 10] = 0.0  # yaw_rate = 0

    # States
    s_t = {
        "pos_world": pos_t.astype(np.float64),
        "yaw": 0.0,
        "vel_world": goal_dir_unit.astype(np.float64),
        "z": float(pos_t[2]),
    }
    s_tp1 = {
        "pos_world": pos_tp1.astype(np.float64),
        "yaw": 0.0,
        "vel_world": goal_dir_unit.astype(np.float64),
        "z": float(pos_tp1[2]),
        "collided": False,
    }

    # Empty obstacles → clearance is large
    obstacle_json = []

    # Pre-fill history for gate satisfaction
    history = create_reward_history(
        trap_window=cfg["trap_window"],
        loop_window=cfg["loop_window"],
        n_base=cfg["n_base"],
        flip_episode_floor=cfg["flip_episode_floor"],
    )
    for _ in range(cfg["trap_window"]):
        history["progress"].append(1.0)  # large progress
        history["d_soft_h"].append(100.0)
        history["d_min"].append(100.0)
        history["risk_gate"].append(0.0)
        history["u_lat_abs"].append(0.0)
        history["yaw_rate_abs"].append(0.0)
        history["m_t"].append(0)
    history["total_steps"] = cfg["trap_window"]

    # ── Step 4: Call compute_reward_v13 ──
    reward, terms, metrics, updated_history = compute_reward_v13(
        s_t=s_t,
        action_chunk=action_chunk,
        s_tp1=s_tp1,
        pointcloud=None,
        obstacle_json=obstacle_json,
        goal_world=goal,
        history=history,
        offline_density=density,
        cfg=cfg,
        clearance_fn=None,
    )

    result["reward_total"] = float(reward)
    result["support_term"] = float(terms.get("support", 0.0))
    result["low_density_cell"] = bool(metrics.get("low_density_cell", False))
    result["support_cell_density"] = float(metrics.get("support_cell_density", 0.0))

    print(f"[smoke] reward_total={reward:.6f}, support_term={terms['support']:.6f}")
    print(f"[smoke] low_density_cell={metrics['low_density_cell']}, "
          f"d_soft_h={metrics['d_soft_h']:.3f}, risk_gate={metrics['risk_gate']:.3f}")

    # Gate diagnostics
    gate_details = {
        "d_soft_h": float(metrics["d_soft_h"]),
        "support_safe": cfg["support_safe"],
        "d_soft_h_pass": bool(metrics["d_soft_h"] > cfg["support_safe"]),
        "total_progress_pass": True,
    }
    result["gate_details"] = gate_details

    if result["support_term"] <= 0:
        if metrics["d_soft_h"] <= cfg["support_safe"]:
            result["blocked_gate"] = "d_soft_h"
        elif not result["low_density_cell"]:
            result["blocked_gate"] = "low_density_cell"
        else:
            # Check progress
            ph = list(history["progress"])
            total_progress = sum(ph[-cfg["trap_window"]:]) if len(ph) >= cfg["trap_window"] else 0
            if total_progress <= cfg["support_progress"]:
                result["blocked_gate"] = "total_progress"
                gate_details["total_progress_pass"] = False
                gate_details["total_progress"] = float(total_progress)
            else:
                result["blocked_gate"] = "unknown"

    # ── Step 5: Terminal support bonus ──
    bonus = compute_episode_terminal_support_bonus(updated_history, cfg, reached=True)
    result["terminal_bonus"] = float(bonus)
    print(f"[smoke] Terminal bonus (reached=True): {bonus}")

    bonus_no = compute_episode_terminal_support_bonus(updated_history, cfg, reached=False)
    assert bonus_no == 0.0, "Bonus should be 0 when not reached"
    print(f"[smoke] Terminal bonus (reached=False): {bonus_no} ✓")

    # ── Step 6: Determine result ──
    checks = {
        "density_loaded": result["density_loaded"],
        "candidate_found": result["candidate_cell"] is not None,
        "low_density_cell": result["low_density_cell"],
        "support_term_positive": result["support_term"] > 0,
    }

    all_pass = all(checks.values())
    if all_pass:
        result["status"] = "PASS"
    else:
        failed = [k for k, v in checks.items() if not v]
        result["status"] = "BLOCKED_SUPPORT_REWARD_GATE_TOO_STRICT"
        result["error"] = f"Failed checks: {failed}, blocked_gate={result.get('blocked_gate', '?')}"

    print(f"\n[smoke] Result: {result['status']}")
    for k, v in checks.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")

    _save(result, args.output)


def _save(result, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[smoke] Saved to {path}")


if __name__ == "__main__":
    main()
