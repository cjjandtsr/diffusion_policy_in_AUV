#!/usr/bin/env python3
"""
calibrate_routea_rl_worlds_kinematic_v1.py
===========================================
Real kinematic baseline calibration for Route A RL worlds.
Reuses _rollout_bc_fm_episode + KinematicUAVEnv + determine_role_active
-- no Gazebo, no reimplementation.

Two-pass logic (design §6.4):
  pass1 = N episodes/world (default 20, --pass1_eps)
  If CI halfwidth > 0.15 or determine_role_active returns "needs_more_eval":
      pass2 = 50 episodes (--pass2_eps)

Per world:
  1. Create KinematicUAVEnv with world_pool=[(world_uid, obstacles_json)]
  2. Run _rollout_bc_fm_episode (frozen BC/FM policy) for N episodes
  3. Compute SR/CR + Wilson 95% CI
  4. Compute ego_sr via A* feasibility (eval_bank.py's faithful replicas)
  5. Feed into determine_role_active → admission decision

Usage:
  python scripts/calibrate_routea_rl_worlds_kinematic_v1.py \
      --world_uids routeA_rl_v1__anticipatory__rl_val__front_offset_obstacle__000002 \
      --pass1_eps 12 --output results.json

Full pool (no subset):
  python scripts/calibrate_routea_rl_worlds_kinematic_v1.py --output results.json
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# ─── Path setup ────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
_CATKIN_STATS = Path("/home/cjj/AUV/catkin_ws0/scripts")
if str(_CATKIN_STATS) not in sys.path:
    sys.path.insert(0, str(_CATKIN_STATS))

# ─── Imports ───────────────────────────────────────────────────────────────────
from diffusion_policy_3d.env.kinematic_uav_env import KinematicUAVEnv
from diffusion_policy_3d.model.common.synth_pointcloud import SynthPointCloud
from diffusion_policy_3d.env.collision_geom import load_obstacles, compute_obstacle_clearance
from routea_rl_calibration_stats import wilson_ci, determine_role_active

# ─── Constants ─────────────────────────────────────────────────────────────────
WORLD_DIR = Path("/home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds")
DEFAULT_CURRICULUM = "C2"

# Role sampling params (replicates KinematicUAVEnv._ROLE_SAMPLING_PARAMS)
ROLE_SAMPLING_PARAMS: Dict[str, Dict] = {
    "anticipatory":         {"d_xy": (10, 24), "start_cl": 0.60, "goal_cl": 2.50},
    "dense_recovery":       {"d_xy": (5, 18),  "start_cl": 0.60, "goal_cl": 1.50},
    "support_expansion":    {"d_xy": (8, 26),  "start_cl": 0.80, "goal_cl": 2.00},
    "yaw_lateral_recovery": {"d_xy": (6, 18),  "start_cl": 0.80, "goal_cl": 2.00},
}


# =══════════════════════════════════════════════════════════════════════════════
# Curriculum range helpers (mirror KinematicUAVEnv)
# =══════════════════════════════════════════════════════════════════════════════

def _curriculum_xy_range(stage: str = "C2") -> Tuple[float, float, float, float]:
    """Return (x_min, x_max, y_min, y_max). Mirrors KinematicUAVEnv."""
    if stage == "C0":
        return (-10.0, 10.0, -6.0, 6.0)
    elif stage == "C1":
        return (-12.0, 12.0, -12.0, 12.0)
    else:  # C2
        return (-14.0, 14.0, -14.0, 14.0)


def _curriculum_z_range(stage: str = "C2", z_max: float = 5.0) -> Tuple[float, float]:
    """Return (z_low, z_high). Mirrors KinematicUAVEnv._curriculum_z_range."""
    if stage == "C0":
        return (1.3, 3.3)
    elif stage == "C1":
        return (1.0, 3.5)
    else:  # C2
        return (1.0, z_max)


# =══════════════════════════════════════════════════════════════════════════════
# Pure functions (testable without GPU)
# =══════════════════════════════════════════════════════════════════════════════

def compute_bc_fm_metrics(successes: int, collisions: int, n: int) -> Dict[str, float]:
    """Compute SR, CR, Wilson 95% CI, halfwidth from binary episode outcomes.

    Args:
        successes: number of episodes where the drone reached the goal.
        collisions: number of episodes where the drone collided.
        n: total number of episodes.

    Returns:
        dict with keys: sr_rate, sr_ci_low, sr_ci_high, cr_rate, sr_ci_halfwidth.
    """
    sr = successes / n if n > 0 else 0.0
    cr = collisions / n if n > 0 else 0.0
    if n > 0:
        ci_low, ci_high = wilson_ci(successes, n)
    else:
        ci_low = ci_high = float("nan")
    ci_hw = (ci_high - ci_low) / 2.0 if not (math.isnan(ci_low) or math.isnan(ci_high)) else 0.0
    return {
        "sr_rate": round(sr, 4),
        "sr_ci_low": round(ci_low, 6),
        "sr_ci_high": round(ci_high, 6),
        "cr_rate": round(cr, 4),
        "sr_ci_halfwidth": round(ci_hw, 6),
    }


def adjudicate_world(
    role_validated: str,
    bc_fm_metrics: Dict[str, float],
    ego_sr: Optional[float] = None,
    semantic_metrics: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], str]:
    """Wrapper around routea_rl_calibration_stats.determine_role_active.

    Args:
        role_validated: the world's current role label.
        bc_fm_metrics: dict from compute_bc_fm_metrics.
        ego_sr: fraction of A*-reachable start/goal pairs (A* feasibility).
        semantic_metrics: optional dict for role-specific semantic fields.

    Returns:
        (role_active, admission_status)
        role_active: assigned role string, or None if rejected / needs_more_eval.
        admission_status: human-readable status string.
    """
    ego_metrics = {"sr_rate": ego_sr} if ego_sr is not None else None
    role_active, status, _beyond = determine_role_active(
        role_validated, bc_fm_metrics, ego_metrics, semantic_metrics or {}
    )
    return role_active, status


def compute_ego_sr(
    obstacles_json: str,
    role: str,
    curriculum_stage: str = "C2",
    n_samples: int = 100,
    seed: int = 42,
) -> float:
    """Compute fraction of A*-reachable start/goal pairs for a world.

    Independent of the env's _sample_start_goal (which already filters by A*).
    Samples start/goal pairs meeting distance + clearance criteria, then checks
    A* reachability via eval_bank.py's faithful replicas of the env's grid + A*.

    Returns:
        float in [0.0, 1.0]: fraction of valid pairs that are A*-reachable.
        Returns 0.0 if no valid pairs could be sampled.
    """
    # Lazy import to avoid circular dependencies
    from scripts.capacity_budget_p0.eval_bank import build_occupancy_grid, astar_3d

    obstacles = load_obstacles(obstacles_json)
    occ_grid, meta = build_occupancy_grid(obstacles, curriculum_stage)

    rp = ROLE_SAMPLING_PARAMS.get(role, ROLE_SAMPLING_PARAMS["anticipatory"])
    d_xy_min, d_xy_max = rp["d_xy"]
    start_cl_min, goal_cl_min = rp["start_cl"], rp["goal_cl"]
    x_min, x_max, y_min, y_max = _curriculum_xy_range(curriculum_stage)
    z_low, z_high = _curriculum_z_range(curriculum_stage)

    rng = np.random.default_rng(seed)
    valid_pairs = 0
    reachable_pairs = 0

    max_attempts = n_samples * 50  # generous; each valid sample may need ~5 attempts
    for _ in range(max_attempts):
        if valid_pairs >= n_samples:
            break

        sx = float(rng.uniform(x_min, x_max))
        sy = float(rng.uniform(y_min, y_max))
        sz = float(rng.uniform(z_low + 0.3, z_high - 0.3))
        gx = float(rng.uniform(x_min, x_max))
        gy = float(rng.uniform(y_min, y_max))
        gz = float(rng.uniform(z_low + 0.5, z_high - 0.5))

        # Distance check (same as env._sample_start_goal)
        d_xy = math.hypot(gx - sx, gy - sy)
        if d_xy < d_xy_min or d_xy > d_xy_max:
            continue
        d_3d = math.hypot(d_xy, gz - sz)
        if d_3d < 4.0:
            continue

        # Clearance check (same as env._sample_start_goal)
        sc, _, _ = compute_obstacle_clearance(sx, sy, sz, obstacles)
        if sc < start_cl_min:
            continue
        gc, _, _ = compute_obstacle_clearance(gx, gy, gz, obstacles)
        if gc < goal_cl_min:
            continue

        valid_pairs += 1
        reachable, _, _, _, _, _ = astar_3d(occ_grid, meta, sx, sy, sz, gx, gy, gz)
        if reachable:
            reachable_pairs += 1

    if valid_pairs == 0:
        return 0.0
    return reachable_pairs / valid_pairs


# =══════════════════════════════════════════════════════════════════════════════
# Calibration logic (GPU required — runs real kinematic rollouts)
# =══════════════════════════════════════════════════════════════════════════════

def _generate_episode_configs(
    wname: str,
    obs_path: str,
    role: str,
    curriculum_stage: str,
    n_configs: int,
    rng_seed: int,
) -> List[Dict]:
    """Generate fixed episode configs for a world, matching real env _sample_start_goal.

    Each config: {start_pos, goal_pos, start_yaw, episode_seed}.
    Uses rejection sampling with role-specific distance/clearance params,
    A* reachability filter, and path_stretch bound for dense_recovery.
    """
    from diffusion_policy_3d.env.collision_geom import load_obstacles, compute_obstacle_clearance
    from scripts.capacity_budget_p0.eval_bank import build_occupancy_grid, astar_3d

    obstacles = load_obstacles(obs_path)
    occ_grid, meta = build_occupancy_grid(obstacles, curriculum_stage)

    rp = ROLE_SAMPLING_PARAMS.get(role, ROLE_SAMPLING_PARAMS["anticipatory"])
    d_xy_min, d_xy_max = rp["d_xy"]
    start_cl_min, goal_cl_min = rp["start_cl"], rp["goal_cl"]
    x_min, x_max, y_min, y_max = _curriculum_xy_range(curriculum_stage)
    z_low, z_high = _curriculum_z_range(curriculum_stage)

    rng = np.random.default_rng(rng_seed)
    configs = []
    for cfg_idx in range(n_configs):
        cfg_seed = int(rng.integers(0, 2**31))
        cfg_rng = np.random.default_rng(cfg_seed)
        found = False
        for _attempt in range(50):
            sx = float(cfg_rng.uniform(x_min, x_max))
            sy = float(cfg_rng.uniform(y_min, y_max))
            sz = float(cfg_rng.uniform(z_low + 0.3, z_high - 0.3))
            gx = float(cfg_rng.uniform(x_min, x_max))
            gy = float(cfg_rng.uniform(y_min, y_max))
            gz = float(cfg_rng.uniform(z_low + 0.5, z_high - 0.5))
            d_xy = math.hypot(gx - sx, gy - sy)
            if d_xy < d_xy_min or d_xy > d_xy_max:
                continue
            d_3d = math.hypot(d_xy, gz - sz)
            if d_3d < 4.0:
                continue
            s_cl, _, _ = compute_obstacle_clearance(sx, sy, sz, obstacles)
            if s_cl < start_cl_min:
                continue
            g_cl, _, _ = compute_obstacle_clearance(gx, gy, gz, obstacles)
            if g_cl < goal_cl_min:
                continue
            # A* reachability (matches real env _sample_start_goal)
            reachable, path_len, _, _, _, _ = astar_3d(
                occ_grid, meta, sx, sy, sz, gx, gy, gz)
            if not reachable:
                continue
            # path_stretch bound for dense_recovery (matches real env)
            if role == "dense_recovery":
                path_stretch = path_len / max(d_3d, 0.1)
                if path_stretch < 1.1 or path_stretch > 1.8:
                    continue
            yaw = math.atan2(gy - sy, gx - sx)
            configs.append({
                "start_pos": [sx, sy, sz],
                "goal_pos": [gx, gy, gz],
                "start_yaw": yaw,
                "episode_seed": cfg_seed,
            })
            found = True
            break
        if not found:
            # Fallback: nominal start/goal
            configs.append({
                "start_pos": [-8.0, 0.0, 2.0],
                "goal_pos": [8.0, 0.0, 2.0],
                "start_yaw": 0.0,
                "episode_seed": cfg_seed,
            })
    return configs


def _make_bc_actor(base_policy: "Phase1aPolicyFM", device: str):
    """Create a zero-init LoRA actor (BC behavior, deterministic eval)."""
    from scripts.train_a2_10k_capability import create_actor_wrapper
    actor = create_actor_wrapper(base_policy, base_policy, device, lora_rank=16)
    actor.eval()
    return actor


def calibrate_world(
    world_entry: dict,
    base_policy: "Phase1aPolicyFM",
    device: str,
    pass1_eps: int = 20,
    pass2_eps: int = 50,
    curriculum_stage: str = "C2",
    rng_seed: int = 42,
) -> Dict[str, Any]:
    """Run kinematic calibration on a single world.

    Uses rollout_episode_a2 with a zero-init LoRA actor (matches the real
    eval harness run_c2_ooc_expanded_eval.py), not _rollout_bc_fm_episode.
    Pre-generates fixed episode configs with rejection-sampled start/goal pairs.

    Args:
        world_entry: dict from world manifest JSON
        base_policy: loaded Phase1aPolicyFM
        device: torch device string
        pass1_eps: episodes for screening pass
        pass2_eps: episodes for full evaluation (if needed)
        curriculum_stage: C0/C1/C2
        rng_seed: base seed for deterministic episodes

    Returns:
        dict with calibration results
    """
    wname = world_entry["world_uid"]
    role_validated = world_entry.get("role_validated") or world_entry["role_generated"]
    split = world_entry.get("split", "rl_val")

    # ─── Locate obstacles JSON ────────────────────────────────────────────
    obs_path = str(WORLD_DIR / f"{wname}.world.obstacles.json")
    if not os.path.isfile(obs_path):
        fallback = str(WORLD_DIR / f"{wname}.obstacles.json")
        if os.path.isfile(fallback):
            obs_path = fallback
        else:
            raise FileNotFoundError(
                f"Cannot find obstacles file for {wname}. "
                f"Tried: {obs_path}, {fallback}"
            )

    # ─── Create zero-init actor ───────────────────────────────────────────
    actor = _make_bc_actor(base_policy, device)

    # ─── Pre-generate episode configs (rejection-sampled start/goal pairs) ─
    total_eps = max(pass1_eps, pass2_eps)
    configs = _generate_episode_configs(
        wname, obs_path, role_validated, curriculum_stage,
        n_configs=total_eps, rng_seed=rng_seed,
    )

    # ─── Lazy import rollout function ─────────────────────────────────────
    from scripts.train_a2_10k_capability import rollout_episode_a2

    # ─── Build SynthPointCloud ONCE per world (cache the expensive ray-tracing) ──
    spc = SynthPointCloud(obs_path, device=device)

    # ─── Helper: run a batch of episodes ──────────────────────────────────
    # Fresh env per episode preserves per-episode state isolation (lambda scoping,
    # _rng seed, obstacles object), while the cached spc in synth_pc_pool avoids
    # the expensive SynthPointCloud ray-tracing rebuild on every episode.
    def _run_episodes(n_eps: int, start_ep: int = 0) -> Tuple[List[int], List[int], List[float]]:
        successes: List[int] = []
        collisions: List[int] = []
        timings: List[float] = []
        for ep_idx in range(start_ep, start_ep + n_eps):
            cfg = configs[ep_idx]
            ep_seed = cfg["episode_seed"]
            t0 = time.time()

            # Fresh env per episode (with cached SynthPointCloud)
            env = KinematicUAVEnv(
                world_id=-1,
                world_pool=[(wname, obs_path)],
                synth_pc_pool={wname: spc},
                rl_action_steps=8,
                h_min=0.3, z_min=1.0, z_max=5.0,
                disturb_enable=False,
                mid_state_init_ratio=0.0,
                seed=ep_seed,
                reward_version="v13",
                curriculum_stage=curriculum_stage,
            )
            # Monkey-patch _sample_start_goal to return fixed pair
            _fixed_start = np.array(cfg["start_pos"], dtype=np.float32)
            _fixed_goal = np.array(cfg["goal_pos"], dtype=np.float32)
            _fixed_yaw = float(cfg["start_yaw"])
            env._sample_start_goal = lambda role=None: (
                _fixed_start.copy(), _fixed_yaw, _fixed_goal.copy()
            )

            _, _, summary = rollout_episode_a2(
                actor, base_policy, env, device,
                episode_seed=ep_seed,
                world_id=-1,
                deterministic=True,
                lazy_teacher=True,
            )
            elapsed = time.time() - t0
            timings.append(elapsed)
            successes.append(int(summary["reached"]))
            collisions.append(int(summary["collided"]))
            env.close()
        return successes, collisions, timings

    # ─── Pass 1: screening ────────────────────────────────────────────────
    print(f"    Pass 1: {pass1_eps} episodes...", end=" ", flush=True)
    successes, collisions, timings = _run_episodes(pass1_eps, start_ep=0)
    n1 = len(successes)
    s1 = sum(successes)
    c1 = sum(collisions)
    metrics = compute_bc_fm_metrics(s1, c1, n1)

    # Compute ego_sr (world property, run once)
    n_ego = max(pass1_eps, 50)
    ego_sr = compute_ego_sr(obs_path, role_validated, curriculum_stage, n_samples=n_ego, seed=rng_seed)
    role_active, status = adjudicate_world(role_validated, metrics, ego_sr)

    needs_pass2 = status == "needs_more_eval"
    print(f"SR={metrics['sr_rate']:.4f}, hw={metrics['sr_ci_halfwidth']:.4f}, "
          f"status={status}, pass2={'YES' if needs_pass2 else 'no'}")

    # ─── Pass 2: full evaluation if needed ────────────────────────────────
    if needs_pass2:
        assert pass2_eps > pass1_eps, \
            f"pass2_eps ({pass2_eps}) must be > pass1_eps ({pass1_eps})"
        extra = pass2_eps - n1
        print(f"    Pass 2: {extra} additional episodes...", end=" ", flush=True)
        s2, c2, t2 = _run_episodes(extra, start_ep=n1)
        successes.extend(s2)
        collisions.extend(c2)
        timings.extend(t2)

        n_final = len(successes)
        s_final = sum(successes)
        c_final = sum(collisions)
        metrics = compute_bc_fm_metrics(s_final, c_final, n_final)
        # Re-evaluate ego_sr with more samples (optional, but consistent)
        ego_sr = compute_ego_sr(obs_path, role_validated, curriculum_stage,
                                 n_samples=max(n_final, 50), seed=rng_seed)
        role_active, status = adjudicate_world(role_validated, metrics, ego_sr)
        print(f"SR={metrics['sr_rate']:.4f}, hw={metrics['sr_ci_halfwidth']:.4f}, "
              f"status={status}")
    else:
        pass2_eps = n1  # mark no pass2
        n_final = n1

    avg_time = float(np.mean(timings)) if timings else 0.0

    # pass2 trigger reason (matches existing manifest schema)
    if needs_pass2:
        pass2_reason = (f"CI half-width {metrics['sr_ci_halfwidth']:.4f} > 0.15; "
                        f"status={status}")
    else:
        pass2_reason = (f"SR {metrics['sr_rate']:.4f} hw={metrics['sr_ci_halfwidth']:.4f} ≤ 0.15")
    # stop reason
    if role_active:
        stop_reason = f"Admitted as {role_active}"
    else:
        stop_reason = status

    return {
        "world_uid": wname,
        "split": split,
        "role_generated": world_entry.get("role_generated", role_validated),
        "role_validated": role_validated,
        "role_active": role_active,
        "admission_status": status,
        "bc_fm_sr": metrics["sr_rate"],
        "bc_fm_cr": metrics["cr_rate"],
        "bc_fm_sr_ci_low": metrics["sr_ci_low"],
        "bc_fm_sr_ci_high": metrics["sr_ci_high"],
        "n_episodes_pass1": n1,
        "n_episodes_final": n_final,
        "ego_sr": round(ego_sr, 4),
        "ego_available": True,
        "beyond_boundary": False,
        "needs_pass2": needs_pass2,
        "pass2_trigger_reason": pass2_reason,
        "stop_reason": stop_reason,
        "avg_episode_time_s": round(avg_time, 3),
    }


# =══════════════════════════════════════════════════════════════════════════════
# Manifest save helper
# =══════════════════════════════════════════════════════════════════════════════

def _save_manifest(args, results: List[Dict], device: str, total_t0: float, out_path: Path):
    """Write admission-manifest-compatible output JSON."""
    total_elapsed = time.time() - total_t0

    # Compute per-role stats
    role_keys = ["anticipatory", "dense_recovery", "support_expansion", "yaw_lateral_recovery"]
    stats = {
        "total": len(results),
        "admitted": sum(1 for r in results if r.get("admission_status", "").startswith("admitted")),
        "rejected": sum(1 for r in results if "error" in r or r.get("admission_status", "").startswith("reject")),
        "needs_more_eval": sum(1 for r in results if r.get("admission_status") == "needs_more_eval"),
        "beyond_boundary": sum(1 for r in results if r.get("beyond_boundary")),
        "stress_only": sum(1 for r in results if r.get("admission_status", "").startswith("stress")),
        "ego_available": any(r.get("ego_available", False) for r in results),
    }
    for rk in role_keys:
        stats[f"{rk}_admitted"] = sum(
            1 for r in results
            if r.get("admission_status", "").startswith("admitted")
            and (r.get("role_active") or r.get("role_validated")) == rk
        )
        stats[f"{rk}_total"] = sum(
            1 for r in results
            if (r.get("role_active") or r.get("role_validated")) == rk
        )

    output_data = {
        "schema_version": "routeA_rl_env_v1",
        "calibration_date": datetime.now().strftime("%Y-%m-%d"),
        "mode": "real_kinematic",
        "stats": stats,
        "config": {
            "world_manifest": args.world_manifest,
            "bc_ckpt": args.bc_ckpt,
            "pass1_eps": args.pass1_eps,
            "pass2_eps": args.pass2_eps,
            "curriculum_stage": args.curriculum_stage,
            "seed": args.seed,
            "device": device,
            "world_uids": args.world_uids,
            "roles": args.roles,
        },
        "results": results,
        "total_time_s": round(total_elapsed, 1),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=2)


# =══════════════════════════════════════════════════════════════════════════════
# CLI
# =══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Kinematic baseline calibration for Route A RL worlds"
    )
    parser.add_argument(
        "--world_manifest", type=str,
        default="/home/cjj/AUV/program/reports/routeA_rl_env_v1/worldgen/"
                "routeA_rl_v1_world_manifest.json",
        help="Path to world manifest JSON",
    )
    parser.add_argument(
        "--world_dir", type=str,
        default=str(WORLD_DIR),
        help="Directory containing .obstacles.json files",
    )
    parser.add_argument(
        "--bc_ckpt", type=str,
        default="/home/cjj/AUV/Improved-3D-Diffusion-Policy-main/"
                "Improved-3D-Diffusion-Policy/data/outputs/"
                "2026.06.17/15.22.50_v9_2_nobs8_bc/checkpoints/latest.ckpt",
        help="BC/FM checkpoint path",
    )
    parser.add_argument(
        "--roles", type=str, nargs="+",
        default=["anticipatory", "dense_recovery", "support_expansion", "yaw_lateral_recovery"],
        help="Only calibrate worlds with these roles",
    )
    parser.add_argument(
        "--world_uids", type=str, nargs="+", default=None,
        help="Subset of world_uids to calibrate (omit for all matching --roles)",
    )
    parser.add_argument(
        "--pass1_eps", type=int, default=20,
        help="Episodes per world for screening pass",
    )
    parser.add_argument(
        "--pass2_eps", type=int, default=50,
        help="Episodes per world if pass2 triggered",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path (default: print only)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--curriculum_stage", type=str, default="C2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip worlds already present in the output file",
    )
    args = parser.parse_args()

    # Override device if GPU unavailable
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    print(f"Device: {device}")

    # ─── Load world manifest ──────────────────────────────────────────────
    if not os.path.isfile(args.world_manifest):
        print(f"[ERROR] World manifest not found: {args.world_manifest}")
        sys.exit(1)
    with open(args.world_manifest) as f:
        manifest = json.load(f)
    worlds = manifest["worlds"]
    print(f"Loaded {len(worlds)} worlds from manifest")

    # Filter by world_uids
    if args.world_uids:
        worlds = [w for w in worlds if w["world_uid"] in args.world_uids]
        found = len(worlds)
        print(f"Filtered by --world_uids: {found}/{len(args.world_uids)} found")
        if found < len(args.world_uids):
            missing = set(args.world_uids) - {w["world_uid"] for w in worlds}
            print(f"  [WARN] World UIDs not found: {missing}")

    # Filter by role
    worlds = [w for w in worlds
              if (w.get("role_validated") or w["role_generated"]) in args.roles]
    print(f"After role filter ({args.roles}): {len(worlds)} worlds")

    if not worlds:
        print("[ERROR] No worlds to calibrate after filtering. Exiting.")
        sys.exit(0)

    # Get unique roles
    unique_roles = set()
    for w in worlds:
        unique_roles.add(w.get("role_validated") or w["role_generated"])
    print(f"Roles in pool: {unique_roles}")

    # ─── Load BC/FM policy ────────────────────────────────────────────────
    if not os.path.isfile(args.bc_ckpt):
        print(f"[ERROR] BC checkpoint not found: {args.bc_ckpt}")
        sys.exit(1)
    from scripts.train_a2_10k_capability import load_v9_2_policy
    print(f"Loading BC/FM checkpoint: {args.bc_ckpt}")
    base_policy = load_v9_2_policy(args.bc_ckpt, device)
    base_policy.eval()
    print(f"  Policy loaded ({sum(p.numel() for p in base_policy.parameters())/1e6:.2f}M params)")

    # ─── Output path ──────────────────────────────────────────────────────
    out_path = Path(args.output) if args.output else None

    # ─── Resume: load existing results ────────────────────────────────────
    existing_uids: set = set()
    if args.resume and out_path and out_path.is_file():
        with open(out_path) as f:
            existing_data = json.load(f)
        existing_results = existing_data.get("results", [])
        existing_uids = {r["world_uid"] for r in existing_results if "error" not in r}
        print(f"Resume mode: loaded {len(existing_results)} existing results, "
              f"{len(existing_uids)} completed worlds will be skipped.")
    elif args.resume and not out_path:
        print("[WARN] --resume requires --output. Ignoring --resume.")

    # ─── Calibrate each world ─────────────────────────────────────────────
    results: List[Dict[str, Any]] = []
    total_t0 = time.time()

    # Carry forward existing results so they appear in the final output
    if existing_uids:
        results = [r for r in existing_data.get("results", [])]

    for wi, w in enumerate(worlds):
        wname = w["world_uid"]
        role = w.get("role_validated") or w["role_generated"]
        split = w.get("split", "?")

        if wname in existing_uids:
            print(f"\n[{wi + 1}/{len(worlds)}] {wname}  [SKIP — already in output]")
            continue

        print(f"\n[{wi + 1}/{len(worlds)}] {wname}")
        print(f"    role={role}, split={split}")
        t0 = time.time()

        try:
            result = calibrate_world(
                w,
                base_policy,
                device,
                pass1_eps=args.pass1_eps,
                pass2_eps=args.pass2_eps,
                curriculum_stage=args.curriculum_stage,
                rng_seed=args.seed,
            )
            result["calibration_time_s"] = round(time.time() - t0, 1)
            results.append(result)

            ci = f"[{result['bc_fm_sr_ci_low']:.4f}, {result['bc_fm_sr_ci_high']:.4f}]"
            p2 = " [P2]" if result["needs_pass2"] else "   "
            print(f"    => n={result['n_episodes_final']:2d} SR={result['bc_fm_sr']:.4f} "
                  f"CR={result['bc_fm_cr']:.4f} CI={ci} "
                  f"ego_sr={result['ego_sr']:.4f} | {result['admission_status']}{p2}")

        except Exception as e:
            print(f"    [ERROR] Calibration failed for {wname}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "world_uid": wname,
                "split": split,
                "role_generated": role,
                "role_validated": role,
                "role_active": None,
                "admission_status": f"error: {e}",
                "bc_fm_sr": 0.0,
                "bc_fm_cr": 0.0,
                "bc_fm_sr_ci_low": 0.0,
                "bc_fm_sr_ci_high": 0.0,
                "n_episodes_pass1": 0,
                "n_episodes_final": 0,
                "ego_sr": 0.0,
                "ego_available": False,
                "beyond_boundary": False,
                "needs_pass2": False,
                "pass2_trigger_reason": "",
                "stop_reason": f"error: {e}",
                "calibration_time_s": round(time.time() - t0, 1),
                "error": str(e),
            })

        # ─── Incremental save after each world ────────────────────────────
        if args.output:
            _save_manifest(args, results, device, total_t0, out_path)

    total_elapsed = time.time() - total_t0
    print(f"\n{'=' * 60}")
    print(f"  CALIBRATION COMPLETE")
    print(f"  Worlds: {len(results)} | Total time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f}min)")
    print(f"{'=' * 60}")

    # ─── Summary table ────────────────────────────────────────────────────
    print(f"\n{'World UID':55s} {'Role':20s} {'n':>3s} {'SR':>7s} {'CR':>7s} "
          f"{'CI':>22s} {'ego':>5s} {'Status':25s}")
    print("-" * 145)
    for r in results:
        wname_short = r["world_uid"]
        if len(wname_short) > 53:
            wname_short = wname_short[:50] + "..."
        ci_str = f"[{r.get('bc_fm_sr_ci_low', 0):.3f},{r.get('bc_fm_sr_ci_high', 1):.3f}]"
        p2 = "P2" if r.get("needs_pass2") else "  "
        print(f"{wname_short:55s} {r.get('role_validated', '?'):20s} "
              f"{r.get('n_episodes_final', 0):3d} "
              f"{r.get('bc_fm_sr', 0):7.4f} {r.get('bc_fm_cr', 0):7.4f} "
              f"{ci_str:>22s} {r.get('ego_sr', 0):5.3f} "
              f"{r.get('admission_status', '?'):25s} {p2}")

    # ─── Final save ───────────────────────────────────────────────────────
    if args.output:
        _save_manifest(args, results, device, total_t0, out_path)

        # Compute and print stats summary
        admitted = sum(1 for r in results if r.get("admission_status", "").startswith("admitted"))
        print(f"\nStats: {admitted}/{len(results)} worlds admitted")


if __name__ == "__main__":
    main()
