#!/usr/bin/env python3
"""audit_b0_replay_regate_v13.py — re-apply v1.3 start/goal gates to a B0 replay.

For every episode (unique world_uid+episode_seed) in the replay, take its STORED
start/goal (RouteATransition.start/goal) and re-run the exact env gate chain used
at sampling time: role-aware distance, 3D>=4m, start/goal clearance, A* reachability,
and (dense_recovery) path_stretch in [1.1,1.8]. Reports pass rate overall / per-role
/ seed-vs-collected. This is the definitive "is the replay v1.3-clean" check the
coverage audit did not perform.

Usage: python scripts/audit_b0_replay_regate_v13.py --replay <pt> --manifest <json>
         [--collect_summary <json>] [--out <json>]
"""
import argparse, json, math, sys
from collections import defaultdict, OrderedDict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from diffusion_policy_3d.rl.routea_replay_buffer import RouteAReplayBuffer
from diffusion_policy_3d.env.kinematic_uav_env import KinematicUAVEnv
from diffusion_policy_3d.env.collision_geom import load_obstacles, compute_obstacle_clearance

import os
WORLD_DIR = Path(os.path.expanduser("~/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds"))
FORMAL_ROLES = ["anticipatory", "dense_recovery", "support_expansion"]


def gate_check(env, role, start, goal):
    """Return (ok, reason). Mirrors _sample_start_goal gate chain."""
    d_xy_min, d_xy_max, start_cl, goal_cl = env._get_role_params(role)
    sx, sy, sz = float(start[0]), float(start[1]), float(start[2])
    gx, gy, gz = float(goal[0]), float(goal[1]), float(goal[2])
    d_xy = math.hypot(gx - sx, gy - sy)
    if d_xy < d_xy_min or d_xy > d_xy_max:
        return False, "distance_xy"
    d_3d = math.hypot(d_xy, gz - sz)
    if d_3d < 4.0:
        return False, "distance_3d"
    if compute_obstacle_clearance(sx, sy, sz, env.obstacles)[0] < start_cl:
        return False, "clearance_start"
    if compute_obstacle_clearance(gx, gy, gz, env.obstacles)[0] < goal_cl:
        return False, "clearance_goal"
    occ = env._get_occupancy_grid()
    reachable, path_len, *_ = env._astar_3d(occ, sx, sy, sz, gx, gy, gz)
    if not reachable:
        return False, "astar_unreachable"
    if role == "dense_recovery":
        ps = path_len / max(d_3d, 0.1)
        if ps < 1.1 or ps > 1.8:
            return False, "path_stretch"
    return True, "pass"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--collect_summary", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    replay = RouteAReplayBuffer.load(args.replay)
    trans = replay.get_all_transitions()
    n = len(trans)
    seed_trans = 0
    if args.collect_summary:
        cs = json.load(open(args.collect_summary))
        seed_trans = n - int(cs.get("transitions_collected", n))

    # First transition of each episode (start/goal are episode-level), tag seed by index
    episodes = OrderedDict()  # (world,seed) -> {role,start,goal,group}
    for i, t in enumerate(trans):
        key = (getattr(t, "world_uid", ""), getattr(t, "episode_seed", -1))
        if key in episodes:
            continue
        episodes[key] = {
            "role": getattr(t, "role_active", "") or getattr(t, "role_validated", ""),
            "start": getattr(t, "start", None),
            "goal": getattr(t, "goal", None),
            "group": "seed" if i < seed_trans else "collected",
        }

    # Group episodes by world -> one env per world
    by_world = defaultdict(list)
    for key, ep in episodes.items():
        by_world[key[0]].append((key, ep))

    results = {"pass": 0, "fail": 0, "by_reason": defaultdict(int),
               "by_role": defaultdict(lambda: {"pass": 0, "fail": 0}),
               "by_group": defaultdict(lambda: {"pass": 0, "fail": 0}),
               "seed_episode_details": [], "failures": []}

    for wname, eps in by_world.items():
        obs_json = WORLD_DIR / f"{wname}.world.obstacles.json"
        if not obs_json.exists():
            for key, ep in eps:
                results["fail"] += 1; results["by_reason"]["obstacles_json_missing"] += 1
                results["by_role"][ep["role"]]["fail"] += 1
                results["by_group"][ep["group"]]["fail"] += 1
            continue
        env = KinematicUAVEnv(world_id=4, curriculum_stage="C2", z_max=5.0, seed=42)
        env.obstacles = load_obstacles(str(obs_json)); env.obstacles_json = str(obs_json)
        env.world_id = 0
        if hasattr(env, "_occ_grid_cache"):
            env._occ_grid_cache = None
        for key, ep in eps:
            if ep["start"] is None or ep["goal"] is None:
                ok, reason = False, "missing_start_goal"
            else:
                ok, reason = gate_check(env, ep["role"], ep["start"], ep["goal"])
            results["by_role"][ep["role"]]["pass" if ok else "fail"] += 1
            results["by_group"][ep["group"]]["pass" if ok else "fail"] += 1
            if ok:
                results["pass"] += 1
            else:
                results["fail"] += 1
                results["by_reason"][reason] += 1
                results["failures"].append({"world": wname, "role": ep["role"],
                                            "group": ep["group"], "reason": reason})
            if ep["group"] == "seed":
                results["seed_episode_details"].append(
                    {"world": wname, "role": ep["role"], "pass": ok, "reason": reason})

    total = results["pass"] + results["fail"]
    report = {
        "replay": args.replay,
        "n_episodes": total,
        "seed_transitions": seed_trans,
        "pass_rate": results["pass"] / max(total, 1),
        "pass": results["pass"], "fail": results["fail"],
        "by_reason": dict(results["by_reason"]),
        "by_role": {r: dict(v) for r, v in results["by_role"].items()},
        "by_group": {g: dict(v) for g, v in results["by_group"].items()},
        "seed_episode_details": results["seed_episode_details"],
        "failures_sample": results["failures"][:30],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    out = args.out or str(Path(args.replay).parent / "b0_replay_regate_v13.json")
    json.dump(report, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"\nSaved: {out}")
    seed_pass = all(d["pass"] for d in results["seed_episode_details"])
    print(f"OVERALL_PASS_RATE: {report['pass_rate']:.4f}")
    print(f"SEED_ALL_PASS: {seed_pass}")


if __name__ == "__main__":
    main()
