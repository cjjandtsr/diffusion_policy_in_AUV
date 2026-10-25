#!/usr/bin/env python3
"""Read-only contact probes; run ONLY in the WSL project Python environment.

python -B scripts/diagnose_executed_path_contact.py
Exit 0: listed oracles agree; 1: contact mismatch; 2: setup/runtime error.
Calls real collision code and the existing linear trajectory interpolator.
No env.step/reset, policy, world assets, synthesized observations or file writes.
This isolates contact semantics, NOT dynamics or historical policy behaviour.
"""

import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
import sys
import traceback


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def source_record(module, relative_path):
    path = Path(module.__file__).resolve()
    expected = (ROOT / relative_path).resolve()
    if path != expected:
        raise RuntimeError("Wrong imported source: {} (expected {})".format(path, expected))
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    import numpy as np

    geom = importlib.import_module("diffusion_policy_3d.env.collision_geom")
    kenv = importlib.import_module("diffusion_policy_3d.env.kinematic_uav_env")
    traj = importlib.import_module("diffusion_policy_3d.env.traj_utils")
    sources = {
        name: source_record(module, "diffusion_policy_3d/env/" + filename)
        for name, module, filename in (
            ("geometry", geom, "collision_geom.py"),
            ("environment", kenv, "kinematic_uav_env.py"),
            ("trajectory", traj, "traj_utils.py"),
        )
    }
    if kenv.compute_obstacle_clearance is not geom.compute_obstacle_clearance:
        raise RuntimeError("Environment does not call the imported clearance function")
    threshold = inspect.signature(kenv.KinematicUAVEnv.__init__).parameters[
        "collision_thresh"
    ].default
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
        raise RuntimeError("Cannot resolve the environment's default collision threshold")

    def check(points, obstacles, with_path=True):
        # Only bypass construction/reset; never replace collision or geometry.
        env = object.__new__(kenv.KinematicUAVEnv)
        env.obstacles = obstacles
        env._drone_pos = points[-1].copy()
        env.collision_thresh = float(threshold)
        env.drone_radius = 0.35  # Existing attribute, not a new calibrated radius.
        env._done_reason = "none"
        world_wp = np.zeros((len(points), 11), dtype=np.float32)
        world_wp[:, :3] = points
        hit = bool(env._check_collision(path_wp=world_wp if with_path else None))
        return hit, env._done_reason

    # Literal oracles: a visited center at/below z=0 intersects the ground.
    # We do NOT redefine the existing 0.2m crash threshold or test its equality.
    ground_cases = (
        ("ground_intermediate_below", [1.0, -0.5, 1.0, 1.0], True),
        ("ground_intermediate_surface", [1.0, 0.0, 1.0, 1.0], True),
        ("ground_final_below_control", [1.0, 1.0, 1.0, -0.5], True),
        ("ground_airborne_control", [1.0, 1.0, 1.0, 1.0], False),
    )
    results = []
    for label, heights, expected in ground_cases:
        points = np.array(list(zip((-0.3, -0.2, 0.2, 0.3), [0.0] * 4, heights)),
                          dtype=np.float32)
        hit, reason = check(points, [])
        results.append({
            "case": label, "group": "ground", "expected_collision": expected,
            "actual_collision": hit, "raw_done_reason": reason,
            "matched": hit == expected, "path_world": points.tolist(),
        })

    # Obstacles occupy z=[0,2]. At y=0, x=0 is strictly INSIDE both shapes.
    # Crossing: all four waypoints clear, but segment 1->2 passes through x=0.
    # Controls: an explicit inside waypoint, lateral bypass, genuine flyover.
    segment_cases = (
        ("between_waypoints", [-0.3, -0.2, 0.2, 0.3], 0.0, 1.0, True),
        ("sampled_contact_control", [-0.3, 0.0, 0.2, 0.3], 0.0, 1.0, True),
        ("lateral_clear_control", [-0.3, -0.2, 0.2, 0.3], 1.0, 1.0, False),
        ("flyover_clear_control", [-0.3, -0.2, 0.2, 0.3], 0.0, 2.5, False),
    )
    fixtures = {}
    for shape in ("box", "cylinder"):
        data = {
            "name": "fixture_" + shape, "shape": shape,
            "x": 0.0, "y": 0.0, "z": 1.0, "yaw": 0.0,
            "radius": 0.05, "height": 2.0,
            "size_x": 0.1, "size_y": 0.4, "size_z": 2.0,
        }
        fixtures[shape] = data
        obstacles = [geom.ObstacleSpec.from_dict(data)]
        for label, xs, y, z, expected in segment_cases:
            points = np.array([[x, y, z] for x in xs], dtype=np.float32)
            hit, reason = check(points, obstacles)
            item = {
                "case": shape + "_" + label, "group": "segments",
                "expected_collision": expected, "actual_collision": hit,
                "raw_done_reason": reason, "matched": hit == expected,
                "path_world": points.tolist(),
            }
            if label == "between_waypoints":
                # Evidence witness, not the oracle: sample the REAL interpolator.
                # No start alignment (cur_pos=None) and no trajectory rewrite.
                world_wp = np.zeros((4, 11), dtype=np.float32)
                world_wp[:, :3] = points
                manager = traj.TrajectoryManager(traj_dt=0.3)
                manager.set_trajectory(world_wp, traj_dt=0.3, cur_pos=None, t_start=0.0)
                midpoint = manager.sample(0.45)[0]  # Halfway along segment 1->2.
                witness_hit, witness_reason = check(np.array([midpoint]), obstacles, False)
                item["witness"] = {
                    "interpolated_position": midpoint.tolist(),
                    "point_collision": witness_hit, "raw_done_reason": witness_reason,
                    "waypoint_collisions": [check(np.array([p]), obstacles, False)[0]
                                            for p in points],
                }
            results.append(item)

    failed = [item for item in results if not item["matched"]]
    exit_code = 1 if failed else 0
    report = {
        "schema": "executed_path_contact_diagnostic_v1",
        "sources": sources,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "contract": {
            "scope": "isolated_kinematic_contact_checker",
            "ground_plane_world_z": 0.0,
            "segment_model": "existing TrajectoryManager linear interpolation",
            "collision_thresh_from_env_default": float(threshold),
            "path_shape": [4, 11],
            "final_state": "last waypoint; disturbances disabled in fixtures",
            "oracle": "hand-derived literal contact expectations, not production outputs",
            "excluded": ["env.step integration", "start-to-first-waypoint wiring",
                         "disturbances", "synthetic point cloud", "vehicle dynamics",
                         "historical exploit frequency", "policy capability"],
        },
        "obstacle_fixtures": fixtures,
        "cases": [{k: v for k, v in item.items() if k != "path_world"} for item in results],
        "summary": {"total": len(results), "matched": len(results) - len(failed),
                    "mismatched": len(failed), "failed_cases": [v["case"] for v in failed]},
        "failed_details": failed,
        "exit_code": exit_code,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return exit_code


if __name__ == "__main__":
    try:
        code = main()
    except Exception as exc:
        print(json.dumps({"schema": "executed_path_contact_diagnostic_v1", "exit_code": 2,
                          "error": "{}: {}".format(type(exc).__name__, exc)},
                         ensure_ascii=False, indent=2))
        traceback.print_exc(file=sys.stderr)
        code = 2
    raise SystemExit(code)
