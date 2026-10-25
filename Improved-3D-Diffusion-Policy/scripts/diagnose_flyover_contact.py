#!/usr/bin/env python3
"""Read-only, CPU-only flyover boundary reproducer; no policy or env rollout.

Run from the WSL project: python -B scripts/diagnose_flyover_contact.py
Exit 0: all listed contact cases agree; 1: a contact mismatch; 2: setup/error.
This tests the existing upright 0.1 m half-height proxy, NOT a measured aircraft.
Positive clearance and an optional 0.05 m safety margin are separate concepts.
Nothing is patched, no world assets/checkpoints are loaded, no files are written.
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
    sources = {
        "geometry": source_record(geom, "diffusion_policy_3d/env/collision_geom.py"),
        "environment": source_record(kenv, "diffusion_policy_3d/env/kinematic_uav_env.py"),
    }
    if kenv.compute_obstacle_clearance is not geom.compute_obstacle_clearance:
        raise RuntimeError("Environment does not call the imported clearance function")
    threshold = inspect.signature(kenv.KinematicUAVEnv.__init__).parameters[
        "collision_thresh"
    ].default
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
        raise RuntimeError("Cannot resolve the environment's default collision threshold")

    # Literal, hand-derived oracles, independent of the production distance code.
    # Top fixtures: obstacle top = 3; UAV bottom = center_z - 0.1.
    # Underpass fixtures: suspended obstacle bottom = 3; UAV top = center_z + 0.1.
    # Every fixture is centered at x=y=0, so horizontal overlap is unambiguous.
    # Underpasses stay well above ground; they do NOT permit underground flight.
    boundaries = (
        ("top_overlap_4cm", "top", 3.06, -0.04, True),
        ("top_contact", "top", 3.10, 0.00, True),
        ("top_clear_2cm", "top", 3.12, 0.02, False),
        ("top_clear_6cm", "top", 3.16, 0.06, False),
        ("under_overlap_4cm", "under", 2.94, -0.04, True),
        ("under_contact", "under", 2.90, 0.00, True),
        ("under_clear_2cm", "under", 2.88, 0.02, False),
        ("under_clear_6cm", "under", 2.84, 0.06, False),
    )
    results = []
    for shape in ("cylinder", "box"):
        for label, side, center_z, nominal_gap, expected_hit in boundaries:
            obstacle_data = {
                "name": "fixture_{}_{}".format(shape, side),
                "shape": shape,
                "x": 0.0,
                "y": 0.0,
                "z": 1.5 if side == "top" else 4.0,
                "radius": 0.5,
                "height": 3.0 if side == "top" else 2.0,
                "size_x": 1.0,
                "size_y": 1.0,
                "size_z": 3.0 if side == "top" else 2.0,
                "yaw": 0.0,
            }
            obstacle = geom.ObstacleSpec.from_dict(obstacle_data)
            # Bypass ONLY construction/reset (world I/O, sampling, point clouds).
            # The two production functions below are real and unmodified.
            # These are all attributes read by the current _check_collision.
            env = object.__new__(kenv.KinematicUAVEnv)
            env.obstacles = [obstacle]
            env._drone_pos = np.array([0.0, 0.0, center_z], dtype=np.float32)
            env.collision_thresh = float(threshold)
            env.drone_radius = 0.35  # Existing attribute, not a new physical claim.
            env._done_reason = "none"
            position = [float(value) for value in env._drone_pos]
            clearance, closest, flyover = geom.compute_obstacle_clearance(
                *position, env.obstacles, check_3d=True
            )
            if not math.isfinite(float(clearance)):
                raise RuntimeError("Non-finite clearance in {} {}".format(shape, label))
            actual_hit = bool(env._check_collision(path_wp=None))
            actual_gap = position[2] - 0.1 - 3.0 if side == "top" else 3.0 - (position[2] + 0.1)
            results.append({
                "case": "{}_{}".format(shape, label),
                "obstacle": obstacle_data,
                "position_world": position,
                "nominal_signed_vertical_gap_m": nominal_gap,
                "float32_signed_vertical_gap_m": actual_gap,
                "expected_collision": expected_hit,
                "actual_collision": actual_hit,
                "clearance_returned": float(clearance),
                "closest_obstacle": closest,
                "flyover_returned": bool(flyover),
                "raw_done_reason": env._done_reason,
                "matches_contact_oracle": actual_hit == expected_hit,
            })

    failed = [item["case"] for item in results if not item["matches_contact_oracle"]]
    exit_code = 1 if failed else 0
    report = {
        "schema": "flyover_contact_diagnostic_v1",
        "scope": "isolated_real_geometry_and_environment_collision_functions",
        "not_evidence_for": ["training_exploitation", "full_dynamics", "whole_environment_readiness"],
        "project_root": str(ROOT),
        "python": sys.version,
        "sources": sources,
        "fixture_contract": {
            "upright_half_height_m": 0.1,
            "half_height_basis": "existing collision_geom.py proxy, not independently measured",
            "collision_thresh_from_constructor": float(threshold),
            "position_dtype": "float32 (same as environment state)",
            "contact_is_collision": True,
            "positive_clearance_is_not_contact": True,
            "safety_margin_not_used_as_contact_threshold": True,
        },
        "cases": results,
        "summary": {"total": len(results), "matched": len(results) - len(failed),
                    "mismatched": len(failed), "failed_cases": failed},
        "exit_code": exit_code,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"schema": "flyover_contact_diagnostic_v1", "exit_code": 2,
                          "error": type(error).__name__, "message": str(error)},
                         ensure_ascii=False), file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(2)
