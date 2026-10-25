"""P8 (ASSUMPTION-5): occupancy-grid construction port — RED tests.

The adapter's A* reachability check needs the pre-v2
kinematic_uav_env.py:497 _build_occupancy_grid construction (the E040-frozen
v2 env has no A* symbols and no grid builder). This module ports it as a pure
function in planner.py; the parity test anchors byte-equivalence with the
pre-v2 builder, quirks preserved: C2 curriculum ranges expanded +-1.0 (xy) /
(-0.5, +0.5) (z), res=0.5, cylinder footprint inflated by drone radius 0.15,
grid[y][x][z] True=free.
"""

import json

import numpy as np
import pytest

from diffusion_policy_3d.env.collision_geom import load_obstacles
from diffusion_policy_3d.rl.dppo_uav.planner import (
    OccGridMeta,
    build_occupancy_grid,
)


def _write_world(tmp_path, obstacles: list) -> str:
    """Minimal obstacles.json in the real schema (collision_geom.load_obstacles)."""
    path = tmp_path / "test_world.obstacles.json"
    path.write_text(json.dumps({"world_id": "test_world",
                                "obstacles": obstacles}))
    return str(path)


def _pre_v2_env(obstacles_json: str, stage: str = "C2", z_max: float = 5.0):
    """Bypass __init__ (pure method anchor) with the state _build_occupancy_grid
    reads: obstacles, _OCC_RES, curriculum_stage, z_max."""
    from diffusion_policy_3d.env.kinematic_uav_env import KinematicUAVEnv

    env = object.__new__(KinematicUAVEnv)
    env.obstacles = load_obstacles(obstacles_json)
    env._OCC_RES = 0.5
    env.curriculum_stage = stage
    env.z_max = z_max
    return env


def test_parity_empty_world(tmp_path):
    """No obstacles -> all-free grid, identical meta."""
    p = _write_world(tmp_path, [])
    ref_env = _pre_v2_env(p)
    ref_grid = ref_env._build_occupancy_grid()
    ref_meta = dict(ref_env._occ_grid_meta)

    grid, meta = build_occupancy_grid(load_obstacles(p))
    assert np.array_equal(grid, ref_grid)
    assert meta == OccGridMeta(**ref_meta)


def test_parity_two_cylinders(tmp_path):
    """Two cylinders (one box-shaped heights) -> byte-identical grid + meta."""
    p = _write_world(tmp_path, [
        {"name": "c1", "shape": "cylinder", "x": 4.9, "y": 0.8,
         "z": 3.6, "radius": 0.36, "height": 7.2},
        {"name": "c2", "shape": "cylinder", "x": -6.0, "y": 5.0,
         "z": 1.5, "radius": 1.1, "height": 3.0},
    ])
    ref_env = _pre_v2_env(p)
    ref_grid = ref_env._build_occupancy_grid()
    ref_meta = dict(ref_env._occ_grid_meta)

    grid, meta = build_occupancy_grid(load_obstacles(p))
    assert np.array_equal(grid, ref_grid)
    assert meta == OccGridMeta(**ref_meta)


def test_cylinder_blocks_footprint(tmp_path):
    """Cylinder at (4, 0) radius 1.2 height 4 -> blocked cells inside the
    inflated footprint (radius + 0.15 margin), free above its top."""
    p = _write_world(tmp_path, [
        {"name": "wall", "shape": "cylinder", "x": 4.0, "y": 0.0,
         "radius": 1.2, "height": 4.0},
    ])
    grid, meta = build_occupancy_grid(load_obstacles(p))
    # cylinder's own center cell by world->grid conversion
    cx = int((4.0 - meta.x_min) / meta.res)
    cy = int((0.0 - meta.y_min) / meta.res)
    assert not grid[cy, cx, 2]  # inside footprint at low z
    assert grid[cy, cx, 9]  # above the cylinder top (z cell ~7.0+)


def test_meta_values_c2_defaults(tmp_path):
    """C2 defaults: res=0.5, xy range (-14,14)+-1 -> (-15,15), z (1,5)
    -> (0.5, 5.5); nx = 61, ny = 61, nz = 11."""
    p = _write_world(tmp_path, [])
    _, meta = build_occupancy_grid(load_obstacles(p))
    assert meta.res == 0.5
    assert (meta.x_min, meta.y_min, meta.z_min) == (-15.0, -15.0, 0.5)
    assert (meta.nx, meta.ny, meta.nz) == (61, 61, 11)


def test_parity_c1_stage(tmp_path):
    """C1 ranges (xy +-12 -> +-13, z (1.0, 3.5) -> (0.5, 4.0)) match the
    pre-v2 builder when the stage differs."""
    p = _write_world(tmp_path, [
        {"name": "c1", "shape": "cylinder", "x": 3.0, "y": -2.0,
         "radius": 0.5, "height": 2.0},
    ])
    ref_env = _pre_v2_env(p, stage="C1", z_max=3.5)
    ref_grid = ref_env._build_occupancy_grid()
    ref_meta = dict(ref_env._occ_grid_meta)

    grid, meta = build_occupancy_grid(
        load_obstacles(p), x_min=-12.0, x_max=12.0,
        y_min=-12.0, y_max=12.0, z_min=1.0, z_max=3.5,
    )
    assert np.array_equal(grid, ref_grid)
    assert meta == OccGridMeta(**ref_meta)
