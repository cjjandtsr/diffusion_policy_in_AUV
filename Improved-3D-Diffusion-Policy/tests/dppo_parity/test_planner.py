"""P8 (ASSUMPTION-5): A* reachability planner port — RED tests.

The planner must be byte-equivalent to the pre-v2
kinematic_uav_env.py:558 _astar_3d (the E040-frozen v2 env has no A*
symbols). The parity test anchors the port; behavior quirks of the legacy
implementation (vertical move cost res*sqrt(2) in g_score vs res per step in
path_len, the start==goal (True, 0.0, 2.0, sz, gz, [start]) return, constant
bottleneck_clr=1.0) are preserved verbatim — they are equivalence properties.
"""

import numpy as np
import pytest

from diffusion_policy_3d.rl.dppo_uav.planner import (
    OccGridMeta,
    astar_3d,
    grid_to_world,
    world_to_grid,
)

META = OccGridMeta(x_min=0.0, y_min=0.0, z_min=0.0, res=1.0, nx=10, ny=10, nz=10)


def _free() -> np.ndarray:
    return np.ones((10, 10, 10), dtype=bool)  # grid[y][x][z]


def test_open_grid_reaches_straight_line():
    """(1.5,1.5,1.5) -> (5.5,1.5,1.5): 4 horizontal steps of res=1.0."""
    ok, path_len, clr, zmin, zmax, path = astar_3d(
        _free(), META, 1.5, 1.5, 1.5, 5.5, 1.5, 1.5
    )
    assert ok is True
    assert path_len == pytest.approx(4.0)
    assert path[0] == (1, 1, 1)
    assert path[-1] == (5, 1, 1)
    assert len(path) == 5
    assert clr == 1.0 and zmin == 1.5 and zmax == 1.5  # legacy constants


def test_wall_blocks_reachability():
    """x=3 plane fully blocked -> no 3D route around -> unreachable."""
    occ = _free()
    occ[:, 3, :] = False
    ok, path_len, clr, zmin, zmax, path = astar_3d(
        occ, META, 1.5, 1.5, 1.5, 5.5, 1.5, 1.5
    )
    assert ok is False
    assert path == []
    assert (path_len, clr, zmin, zmax) == (0.0, 0.0, 0.0, 0.0)


def test_start_goal_same_cell_legacy_quirk():
    """start == goal -> (True, 0.0, 2.0, sz, gz, [start]) verbatim."""
    ok, path_len, clr, zmin, zmax, path = astar_3d(
        _free(), META, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5
    )
    assert ok is True
    assert (path_len, clr) == (0.0, 2.0)
    assert (zmin, zmax) == (1.5, 1.5)
    assert path == [(1, 1, 1)]


def test_occupied_start_or_goal_rejected():
    """Occupied start or goal cell -> immediate unreachable."""
    occ = _free()
    occ[1, 1, 1] = False  # start cell (y=1, x=1, z=1)
    ok, *rest = astar_3d(occ, META, 1.5, 1.5, 1.5, 5.5, 1.5, 1.5)
    assert ok is False
    occ = _free()
    occ[1, 5, 1] = False  # goal cell
    ok, *rest = astar_3d(occ, META, 1.5, 1.5, 1.5, 5.5, 1.5, 1.5)
    assert ok is False


def test_out_of_bounds_start_clamped():
    """Start far outside the grid clamps to the border and still works."""
    ok, path_len, clr, zmin, zmax, path = astar_3d(
        _free(), META, -50.0, -50.0, -50.0, 5.5, 1.5, 1.5
    )
    assert ok is True
    assert path[0] == (0, 0, 0)  # clamped to origin cell


def test_vertical_move_path_len():
    """Vertical step costs res=1.0 in path_len (legacy path_len semantics)."""
    ok, path_len, clr, zmin, zmax, path = astar_3d(
        _free(), META, 1.5, 1.5, 1.5, 1.5, 1.5, 5.5
    )
    assert ok is True
    assert path_len == pytest.approx(4.0)
    assert path[-1] == (1, 1, 5)


def test_world_grid_roundtrip():
    """world <-> grid conversions match the pre-v2 formulas."""
    assert world_to_grid(META, 1.5, 2.5, 3.5) == (1, 2, 3)
    assert grid_to_world(META, 1, 2, 3) == (1.5, 2.5, 3.5)


def test_parity_with_pre_v2_env():
    """Byte-equivalence: identical 6-tuples vs kinematic_uav_env._astar_3d
    across reachable/unreachable/same-cell/occupied/out-of-bounds cases."""
    from diffusion_policy_3d.env.kinematic_uav_env import KinematicUAVEnv

    env = object.__new__(KinematicUAVEnv)  # bypass __init__; pure method
    env._occ_grid_meta = {
        "x_min": 0.0, "y_min": 0.0, "z_min": 0.0,
        "res": 1.0, "nx": 10, "ny": 10, "nz": 10,
    }

    cases = [
        # (sx, sy, sz, gx, gy, gz)
        (1.5, 1.5, 1.5, 5.5, 1.5, 1.5),   # straight
        (1.5, 1.5, 1.5, 1.5, 1.5, 5.5),   # vertical
        (1.5, 1.5, 1.5, 1.5, 1.5, 1.5),   # same cell
        (1.5, 1.5, 1.5, 8.5, 7.5, 3.5),   # diagonal
        (1.5, 1.5, 1.5, 9.9, 9.9, 9.9),   # corner
        (-50.0, -50.0, -50.0, 5.5, 1.5, 1.5),  # out-of-bounds start
        (0.3, 0.2, 0.1, 1.5, 1.5, 1.5),   # non-center start
    ]
    for c in cases:
        ref = env._astar_3d(_free(), *c)
        got = astar_3d(_free(), META, *c)
        assert got == ref, f"mismatch for {c}: {got} vs {ref}"

    # blocked start cell
    occ = _free()
    occ[1, 1, 1] = False
    assert astar_3d(occ, META, 1.5, 1.5, 1.5, 5.5, 1.5, 1.5) == \
        env._astar_3d(occ, 1.5, 1.5, 1.5, 5.5, 1.5, 1.5)

    # walled corridor
    occ = _free()
    occ[:, 3, :] = False
    assert astar_3d(occ, META, 1.5, 1.5, 1.5, 5.5, 1.5, 1.5) == \
        env._astar_3d(occ, 1.5, 1.5, 1.5, 5.5, 1.5, 1.5)
