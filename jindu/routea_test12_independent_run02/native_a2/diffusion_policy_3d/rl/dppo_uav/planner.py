"""A* reachability planner.

Ports the pre-v2 ``kinematic_uav_env.py:558 _astar_3d`` (occupancy-grid A*,
6-tuple return) into the official core. The E040-frozen v2 env has no A*
symbols. Byte-equivalent behavior is anchored by a parity test
(``test_planner_parity_with_pre_v2_env``); legacy quirks are preserved as
equivalence properties:

- occ grid layout is ``grid[y][x][z]``, True = free;
- g_score move cost: horizontal ``res``, vertical ``hypot(res, res)``;
- path_len (reconstructed) counts every step as ``res`` (index-space
  Euclidean step, NOT the g_score cost) — legacy semantics;
- ``start == goal`` -> ``(True, 0.0, 2.0, sz, gz, [start])``;
- found path -> ``(True, path_len, 1.0, sz, gz, path)`` (bottleneck_clr and
  z_min/z_max are legacy constants, not computed);
- world->grid conversion truncates (``int(...)``), not floor.

Wiring: AdmissionChecker.planner gets ``lambda s, g: astar_3d(occ,
meta, *s, *g)``-style adapter at the integration point (the pre-v2 method
takes ``(occ_grid, sx, sy, sz, gx, gy, gz)`` — signature noted in
environment_adapter.py AdmissionChecker docstring).
"""

from dataclasses import dataclass
import heapq
import math
from typing import List, Set, Tuple

import numpy as np

# grid[y][x][z] layout, True = free (pre-v2 convention)
OccGrid = np.ndarray


@dataclass(frozen=True)
class OccGridMeta:
    """Occupancy-grid metadata (mirrors pre-v2 ``_occ_grid_meta``)."""

    x_min: float
    y_min: float
    z_min: float
    res: float
    nx: int
    ny: int
    nz: int


def build_occupancy_grid(
    obstacles: List[object],
    *,
    res: float = 0.5,
    x_min: float = -14.0,
    x_max: float = 14.0,
    y_min: float = -14.0,
    y_max: float = 14.0,
    z_min: float = 1.0,
    z_max: float = 5.0,
) -> Tuple[OccGrid, OccGridMeta]:
    """Build the 3D occupancy grid from cylinder obstacles (verbatim port of
    the pre-v2 kinematic_uav_env.py:497 _build_occupancy_grid).

    The curriculum ranges are expanded exactly like the legacy builder:
    xy +-1.0, z (-0.5, +0.5); cylinder footprints are inflated by the drone
    radius margin 0.15. Byte-equivalence is anchored by the parity test
    (``test_parity_*`` in test_planner_occupancy.py).
    """
    # Expand slightly for safety margin (legacy behavior)
    x_min, x_max = x_min - 1.0, x_max + 1.0
    y_min, y_max = y_min - 1.0, y_max + 1.0
    z_min, z_max = max(z_min - 0.5, 0.0), z_max + 0.5

    nx = max(int((x_max - x_min) / res) + 1, 2)
    ny = max(int((y_max - y_min) / res) + 1, 2)
    nz = max(int((z_max - z_min) / res) + 1, 2)

    # grid[y][x][z] — True = free, False = blocked
    grid = np.ones((ny, nx, nz), dtype=bool)

    for obs in obstacles:
        cx, cy, cz = obs.x, obs.y, obs.z
        r = obs.radius + 0.15  # add drone radius margin
        h = obs.height

        # Bounding box in grid indices
        ix0 = max(0, int((cx - r - x_min) / res))
        ix1 = min(nx - 1, int((cx + r - x_min) / res))
        iy0 = max(0, int((cy - r - y_min) / res))
        iy1 = min(ny - 1, int((cy + r - y_min) / res))
        iz0 = max(0, int((cz - h / 2 - z_min) / res))
        iz1 = min(nz - 1, int((cz + h / 2 - z_min) / res))

        for iy in range(iy0, iy1 + 1):
            for ix in range(ix0, ix1 + 1):
                wx = x_min + (ix + 0.5) * res
                wy = y_min + (iy + 0.5) * res
                d2 = (wx - cx) ** 2 + (wy - cy) ** 2
                if d2 < r ** 2:
                    grid[iy, ix, iz0:iz1 + 1] = False

    meta = OccGridMeta(
        x_min=x_min, y_min=y_min, z_min=z_min,
        res=res, nx=nx, ny=ny, nz=nz,
    )
    return grid, meta


def world_to_grid(meta: OccGridMeta, x: float, y: float, z: float) -> Tuple[int, int, int]:
    """World -> grid indices (truncating int(), like the pre-v2 method)."""
    ix = int((x - meta.x_min) / meta.res)
    iy = int((y - meta.y_min) / meta.res)
    iz = int((z - meta.z_min) / meta.res)
    return ix, iy, iz


def grid_to_world(meta: OccGridMeta, ix: int, iy: int, iz: int) -> Tuple[float, float, float]:
    """Grid indices -> cell-center world coordinates (pre-v2 formula)."""
    wx = meta.x_min + (ix + 0.5) * meta.res
    wy = meta.y_min + (iy + 0.5) * meta.res
    wz = meta.z_min + (iz + 0.5) * meta.res
    return wx, wy, wz


def astar_3d(
    occ_grid: OccGrid,
    meta: OccGridMeta,
    sx: float,
    sy: float,
    sz: float,
    gx: float,
    gy: float,
    gz: float,
) -> Tuple[bool, float, float, float, float, list]:
    """3D A* on the occupancy grid (byte-equivalent port of _astar_3d).

    Returns:
        (reachable, path_len, bottleneck_clr, z_min, z_max, path).
    """
    six, siy, siz = world_to_grid(meta, sx, sy, sz)
    gix, giy, giz = world_to_grid(meta, gx, gy, gz)

    # Clamp to grid bounds
    six = max(0, min(meta.nx - 1, six))
    siy = max(0, min(meta.ny - 1, siy))
    siz = max(0, min(meta.nz - 1, siz))
    gix = max(0, min(meta.nx - 1, gix))
    giy = max(0, min(meta.ny - 1, giy))
    giz = max(0, min(meta.nz - 1, giz))

    if not occ_grid[siy, six, siz] or not occ_grid[giy, gix, giz]:
        return False, 0.0, 0.0, 0.0, 0.0, []

    start = (six, siy, siz)
    goal = (gix, giy, giz)
    if start == goal:
        return True, 0.0, 2.0, sz, gz, [start]

    dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    def _heuristic(a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

    open_set = [(0.0, 0, start)]
    g_score = {start: 0.0}
    parent: dict = {start: None}
    visited: Set[Tuple[int, int, int]] = set()

    max_iter = meta.nx * meta.ny * meta.nz // 5
    iters = 0

    while open_set and iters < max_iter:
        iters += 1
        _, _, current = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            # Reconstruct path
            path: List[Tuple[int, int, int]] = []
            cur = goal
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()

            path_len = 0.0
            for i in range(1, len(path)):
                di = 0.0
                for dim in range(3):
                    di += (path[i][dim] - path[i - 1][dim]) ** 2
                path_len += math.sqrt(di) * meta.res

            return True, path_len, 1.0, sz, gz, path

        ci, cj, ck = current
        for d in dirs:
            ni, nj, nk = ci + d[0], cj + d[1], ck + d[2]
            if not (0 <= ni < meta.nx and 0 <= nj < meta.ny and 0 <= nk < meta.nz):
                continue
            if not occ_grid[nj, ni, nk]:
                continue
            neighbor = (ni, nj, nk)
            move_cost = meta.res if d[2] == 0 else math.hypot(meta.res, meta.res)
            tentative_g = g_score[current] + move_cost
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f_score = tentative_g + _heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, iters, neighbor))
                parent[neighbor] = current

    return False, 0.0, 0.0, 0.0, 0.0, []
