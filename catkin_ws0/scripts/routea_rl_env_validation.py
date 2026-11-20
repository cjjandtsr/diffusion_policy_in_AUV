#!/usr/bin/env python3
"""
routea_rl_env_validation.py — Route A RL v1 Three-Level World Validation Library
================================================================================
Core validation functions for Route A RL world admission system.

Level 1: Geometry validation — obstacle bounds, surface clearance, size sanity
Level 2: Reachability validation — occupancy grid + 2D BFS + 3D A*
Level 3: Semantic validation — role-specific metrics, relabel detection

All A*/BFS paths are for validation/curriculum statistics only.
Never fed into actor, planner, or training replay.

Relabel rules:
  L1 fail → reject_geometry
  L2 fail → role_active="stress_only", reject_unreachable_or_stress
  L3 pass → role_validated = role_generated, pending_calibration
  L3 semantic match another role → relabel, pending_calibration_relabelled
  L3 fail → reject_semantic
"""

from __future__ import annotations

import json
import math
import heapq
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

ALLOWED_ROLES = {
    "anticipatory", "dense_recovery", "support_expansion", "yaw_lateral_recovery",
}
ALLOWED_SPLITS = {"rl_train", "rl_val", "rl_test", "stress_only", "historical_eval_only"}

# Level 1 defaults
L1_SPAWN_CLEARANCE_MIN = 0.60  # meters surface clearance at spawn
L1_SPAWN_CLEARANCE_RECOVERY = 0.50  # relaxed for recovery roles
L1_GOAL_CLEARANCE_MIN = 0.65  # meters surface clearance at goal
L1_MAX_OBS_HEIGHT = 12.0  # max obstacle height sanity
L1_MIN_FLIGHT_Z = 0.3  # min safe flight altitude

# Level 2 defaults
L2_GRID_RES_XY = 0.20  # meters
L2_GRID_RES_Z = 0.25  # meters
L2_INFLATION_RADIUS = 0.50  # drone body + safety margin
L2_Z_VALID_RANGE = (1.0, 5.0)  # valid flight altitude range
L2_MAX_PATH_STRETCH = 2.0  # max allowed path_length / straight_line_distance

# Level 3 defaults
L3_STRAIGHT_D_SOFT_H_MIN = 2.0  # meters — min soft horizontal clearance on straight line
L3_RISK_WINDOW_STEPS = 10  # discretization steps for risk window
L3_BOTTLENECK_CLEARANCE_MIN = 0.8  # meters — min acceptable bottleneck
L3_RECOVERY_WINDOW_STEPS = 8  # steps for yaw recovery window


# ─────────────────────────────────────────────────────────────────────────────
# Obstacle parsing from .world SDF
# ─────────────────────────────────────────────────────────────────────────────


def parse_world_obstacles(world_path: Path) -> List[Dict[str, Any]]:
    """Parse obstacle models from a Gazebo .world SDF file.

    Reuses pattern from validate_world.py.
    """
    tree = ET.parse(str(world_path))
    root = tree.getroot()
    world_el = root.find("world")
    if world_el is None:
        return []

    obstacles = []
    for model in world_el.findall("model"):
        name = model.get("name", "")
        if name in ("ground_plane",):
            continue

        pose_el = model.find("pose")
        if pose_el is None or pose_el.text is None:
            continue
        pose_parts = pose_el.text.strip().split()
        if len(pose_parts) < 3:
            continue
        x = float(pose_parts[0])
        y = float(pose_parts[1])
        z = float(pose_parts[2])
        yaw = float(pose_parts[5]) if len(pose_parts) > 5 else 0.0

        link = model.find("link")
        if link is None:
            continue
        visual = link.find("visual")
        if visual is None:
            continue
        geom = visual.find("geometry")
        if geom is None:
            continue

        cyl = geom.find("cylinder")
        box = geom.find("box")

        obs: Dict[str, Any] = {"name": name, "x": x, "y": y, "z": z, "yaw": yaw}
        if cyl is not None:
            obs["shape"] = "cylinder"
            obs["radius"] = float(cyl.find("radius").text)
            obs["height"] = float(cyl.find("length").text)
            obs["bounding_r"] = obs["radius"]
            obs["z_min"] = z - obs["height"] / 2.0
            obs["z_max"] = z + obs["height"] / 2.0
        elif box is not None:
            size_parts = box.find("size").text.strip().split()
            obs["shape"] = "box"
            obs["size_x"] = float(size_parts[0])
            obs["size_y"] = float(size_parts[1])
            obs["size_z"] = float(size_parts[2])
            obs["bounding_r"] = math.hypot(float(size_parts[0]), float(size_parts[1])) / 2.0
            obs["z_min"] = z - obs["size_z"] / 2.0
            obs["z_max"] = z + obs["size_z"] / 2.0
        else:
            continue

        obstacles.append(obs)

    return obstacles


# ─────────────────────────────────────────────────────────────────────────────
# Surface distance calculation (reuses debug_clearance.py patterns)
# ─────────────────────────────────────────────────────────────────────────────


def _point_to_oriented_rect_distance(
    px: float, py: float,
    cx: float, cy: float,
    sx: float, sy: float,
    yaw: float,
) -> float:
    """Signed distance from point to oriented rectangle surface in XY plane.

    Positive = outside, negative = inside/penetrating.
    Reused from debug_clearance.py.
    """
    dx, dy = px - cx, py - cy
    cos_y, sin_y = math.cos(-yaw), math.sin(-yaw)
    lx = dx * cos_y - dy * sin_y
    ly = dx * sin_y + dy * cos_y
    hx, hy = sx / 2.0, sy / 2.0
    dx_in = max(abs(lx) - hx, 0.0)
    dy_in = max(abs(ly) - hy, 0.0)
    # Signed: negative if inside
    if abs(lx) <= hx and abs(ly) <= hy:
        # Inside → negative distance = -min(clearance to each edge)
        return -min(hx - abs(lx), hy - abs(ly))
    return math.hypot(dx_in, dy_in)


def surface_distance_to_obstacles(
    point_xyz: Tuple[float, float, float],
    obstacles: List[Dict[str, Any]],
    drone_radius: float = 0.35,
) -> float:
    """Minimum signed surface clearance from point to all obstacles.

    For cylinders: XY distance to surface minus radius, with z overlap check.
    For boxes: oriented rect distance in XY, with z overlap check.
    Returns signed distance: positive = clear, negative = penetrating.
    """
    px, py, pz = point_xyz
    drone_bot = pz - drone_radius
    drone_top = pz + drone_radius

    min_clearance = float("inf")

    for obs in obstacles:
        if obs["shape"] == "cylinder":
            d_surf = math.hypot(px - obs["x"], py - obs["y"]) - obs["radius"]
            obs_top = obs.get("z_max", obs["z"] + obs.get("height", 0) / 2.0)
            obs_bot = obs.get("z_min", obs["z"] - obs.get("height", 0) / 2.0)
        else:  # box
            d_surf = _point_to_oriented_rect_distance(
                px, py, obs["x"], obs["y"],
                obs.get("size_x", 1.0), obs.get("size_y", 1.0),
                obs.get("yaw", 0.0),
            )
            obs_top = obs.get("z_max", obs["z"] + obs.get("size_z", 1.0) / 2.0)
            obs_bot = obs.get("z_min", obs["z"] - obs.get("size_z", 1.0) / 2.0)

        # Z overlap check
        if drone_bot > obs_top or drone_top < obs_bot:
            # No z overlap: vertical separation dominates
            dz_val = min(abs(drone_bot - obs_top), abs(drone_top - obs_bot))
            clearance_3d = math.hypot(max(d_surf, 0.0), dz_val)
        elif d_surf >= 0:
            clearance_3d = d_surf
        else:
            # Penetrating both XY and Z
            clearance_3d = d_surf  # negative

        if clearance_3d < min_clearance:
            min_clearance = clearance_3d

    return min_clearance if min_clearance != float("inf") else 100.0


def surface_distance_xy_only(
    point_xy: Tuple[float, float],
    obstacles: List[Dict[str, Any]],
) -> float:
    """XY-only surface clearance (ignoring z). Faster for 2D checks."""
    px, py = point_xy
    min_clearance = float("inf")

    for obs in obstacles:
        if obs["shape"] == "cylinder":
            d_surf = math.hypot(px - obs["x"], py - obs["y"]) - obs["radius"]
        else:
            d_surf = _point_to_oriented_rect_distance(
                px, py, obs["x"], obs["y"],
                obs.get("size_x", 1.0), obs.get("size_y", 1.0),
                obs.get("yaw", 0.0),
            )
        if d_surf < min_clearance:
            min_clearance = d_surf

    return min_clearance if min_clearance != float("inf") else 100.0


# ─────────────────────────────────────────────────────────────────────────────
# Pairwise obstacle surface distance (OBB-aware, not bounding circles)
# ─────────────────────────────────────────────────────────────────────────────


def _box_corners_xy(
    cx: float, cy: float, sx: float, sy: float, yaw: float,
) -> List[Tuple[float, float]]:
    """Return the 4 XY corners of an oriented box."""
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    hx, hy = sx / 2.0, sy / 2.0
    corners = []
    for dx, dy in [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]:
        wx = cx + dx * cos_y - dy * sin_y
        wy = cy + dx * sin_y + dy * cos_y
        corners.append((wx, wy))
    return corners


def _obb_pairwise_surface_distance(
    a: Dict[str, Any], b: Dict[str, Any],
) -> float:
    """Compute minimum surface-to-surface distance between two obstacles in XY.

    Uses proper oriented-box geometry, not bounding-circle approximation.
    Positive = clear gap, negative = overlapping.
    """
    # Cylinder-cylinder: center distance minus radii (exact)
    if a["shape"] == "cylinder" and b["shape"] == "cylinder":
        d_center = math.hypot(a["x"] - b["x"], a["y"] - b["y"])
        return d_center - a["radius"] - b["radius"]

    # Cylinder-box: distance from cylinder center to box surface minus radius
    if a["shape"] == "cylinder" and b["shape"] == "box":
        d_surf = _point_to_oriented_rect_distance(
            a["x"], a["y"], b["x"], b["y"],
            b.get("size_x", 1.0), b.get("size_y", 1.0),
            b.get("yaw", 0.0),
        )
        return d_surf - a["radius"]
    if b["shape"] == "cylinder" and a["shape"] == "box":
        return _obb_pairwise_surface_distance(b, a)

    # Box-box: sample corners to find min surface distance
    if a["shape"] == "box" and b["shape"] == "box":
        ax, ay = a["x"], a["y"]
        bx, by = b["x"], b["y"]
        asx, asy = a.get("size_x", 1.0), a.get("size_y", 1.0)
        bsx, bsy = b.get("size_x", 1.0), b.get("size_y", 1.0)
        ayaw = a.get("yaw", 0.0)
        byaw = b.get("yaw", 0.0)

        corners_a = _box_corners_xy(ax, ay, asx, asy, ayaw)
        corners_b = _box_corners_xy(bx, by, bsx, bsy, byaw)

        min_dist = float("inf")
        for cx, cy in corners_a:
            d = _point_to_oriented_rect_distance(cx, cy, bx, by, bsx, bsy, byaw)
            min_dist = min(min_dist, d)
        for cx, cy in corners_b:
            d = _point_to_oriented_rect_distance(cx, cy, ax, ay, asx, asy, ayaw)
            min_dist = min(min_dist, d)

        return min_dist

    # Fallback
    r_a = a.get("bounding_r", 0.5)
    r_b = b.get("bounding_r", 0.5)
    d_center = math.hypot(a["x"] - b["x"], a["y"] - b["y"])
    return d_center - r_a - r_b


# ─────────────────────────────────────────────────────────────────────────────
# Level 1: Geometry validation
# ─────────────────────────────────────────────────────────────────────────────


def validate_level1_geometry(
    metadata: Dict[str, Any],
    obstacles: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run Level 1 geometry validation checks.

    Returns dict with 'passed', 'fail_reasons', and clearance measurements.
    """
    fail_reasons: List[str] = []
    warnings: List[str] = []

    start_world = metadata.get("start_world", [0, 0, 2.0])
    goal_world = metadata.get("goal_world", [16, 0, 2.0])
    world_bounds = metadata.get("world_bounds", [-10, 10, -10, 10])
    obs_region = metadata.get("obstacle_region", [-10, 10])
    role = metadata.get("role_generated", "")

    # Determine clearance thresholds
    spawn_clearance_min = L1_SPAWN_CLEARANCE_MIN
    if role in ("dense_recovery", "yaw_lateral_recovery"):
        spawn_clearance_min = L1_SPAWN_CLEARANCE_RECOVERY

    # --- Check 1: obstacle bounds ---
    for obs in obstacles:
        obs_name = obs.get("name", "?")
        x, y = obs["x"], obs["y"]

        if not (obs_region[0] - 0.5 <= x <= obs_region[1] + 0.5):
            fail_reasons.append(f"{obs_name}: x={x:.2f} out of obstacle_region {obs_region}")
        if not (obs_region[0] - 0.5 <= y <= obs_region[1] + 0.5):
            fail_reasons.append(f"{obs_name}: y={y:.2f} out of obstacle_region {obs_region}")

        # Height sanity
        if obs["shape"] == "cylinder":
            h = obs.get("height", 0)
            r = obs.get("radius", 0)
            if h > L1_MAX_OBS_HEIGHT:
                fail_reasons.append(f"{obs_name}: height={h:.1f}m exceeds max {L1_MAX_OBS_HEIGHT}m")
            if r <= 0:
                fail_reasons.append(f"{obs_name}: non-positive radius={r}")
        elif obs["shape"] == "box":
            sz = obs.get("size_z", 0)
            if sz > L1_MAX_OBS_HEIGHT:
                fail_reasons.append(f"{obs_name}: size_z={sz:.1f}m exceeds max {L1_MAX_OBS_HEIGHT}m")

        # z bounds: obstacles shouldn't be below safe flight floor
        z_min = obs.get("z_min", obs["z"])
        if z_min < -0.5:
            fail_reasons.append(f"{obs_name}: z_min={z_min:.2f}m below ground")

    # --- Check 2: start surface clearance ---
    start_pt = (start_world[0], start_world[1], start_world[2])
    spawn_clearance = surface_distance_to_obstacles(start_pt, obstacles)
    if spawn_clearance < spawn_clearance_min:
        fail_reasons.append(
            f"spawn surface clearance={spawn_clearance:.3f}m < {spawn_clearance_min}m"
        )
    if spawn_clearance < L1_SPAWN_CLEARANCE_MIN and role in ("dense_recovery", "yaw_lateral_recovery"):
        warnings.append(
            f"spawn clearance={spawn_clearance:.3f}m below standard {L1_SPAWN_CLEARANCE_MIN}m "
            f"(relaxed for {role})"
        )

    # --- Check 3: goal surface clearance ---
    goal_pt = (goal_world[0], goal_world[1], goal_world[2])
    goal_clearance = surface_distance_to_obstacles(goal_pt, obstacles)
    if goal_clearance < L1_GOAL_CLEARANCE_MIN:
        fail_reasons.append(
            f"goal surface clearance={goal_clearance:.3f}m < {L1_GOAL_CLEARANCE_MIN}m"
        )

    # --- Check 4: obstacle pairwise gap (OBB-aware, not bounding circles) ---
    for i in range(len(obstacles)):
        for j in range(i + 1, len(obstacles)):
            a, b = obstacles[i], obstacles[j]
            gap = _obb_pairwise_surface_distance(a, b)
            if gap < -0.1:  # significant overlap
                fail_reasons.append(
                    f"{a['name']}↔{b['name']}: XY overlap gap={gap:.2f}m"
                )
            elif gap < 0.2:
                warnings.append(
                    f"{a['name']}↔{b['name']}: tight gap={gap:.2f}m"
                )

    # --- Check 5: z bounds sanity ---
    for obs in obstacles:
        z_max = obs.get("z_max", obs["z"])
        if z_max > 20.0:
            fail_reasons.append(f"{obs['name']}: z_max={z_max:.1f}m unreasonably high")

    passed = len(fail_reasons) == 0

    return {
        "passed": passed,
        "fail_reasons": fail_reasons,
        "warnings": warnings,
        "spawn_clearance_surface": round(spawn_clearance, 4),
        "goal_clearance_surface": round(goal_clearance, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Level 2: Reachability validation (occupancy grid + BFS + A*)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class OccupancyGrid:
    """3D occupancy grid for path planning."""
    x_min: float
    y_min: float
    z_min: float
    res_xy: float
    res_z: float
    nx: int
    ny: int
    nz: int
    grid: List[List[List[bool]]]  # grid[x][y][z] = True if occupied

    def is_valid(self, ix: int, iy: int, iz: int) -> bool:
        return 0 <= ix < self.nx and 0 <= iy < self.ny and 0 <= iz < self.nz

    def is_free(self, ix: int, iy: int, iz: int) -> bool:
        if not self.is_valid(ix, iy, iz):
            return False
        return not self.grid[ix][iy][iz]

    def world_to_grid(self, wx: float, wy: float, wz: float) -> Tuple[int, int, int]:
        ix = int((wx - self.x_min) / self.res_xy)
        iy = int((wy - self.y_min) / self.res_xy)
        iz = int((wz - self.z_min) / self.res_z)
        return ix, iy, iz

    def grid_to_world(self, ix: int, iy: int, iz: int) -> Tuple[float, float, float]:
        wx = self.x_min + (ix + 0.5) * self.res_xy
        wy = self.y_min + (iy + 0.5) * self.res_xy
        wz = self.z_min + (iz + 0.5) * self.res_z
        return wx, wy, wz


def build_occupancy_grid(
    obstacles: List[Dict[str, Any]],
    world_bounds: List[float],
    inflation_radius: float = L2_INFLATION_RADIUS,
    res_xy: float = L2_GRID_RES_XY,
    res_z: float = L2_GRID_RES_Z,
    z_range: Tuple[float, float] = L2_Z_VALID_RANGE,
) -> OccupancyGrid:
    """Build an inflated 3D occupancy grid from obstacles."""
    x_min, x_max = world_bounds[0], world_bounds[1]
    y_min, y_max = world_bounds[2], world_bounds[3]
    z_min, z_max = z_range[0], z_range[1]

    nx = int((x_max - x_min) / res_xy) + 1
    ny = int((y_max - y_min) / res_xy) + 1
    nz = int((z_max - z_min) / res_z) + 1

    # Initialize grid: all free
    grid = [[[False for _ in range(nz)] for _ in range(ny)] for _ in range(nx)]

    # Inflate each obstacle into the grid
    for obs in obstacles:
        inflate_r = obs.get("bounding_r", 0.5) + inflation_radius
        z_min_obs = obs.get("z_min", obs["z"]) - inflation_radius
        z_max_obs = obs.get("z_max", obs["z"]) + inflation_radius

        # Grid index range for this obstacle
        ix_min = max(0, int((obs["x"] - inflate_r - x_min) / res_xy))
        ix_max = min(nx - 1, int((obs["x"] + inflate_r - x_min) / res_xy))
        iy_min = max(0, int((obs["y"] - inflate_r - y_min) / res_xy))
        iy_max = min(ny - 1, int((obs["y"] + inflate_r - y_min) / res_xy))
        iz_min = max(0, int((z_min_obs - z_min) / res_z))
        iz_max = min(nz - 1, int((z_max_obs - z_min) / res_z))

        for ix in range(ix_min, ix_max + 1):
            wx = x_min + (ix + 0.5) * res_xy
            for iy in range(iy_min, iy_max + 1):
                wy = y_min + (iy + 0.5) * res_xy
                # Check actual XY distance to obstacle surface
                if obs["shape"] == "cylinder":
                    d_surf = math.hypot(wx - obs["x"], wy - obs["y"]) - obs["radius"]
                else:
                    d_surf = _point_to_oriented_rect_distance(
                        wx, wy, obs["x"], obs["y"],
                        obs.get("size_x", 1.0), obs.get("size_y", 1.0),
                        obs.get("yaw", 0.0),
                    )
                if d_surf > inflation_radius:
                    continue  # outside inflated region in XY

                for iz in range(iz_min, iz_max + 1):
                    wz = z_min + (iz + 0.5) * res_z
                    # Mark as occupied
                    grid[ix][iy][iz] = True

    return OccupancyGrid(
        x_min=x_min, y_min=y_min, z_min=z_min,
        res_xy=res_xy, res_z=res_z,
        nx=nx, ny=ny, nz=nz,
        grid=grid,
    )


def _find_nearest_free(
    grid: OccupancyGrid,
    wx: float, wy: float, wz: float,
    max_search_radius: float = 3.0,
) -> Optional[Tuple[int, int, int]]:
    """Find nearest free grid cell to a world point."""
    ix_c, iy_c, iz_c = grid.world_to_grid(wx, wy, wz)

    # Clamp z to valid range
    iz_c = max(0, min(grid.nz - 1, iz_c))

    search_range = int(max_search_radius / grid.res_xy) + 1
    for r in range(search_range):
        for dix in range(-r, r + 1):
            for diy in range(-r, r + 1):
                if abs(dix) != r and abs(diy) != r:
                    continue  # only check perimeter of expanding square
                ix = ix_c + dix
                iy = iy_c + diy
                if grid.is_free(ix, iy, iz_c):
                    return ix, iy, iz_c

    # If not found, check all cells within radius
    for dix in range(-search_range, search_range + 1):
        for diy in range(-search_range, search_range + 1):
            for diz in range(-3, 4):
                ix = ix_c + dix
                iy = iy_c + diy
                iz = iz_c + diz
                if grid.is_free(ix, iy, iz):
                    return ix, iy, iz

    return None


def _bfs_2d(
    grid: OccupancyGrid,
    start_ix: int, start_iy: int, start_iz: int,
    goal_ix: int, goal_iy: int, goal_iz: int,
) -> Tuple[bool, float, List[Tuple[int, int, int]]]:
    """2D BFS at fixed z level for quick reachability check. Returns (reachable, path_len, path)."""
    # Use the z level of start
    z_level = start_iz

    visited: Set[Tuple[int, int]] = set()
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {}
    queue = deque()

    start_xy = (start_ix, start_iy)
    goal_xy = (goal_ix, goal_iy)

    queue.append(start_xy)
    visited.add(start_xy)
    parent[start_xy] = None

    # 4-connected neighbors
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while queue:
        cx, cy = queue.popleft()
        if (cx, cy) == goal_xy:
            # Reconstruct path
            path = []
            cur = goal_xy
            while cur is not None:
                path.append((cur[0], cur[1], z_level))
                cur = parent[cur]
            path.reverse()
            path_len = len(path) * grid.res_xy  # approximate
            return True, path_len, path

        for dx, dy in dirs:
            nx, ny = cx + dx, cy + dy
            if (nx, ny) in visited:
                continue
            if grid.is_free(nx, ny, z_level):
                visited.add((nx, ny))
                parent[(nx, ny)] = (cx, cy)
                queue.append((nx, ny))

    return False, 0.0, []


def _astar_3d(
    grid: OccupancyGrid,
    start_ix: int, start_iy: int, start_iz: int,
    goal_ix: int, goal_iy: int, goal_iz: int,
) -> Tuple[bool, float, float, float, float, List[Tuple[int, int, int]]]:
    """3D A* search. Returns (reachable, path_len, bottleneck_clearance, z_span_min, z_span_max, path)."""
    start = (start_ix, start_iy, start_iz)
    goal = (goal_ix, goal_iy, goal_iz)

    def heuristic(a, b):
        return math.sqrt(
            (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
        )

    open_set: List[Tuple[float, int, Tuple[int, int, int]]] = []
    heapq.heappush(open_set, (0.0, 0, start))

    g_score: Dict[Tuple[int, int, int], float] = {start: 0.0}
    parent: Dict[Tuple[int, int, int], Optional[Tuple[int, int, int]]] = {start: None}
    visited: Set[Tuple[int, int, int]] = set()

    # 6-connected in 3D
    dirs = [
        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
    ]

    max_iterations = grid.nx * grid.ny * grid.nz // 10  # prevent infinite loops
    iterations = 0

    while open_set and iterations < max_iterations:
        iterations += 1
        _, _, current = heapq.heappop(open_set)

        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            # Reconstruct path
            path = []
            cur = goal
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()

            # Compute metrics
            path_len = 0.0
            z_span_min = float("inf")
            z_span_max = float("-inf")
            bottleneck = float("inf")
            for i in range(len(path)):
                wx, wy, wz = grid.grid_to_world(*path[i])
                z_span_min = min(z_span_min, wz)
                z_span_max = max(z_span_max, wz)
                if i > 0:
                    pw, _, _ = grid.grid_to_world(*path[i - 1])
                    path_len += grid.res_xy  # approximate
                # Cell-level bottleneck: just count free neighbors as proxy
                ix, iy, iz = path[i]
                free_nbrs = sum(
                    1 for d in dirs if grid.is_free(ix + d[0], iy + d[1], iz + d[2])
                )
                # Fewer free neighbors = tighter bottleneck
                if free_nbrs < 4:
                    bottleneck = min(bottleneck, free_nbrs * grid.res_xy)

            turning_angle_sum = 0.0
            for i in range(2, len(path)):
                p0 = grid.grid_to_world(*path[i - 2])
                p1 = grid.grid_to_world(*path[i - 1])
                p2 = grid.grid_to_world(*path[i])
                v1 = (p1[0] - p0[0], p1[1] - p0[1])
                v2 = (p2[0] - p1[0], p2[1] - p1[1])
                d1 = math.hypot(*v1)
                d2 = math.hypot(*v2)
                if d1 > 0.01 and d2 > 0.01:
                    dot = (v1[0] * v2[0] + v1[1] * v2[1]) / (d1 * d2)
                    dot = max(-1.0, min(1.0, dot))
                    turning_angle_sum += math.acos(dot)

            bottleneck_clr = bottleneck if bottleneck != float("inf") else 2.0
            z_span = (z_span_min, z_span_max)
            return True, path_len, bottleneck_clr, z_span[0], z_span[1], path

        ci, cj, ck = current
        for d in dirs:
            ni, nj, nk = ci + d[0], cj + d[1], ck + d[2]
            neighbor = (ni, nj, nk)
            if not grid.is_free(ni, nj, nk):
                continue
            # Diagonal cost for z movement
            move_cost = grid.res_xy if d[2] == 0 else math.hypot(grid.res_xy, grid.res_z)
            tentative_g = g_score[current] + move_cost
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, iterations, neighbor))
                parent[neighbor] = current

    return False, 0.0, 0.0, 0.0, 0.0, []


def validate_level2_reachability(
    metadata: Dict[str, Any],
    obstacles: List[Dict[str, Any]],
    grid_res_xy: float = L2_GRID_RES_XY,
    grid_res_z: float = L2_GRID_RES_Z,
    inflation_radius: float = L2_INFLATION_RADIUS,
    z_valid_range: Tuple[float, float] = L2_Z_VALID_RANGE,
    max_path_stretch: float = L2_MAX_PATH_STRETCH,
) -> Dict[str, Any]:
    """Run Level 2 reachability validation using occupancy grid + BFS + A*."""
    start_world = metadata.get("start_world", [-8.0, 0.0, 2.0])
    goal_world = metadata.get("goal_world", [8.0, 0.0, 2.0])
    world_bounds = metadata.get("world_bounds", [-10.0, 10.0, -10.0, 10.0])

    sx, sy, sz = start_world[0], start_world[1], start_world[2]
    gx, gy, gz = goal_world[0], goal_world[1], goal_world[2]

    straight_line_dist = math.hypot(gx - sx, gy - sy)

    if not obstacles:
        # Empty world: trivially reachable
        return {
            "reachable_2d": True,
            "reachable_3d": True,
            "path_length_2d": straight_line_dist,
            "path_length_3d": straight_line_dist,
            "path_stretch": 1.0,
            "bottleneck_clearance_surface": 100.0,
            "z_span_oracle": [sz, sz],
            "turning_angle_sum": 0.0,
            "reachability_label": "reachable_trivial",
        }

    # Build occupancy grid
    grid = build_occupancy_grid(
        obstacles, world_bounds,
        inflation_radius=inflation_radius,
        res_xy=grid_res_xy,
        res_z=grid_res_z,
        z_range=z_valid_range,
    )

    # Find nearest free cells for start and goal
    start_cell = _find_nearest_free(grid, sx, sy, sz)
    goal_cell = _find_nearest_free(grid, gx, gy, gz)

    if start_cell is None:
        return {
            "reachable_2d": False,
            "reachable_3d": False,
            "path_length_2d": 0.0,
            "path_length_3d": 0.0,
            "path_stretch": 0.0,
            "bottleneck_clearance_surface": 0.0,
            "z_span_oracle": [sz, sz],
            "turning_angle_sum": 0.0,
            "reachability_label": "reachable_no_start_cell",
        }

    if goal_cell is None:
        return {
            "reachable_2d": False,
            "reachable_3d": False,
            "path_length_2d": 0.0,
            "path_length_3d": 0.0,
            "path_stretch": 0.0,
            "bottleneck_clearance_surface": 0.0,
            "z_span_oracle": [sz, sz],
            "turning_angle_sum": 0.0,
            "reachability_label": "reachable_no_goal_cell",
        }

    # 2D BFS quick check
    reachable_2d, path_len_2d, path_2d = _bfs_2d(
        grid, *start_cell, *goal_cell,
    )

    # 3D A* full check
    reachable_3d, path_len_3d, bottleneck_clr, z_min, z_max, path_3d = _astar_3d(
        grid, *start_cell, *goal_cell,
    )

    # Path stretch
    if path_len_3d > 0 and straight_line_dist > 0:
        stretch = path_len_3d / straight_line_dist
    else:
        stretch = 0.0

    # Reachability label
    if reachable_3d:
        if stretch <= 1.1:
            label = "reachable_direct"
        elif stretch <= max_path_stretch:
            label = "reachable_frontier"
        else:
            label = "reachable_detour"
    elif reachable_2d:
        label = "reachable_2d_only"
    else:
        label = "unreachable"

    return {
        "reachable_2d": reachable_2d,
        "reachable_3d": reachable_3d,
        "path_length_2d": round(path_len_2d, 4),
        "path_length_3d": round(path_len_3d, 4),
        "path_stretch": round(stretch, 4),
        "bottleneck_clearance_surface": round(bottleneck_clr, 4),
        "z_span_oracle": [round(z_min, 4), round(z_max, 4)],
        "turning_angle_sum": round(0.0, 4),  # computed in A*
        "reachability_label": label,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Level 3: Semantic validation
# ─────────────────────────────────────────────────────────────────────────────


def _compute_straight_path_clearance(
    start_world: List[float],
    goal_world: List[float],
    obstacles: List[Dict[str, Any]],
    n_steps: int = 20,
) -> Dict[str, float]:
    """Compute clearance stats along the straight start→goal path."""
    sx, sy = start_world[0], start_world[1]
    gx, gy = goal_world[0], goal_world[1]

    clearances = []
    for i in range(n_steps + 1):
        t = i / n_steps
        px = sx + t * (gx - sx)
        py = sy + t * (gy - sy)
        clr = surface_distance_xy_only((px, py), obstacles)
        clearances.append(clr)

    return {
        "min_clearance": min(clearances),
        "mean_clearance": sum(clearances) / len(clearances),
        "clearance_below_2m_ratio": sum(1 for c in clearances if c < 2.0) / len(clearances),
    }


def _count_path_crossings(
    start_world: List[float],
    goal_world: List[float],
    obstacles: List[Dict[str, Any]],
) -> int:
    """Count how many obstacles intersect the straight start→goal line."""
    sx, sy = start_world[0], start_world[1]
    gx, gy = goal_world[0], goal_world[1]
    dx, dy = gx - sx, gy - sy
    path_len = math.hypot(dx, dy)
    if path_len < 0.01:
        return 0

    count = 0
    for obs in obstacles:
        ox_rel = obs["x"] - sx
        oy_rel = obs["y"] - sy
        t = (ox_rel * dx + oy_rel * dy) / (path_len * path_len)
        t = max(0.0, min(1.0, t))
        closest_x = sx + t * dx
        closest_y = sy + t * dy
        dist = math.hypot(obs["x"] - closest_x, obs["y"] - closest_y)
        if dist < obs.get("bounding_r", 0.5):
            count += 1

    return count


def _has_bypass_corridor(
    start_world: List[float],
    goal_world: List[float],
    obstacles: List[Dict[str, Any]],
    min_width: float = 1.5,
) -> bool:
    """Check if there exists an alternative corridor bypassing the straight path."""
    sx, sy = start_world[0], start_world[1]
    gx, gy = goal_world[0], goal_world[1]

    # Sample points offset from the direct line
    path_len = math.hypot(gx - sx, gy - sy)
    dx, dy = (gx - sx) / path_len, (gy - sy) / path_len
    # Perpendicular
    px, py = -dy, dx

    for offset_sign in [-1, 1]:
        for offset in [2.0, 3.0, 4.0]:
            all_clear = True
            for t in [0.2, 0.4, 0.6, 0.8]:
                cx = sx + t * (gx - sx) + offset_sign * offset * px
                cy = sy + t * (gy - sy) + offset_sign * offset * py
                clr = surface_distance_xy_only((cx, cy), obstacles)
                if clr < min_width / 2.0:
                    all_clear = False
                    break
            if all_clear:
                return True
    return False


def validate_level3_anticipatory(
    metadata: Dict[str, Any],
    obstacles: List[Dict[str, Any]],
    level2_report: Dict[str, Any],
) -> Dict[str, Any]:
    """Semantic checks for anticipatory role."""
    start_world = metadata.get("start_world", [-8.0, 0.0, 2.0])
    goal_world = metadata.get("goal_world", [8.0, 0.0, 2.0])

    straight_clr = _compute_straight_path_clearance(start_world, goal_world, obstacles)
    path_crossings = _count_path_crossings(start_world, goal_world, obstacles)
    has_bypass = _has_bypass_corridor(start_world, goal_world, obstacles, min_width=1.5)

    risk_window_steps = 0
    sx, sy = start_world[0], start_world[1]
    gx, gy = goal_world[0], goal_world[1]
    for i in range(L3_RISK_WINDOW_STEPS):
        t = i / L3_RISK_WINDOW_STEPS
        px = sx + t * (gx - sx)
        py = sy + t * (gy - sy)
        clr = surface_distance_xy_only((px, py), obstacles)
        if clr < L3_STRAIGHT_D_SOFT_H_MIN:
            risk_window_steps += 1

    # anticipatory requires: straight path has obstacle that forces early planning.
    # Two valid patterns:
    #   a) obstacle intersects direct path (path_crossings >= 1) AND bypass exists
    #   b) obstacle is NEAR the direct path (within 3.5m soft clearance) AND bypass exists
    #      → drone must anticipate early to smoothly go around, even if technically
    #        the straight line is physically clear
    obstacle_near_path = straight_clr["min_clearance"] < 3.5 and path_crossings == 0
    is_anticipatory = (
        (path_crossings >= 1 and has_bypass)
        or (obstacle_near_path and has_bypass and len(obstacles) >= 1)
    )

    return {
        "straight_d_soft_h_min": round(straight_clr["min_clearance"], 4),
        "risk_window_steps": risk_window_steps,
        "best_alt_corridor_d_soft_h": round(straight_clr["mean_clearance"], 4),
        "path_stretch_oracle": level2_report.get("path_stretch", 0.0),
        "has_bypass_corridor": has_bypass,
        "path_crossing_count": path_crossings,
        "matches_role": is_anticipatory,
    }


def _compute_path_bottleneck(
    start_world: List[float],
    goal_world: List[float],
    obstacles: List[Dict[str, Any]],
    n_samples: int = 30,
) -> float:
    """Compute minimum surface clearance along the straight start→goal line."""
    sx, sy = start_world[0], start_world[1]
    gx, gy = goal_world[0], goal_world[1]
    min_clr = float("inf")
    for i in range(n_samples + 1):
        t = i / n_samples
        px = sx + t * (gx - sx)
        py = sy + t * (gy - sy)
        clr = surface_distance_xy_only((px, py), obstacles)
        min_clr = min(min_clr, clr)
    return min_clr


def validate_level3_dense_recovery(
    metadata: Dict[str, Any],
    obstacles: List[Dict[str, Any]],
    level2_report: Dict[str, Any],
) -> Dict[str, Any]:
    """Semantic checks for dense_recovery role."""
    start_world = metadata.get("start_world", [-8.0, 0.0, 2.0])
    goal_world = metadata.get("goal_world", [8.0, 0.0, 2.0])

    # Compute actual path bottleneck using surface clearance along straight line
    path_bottleneck = _compute_path_bottleneck(start_world, goal_world, obstacles)
    path_crossings = _count_path_crossings(start_world, goal_world, obstacles)

    # Branch ambiguity: count obstacles that create multi-modal choices
    branch_ambiguity = 0.0
    if len(obstacles) >= 2:
        gaps = []
        for i in range(len(obstacles)):
            for j in range(i + 1, len(obstacles)):
                gap = _obb_pairwise_surface_distance(obstacles[i], obstacles[j])
                if gap > 0:
                    gaps.append(gap)
        if gaps:
            mean_gap = sum(gaps) / len(gaps)
            variance = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
            branch_ambiguity = round(1.0 / (1.0 + math.sqrt(variance)), 4)

    # Loop propensity: ratio of above/below path obstacles (U-shape detection)
    loop_propensity = 0.0
    if obstacles:
        n_above = sum(1 for o in obstacles if o["y"] > 0.5)
        n_below = sum(1 for o in obstacles if o["y"] < -0.5)
        if n_above > 0 and n_below > 0:
            loop_propensity = round(min(n_above, n_below) / max(n_above, n_below), 4)

    # dense_recovery: tight clearance on the path (bottleneck) or many crossings
    has_tight_bottleneck = path_bottleneck < 2.0
    has_many_crossings = path_crossings >= 2
    has_trap_structure = loop_propensity > 0.3
    high_obstacle_count = len(obstacles) >= 5

    is_dense_recovery = (
        (has_tight_bottleneck and has_many_crossings)
        or (has_trap_structure and path_crossings >= 1)
        or (high_obstacle_count and path_crossings >= 2)
        or (len(obstacles) >= 3 and path_bottleneck < 1.0)
    )

    return {
        "bottleneck_clearance_surface": round(path_bottleneck, 4),
        "trap_depth": round(max(0.0, path_crossings * 0.5), 4),
        "branch_ambiguity_score": branch_ambiguity,
        "loop_propensity_score": loop_propensity,
        "path_crossing_count": path_crossings,
        "matches_role": is_dense_recovery,
    }


def validate_level3_support_expansion(
    metadata: Dict[str, Any],
    obstacles: List[Dict[str, Any]],
    level2_report: Dict[str, Any],
) -> Dict[str, Any]:
    """Semantic checks for support_expansion role.

    Requires offline density data for full evaluation.
    Without it, marks as pending_offline_density.
    """
    # Check if offline density data exists in metadata
    has_density = False

    # Count obstacles of different types
    n_low = sum(1 for o in obstacles
                if o.get("z_max", o.get("z", 0) + o.get("height", o.get("size_z", 0)) / 2.0) < 2.5)
    n_high = sum(1 for o in obstacles
                 if o.get("z_min", o.get("z", 0) - o.get("height", o.get("size_z", 0)) / 2.0) > 2.0)
    n_mid = len(obstacles) - n_low - n_high

    # Low density axes: count directions with few obstacles
    # Check ±x, ±y quadrants
    axes_with_obs = 0
    for sign_x in [-1, 1]:
        for sign_y in [-1, 1]:
            n_quad = sum(
                1 for o in obstacles
                if (o["x"] * sign_x > 0 and o["y"] * sign_y > 0)
            )
            if n_quad > 0:
                axes_with_obs += 1
    low_density_axes = 4 - axes_with_obs

    # Altitude novelty: obstacles span different z ranges
    z_ranges = []
    for o in obstacles:
        z_min = o.get("z_min", o.get("z", 0) - 0.5)
        z_max = o.get("z_max", o.get("z", 0) + 0.5)
        z_ranges.append((z_min, z_max))

    # Check if there's altitude-only novelty (must go over/under, not just around)
    has_altitude_novelty = n_low >= 1 and n_high >= 1

    # support_expansion requires: mixed height obstacles with low density axes
    is_support_expansion = (
        len(obstacles) >= 3
        and n_low >= 1 and n_high >= 1
        and low_density_axes >= 1
    )

    result = {
        "offline_density_path_p20_ratio": None,
        "low_density_axes_count": low_density_axes,
        "altitude_only_novelty_ratio": 1.0 if has_altitude_novelty else 0.0,
        "n_low_obs": n_low,
        "n_mid_obs": n_mid,
        "n_high_obs": n_high,
        "semantic_status": "ok",  # may be overridden
        "matches_role": is_support_expansion,
    }

    # If no offline density data, mark as pending
    if not has_density:
        result["semantic_status"] = "pending_offline_density"
        # Don't reject — just mark pending. Full admission requires density.

    return result


def validate_level3_yaw_lateral_recovery(
    metadata: Dict[str, Any],
    obstacles: List[Dict[str, Any]],
    level2_report: Dict[str, Any],
) -> Dict[str, Any]:
    """Semantic checks for yaw_lateral_recovery role."""
    start_world = metadata.get("start_world", [-8.0, 0.0, 2.0])
    start_yaw = metadata.get("start_yaw", 0.0)

    # Recovery corridor exists: is there a clear path after the initial offset?
    # Check if obstacles allow lateral recovery
    recovery_corridor = False
    if obstacles:
        # Sample forward from start at offset directions
        sx, sy = start_world[0], start_world[1]
        # Directions perpendicular to start_yaw
        cos_y = math.cos(start_yaw)
        sin_y = math.sin(start_yaw)
        perp_x, perp_y = -sin_y, cos_y  # left perpendicular

        for offset in [2.0, 3.0]:
            for sign in [-1, 1]:
                check_x = sx + sign * offset * perp_x + cos_y * 2.0
                check_y = sy + sign * offset * perp_y + sin_y * 2.0
                clr = surface_distance_xy_only((check_x, check_y), obstacles)
                if clr > 1.0:  # enough clearance for recovery
                    recovery_corridor = True
                    break
            if recovery_corridor:
                break

    # Recovery window: how many steps until clear path is available
    recovery_window = 0
    if start_yaw != 0.0:
        # Higher yaw offset = longer recovery needed
        recovery_window = int(abs(start_yaw) / (math.pi / L3_RECOVERY_WINDOW_STEPS))
        recovery_window = min(recovery_window, L3_RECOVERY_WINDOW_STEPS)

    # yaw_lateral_recovery requires: non-zero start_yaw AND recovery corridor exists
    is_yaw_recovery = abs(start_yaw) > 0.1 and recovery_corridor

    # Also check: if role is yaw_lateral_recovery but start_yaw is 0,
    # it doesn't match (generator misconfiguration)
    if abs(start_yaw) < 0.05:
        is_yaw_recovery = False

    return {
        "recovery_corridor_exists": recovery_corridor,
        "recovery_window_steps": recovery_window,
        "start_yaw_offset": abs(start_yaw),
        "matches_role": is_yaw_recovery,
    }


def validate_level3_semantic(
    metadata: Dict[str, Any],
    obstacles: List[Dict[str, Any]],
    level2_report: Dict[str, Any],
) -> Dict[str, Any]:
    """Run Level 3 semantic validation, dispatching to role-specific checks.

    Also computes best-matching role for possible relabel.
    """
    role = metadata.get("role_generated", "")

    role_checks = {
        "anticipatory": validate_level3_anticipatory,
        "dense_recovery": validate_level3_dense_recovery,
        "support_expansion": validate_level3_support_expansion,
        "yaw_lateral_recovery": validate_level3_yaw_lateral_recovery,
    }

    # Run all role checks to find best match
    all_results: Dict[str, Dict[str, Any]] = {}
    for r, check_fn in role_checks.items():
        all_results[r] = check_fn(metadata, obstacles, level2_report)

    # Primary result is for the generated role
    primary = all_results.get(role, {})

    # Find best matching role (for relabel)
    best_role = role
    best_match = primary.get("matches_role", False)
    for r, result in all_results.items():
        if result.get("matches_role", False) and r != role:
            # If another role also matches and generated role doesn't, relabel
            if not best_match:
                best_role = r
                best_match = True

    relabel_reason = None
    if best_role != role:
        relabel_reason = (
            f"Generated as '{role}' but semantics best match '{best_role}': "
            f"generated_role_match={primary.get('matches_role', False)}, "
            f"best_role_match=True"
        )

    # Only propagate pending_offline_density for support_expansion role
    semantic_status = "ok"
    if role == "support_expansion":
        semantic_status = all_results.get("support_expansion", {}).get(
            "semantic_status", "ok"
        )

    return {
        "primary_role": role,
        "matches_generated_role": primary.get("matches_role", False),
        "best_matching_role": best_role,
        "needs_relabel": best_role != role and best_match,
        "relabel_reason": relabel_reason,
        "role_checks": {
            r: {k: v for k, v in checks.items()}
            for r, checks in all_results.items()
        },
        "semantic_status": semantic_status,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Relabel and admission logic
# ─────────────────────────────────────────────────────────────────────────────


def compute_admission(
    level1_report: Dict[str, Any],
    level2_report: Dict[str, Any],
    level3_report: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply relabel rules to determine final role_validated, role_active, admission_status.

    Rules:
      L1 fail → reject_geometry
      L2 fail → role_active="stress_only", reject_unreachable_or_stress
      L3 pass → role_validated = role_generated, pending_calibration
      L3 semantic match another → relabel, pending_calibration_relabelled
      L3 fail → reject_semantic
    """
    l1_pass = level1_report.get("passed", False)
    l2_reachable = level2_report.get("reachable_3d", False)
    l3_matches = level3_report.get("matches_generated_role", False)
    l3_needs_relabel = level3_report.get("needs_relabel", False)
    l3_best_role = level3_report.get("best_matching_role", "")
    l3_semantic_status = level3_report.get("semantic_status", "ok")

    role_generated = level3_report.get("primary_role", "")

    if not l1_pass:
        return {
            "role_validated": None,
            "role_active": None,
            "admission_status": "reject_geometry",
            "fail_reasons": level1_report.get("fail_reasons", []),
        }

    if not l2_reachable:
        return {
            "role_validated": None,
            "role_active": "stress_only",
            "admission_status": "reject_unreachable_or_stress",
            "fail_reasons": [f"L2 unreachable: {level2_report.get('reachability_label', '?')}"],
        }

    # Check for pending density
    if l3_semantic_status == "pending_offline_density":
        return {
            "role_validated": None,
            "role_active": None,
            "admission_status": "pending_offline_density",
            "fail_reasons": [],
        }

    if l3_needs_relabel:
        return {
            "role_validated": l3_best_role,
            "role_active": None,
            "admission_status": "pending_calibration_relabelled",
            "fail_reasons": [],
        }

    if l3_matches:
        return {
            "role_validated": role_generated,
            "role_active": None,
            "admission_status": "pending_calibration",
            "fail_reasons": [],
        }

    # L3 fail
    return {
        "role_validated": None,
        "role_active": None,
        "admission_status": "reject_semantic",
        "fail_reasons": [
            f"role_generated='{role_generated}' does not match semantic profile"
        ],
    }
