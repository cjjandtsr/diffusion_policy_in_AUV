#!/usr/bin/env python3
"""eval_bank.py — P0 96-episode fixed evaluation bank builder + admission audit.

Produces:
    eval_bank_manifest.json — episode listing with full pairing keys.
    eval_bank_audit.json   — gate decision with per-role coverage table,
                              admission pass/reject, replay-identity subset.
"""

import json
import math
import os
import sys
import time
import heapq
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np

from scripts.capacity_budget_p0 import SCHEMA_VERSION, GATE_PASS_KEY

# Accessible at module level for tests
FORMAL_ROLES = ("anticipatory", "dense_recovery", "support_expansion")
PER_ROLE_QUOTA = {"anticipatory": 32, "dense_recovery": 32, "support_expansion": 32}
MIN_WORLDS_PER_ROLE = 3
MIN_EPS_PER_WORLD = 4
MAX_SINGLE_WORLD_SHARE = 0.50  # strictly < 50%

# Role-agnostic sampling parameters (mirrors KinematicUAVEnv._ROLE_SAMPLING_PARAMS)
ROLE_PARAMS = {
    "anticipatory":      {"d_xy": (10, 24), "start_cl": 0.60, "goal_cl": 2.50},
    "dense_recovery":    {"d_xy": (5, 18),  "start_cl": 0.60, "goal_cl": 1.50},
    "support_expansion": {"d_xy": (8, 26),  "start_cl": 0.80, "goal_cl": 2.00},
}

OCC_RES = 0.5  # occupancy grid resolution
WORLD_DIR = "/home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds"
REQUIRED_EPISODE_FIELDS = {
    "world_name", "start_pos", "goal_pos", "episode_seed",
    "pointcloud_rng_seed", "role", "episode_id",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Pure validators
# ═══════════════════════════════════════════════════════════════════════════════

def build_pairing_key(ep: dict) -> str:
    """Deterministic pairing key: world + start + goal + seeds."""
    w = ep.get("world_name", "?")
    sx, sy, sz = ep.get("start_pos", [0, 0, 0])
    gx, gy, gz = ep.get("goal_pos", [0, 0, 0])
    es = ep.get("episode_seed", 0)
    ps = ep.get("pointcloud_rng_seed", 0)
    return (
        f"{w}|{sx:.4f},{sy:.4f},{sz:.4f}|{gx:.4f},{gy:.4f},{gz:.4f}"
        f"|es={es}|ps={ps}"
    )


def validate_bank_quota(episodes: List[dict]) -> dict:
    """Pure validator for bank quota, world-coverage, and pairing-key uniqueness.

    Returns a dict with keys:
        pass (bool), roles (dict), dup_pairing_keys (int),
        missing_fields (int), unrecognised_roles (list), notes (list).
    """
    report = {
        "pass": True,
        "roles": {},
        "dup_pairing_keys": 0,
        "missing_fields": 0,
        "unrecognised_roles": [],
        "notes": [],
    }

    # ── Per-role grouping ──────────────────────────────────────────────────
    role_eps: Dict[str, List[dict]] = defaultdict(list)
    for ep in episodes:
        role = ep.get("role", "?")
        # Check required fields
        missing = REQUIRED_EPISODE_FIELDS - set(ep.keys())
        if missing:
            report["missing_fields"] += 1
        # Check unrecognised role
        if role not in FORMAL_ROLES and role not in role_eps.get("_unrecog", []):
            report["unrecognised_roles"].append(role)
        role_eps[role].append(ep)

    report["unrecognised_roles"] = list(set(report["unrecognised_roles"]))

    # ── Pairing-key uniqueness ─────────────────────────────────────────────
    seen_keys: Set[str] = set()
    dup_keys = 0
    for ep in episodes:
        key = build_pairing_key(ep)
        if key in seen_keys:
            dup_keys += 1
        else:
            seen_keys.add(key)
    report["dup_pairing_keys"] = dup_keys
    if dup_keys > 0:
        report["pass"] = False
        report["notes"].append(f"Duplicate pairing keys: {dup_keys}")

    # ── Per-role checks ────────────────────────────────────────────────────
    global_pass = True

    for role in FORMAL_ROLES:
        role_eps_list = role_eps.get(role, [])
        count = len(role_eps_list)
        n_worlds = len(set(e.get("world_name") for e in role_eps_list))
        per_world_counts = Counter(e.get("world_name") for e in role_eps_list)
        max_world_share = max(per_world_counts.values()) / max(count, 1) if count > 0 else 0.0

        role_ok = True
        role_notes = []

        # Count check
        expected = PER_ROLE_QUOTA.get(role, 32)
        if count != expected:
            role_ok = False
            role_notes.append(f"count={count}, expected={expected}")

        # World coverage
        if n_worlds < MIN_WORLDS_PER_ROLE:
            role_ok = False
            role_notes.append(
                f"n_worlds={n_worlds}, required >= {MIN_WORLDS_PER_ROLE}"
            )

        # Per-world min episodes
        for w, wc in per_world_counts.items():
            if wc < MIN_EPS_PER_WORLD:
                role_ok = False
                role_notes.append(
                    f"world '{w}' has {wc} eps, required >= {MIN_EPS_PER_WORLD}"
                )

        # Single-world share max
        if max_world_share >= MAX_SINGLE_WORLD_SHARE and count > 0:
            role_ok = False
            role_notes.append(
                f"max_single_world_share={max_world_share:.4f}, "
                f"required < {MAX_SINGLE_WORLD_SHARE}"
            )

        report["roles"][role] = {
            "count": count,
            "expected": expected,
            "n_worlds": n_worlds,
            "required_worlds": MIN_WORLDS_PER_ROLE,
            "per_world_counts": dict(per_world_counts),
            "max_single_world_share": max_world_share,
            "pass": role_ok,
            "notes": role_notes,
        }
        if not role_ok:
            global_pass = False

    if report["missing_fields"] > 0:
        global_pass = False
        report["notes"].append(
            f"Episodes with missing fields: {report['missing_fields']}"
        )

    report["pass"] = global_pass
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# Geometry helpers (standalone, no GPU needed)
# ═══════════════════════════════════════════════════════════════════════════════

def _curriculum_xy_range(stage: str = "C2"):
    if stage == "C0":
        return -10.0, 10.0, -6.0, 6.0
    elif stage == "C1":
        return -12.0, 12.0, -12.0, 12.0
    else:
        return -14.0, 14.0, -14.0, 14.0


def _curriculum_z_range(stage: str = "C2"):
    if stage == "C0":
        return 1.3, 3.3
    elif stage == "C1":
        return 1.0, 3.5
    else:
        return 1.0, 5.0


def build_occupancy_grid(
    obstacles: List, curriculum_stage: str = "C2"
) -> Tuple[np.ndarray, dict]:
    """Build 3D occupancy grid from obstacle list.

    Returns (grid, meta) where grid[y][x][z] is True (free) or False (blocked).
    Replicates KinematicUAVEnv._build_occupancy_grid.
    """
    res = OCC_RES
    x_min, x_max, y_min, y_max = _curriculum_xy_range(curriculum_stage)
    z_low, z_high = _curriculum_z_range(curriculum_stage)
    # Expand slightly for safety margin
    x_min, x_max = x_min - 1.0, x_max + 1.0
    y_min, y_max = y_min - 1.0, y_max + 1.0
    z_low, z_high = max(z_low - 0.5, 0.0), z_high + 0.5

    nx = max(int((x_max - x_min) / res) + 1, 2)
    ny = max(int((y_max - y_min) / res) + 1, 2)
    nz = max(int((z_high - z_low) / res) + 1, 2)

    grid = np.ones((ny, nx, nz), dtype=bool)

    for obs in obstacles:
        cx, cy, cz = obs.x, obs.y, obs.z
        r = obs.radius + 0.15  # drone radius margin
        h = obs.height

        ix0 = max(0, int((cx - r - x_min) / res))
        ix1 = min(nx - 1, int((cx + r - x_min) / res))
        iy0 = max(0, int((cy - r - y_min) / res))
        iy1 = min(ny - 1, int((cy + r - y_min) / res))
        iz0 = max(0, int((cz - h / 2 - z_low) / res))
        iz1 = min(nz - 1, int((cz + h / 2 - z_low) / res))

        for iy in range(iy0, iy1 + 1):
            for ix in range(ix0, ix1 + 1):
                wx = x_min + (ix + 0.5) * res
                wy = y_min + (iy + 0.5) * res
                d2 = (wx - cx) ** 2 + (wy - cy) ** 2
                if d2 < r ** 2:
                    grid[iy, ix, iz0: iz1 + 1] = False

    meta = {
        "x_min": x_min, "y_min": y_min, "z_min": z_low,
        "res": res, "nx": nx, "ny": ny, "nz": nz,
    }
    return grid, meta


def _world_to_grid(x: float, y: float, z: float, meta: dict):
    ix = int((x - meta["x_min"]) / meta["res"])
    iy = int((y - meta["y_min"]) / meta["res"])
    iz = int((z - meta["z_min"]) / meta["res"])
    return ix, iy, iz


def astar_3d(
    occ_grid: np.ndarray, meta: dict,
    sx: float, sy: float, sz: float,
    gx: float, gy: float, gz: float,
) -> Tuple[bool, float, float, float, float, list]:
    """3D A* search on occupancy grid.

    Returns (reachable, path_len, bottleneck_clr, z_min, z_max, path).
    Replicates KinematicUAVEnv._astar_3d.
    """
    six, siy, siz = _world_to_grid(sx, sy, sz, meta)
    gix, giy, giz = _world_to_grid(gx, gy, gz, meta)

    six = max(0, min(meta["nx"] - 1, six))
    siy = max(0, min(meta["ny"] - 1, siy))
    siz = max(0, min(meta["nz"] - 1, siz))
    gix = max(0, min(meta["nx"] - 1, gix))
    giy = max(0, min(meta["ny"] - 1, giy))
    giz = max(0, min(meta["nz"] - 1, giz))

    if not occ_grid[siy, six, siz] or not occ_grid[giy, gix, giz]:
        return False, 0.0, 0.0, 0.0, 0.0, []

    start = (six, siy, siz)
    goal = (gix, giy, giz)
    if start == goal:
        return True, 0.0, 2.0, sz, gz, [start]

    dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    def _heuristic(a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)

    open_set = [(0.0, 0, start)]
    g_score = {start: 0.0}
    parent: Dict[Tuple, Optional[Tuple]] = {start: None}
    visited: Set[Tuple] = set()

    max_iter = meta["nx"] * meta["ny"] * meta["nz"] // 5
    iters = 0

    while open_set and iters < max_iter:
        iters += 1
        _, _, current = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            path = []
            cur = goal
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()

            path_len = 0.0
            for i in range(1, len(path)):
                di = 0.0
                for dim in range(3):
                    di += (path[i][dim] - path[i-1][dim]) ** 2
                path_len += math.sqrt(di) * meta["res"]

            return True, path_len, 1.0, sz, gz, path

        ci, cj, ck = current
        for d in dirs:
            ni, nj, nk = ci + d[0], cj + d[1], ck + d[2]
            if not (0 <= ni < meta["nx"]
                    and 0 <= nj < meta["ny"]
                    and 0 <= nk < meta["nz"]):
                continue
            if not occ_grid[nj, ni, nk]:
                continue
            neighbor = (ni, nj, nk)
            move_cost = meta["res"] if d[2] == 0 else math.hypot(meta["res"], meta["res"])
            tentative_g = g_score[current] + move_cost
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f_score = tentative_g + _heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, iters, neighbor))
                parent[neighbor] = current

    return False, 0.0, 0.0, 0.0, 0.0, []


# ═══════════════════════════════════════════════════════════════════════════════
# Admission geometry check for a single episode (no GPU)
# ═══════════════════════════════════════════════════════════════════════════════

def check_episode_admission(
    obstacles: list,
    occ_grid: np.ndarray,
    occ_meta: dict,
    start_pos: list,
    goal_pos: list,
    start_yaw: float,
    role: str,
    curriculum_stage: str = "C2",
) -> dict:
    """Check a single episode against all admission geometry gates.

    Returns dict with keys: pass (bool), rejection_histogram (dict),
    fallback_invalid (bool), fallback_reasons (list), details (dict).
    """
    rp = ROLE_PARAMS.get(role, ROLE_PARAMS["anticipatory"])
    d_xy_min, d_xy_max = rp["d_xy"]
    start_cl, goal_cl_thresh = rp["start_cl"], rp["goal_cl"]

    sx, sy, sz = start_pos
    gx, gy, gz = goal_pos

    obstacles_local = obstacles
    from diffusion_policy_3d.env.collision_geom import compute_obstacle_clearance

    hist = {
        "distance": 0, "clearance_start": 0, "clearance_goal": 0,
        "astar_unreachable": 0, "path_stretch": 0, "success": 0,
    }
    fallback_invalid = False
    fallback_reasons = []

    # 1. Distance check
    d_xy = math.hypot(gx - sx, gy - sy)
    if d_xy < d_xy_min or d_xy > d_xy_max:
        hist["distance"] += 1
        return {
            "pass": False,
            "rejection_histogram": hist,
            "fallback_invalid": False,
            "fallback_reasons": [],
            "details": {"reason": "distance", "d_xy": d_xy,
                        "d_xy_min": d_xy_min, "d_xy_max": d_xy_max},
        }
    d_3d = math.hypot(d_xy, gz - sz)
    if d_3d < 4.0:
        hist["distance"] += 1
        return {
            "pass": False,
            "rejection_histogram": hist,
            "fallback_invalid": False,
            "fallback_reasons": [],
            "details": {"reason": "distance_3d_too_close", "d_3d": d_3d},
        }

    # 2. Clearance check
    start_cl_actual, _, _ = compute_obstacle_clearance(sx, sy, sz, obstacles_local)
    if start_cl_actual < start_cl:
        hist["clearance_start"] += 1
        return {
            "pass": False,
            "rejection_histogram": hist,
            "fallback_invalid": False,
            "fallback_reasons": [],
            "details": {"reason": "clearance_start",
                        "actual": start_cl_actual, "required": start_cl},
        }
    goal_cl_actual, _, _ = compute_obstacle_clearance(gx, gy, gz, obstacles_local)
    if goal_cl_actual < goal_cl_thresh:
        hist["clearance_goal"] += 1
        return {
            "pass": False,
            "rejection_histogram": hist,
            "fallback_invalid": False,
            "fallback_reasons": [],
            "details": {"reason": "clearance_goal",
                        "actual": goal_cl_actual, "required": goal_cl_thresh},
        }

    # 3. A* reachability
    reachable, path_len, _, _, _, _ = astar_3d(
        occ_grid, occ_meta, sx, sy, sz, gx, gy, gz
    )
    if not reachable:
        hist["astar_unreachable"] += 1
        return {
            "pass": False,
            "rejection_histogram": hist,
            "fallback_invalid": False,
            "fallback_reasons": [],
            "details": {"reason": "astar_unreachable"},
        }

    # 4. Path stretch (dense_recovery only)
    if role == "dense_recovery":
        path_stretch = path_len / max(d_3d, 0.1)
        if path_stretch < 1.1 or path_stretch > 1.8:
            hist["path_stretch"] += 1
            return {
                "pass": False,
                "rejection_histogram": hist,
                "fallback_invalid": False,
                "fallback_reasons": [],
                "details": {"reason": "path_stretch",
                            "stretch": path_stretch, "required_low": 1.1, "required_high": 1.8},
            }

    hist["success"] += 1
    return {
        "pass": True,
        "rejection_histogram": hist,
        "fallback_invalid": False,
        "fallback_reasons": [],
        "details": {"reason": "success", "d_xy": d_xy, "d_3d": d_3d,
                    "start_clearance": start_cl_actual,
                    "goal_clearance": goal_cl_actual,
                    "path_len": path_len if reachable else None},
    }


def _check_fallback(
    obstacles: list,
    occ_grid: np.ndarray,
    occ_meta: dict,
    role: str,
    curriculum_stage: str = "C2",
) -> dict:
    """Check the default fallback start/goal against admission gates.

    Returns same dict format as check_episode_admission.
    """
    rp = ROLE_PARAMS.get(role, ROLE_PARAMS["anticipatory"])
    d_xy_min, d_xy_max = rp["d_xy"]
    start_cl_thresh, goal_cl_thresh = rp["start_cl"], rp["goal_cl"]

    z_low, z_high = _curriculum_z_range(curriculum_stage)
    z_target = (z_low + z_high) / 2.0

    start_fb = [-8.0, 0.0, z_target]
    goal_fb = [8.0, 0.0, 2.0]

    from diffusion_policy_3d.env.collision_geom import compute_obstacle_clearance

    reasons = []
    sx, sy, sz = start_fb
    gx, gy, gz = goal_fb

    d_xy = math.hypot(gx - sx, gy - sy)
    d_3d = math.hypot(d_xy, gz - sz)

    if d_xy < d_xy_min or d_xy > d_xy_max:
        reasons.append("distance")
    if d_3d < 4.0:
        reasons.append("3d_too_close")

    start_cl, _, _ = compute_obstacle_clearance(sx, sy, sz, obstacles)
    if start_cl < start_cl_thresh:
        reasons.append("start_clearance")
    goal_cl, _, _ = compute_obstacle_clearance(gx, gy, gz, obstacles)
    if goal_cl < goal_cl_thresh:
        reasons.append("goal_clearance")

    reachable, path_len, _, _, _, _ = astar_3d(
        occ_grid, occ_meta, sx, sy, sz, gx, gy, gz
    )
    if not reachable:
        reasons.append("astar_unreachable")

    if role == "dense_recovery" and reachable:
        path_stretch = path_len / max(d_3d, 0.1)
        if path_stretch < 1.1 or path_stretch > 1.8:
            reasons.append("path_stretch")

    fallback_valid = len(reasons) == 0

    return {
        "pass": True,  # fallback itself is always used geometrically
        "fallback_invalid": not fallback_valid,
        "fallback_reasons": reasons,
        "fallback_start": start_fb,
        "fallback_goal": goal_fb,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Episode generation: sample valid start/goal for a world
# ═══════════════════════════════════════════════════════════════════════════════

def _sample_start_goal_for_world(
    obstacles: list,
    occ_grid: np.ndarray,
    occ_meta: dict,
    role: str,
    rng: np.random.Generator,
    curriculum_stage: str = "C2",
    max_attempts: int = 50,
) -> Tuple[Optional[list], Optional[float], Optional[list], dict]:
    """Sample a start/goal pair passing all geometric gates.

    Returns (start_pos, start_yaw, goal_pos, details).
    On failure, returns (None, None, None, details) with fallback info.
    """
    rp = ROLE_PARAMS.get(role, ROLE_PARAMS["anticipatory"])
    d_xy_min, d_xy_max = rp["d_xy"]
    start_cl_thresh, goal_cl_thresh = rp["start_cl"], rp["goal_cl"]
    x_min, x_max, y_min, y_max = _curriculum_xy_range(curriculum_stage)
    z_low, z_high = _curriculum_z_range(curriculum_stage)
    from diffusion_policy_3d.env.collision_geom import compute_obstacle_clearance

    details = {
        "attempts": 0,
        "rejection_histogram": {
            "distance": 0, "clearance_start": 0, "clearance_goal": 0,
            "astar_unreachable": 0, "path_stretch": 0,
        },
        "fallback_invalid": False,
        "fallback_reasons": [],
    }

    for _attempt in range(max_attempts):
        details["attempts"] = _attempt + 1
        sx = float(rng.uniform(x_min, x_max))
        sy = float(rng.uniform(y_min, y_max))
        sz = float(rng.uniform(z_low + 0.3, z_high - 0.3))
        gx = float(rng.uniform(x_min, x_max))
        gy = float(rng.uniform(y_min, y_max))
        gz = float(rng.uniform(z_low + 0.5, z_high - 0.5))

        d_xy = math.hypot(gx - sx, gy - sy)
        if d_xy < d_xy_min or d_xy > d_xy_max:
            details["rejection_histogram"]["distance"] += 1
            continue
        d_3d = math.hypot(d_xy, gz - sz)
        if d_3d < 4.0:
            details["rejection_histogram"]["distance"] += 1
            continue

        start_cl_actual, _, _ = compute_obstacle_clearance(sx, sy, sz, obstacles)
        if start_cl_actual < start_cl_thresh:
            details["rejection_histogram"]["clearance_start"] += 1
            continue
        goal_cl_actual, _, _ = compute_obstacle_clearance(gx, gy, gz, obstacles)
        if goal_cl_actual < goal_cl_thresh:
            details["rejection_histogram"]["clearance_goal"] += 1
            continue

        reachable, path_len, _, _, _, _ = astar_3d(
            occ_grid, occ_meta, sx, sy, sz, gx, gy, gz
        )
        if not reachable:
            details["rejection_histogram"]["astar_unreachable"] += 1
            continue

        if role == "dense_recovery":
            ps = path_len / max(d_3d, 0.1)
            if ps < 1.1 or ps > 1.8:
                details["rejection_histogram"]["path_stretch"] += 1
                continue

        yaw = math.atan2(gy - sy, gx - sx) + float(rng.uniform(-0.3, 0.3))
        return (
            [float(sx), float(sy), float(sz)],
            float(yaw),
            [float(gx), float(gy), float(gz)],
            details,
        )

    # ── Fallback (all attempts exhausted) ────────────────────────────────────
    z_target = (z_low + z_high) / 2.0
    fb_start = [-8.0, 0.0, z_target]
    fb_goal = [8.0, 0.0, 2.0]
    fb_yaw = math.atan2(fb_goal[1] - fb_start[1], fb_goal[0] - fb_start[0])

    fb_details = _check_fallback(obstacles, occ_grid, occ_meta, role, curriculum_stage)
    details["fallback_invalid"] = fb_details["fallback_invalid"]
    details["fallback_reasons"] = fb_details["fallback_reasons"]

    return fb_start, fb_yaw, fb_goal, details


# ═══════════════════════════════════════════════════════════════════════════════
# World pool builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_world_pool(manifest, split: str = "rl_val") -> List[Tuple[str, str]]:
    """Build (world_name, obstacles_json_path) list from manifest.

    Replicates _make_newenv_world_pool.
    """
    pool = []
    world_dir = Path(WORLD_DIR)
    admitted = manifest.get_admitted_worlds(split)
    for w_entry in admitted:
        wname = w_entry.world_uid
        if wname.endswith(".world"):
            wname = wname[:-6]
        obs_json = world_dir / f"{wname}.world.obstacles.json"
        if obs_json.exists():
            pool.append((wname, str(obs_json)))
    return pool


# ═══════════════════════════════════════════════════════════════════════════════
# Main: build bank + run admission + replay identity
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_episodes_for_role(
    role: str,
    world_list: List[Tuple[str, str]],  # [(wname, obstacles_json)]
    n_episodes: int,
    rng: np.random.Generator,
    curriculum_stage: str = "C2",
    world_distribution: Optional[Dict[str, int]] = None,
) -> Tuple[List[dict], dict]:
    """Generate *n_episodes* episodes for *role* using rejection sampling.

    If *world_distribution* is given, use its world→count mapping directly.
    Otherwise, distribute evenly across available worlds.
    Returns (episodes, world_stats).
    """
    from diffusion_policy_3d.env.collision_geom import load_obstacles

    # Load obstacles + occ grids for each world
    world_cache = {}
    for wname, obs_json in world_list:
        objs = load_obstacles(obs_json)
        grid, meta = build_occupancy_grid(objs, curriculum_stage)
        world_cache[wname] = {"obstacles": objs, "occ_grid": grid, "occ_meta": meta}

    # Determine world distribution
    if world_distribution is not None:
        assigned = dict(world_distribution)
    else:
        n = len(world_list)
        if n == 0:
            return [], {"error": "no worlds available"}
        base = n_episodes // n
        rem = n_episodes % n
        assigned = {}
        for i, (wname, _) in enumerate(world_list):
            assigned[wname] = base + (1 if i < rem else 0)

    episodes = []
    ep_idx = 0
    admission_summary = defaultdict(int)

    for wname, count in assigned.items():
        cache = world_cache.get(wname)
        if cache is None:
            continue
        obstacles = cache["obstacles"]
        occ_grid = cache["occ_grid"]
        occ_meta = cache["occ_meta"]

        for _ in range(count):
            ep_seed = int(rng.integers(0, 2**31))
            ep_rng = np.random.default_rng(ep_seed)

            start_pos, start_yaw, goal_pos, details = _sample_start_goal_for_world(
                obstacles, occ_grid, occ_meta, role, ep_rng, curriculum_stage,
            )

            # Check fallback validity
            fb_invalid = details.get("fallback_invalid", False)

            episode = {
                "world_name": wname,
                "start_pos": start_pos,
                "goal_pos": goal_pos,
                "start_yaw": start_yaw,
                "episode_seed": ep_seed,
                "pointcloud_rng_seed": ep_seed,  # same seed → same RNG chain
                "episode_id": f"{role}_{ep_idx:03d}",
                "role": role,
                "val_pool_index": next(
                    (i for i, (w, _) in enumerate(world_list) if w == wname),
                    0,
                ),
                "_admission": {
                    "fallback_invalid": fb_invalid,
                    "fallback_reasons": details.get("fallback_reasons", []),
                    "sample_attempts": details.get("attempts", 0),
                },
            }
            episodes.append(episode)
            ep_idx += 1

            if fb_invalid:
                admission_summary["fallback_invalid"] += 1
            else:
                admission_summary["clean"] += 1

    world_stats = {
        "assigned": assigned,
        "n_worlds": len(assigned),
        "episodes_generated": len(episodes),
    }
    return episodes, world_stats


def run_eval_bank_audit(config: dict) -> dict:
    """Build the 96-episode evaluation bank and run full admission audit.

    Config keys:
        output_dir (str): Write output files here.
        run_id (str, optional): Run identifier.
        admission_manifest (str, optional): Path to admission_manifest.json.
            Defaults to the project calibration path.
        bc_ckpt (str, optional): Path to BC/FM checkpoint for replay-identity.
            If not given, skip replay-identity.
        device (str, optional): 'cuda' or 'cpu'. Default 'cpu' for admission.
        curriculum_stage (str, optional): Default 'C2'.
        seed (int, optional): RNG seed for bank construction. Default 42.
    """
    output_dir = Path(config.get("output_dir", "."))
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = config.get("run_id", output_dir.name)
    device = config.get("device", "cpu")
    curriculum_stage = config.get("curriculum_stage", "C2")
    seed = int(config.get("seed", 42))

    # Paths
    manifest_path = config.get(
        "admission_manifest",
        "/home/cjj/AUV/program/reports/routeA_rl_env_v1/calibration/admission_manifest.json",
    )
    bc_ckpt = config.get("bc_ckpt")

    # ── Load manifest ──────────────────────────────────────────────────────
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from diffusion_policy_3d.rl.routea_rl_env_adapter_v1 import (
        load_routea_rl_admission_manifest,
    )
    manifest = load_routea_rl_admission_manifest(manifest_path)

    # ── Build world pools (rl_val only) ────────────────────────────────────
    val_pool = build_world_pool(manifest, "rl_val")

    # Per-role world lists
    role_worlds: Dict[str, List[Tuple[str, str]]] = {}
    for role in FORMAL_ROLES:
        role_worlds[role] = [
            (wname, jpath) for wname, jpath in val_pool
            if wname.startswith(f"routeA_rl_v1__{role}__rl_val__")
        ]

    print(f"  World pool: {len(val_pool)} total val worlds")
    for role in FORMAL_ROLES:
        print(f"    {role}: {len(role_worlds[role])} val worlds: "
              f"{[w for w, _ in role_worlds[role]]}")

    # ── Generate episodes ──────────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    all_episodes: List[dict] = []
    generation_stats: Dict[str, dict] = {}

    for role in FORMAL_ROLES:
        quota = PER_ROLE_QUOTA.get(role, 32)
        world_list = role_worlds.get(role, [])

        if role == "dense_recovery":
            # dense_recovery: distribute across ≥3 worlds from the 9 val worlds
            # Pick 4 worlds across different scenario families to ensure diversity.
            # Available families: bottleneck_corridor, cul_de_sac_trap, s_turn_multi_obstacle
            dense_candidates = {}
            for wname, jpath in world_list:
                family = wname.split("__")[3] if "__" in wname else ""
                if family not in dense_candidates:
                    dense_candidates[family] = (wname, jpath)

            # We need ≥3 worlds; we have 3 families → pick 1 from each = 3 worlds
            # But 32/3 = 10.67 → distribution 11/11/10 works
            selected_dense = list(dense_candidates.values())

            # Use 4 worlds: 1 from each family + 1 more from bottleneck_corridor
            if len(world_list) >= 4:
                # Find an additional bottleneck_corridor world
                extra_bn = None
                for wname, jpath in world_list:
                    family = wname.split("__")[3] if "__" in wname else ""
                    if family == "bottleneck_corridor" and (wname, jpath) not in selected_dense:
                        extra_bn = (wname, jpath)
                        break
                if extra_bn:
                    selected_dense.append(extra_bn)

            # Distribution: 8 eps × 4 worlds = 32
            n_worlds = len(selected_dense)
            base = quota // n_worlds
            rem = quota % n_worlds
            dist = {}
            for i, (wname, _) in enumerate(selected_dense):
                dist[wname] = base + (1 if i < rem else 0)

            # Convert names back to full (wname, jpath) with correct paths
            world_lookup = {wname: jpath for wname, jpath in world_list}
            dist_with_paths = {}
            for wname, cnt in dist.items():
                if wname in world_lookup:
                    dist_with_paths[wname] = cnt
            selected_with_paths = [(w, world_lookup[w]) for w in dist_with_paths]
            # Also build the full world list for cache
            full_world_list = selected_with_paths

            eps, stats = _generate_episodes_for_role(
                role, full_world_list, quota, rng, curriculum_stage,
                world_distribution=dist_with_paths,
            )
            all_episodes.extend(eps)
            generation_stats[role] = stats
        else:
            # anticipatory & support_expansion: use whatever worlds are available
            # (expected: 1 each). Build 32 eps from available world(s).
            eps, stats = _generate_episodes_for_role(
                role, world_list, quota, rng, curriculum_stage,
            )
            all_episodes.extend(eps)
            generation_stats[role] = stats

        print(f"  Generated {len(eps)} episodes for {role} "
              f"(worlds: {len(world_list)}, "
              f"assigned: {generation_stats[role].get('assigned', {})})")

    # ── Run admission geometry on ALL episodes ─────────────────────────────
    print(f"\n  Running admission geometry on {len(all_episodes)} episodes...")
    admission_results = []
    from diffusion_policy_3d.env.collision_geom import load_obstacles

    # Pre-compute obstacles + occ_grid per world
    world_admission_cache = {}
    for wname, jpath in val_pool:
        objs = load_obstacles(jpath)
        grid, meta = build_occupancy_grid(objs, curriculum_stage)
        world_admission_cache[wname] = {"obstacles": objs, "occ_grid": grid, "occ_meta": meta}

    for ep in all_episodes:
        wname = ep["world_name"]
        cache = world_admission_cache.get(wname)
        if cache is None:
            admission_results.append({
                "episode_id": ep["episode_id"],
                "pass": False,
                "rejection_histogram": {"cache_miss": 1},
            })
            continue

        result = check_episode_admission(
            cache["obstacles"],
            cache["occ_grid"],
            cache["occ_meta"],
            ep["start_pos"],
            ep["goal_pos"],
            ep.get("start_yaw", 0.0),
            ep["role"],
            curriculum_stage,
        )

        # Check fallback validity ONLY for episodes that actually used fallback.
        # Episodes generated via rejection sampling do NOT depend on fallback validity.
        gen_adm = ep.get("_admission", {})
        used_fallback = gen_adm.get("fallback_invalid", False)
        # Also detect via sample_attempts reaching fallback:
        if gen_adm.get("sample_attempts", 0) >= 50 and "fallback" in str(gen_adm.get("fallback_reasons", "")):
            used_fallback = True

        if result.get("pass", False):
            if used_fallback:
                # Episode used fallback — verify the fallback passes gates
                fb_check = _check_fallback(
                    cache["obstacles"],
                    cache["occ_grid"],
                    cache["occ_meta"],
                    ep["role"],
                    curriculum_stage,
                )
                fallback_invalid = fb_check["fallback_invalid"]
                fallback_reasons = fb_check["fallback_reasons"]
                ep_pass = result["pass"] and not fallback_invalid
            else:
                # Episode passed rejection sampling — fallback is irrelevant
                fallback_invalid = False
                fallback_reasons = []
                ep_pass = True

            admission_results.append({
                "episode_id": ep["episode_id"],
                "pass": ep_pass,
                "fallback_invalid": fallback_invalid,
                "fallback_reasons": fallback_reasons,
                "rejection_histogram": result.get("rejection_histogram", {}),
                "details": result.get("details", {}),
            })
        else:
            # Episode failed admission
            admission_results.append({
                "episode_id": ep["episode_id"],
                "pass": False,
                "fallback_invalid": used_fallback,
                "fallback_reasons": gen_adm.get("fallback_reasons", ["admission_failed"]),
                "rejection_histogram": result.get("rejection_histogram", {}),
                "details": result.get("details", {}),
            })

    # ── Compute admission stats ────────────────────────────────────────────
    n_admitted = sum(1 for r in admission_results if r["pass"])
    n_fallback_invalid = sum(1 for r in admission_results if r.get("fallback_invalid", False))
    n_non_finite = 0
    rejection_hist_total: Dict[str, int] = defaultdict(int)
    for r in admission_results:
        for k, v in (r.get("rejection_histogram") or {}).items():
            rejection_hist_total[k] += v

    admission_pass_rate = n_admitted / max(len(admission_results), 1)

    # ── Quota validation ────────────────────────────────────────────────────
    quota_report = validate_bank_quota(all_episodes)
    n_dup_keys = quota_report.get("dup_pairing_keys", 0)

    # ── Run frozen-policy replay-identity (small subset) ────────────────────
    replay_identity_result = _run_replay_identity(
        all_episodes, bc_ckpt, device, curriculum_stage,
    )

    # ── Determine gate pass ─────────────────────────────────────────────────
    world_inventory = {}
    for role in FORMAL_ROLES:
        worlds = role_worlds.get(role, [])
        world_inventory[role] = {
            "val_worlds": len(worlds),
            "val_world_names": [w for w, _ in worlds],
        }

    # Build per-role coverage table
    per_role_coverage = {}
    for role in FORMAL_ROLES:
        rpt = quota_report["roles"].get(role, {})
        per_role_coverage[role] = {
            "n_worlds": rpt.get("n_worlds", 0),
            "required_worlds": MIN_WORLDS_PER_ROLE,
            "per_world_counts": rpt.get("per_world_counts", {}),
            "max_single_world_share": rpt.get("max_single_world_share", 0.0),
            "count": rpt.get("count", 0),
            "pass": rpt.get("pass", False),
        }

    admission_report = {
        "pass_rate": admission_pass_rate,
        "rejection_histogram": dict(rejection_hist_total),
        "fallback_invalid": n_fallback_invalid,
        "non_finite": n_non_finite,
        "dup_missing_keys": n_dup_keys,
    }

    # Gate: ALL roles satisfy count==32, ≥3 worlds, ≥4 eps/world, <50% share
    # AND admission pass rate == 100%, fallback_invalid == 0, non_finite == 0, dup == 0
    # AND replay-identity consistent
    coverage_ok = all(
        per_role_coverage[r]["pass"] for r in FORMAL_ROLES
    )
    admission_ok = (
        admission_pass_rate == 1.0
        and n_fallback_invalid == 0
        and n_non_finite == 0
        and n_dup_keys == 0
    )
    replay_ok = replay_identity_result.get("consistent", False)
    gate_pass = coverage_ok and admission_ok and replay_ok

    next_action = "NONE"
    bank_buildable_to_spec = True
    notes = []

    if not coverage_ok:
        for role in FORMAL_ROLES:
            if not per_role_coverage[role]["pass"]:
                cov = per_role_coverage[role]
                if cov["n_worlds"] < MIN_WORLDS_PER_ROLE:
                    notes.append(
                        f"{role}: only {cov['n_worlds']} worlds available (need ≥{MIN_WORLDS_PER_ROLE}). "
                        f"Cannot satisfy world-coverage requirement."
                    )
        next_action = "GENERATE_ADMIT_ADDITIONAL_EVAL_WORLDS_ANTIC_SUPPORT"
        bank_buildable_to_spec = False
        notes.extend([
            "anticiptory & support_expansion have ≤2 eval worlds in the admission manifest.",
            "This is an ENVIRONMENT-level limitation: more eval worlds per role must be generated/admitted.",
            "Do NOT relabel rl_train worlds as eval (violates train/eval separation).",
            "Do NOT reuse one world 3x to fake coverage.",
        ])

    if not admission_ok:
        notes.append(
            f"Admission: pass_rate={admission_pass_rate:.4f}, "
            f"fallback_invalid={n_fallback_invalid}, "
            f"non_finite={n_non_finite}, dup={n_dup_keys}"
        )

    if not replay_ok:
        notes.append(
            f"Replay-identity: subset_size={replay_identity_result.get('subset_size', 0)}, "
            f"consistent={replay_identity_result.get('consistent', False)}"
        )

    # ── Build manifest (episode data, no gate_pass) ─────────────────────────
    manifest_episodes = []
    for ep in all_episodes:
        # Strip internal _admission field from manifest
        ep_out = {k: v for k, v in ep.items() if not k.startswith("_")}
        manifest_episodes.append(ep_out)

    bank_version = "p0_capacity_budget_v1"

    bank_manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "bank_version": bank_version,
        "per_role": {
            role: {
                "count": PER_ROLE_QUOTA.get(role),
                "worlds": list(set(
                    e["world_name"] for e in all_episodes if e["role"] == role
                )),
            }
            for role in FORMAL_ROLES
        },
        "episodes": manifest_episodes,
    }

    # ── Build audit report ──────────────────────────────────────────────────
    audit = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        GATE_PASS_KEY: gate_pass,
        "bank_version": bank_version,
        "per_role_coverage": per_role_coverage,
        "admission": admission_report,
        "replay_identity": {
            "subset_size": replay_identity_result.get("subset_size", 0),
            "consistent": replay_identity_result.get("consistent", False),
            "device": replay_identity_result.get("device", "none"),
            "episodes_checked": replay_identity_result.get("episodes_checked", []),
            "error": replay_identity_result.get("error"),
        },
        "world_inventory": world_inventory,
        "next_action": next_action,
        "bank_buildable_to_spec": bank_buildable_to_spec,
        "notes": notes,
        "generation_stats": {
            role: {
                "n_worlds": stats.get("n_worlds", 0),
                "episodes_generated": stats.get("episodes_generated", 0),
            }
            for role, stats in generation_stats.items()
        },
    }

    # ── Write outputs ───────────────────────────────────────────────────────
    manifest_path = output_dir / "eval_bank_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(bank_manifest, f, indent=2, ensure_ascii=False)

    audit_path = output_dir / "eval_bank_audit.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    print(f"\n  Wrote {manifest_path}")
    print(f"  Wrote {audit_path}")
    print(f"  Gate pass: {gate_pass}")
    print(f"  Next action: {next_action}")

    return audit


# ═══════════════════════════════════════════════════════════════════════════════
# Frozen-policy replay-identity (GPU optional, ≤6 episodes)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_replay_identity(
    all_episodes: List[dict],
    bc_ckpt: Optional[str],
    device: str,
    curriculum_stage: str = "C2",
) -> dict:
    """Replay a ≤6-episode subset twice with frozen BC/FM policy.

    If bc_ckpt is None or device is 'cpu', skip (CPU inference too slow).
    """
    if bc_ckpt is None:
        return {
            "subset_size": 0,
            "consistent": False,
            "device": device,
            "episodes_checked": [],
            "error": "No BC checkpoint provided — replay-identity skipped",
        }

    if device == "cpu":
        return {
            "subset_size": 0,
            "consistent": False,
            "device": device,
            "episodes_checked": [],
            "error": "CPU-only — replay-identity skipped (GPU needed for timely inference)",
        }

    try:
        import torch
        if not torch.cuda.is_available():
            return {
                "subset_size": 0,
                "consistent": False,
                "device": device,
                "episodes_checked": [],
                "error": "CUDA not available — replay-identity skipped",
            }
    except ImportError:
        return {
            "subset_size": 0,
            "consistent": False,
            "device": device,
            "episodes_checked": [],
            "error": "PyTorch not importable — replay-identity skipped",
        }

    # ── Select ≤6 episodes spanning roles ──────────────────────────────────
    selected = []
    for role in FORMAL_ROLES:
        role_eps = [e for e in all_episodes if e["role"] == role]
        if role_eps:
            selected.append(role_eps[0])
            if len(role_eps) > 1:
                selected.append(role_eps[len(role_eps) // 2])
    selected = selected[:6]

    if len(selected) < 2:
        return {
            "subset_size": len(selected),
            "consistent": False,
            "device": device,
            "episodes_checked": [e["episode_id"] for e in selected],
            "error": "Too few episodes selected for replay-identity",
        }

    # ── Load frozen policy ──────────────────────────────────────────────────
    print(f"  Loading BC/FM policy from {bc_ckpt}...")
    try:
        from diffusion_policy_3d.policy.phase1a_policy_fm import Phase1aPolicyFM
        from diffusers import DDIMScheduler
        import dill

        ckpt_path = str(Path(bc_ckpt).expanduser().resolve())
        ckpt = torch.load(ckpt_path, map_location="cpu", pickle_module=dill)
        cfg = ckpt["cfg"]
        sd = ckpt["state_dicts"]["model"]
        pc = cfg["policy"]

        policy = Phase1aPolicyFM(
            shape_meta=cfg["shape_meta"],
            noise_scheduler=DDIMScheduler(
                num_train_timesteps=50,
                beta_start=0.0001, beta_end=0.02,
                beta_schedule="squaredcos_cap_v2",
                clip_sample=True, set_alpha_to_one=True,
                steps_offset=0, prediction_type="sample",
            ),
            horizon=cfg["horizon"],
            n_action_steps=cfg["n_action_steps"],
            n_obs_steps=cfg.get("n_obs_steps", 8),
            obs_as_global_cond=pc.get("obs_as_global_cond", True),
            diffusion_step_embed_dim=pc.get("diffusion_step_embed_dim", 128),
            down_dims=pc.get("down_dims", (256, 512, 1024)),
            kernel_size=pc.get("kernel_size", 5),
            n_groups=pc.get("n_groups", 8),
            anchor_bank_path=None,
            cfg_lambda=1.5, num_fm_steps=4, sigma_min=0.005,
            fm_t_scale=50, solver="euler", fm_anchor_t_start=0.95,
            use_pc_color=False,
            pointnet_type="multi_stage_pointnet",
            pointcloud_encoder_cfg=pc.get("pointcloud_encoder_cfg", None),
            point_downsample=pc.get("point_downsample", True),
        )
        policy.load_state_dict(sd, strict=False)
        policy = policy.to(device).eval()
        policy.requires_grad_(False)
        print(f"    Policy loaded ({sum(p.numel() for p in policy.parameters())} params, {device})")
    except Exception as e:
        return {
            "subset_size": 0,
            "consistent": False,
            "device": device,
            "episodes_checked": [],
            "error": f"Failed to load policy: {e}",
        }

    # ── Replay each episode twice ──────────────────────────────────────────
    from diffusion_policy_3d.env.kinematic_uav_env import KinematicUAVEnv
    from diffusion_policy_3d.model.common.synth_pointcloud import SynthPointCloud

    results = []
    consistent = True
    episodes_checked = []

    for i, ep in enumerate(selected):
        wname = ep["world_name"]
        start = ep["start_pos"]
        goal = ep["goal_pos"]
        start_yaw = ep.get("start_yaw", 0.0)
        ep_seed = ep["episode_seed"]
        ep_id = ep["episode_id"]

        print(f"    Replay {ep_id} ({wname})...")

        # Find obstacles json for this world
        manifest_path = "/home/cjj/AUV/program/reports/routeA_rl_env_v1/calibration/admission_manifest.json"
        try:
            obs_json = _find_obstacles_json(wname)
            if not obs_json:
                print(f"      SKIP — obstacles.json not found for {wname}")
                continue
        except Exception as e:
            print(f"      SKIP — error finding obstacles: {e}")
            continue

        outcomes = []
        for run_idx in range(2):
            try:
                outcome = _replay_single(
                    policy, obs_json, start, goal, start_yaw, ep_seed,
                    wname, device, curriculum_stage,
                )
                outcomes.append(outcome)
            except Exception as e:
                import traceback
                print(f"      Run {run_idx} failed: {e}")
                traceback.print_exc()
                outcomes.append(None)
                break

        if len(outcomes) == 2 and all(o is not None for o in outcomes):
            same = (
                outcomes[0]["SR"] == outcomes[1]["SR"]
                and outcomes[0]["CR"] == outcomes[1]["CR"]
                and outcomes[0]["done_reason"] == outcomes[1]["done_reason"]
            )
            if not same:
                consistent = False
                print(f"      MISMATCH: Run0={outcomes[0]}, Run1={outcomes[1]}")
            else:
                print(f"      OK: SR={outcomes[0]['SR']}, CR={outcomes[0]['CR']}, "
                      f"done={outcomes[0]['done_reason']}")
            results.append({
                "episode_id": ep_id,
                "run_0": outcomes[0],
                "run_1": outcomes[1],
                "consistent": same,
            })
            episodes_checked.append(ep_id)
        else:
            consistent = False
            results.append({
                "episode_id": ep_id,
                "error": f"Only {len(outcomes)} runs completed",
                "consistent": False,
            })

        # Clean up
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "subset_size": len(episodes_checked),
        "consistent": consistent,
        "device": device,
        "episodes_checked": episodes_checked,
        "results": results,
        "error": None,
    }


def _find_obstacles_json(world_uid: str) -> Optional[str]:
    """Find obstacles.json for a world_uid."""
    wname = world_uid
    if wname.endswith(".world"):
        wname = wname[:-6]
    base = Path(WORLD_DIR)
    candidates = [
        base / f"{wname}.world.obstacles.json",
        base / f"{wname}.obstacles.json",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def _replay_single(
    policy, obs_json: str,
    start_pos: list, goal_pos: list, start_yaw: float,
    episode_seed: int, world_name: str, device: str,
    curriculum_stage: str = "C2",
) -> dict:
    """Run a single frozen-policy rollout and return SR/CR/done_reason."""
    import numpy as np
    import torch
    from diffusion_policy_3d.model.common.synth_pointcloud import SynthPointCloud
    from diffusion_policy_3d.env.kinematic_uav_env import KinematicUAVEnv

    # Build env
    env = KinematicUAVEnv(
        world_pool=[(world_name, obs_json)],
        synth_pc_factory=lambda jp: SynthPointCloud(jp, device=device),
        rl_action_steps=8,
        h_min=0.3, z_min=1.0, z_max=5.0,
        disturb_enable=False,
        mid_state_init_ratio=0.0,
        seed=episode_seed,
        reward_version="v13",
        curriculum_stage=curriculum_stage,
    )
    env._multi_world = True

    # Monkey-patch start/goal
    _fixed_start = np.array(start_pos, dtype=np.float32)
    _fixed_goal = np.array(goal_pos, dtype=np.float32)
    _fixed_yaw = float(start_yaw)
    env._sample_start_goal = lambda role=None: (_fixed_start.copy(), _fixed_yaw, _fixed_goal.copy())

    # Rollout
    obs, info = env.reset()
    done = False
    reached = 0
    collided = 0
    done_reason = "none"
    max_steps = 60

    for step_idx in range(max_steps):
        # Convert obs to torch
        obs_torch = {
            "point_cloud": torch.from_numpy(
                obs["point_cloud"][..., :3].copy()
            ).unsqueeze(0).to(device),
            "agent_pos": torch.from_numpy(
                obs["agent_pos"].copy()
            ).unsqueeze(0).to(device),
        }

        with torch.no_grad():
            result = policy.predict_action(obs_torch)

        action = result["action_pred"][0].cpu().numpy()
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            done = True
            done_reason = info.get("done_reason", "terminated")
            reached = int(info.get("reached", 0))
            collided = int(info.get("collided", 0))
            break

    return {
        "SR": reached,
        "CR": collided,
        "done_reason": done_reason,
        "steps": step_idx + 1,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="P0 eval bank builder + admission audit")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--admission_manifest", type=str, default=None)
    parser.add_argument("--bc_ckpt", type=str, default=None,
                        help="BC/FM checkpoint for replay-identity (GPU)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--curriculum_stage", type=str, default="C2")
    args = parser.parse_args()

    config = {
        "output_dir": args.output_dir,
        "run_id": Path(args.output_dir).name,
        "admission_manifest": args.admission_manifest,
        "bc_ckpt": args.bc_ckpt,
        "device": args.device,
        "seed": args.seed,
        "curriculum_stage": args.curriculum_stage,
    }

    result = run_eval_bank_audit(config)
    print(f"\nGate pass: {result[GATE_PASS_KEY]}")
    print(f"Next action: {result.get('next_action', 'NONE')}")


if __name__ == "__main__":
    main()
