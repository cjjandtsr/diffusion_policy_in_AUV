#!/usr/bin/env python3
"""
generate_worlds_routeA_rl_v1.py — Route A RL v1 Environment World Generator
================================================================================
iDP3-UAV Route A RL Environment Design v1.2

Generates role-driven, metadata-complete Gazebo world pool for RL training.
Each world carries three-stage role semantics:
  role_generated  — assigned by generator (this script)
  role_validated  — null, filled by validator later
  role_active     — null, filled by training pipeline later

Output:
  {world_output_dir}/routeA_rl_v1__*.world              — Gazebo SDF
  {report_dir}/routeA_rl_v1_world_manifest.json          — full manifest
  {report_dir}/routeA_rl_v1_world_manifest.csv           — CSV summary
  {report_dir}/worldgen_summary.md                        — human-readable summary

Roles and generator families:
  anticipatory       → front_offset_obstacle, double_gate_progressive
  dense_recovery     → bottleneck_corridor, cul_de_sac_trap, s_turn_multi_obstacle
  support_expansion  → mixed_3d_low_density_route, xy_z_combined_bypass
  yaw_lateral_recovery → initial_yaw_offset_corridor, mid_impulse_recovery

Usage:
  # Smoke mode (small pool for verification)
  python generate_worlds_routeA_rl_v1.py --smoke

  # Full generation
  python generate_worlds_routeA_rl_v1.py --seed 20260703

  # Custom counts
  python generate_worlds_routeA_rl_v1.py \
    --n_train_per_role 40 --n_val_per_role 12 --n_test_per_role 20
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

ROLE_GENERATED = (
    "anticipatory",
    "dense_recovery",
    "support_expansion",
    "yaw_lateral_recovery",
)

SPLITS = ("rl_train", "rl_val", "rl_test", "stress_only", "historical_eval_only")

SCHEMA_VERSION = "routeA_rl_env_v1"

# Allowed role_generated values for validation
ALLOWED_ROLES = set(ROLE_GENERATED)

# Forbidden legacy world IDs (w4/w8/w11/w12/w18 must not appear)
FORBIDDEN_WORLD_IDS = {4, 8, 11, 12, 18}

# Forbidden legacy scenario names that must not appear as formal role
FORBIDDEN_LEGACY_SCENARIOS = {
    "canonical", "support", "default", "dense", "random_stress",
    "empty", "sparse", "medium_A", "medium_B", "super_dense", "special",
    "narrow_gap", "symmetric", "trap", "ood",
}

# Geometric defaults (reused from generate_worlds_ood.py)
SPAWN_WORLD = (-8.0, 0.0)
SPAWN_WORLD_Z = 0.1
GOAL_WORLD_DEFAULT = (8.0, 0.0, 2.0)
SPAWN_CLEARANCE = 5.0
GOAL_CLEARANCE = 3.0
OBS_REGION = (-14.0, 14.0)
INFLATION_RADIUS = 0.35  # drone safety radius

# Obstacle size ranges
CYL_R_MIN, CYL_R_MAX = 0.15, 2.0
CYL_H_MIN, CYL_H_MAX = 1.0, 8.0
BOX_S_MIN, BOX_S_MAX = 0.5, 4.0
BOX_H_MIN, BOX_H_MAX = 1.0, 6.0

MIN_DIST_BETWEEN_OBS = 2.0
MAX_POISSON_ATTEMPTS = 200

# Gazebo material colors
COLORS = [
    ("Gazebo/Grey",     "0.5 0.5 0.5 1"),
    ("Gazebo/Blue",     "0.447 0.624 0.812 1"),
    ("Gazebo/Red",      "0.8 0.2 0.2 1"),
    ("Gazebo/Orange",   "0.9 0.5 0.1 1"),
    ("Gazebo/White",    "0.9 0.9 0.9 1"),
    ("Gazebo/DarkGrey", "0.3 0.3 0.3 1"),
    ("Gazebo/Yellow",   "0.9 0.9 0.1 1"),
]

# Seed group tracking: each seed spawns a paired_seed_group
# Worlds with same (split, role, family, seed) share a group
_PAIRED_SEED_GROUP_COUNTER = 0


def _next_seed_group() -> str:
    global _PAIRED_SEED_GROUP_COUNTER
    _PAIRED_SEED_GROUP_COUNTER += 1
    return f"sg_{_PAIRED_SEED_GROUP_COUNTER:06d}"


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Obstacle:
    """Reuses schema from generate_worlds_ood.py."""
    name: str
    shape: str          # "cylinder" or "box"
    x: float
    y: float
    z: float            # center z (= h/2 for ground-sitting)
    yaw: float          # rotation around z-axis (radians)
    # cylinder params
    radius: float = 0.0
    height: float = 0.0
    # box params
    size_x: float = 0.0
    size_y: float = 0.0
    size_z: float = 0.0
    # for collision checking (bounding circle radius)
    bounding_r: float = 0.0


@dataclass
class RouteAWorldMetadata:
    """Three-stage role semantics metadata per design v1.2."""
    schema_version: str
    world_uid: str
    world_id: int
    split: str
    role_generated: str
    role_validated: Optional[str]
    role_active: Optional[str]
    role_version: str
    generator_family: str
    generator_seed: int
    generator_params: Dict[str, Any]
    start_world: List[float]
    goal_world: List[float]
    start_yaw: float
    goal_tolerance: float
    spawn_clearance_surface: Optional[float]
    goal_clearance_surface: Optional[float]
    world_bounds: List[float]
    obstacle_region: List[float]
    inflation_radius: float
    obstacle_stats: Dict[str, Any]
    difficulty_stats: Dict[str, Any]
    semantic_stats: Dict[str, Any]
    validation: Dict[str, Any]
    baseline_calibration: Dict[str, Any]
    disturb_profile: Optional[Dict[str, Any]]
    paired_seed_group: str
    admission_status: str


@dataclass
class WorldGenerationResult:
    """Return value from each generator family function."""
    obstacles: List[Obstacle]
    start_world: List[float]
    goal_world: List[float]
    start_yaw: float
    world_bounds: List[float]
    obstacle_region: List[float]
    generator_params: Dict[str, Any]
    disturb_profile: Optional[Dict[str, Any]]
    goal_tolerance: float


# ─────────────────────────────────────────────────────────────────────────────
# SDF generation (reused from generate_worlds_ood.py)
# ─────────────────────────────────────────────────────────────────────────────


def obstacle_to_sdf(obs: Obstacle, color_name: str, ambient: str) -> str:
    """Convert an Obstacle to Gazebo SDF model XML."""
    if obs.shape == "cylinder":
        geom_xml = f"""<cylinder>
              <radius>{obs.radius:.4f}</radius>
              <length>{obs.height:.4f}</length>
            </cylinder>"""
    else:
        geom_xml = f"""<box>
              <size>{obs.size_x:.4f} {obs.size_y:.4f} {obs.size_z:.4f}</size>
            </box>"""

    return f"""    <model name='{obs.name}'>
      <static>1</static>
      <link name='link'>
        <visual name='visual'>
          <geometry>
            {geom_xml}
          </geometry>
          <material>
            <script>
              <n>{color_name}</n>
              <uri>file://media/materials/scripts/gazebo.material</uri>
            </script>
            <ambient>{ambient}</ambient>
          </material>
        </visual>
        <collision name='collision'>
          <geometry>
            {geom_xml}
          </geometry>
          <max_contacts>10</max_contacts>
          <surface>
            <contact><ode/></contact>
            <bounce/>
            <friction>
              <ode><mu>1</mu><mu2>1</mu2></ode>
              <torsional><ode/></torsional>
            </friction>
          </surface>
        </collision>
      </link>
      <pose>{obs.x:.4f} {obs.y:.4f} {obs.z:.4f} 0 0 {obs.yaw:.4f}</pose>
    </model>
"""


def generate_world_sdf(
    obstacles: List[Obstacle],
    world_name: str = "default",
) -> str:
    """Generate a complete Gazebo .world SDF file."""
    obstacle_xml_parts = []
    for obs in obstacles:
        color_name, ambient = random.choice(COLORS)
        obstacle_xml_parts.append(obstacle_to_sdf(obs, color_name, ambient))
    obstacle_xml = "\n".join(obstacle_xml_parts)

    return f"""<?xml version='1.0'?>
<sdf version='1.6'>
  <world name='{world_name}'>
    <!-- Sun light -->
    <light name='sun' type='directional'>
      <cast_shadows>1</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <!-- Ground plane -->
    <model name='ground_plane'>
      <static>1</static>
      <link name='link'>
        <collision name='collision'>
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <surface>
            <contact>
              <collide_bitmask>65535</collide_bitmask>
              <ode/>
            </contact>
            <friction>
              <ode>
                <mu>100</mu>
                <mu2>50</mu2>
              </ode>
              <torsional><ode/></torsional>
            </friction>
            <bounce/>
          </surface>
          <max_contacts>10</max_contacts>
        </collision>
        <visual name='visual'>
          <cast_shadows>0</cast_shadows>
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <material>
            <script>
              <uri>file://media/materials/scripts/gazebo.material</uri>
              <n>Gazebo/Grey</n>
            </script>
          </material>
        </visual>
      </link>
    </model>

    <!-- Environment -->
    <magnetic_field>6e-06 2.3e-05 -4.2e-05</magnetic_field>
    <atmosphere type='adiabatic'/>
    <physics name='default_physics' default='0' type='ode'>
      <gravity>0 0 -9.8</gravity>
      <max_step_size>0.004</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>250</real_time_update_rate>
    </physics>
    <scene>
      <ambient>0.4 0.4 0.4 1</ambient>
      <background>0.7 0.7 0.7 1</background>
      <shadows>1</shadows>
    </scene>
    <wind/>
    <spherical_coordinates>
      <surface_model>EARTH_WGS84</surface_model>
      <latitude_deg>0</latitude_deg>
      <longitude_deg>0</longitude_deg>
      <elevation>0</elevation>
      <heading_deg>0</heading_deg>
    </spherical_coordinates>

    <!-- ============ OBSTACLES ({len(obstacles)}) ============ -->
{obstacle_xml}
    <!-- GUI camera -->
    <gui fullscreen='0'>
      <camera name='user_camera'>
        <pose>-15 0 18 0 0.85 0.15</pose>
        <view_controller>orbit</view_controller>
        <projection_type>perspective</projection_type>
      </camera>
    </gui>
  </world>
</sdf>
"""


# ─────────────────────────────────────────────────────────────────────────────
# UID and file helpers
# ─────────────────────────────────────────────────────────────────────────────


def make_world_uid(
    split: str,
    role_generated: str,
    generator_family: str,
    world_id: int,
) -> str:
    """Generate unique world UID with role_generated snapshot."""
    return (
        f"routeA_rl_v1__{role_generated}__{split}"
        f"__{generator_family}__{world_id:06d}"
    )


def make_world_filename(world_uid: str) -> str:
    """Generate .world filename from UID."""
    return f"{world_uid}.world"


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON with consistent formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Poisson disk sampling (reused from generate_worlds_ood.py)
# ─────────────────────────────────────────────────────────────────────────────


def poisson_disk_sample(
    n: int,
    bounds: Tuple[float, float],
    min_dist: float,
    exclusion_zones: List[Tuple[float, float, float]],
    rng: random.Random,
    existing: Optional[List[Tuple[float, float]]] = None,
    max_attempts: int = MAX_POISSON_ATTEMPTS,
) -> Tuple[List[Tuple[float, float]], int]:
    """Sample n points with minimum distance between points and away from exclusion zones."""
    points = list(existing) if existing else []

    for _ in range(n):
        placed = False
        for _ in range(max_attempts):
            x = rng.uniform(bounds[0], bounds[1])
            y = rng.uniform(bounds[0], bounds[1])

            ok = True
            for ex, ey, er in exclusion_zones:
                if math.hypot(x - ex, y - ey) < er:
                    ok = False
                    break
            if not ok:
                continue

            for px, py in points:
                if math.hypot(x - px, y - py) < min_dist:
                    ok = False
                    break
            if not ok:
                continue

            points.append((x, y))
            placed = True
            break

        if not placed:
            break

    n_existing = len(existing) if existing else 0
    n_placed = len(points) - n_existing
    return points[n_existing:], n_placed


# ─────────────────────────────────────────────────────────────────────────────
# Obstacle factory helpers
# ─────────────────────────────────────────────────────────────────────────────


def make_cylinder(
    name: str,
    x: float,
    y: float,
    radius: float,
    height: float,
    yaw: float = 0.0,
) -> Obstacle:
    """Create a cylinder obstacle sitting on ground."""
    return Obstacle(
        name=name, shape="cylinder",
        x=x, y=y, z=height / 2.0, yaw=yaw,
        radius=radius, height=height,
        bounding_r=radius,
    )


def make_box(
    name: str,
    x: float,
    y: float,
    size_x: float,
    size_y: float,
    size_z: float,
    yaw: float = 0.0,
) -> Obstacle:
    """Create a box obstacle sitting on ground."""
    return Obstacle(
        name=name, shape="box",
        x=x, y=y, z=size_z / 2.0, yaw=yaw,
        size_x=size_x, size_y=size_y, size_z=size_z,
        bounding_r=math.hypot(size_x, size_y) / 2.0,
    )


def make_random_obstacle(
    name: str, x: float, y: float, rng: random.Random,
) -> Obstacle:
    """Generate a random cylinder or box obstacle at (x, y)."""
    is_cylinder = rng.random() < 0.667  # 2:1 cylinder:box
    yaw = rng.uniform(0, math.pi)

    if is_cylinder:
        r = rng.uniform(CYL_R_MIN, CYL_R_MAX)
        h = rng.uniform(CYL_H_MIN, CYL_H_MAX)
        return make_cylinder(name, x, y, r, h, yaw)
    else:
        sx = rng.uniform(BOX_S_MIN, BOX_S_MAX)
        sy = rng.uniform(BOX_S_MIN, BOX_S_MAX)
        sz = rng.uniform(BOX_H_MIN, BOX_H_MAX)
        return make_box(name, x, y, sx, sy, sz, yaw)


# ─────────────────────────────────────────────────────────────────────────────
# Difficulty metrics (reused from generate_worlds_ood.py)
# ─────────────────────────────────────────────────────────────────────────────


def compute_difficulty_metrics(
    obstacles: List[Obstacle],
    start_world: List[float],
    goal_world: List[float],
) -> Dict[str, Any]:
    """Compute TO/VO/AOL difficulty metrics."""
    sx, sy = start_world[0], start_world[1]
    gx, gy = goal_world[0], goal_world[1]

    dx, dy = gx - sx, gy - sy
    path_len = math.hypot(dx, dy)
    if path_len < 0.01:
        return {"TO": 0.0, "VO": 1.0, "AOL": 0.0, "n_obstacles": len(obstacles)}

    # TO: angular deviation
    to_values = []
    for obs in obstacles:
        odx, ody = obs.x - sx, obs.y - sy
        obs_dist = math.hypot(odx, ody)
        if obs_dist < 0.01:
            to_values.append(math.pi / 2)
            continue
        cos_angle = (odx * dx + ody * dy) / (obs_dist * path_len)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle = math.acos(cos_angle)
        to_values.append(angle)

    TO = float(sum(to_values) / len(to_values)) if to_values else 0.0

    # VO: line-of-sight visibility
    los_blocked = False
    for obs in obstacles:
        r_eff = obs.bounding_r
        if r_eff <= 0:
            continue
        ox_rel, oy_rel = obs.x - sx, obs.y - sy
        t = (ox_rel * dx + oy_rel * dy) / (path_len * path_len)
        t = max(0.0, min(1.0, t))
        closest_x = sx + t * dx
        closest_y = sy + t * dy
        dist_to_seg = math.hypot(obs.x - closest_x, obs.y - closest_y)
        if dist_to_seg < r_eff:
            los_blocked = True
            break

    VO = 0.0 if los_blocked else 1.0

    area = (OBS_REGION[1] - OBS_REGION[0]) ** 2
    AOL = float(len(obstacles) / area) if area > 0 else 0.0

    return {
        "TO": round(TO, 4),
        "VO": round(VO, 4),
        "AOL": round(AOL, 4),
        "n_obstacles": len(obstacles),
    }


def compute_obstacle_stats(obstacles: List[Obstacle]) -> Dict[str, Any]:
    """Compute obstacle statistics."""
    n_cyl = sum(1 for o in obstacles if o.shape == "cylinder")
    n_box = sum(1 for o in obstacles if o.shape == "box")
    cyl_radii = [o.radius for o in obstacles if o.shape == "cylinder"]
    cyl_heights = [o.height for o in obstacles if o.shape == "cylinder"]
    box_volumes = [
        o.size_x * o.size_y * o.size_z for o in obstacles if o.shape == "box"
    ]
    return {
        "n_total": len(obstacles),
        "n_cylinder": n_cyl,
        "n_box": n_box,
        "cyl_radius_mean": round(sum(cyl_radii) / len(cyl_radii), 4) if cyl_radii else 0.0,
        "cyl_radius_range": [round(min(cyl_radii), 4), round(max(cyl_radii), 4)] if cyl_radii else [0, 0],
        "cyl_height_mean": round(sum(cyl_heights) / len(cyl_heights), 4) if cyl_heights else 0.0,
        "box_volume_mean": round(sum(box_volumes) / len(box_volumes), 4) if box_volumes else 0.0,
    }


def compute_semantic_stats(
    obstacles: List[Obstacle],
    start_world: List[float],
    goal_world: List[float],
) -> Dict[str, Any]:
    """Compute semantic statistics about obstacle layout."""
    if not obstacles:
        return {"path_crossing_count": 0, "mean_path_offset": 0.0}

    sx, sy = start_world[0], start_world[1]
    gx, gy = goal_world[0], goal_world[1]
    dx, dy = gx - sx, gy - sy
    path_len = math.hypot(dx, dy)

    path_crossing = 0
    path_offsets = []
    for obs in obstacles:
        # perpendicular distance from obstacle center to start→goal line
        ox_rel, oy_rel = obs.x - sx, obs.y - sy
        cross = abs(ox_rel * dy - oy_rel * dx) / path_len if path_len > 0 else 0.0
        path_offsets.append(cross)
        if cross < obs.bounding_r:
            path_crossing += 1

    return {
        "path_crossing_count": path_crossing,
        "mean_path_offset": round(sum(path_offsets) / len(path_offsets), 4),
        "min_path_offset": round(min(path_offsets), 4),
        "max_path_offset": round(max(path_offsets), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Role family generators
# ─────────────────────────────────────────────────────────────────────────────


# --- anticipatory: front_offset_obstacle ---

def gen_front_offset_obstacle(
    rng: random.Random,
    variant: int = 0,
) -> WorldGenerationResult:
    """
    Single obstacle placed offset from the direct start→goal line,
    requiring the drone to anticipate and adjust trajectory early.
    Variant controls offset direction and magnitude.
    """
    offset_signs = [1, -1, 1, -1]
    offset_mags = [3.0, 4.0, 5.0, 2.5]
    idx = variant % len(offset_signs)
    sign = offset_signs[idx]
    offset = offset_mags[idx]

    start = [-8.0, 0.0, 2.0]
    goal = [8.0, 0.0, 2.0]

    # Obstacle at x=0, offset in y
    obs_x = 0.0
    obs_y = sign * offset

    obstacles = [
        make_cylinder(
            name="front_offset_cyl",
            x=obs_x, y=obs_y,
            radius=rng.uniform(0.8, 1.5),
            height=rng.uniform(4.0, 7.0),
        ),
    ]

    return WorldGenerationResult(
        obstacles=obstacles,
        start_world=start,
        goal_world=goal,
        start_yaw=0.0,
        world_bounds=[-14.0, 14.0, -14.0, 14.0],
        obstacle_region=list(OBS_REGION),
        generator_params={
            "family": "front_offset_obstacle",
            "variant": variant,
            "offset_y": obs_y,
        },
        disturb_profile=None,
        goal_tolerance=0.5,
    )


# --- anticipatory: double_gate_progressive ---

def gen_double_gate_progressive(
    rng: random.Random,
    variant: int = 0,
) -> WorldGenerationResult:
    """
    Two gates placed progressively along the path. The second gate is narrower
    or offset differently, requiring the drone to plan two moves ahead.
    """
    gate_widths = [3.0, 2.5, 2.0, 3.5]
    gate_offsets = [0.0, 0.5, -0.5, 0.0]  # y-offset of second gate relative to first
    idx = variant % len(gate_widths)
    gate_w = gate_widths[idx]
    gate_offset_y = gate_offsets[idx]

    start = [-8.0, 0.0, 2.0]
    goal = [8.0, gate_offset_y, 2.0]

    wall_thickness = 0.5
    wall_height = 5.0
    half_gap = gate_w / 2.0

    obstacles = []

    # Gate posts: thin walls on each side of the gap
    post_depth = wall_thickness  # size_y = thin post, not wide wall
    post_width = wall_thickness  # size_x

    # Gate 1 at x=-3
    for side, sign in enumerate([-1, 1]):
        wall_y = sign * (half_gap + post_depth / 2.0)
        obstacles.append(make_box(
            name=f"gate1_{['l', 'r'][side]}",
            x=-3.0, y=wall_y,
            size_x=post_width, size_y=post_depth, size_z=wall_height,
        ))

    # Gate 2 at x=3, possibly offset and narrower
    gate2_half_gap = half_gap * 0.85  # slightly narrower than gate 1
    gate2_center_y = gate_offset_y
    for side, sign in enumerate([-1, 1]):
        wall_y = gate2_center_y + sign * (gate2_half_gap + post_depth / 2.0)
        obstacles.append(make_box(
            name=f"gate2_{['l', 'r'][side]}",
            x=3.0, y=wall_y,
            size_x=post_width, size_y=post_depth, size_z=wall_height,
        ))

    return WorldGenerationResult(
        obstacles=obstacles,
        start_world=start,
        goal_world=goal,
        start_yaw=0.0,
        world_bounds=[-14.0, 14.0, -14.0, 14.0],
        obstacle_region=list(OBS_REGION),
        generator_params={
            "family": "double_gate_progressive",
            "variant": variant,
            "gate_width": gate_w,
            "gate2_offset_y": gate_offset_y,
        },
        disturb_profile=None,
        goal_tolerance=0.5,
    )


# --- dense_recovery: bottleneck_corridor ---

def gen_bottleneck_corridor(
    rng: random.Random,
    variant: int = 0,
) -> WorldGenerationResult:
    """
    Corridor that narrows to a bottleneck, forcing the drone to squeeze through
    or recover if it enters too fast / at a bad angle.
    """
    bottleneck_widths = [1.5, 1.2, 1.0, 1.8]  # meters
    idx = variant % len(bottleneck_widths)
    bn_width = bottleneck_widths[idx]

    start = [-8.0, 0.0, 2.0]
    goal = [8.0, 0.0, 2.0]

    wall_height = 5.0
    wall_thickness = 0.5
    corridor_length = 10.0  # entrance width
    half_bn = bn_width / 2.0

    obstacles = []

    # Funnel walls: start wide, narrow to bottleneck entrance at x=-gap
    # Gap between entrance and exit sections prevents cylinder overlap
    gap = 0.8  # meters between last entrance cylinder and first exit cylinder
    n_segments = 8  # increased from 5 for higher density (design doc: 25-45 obstacles)
    for i in range(n_segments):
        t = i / (n_segments - 1)  # 0 at entrance, 1 at bottleneck
        x = -4.0 + t * (4.0 - gap / 2.0)  # from x=-4 to x=-gap/2
        # y width narrows from ±3.5 to ±half_bn
        y_width = 3.5 - t * (3.5 - half_bn)
        for sign in [-1, 1]:
            obstacles.append(make_cylinder(
                name=f"funnel_{['l', 'r'][0 if sign==-1 else 1]}_{i}",
                x=x,
                y=sign * (y_width + 0.3),
                radius=rng.uniform(0.25, 0.40),
                height=rng.uniform(4.0, 6.0),
            ))

    # After bottleneck, corridor widens again
    for i in range(n_segments):
        t = i / (n_segments - 1)
        x = gap / 2.0 + t * (4.0 - gap / 2.0)  # from x=+gap/2 to x=4
        y_width = half_bn + t * (3.5 - half_bn)
        for sign in [-1, 1]:
            obstacles.append(make_cylinder(
                name=f"funnel_out_{['l', 'r'][0 if sign==-1 else 1]}_{i}",
                x=x,
                y=sign * (y_width + 0.3),
                radius=rng.uniform(0.25, 0.40),
                height=rng.uniform(4.0, 6.0),
            ))

    # Scatter obstacles for density: 25-45 target (design doc §2.3)
    n_scatter = 8
    for i in range(n_scatter):
        x = rng.uniform(-6.0, 6.0)
        y = rng.uniform(-5.0, 5.0)
        # Avoid placing directly in the corridor path
        if abs(x) < 1.0 and abs(y) < half_bn + 0.8:
            y = (half_bn + 0.8 + rng.uniform(0.5, 2.5)) * (1 if y >= 0 else -1)
        obstacles.append(make_cylinder(
            name=f"bn_scatter_{i}",
            x=x, y=y,
            radius=rng.uniform(0.2, 0.6),
            height=rng.uniform(3.0, 6.0),
        ))

    return WorldGenerationResult(
        obstacles=obstacles,
        start_world=start,
        goal_world=goal,
        start_yaw=0.0,
        world_bounds=[-14.0, 14.0, -14.0, 14.0],
        obstacle_region=list(OBS_REGION),
        generator_params={
            "family": "bottleneck_corridor",
            "variant": variant,
            "bottleneck_width": bn_width,
        },
        disturb_profile=None,
        goal_tolerance=0.5,
    )


# --- dense_recovery: cul_de_sac_trap ---

def gen_cul_de_sac_trap(
    rng: random.Random,
    variant: int = 0,
) -> WorldGenerationResult:
    """
    U-shaped dead-end trap on the direct path. Drone must recognize the trap
    and go around rather than entering.
    """
    depths = [4.0, 5.0, 3.0, 6.0]
    openings = [3.0, 2.5, 3.5, 2.0]
    idx = variant % len(depths)
    depth = depths[idx]
    opening = openings[idx]

    start = [-10.0, 0.0, 2.0]
    goal = [8.0, 0.0, 2.0]

    wall_thickness = 0.5
    wall_height = 4.0

    obstacles = []

    # Side walls extending from opening toward goal
    side_center_x = depth / 2.0
    for side, sign in enumerate([-1, 1]):
        obstacles.append(make_box(
            name=f"trap_side_{['l', 'r'][side]}",
            x=side_center_x,
            y=sign * (opening / 2.0 + wall_thickness / 2.0),
            size_x=depth, size_y=wall_thickness, size_z=wall_height,
        ))

    # Back wall
    obstacles.append(make_box(
        name="trap_back",
        x=depth, y=0.0,
        size_x=wall_thickness, size_y=opening, size_z=wall_height,
    ))

    # Funnel entrance flares: walls extending toward start
    funnel_length = depth * 0.5
    funnel_flare = opening * 0.4
    funnel_center_x = -funnel_length / 2.0
    for side, sign in enumerate([-1, 1]):
        obstacles.append(make_box(
            name=f"trap_funnel_{['l', 'r'][side]}",
            x=funnel_center_x,
            y=sign * (opening / 2.0 + funnel_flare / 2.0),
            size_x=funnel_length, size_y=wall_thickness, size_z=wall_height,
        ))

    # Scatter obstacles for density: 25-45 target (design doc §2.3)
    # Place cylinders around the trap entrance and perimeter — avoid blocking
    # the escape path inside the cavity
    n_scatter = 25
    for i in range(n_scatter):
        x = rng.uniform(-4.0, depth + 2.0)
        y = rng.uniform(-(opening + 5.0), opening + 5.0)
        # Avoid placing inside the trap cavity (drone needs an escape corridor)
        if 0.0 < x < depth and abs(y) < opening / 2.0 + 0.3:
            continue
        obstacles.append(make_cylinder(
            name=f"trap_scatter_{i}",
            x=x, y=y,
            radius=rng.uniform(0.2, 0.6),
            height=rng.uniform(3.0, 5.0),
        ))

    return WorldGenerationResult(
        obstacles=obstacles,
        start_world=start,
        goal_world=goal,
        start_yaw=0.0,
        world_bounds=[-14.0, 14.0, -14.0, 14.0],
        obstacle_region=list(OBS_REGION),
        generator_params={
            "family": "cul_de_sac_trap",
            "variant": variant,
            "depth": depth,
            "opening": opening,
        },
        disturb_profile=None,
        goal_tolerance=0.5,
    )


# --- dense_recovery: s_turn_multi_obstacle ---

def gen_s_turn_multi_obstacle(
    rng: random.Random,
    variant: int = 0,
) -> WorldGenerationResult:
    """
    S-turn path with multiple staggered obstacles. Drone must weave through
    and recover if it clips an obstacle. Two rows of obstacles with different
    amplitudes create a dense weave pattern (design doc: 25-45 obstacles).
    """
    amplitudes = [2.5, 3.0, 2.0, 3.5]
    idx = variant % len(amplitudes)
    amp = amplitudes[idx]

    start = [-8.0, 0.0, 2.0]
    goal = [8.0, 0.0, 2.0]

    obstacles = []
    # Primary S-curve row
    n_obs = 16
    for i in range(n_obs):
        t = i / (n_obs - 1)
        x = -6.0 + t * 12.0  # spread along x
        y = amp * math.sin(t * 3.0 * math.pi)  # S-curve offset
        y += rng.uniform(-0.4, 0.4)
        r = rng.uniform(0.3, 0.7)
        h = rng.uniform(3.0, 6.0)
        obstacles.append(make_cylinder(
            name=f"s_turn_obs_{i}",
            x=x, y=y, radius=r, height=h,
        ))

    # Secondary S-curve row (offset amplitude, slightly different phase)
    amp2 = amp * 0.65
    n_obs2 = 12
    for i in range(n_obs2):
        t = i / (n_obs2 - 1)
        x = -5.0 + t * 10.0
        y = amp2 * math.sin(t * 2.5 * math.pi + math.pi / 3)
        y += rng.uniform(-0.3, 0.3)
        r = rng.uniform(0.2, 0.5)
        h = rng.uniform(3.0, 5.0)
        obstacles.append(make_cylinder(
            name=f"s_turn_obs2_{i}",
            x=x, y=y, radius=r, height=h,
        ))

    return WorldGenerationResult(
        obstacles=obstacles,
        start_world=start,
        goal_world=goal,
        start_yaw=0.0,
        world_bounds=[-14.0, 14.0, -14.0, 14.0],
        obstacle_region=list(OBS_REGION),
        generator_params={
            "family": "s_turn_multi_obstacle",
            "variant": variant,
            "amplitude": amp,
            "n_obstacles": n_obs + n_obs2,
        },
        disturb_profile=None,
        goal_tolerance=0.5,
    )


# --- support_expansion: mixed_3d_low_density_route ---

def gen_mixed_3d_low_density_route(
    rng: random.Random,
    variant: int = 0,
) -> WorldGenerationResult:
    """
    Mixed 3D obstacle field at low density. Obstacles of varying heights
    create a 3D routing problem requiring combined XY+Z maneuvering.
    """
    start = [-8.0, 0.0, 2.0]
    goal = [8.0, 0.0, 2.0]

    obstacles = []
    n_obs = 6 + variant % 3  # 6-8 obstacles

    # Low obstacles (can fly over)
    low_positions = [
        (-4.0, -1.5), (0.0, 2.5), (5.0, 1.0),
    ]
    for i, (px, py) in enumerate(low_positions[:3]):
        obstacles.append(make_cylinder(
            name=f"low_obs_{i}",
            x=px, y=py,
            radius=rng.uniform(0.3, 0.7),  # smaller to avoid overlap
            height=rng.uniform(0.8, 1.5),  # low enough to fly over
        ))

    # Tall obstacles (must go around) — well-separated from low obs
    tall_positions = [
        (-2.0, -2.5), (2.0, 2.0), (6.0, -2.0),
    ]
    for i, (px, py) in enumerate(tall_positions[:3]):
        obstacles.append(make_cylinder(
            name=f"tall_obs_{i}",
            x=px, y=py,
            radius=rng.uniform(0.6, 1.5),
            height=rng.uniform(4.0, 7.0),  # must go around
        ))

    # Medium obstacles (marginal height — choice: over or around?)
    medium_positions = [
        (-5.0, 2.5), (1.0, -1.5),
    ]
    for i, (px, py) in enumerate(medium_positions[:2]):
        obstacles.append(make_box(
            name=f"med_obs_{i}",
            x=px, y=py,
            size_x=rng.uniform(0.8, 1.5),
            size_y=rng.uniform(0.8, 1.5),
            size_z=rng.uniform(2.5, 3.5),  # marginal: can the drone fly over?
        ))

    return WorldGenerationResult(
        obstacles=obstacles,
        start_world=start,
        goal_world=goal,
        start_yaw=0.0,
        world_bounds=[-14.0, 14.0, -14.0, 14.0],
        obstacle_region=list(OBS_REGION),
        generator_params={
            "family": "mixed_3d_low_density_route",
            "variant": variant,
            "n_obstacles": len(obstacles),
        },
        disturb_profile=None,
        goal_tolerance=0.5,
    )


# --- support_expansion: xy_z_combined_bypass ---

def gen_xy_z_combined_bypass(
    rng: random.Random,
    variant: int = 0,
) -> WorldGenerationResult:
    """
    Combined XY and Z obstacle layout. Horizontal walls block lateral paths
    while overhanging obstacles force vertical decisions. Drone must combine
    lateral bypass with vertical ascent/descent.
    """
    start = [-8.0, 0.0, 2.0]
    goal = [8.0, 0.0, 2.0]

    obstacles = []

    # Horizontal wall at y=+2 (blocks upper path)
    obstacles.append(make_box(
        name="wall_upper",
        x=0.0, y=2.5,
        size_x=8.0, size_y=0.5, size_z=3.0,  # height=3: can fly over if go high
    ))

    # Horizontal wall at y=-2 (blocks lower path)
    obstacles.append(make_box(
        name="wall_lower",
        x=1.0, y=-2.5,
        size_x=8.0, size_y=0.5, size_z=5.0,  # height=5: harder to fly over
    ))

    # Overhanging obstacle at center (ceiling-like)
    obstacles.append(make_box(
        name="overhang",
        x=0.0, y=0.0,
        size_x=2.0, size_y=3.0, size_z=0.5,
    ))
    # Set its z so it hangs at z=4 (supported by invisible structure, or
    # practically: the box sits at z=4 with size_z=0.5, so it spans 3.75–4.25)
    obstacles[-1].z = 4.0  # override: this is an elevated obstacle
    obstacles[-1].size_z = 0.5

    # A few cylinders in clear space to create combined XY+Z challenges
    # Wall_upper at y=2.0-3.0, wall_lower at y=-2.75 to -2.25, overhang at 0,0 (2×3)
    for i, (px, py) in enumerate([(-3.0, 0.0), (3.0, 0.0), (1.5, 1.0)]):
        obstacles.append(make_cylinder(
            name=f"bypass_cyl_{i}",
            x=px, y=py,
            radius=rng.uniform(0.3, 0.7),
            height=rng.uniform(2.0, 4.0),
        ))

    return WorldGenerationResult(
        obstacles=obstacles,
        start_world=start,
        goal_world=goal,
        start_yaw=0.0,
        world_bounds=[-14.0, 14.0, -14.0, 14.0],
        obstacle_region=list(OBS_REGION),
        generator_params={
            "family": "xy_z_combined_bypass",
            "variant": variant,
            "n_obstacles": len(obstacles),
        },
        disturb_profile=None,
        goal_tolerance=0.5,
    )


# --- yaw_lateral_recovery: initial_yaw_offset_corridor ---

def gen_initial_yaw_offset_corridor(
    rng: random.Random,
    variant: int = 0,
) -> WorldGenerationResult:
    """
    Corridor with initial yaw misalignment. The drone starts facing away from
    the corridor entrance and must recover orientation to enter.
    """
    yaw_offsets = [math.pi / 2, -math.pi / 2, math.pi * 0.75, -math.pi * 0.75]
    idx = variant % len(yaw_offsets)
    start_yaw = yaw_offsets[idx]

    start = [-8.0, 0.0, 2.0]
    goal = [8.0, 0.0, 2.0]

    corridor_width = 2.5 + variant * 0.3
    half_w = corridor_width / 2.0
    wall_length = 14.0
    wall_height = 5.0
    wall_thickness = 0.5

    obstacles = []
    for side, sign in enumerate([-1, 1]):
        obstacles.append(make_box(
            name=f"corridor_wall_{['l', 'r'][side]}",
            x=0.0, y=sign * (half_w + wall_thickness / 2.0),
            size_x=wall_length, size_y=wall_thickness, size_z=wall_height,
        ))

    return WorldGenerationResult(
        obstacles=obstacles,
        start_world=start,
        goal_world=goal,
        start_yaw=start_yaw,
        world_bounds=[-14.0, 14.0, -14.0, 14.0],
        obstacle_region=list(OBS_REGION),
        generator_params={
            "family": "initial_yaw_offset_corridor",
            "variant": variant,
            "start_yaw": start_yaw,
            "corridor_width": corridor_width,
        },
        disturb_profile=None,
        goal_tolerance=0.5,
    )


# --- yaw_lateral_recovery: mid_impulse_recovery ---

def gen_mid_impulse_recovery(
    rng: random.Random,
    variant: int = 0,
) -> WorldGenerationResult:
    """
    Mid-path surprise obstacle requiring lateral impulse recovery.
    A clear path suddenly has an obstacle appearing off the direct line,
    requiring a quick lateral maneuver.
    """
    impulse_x_positions = [1.0, 2.0, 0.0, 3.0]
    impulse_y_offsets = [1.5, -1.5, 2.0, -2.0]
    idx = variant % len(impulse_x_positions)
    imp_x = impulse_x_positions[idx]
    imp_y = impulse_y_offsets[idx]

    start = [-8.0, 0.0, 2.0]
    goal = [8.0, 0.0, 2.0]

    obstacles = []

    # Guiding walls that make the drone think it's a straight corridor
    # Then an impulse obstacle appears
    guide_length = 6.0
    guide_half_w = 3.0
    for side, sign in enumerate([-1, 1]):
        obstacles.append(make_box(
            name=f"guide_wall_{['l', 'r'][side]}",
            x=-3.0, y=sign * (guide_half_w + 0.25),
            size_x=guide_length, size_y=0.5, size_z=4.0,
        ))

    # Impulse obstacle: suddenly in the path
    obstacles.append(make_cylinder(
        name="impulse_obs",
        x=imp_x, y=imp_y,
        radius=rng.uniform(0.5, 1.0),
        height=rng.uniform(3.0, 5.0),
    ))

    # Secondary obstacle further ahead to complicate recovery
    obstacles.append(make_cylinder(
        name="recovery_obs",
        x=imp_x + 3.0, y=-imp_y * 0.5,
        radius=rng.uniform(0.4, 0.8),
        height=rng.uniform(2.0, 4.0),
    ))

    return WorldGenerationResult(
        obstacles=obstacles,
        start_world=start,
        goal_world=goal,
        start_yaw=0.0,
        world_bounds=[-14.0, 14.0, -14.0, 14.0],
        obstacle_region=list(OBS_REGION),
        generator_params={
            "family": "mid_impulse_recovery",
            "variant": variant,
            "impulse_x": imp_x,
            "impulse_y": imp_y,
        },
        disturb_profile=None,
        goal_tolerance=0.5,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Generator family registry
# ─────────────────────────────────────────────────────────────────────────────

# Maps role → list of (family_name, generator_function)
GENERATOR_REGISTRY: Dict[str, List[Tuple[str, callable]]] = {
    "anticipatory": [
        ("front_offset_obstacle", gen_front_offset_obstacle),
        ("double_gate_progressive", gen_double_gate_progressive),
    ],
    "dense_recovery": [
        ("bottleneck_corridor", gen_bottleneck_corridor),
        ("cul_de_sac_trap", gen_cul_de_sac_trap),
        ("s_turn_multi_obstacle", gen_s_turn_multi_obstacle),
    ],
    "support_expansion": [
        ("mixed_3d_low_density_route", gen_mixed_3d_low_density_route),
        ("xy_z_combined_bypass", gen_xy_z_combined_bypass),
    ],
    "yaw_lateral_recovery": [
        ("initial_yaw_offset_corridor", gen_initial_yaw_offset_corridor),
        ("mid_impulse_recovery", gen_mid_impulse_recovery),
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# World generation orchestration
# ─────────────────────────────────────────────────────────────────────────────


def generate_world_pool(
    rng: random.Random,
    world_output_dir: Path,
    report_dir: Path,
    n_train_per_role: int,
    n_val_per_role: int,
    n_test_per_role: int,
) -> Tuple[List[RouteAWorldMetadata], Dict[str, Any]]:
    """
    Generate the complete world pool and write all output files.

    Returns (metadata_list, summary_dict).
    """
    world_output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    split_counts = {
        "rl_train": n_train_per_role,
        "rl_val": n_val_per_role,
        "rl_test": n_test_per_role,
    }

    metadata_list: List[RouteAWorldMetadata] = []
    world_id_counter = 0
    failures: List[Dict[str, Any]] = []

    for role in ROLE_GENERATED:
        families = GENERATOR_REGISTRY[role]
        for split, n_worlds in split_counts.items():
            if n_worlds <= 0:
                continue
            # Distribute worlds across families for this role
            n_families = len(families)
            base_per_family = n_worlds // n_families
            remainder = n_worlds % n_families

            for fi, (family_name, gen_func) in enumerate(families):
                n_this_family = base_per_family + (1 if fi < remainder else 0)
                for vi in range(n_this_family):
                    try:
                        result = gen_func(rng, variant=vi)
                    except Exception as e:
                        failures.append({
                            "role": role,
                            "split": split,
                            "family": family_name,
                            "variant": vi,
                            "error": str(e),
                        })
                        continue

                    world_uid = make_world_uid(
                        split=split,
                        role_generated=role,
                        generator_family=family_name,
                        world_id=world_id_counter,
                    )

                    # Compute stats
                    diff_stats = compute_difficulty_metrics(
                        result.obstacles,
                        result.start_world,
                        result.goal_world,
                    )
                    obs_stats = compute_obstacle_stats(result.obstacles)
                    sem_stats = compute_semantic_stats(
                        result.obstacles,
                        result.start_world,
                        result.goal_world,
                    )

                    metadata = RouteAWorldMetadata(
                        schema_version=SCHEMA_VERSION,
                        world_uid=world_uid,
                        world_id=world_id_counter,
                        split=split,
                        role_generated=role,
                        role_validated=None,
                        role_active=None,
                        role_version="v1",
                        generator_family=family_name,
                        generator_seed=20260703,
                        generator_params=result.generator_params,
                        start_world=result.start_world,
                        goal_world=result.goal_world,
                        start_yaw=result.start_yaw,
                        goal_tolerance=result.goal_tolerance,
                        spawn_clearance_surface=SPAWN_CLEARANCE,
                        goal_clearance_surface=GOAL_CLEARANCE,
                        world_bounds=result.world_bounds,
                        obstacle_region=result.obstacle_region,
                        inflation_radius=INFLATION_RADIUS,
                        obstacle_stats=obs_stats,
                        difficulty_stats=diff_stats,
                        semantic_stats=sem_stats,
                        validation={},
                        baseline_calibration={},
                        disturb_profile=result.disturb_profile,
                        paired_seed_group=_next_seed_group(),
                        admission_status="pending_validation",
                    )

                    # Write .world file
                    world_filename = make_world_filename(world_uid)
                    world_path = world_output_dir / world_filename
                    sdf_content = generate_world_sdf(
                        result.obstacles,
                        world_name=world_uid,
                    )
                    world_path.write_text(sdf_content, encoding="utf-8")

                    # Write companion metadata JSON
                    meta_filename = f"{world_uid}.metadata.json"
                    meta_path = report_dir / meta_filename
                    write_json(meta_path, asdict(metadata))

                    metadata_list.append(metadata)
                    world_id_counter += 1

                    print(
                        f"  [{world_id_counter:04d}] {world_uid[:80]}"
                        f"  obs={len(result.obstacles)}"
                    )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "total_worlds": len(metadata_list),
        "total_failures": len(failures),
        "failures": failures,
        "role_counts": {
            role: sum(
                1 for m in metadata_list if m.role_generated == role
            )
            for role in ROLE_GENERATED
        },
        "split_counts": {
            split: sum(1 for m in metadata_list if m.split == split)
            for split in SPLITS
        },
        "family_counts": {},
    }

    # Count by family
    for m in metadata_list:
        key = f"{m.role_generated}/{m.generator_family}"
        summary["family_counts"][key] = summary["family_counts"].get(key, 0) + 1

    return metadata_list, summary


# ─────────────────────────────────────────────────────────────────────────────
# Manifest writers
# ─────────────────────────────────────────────────────────────────────────────


def write_manifest_json(
    metadata_list: List[RouteAWorldMetadata],
    summary: Dict[str, Any],
    report_dir: Path,
) -> Path:
    """Write full JSON manifest."""
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "summary": summary,
        "worlds": [asdict(m) for m in metadata_list],
    }
    path = report_dir / "routeA_rl_v1_world_manifest.json"
    write_json(path, manifest)
    return path


def write_manifest_csv(
    metadata_list: List[RouteAWorldMetadata],
    report_dir: Path,
) -> Path:
    """Write CSV summary manifest."""
    path = report_dir / "routeA_rl_v1_world_manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "world_uid", "world_id", "split", "role_generated",
        "role_validated", "role_active", "generator_family",
        "generator_seed", "start_world_x", "start_world_y", "start_world_z",
        "goal_world_x", "goal_world_y", "goal_world_z",
        "start_yaw", "world_bounds", "obstacle_count",
        "paired_seed_group", "admission_status",
        "world_file", "metadata_file",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in metadata_list:
            writer.writerow({
                "world_uid": m.world_uid,
                "world_id": m.world_id,
                "split": m.split,
                "role_generated": m.role_generated,
                "role_validated": m.role_validated or "",
                "role_active": m.role_active or "",
                "generator_family": m.generator_family,
                "generator_seed": m.generator_seed,
                "start_world_x": m.start_world[0],
                "start_world_y": m.start_world[1],
                "start_world_z": m.start_world[2] if len(m.start_world) > 2 else 2.0,
                "goal_world_x": m.goal_world[0],
                "goal_world_y": m.goal_world[1],
                "goal_world_z": m.goal_world[2] if len(m.goal_world) > 2 else 2.0,
                "start_yaw": m.start_yaw,
                "world_bounds": str(m.world_bounds),
                "obstacle_count": m.obstacle_stats.get("n_total", 0),
                "paired_seed_group": m.paired_seed_group,
                "admission_status": m.admission_status,
                "world_file": f"{m.world_uid}.world",
                "metadata_file": f"{m.world_uid}.metadata.json",
            })

    return path


def write_summary_md(
    metadata_list: List[RouteAWorldMetadata],
    summary: Dict[str, Any],
    report_dir: Path,
) -> Path:
    """Write human-readable worldgen summary markdown."""
    path = report_dir / "worldgen_summary.md"

    lines = [
        "# Route A RL v1 World Generation Summary",
        "",
        f"- **Schema version**: {SCHEMA_VERSION}",
        f"- **Total worlds**: {summary['total_worlds']}",
        f"- **Failures**: {summary['total_failures']}",
        f"- **Date**: 2026-07-03",
        "",
        "## Role Distribution",
        "",
        "| Role | Count |",
        "|------|-------|",
    ]
    for role in ROLE_GENERATED:
        lines.append(f"| {role} | {summary['role_counts'].get(role, 0)} |")

    lines += [
        "",
        "## Split Distribution",
        "",
        "| Split | Count |",
        "|-------|-------|",
    ]
    for split in SPLITS:
        cnt = summary["split_counts"].get(split, 0)
        if cnt > 0:
            lines.append(f"| {split} | {cnt} |")

    lines += [
        "",
        "## Generator Family Distribution",
        "",
        "| Role | Family | Count |",
        "|------|--------|-------|",
    ]
    for key, cnt in sorted(summary.get("family_counts", {}).items()):
        role, family = key.split("/", 1)
        lines.append(f"| {role} | {family} | {cnt} |")

    lines += [
        "",
        "## Metadata Integrity",
        "",
        "- `role_validated`: all `null` (pending validator)",
        "- `role_active`: all `null` (pending training pipeline)",
        "- `validation`: all `{}` (empty, pending)",
        "- `baseline_calibration`: all `{}` (empty, pending)",
        "- `admission_status`: all `pending_validation`",
        "",
        "## Forbidden Items Check",
        "",
        f"- Legacy world IDs (w4/w8/w11/w12/w18): **not present**",
        f"- Legacy scenario names as formal role: **not present**",
        f"- Falsified validation/calibration results: **none**",
        "",
        "## Files",
        "",
        f"- Manifest JSON: `{report_dir}/routeA_rl_v1_world_manifest.json`",
        f"- Manifest CSV: `{report_dir}/routeA_rl_v1_world_manifest.csv`",
    ]

    if summary["failures"]:
        lines += [
            "",
            "## Failures",
            "",
        ]
        for f in summary["failures"]:
            lines.append(
                f"- {f['role']}/{f['family']} variant={f['variant']}: "
                f"{f['error']}"
            )

    content = "\n".join(lines) + "\n"
    path.write_text(content, encoding="utf-8")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Route A RL v1 Environment World Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Smoke mode (small pool for verification)
  %(prog)s --smoke

  # Full generation with defaults
  %(prog)s --seed 20260703

  # Custom counts
  %(prog)s --n_train_per_role 40 --n_val_per_role 12 --n_test_per_role 20
        """,
    )

    parser.add_argument(
        "--world_output_dir", type=str,
        default="/home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds",
        help="Output directory for .world files",
    )
    parser.add_argument(
        "--report_dir", type=str,
        default="/home/cjj/AUV/program/reports/routeA_rl_env_v1/worldgen",
        help="Output directory for manifest and metadata files",
    )
    parser.add_argument(
        "--seed", type=int, default=20260703,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--n_train_per_role", type=int, default=40,
        help="Number of training worlds per role",
    )
    parser.add_argument(
        "--n_val_per_role", type=int, default=12,
        help="Number of validation worlds per role",
    )
    parser.add_argument(
        "--n_test_per_role", type=int, default=20,
        help="Number of test worlds per role",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Generate small smoke-test pool (2 train + 1 val + 1 test per role)",
    )

    args = parser.parse_args()

    # Smoke mode overrides counts
    if args.smoke:
        n_train = 2
        n_val = 1
        n_test = 1
        print("=== SMOKE MODE ===")
    else:
        n_train = args.n_train_per_role
        n_val = args.n_val_per_role
        n_test = args.n_test_per_role

    world_output_dir = Path(args.world_output_dir)
    report_dir = Path(args.report_dir)
    rng = random.Random(args.seed)

    print(f"Route A RL v1 World Generator")
    print(f"{'=' * 60}")
    print(f"Schema version: {SCHEMA_VERSION}")
    print(f"World output:   {world_output_dir}")
    print(f"Report dir:     {report_dir}")
    print(f"Seed:           {args.seed}")
    print(f"Roles:          {', '.join(ROLE_GENERATED)}")
    print(f"Train/role:     {n_train}")
    print(f"Val/role:       {n_val}")
    print(f"Test/role:      {n_test}")
    print(f"Total expected: {4 * (n_train + n_val + n_test)}")
    print()

    # Generate
    metadata_list, summary = generate_world_pool(
        rng=rng,
        world_output_dir=world_output_dir,
        report_dir=report_dir,
        n_train_per_role=n_train,
        n_val_per_role=n_val,
        n_test_per_role=n_test,
    )

    # Write manifests
    manifest_json_path = write_manifest_json(metadata_list, summary, report_dir)
    manifest_csv_path = write_manifest_csv(metadata_list, report_dir)
    summary_md_path = write_summary_md(metadata_list, summary, report_dir)

    # Summary
    print()
    print(f"{'=' * 60}")
    print(f"Generated {len(metadata_list)} worlds")
    print(f"Failures: {summary['total_failures']}")
    print()
    print("Role distribution:")
    for role in ROLE_GENERATED:
        print(f"  {role}: {summary['role_counts'].get(role, 0)}")
    print()
    print("Split distribution:")
    for split in SPLITS:
        cnt = summary["split_counts"].get(split, 0)
        if cnt > 0:
            print(f"  {split}: {cnt}")
    print()
    print(f"Manifest JSON: {manifest_json_path}")
    print(f"Manifest CSV:  {manifest_csv_path}")
    print(f"Summary MD:    {summary_md_path}")
    print()
    print("Metadata integrity:")
    n_validated = sum(1 for m in metadata_list if m.role_validated is not None)
    n_active = sum(1 for m in metadata_list if m.role_active is not None)
    n_nonempty_val = sum(1 for m in metadata_list if m.validation)
    n_nonempty_cal = sum(1 for m in metadata_list if m.baseline_calibration)
    print(f"  role_validated != null:  {n_validated} (should be 0)")
    print(f"  role_active != null:     {n_active} (should be 0)")
    print(f"  validation non-empty:    {n_nonempty_val} (should be 0)")
    print(f"  calibration non-empty:   {n_nonempty_cal} (should be 0)")

    all_ok = (
        n_validated == 0 and n_active == 0
        and n_nonempty_val == 0 and n_nonempty_cal == 0
    )
    if all_ok:
        print("  ✓ All metadata constraints satisfied")
    else:
        print("  ✗ Metadata constraint violations detected!")

    print()
    print("Done.")

    return 0 if (all_ok and summary["total_failures"] == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
