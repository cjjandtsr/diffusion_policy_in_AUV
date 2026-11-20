#!/usr/bin/env python3
"""
generate_worlds_ood.py — OOD/Corner-Case Gazebo world generation
================================================================================
iDP3-UAV OOD Experiment Design v1 — Layer A / Layer B / Corner-Case worlds

Copied from generate_worlds_v8_3.py and parameterized for OOD extrapolation.
Original script remains unchanged for training-environment reproducibility.

Key changes from v8_3:
  - All geometric constants exposed as CLI args
  - Single-tier generation (not the fixed 7-tier distribution)
  - Shape ratio control (cylinder:box proportion)
  - Corner-case world types (narrow_gap, symmetric, trap)
  - Increased Poisson attempts for tight min_dist

Output (same format as v8_3):
  {output_dir}/env_{id:03d}_obs{count}.world        — Gazebo SDF
  {output_dir}/env_{id:03d}_obs{count}.obstacles.json — metadata + difficulty
  {output_dir}/manifest.json — summary

Usage:
  # Layer A: same distribution, different seed (use original script)
  python generate_worlds_v8_3.py --output_dir worlds_layerA --seed 9001

  # Layer B1: moderate OOD extrapolation
  python generate_worlds_ood.py --output_dir worlds_layerB1 \
    --density_min 35 --density_max 42 --radius_max 2.5 --box_max 5.0 \
    --min_dist 1.5 --shape_ratio 1.0 --n_worlds 10 --seed 8001

  # Layer B2: extreme OOD extrapolation
  python generate_worlds_ood.py --output_dir worlds_layerB2 \
    --density_min 43 --density_max 50 --radius_max 3.0 --box_max 6.0 \
    --min_dist 1.0 --n_worlds 10 --seed 8002

  # Corner-case: narrow gap
  python generate_worlds_ood.py --output_dir worlds_corner_narrow_gap \
    --corner_type narrow_gap --n_worlds 5 --seed 7001

  # Corner-case: symmetric
  python generate_worlds_ood.py --output_dir worlds_corner_symmetric \
    --corner_type symmetric --n_worlds 5 --seed 7002

  # Corner-case: trap
  python generate_worlds_ood.py --output_dir worlds_corner_trap \
    --corner_type trap --n_worlds 5 --seed 7003
"""

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Configuration (defaults = training range from v8_3)
# ─────────────────────────────────────────────────────────────────────────────

SPAWN_WORLD = (-8.0, 0.0)        # drone spawn in world frame (实测: fast_test.launch 真实起飞点)
SPAWN_WORLD_Z = 0.1               # drone spawn height
GOAL_WORLD_RANDOM = (8.0, 0.0)   # 随机world的实际goal world(8,0)=local(16,0)+spawn(-8,0). 起终点(-8,0)→(8,0)充分利用地图
SPAWN_CLEARANCE = 5.0              # meters, no obstacles within this of spawn
GOAL_CLEARANCE = 3.0               # meters, no obstacles within this of goal
OBS_REGION = (-10.0, 10.0)         # obstacle placement bounds (x and y)

# Default obstacle size ranges (training ranges)
CYL_R_MIN_DEFAULT, CYL_R_MAX_DEFAULT = 0.15, 2.0
CYL_H_MIN_DEFAULT, CYL_H_MAX_DEFAULT = 1.0, 8.0
BOX_S_MIN_DEFAULT, BOX_S_MAX_DEFAULT = 0.5, 4.0
BOX_H_MIN_DEFAULT, BOX_H_MAX_DEFAULT = 1.0, 6.0

MIN_DIST_DEFAULT = 2.0             # Poisson disk minimum distance
MAX_POISSON_ATTEMPTS_DEFAULT = 200 # rejection sampling attempts per point

# Gazebo material colors for visual variety
COLORS = [
    ("Gazebo/Grey",     "0.5 0.5 0.5 1"),
    ("Gazebo/Blue",     "0.447 0.624 0.812 1"),
    ("Gazebo/Red",      "0.8 0.2 0.2 1"),
    ("Gazebo/Orange",   "0.9 0.5 0.1 1"),
    ("Gazebo/White",    "0.9 0.9 0.9 1"),
    ("Gazebo/DarkGrey", "0.3 0.3 0.3 1"),
    ("Gazebo/Yellow",   "0.9 0.9 0.1 1"),
]

# Drone safety envelope (F450 + prop guard + margin)
DRONE_RADIUS = 0.35  # meters (half of F450 diagonal + safety)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Obstacle:
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
class WorldSpec:
    world_id: int
    tier: str
    n_obstacles: int
    filename: str
    obstacles: List[Obstacle] = field(default_factory=list)
    start_world: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    goal_world: Tuple[float, float, float] = (0.0, 0.0, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Poisson disk sampling
# ─────────────────────────────────────────────────────────────────────────────

def poisson_disk_sample(
    n: int,
    bounds: Tuple[float, float],
    min_dist: float,
    exclusion_zones: List[Tuple[float, float, float]],
    rng: random.Random,
    existing: Optional[List[Tuple[float, float]]] = None,
    max_attempts: int = MAX_POISSON_ATTEMPTS_DEFAULT,
) -> Tuple[List[Tuple[float, float]], int]:
    """
    Sample n points in [bounds[0], bounds[1]]^2 with minimum distance
    between points and away from exclusion zones.

    Returns (points, n_placed) — n_placed may be < n if placement fails.
    """
    points = list(existing) if existing else []

    for _ in range(n):
        placed = False
        for _ in range(max_attempts):
            x = rng.uniform(bounds[0], bounds[1])
            y = rng.uniform(bounds[0], bounds[1])

            # Check exclusion zones
            ok = True
            for ex, ey, er in exclusion_zones:
                if math.hypot(x - ex, y - ey) < er:
                    ok = False
                    break
            if not ok:
                continue

            # Check min distance to existing points
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
# Obstacle generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_random_obstacle(
    name: str, x: float, y: float, rng: random.Random,
    cyl_r_min: float, cyl_r_max: float,
    cyl_h_min: float, cyl_h_max: float,
    box_s_min: float, box_s_max: float,
    box_h_min: float, box_h_max: float,
    shape_ratio: float = 0.667,  # 2:1 cylinder:box = 0.667
) -> Obstacle:
    """
    Generate a random cylinder or box obstacle at (x, y).

    shape_ratio: probability of cylinder (0=all boxes, 1=all cylinders)
    """
    is_cylinder = rng.random() < shape_ratio
    yaw = rng.uniform(0, math.pi)

    if is_cylinder:
        r = rng.uniform(cyl_r_min, cyl_r_max)
        h = rng.uniform(cyl_h_min, cyl_h_max)
        return Obstacle(
            name=name, shape="cylinder",
            x=x, y=y, z=h/2, yaw=yaw,
            radius=r, height=h,
            bounding_r=r,
        )
    else:
        sx = rng.uniform(box_s_min, box_s_max)
        sy = rng.uniform(box_s_min, box_s_max)
        sz = rng.uniform(box_h_min, box_h_max)
        return Obstacle(
            name=name, shape="box",
            x=x, y=y, z=sz/2, yaw=yaw,
            size_x=sx, size_y=sy, size_z=sz,
            bounding_r=math.hypot(sx, sy) / 2,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Corner-case generators
# ─────────────────────────────────────────────────────────────────────────────

def generate_narrow_gap_world(
    rng: random.Random, rgs: float, world_id: int = 0
) -> WorldSpec:
    """
    Narrow gap corner-case.

    Two LONG PARALLEL WALLS (boxes) forming a constricted corridor along
    the start→goal line. The walls extend far enough that going around is
    not feasible — the drone MUST pass through the gap.

    RGS (Relative Gap Size): gap_width / drone_diameter
      RGS=2.0 → 1.40m gap, easy (2× drone diameter)
      RGS=1.5 → 1.05m gap, moderate
      RGS=1.2 → 0.84m gap, narrow
      RGS=1.05 → 0.74m gap, very narrow
      RGS=1.0 → 0.70m gap, exactly drone diameter, physically critical

    Layout (top-down view):
      start (-8,0) ═══════════════╗         ╔═══════════════ goal (8,0)
                                  ║  gap_w  ║
                      left_wall   ║         ║  right_wall
                      (long box)  ║         ║  (long box)
                                  ║         ║
                                  ╚═════════╝
                                  ←  wall_length=8m →
    """
    drone_diameter = 2 * DRONE_RADIUS  # 0.70m
    gap_width = rgs * drone_diameter
    half_gap = gap_width / 2.0

    start = (-8.0, 0.0, 2.0)
    goal = (8.0, 0.0, 2.0)

    # Wall dimensions
    wall_length = 8.0   # long enough to prevent going around
    wall_thickness = 0.5  # thin wall
    wall_height = 6.0

    # Walls centered at x=0, placed ±half_gap from center line
    obstacles = []

    for side, sign in enumerate([-1, 1]):
        # Wall center y = sign * (half_gap + wall_thickness/2)
        # So the inner surface is at y = sign * half_gap
        wall_y = sign * (half_gap + wall_thickness / 2.0)
        wall = Obstacle(
            name=f"gap_wall_{['left', 'right'][side]}",
            shape="box",
            x=0.0, y=wall_y, z=wall_height / 2.0, yaw=0.0,
            size_x=wall_length, size_y=wall_thickness, size_z=wall_height,
            bounding_r=math.hypot(wall_length, wall_thickness) / 2.0,
        )
        obstacles.append(wall)

    return WorldSpec(
        world_id=world_id,
        tier=f"narrow_gap_rgs{rgs:.2f}",
        n_obstacles=len(obstacles),
        filename="",
        obstacles=obstacles,
        start_world=start,
        goal_world=goal,
    )


def generate_symmetric_world(
    rng: random.Random, center_width: float, world_id: int = 0
) -> WorldSpec:
    """
    Symmetric obstacle corner-case.

    A single wide box centered ON the start→goal line, forcing the drone
    to pick left or right. Both sides are geometrically symmetric — tests
    whether the policy handles multi-modal choices consistently.

    center_width: width (Y) of the central obstacle.
      Small (1-2m) → drone can easily go either side
      Medium (3-5m) → drone must make clear left/right choice
      Large (8m) → drone forced to make a significant detour

    Layout (top-down):
      start(-8,0) ──→ ┌─────────────┐ ──→ goal(8,0)
                      │   CENTER    │
                      │   BLOCK     │   ← drone must go
                      │  (width×3m)│     left OR right
                      └─────────────┘
    """
    start = (-8.0, 0.0, 2.0)
    goal = (8.0, 0.0, 2.0)

    box_length = 3.0    # along X (thickness)
    box_height = 5.0    # tall enough to block

    obstacles = []
    center = Obstacle(
        name="center_block",
        shape="box",
        x=0.0, y=0.0, z=box_height / 2.0, yaw=0.0,
        size_x=box_length, size_y=center_width, size_z=box_height,
        bounding_r=math.hypot(box_length, center_width) / 2.0,
    )
    obstacles.append(center)

    # start, goal, center are collinear by design (all at y=0)
    return WorldSpec(
        world_id=world_id,
        tier=f"symmetric_w{center_width:.1f}",
        n_obstacles=len(obstacles),
        filename="",
        obstacles=obstacles,
        start_world=start,
        goal_world=goal,
    )


def generate_trap_world(
    rng: random.Random, depth: float, opening: float, world_id: int = 0
) -> WorldSpec:
    """
    Trap / cul-de-sac corner-case.

    U-shaped obstacle placed ON the direct start→goal path. The U opens
    toward the START point, so a drone flying straight from start toward
    goal enters the trap. The drone must recognize the dead-end and go
    around. Goal is behind the trap.

    depth: depth of the U along the X axis (how far back it extends)
    opening: width of the U opening in Y (smaller = harder to escape)

    Layout (top-down):
                  ┌──────────┐
      start       │  DEAD    │       goal
      (-10,0) ──→ │  END     │ ──→  (8,0)
                  │ (trap)   │
                  └──────────┘
                       ↑ U opens toward start
      Drone must NOT enter but go around (above or below).
    """
    start = (-10.0, 0.0, 2.0)
    goal = (8.0, 0.0, 2.0)

    # Trap ON the direct path (y=0), NOT offset.
    # Opening at x=0 faces START (-x direction), back wall at x=depth
    # toward GOAL (+x direction). Drone flies from (-10,0) toward (8,0),
    # encounters opening first, enters U, hits back wall — must back out.
    #
    # Layout (top-down, X→):
    #   start(-10)  →  | opening  |  →  | back wall |  →  goal(8)
    #                   | at x=0   |     | at x=depth|
    #   If the drone goes straight, it enters the U and gets trapped.

    wall_thickness = 0.5
    wall_height = 4.0

    obstacles = []

    # Side walls: extend from x=0 (opening) to x=depth (back wall)
    # Center at x=depth/2, length=depth
    side_center_x = depth / 2.0
    for side, sign in enumerate([-1, 1]):
        wall = Obstacle(
            name=f"trap_{['left', 'right'][side]}",
            shape="box",
            x=side_center_x,
            y=sign * opening / 2.0,
            z=wall_height / 2.0, yaw=0.0,
            size_x=depth, size_y=wall_thickness, size_z=wall_height,
            bounding_r=math.hypot(depth, wall_thickness) / 2.0,
        )
        obstacles.append(wall)

    # Back wall: at x=depth, spans across the opening width
    back_wall = Obstacle(
        name="trap_back",
        shape="box",
        x=depth, y=0.0,
        z=wall_height / 2.0, yaw=0.0,
        size_x=wall_thickness, size_y=opening, size_z=wall_height,
        bounding_r=math.hypot(wall_thickness, opening) / 2.0,
    )
    obstacles.append(back_wall)

    # Funnel walls extending from near x=0 toward start (-x),
    # making the trap harder to avoid by flaring outward.
    funnel_length = depth * 0.8
    funnel_flare = opening * 0.5
    funnel_center_x = -funnel_length / 2.0  # extends from x=0 toward -x

    for side, sign in enumerate([-1, 1]):
        funnel = Obstacle(
            name=f"trap_funnel_{['l', 'r'][side]}",
            shape="box",
            x=funnel_center_x,
            y=sign * (opening / 2.0 + funnel_flare / 2.0),
            z=wall_height / 2.0, yaw=0.0,
            size_x=funnel_length, size_y=wall_thickness, size_z=wall_height,
            bounding_r=math.hypot(funnel_length, wall_thickness) / 2.0,
        )
        obstacles.append(funnel)

    return WorldSpec(
        world_id=world_id,
        tier=f"trap_d{depth:.1f}_o{opening:.1f}",
        n_obstacles=len(obstacles),
        filename="",
        obstacles=obstacles,
        start_world=start,
        goal_world=goal,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SDF generation
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


def generate_world_sdf(spec: WorldSpec) -> str:
    """Generate a complete Gazebo .world SDF file."""
    import random as _random
    obstacle_xml_parts = []
    for obs in spec.obstacles:
        color_name, ambient = _random.choice(COLORS)
        obstacle_xml_parts.append(obstacle_to_sdf(obs, color_name, ambient))
    obstacle_xml = "\n".join(obstacle_xml_parts)

    return f"""<?xml version='1.0'?>
<sdf version='1.6'>
  <world name='default'>
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

    <!-- ============ OBSTACLES ({len(spec.obstacles)}) ============ -->
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
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_world(
    spec: WorldSpec,
    cyl_r_min: float, cyl_r_max: float,
    cyl_h_min: float, cyl_h_max: float,
    box_s_min: float, box_s_max: float,
    box_h_min: float, box_h_max: float,
    min_dist: float,
) -> List[str]:
    """Validate a world spec, returning list of issues (empty = OK)."""
    issues = []

    for i, obs in enumerate(spec.obstacles):
        # Check within bounds
        if not (OBS_REGION[0] <= obs.x <= OBS_REGION[1]):
            issues.append(f"{obs.name}: x={obs.x:.2f} out of bounds")
        if not (OBS_REGION[0] <= obs.y <= OBS_REGION[1]):
            issues.append(f"{obs.name}: y={obs.y:.2f} out of bounds")

        # Check spawn clearance
        d_spawn = math.hypot(obs.x - SPAWN_WORLD[0], obs.y - SPAWN_WORLD[1])
        if d_spawn < SPAWN_CLEARANCE:
            issues.append(
                f"{obs.name}: too close to spawn "
                f"({d_spawn:.2f}m < {SPAWN_CLEARANCE}m)"
            )

        # Check goal clearance
        if spec.goal_world != (0.0, 0.0, 0.0):
            # corner world: 用自己的 goal_world
            gx_chk, gy_chk = spec.goal_world[0], spec.goal_world[1]
        else:
            # 随机 world: goal_world=(0,0,0) 是占位, 实际 goal 在 world(4,0)
            gx_chk, gy_chk = GOAL_WORLD_RANDOM[0], GOAL_WORLD_RANDOM[1]
        d_goal = math.hypot(obs.x - gx_chk, obs.y - gy_chk)
        if d_goal < GOAL_CLEARANCE:
            issues.append(
                f"{obs.name}: too close to goal "
                f"({d_goal:.2f}m < {GOAL_CLEARANCE}m)"
            )

        # Check size ranges
        if obs.shape == "cylinder":
            if not (cyl_r_min - 0.001 <= obs.radius <= cyl_r_max + 0.001):
                issues.append(
                    f"{obs.name}: radius {obs.radius:.2f} out of "
                    f"[{cyl_r_min}, {cyl_r_max}]"
                )
            if not (cyl_h_min - 0.001 <= obs.height <= cyl_h_max + 0.001):
                issues.append(
                    f"{obs.name}: height {obs.height:.2f} out of "
                    f"[{cyl_h_min}, {cyl_h_max}]"
                )
        elif obs.shape == "box":
            if not (box_s_min - 0.001 <= obs.size_x <= box_s_max + 0.001):
                issues.append(
                    f"{obs.name}: size_x {obs.size_x:.2f} out of "
                    f"[{box_s_min}, {box_s_max}]"
                )
            if not (box_s_min - 0.001 <= obs.size_y <= box_s_max + 0.001):
                issues.append(
                    f"{obs.name}: size_y {obs.size_y:.2f} out of "
                    f"[{box_s_min}, {box_s_max}]"
                )
            if not (box_h_min - 0.001 <= obs.size_z <= box_h_max + 0.001):
                issues.append(
                    f"{obs.name}: size_z {obs.size_z:.2f} out of "
                    f"[{box_h_min}, {box_h_max}]"
                )

        # Check pairwise min distance
        for j in range(i + 1, len(spec.obstacles)):
            obs2 = spec.obstacles[j]
            d = math.hypot(obs.x - obs2.x, obs.y - obs2.y)
            if d < min_dist * 0.9:  # 10% tolerance
                issues.append(
                    f"{obs.name}<->{obs2.name}: distance {d:.2f}m < "
                    f"{min_dist}m"
                )

    return issues


# ─────────────────────────────────────────────────────────────────────────────
# Difficulty metrics (TO / VO / AOL)
# ─────────────────────────────────────────────────────────────────────────────

def compute_difficulty_metrics(
    spec: WorldSpec,
) -> dict:
    """
    Compute FlightBench-inspired difficulty metrics for a world.

    TO (Target Orientation): average angular deviation of obstacles from
      the start→goal line. Higher TO = more obstacles block direct path.
    VO (Visibility Occlusion): fraction of goal "visible" from start
      (line-of-sight blocked by obstacles). 0 = fully occluded, 1 = clear.
    AOL (Average Obstacle Load): normalized obstacle density per unit area.

    These are computed from obstacle geometry extracted from .obstacles.json,
    so this function can also be used standalone on existing JSON files.
    """
    obstacles = spec.obstacles
    start = spec.start_world if spec.start_world != (0, 0, 0) else SPAWN_WORLD + (SPAWN_WORLD_Z,)
    goal = spec.goal_world if spec.goal_world != (0, 0, 0) else (12.0, 0.0, 2.0)

    sx, sy = start[0], start[1]
    gx, gy = goal[0], goal[1]

    # Direct path vector
    dx, dy = gx - sx, gy - sy
    path_len = math.hypot(dx, dy)
    if path_len < 0.01:
        return {"TO": 0.0, "VO": 1.0, "AOL": 0.0, "n_obstacles": len(obstacles)}

    # --- TO: Target Orientation ---
    # For each obstacle, compute angular offset from start→goal line.
    # TO = mean over obstacles of |angle(obs_from_start, path_direction)|
    to_values = []
    for obs in obstacles:
        ox, oy = obs.x, obs.y
        odx, ody = ox - sx, oy - sy
        obs_dist = math.hypot(odx, ody)
        if obs_dist < 0.01:
            # obstacle at start — worst case
            to_values.append(math.pi / 2)
            continue
        # Dot product to get cos(angle)
        cos_angle = (odx * dx + ody * dy) / (obs_dist * path_len)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle = math.acos(cos_angle)
        to_values.append(angle)

    TO = float(sum(to_values) / len(to_values)) if to_values else 0.0

    # --- VO: Visibility Occlusion ---
    # Check line-of-sight from start to goal against each obstacle cylinder
    # (approximate obstacles as bounding circles in the XY plane)
    los_blocked = False
    for obs in obstacles:
        r_eff = obs.bounding_r
        if r_eff <= 0:
            continue
        # Distance from obstacle center to line segment (start→goal)
        # Using point-to-segment distance
        ox, oy = obs.x, obs.y
        # Vector from start to obstacle
        ox_rel, oy_rel = ox - sx, oy - sy
        # Project onto path direction
        t = (ox_rel * dx + oy_rel * dy) / (path_len * path_len)
        t = max(0.0, min(1.0, t))  # clamp to segment
        # Closest point on segment
        closest_x = sx + t * dx
        closest_y = sy + t * dy
        dist_to_seg = math.hypot(ox - closest_x, oy - closest_y)
        if dist_to_seg < r_eff:
            los_blocked = True
            break

    VO = 0.0 if los_blocked else 1.0

    # --- AOL: Average Obstacle Load ---
    # Obstacle count per unit area of the placement region
    area = (OBS_REGION[1] - OBS_REGION[0]) ** 2  # 400 m²
    AOL = float(len(obstacles) / area) if area > 0 else 0.0

    return {
        "TO": round(TO, 4),
        "VO": round(VO, 4),
        "AOL": round(AOL, 4),
        "n_obstacles": len(obstacles),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="OOD/Corner-Case Gazebo world generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Layer B1 moderate OOD
  %(prog)s --output_dir worlds_layerB1 --density_min 35 --density_max 42 \\
    --radius_max 2.5 --box_max 5.0 --min_dist 1.5 --shape_ratio 1.0 \\
    --n_worlds 10 --seed 8001

  # Layer B2 extreme OOD
  %(prog)s --output_dir worlds_layerB2 --density_min 43 --density_max 50 \\
    --radius_max 3.0 --box_max 6.0 --min_dist 1.0 --n_worlds 10 --seed 8002

  # Narrow gap corner-case (RGS gradient)
  %(prog)s --output_dir worlds_corner_narrow_gap --corner_type narrow_gap \\
    --n_worlds 5 --seed 7001

  # Symmetric obstacle corner-case
  %(prog)s --output_dir worlds_corner_symmetric --corner_type symmetric \\
    --n_worlds 5 --seed 7002

  # Trap corner-case
  %(prog)s --output_dir worlds_corner_trap --corner_type trap \\
    --n_worlds 5 --seed 7003
        """,
    )

    # ── Output control ──
    parser.add_argument("--output_dir", default="./worlds_ood",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--validate_only", action="store_true",
                        help="Only validate existing worlds")
    parser.add_argument("--n_worlds", type=int, default=10,
                        help="Number of worlds to generate")

    # ── Tier metadata ──
    parser.add_argument("--tier_name", default="ood",
                        help="Tier label for manifest")

    # ── Density ──
    parser.add_argument("--density_min", type=int, default=0,
                        help="Minimum obstacle count (default: 0)")
    parser.add_argument("--density_max", type=int, default=0,
                        help="Maximum obstacle count (default: 0)")

    # ── Obstacle size ranges ──
    parser.add_argument("--radius_min", type=float, default=CYL_R_MIN_DEFAULT,
                        help=f"Cylinder radius min (default: {CYL_R_MIN_DEFAULT})")
    parser.add_argument("--radius_max", type=float, default=CYL_R_MAX_DEFAULT,
                        help=f"Cylinder radius max (default: {CYL_R_MAX_DEFAULT})")
    parser.add_argument("--cyl_h_min", type=float, default=CYL_H_MIN_DEFAULT,
                        help=f"Cylinder height min (default: {CYL_H_MIN_DEFAULT})")
    parser.add_argument("--cyl_h_max", type=float, default=CYL_H_MAX_DEFAULT,
                        help=f"Cylinder height max (default: {CYL_H_MAX_DEFAULT})")
    parser.add_argument("--box_min", type=float, default=BOX_S_MIN_DEFAULT,
                        help=f"Box side min (default: {BOX_S_MIN_DEFAULT})")
    parser.add_argument("--box_max", type=float, default=BOX_S_MAX_DEFAULT,
                        help=f"Box side max (default: {BOX_S_MAX_DEFAULT})")
    parser.add_argument("--box_h_min", type=float, default=BOX_H_MIN_DEFAULT,
                        help=f"Box height min (default: {BOX_H_MIN_DEFAULT})")
    parser.add_argument("--box_h_max", type=float, default=BOX_H_MAX_DEFAULT,
                        help=f"Box height max (default: {BOX_H_MAX_DEFAULT})")

    # ── Placement ──
    parser.add_argument("--min_dist", type=float, default=MIN_DIST_DEFAULT,
                        help=f"Poisson disk min distance (default: {MIN_DIST_DEFAULT})")
    parser.add_argument("--max_poisson_attempts", type=int,
                        default=MAX_POISSON_ATTEMPTS_DEFAULT,
                        help="Max Poisson rejection attempts per point")

    # ── Shape ──
    parser.add_argument("--shape_ratio", type=float, default=0.667,
                        help="Cylinder:box ratio (0=all boxes, 0.667=2:1, "
                             "1.0=all cylinders, default: 0.667)")

    # ── Corner-case ──
    parser.add_argument("--corner_type", type=str, default=None,
                        choices=["narrow_gap", "symmetric", "trap"],
                        help="Generate corner-case worlds instead of random")

    args = parser.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Adjust Poisson attempts for tight min_dist
    if args.min_dist < 1.5:
        effective_attempts = int(args.max_poisson_attempts * (2.0 / args.min_dist))
        effective_attempts = min(effective_attempts, 2000)  # cap
    else:
        effective_attempts = args.max_poisson_attempts

    # Exclusion zones: spawn point (always), goal point (for corner-cases)
    exclusion_zones = [
        (SPAWN_WORLD[0], SPAWN_WORLD[1], SPAWN_CLEARANCE),
    ]
    # 随机 world (非corner) 的实际 goal=world(4,0) 也必须保护, 否则障碍占据目标点
    # → 任务物理不可解. corner world 有自己的 goal_world, 由下面 per-spec 逻辑处理.
    if args.corner_type in (None, "", "none"):
        exclusion_zones.append(
            (GOAL_WORLD_RANDOM[0], GOAL_WORLD_RANDOM[1], GOAL_CLEARANCE)
        )

    print(f"OOD World Generator")
    print(f"{'='*60}")
    print(f"Output: {args.output_dir}/")
    print(f"Seed: {args.seed}")
    print(f"Tier: {args.tier_name}")
    print(f"Spawn: {SPAWN_WORLD}, clearance: {SPAWN_CLEARANCE}m")
    print(f"Obstacle region: [{OBS_REGION[0]}, {OBS_REGION[1]}]²")
    print(f"Min distance: {args.min_dist}m (attempts: {effective_attempts})")
    print(f"Shape ratio (cyl:box): {args.shape_ratio:.2f}")
    print()

    # ── Generate world specs ──
    worlds = []

    if args.corner_type == "narrow_gap":
        # RGS gradient: 2.0, 1.5, 1.2, 1.05, 1.0
        rgs_values = [2.0, 1.5, 1.2, 1.05, 1.0]
        n_gen = min(args.n_worlds, len(rgs_values))
        print(f"Corner-case: NARROW GAP (RGS gradient)")
        print(f"Drone diameter: {2*DRONE_RADIUS:.2f}m")
        print()
        for i, rgs in enumerate(rgs_values[:n_gen]):
            spec = generate_narrow_gap_world(rng, rgs, world_id=i)
            worlds.append(spec)

    elif args.corner_type == "symmetric":
        # Center width gradient: 1.0, 2.0, 3.0, 5.0, 8.0
        widths = [1.0, 2.0, 3.0, 5.0, 8.0]
        n_gen = min(args.n_worlds, len(widths))
        print(f"Corner-case: SYMMETRIC (center width gradient)")
        print()
        for i, w in enumerate(widths[:n_gen]):
            spec = generate_symmetric_world(rng, w, world_id=i)
            worlds.append(spec)

    elif args.corner_type == "trap":
        # Depth/opening gradient
        # (depth, opening) pairs from easy to hard
        trap_configs = [
            (2.0, 4.0),   # shallow, wide — easy to avoid
            (3.0, 3.0),   # medium depth, medium opening
            (4.0, 2.5),   # deeper, narrower
            (5.0, 2.0),   # deep, narrow — hard
            (6.0, 1.5),   # very deep, very narrow — hardest
        ]
        n_gen = min(args.n_worlds, len(trap_configs))
        print(f"Corner-case: TRAP (depth/opening gradient)")
        print()
        for i, (depth, opening) in enumerate(trap_configs[:n_gen]):
            spec = generate_trap_world(rng, depth, opening, world_id=i)
            worlds.append(spec)

    else:
        # Standard random world generation with parameterized ranges
        if args.density_max <= 0:
            print("ERROR: --density_max is required for random world generation")
            print("  (or use --corner_type for corner-case worlds)")
            return 1

        print(f"Density range: [{args.density_min}, {args.density_max}]")
        print(f"Cylinder radius: [{args.radius_min}, {args.radius_max}]m")
        print(f"Cylinder height:  [{args.cyl_h_min}, {args.cyl_h_max}]m")
        print(f"Box side:        [{args.box_min}, {args.box_max}]m")
        print(f"Box height:      [{args.box_h_min}, {args.box_h_max}]m")
        print()

        for world_id in range(args.n_worlds):
            if args.density_min == args.density_max:
                n_obs = args.density_min
            else:
                n_obs = rng.randint(args.density_min, args.density_max)

            spec = WorldSpec(
                world_id=world_id,
                tier=args.tier_name,
                n_obstacles=n_obs,
                filename="",
            )
            worlds.append(spec)

    # ── Generate obstacles for each world ──
    manifest = []
    total_issues = 0

    for spec in worlds:
        if args.corner_type:
            # Corner-case worlds already have obstacles
            pass
        elif spec.n_obstacles == 0:
            spec.obstacles = []
        else:
            # Poisson disk sampling for positions
            positions, n_placed = poisson_disk_sample(
                n=spec.n_obstacles,
                bounds=OBS_REGION,
                min_dist=args.min_dist,
                exclusion_zones=exclusion_zones,
                rng=rng,
                max_attempts=effective_attempts,
            )

            if n_placed < spec.n_obstacles:
                shortfall = spec.n_obstacles - n_placed
                print(f"  [{spec.world_id:03d}] ⚠ Poisson disk could only place "
                      f"{n_placed}/{spec.n_obstacles} obstacles "
                      f"(shortfall={shortfall}, min_dist={args.min_dist}m)")

            spec.n_obstacles = n_placed
            spec.obstacles = []
            for i, (px, py) in enumerate(positions):
                obs = generate_random_obstacle(
                    name=f"obs_{i:02d}",
                    x=px, y=py, rng=rng,
                    cyl_r_min=args.radius_min,
                    cyl_r_max=args.radius_max,
                    cyl_h_min=args.cyl_h_min,
                    cyl_h_max=args.cyl_h_max,
                    box_s_min=args.box_min,
                    box_s_max=args.box_max,
                    box_h_min=args.box_h_min,
                    box_h_max=args.box_h_max,
                    shape_ratio=args.shape_ratio,
                )
                spec.obstacles.append(obs)

        # Set filename
        spec.filename = f"env_{spec.world_id:03d}_obs{spec.n_obstacles}.world"

        # Validate
        issues = validate_world(
            spec,
            cyl_r_min=args.radius_min,
            cyl_r_max=args.radius_max,
            cyl_h_min=args.cyl_h_min,
            cyl_h_max=args.cyl_h_max,
            box_s_min=args.box_min,
            box_s_max=args.box_max,
            box_h_min=args.box_h_min,
            box_h_max=args.box_h_max,
            min_dist=args.min_dist,
        )
        total_issues += len(issues)

        status = "✓" if not issues else f"✗ ({len(issues)} issues)"
        tier_display = spec.tier[:40]
        print(f"  [{spec.world_id:03d}] {tier_display:40s} "
              f"obs={spec.n_obstacles:2d}  {spec.filename:30s}  {status}")
        for issue in issues:
            print(f"         ⚠ {issue}")

        # Compute difficulty
        difficulty = compute_difficulty_metrics(spec)

        # Generate SDF
        sdf_content = generate_world_sdf(spec)
        world_path = os.path.join(args.output_dir, spec.filename)
        with open(world_path, "w") as f:
            f.write(sdf_content)

        # Generate companion JSON
        obs_data = []
        for obs in spec.obstacles:
            d = {
                "name": obs.name,
                "shape": obs.shape,
                "x": round(obs.x, 4),
                "y": round(obs.y, 4),
                "yaw": round(obs.yaw, 4),
            }
            if obs.shape == "cylinder":
                d["radius"] = round(obs.radius, 4)
                d["height"] = round(obs.height, 4)
            else:
                d["size_x"] = round(obs.size_x, 4)
                d["size_y"] = round(obs.size_y, 4)
                d["size_z"] = round(obs.size_z, 4)
            obs_data.append(d)

        json_data = {
            "world_id": spec.world_id,
            "tier": spec.tier,
            "n_obstacles": spec.n_obstacles,
            "spawn_world": list(SPAWN_WORLD),
            "obstacle_region": list(OBS_REGION),
            "obstacles": obs_data,
            "difficulty": difficulty,
            "generator_params": {
                "density_range": [args.density_min, args.density_max],
                "radius_range": [args.radius_min, args.radius_max],
                "box_range": [args.box_min, args.box_max],
                "min_dist": args.min_dist,
                "shape_ratio": args.shape_ratio,
                "corner_type": args.corner_type,
            },
        }

        # Add corner-case metadata
        if spec.start_world != (0.0, 0.0, 0.0):
            json_data["start_world"] = list(spec.start_world)
        if spec.goal_world != (0.0, 0.0, 0.0):
            json_data["goal_world"] = list(spec.goal_world)

        json_path = os.path.join(
            args.output_dir,
            spec.filename.replace(".world", ".obstacles.json")
        )
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

        manifest.append({
            "world_id": spec.world_id,
            "tier": spec.tier,
            "n_obstacles": spec.n_obstacles,
            "filename": spec.filename,
            "n_issues": len(issues),
            "difficulty": difficulty,
        })

    # ── Write manifest ──
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({
            "version": "ood_v1",
            "seed": args.seed,
            "total_worlds": len(worlds),
            "spawn_world": list(SPAWN_WORLD),
            "spawn_clearance": SPAWN_CLEARANCE,
            "obstacle_region": list(OBS_REGION),
            "generator_params": {
                "density_range": [args.density_min, args.density_max],
                "radius_range": [args.radius_min, args.radius_max],
                "box_range": [args.box_min, args.box_max],
                "min_dist": args.min_dist,
                "shape_ratio": args.shape_ratio,
                "corner_type": args.corner_type,
            },
            "worlds": manifest,
        }, f, indent=2)

    # ── Summary ──
    print()
    print(f"{'='*60}")
    print(f"Generated {len(worlds)} worlds → {args.output_dir}/")
    print(f"Total validation issues: {total_issues}")
    if total_issues == 0:
        print("All worlds passed geometric validation ✓")
    else:
        print(f"⚠ {total_issues} issue(s) found — review above")

    # Difficulty summary
    if worlds:
        to_vals = [m["difficulty"]["TO"] for m in manifest if "difficulty" in m]
        vo_vals = [m["difficulty"]["VO"] for m in manifest if "difficulty" in m]
        aol_vals = [m["difficulty"]["AOL"] for m in manifest if "difficulty" in m]
        if to_vals:
            print()
            print("Difficulty summary:")
            print(f"  TO  (Target Orientation):   "
                  f"mean={sum(to_vals)/len(to_vals):.3f}, "
                  f"range=[{min(to_vals):.3f}, {max(to_vals):.3f}]")
            print(f"  VO  (Visibility Occlusion): "
                  f"mean={sum(vo_vals)/len(vo_vals):.3f}, "
                  f"range=[{min(vo_vals):.3f}, {max(vo_vals):.3f}]")
            print(f"  AOL (Avg Obstacle Load):    "
                  f"mean={sum(aol_vals)/len(aol_vals):.4f}, "
                  f"range=[{min(aol_vals):.4f}, {max(aol_vals):.4f}]")

    print()
    print(f"Manifest: {manifest_path}")
    print("Done.")

    return 0 if total_issues == 0 else 1


if __name__ == "__main__":
    sys.exit(main())