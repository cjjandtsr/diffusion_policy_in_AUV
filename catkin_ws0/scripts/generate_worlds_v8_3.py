#!/usr/bin/env python3
"""
generate_worlds_v8_3.py — Phase 0.1: Programmatic Gazebo world generation
================================================================================
iDP3-UAV v8.3 §6.1 specification

Generates 60 Gazebo .world SDF files with stratified obstacle distributions,
plus companion .obstacles.json files for random_target_generator.

Distribution (v8.3 §6.1):
  empty      13%  →  8 worlds  (0 obstacles)
  sparse     20%  → 12 worlds  (1–5 obstacles)
  medium-A   25%  → 15 worlds  (6–12 obstacles)
  medium-B   22%  → 13 worlds  (13–20 obstacles)
  dense      15%  →  9 worlds  (21–30 obstacles)
  super-dense 3%  →  2 worlds  (31–40 obstacles)
  special     2%  →  1 world   (corridor with funnel)

Obstacle placement:
  - World region: [-10, 10] × [-10, 10] (world coords)
  - Drone spawn: (-12, 0, 0.1) — avoid 5m radius
  - Poisson disk sampling, min_dist=2.0m between obstacle centers
  - Cylinders: r ∈ [0.15, 2.0]m, h ∈ [1, 8]m
  - Boxes: side ∈ [0.5, 4.0]m, h ∈ [1, 6]m
  - Obstacle z-pose = h/2 (sitting on ground)

Output:
  {output_dir}/env_{id:03d}_obs{count}.world   — Gazebo SDF
  {output_dir}/env_{id:03d}_obs{count}.obstacles.json — metadata
  {output_dir}/manifest.json — summary of all worlds

Usage:
  python3 generate_worlds_v8_3.py [--output_dir ./worlds_v8] [--seed 42]
"""

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

SPAWN_WORLD = (-12.0, 0.0)   # drone spawn in world frame
SPAWN_CLEARANCE = 5.0         # meters, no obstacles within this of spawn
OBS_REGION = (-10.0, 10.0)    # obstacle placement bounds (x and y)
MIN_DIST_BETWEEN_OBS = 2.0    # Poisson disk minimum distance
MAX_POISSON_ATTEMPTS = 200    # rejection sampling attempts per point

# Obstacle size ranges
CYL_R_MIN, CYL_R_MAX = 0.15, 2.0
CYL_H_MIN, CYL_H_MAX = 1.0, 8.0
BOX_S_MIN, BOX_S_MAX = 0.5, 4.0
BOX_H_MIN, BOX_H_MAX = 1.0, 6.0

# Distribution (tier_name, fraction, obs_min, obs_max)
DISTRIBUTION = [
    ("empty",       8,  0,  0),
    ("sparse",     12,  1,  5),
    ("medium_A",   15,  6, 12),
    ("medium_B",   13, 13, 20),
    ("dense",       9, 21, 30),
    ("super_dense",  2, 31, 40),
    ("special",      1, -1, -1),  # special: handled separately
]

# Gazebo material colors for visual variety
COLORS = [
    ("Gazebo/Grey",    "0.5 0.5 0.5 1"),
    ("Gazebo/Blue",    "0.447 0.624 0.812 1"),
    ("Gazebo/Red",     "0.8 0.2 0.2 1"),
    ("Gazebo/Green",   "0.2 0.8 0.2 1"),
    ("Gazebo/Orange",  "0.9 0.5 0.1 1"),
    ("Gazebo/White",   "0.9 0.9 0.9 1"),
    ("Gazebo/DarkGrey","0.3 0.3 0.3 1"),
]


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
) -> List[Tuple[float, float]]:
    """
    Sample n points in [bounds[0], bounds[1]]^2 with minimum distance
    between points and away from exclusion zones.
    
    exclusion_zones: list of (cx, cy, radius) to avoid
    existing: previously placed points to respect min_dist from
    """
    points = list(existing) if existing else []
    
    for _ in range(n):
        placed = False
        for _ in range(MAX_POISSON_ATTEMPTS):
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
            # Could not place all obstacles; return what we have
            break
    
    # Return only the newly placed points (exclude existing)
    n_existing = len(existing) if existing else 0
    return points[n_existing:]


# ─────────────────────────────────────────────────────────────────────────────
# Obstacle generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_random_obstacle(
    name: str, x: float, y: float, rng: random.Random
) -> Obstacle:
    """Generate a random cylinder or box obstacle at (x, y)."""
    shape = rng.choice(["cylinder", "cylinder", "box"])  # 2:1 cylinder bias
    yaw = rng.uniform(0, math.pi)  # random orientation (matters for boxes)
    
    if shape == "cylinder":
        r = rng.uniform(CYL_R_MIN, CYL_R_MAX)
        h = rng.uniform(CYL_H_MIN, CYL_H_MAX)
        return Obstacle(
            name=name, shape="cylinder",
            x=x, y=y, z=h/2, yaw=yaw,
            radius=r, height=h,
            bounding_r=r,
        )
    else:
        sx = rng.uniform(BOX_S_MIN, BOX_S_MAX)
        sy = rng.uniform(BOX_S_MIN, BOX_S_MAX)
        sz = rng.uniform(BOX_H_MIN, BOX_H_MAX)
        return Obstacle(
            name=name, shape="box",
            x=x, y=y, z=sz/2, yaw=yaw,
            size_x=sx, size_y=sy, size_z=sz,
            bounding_r=math.hypot(sx, sy) / 2,
        )


def generate_corridor_world(rng: random.Random) -> List[Obstacle]:
    """Generate a special corridor/funnel world."""
    obstacles = []
    idx = 0
    
    # Two parallel walls creating a corridor along x-axis
    # Corridor narrows from 6m wide to 2m wide (funnel)
    for side in [-1, 1]:
        for xi in range(-8, 9, 2):
            # Width narrows linearly from 3.0 at x=-8 to 1.0 at x=8
            offset = 3.0 - (xi + 8) / 16.0 * 2.0
            y = side * offset
            h = rng.uniform(2.0, 4.0)
            r = rng.uniform(0.2, 0.4)
            obs = Obstacle(
                name=f"corridor_cyl_{idx}",
                shape="cylinder",
                x=float(xi), y=y, z=h/2, yaw=0,
                radius=r, height=h, bounding_r=r,
            )
            obstacles.append(obs)
            idx += 1
    
    # Add a few random obstacles inside the corridor (with spacing check)
    for attempt in range(20):
        if len([o for o in obstacles if "inner" in o.name]) >= 3:
            break
        x = rng.uniform(-4, 4)
        y = rng.uniform(-0.8, 0.8)  # narrower range to avoid walls
        r = rng.uniform(0.15, 0.25)
        h = rng.uniform(1.5, 3.0)
        # Check distance to all existing obstacles
        too_close = False
        for existing in obstacles:
            d = math.hypot(x - existing.x, y - existing.y)
            if d < MIN_DIST_BETWEEN_OBS * 0.9:
                too_close = True
                break
        if too_close:
            continue
        idx = len(obstacles)
        obs = Obstacle(
            name=f"corridor_inner_{idx}",
            shape="cylinder",
            x=x, y=y, z=h/2, yaw=0,
            radius=r, height=h, bounding_r=r,
        )
        obstacles.append(obs)
        idx += 1
    
    return obstacles


# ─────────────────────────────────────────────────────────────────────────────
# SDF generation
# ─────────────────────────────────────────────────────────────────────────────

def obstacle_to_sdf(obs: Obstacle) -> str:
    """Convert an Obstacle to Gazebo SDF model XML."""
    color_name, ambient = random.choice(COLORS)
    
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


def generate_world_sdf(obstacles: List[Obstacle]) -> str:
    """Generate a complete Gazebo .world SDF file."""
    obstacle_xml = "\n".join(obstacle_to_sdf(obs) for obs in obstacles)
    
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
# World plan generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_world_plan(rng: random.Random) -> List[WorldSpec]:
    """Generate the list of 60 worlds with obstacle counts per tier."""
    worlds = []
    world_id = 0
    
    for tier_name, count, obs_min, obs_max in DISTRIBUTION:
        if tier_name == "special":
            # Special corridor world
            worlds.append(WorldSpec(
                world_id=world_id,
                tier=tier_name,
                n_obstacles=-1,  # determined by generation
                filename="",
            ))
            world_id += 1
            continue
        
        for i in range(count):
            if obs_min == obs_max:
                n_obs = obs_min
            else:
                n_obs = rng.randint(obs_min, obs_max)
            
            worlds.append(WorldSpec(
                world_id=world_id,
                tier=tier_name,
                n_obstacles=n_obs,
                filename="",
            ))
            world_id += 1
    
    # Shuffle to avoid tier-sorted ordering in filenames
    # But keep IDs sequential
    rng.shuffle(worlds)
    for i, w in enumerate(worlds):
        w.world_id = i
    
    return worlds


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_world(spec: WorldSpec) -> List[str]:
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
            issues.append(f"{obs.name}: too close to spawn ({d_spawn:.2f}m < {SPAWN_CLEARANCE}m)")
        
        # Check size ranges
        if obs.shape == "cylinder":
            if not (CYL_R_MIN <= obs.radius <= CYL_R_MAX):
                issues.append(f"{obs.name}: radius {obs.radius:.2f} out of range")
            if not (CYL_H_MIN <= obs.height <= CYL_H_MAX):
                issues.append(f"{obs.name}: height {obs.height:.2f} out of range")
        elif obs.shape == "box":
            if not (BOX_S_MIN <= obs.size_x <= BOX_S_MAX):
                issues.append(f"{obs.name}: size_x {obs.size_x:.2f} out of range")
            if not (BOX_S_MIN <= obs.size_y <= BOX_S_MAX):
                issues.append(f"{obs.name}: size_y {obs.size_y:.2f} out of range")
            if not (BOX_H_MIN <= obs.size_z <= BOX_H_MAX):
                issues.append(f"{obs.name}: size_z {obs.size_z:.2f} out of range")
        
        # Check pairwise min distance
        for j in range(i + 1, len(spec.obstacles)):
            obs2 = spec.obstacles[j]
            d = math.hypot(obs.x - obs2.x, obs.y - obs2.y)
            if d < MIN_DIST_BETWEEN_OBS * 0.9:  # 10% tolerance
                issues.append(
                    f"{obs.name}↔{obs2.name}: distance {d:.2f}m < "
                    f"{MIN_DIST_BETWEEN_OBS}m"
                )
    
    return issues


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate v8.3 Gazebo worlds")
    parser.add_argument("--output_dir", default="./worlds_v8",
                        help="Output directory for .world files")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--validate_only", action="store_true",
                        help="Only validate existing worlds in output_dir")
    args = parser.parse_args()
    
    rng = random.Random(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate world plan
    worlds = generate_world_plan(rng)
    
    # Exclusion zones: spawn point
    exclusion_zones = [
        (SPAWN_WORLD[0], SPAWN_WORLD[1], SPAWN_CLEARANCE),
    ]
    
    manifest = []
    total_issues = 0
    
    print(f"Generating {len(worlds)} worlds in {args.output_dir}/")
    print(f"Seed: {args.seed}")
    print(f"Spawn: {SPAWN_WORLD}, clearance: {SPAWN_CLEARANCE}m")
    print(f"Obstacle region: [{OBS_REGION[0]}, {OBS_REGION[1]}]²")
    print(f"Min distance: {MIN_DIST_BETWEEN_OBS}m")
    print()
    
    # Tier statistics
    tier_counts = {}
    for w in worlds:
        tier_counts[w.tier] = tier_counts.get(w.tier, 0) + 1
    print("Tier distribution:")
    for tier, count in tier_counts.items():
        print(f"  {tier:12s}: {count} worlds")
    print()
    
    for spec in worlds:
        # Generate obstacles
        if spec.tier == "special":
            spec.obstacles = generate_corridor_world(rng)
            spec.n_obstacles = len(spec.obstacles)
        elif spec.n_obstacles == 0:
            spec.obstacles = []
        else:
            # Poisson disk sampling for positions
            positions = poisson_disk_sample(
                n=spec.n_obstacles,
                bounds=OBS_REGION,
                min_dist=MIN_DIST_BETWEEN_OBS,
                exclusion_zones=exclusion_zones,
                rng=rng,
            )
            
            spec.n_obstacles = len(positions)  # may be less if couldn't place all
            spec.obstacles = []
            for i, (px, py) in enumerate(positions):
                obs = generate_random_obstacle(
                    name=f"obs_{i:02d}",
                    x=px, y=py, rng=rng,
                )
                spec.obstacles.append(obs)
        
        # Set filename
        spec.filename = f"env_{spec.world_id:03d}_obs{spec.n_obstacles}.world"
        
        # Validate
        issues = validate_world(spec)
        total_issues += len(issues)
        
        status = "✓" if not issues else f"✗ ({len(issues)} issues)"
        print(f"  [{spec.world_id:03d}] {spec.tier:12s} "
              f"obs={spec.n_obstacles:2d}  {spec.filename:30s}  {status}")
        for issue in issues:
            print(f"         ⚠ {issue}")
        
        # Generate SDF
        sdf_content = generate_world_sdf(spec.obstacles)
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
        
        json_path = os.path.join(
            args.output_dir,
            spec.filename.replace(".world", ".obstacles.json")
        )
        with open(json_path, "w") as f:
            json.dump({
                "world_id": spec.world_id,
                "tier": spec.tier,
                "n_obstacles": spec.n_obstacles,
                "spawn_world": list(SPAWN_WORLD),
                "obstacle_region": list(OBS_REGION),
                "obstacles": obs_data,
            }, f, indent=2)
        
        # Manifest entry
        manifest.append({
            "world_id": spec.world_id,
            "tier": spec.tier,
            "n_obstacles": spec.n_obstacles,
            "filename": spec.filename,
            "n_issues": len(issues),
        })
    
    # Write manifest
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({
            "version": "v8.3",
            "seed": args.seed,
            "total_worlds": len(worlds),
            "spawn_world": list(SPAWN_WORLD),
            "spawn_clearance": SPAWN_CLEARANCE,
            "obstacle_region": list(OBS_REGION),
            "min_dist": MIN_DIST_BETWEEN_OBS,
            "tier_distribution": tier_counts,
            "worlds": manifest,
        }, f, indent=2)
    
    # Summary
    print()
    print(f"{'='*60}")
    print(f"Generated {len(worlds)} worlds → {args.output_dir}/")
    print(f"Total validation issues: {total_issues}")
    if total_issues == 0:
        print("All worlds passed geometric validation ✓")
    else:
        print(f"⚠ {total_issues} issues found — review above")
    
    # Tier summary with obstacle stats
    print()
    print("Summary by tier:")
    for tier_name, _, _, _ in DISTRIBUTION:
        tier_worlds = [m for m in manifest if m["tier"] == tier_name]
        if not tier_worlds:
            continue
        obs_counts = [m["n_obstacles"] for m in tier_worlds]
        print(f"  {tier_name:12s}: {len(tier_worlds):2d} worlds, "
              f"obs range [{min(obs_counts):2d}, {max(obs_counts):2d}], "
              f"mean {sum(obs_counts)/len(obs_counts):.1f}")
    
    print()
    print(f"Manifest: {manifest_path}")
    print(f"Done.")
    
    return 0 if total_issues == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
