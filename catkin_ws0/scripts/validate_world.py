#!/usr/bin/env python3
"""
validate_world.py — v8.3 §16.7: Geometric + physical validation of generated worlds
================================================================================
Run after generate_worlds_v8_3.py to verify all worlds pass constraints.
Can also validate individual .world files.

Usage:
  python3 validate_world.py worlds_v8/                    # validate all
  python3 validate_world.py worlds_v8/env_003_obs26.world  # validate one
  python3 validate_world.py worlds_v8/ --ego-flyable 5     # spot-check EGO flyability
"""

import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from typing import List, Tuple


SPAWN_WORLD = (-12.0, 0.0)
SPAWN_CLEARANCE = 5.0
OBS_REGION = (-10.0, 10.0)
MIN_DIST = 2.0
FLIGHT_HEIGHT_RANGE = (0.5, 5.0)  # drone flight altitude range


def parse_world_obstacles(world_path: str) -> List[dict]:
    """Parse obstacle models from a .world SDF file."""
    tree = ET.parse(world_path)
    root = tree.getroot()
    world = root.find("world")
    if world is None:
        return []
    
    obstacles = []
    for model in world.findall("model"):
        name = model.get("name", "")
        if name in ("ground_plane",):
            continue
        
        pose_el = model.find("pose")
        if pose_el is None or pose_el.text is None:
            continue
        pose_parts = pose_el.text.strip().split()
        x, y, z = float(pose_parts[0]), float(pose_parts[1]), float(pose_parts[2])
        yaw = float(pose_parts[5]) if len(pose_parts) > 5 else 0.0
        
        # Find geometry
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
        
        obs = {"name": name, "x": x, "y": y, "z": z, "yaw": yaw}
        if cyl is not None:
            obs["shape"] = "cylinder"
            obs["radius"] = float(cyl.find("radius").text)
            obs["height"] = float(cyl.find("length").text)
            obs["bounding_r"] = obs["radius"]
        elif box is not None:
            size = box.find("size").text.strip().split()
            obs["shape"] = "box"
            obs["size_x"] = float(size[0])
            obs["size_y"] = float(size[1])
            obs["size_z"] = float(size[2])
            obs["bounding_r"] = math.hypot(float(size[0]), float(size[1])) / 2
        else:
            obs["shape"] = "unknown"
            obs["bounding_r"] = 0
        
        obstacles.append(obs)
    
    return obstacles


def validate_obstacles(obstacles: List[dict], filename: str) -> List[str]:
    """Run all validation checks. Returns list of issues."""
    issues = []
    
    for obs in obstacles:
        # Bounds check
        if not (OBS_REGION[0] - 0.5 <= obs["x"] <= OBS_REGION[1] + 0.5):
            issues.append(f'{obs["name"]}: x={obs["x"]:.2f} out of bounds')
        if not (OBS_REGION[0] - 0.5 <= obs["y"] <= OBS_REGION[1] + 0.5):
            issues.append(f'{obs["name"]}: y={obs["y"]:.2f} out of bounds')
        
        # Spawn clearance
        d = math.hypot(obs["x"] - SPAWN_WORLD[0], obs["y"] - SPAWN_WORLD[1])
        if d < SPAWN_CLEARANCE:
            issues.append(
                f'{obs["name"]}: {d:.2f}m from spawn (need {SPAWN_CLEARANCE}m)'
            )
        
        # Height sanity
        if obs["shape"] == "cylinder":
            if obs["height"] > 12:
                issues.append(f'{obs["name"]}: height {obs["height"]:.1f}m too tall')
        elif obs["shape"] == "box":
            if obs["size_z"] > 10:
                issues.append(f'{obs["name"]}: height {obs["size_z"]:.1f}m too tall')
    
    # Pairwise distance
    for i in range(len(obstacles)):
        for j in range(i + 1, len(obstacles)):
            d = math.hypot(
                obstacles[i]["x"] - obstacles[j]["x"],
                obstacles[i]["y"] - obstacles[j]["y"],
            )
            if d < MIN_DIST * 0.8:  # allow some tolerance
                issues.append(
                    f'{obstacles[i]["name"]}↔{obstacles[j]["name"]}: '
                    f'{d:.2f}m apart (min {MIN_DIST}m)'
                )
    
    # XML well-formedness (already parsed if we got here)
    
    return issues


def check_flyability(obstacles: List[dict], n_checks: int = 20) -> dict:
    """
    Estimate EGO planner flyability by checking if straight-line paths
    between random start/goal pairs have enough clearance.
    This is a rough geometric heuristic, not a real planner check.
    """
    import random
    rng = random.Random(123)
    
    passable = 0
    total = n_checks
    
    for _ in range(total):
        # Sample start near spawn, goal in opposite direction
        sx = rng.uniform(-1, 3)  # local frame
        sy = rng.uniform(-8, 8)
        gx = rng.uniform(10, 20)
        gy = rng.uniform(-8, 8)
        
        # Convert to world frame (spawn at -12, 0)
        sx_w = sx + SPAWN_WORLD[0]
        sy_w = sy + SPAWN_WORLD[1]
        gx_w = gx + SPAWN_WORLD[0]
        gy_w = gy + SPAWN_WORLD[1]
        
        # Check 50 points along the line
        blocked = False
        for t in [i / 50 for i in range(51)]:
            px = sx_w + t * (gx_w - sx_w)
            py = sy_w + t * (gy_w - sy_w)
            
            for obs in obstacles:
                d = math.hypot(px - obs["x"], py - obs["y"])
                clearance = d - obs["bounding_r"]
                if clearance < 0.5:  # drone radius + margin
                    blocked = True
                    break
            if blocked:
                break
        
        if not blocked:
            passable += 1
    
    return {
        "straight_line_passable": passable,
        "total_checks": total,
        "passable_ratio": passable / total,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate v8.3 Gazebo worlds")
    parser.add_argument("path", help="Directory of .world files or single .world file")
    parser.add_argument("--ego-flyable", type=int, default=0,
                        help="Number of worlds to spot-check for EGO flyability")
    args = parser.parse_args()
    
    if os.path.isfile(args.path):
        files = [args.path]
    elif os.path.isdir(args.path):
        files = sorted([
            os.path.join(args.path, f)
            for f in os.listdir(args.path)
            if f.endswith(".world")
        ])
    else:
        print(f"Error: {args.path} not found")
        return 1
    
    total_issues = 0
    total_obstacles = 0
    results = []
    
    print(f"Validating {len(files)} world files...")
    print()
    
    for fpath in files:
        fname = os.path.basename(fpath)
        try:
            obstacles = parse_world_obstacles(fpath)
        except ET.ParseError as e:
            print(f"  ✗ {fname}: XML parse error: {e}")
            total_issues += 1
            continue
        
        issues = validate_obstacles(obstacles, fname)
        total_issues += len(issues)
        total_obstacles += len(obstacles)
        
        status = "✓" if not issues else f"✗ ({len(issues)})"
        print(f"  {status} {fname:35s} [{len(obstacles):2d} obstacles]")
        for iss in issues:
            print(f"       ⚠ {iss}")
        
        results.append({
            "file": fname,
            "n_obstacles": len(obstacles),
            "n_issues": len(issues),
        })
    
    print()
    print(f"{'='*55}")
    print(f"Total: {len(files)} files, {total_obstacles} obstacles, {total_issues} issues")
    
    if total_issues == 0:
        print("All worlds PASSED geometric validation ✓")
    else:
        print(f"⚠ {total_issues} issues found")
    
    # Flyability check
    if args.ego_flyable > 0:
        import random
        rng = random.Random(99)
        check_files = rng.sample(files, min(args.ego_flyable, len(files)))
        
        print()
        print(f"EGO flyability spot-check ({len(check_files)} worlds):")
        for fpath in check_files:
            fname = os.path.basename(fpath)
            obstacles = parse_world_obstacles(fpath)
            result = check_flyability(obstacles)
            ratio = result["passable_ratio"]
            status = "✓" if ratio >= 0.5 else "⚠"
            print(f"  {status} {fname:35s} "
                  f"straight-line passable: {result['straight_line_passable']}"
                  f"/{result['total_checks']} ({ratio:.0%})")
    
    return 0 if total_issues == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
