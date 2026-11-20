#!/usr/bin/env python3
"""
generate_obstacles_json_routeA_rl_v1.py
========================================
Parse Route A RL v1 .world SDF files and extract obstacle geometry into
.obstacles.json format that KinematicUAVEnv.load_obstacles() can read.

Output: {world_file}.obstacles.json alongside each .world file.
"""

from __future__ import annotations

import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List

WORLDS_DIR = Path("/home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds")


def parse_world_sdf(world_path: Path) -> Dict[str, Any]:
    """Parse a .world SDF file and extract obstacle specs."""
    tree = ET.parse(str(world_path))
    root = tree.getroot()

    obstacles: List[Dict[str, Any]] = []

    for model in root.iter("model"):
        name = model.get("name", "")
        if name == "ground_plane":
            continue

        # Get pose
        pose_elem = model.find("pose")
        if pose_elem is None:
            # Check in link
            link = model.find("link")
            if link is not None:
                pose_elem = link.find("pose")
        if pose_elem is None:
            print(f"  [WARN] {name}: no pose found, skipping")
            continue

        pose_str = pose_elem.text.strip() if pose_elem.text else "0 0 0 0 0 0"
        pose_parts = pose_str.split()
        x, y, z = float(pose_parts[0]), float(pose_parts[1]), float(pose_parts[2])
        # roll=pose_parts[3], pitch=pose_parts[4]
        yaw = float(pose_parts[5]) if len(pose_parts) >= 6 else 0.0

        # Find geometry: search in collision > geometry (preferred), fallback to any geometry
        geom = None
        for collision in model.iter("collision"):
            g = collision.find("geometry")
            if g is not None:
                geom = g
                break
        if geom is None:
            for g in model.iter("geometry"):
                geom = g
                break

        if geom is None:
            print(f"  [WARN] {name}: no geometry found, skipping")
            continue

        cyl = geom.find("cylinder")
        box = geom.find("box")

        if cyl is not None:
            radius_elem = cyl.find("radius")
            length_elem = cyl.find("length")
            radius = float(radius_elem.text.strip()) if radius_elem is not None and radius_elem.text else 0.0
            height = float(length_elem.text.strip()) if length_elem is not None and length_elem.text else 0.0
            obstacles.append({
                "name": name,
                "shape": "cylinder",
                "x": x, "y": y, "z": z,
                "yaw": yaw,
                "radius": radius,
                "height": height,
            })
        elif box is not None:
            size_elem = box.find("size")
            if size_elem is not None and size_elem.text:
                size_parts = size_elem.text.strip().split()
                sx, sy, sz = float(size_parts[0]), float(size_parts[1]), float(size_parts[2])
            else:
                sx = sy = sz = 0.0
            obstacles.append({
                "name": name,
                "shape": "box",
                "x": x, "y": y, "z": z,
                "yaw": yaw,
                "size_x": sx, "size_y": sy, "size_z": sz,
            })
        else:
            print(f"  [WARN] {name}: unknown geometry type, skipping")

    # Determine spawn from metadata if available
    metadata_path = Path(str(world_path).replace(".world", ".metadata.json"))
    spawn_world = [0.0, 0.0, 2.0]
    obstacle_region = [-10.0, -10.0, 10.0, 10.0]
    world_uid = world_path.stem

    if metadata_path.exists():
        with open(metadata_path) as f:
            meta = json.load(f)
        spawn_world = meta.get("start_world", spawn_world)
        ob = meta.get("obstacle_region", obstacle_region)
        if isinstance(ob, list) and len(ob) >= 4:
            obstacle_region = ob[:4]
        world_uid = meta.get("world_uid", world_uid)

    return {
        "world_id": world_uid,
        "tier": "routeA_rl_v1",
        "n_obstacles": len(obstacles),
        "spawn_world": spawn_world,
        "obstacle_region": obstacle_region,
        "obstacles": obstacles,
    }


def main() -> int:
    world_files = sorted(WORLDS_DIR.glob("routeA_rl_v1__*.world"))
    if not world_files:
        print("No Route A RL v1 world files found.")
        return 1

    print(f"Found {len(world_files)} world files")
    generated = 0
    skipped = 0

    for wf in world_files:
        out_path = Path(str(wf) + ".obstacles.json")
        if out_path.exists():
            print(f"  SKIP {wf.name} — already exists")
            skipped += 1
            continue

        try:
            data = parse_world_sdf(wf)
            with open(out_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"  OK   {wf.name} → {len(data['obstacles'])} obstacles")
            generated += 1
        except Exception as e:
            print(f"  FAIL {wf.name}: {e}")
            return 1

    print(f"\nDone: {generated} generated, {skipped} skipped, {len(world_files)} total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
