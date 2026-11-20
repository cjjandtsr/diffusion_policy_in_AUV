#!/usr/bin/env python3
"""Generate the SUPER Gazebo-arm config from the official ROS1 config.

The Gazebo arm runs SUPER the way upstream runs it (timers on, ROS callbacks on),
so this is the official click_smooth_ros1.yaml with only deployment overrides:
topics are retargeted to the paper bridge and the map/robot envelope follows the
task world. Every change is reported with its YAML path, so the effective config
is auditable against the official file.

Usage:
  python3 prepare_paper_super_gazebo_config.py <output.yaml> \
      --map-size X Y Z --virtual-ceiling C --virtual-ground G \
      --max-vel V --max-acc A --robot-r R
"""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import yaml

OFFICIAL = Path('/home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy/'
                'third_party/super_official_2ad3419/super_planner/config/click_smooth_ros1.yaml')


def walk(node, path=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield path + (key,), value
            yield from walk(value, path + (key,))


def set_unique(doc, key, value, changes):
    hits = [path for path, _ in walk(doc) if path[-1] == key]
    if len(hits) != 1:
        raise SystemExit('key %r is ambiguous: %d occurrences at %s'
                         % (key, len(hits), ['.'.join(p) for p in hits]))
    path = hits[0]
    parent = doc
    for step in path[:-1]:
        parent = parent[step]
    old = parent[key]
    parent[key] = value
    changes.append(('.'.join(path), old, value))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output')
    parser.add_argument('--map-size', nargs=3, type=float, required=True)
    parser.add_argument('--virtual-ceiling', type=float, required=True)
    parser.add_argument('--virtual-ground', type=float, required=True)
    parser.add_argument('--max-vel', type=float, default=None)
    parser.add_argument('--max-acc', type=float, default=None)
    parser.add_argument('--robot-r', type=float, default=None)
    parser.add_argument('--super-cmd-topic', default='/paper/super/pos_cmd')
    parser.add_argument('--goal-topic', default='/paper_super/goal')
    parser.add_argument('--cloud-topic', default='/paper/cloud_local')
    parser.add_argument('--odom-topic', default='/paper/odom_local')
    args = parser.parse_args()

    raw = OFFICIAL.read_bytes()
    doc = yaml.safe_load(raw.decode())
    before = copy.deepcopy(doc)
    changes = []

    set_unique(doc, 'cmd_topic', args.super_cmd_topic, changes)
    set_unique(doc, 'click_goal_topic', args.goal_topic, changes)
    set_unique(doc, 'click_goal_en', True, changes)
    # No z override: the task goal's own z is what the paper judges.
    set_unique(doc, 'click_height', -10.0, changes)
    set_unique(doc, 'cloud_topic', args.cloud_topic, changes)
    set_unique(doc, 'odom_topic', args.odom_topic, changes)
    set_unique(doc, 'map_size', [float(v) for v in args.map_size], changes)
    set_unique(doc, 'virtual_ceil_height', args.virtual_ceiling, changes)
    set_unique(doc, 'virtual_ground_height', args.virtual_ground, changes)
    # Upstream ROS input path stays on for the Gazebo arm (the sync arm is the
    # one that disables it).
    for path in [p for p, _ in walk(doc) if p[-1] == 'enable']:
        parent = doc
        for step in path[:-1]:
            parent = parent[step]
        if len(path) >= 2 and path[-2] == 'ros_callback':
            changes.append(('.'.join(path), parent['enable'], True))
            parent['enable'] = True
        elif len(path) >= 2 and path[-2] == 'visualization':
            changes.append(('.'.join(path), parent['enable'], False))
            parent['enable'] = False
    set_unique(doc, 'visualization_en', False, changes)
    if args.max_vel is not None:
        set_unique(doc, 'max_vel', args.max_vel, changes)
    if args.max_acc is not None:
        set_unique(doc, 'max_acc', args.max_acc, changes)
    if args.robot_r is not None:
        set_unique(doc, 'robot_r', args.robot_r, changes)

    out = Path(args.output).resolve()
    if out.exists():
        raise SystemExit('refusing to overwrite %s' % out)
    out.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding='utf-8')
    audit = {
        'official_config': str(OFFICIAL),
        'official_config_sha256': hashlib.sha256(raw).hexdigest(),
        'effective_config': str(out),
        'effective_config_sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
        'changes': [{'path': p, 'from': f, 'to': t} for p, f, t in changes],
    }
    Path(str(out) + '.changes.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2),
                                                encoding='utf-8')
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    assert doc != before
    return 0


if __name__ == '__main__':
    sys.exit(main())
