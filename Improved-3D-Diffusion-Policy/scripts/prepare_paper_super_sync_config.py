#!/usr/bin/env python3
"""Generate the effective SUPER config for the synchronous kinematics arm.

Reuses the delivered prepare_paper_super.py (config command) unchanged, keeps
its output verbatim as <output>.base.yaml, then writes the effective YAML with
the sync-specific overrides required by the adapter design (P1_design):
  fsm.timer_en = false
  super_planner.visualization_en = false
  rog_map.ros_callback.enable = false
Upstream ROGMap::updateMap refuses to run while ros_callback is enabled, and
the sync node creates no timers/subscribers at all; the effective file is the
config the node actually loads and is recorded in the run identity.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    for name in ['cloud-topic', 'odom-topic', 'goal-topic']:
        parser.add_argument('--' + name, required=True)
    for name in ['radius', 'max-velocity', 'max-acceleration', 'ground',
                 'ceiling', 'map-height', 'min-range', 'max-range']:
        parser.add_argument('--' + name, type=float, required=True)
    args = parser.parse_args()

    base = args.output.with_suffix(args.output.suffix + '.base.yaml')
    command = [sys.executable,
               str(args.repo / 'scripts/prepare_paper_super.py'), 'config',
               '--source', str(args.repo / 'third_party/super_official_2ad3419'),
               '--output', str(base),
               '--cloud-topic', args.cloud_topic,
               '--odom-topic', args.odom_topic,
               '--goal-topic', args.goal_topic,
               '--radius', repr(args.radius),
               '--max-velocity', repr(args.max_velocity),
               '--max-acceleration', repr(args.max_acceleration),
               '--ground', repr(args.ground),
               '--ceiling', repr(args.ceiling),
               '--map-height', repr(args.map_height),
               '--min-range', repr(args.min_range),
               '--max-range', repr(args.max_range)]
    subprocess.run(command, check=True)

    cfg = yaml.safe_load(base.read_text(encoding='utf-8'))
    cfg['fsm']['timer_en'] = False
    cfg['super_planner']['visualization_en'] = False
    cfg['rog_map']['ros_callback']['enable'] = False
    # ROGMapROS creates a wall-clock viz timer (time_rate) when its own
    # visualization section is enabled; the sync node must create no timers.
    cfg['rog_map']['visualization']['enable'] = False
    with args.output.open('x', encoding='utf-8') as stream:
        yaml.safe_dump(cfg, stream, sort_keys=False)
    print('SUPER_SYNC_CONFIG_WRITTEN', args.output, flush=True)
    print('SUPER_SYNC_CONFIG_BASE', base, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
