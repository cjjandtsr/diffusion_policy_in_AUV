#!/usr/bin/env python3
"""Plot top-down view of all rl_test worlds with obstacles, start, goal."""
import json, os, sys, numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from diffusion_policy_3d.rl.routea_rl_env_adapter_v1 import (
    load_routea_rl_admission_manifest, build_role_buckets)

WORLDS_DIR = os.path.expanduser('~/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds')
MANIFEST = os.path.expanduser(
    '~/AUV/program/reports/routeA_rl_env_v1/calibration/admission_manifest_kinematic_v1.json')

manifest = load_routea_rl_admission_manifest(MANIFEST)
buckets = build_role_buckets(manifest, 'rl_test')

# Collect all worlds
all_worlds = []
for role in ['anticipatory', 'dense_recovery', 'support_expansion', 'yaw_lateral_recovery']:
    for w in buckets.get(role, []):
        all_worlds.append((role, w))

n = len(all_worlds)
cols = 4
rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4))
axes = axes.flatten() if n > 1 else [axes]

for idx, (role, w) in enumerate(all_worlds):
    ax = axes[idx]
    json_path = os.path.join(WORLDS_DIR, f'{w.world_uid}.world.obstacles.json')
    with open(json_path) as f:
        data = json.load(f)

    # Plot obstacles (cylinders)
    for obs in data.get('obstacles', []):
        x, y = obs.get('x', 0), obs.get('y', 0)
        r = obs.get('radius', obs.get('size', 0.5))
        circle = plt.Circle((x, y), r, color='gray', alpha=0.7, ec='black', lw=0.5)
        ax.add_patch(circle)

    # Spawn region center
    spawn = data.get('spawn_world', [0, 0, 5])
    if isinstance(spawn, list) and len(spawn) >= 2:
        sx, sy = spawn[0], spawn[1]
        ax.plot(sx, sy, 'go', markersize=10, label='Spawn center', zorder=5)
        # Spawn radius
        if len(spawn) >= 3:
            ax.add_patch(plt.Circle((sx, sy), spawn[2], color='green', alpha=0.1))

    # Bounds from obstacle_region
    region = data.get('obstacle_region', [-15, -15, 15, 15])
    if isinstance(region, list) and len(region) == 4:
        x_min, y_min, x_max, y_max = region
    else:
        ob_x = [o['x'] for o in data.get('obstacles', [])]
        ob_y = [o['y'] for o in data.get('obstacles', [])]
        x_min, x_max = (min(ob_x)-5, max(ob_x)+5) if ob_x else (-15, 15)
        y_min, y_max = (min(ob_y)-5, max(ob_y)+5) if ob_y else (-15, 15)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Title
    short_name = w.world_uid.split('__')[-1]
    ax.set_title(f'{role}\n{short_name}\nbc_sr={w.bc_fm_sr:.2f} bc_cr={w.bc_fm_cr:.2f}', fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

# Hide unused subplots
for idx in range(n, len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('rl_test Worlds (12 total) — Top-down view, gray=obstacles, green=start area', fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig('/home/cjj/AUV/rl_test_worlds.png', dpi=150, bbox_inches='tight')
print('Saved to ~/AUV/rl_test_worlds.png')
