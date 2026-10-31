"""Spot-check figures (spec §8 item 3): 3 tasks per stratum per dimension, seed 20261031.

Top view: obstacles, corridor (w=1.0 m), route band (w=3.0 m), smoothed shortest path, S/G.
Side view: z against distance along S-G for the path, plus obstacles that touch the route band.
Usage: python plots.py task_geometry.json task_metrics.json bins.json <out_dir>
"""
import json
import math
import os
import random
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

plt.rcParams['font.family'] = ['DejaVu Sans']
DIMS = ('density_1', 'bottleneck', 'detour')
LABEL_EN = {'无障碍': 'empty', '低': 'low', '中': 'mid', '高': 'high', '窄': 'narrow', '宽': 'wide',
            '带内不可达': 'band-unreachable', '无需绕行': 'no-detour', '小绕行': 'small-detour',
            '大绕行': 'large-detour'}


def rect(u, v, S, L, w):
    pts = []
    for t, s in ((0, -w), (L, -w), (L, w), (0, w)):
        pts.append((S[0] + t * u[0] + s * v[0], S[1] + t * u[1] + s * v[1]))
    return pts


def draw(ax_top, ax_side, task, m):
    S, G = task['start'], task['goal']
    d = (G[0] - S[0], G[1] - S[1])
    L = math.hypot(*d)
    u = (d[0] / L, d[1] / L)
    v = (-u[1], u[0])
    for o in task['geometry']:
        if o['shape'] == 'cylinder':
            ax_top.add_patch(Circle((o['x'], o['y']), o['radius'], color='0.4'))
            hx = o['radius']
            top = o['z'] + o['height'] / 2
        else:
            c, s = math.cos(o['yaw']), math.sin(o['yaw'])
            corners = [(o['x'] + c * a - s * b, o['y'] + s * a + c * b)
                       for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))
                       for a, b in [(a * o['size_x'] / 2, b * o['size_y'] / 2)]]
            ax_top.add_patch(Polygon(corners, color='0.4'))
            hx = math.hypot(o['size_x'], o['size_y']) / 2
            top = o['z'] + o['size_z'] / 2
        t = (o['x'] - S[0]) * u[0] + (o['y'] - S[1]) * u[1]
        s = (o['x'] - S[0]) * v[0] + (o['y'] - S[1]) * v[1]
        if abs(s) - hx <= 3.0 and -1 <= t <= L + 1:
            bottom = top - (o['height'] if o['shape'] == 'cylinder' else o['size_z'])
            ax_side.add_patch(Polygon([(t - hx, bottom), (t + hx, bottom), (t + hx, top), (t - hx, top)],
                                      color='0.4' if abs(s) - hx <= 1.0 else '0.75'))
    ax_top.add_patch(Polygon(rect(u, v, S, L, 1.0), fill=False, ec='tab:blue', lw=1))
    ax_top.add_patch(Polygon(rect(u, v, S, L, 3.0), fill=False, ec='tab:cyan', lw=0.8, ls='--'))
    path = m['path'] or [S, G]
    ax_top.plot([p[0] for p in path], [p[1] for p in path], 'r-', lw=1.2)
    ax_top.plot(S[0], S[1], 'go'); ax_top.plot(G[0], G[1], 'r*', ms=10)
    ax_top.set_aspect('equal'); ax_top.autoscale_view()
    ts = [(p[0] - S[0]) * u[0] + (p[1] - S[1]) * u[1] for p in path]
    ax_side.plot(ts, [p[2] for p in path], 'r-', lw=1.2)
    ax_side.plot(0, S[2], 'go'); ax_side.plot(L, G[2], 'r*', ms=10)
    ax_side.axhline(max(S[2], G[2]) + 1.0, color='tab:cyan', ls='--', lw=0.8)
    ax_side.set_xlim(-1.5, L + 1.5); ax_side.set_ylim(0, 7)
    ax_side.set_xlabel('distance along S-G (m)'); ax_side.set_ylabel('z (m)')
    b = m['b_band_m']
    ax_top.set_title('%s\nrho1=%.3f  b_band=%s  kappa=%.3f  blocked=%s' % (
        task['world'].split('__', 2)[-1][:40], m['rho_1.0'], 'None' if b is None else '%.2f' % b,
        m['kappa'], m['blocked']), fontsize=7)


def main(geom_path, metrics_path, bins_path, out_dir):
    geom = {t['task_id']: t for t in json.load(open(geom_path))}
    metrics = {m['task_id']: m for m in json.load(open(metrics_path))}
    labels = json.load(open(bins_path))['labels']
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(20261031)
    picked = {}
    for dim in DIMS:
        groups = {}
        for tid in sorted(labels):
            groups.setdefault(labels[tid][dim], []).append(tid)
        rows = []
        for lab in sorted(groups):
            chosen = rng.sample(groups[lab], min(3, len(groups[lab])))
            rows += [(lab, tid) for tid in chosen]
        picked[dim] = rows
        fig, axes = plt.subplots(len(rows), 2, figsize=(10, 3.2 * len(rows)))
        for (lab, tid), (a, b) in zip(rows, axes):
            draw(a, b, geom[tid], metrics[tid])
            b.set_title('stratum: %s (%s)' % (LABEL_EN.get(lab, lab), metrics[tid]['split']), fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'spotcheck_%s.png' % dim), dpi=80)
        plt.close(fig)
    json.dump(picked, open(os.path.join(out_dir, 'spotcheck_tasks.json'), 'w'), ensure_ascii=False, indent=1)
    print('figures written', {k: len(v) for k, v in picked.items()})


if __name__ == '__main__':
    main(*sys.argv[1:5])
