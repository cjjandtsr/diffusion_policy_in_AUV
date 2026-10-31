"""Offline upper bound for terminal straight-line guidance
(guide: Improved-3D-Diffusion-Policy/zhidao/2026-10-30_末段直线引导_离线上限统计_执行指导_v1_0.md).

dev420 only; P0, P2, P4, P6 (policy numbering spec 2026-10-31). Ground-truth obstacles, no
perception error -> an UPPER BOUND, not a prediction.
  entry P_r   first executed-path point with |p - goal| < r, before any termination event
  clear       segment_obstacle_collision(P_r, goal, obstacles, thresh) is None
  feasible    ceil(|P_r - goal| / v_cap) <= 60 - i_r   (decision budget from the evaluator)
  v_cap       main: p50 of per-decision displacement of that policy (all its episodes)
              axis: max(ceil(dxy / p95 dxy), ceil(dz / p95 dz))
  rescuable   failed episode that entered, clear and feasible
  harmed      reached episode that entered, clear but infeasible
Usage: python upper_bound.py <repo_root> <out.json>
"""
import collections
import glob
import json
import math
import random
import sys

import numpy as np

FIELDS = ('name', 'shape', 'x', 'y', 'z', 'yaw', 'radius', 'height', 'size_x', 'size_y', 'size_z')
MAX_DECISIONS = 60
RADII = (1.0, 1.5, 2.0)


def load(directory):
    out = {}
    for f in sorted(glob.glob(directory + '/traces/*.jsonl')):
        st, trs, reason = None, [], None
        for line in open(f):
            x = json.loads(line)
            k = x.get('type')
            if k == 'episode_start':
                st = x
            elif k == 'transition':
                trs.append(x)
                reason = x.get('done_reason')
            elif k == 'episode_end':
                reason = x['done_reason']
        if reason == 'none':
            continue
        out[(st['task_id'], int(st['repeat']))] = (st, trs, reason)
    return out


def classify(st, trs, reason):
    g = st['task']['goal_pos']
    pts = []
    for x in trs:
        pts += x['executed_path'] if not pts else x['executed_path'][1:]
    d = [math.dist(p, g) for p in pts]
    i = int(np.argmin(d))
    if reason == 'reached':
        return 'reached'
    if reason in ('collision', 'ground_crash'):
        return 'collision'
    return 'below_goal_timeout' if d[i] < 1.0 and pts[i][2] - g[2] < -0.3 else 'other_timeout'


def main(repo, out_path):
    sys.path.insert(0, repo + '/Improved-3D-Diffusion-Policy')
    from diffusion_policy_3d.env.collision_geom import ObstacleSpec, segment_obstacle_collision
    base = repo + '/jindu/routea_a2_selective_keep_run01/full/'
    pols = {'P0': base + 'bc', 'P2': base + 'rl_391a6656a0f7_u128',
            'P4': base + 'rl_910559bb5de2_u256', 'P6': base + 'rl_d987235a0cbb_u256'}
    result = {}
    for pol, d in pols.items():
        eps = load(d)
        steps, dxy, dz = [], [], []
        for st, trs, _ in eps.values():
            for x in trs:
                a, b = x['pre_state']['position'], x['post_state']['position']
                steps.append(math.dist(a, b))
                dxy.append(math.hypot(b[0] - a[0], b[1] - a[1]))
                dz.append(abs(b[2] - a[2]))
        vcap = float(np.median(steps))
        vxy, vz = float(np.quantile(dxy, .95)), float(np.quantile(dz, .95))
        res = dict(v_cap_m_per_decision=vcap, p95_dxy=vxy, p95_dz=vz,
                   outcomes=dict(collections.Counter(classify(*e) for e in eps.values())), radii={})
        for thresh in (0.1, 0.3):
            for r in RADII:
                cells = collections.Counter()
                rescued_by = {'main': collections.Counter(), 'axis': collections.Counter()}
                rescued_worlds = set()
                examples = collections.defaultdict(list)
                for key, (st, trs, reason) in eps.items():
                    cls = classify(st, trs, reason)
                    g = st['task']['goal_pos']
                    obs = [ObstacleSpec(**{k: o[k] for k in FIELDS}) for o in st['geometry']]
                    entry = None
                    for i, x in enumerate(trs):
                        path = x['executed_path']
                        last = i == len(trs) - 1
                        limit = len(path)
                        if last and x.get('event_segment') is not None:
                            limit = x['event_segment'] + 1  # points up to the start of the event segment
                        for p in path[:limit]:
                            if math.dist(p, g) < r:
                                entry = (i, p)
                                break
                        if entry:
                            break
                    if entry is None:
                        cells[cls + '|not_entered'] += 1
                        continue
                    i, p = entry
                    cells[cls + '|entered'] += 1
                    if segment_obstacle_collision(p, g, obs, thresh) is not None:
                        cells[cls + '|blocked_line'] += 1
                        if cls != 'reached':
                            examples['blocked'].append(key)
                        continue
                    left = MAX_DECISIONS - i
                    dist = math.dist(p, g)
                    need_main = math.ceil(dist / vcap)
                    need_axis = max(math.ceil(math.hypot(g[0] - p[0], g[1] - p[1]) / vxy),
                                    math.ceil(abs(g[2] - p[2]) / vz))
                    for name, need in (('main', need_main), ('axis', need_axis)):
                        if need <= left:
                            if cls != 'reached':
                                rescued_by[name][cls] += 1
                                if name == 'main':
                                    rescued_worlds.add(st['world_id'])
                                    examples['rescuable'].append(key)
                        elif cls == 'reached':
                            rescued_by[name]['harmed'] += 1
                reached = res['outcomes'].get('reached', 0)
                row = dict(cells=dict(cells),
                           rescuable_main=dict(rescued_by['main']), rescuable_axis=dict(rescued_by['axis']),
                           system_upper_main=reached + sum(v for k, v in rescued_by['main'].items() if k != 'harmed')
                           - rescued_by['main']['harmed'],
                           system_upper_axis=reached + sum(v for k, v in rescued_by['axis'].items() if k != 'harmed')
                           - rescued_by['axis']['harmed'],
                           rescuable_worlds=len(rescued_worlds))
                if thresh == 0.1 and r == 1.5:
                    rng = random.Random(20261030)
                    row['spot_check'] = {k: [list(x) for x in rng.sample(v, min(5, len(v)))] for k, v in examples.items()}
                res['radii']['thresh%.1f_r%.1f' % (thresh, r)] = row
        result[pol] = res
        print(pol, 'outcomes', res['outcomes'], 'v_cap %.2f p95 dxy %.2f dz %.2f' % (vcap, vxy, vz))
        for k, row in res['radii'].items():
            print('   ', k, 'rescuable(main)', row['rescuable_main'], '-> system upper', row['system_upper_main'],
                  '| axis', row['rescuable_axis'], '->', row['system_upper_axis'], '| worlds', row['rescuable_worlds'])
    json.dump(result, open(out_path, 'w'), indent=1)


if __name__ == '__main__':
    main(*sys.argv[1:3])
