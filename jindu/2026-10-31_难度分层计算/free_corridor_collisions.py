"""Where do collisions happen in tasks whose 1.0 m corridor contains no obstacle?

Collision point = executed_path[event_segment] + event_fraction * (next - current) of the
terminal transition. Reports lateral offset from the horizontal S-G line, axial position
(0 = start, 1 = goal), and the nearest obstacle's lateral offset and footprint reach.
Duplicate attempt files are de-duplicated by (task, repeat), later attempt wins.
Usage: python free_corridor_collisions.py <main_checkout_root> <out.json>
"""
import collections
import glob
import json
import math
import sys

import metrics as M
import stratify as S


def main(main_root, out_path):
    labels = json.load(open('bins.json'))['labels']
    geo = {t['task_id']: t for t in json.load(open('task_geometry.json'))}
    rows = {}
    for split, pols in S.sources(main_root).items():
        for pol, d in pols.items():
            for f in sorted(glob.glob(d + '/traces/*.jsonl')):
                st, last, end = None, None, None
                for line in open(f):
                    x = json.loads(line)
                    kind = x.get('type')
                    if kind == 'episode_start':
                        st = x
                    elif kind == 'transition':
                        last = x
                    elif kind == 'episode_end':
                        end = x
                tid = st['task_id']
                if labels[tid]['density_1'] != '无障碍':
                    continue
                t = geo[tid]
                s, g = t['start'], t['goal']
                L = math.hypot(g[0] - s[0], g[1] - s[1])
                u = ((g[0] - s[0]) / L, (g[1] - s[1]) / L)
                lat = lambda p: -(p[0] - s[0]) * u[1] + (p[1] - s[1]) * u[0]
                axial = lambda p: ((p[0] - s[0]) * u[0] + (p[1] - s[1]) * u[1]) / L
                outcome = end['done_reason'] if end else last['done_reason']
                r = dict(split=split, policy=pol, task_id=tid, repeat=int(st['repeat']), outcome=outcome)
                if outcome == 'collision':
                    path, seg, fr = last['executed_path'], last['event_segment'], last['event_fraction']
                    p = [path[seg][i] + fr * (path[seg + 1][i] - path[seg][i]) for i in range(3)]
                    best = None
                    for o in t['geometry']:
                        hd = float(M.horizontal_distance(o, p[0], p[1]))
                        vz = max(abs(p[2] - o['z']) - M._half_height(o), 0.0)
                        if best is None or math.hypot(hd, vz) < best[0]:
                            best = (math.hypot(hd, vz), o)
                    o = best[1]
                    r.update(point=p, lateral_m=lat(p), axial=axial(p), goal_z=g[2],
                             obstacle=o['name'], obstacle_lateral_m=lat((o['x'], o['y'])),
                             obstacle_reach_m=M._xy_reach(o))
                rows[(split, pol, tid, r['repeat'])] = r
    rows = list(rows.values())
    json.dump(rows, open(out_path, 'w'), indent=1)
    for split in ('dev420', 'test12'):
        for pol in S.sources(main_root)[split]:
            R = [r for r in rows if r['split'] == split and r['policy'] == pol]
            C = [r for r in R if r['outcome'] == 'collision']
            if not C:
                continue
            lat_abs = sorted(abs(r['lateral_m']) for r in C)
            outside = sum(abs(r['obstacle_lateral_m']) - r['obstacle_reach_m'] > 1.0 for r in C)
            print(split, pol, dict(collections.Counter(r['outcome'] for r in R)),
                  '| collisions %d in %d tasks, obstacle outside corridor %d, |lateral| median %.2f m,'
                  ' beyond goal (axial>1) %d' % (len(C), len({r['task_id'] for r in C}), outside,
                                                 lat_abs[len(lat_abs) // 2], sum(r['axial'] > 1 for r in C)))


if __name__ == '__main__':
    main(*sys.argv[1:3])
