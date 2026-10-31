"""Exploratory (defined after outcomes were seen): how each policy differs from P0.

Strata
  A (main)     world density: obstacle count per world. Natural gap in the data
               (1, 2, 4, 6, 8 vs 25-40) -> 稀疏世界 (<=8) / 密集世界 (>=25).
  B (appendix) task route band: horizontal |lateral| <= 3 m from the S-G line,
               axial [-1, L+3] m (3 m beyond the goal), z in [min(zS,zG)-1, max+1];
               obstacle volume fraction at 0.1 m; 带内无障碍 (=0) / 带内低 / 带内高 (median split).
  goal height  goal z < 3 m / >= 3 m (demonstration goal heights are within [1, 3] m).

Episode classes (from the executed H4 path; goal radius 0.5 m)
  到达             reached
  途中碰撞         collision (or ground crash) before ever coming within 2 m of the goal
  近目标后碰撞     collision after having come within 2 m of the goal
  目标下方超时     time limit, closest approach < 1 m and > 0.3 m below the goal at that point
  近目标其他超时   time limit, closest approach < 2 m, not 目标下方
  远处超时         time limit, never within 2 m of the goal
Usage: python outcome_change.py <main_checkout_root> <out.json>
"""
import collections
import glob
import json
import math
import random
import sys

import numpy as np

import stratify as S

CLASSES = ('到达', '途中碰撞', '近目标后碰撞', '目标下方超时', '近目标其他超时', '远处超时')


def band_density(task, half=3.0, beyond=3.0, res=0.1):
    import metrics as M
    s, g = np.array(task['start']), np.array(task['goal'])
    d = g[:2] - s[:2]
    L = float(np.hypot(*d))
    u = d / L
    v = np.array([-u[1], u[0]])
    ts = np.arange(-1.0 + res / 2, L + beyond, res)
    ss = np.arange(-half + res / 2, half, res)
    zlo = max(min(s[2], g[2]) - 1.0, 0.2)
    zs = np.arange(zlo + res / 2, max(s[2], g[2]) + 1.0, res)
    tt, sv = np.meshgrid(ts, ss, indexing='ij')
    px = (s[0] + tt * u[0] + sv * v[0]).ravel()
    py = (s[1] + tt * u[1] + sv * v[1]).ravel()
    occ = np.zeros((px.size, zs.size), dtype=bool)
    for o in task['geometry']:
        foot = M.horizontal_distance(o, px, py) <= 0.0
        if not foot.any():
            continue
        zm = np.abs(zs - o['z']) <= M._half_height(o)
        occ[np.ix_(np.nonzero(foot)[0], np.nonzero(zm)[0])] = True
    return float(occ.mean())


def classify(start, transitions, reason):
    g = start['task']['goal_pos']
    pts = []
    for x in transitions:
        pts += x['executed_path'] if not pts else x['executed_path'][1:]
    if reason in ('collision', 'ground_crash'):
        last = transitions[-1]
        p, seg, fr = last['executed_path'], last['event_segment'], last['event_fraction']
        k = len(pts) - len(p) + seg + 1
        before = pts[:k] + [[p[seg][i] + fr * (p[seg + 1][i] - p[seg][i]) for i in range(3)]]
    else:
        before = pts
    dists = [math.dist(q, g) for q in before]
    i = int(np.argmin(dists))
    near, dz = dists[i], before[i][2] - g[2]
    if reason == 'reached':
        cls = '到达'
    elif reason in ('collision', 'ground_crash'):
        cls = '近目标后碰撞' if near < 2.0 else '途中碰撞'
    elif reason == 'time_limit':
        cls = ('目标下方超时' if near < 1.0 and dz < -0.3 else '近目标其他超时' if near < 2.0 else '远处超时')
    else:
        raise ValueError(reason)
    length = sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))
    return dict(cls=cls, near=near, dz=dz, decisions=len(transitions), length=length,
                zmax=max(q[2] for q in pts), chord=math.dist(start['task']['start_pos'], g))


def load(directory):
    out = {}
    for f in sorted(glob.glob(directory + '/traces/*.jsonl')):
        st, tr, reason = None, [], None
        for line in open(f):
            x = json.loads(line)
            k = x.get('type')
            if k == 'episode_start':
                st = x
            elif k == 'transition':
                tr.append(x)
                reason = x.get('done_reason')
            elif k == 'episode_end':
                reason = x['done_reason']
        if reason == 'none':
            continue  # interrupted attempt; the re-run attempt file (sorted later) holds the episode
        out[(st['task_id'], int(st['repeat']))] = classify(st, tr, reason)
    return out


def world_ci(keys, world_of, ref, pol, seed=2026102601, draws=20000):
    worlds = sorted({world_of[k[0]] for k in keys})
    by = collections.defaultdict(list)
    for k in keys:
        by[world_of[k[0]]].append(k)
    f = lambda k: (pol[k]['cls'] == '到达') - (ref[k]['cls'] == '到达')
    point = 100.0 * sum(f(k) for k in keys) / len(keys)
    if len(worlds) < 5:
        return point, None
    rng = random.Random(seed)
    ds = []
    for _ in range(draws):
        ks = [k for _ in worlds for k in by[rng.choice(worlds)]]
        ds.append(100.0 * sum(f(k) for k in ks) / len(ks))
    ds.sort()
    return point, [ds[int(.025 * draws)], ds[int(.975 * draws) - 1]]


def main(root, out_path):
    geo = {t['task_id']: t for t in json.load(open('task_geometry.json'))}
    nobs = {t: len(g['geometry']) for t, g in geo.items()}
    world_of = {t: g['world'] for t, g in geo.items()}
    band = {t: band_density(g) for t, g in geo.items()}
    pos = [v for v in band.values() if v > 0]
    band_median = float(np.median(pos))
    strata = {}
    for t, g in geo.items():
        strata[t] = dict(world='稀疏世界' if nobs[t] <= 8 else '密集世界',
                         band='带内无障碍' if band[t] == 0 else '带内低' if band[t] <= band_median else '带内高',
                         goal='目标<3m' if g['goal'][2] < 3.0 else '目标≥3m')
    result = dict(band_median=band_median, band_density=band, strata=strata, splits={})
    for split, pols in S.sources(root).items():
        data = {p: load(d) for p, d in pols.items()}
        keys = sorted(data['P0'])
        res = dict(transitions={}, cells={})
        for p in pols:
            if p == 'P0':
                continue
            m = collections.Counter((data['P0'][k]['cls'], data[p][k]['cls']) for k in keys)
            res['transitions'][p] = {'%s→%s' % a: n for a, n in sorted(m.items())}
        for dim in ('world', 'band', 'goal'):
            for lab in sorted({strata[k[0]][dim] for k in keys}):
                ks = [k for k in keys if strata[k[0]][dim] == lab]
                cell = dict(tasks=len({k[0] for k in ks}), conditions=len(ks),
                            worlds=len({world_of[k[0]] for k in ks}), policies={})
                for p in pols:
                    c = collections.Counter(data[p][k]['cls'] for k in ks)
                    row = {cl: c.get(cl, 0) for cl in CLASSES}
                    if p != 'P0':
                        row['gain'] = sum(data[p][k]['cls'] == '到达' != (data['P0'][k]['cls'] == '到达') for k in ks)
                        row['gain'] = sum(data[p][k]['cls'] == '到达' and data['P0'][k]['cls'] != '到达' for k in ks)
                        row['loss'] = sum(data[p][k]['cls'] != '到达' and data['P0'][k]['cls'] == '到达' for k in ks)
                        row['delta_pp'], row['ci95'] = world_ci(ks, world_of, data['P0'], data[p])
                    cell['policies'][p] = row
                res['cells']['%s|%s' % (dim, lab)] = cell
        # how new successes succeed: P0 class on those conditions, and path features
        how = {}
        for p in pols:
            if p == 'P0':
                continue
            new = [k for k in keys if data[p][k]['cls'] == '到达' and data['P0'][k]['cls'] != '到达']
            kept = [k for k in keys if data[p][k]['cls'] == '到达' and data['P0'][k]['cls'] == '到达']
            med = lambda xs: float(np.median(xs)) if xs else None
            how[p] = dict(
                n_new=len(new),
                p0_class_of_new=dict(collections.Counter(data['P0'][k]['cls'] for k in new)),
                new_goal_ge3=sum(geo[k[0]]['goal'][2] >= 3.0 for k in new),
                new_dense_world=sum(nobs[k[0]] > 8 for k in new),
                new_band_obstacle=sum(band[k[0]] > 0 for k in new),
                new_length_over_chord=med([data[p][k]['length'] / data[p][k]['chord'] for k in new]),
                new_decisions=med([data[p][k]['decisions'] for k in new]),
                kept_decisions_policy=med([data[p][k]['decisions'] for k in kept]),
                kept_decisions_p0=med([data['P0'][k]['decisions'] for k in kept]),
                kept_length_ratio_policy=med([data[p][k]['length'] / data[p][k]['chord'] for k in kept]),
                kept_length_ratio_p0=med([data['P0'][k]['length'] / data['P0'][k]['chord'] for k in kept]),
                lost_policy_class=dict(collections.Counter(data[p][k]['cls'] for k in keys
                                                           if data['P0'][k]['cls'] == '到达' and data[p][k]['cls'] != '到达')),
                same_failure_class=sum(data['P0'][k]['cls'] != '到达' and data[p][k]['cls'] == data['P0'][k]['cls'] for k in keys),
                both_fail=sum(data['P0'][k]['cls'] != '到达' and data[p][k]['cls'] != '到达' for k in keys))
        res['how'] = how
        res['totals'] = {p: dict(collections.Counter(v['cls'] for v in data[p].values())) for p in pols}
        result['splits'][split] = res
    json.dump(result, open(out_path, 'w'), ensure_ascii=False, indent=1)
    print('band median', band_median)
    for split, res in result['splits'].items():
        print('\n######', split)
        print('totals by class:')
        for p, c in res['totals'].items():
            print('  ', p, ' '.join('%s %d' % (cl, c.get(cl, 0)) for cl in CLASSES))
        for key, cell in res['cells'].items():
            print(' #', key, 'tasks', cell['tasks'], 'cond', cell['conditions'], 'worlds', cell['worlds'])
            for p, r in cell['policies'].items():
                extra = '' if p == 'P0' else ' | +%d/-%d %+.1f %s' % (r['gain'], r['loss'], r['delta_pp'],
                                                                      '' if r['ci95'] is None else '[%.0f,%.0f]' % tuple(r['ci95']))
                print('    ', p, ' '.join('%d' % r[cl] for cl in CLASSES) + extra)
        for p, h in res['how'].items():
            print(' how', p, json.dumps(h, ensure_ascii=False))


if __name__ == '__main__':
    main(*sys.argv[1:3])
