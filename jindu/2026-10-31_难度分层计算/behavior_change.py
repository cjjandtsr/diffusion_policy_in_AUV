"""Paired behaviour change vs P0 over all episodes, linked to reward v2 terms.

Per episode (executed H4 path; 1 decision = 4 x 0.3 s = 1.2 s):
  reward terms (summed over decisions, from trace reward_terms): terminal, newbest, risk, time
  space         training-only term, recomputed offline with reward_v2.space_running_cost
                (budget 1.0, horizon 60 x 4 x 0.3 s); not present in evaluation traces
  decisions     number of decisions
  step_m        median displacement per decision
  closed_frac   1 - (best goal distance) / (initial distance)  (= newbest_sum / 10)
  t2m           decisions until first within 2 m of the goal (None if never)
  dz_close      z(closest point) - z(goal)
  vert_ratio    total |dz| travelled / |z_goal - z_start| (tasks with |required dz| > 0.5 m)
  minclr_m      minimum obstacle clearance along the path (raw geometry)
  risk_frac     fraction of decisions with a nonzero risk term
  straight      path length / (initial distance - final distance), episodes that closed >= 1 m
  xtrack_m      max horizontal distance from the S-G line
  yaw_misalign  median |yaw - heading of the decision displacement| (deg), decisions moving > 0.1 m
  zmin_m        minimum altitude
Groups: all conditions; both reached; both failed.
Usage: python behavior_change.py <main_checkout_root> <out.json>
"""
import collections
import glob
import json
import math
import sys

import numpy as np

import metrics as M
import stratify as S

KEYS = ('return', 'terminal', 'newbest', 'risk', 'time', 'space', 'decisions', 'step_m', 'closed_frac',
        't2m', 'dz_close', 'vert_ratio', 'minclr_m', 'risk_frac', 'straight', 'xtrack_m',
        'yaw_misalign_deg', 'zmin_m')


def clearance(p, geometry):
    best = 1e9
    for o in geometry:
        hd = float(M.horizontal_distance(o, p[0], p[1]))
        vz = max(abs(p[2] - o['z']) - M._half_height(o), 0.0)
        best = min(best, math.hypot(hd, vz))
    return best


def space_cost(path, s, g, budget=1.0, horizon=72.0, dt=0.3):
    path = np.asarray(path)
    d0 = np.linalg.norm(np.subtract(s, g))
    q = np.maximum(0.0, 1.0 - d0 / (np.linalg.norm(path - s, axis=1) + np.linalg.norm(path - g, axis=1)))
    return -float(budget / horizon * np.sum(dt * (q[:-1] + q[1:]) / 2.0))


def episode(start, trs, reason):
    s, g = start['task']['start_pos'], start['task']['goal_pos']
    geom = start['geometry']
    pts = []
    for x in trs:
        pts += x['executed_path'] if not pts else x['executed_path'][1:]
    if reason in ('collision', 'ground_crash'):
        last = trs[-1]
        p, seg, fr = last['executed_path'], last['event_segment'], last['event_fraction']
        cut = len(pts) - len(p) + seg + 1
        pts = pts[:cut] + [[p[seg][i] + fr * (p[seg + 1][i] - p[seg][i]) for i in range(3)]]
    elif reason == 'reached':
        last = trs[-1]
        p, seg, fr = last['executed_path'], last['event_segment'], last['event_fraction']
        if seg is not None:
            cut = len(pts) - len(p) + seg + 1
            pts = pts[:cut] + [[p[seg][i] + fr * (p[seg + 1][i] - p[seg][i]) for i in range(3)]]
    d0 = math.dist(s, g)
    dist = [math.dist(q, g) for q in pts]
    i = int(np.argmin(dist))
    terms = collections.Counter()
    for x in trs:
        for k, v in x['reward_terms'].items():
            terms[k] += v
    sx, sy = s[0], s[1]
    L = math.hypot(g[0] - sx, g[1] - sy)
    u = ((g[0] - sx) / L, (g[1] - sy) / L)
    steps = [math.dist(a['pre_state']['position'], a['post_state']['position']) for a in trs]
    t2m = next((k for k, x in enumerate(trs) if min(math.dist(q, g) for q in x['executed_path']) < 2.0), None)
    req = abs(g[2] - s[2])
    vert = sum(abs(b[2] - a[2]) for a, b in zip(pts, pts[1:]))
    closed = d0 - dist[-1]
    length = sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))
    mis = []
    for x in trs:
        # heading of the decision's executed horizontal displacement (instantaneous velocity at
        # pre_state lags the commanded motion and gave misleading angles)
        a, b = x['executed_path'][0], x['executed_path'][-1]
        dx, dy = b[0] - a[0], b[1] - a[1]
        if math.hypot(dx, dy) > 0.1:
            y0, y1 = x['pre_state']['yaw'], x['post_state']['yaw']
            ym = y0 + ((y1 - y0 + math.pi) % (2 * math.pi) - math.pi) / 2
            dlt = (ym - math.atan2(dy, dx) + math.pi) % (2 * math.pi) - math.pi
            mis.append(abs(math.degrees(dlt)))
    out = dict(
        reason=reason, terminal=terms['terminal'], newbest=terms['newbest'], risk=terms['risk'],
        time=terms['time'], space=space_cost(pts, s, g), decisions=len(trs),
        step_m=float(np.median(steps)), closed_frac=1 - min(dist) / d0, t2m=t2m,
        dz_close=pts[i][2] - g[2], vert_ratio=vert / req if req > 0.5 else None,
        minclr_m=min(clearance(q, geom) for q in pts[::2] + [pts[-1]]),
        risk_frac=sum(x['reward_terms']['risk'] < 0 for x in trs) / len(trs),
        straight=length / closed if closed >= 1.0 else None,
        xtrack_m=max(abs(-(q[0] - sx) * u[1] + (q[1] - sy) * u[0]) for q in pts),
        yaw_misalign_deg=float(np.median(mis)) if mis else None, zmin_m=min(q[2] for q in pts))
    out['return'] = out['terminal'] + out['newbest'] + out['risk'] + out['time'] + out['space']
    return out


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
        out[(st['task_id'], int(st['repeat']))] = episode(st, trs, reason)
    return out


def summarize(data, pols, keys):
    res = {}
    for p in pols:
        row = {}
        for k in KEYS:
            vals = [data[p][c][k] for c in keys if data[p][c][k] is not None]
            row[k] = dict(median=float(np.median(vals)) if vals else None,
                          mean=float(np.mean(vals)) if vals else None, n=len(vals))
            if p != 'P0':
                diffs = [data[p][c][k] - data['P0'][c][k] for c in keys
                         if data[p][c][k] is not None and data['P0'][c][k] is not None]
                row[k]['paired_median_diff'] = float(np.median(diffs)) if diffs else None
                row[k]['paired_mean_diff'] = float(np.mean(diffs)) if diffs else None
        res[p] = row
    return res


def main(root, out_path):
    result = {}
    for split, pols in S.sources(root).items():
        if split == 'dev420':
            pols = {p: pols[p] for p in ('P0', 'P2', 'P4', 'P5', 'P6', 'P7')}
        data = {p: load(d) for p, d in pols.items()}
        keys = sorted(data['P0'])
        groups = {'all': keys}
        for p in pols:
            if p == 'P0':
                continue
            groups['both_reached_%s' % p] = [k for k in keys if data[p][k]['reason'] == 'reached' == data['P0'][k]['reason']]
            groups['both_failed_%s' % p] = [k for k in keys if data[p][k]['reason'] != 'reached' and data['P0'][k]['reason'] != 'reached']
        result[split] = {g: dict(n=len(ks), stats=summarize(data, pols, ks)) for g, ks in groups.items()}
    json.dump(result, open(out_path, 'w'), ensure_ascii=False, indent=1)
    for split, res in result.items():
        for g in ['all'] + [x for x in res if x.endswith('P6')]:
            print('\n##', split, g, 'n', res[g]['n'])
            st = res[g]['stats']
            pols = list(st)
            print('%-17s' % 'metric' + ''.join('%14s' % p for p in pols))
            for k in KEYS:
                cells = []
                for p in pols:
                    c = st[p][k]
                    if c['median'] is None:
                        cells.append('%14s' % '-')
                    elif p == 'P0':
                        cells.append('%14s' % ('%.3g/%.3g' % (c['median'], c['mean'])))
                    else:
                        cells.append('%14s' % ('%.3g(%+.3g)' % (c['median'], c['paired_mean_diff'])))
                print('%-17s' % k + ''.join(cells))


if __name__ == '__main__':
    main(*sys.argv[1:3])
