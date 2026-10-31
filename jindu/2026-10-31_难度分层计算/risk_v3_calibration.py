"""Offline calibration of the reward v3 closing-speed risk term on existing dev420 traces.

Per decision (executed H4 path p0..p4, 0.3 s per segment):
  c_k   = point clearance of p_k to the nearest obstacle (raw geometry, same as behavior_change.py)
  term  = sum_k max(0, c_{k-1} - c_k) * barrier(c_k; d_w, 0.10)
  barrier(c; w, b) = clip((w - c) / (w - b), 0, 1) ** 2      (same shape as v2 _barrier)
i.e. clearance lost per decision, weighted by proximity. Unit weight here; the coefficient is chosen
from these numbers. Reported per episode (sum over decisions) by outcome, next to the v2 risk sum
recorded in the traces. Collision decisions are cut at the event point.
Usage: python risk_v3_calibration.py <repo_root> <out.json>
"""
import glob
import json
import math
import sys

import numpy as np

import metrics as M

WIDTHS = (1.0, 1.2, 1.5)


def clr(p, geom):
    best = 1e9
    for o in geom:
        hd = float(M.horizontal_distance(o, p[0], p[1]))
        vz = max(abs(p[2] - o['z']) - M._half_height(o), 0.0)
        best = min(best, math.hypot(hd, vz))
    return best


def barrier(c, w, b=0.10):
    return min(max((w - c) / (w - b), 0.0), 1.0) ** 2


def episode(path):
    st, trs, reason = None, [], None
    for line in open(path):
        x = json.loads(line)
        if x.get('type') == 'episode_start':
            st = x
        elif x.get('type') == 'transition':
            trs.append(x)
            reason = x.get('done_reason')
        elif x.get('type') == 'episode_end':
            reason = x['done_reason']
    if reason in (None, 'none'):
        return None
    geom = st['geometry']
    out = {w: 0.0 for w in WIDTHS}
    per_dec = {w: [] for w in WIDTHS}
    first_nonzero = {w: None for w in WIDTHS}
    v2 = 0.0
    for i, x in enumerate(trs):
        pts = list(x['executed_path'])
        if i == len(trs) - 1 and reason in ('collision', 'ground_crash') and x.get('event_segment') is not None:
            s, f = x['event_segment'], x['event_fraction']
            pts = pts[:s + 1] + [[pts[s][k] + f * (pts[s + 1][k] - pts[s][k]) for k in range(3)]]
        cs = [clr(p, geom) for p in pts]
        for w in WIDTHS:
            t = sum(max(0.0, a - b) * barrier(b, w) for a, b in zip(cs, cs[1:]))
            out[w] += t
            per_dec[w].append(t)
            if t > 0 and first_nonzero[w] is None:
                first_nonzero[w] = len(trs) - i
        v2 += -x['reward_terms']['risk']
    return dict(reason=reason, v3=out, v2=v2, decisions=len(trs),
                max_dec={w: max(per_dec[w]) for w in WIDTHS},
                lead={w: first_nonzero[w] for w in WIDTHS})


def main(repo, out_path):
    base = repo + '/jindu/routea_a2_selective_keep_run01/full/'
    pols = {'P0': base + 'bc', 'P6': base + 'rl_d987235a0cbb_u256'}
    res = {}
    for p, d in pols.items():
        eps = [e for e in (episode(f) for f in sorted(glob.glob(d + '/traces/*.jsonl'))) if e]
        res[p] = {}
        for grp in ('reached', 'collision', 'time_limit'):
            E = [e for e in eps if (e['reason'] == grp or (grp == 'collision' and e['reason'] == 'ground_crash'))]
            row = dict(n=len(E), v2_mean=float(np.mean([e['v2'] for e in E])) if E else None)
            for w in WIDTHS:
                v = [e['v3'][w] for e in E]
                row['v3_w%.1f' % w] = dict(mean=float(np.mean(v)), median=float(np.median(v)),
                                           p90=float(np.quantile(v, .9)),
                                           max_per_decision_p95=float(np.quantile([e['max_dec'][w] for e in E], .95)))
                if grp == 'collision':
                    leads = [e['lead'][w] for e in E if e['lead'][w] is not None]
                    row['v3_w%.1f' % w]['decisions_before_end_first_nonzero_median'] = float(np.median(leads)) if leads else None
                    row['v3_w%.1f' % w]['frac_collisions_with_any_signal'] = len(leads) / len(E)
            res[p][grp] = row
    json.dump(res, open(out_path, 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:3])
