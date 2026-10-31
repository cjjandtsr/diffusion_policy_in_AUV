"""Training-rollout outcomes by difficulty, per run and update window (exploratory).

Source: rollout_audit.transition_records in metrics_update_<u>.json. The rollout in file u
was collected by the policy after update u-1 (policy_state_update_index).
An episode = all records with one episode_id inside one file; start = position_before_xyz of
its first decision (includes mid-state initialisation); outcome = terminal_facts.done_reason of
its terminated/truncated record. Episodes cut by the collection boundary are dropped.
Strata (same definitions as outcome_change.py, computed from each episode's own start/goal):
  world  稀疏世界 (<=8 obstacles) / 密集世界 (>=25)
  band   route band +-3 m, 3 m beyond the goal, z in [min-1, max+1]; 带内无障碍 / 低 / 高,
         using the evaluation band median 0.0282 as the low/high cut
  goal   目标<3m / 目标≥3m
Usage: python training_strata.py <main_checkout_root> <out.json> [workers]
"""
import collections
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

BAND_MEDIAN = 0.028221816244195782   # from outcome_change.json (evaluation tasks)
GEOM = None


def runs(root):
    a1 = '/projects/sandbox/diffusion_policy_in_AUV/jindu/routea_common_a1_v1_run01/training_b2560'
    a2 = root + '/jindu/routea_a2_selective_keep_run01/arms'
    xk = root + '/jindu/routea_xk_run01/main'
    return {'P2/P3': a1, 'P4': a2 + '/C', 'P6': a2 + '/T', 'P5': xk + '/K', 'P7': xk + '/X'}


def episodes_of(path):
    recs = json.load(open(path))['rollout_audit']['transition_records']
    by = collections.defaultdict(list)
    for r in recs:
        by[r['episode_id']].append(r)
    out = []
    for eid, rs in by.items():
        rs.sort(key=lambda r: r['decision_index'])
        end = [r for r in rs if r['terminated'] or r['truncated']]
        if not end:
            continue
        reason = end[-1]['terminal_facts']['done_reason']
        out.append(dict(episode_id=eid, world=rs[0]['world_id'], update=rs[0]['update_index'],
                        policy_update=rs[0]['policy_state_update_index'], role=rs[0]['role'],
                        start=rs[0]['position_before_xyz'], goal=rs[0]['goal_xyz'], reason=reason,
                        decisions=len(rs),
                        min_goal_dist=min(min(r['goal_distance_before_3d'], r['goal_distance_after_3d']) for r in rs),
                        dz_at_min=None))
    return out


def label(ep):
    from outcome_change import band_density
    g = GEOM[ep['world']]
    task = dict(start=ep['start'], goal=ep['goal'], geometry=g)
    rho = band_density(task)
    ep['band_rho'] = rho
    ep['band'] = '带内无障碍' if rho == 0 else '带内低' if rho <= BAND_MEDIAN else '带内高'
    ep['world_class'] = '稀疏世界' if len(g) <= 8 else '密集世界'
    ep['goal_class'] = '目标<3m' if ep['goal'][2] < 3.0 else '目标≥3m'
    return ep


def init(geom):
    global GEOM
    GEOM = geom


def main(root, out_path, workers='6'):
    geo = json.load(open('task_geometry.json'))
    geom = {}
    for t in geo:
        geom.setdefault(t['world'], t['geometry'])
    result = {}
    allrows = []
    with Pool(int(workers), initializer=init, initargs=(geom,)) as pool:
        for run, d in runs(root).items():
            files = sorted(glob.glob(d + '/metrics_update_*.json'))
            eps = [e for f in files for e in episodes_of(f)]
            eps = pool.map(label, eps, chunksize=20)
            for e in eps:
                e['run'] = run
            allrows += eps
            print(run, 'files', len(files), 'episodes', len(eps), flush=True)
    json.dump(allrows, open(out_path, 'w'))
    summarize(allrows)


def summarize(rows):
    def window(u):
        return '%03d-%03d' % ((u - 1) // 32 * 32 + 1, (u - 1) // 32 * 32 + 32)
    for dim in ('band', 'world_class', 'goal_class'):
        print('\n#####', dim)
        for run in ('P2/P3', 'P4', 'P5', 'P6', 'P7'):
            R = [r for r in rows if r['run'] == run]
            if not R:
                continue
            print('==', run)
            for lab in sorted({r[dim] for r in R}):
                cells = []
                for w in sorted({window(r['update']) for r in R}):
                    S = [r for r in R if r[dim] == lab and window(r['update']) == w]
                    if S:
                        rr = sum(r['reason'] == 'reached' for r in S) / len(S)
                        cr = sum(r['reason'] in ('collision', 'ground_crash') for r in S) / len(S)
                        cells.append('%s n%4d R%.2f C%.2f' % (w, len(S), rr, cr))
                print('  ', lab, ' | '.join(cells))


if __name__ == '__main__':
    main(*sys.argv[1:])
