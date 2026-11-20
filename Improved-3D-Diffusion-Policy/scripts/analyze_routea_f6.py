#!/usr/bin/env python3
"""F6 completed-episode monitoring and frozen paired engineering verdicts."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_routea_a2 import read, write
from scripts.analyze_routea_f5 import check_bank, compare, load_evaluation, table
from scripts.routea_paired_support.core import canonical_hash
from scripts.run_routea_xk_evaluation import closest_path_point, read_accepted_trace

PAIRS = (('P12', 'P9'), ('P13', 'P11'))


def pair_verdict(values):
    gates = dict(dev=values['dev_arrival_delta'] >= 12,
        collision=values['dev_collision_delta'] <= 0,
        val=values['val_arrival_delta'] >= -4,
        retention=values['zero_band_loss_delta'] <= 8)
    return dict(passed=all(gates.values()), gates=gates, values=values)


def combined_verdict(passed):
    if len(passed) != 2: raise ValueError('exactly two seed pairs required')
    return ('engineering_criteria_met_both_seeds' if all(passed) else
            'inconclusive_between_seeds' if any(passed) else 'adoption_criteria_not_met_both_seeds')


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values): return dict(count=0, sum=0., min=None, median=None, max=None)
    if not np.isfinite(values).all(): raise ValueError('nonfinite monitor value')
    return dict(count=len(values), sum=float(values.sum()), min=float(values.min()),
        median=float(np.median(values)), max=float(values.max()))


def monitor_records(rows):
    episodes = defaultdict(list)
    seen = set()
    for r in rows:
        key = (r['update_index'], r['transition_id'])
        if key in seen: raise ValueError('duplicate transition')
        seen.add(key)
        episodes[r['episode_id']].append(r)
    counts, shaping, terminals = Counter(), [], []
    for group in episodes.values():
        group.sort(key=lambda r: (r['update_index'], int(r['transition_id'].split(':t')[-1])))
        first, last = group[0], group[-1]
        complete = bool(last['terminated'] or last['truncated'])
        counts['completed'] += complete
        counts['partial_episodes'] += not complete
        if not complete: continue
        if 'critic_episode_context' in first and first['critic_episode_context'][5] != 1.:
            raise ValueError('completed episode has missing start; read all preceding metrics')
        reason = last['terminal_facts']['done_reason']
        counts[reason] += 1
        path = np.concatenate([np.asarray(r['executed_path'], dtype=np.float64) for r in group])
        goal = np.asarray(last['goal_xyz'], dtype=np.float64)
        near = np.linalg.norm(closest_path_point(path, goal) - goal) < 1.
        collision = reason in ('collision', 'ground_crash') and np.linalg.norm(path[-1]-goal) <= 2.
        counts['near_miss_timeouts'] += int(reason == 'time_limit' and near)
        counts['near_goal_collisions'] += int(collision)
        shaping.append(sum(r['reward_terms']['goal_potential'] for r in group))
        terminals.append(dict(episode_id=last['episode_id'], end_update=last['update_index'],
            reason=reason, near_miss=bool(near and reason == 'time_limit'),
            near_goal_collision=bool(collision), shaping=shaping[-1], decisions=len(group)))
    n = counts['completed']
    return dict(completed=n, partial_episodes=counts['partial_episodes'], outcomes=dict(counts),
        reached_rate=counts['reached']/n if n else None,
        collision_rate=(counts['collision']+counts['ground_crash'])/n if n else None,
        timeout_rate=counts['time_limit']/n if n else None,
        near_miss_timeouts=counts['near_miss_timeouts'], near_goal_collisions=counts['near_goal_collisions'],
        episode_shaping=stats(shaping), transitions=len(rows),
        shaping_nonzero_rate=sum(r['reward_terms']['goal_potential'] != 0. for r in rows)/len(rows) if rows else None,
        completed_episodes=terminals)


def monitor(args):
    metrics = [read(p) for p in sorted(Path(args.metrics).glob('metrics_update_*.json'))]
    if not metrics: raise ValueError('no metrics files')
    rows = [r for m in metrics for r in m['rollout_audit']['transition_records']]
    summary = monitor_records(rows)
    blocks = []
    for index in range(0, len(metrics), 8):
        chunk = metrics[index:index+8]
        upto = [r for m in metrics[:index+len(chunk)] for r in m['rollout_audit']['transition_records']]
        joined = monitor_records(upto)
        updates = {r['update_index'] for m in chunk for r in m['rollout_audit']['transition_records']}
        completed = [e for e in joined['completed_episodes'] if e['end_update'] in updates]
        reasons = Counter(e['reason'] for e in completed)
        n = len(completed)
        ratios = [u['pre_update']['keep_diagnostics']['ratio'] for m in chunk for u in m['updates'] if u['status']=='committed']
        undefined_ratios = sum(ratio is None for ratio in ratios)
        ratios = [ratio for ratio in ratios if ratio is not None]
        ev = [m['rollout_audit']['critic']['explained_variance'] for m in chunk
              if m['rollout_audit']['critic']['explained_variance'] is not None]
        values = [r['reward_terms']['goal_potential'] for m in chunk for r in m['rollout_audit']['transition_records']]
        blocks.append(dict(updates=sorted(updates), completed=n, outcomes=dict(reasons),
            reached_rate=reasons['reached']/n if n else None,
            collision_rate=(reasons['collision']+reasons['ground_crash'])/n if n else None,
            timeout_rate=reasons['time_limit']/n if n else None,
            near_miss_timeouts=sum(e['near_miss'] for e in completed),
            near_goal_collisions=sum(e['near_goal_collision'] for e in completed),
            episode_shaping=stats([e['shaping'] for e in completed]),
            shaping_nonzero_rate=sum(x!=0. for x in values)/len(values),
            critic_explained_variance=stats(ev), keep_to_ppo_gradient_ratio=stats(ratios),
            undefined_keep_to_ppo_gradient_ratios=undefined_ratios))
    write(args.output, dict(schema='routea_f6_monitor_v1', summary=summary, windows=blocks,
        automatic_training_stop=False, policy_note='metrics u uses rollout collected by u-1'))


def mechanisms(job):
    counts, durations, height, closest_z = Counter(), [], defaultdict(Counter), []
    for p in sorted((Path(job)/'commits').glob('*.json')):
        start, rows, end = read_accepted_trace(job, p)
        goal = np.asarray(start['task']['goal_pos'], dtype=np.float64)
        path = np.concatenate([np.asarray(r['executed_path']) for r in rows])
        nearest = closest_path_point(path, goal)
        reason = end['done_reason']
        counts['near_miss_timeouts'] += int(reason == 'time_limit' and np.linalg.norm(nearest-goal) < 1.)
        counts['near_goal_collisions'] += int(reason in ('collision','ground_crash') and np.linalg.norm(path[-1]-goal) <= 2.)
        height['goal_le_3m' if goal[2] <= 3. else 'goal_gt_3m'][reason] += 1
        closest_z.append(float(nearest[2]-goal[2]))
        if reason == 'reached': durations.append(end['decisions'])
        for r in rows:
            plan = np.asarray(r['planned_policy_path'], dtype=np.float64)
            counts['h8_plans'] += 1
            counts['h8_intersects_goal'] += int(np.linalg.norm(closest_path_point(plan, goal)-goal) <= .5)
    return dict(counts=dict(counts), arrival_decisions=stats(durations), goal_height=dict(height),
        closest_point_minus_goal_z=stats(closest_z))


def evaluate(args):
    from scripts.routea_f6_support import require_reproduction_gates
    require_reproduction_gates(args.gates, read(Path(args.prepared)/'prepared.json'))
    data, strata, tables, diagnostic = {}, {}, {}, {}
    for split in ('dev', 'val'):
        bank = read(getattr(args, split+'_bank'))
        check_bank(bank, 'dev420' if split == 'dev' else 'val20')
        frozen = read(getattr(args, split+'_strata'))
        if (frozen['bank_hash'] != bank['content_hash'] or canonical_hash({k:v for k,v in frozen.items() if k!='content_hash'}) != frozen['content_hash']
                or {t['task_id'] for t in frozen['tasks']} != {t['task_id'] for t in bank['tasks']}):
            raise ValueError('strata identity/coverage differs')
        old, new = getattr(args, 'old_'+split), getattr(args, split)
        rows = load_evaluation(old, bank, names=('bc','P9','P11')) + load_evaluation(new, bank, names=('P12','P13'))
        data[split], strata[split] = rows, frozen['tasks']
        tables[split] = table(rows, frozen['tasks'], seed_pairs=PAIRS)
        diagnostic[split] = {n: mechanisms(Path(root)/'full'/n) for root,names in
            [(old,('P9','P11')), (new,('P12','P13'))] for n in names}
    pairs, transitions = {}, {}
    def count(rows, name, reasons): return sum(r['policy']==name and r['done_reason'] in reasons for r in rows)
    zero = {t['task_id'] for t in strata['dev'] if t['labels']['band']=='zero'}
    zero_rows = [r for r in data['dev'] if r['task_id'] in zero]
    for a, b in PAIRS:
        v = dict(dev_arrival_delta=count(data['dev'],a,['reached'])-count(data['dev'],b,['reached']),
            dev_collision_delta=count(data['dev'],a,['collision','ground_crash'])-count(data['dev'],b,['collision','ground_crash']),
            val_arrival_delta=count(data['val'],a,['reached'])-count(data['val'],b,['reached']),
            zero_band_loss_delta=compare(zero_rows,a,'P0')['lost']-compare(zero_rows,b,'P0')['lost'])
        pairs[a+'_vs_'+b] = pair_verdict(v)
        for split, rows in data.items():
            left={(r['task_id'],r['repeat']):r for r in rows if r['policy']==a}
            right={(r['task_id'],r['repeat']):r for r in rows if r['policy']==b}
            if set(left)!=set(right): raise ValueError('unpaired transitions')
            transitions[split+'_'+b+'_to_'+a]=dict(Counter(right[k]['done_reason']+' -> '+left[k]['done_reason'] for k in left))
    write(args.output, dict(schema='routea_f6_analysis_v1', protocol='#E2', pairs=pairs,
        verdict=combined_verdict([v['passed'] for v in pairs.values()]), tables=tables,
        verdict_zh={'engineering_criteria_met_both_seeds': '达到采用条件',
                    'inconclusive_between_seeds': '不确定',
                    'adoption_criteria_not_met_both_seeds': '本预算下未达采用条件'}[
                        combined_verdict([v['passed'] for v in pairs.values()])],
        transitions=transitions, mechanisms=diagnostic, automatic_continuation=False,
        scope='two paired seeds at u192; engineering criteria, not significance or mechanism causality'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('monitor')
    for key in ('metrics','output'): a.add_argument('--'+key, required=True)
    a = sub.add_parser('evaluate')
    for key in ('dev','val','old-dev','old-val','dev-bank','val-bank','dev-strata','val-strata','prepared','gates','output'):
        a.add_argument('--'+key, required=True)
    args=p.parse_args()
    globals()[args.command](args)


if __name__ == '__main__': main()
