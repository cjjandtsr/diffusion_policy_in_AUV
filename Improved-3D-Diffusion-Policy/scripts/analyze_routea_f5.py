#!/usr/bin/env python3
"""F5 paired engineering criteria and training exposure; no training intervention."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_routea_xk_evaluation import read, write, paired_cluster
from scripts.routea_paired_support.core import canonical_hash


def pair_verdict(values):
    gates = dict(high=values['high_arrival_delta'] >= 8,
                 dev=values['dev_arrival_delta'] >= -10,
                 retention=values['zero_band_loss_delta'] <= 8,
                 val=values['val_arrival_delta'] >= -4)
    collision_increase = values['dev_collision_delta'] > 0 or values['high_collision_delta'] > 0
    status = ('manual_review' if collision_increase else
              'meets_adoption_criteria' if all(gates.values()) else 'criteria_not_met')
    return dict(status=status, gates=gates, collision_increase=collision_increase, values=values)


def combined_verdict(statuses):
    if len(statuses) != 2:
        raise ValueError('exactly two seed pairs required')
    if 'manual_review' in statuses:
        return 'manual_review'
    if statuses == ['meets_adoption_criteria'] * 2:
        return 'engineering_criteria_met_both_seeds'
    if statuses == ['criteria_not_met'] * 2:
        return 'adoption_criteria_not_met_both_seeds'
    return 'inconclusive_between_seeds'


def compare(rows, treatment, reference):
    groups = {n: {(r['task_id'], r['repeat']): r for r in rows if r['policy'] == n}
              for n in (treatment, reference)}
    if not groups[treatment] or set(groups[treatment]) != set(groups[reference]):
        raise ValueError('paired conditions differ')
    for name, group in groups.items():
        if len(group) != sum(r['policy'] == name for r in rows):
            raise ValueError('duplicate paired condition')
    changes = []
    for key, a in groups[treatment].items():
        b = groups[reference][key]
        if a['world_id'] != b['world_id']:
            raise ValueError('paired world differs')
        changes.append(int(a['safe_arrival']) - int(b['safe_arrival']))
    worlds = {r['world_id'] for r in groups[treatment].values()}
    result = dict(treatment=treatment, reference=reference, pairs=len(changes), worlds=len(worlds),
                  rescued=sum(v > 0 for v in changes), lost=sum(v < 0 for v in changes),
                  delta_pp=100 * sum(changes) / len(changes), ci95_delta_pp=None)
    if len(worlds) >= 5:
        result = paired_cluster(rows, treatment, reference)
    return result


def labels(row):
    def density(value, boundary):
        return 'zero' if value == 0 else 'low' if value <= boundary else 'high'
    b, k = row['b_band_m'], row['kappa']
    return dict(w1=density(row['rho_1.0'], .04623), w2=density(row['rho_2.0'], .03919),
                band=density(row['band_rho'], .028221816244195782),
                bottleneck='unreachable' if b is None else 'narrow' if b <= 2.303 else 'middle' if b <= 3.144 else 'wide',
                detour='unavailable' if k is None else 'clear' if not row['blocked'] else 'small' if k <= 1.0186 else 'large')


def check_bank(bank, split):
    expected = (84, 420, 21) if split == 'dev420' else (80, 80, 20)
    if (len(bank['tasks']), len(bank['episodes']), len(bank['worlds'])) != expected:
        raise ValueError('wrong fixed bank denominator')
    if canonical_hash({k: v for k, v in bank.items() if k != 'content_hash'}) != bank['content_hash']:
        raise ValueError('bank content hash differs')


def freeze_strata(args):
    from scripts.routea_paired_support.task_geometry_metrics import compute, band_density
    bank = read(args.bank)
    check_bank(bank, args.split)
    rows = []
    for task in bank['tasks']:
        p = task['payload']
        item = dict(task_id=task['task_id'], world=task['world_id'], role=task['role'], split=args.split,
                    start=p['start_pos'], goal=p['goal_pos'], geometry=bank['worlds'][task['world_id']]['geometry'])
        row = compute(item)
        row['band_rho'] = band_density(item)
        row.pop('path', None)
        row['labels'] = labels(row)
        rows.append(row)
        print('STRATA', len(rows), len(bank['tasks']), flush=True)
    if args.split == 'dev420' and (sum(r['labels']['w1'] == 'high' for r in rows),
                                   sum(r['labels']['band'] == 'high' for r in rows)) != (20, 30):
        raise ValueError('frozen dev strata counts differ; do not change thresholds')
    result = dict(schema='routea_f5_strata_v1', split=args.split, bank_hash=bank['content_hash'], tasks=rows)
    result['content_hash'] = canonical_hash(result)
    write(args.output, result)


def load_evaluation(root, bank):
    from scripts.routea_paired_support.episodes import load_commit
    rows = []
    tasks = {t['task_id']: t for t in bank['tasks']}
    for name in ('bc', 'P6', 'P8', 'P9', 'P10', 'P11'):
        job = Path(root) / 'full' / name
        provenance = read(job / 'policy_provenance.json')
        protocol = provenance['evaluation_sampling_protocol']
        if (protocol['clip_protocol']['override'] != 2.0 or protocol['sampling_mode'] != 'deterministic'
                or provenance['bank_hash'] != bank['content_hash']):
            raise ValueError('evaluation is not paired #E2')
        for field in ('min_sampling_denoising_std', 'min_logprob_denoising_std'):
            values = protocol[field]
            values = values if isinstance(values, list) else [values]
            if any(abs(v - .1) > 1e-7 for v in values):
                raise ValueError('evaluation noise floor differs')
        if {p.stem for p in (job / 'commits').glob('*.json')} != {e['key'] for e in bank['episodes']}:
            raise ValueError('incomplete evaluation commits')
        verified = []
        for entry in bank['episodes']:
            h, _, end = load_commit(job, job / 'commits' / (entry['key'] + '.json'))
            task = tasks[entry['task_id']]
            if (h['bank_hash'] != bank['content_hash'] or h['initial_hash'] != entry['initial_hash']
                    or h['task_id'] != entry['task_id'] or h['repeat'] != entry['repeat']
                    or h['world_id'] != task['world_id'] or h['task'] != task['payload']
                    or h['reset_seed'] != entry['reset_seed']
                    or h['policy'] != name or h['use_base_policy'] != (name == 'bc')):
                raise ValueError('raw pairing/branch differs')
            verified.append(end)
        if verified != read(job / 'summary.json')['records']:
            raise ValueError('summary differs from raw evidence')
        rows.extend(dict(r, policy='P0' if name == 'bc' else name) for r in verified)
    return rows


def table(rows, tasks):
    result = {}
    for dim in ('w1', 'w2', 'bottleneck', 'detour', 'band'):
        result[dim] = {}
        for label in sorted({t['labels'][dim] for t in tasks}):
            ids = {t['task_id'] for t in tasks if t['labels'][dim] == label}
            selected = [r for r in rows if r['task_id'] in ids]
            names = sorted({r['policy'] for r in selected})
            result[dim][label] = dict(tasks=len(ids), conditions=sum(r['policy'] == 'P0' for r in selected),
                worlds=len({r['world_id'] for r in selected}),
                outcomes={n: dict(Counter(r['done_reason'] for r in selected if r['policy'] == n)) for n in names},
                vs_P0={n: compare(selected, n, 'P0') for n in names if n != 'P0'},
                seed_pairs={a + '_vs_' + b: compare(selected, a, b)
                            for a, b in [('P8', 'P9'), ('P10', 'P11')]})
    return result


def evaluate(args):
    data, strata, tables = {}, {}, {}
    for split, root, bank_path, strata_path in [('dev420', args.dev, args.dev_bank, args.dev_strata),
                                               ('val20', args.val, args.val_bank, args.val_strata)]:
        bank = read(bank_path)
        check_bank(bank, split)
        frozen = read(strata_path)
        if frozen['bank_hash'] != bank['content_hash'] or canonical_hash({k: v for k, v in frozen.items() if k != 'content_hash'}) != frozen['content_hash']:
            raise ValueError('frozen strata identity differs')
        if {r['task_id'] for r in frozen['tasks']} != {r['task_id'] for r in bank['tasks']}:
            raise ValueError('strata task coverage differs')
        data[split] = load_evaluation(root, bank)
        strata[split] = frozen['tasks']
        tables[split] = table(data[split], frozen['tasks'])
    dev, val = data['dev420'], data['val20']
    high = {r['task_id'] for r in strata['dev420'] if r['labels']['band'] == 'high'}
    zero = {r['task_id'] for r in strata['dev420'] if r['labels']['band'] == 'zero'}
    def count(rows, name, reasons, ids=None):
        return sum(r['policy'] == name and r['done_reason'] in reasons and (ids is None or r['task_id'] in ids) for r in rows)
    pairs = {}
    for treatment, control in [('P8', 'P9'), ('P10', 'P11')]:
        zero_rows = [r for r in dev if r['task_id'] in zero]
        values = dict(high_arrival_delta=count(dev, treatment, ['reached'], high) - count(dev, control, ['reached'], high),
            dev_arrival_delta=count(dev, treatment, ['reached']) - count(dev, control, ['reached']),
            zero_band_loss_delta=compare(zero_rows, treatment, 'P0')['lost'] - compare(zero_rows, control, 'P0')['lost'],
            val_arrival_delta=count(val, treatment, ['reached']) - count(val, control, ['reached']),
            dev_collision_delta=count(dev, treatment, ['collision', 'ground_crash']) - count(dev, control, ['collision', 'ground_crash']),
            high_collision_delta=count(dev, treatment, ['collision', 'ground_crash'], high) - count(dev, control, ['collision', 'ground_crash'], high))
        pairs[treatment + '_vs_' + control] = pair_verdict(values)
    write(args.output, dict(protocol='#E2', tables=tables, pairs=pairs,
        verdict=combined_verdict([p['status'] for p in pairs.values()]),
        F3_seed_difference=compare(dev, 'P9', 'P11'),
        historical_F2_range='unavailable under common #E2; do not substitute historical #E1',
        scope='two paired seeds; engineering adoption criteria, not factor causality or significance'))


def summarize_training_records(rows):
    episodes = defaultdict(list)
    for row in rows:
        episodes[row['episode_id']].append(row)
    result = {'by_source': {}, 'by_tier': {}, 'by_world': {}}
    for values in episodes.values():
        values.sort(key=lambda r: r['decision_index'])
        first, last = values[0], values[-1]
        metadata = first.get('task_bank') or dict(source='original', tier=None)
        completed = bool(last['terminated'] or last['truncated'])
        reason = (last.get('terminal_facts') or {}).get('done_reason', 'none')
        for dimension, label in [('by_source', metadata['source']), ('by_tier', metadata.get('tier') or 'original_unclassified'), ('by_world', first['world_id'])]:
            c = result[dimension].setdefault(label, Counter())
            c.update(started=int(first['decision_index'] == 0), completed=int(completed),
                     collection_cut=int(not completed and last.get('collection_boundary', False)),
                     truncated=int(last['truncated']), transitions=len(values),
                     reached=int(reason == 'reached'), collision=int(reason in ('collision', 'ground_crash')),
                     successful_episode_transitions=len(values) if reason == 'reached' else 0)
    for dimension in result.values():
        for c in dimension.values():
            c['success_rate_completed'] = c['reached'] / c['completed'] if c['completed'] else None
            c['success_rate_started'] = c['reached'] / c['started'] if c['started'] else None
    return result


def monitor(args):
    updates = []
    for path in sorted(Path(args.metrics).glob('metrics_update_*.json')):
        rows = read(path)['rollout_audit']['transition_records']
        if not rows:
            raise ValueError('missing training transition audit')
        updates.append(dict(update=rows[0]['update_index'], statistics=summarize_training_records(rows)))
    if not updates:
        raise ValueError('no metrics files')
    updates.sort(key=lambda r: r['update'])
    write(args.output, dict(updates=updates, automatic_training_stop=False,
        interpretation='original strata are unclassified; differing on-policy distributions are not a forgetting stop criterion'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('freeze-strata')
    p.add_argument('--split', choices=['dev420', 'val20'], required=True)
    for key in ('bank', 'output'):
        p.add_argument('--' + key, required=True)
    p = sub.add_parser('evaluate')
    for key in ('dev', 'dev-bank', 'dev-strata', 'val', 'val-bank', 'val-strata', 'output'):
        p.add_argument('--' + key, required=True)
    p = sub.add_parser('monitor')
    for key in ('metrics', 'output'):
        p.add_argument('--' + key, required=True)
    args = parser.parse_args()
    globals()[args.command.replace('-', '_')](args)


if __name__ == '__main__':
    main()
