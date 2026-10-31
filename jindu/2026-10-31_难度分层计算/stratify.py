"""Join registered strata (bins.json) with policy outcomes; spec §7 report tables.

Policy numbering: jindu/2026-10-31_策略编号规范_v1_0.md. Evaluation protocol #E1.
Usage: python stratify.py bins.json task_metrics.json <main_checkout_root> <out.json>
"""
import collections
import glob
import json
import random
import sys

REPO = '/projects/sandbox/diffusion_policy_in_AUV/jindu/'


def sources(main_root):
    a2 = REPO + 'routea_a2_selective_keep_run01/full/'
    a1 = REPO + 'routea_common_a1_v1_run01/evaluation/full/'
    xk = main_root + '/jindu/routea_xk_run01/e1_dev_original/full/'
    t12 = main_root + '/jindu/routea_test12_independent_run02/full/'
    return {
        'dev420': {'P0': a2 + 'bc', 'P2': a2 + 'rl_391a6656a0f7_u128', 'P3': a1 + 'rl_391a6656a0f7_u256',
                   'P4': a2 + 'rl_910559bb5de2_u256', 'P5': xk + 'K', 'P6': a2 + 'rl_d987235a0cbb_u256',
                   'P7': xk + 'X'},
        'test12': {'P0': t12 + 'bc', 'P2': t12 + 'u128', 'P4': t12 + 'C', 'P6': t12 + 'T'},
    }


def outcomes(directory):
    out = {}
    for path in sorted(glob.glob(directory + '/traces/*.jsonl')):  # attempt002 sorts after attempt001
        start, reason = None, None
        for line in open(path):
            record = json.loads(line)
            kind = record.get('type')
            if kind == 'episode_start':
                start = record
            elif kind == 'transition':
                reason = record.get('done_reason')
            elif kind == 'episode_end':
                reason = record['done_reason']
        out[(start['task_id'], int(start['repeat']))] = reason
    return out


def bootstrap(keys, world_of, ref, pol, draws=20000, seed=2026102601):
    worlds = sorted({world_of[k[0]] for k in keys})
    by_world = collections.defaultdict(list)
    for k in keys:
        by_world[world_of[k[0]]].append(k)
    point = 100.0 * sum((pol[k] == 'reached') - (ref[k] == 'reached') for k in keys) / len(keys)
    if len(worlds) < 5:
        return point, None
    rng = random.Random(seed)
    deltas = []
    for _ in range(draws):
        sample = [k for _ in worlds for k in by_world[rng.choice(worlds)]]
        deltas.append(100.0 * sum((pol[k] == 'reached') - (ref[k] == 'reached') for k in sample) / len(sample))
    deltas.sort()
    return point, [deltas[int(0.025 * draws)], deltas[int(0.975 * draws) - 1]]


def main(bins_path, metrics_path, main_root, out_path):
    labels = json.load(open(bins_path))['labels']
    metrics = {m['task_id']: m for m in json.load(open(metrics_path))}
    world_of = {t: m['world'] for t, m in metrics.items()}
    result = {}
    for split, pols in sources(main_root).items():
        data = {p: outcomes(d) for p, d in pols.items()}
        keys = sorted(data['P0'])
        for p, v in data.items():
            if sorted(v) != keys:
                raise ValueError('unpaired keys for %s %s' % (split, p))
        totals = {p: collections.Counter(v.values()) for p, v in data.items()}
        result[split] = dict(totals={p: dict(c) for p, c in totals.items()}, dims={})
        for dim in ('density_1', 'density_2', 'bottleneck', 'detour'):
            table = {}
            for lab in sorted({labels[k[0]][dim] for k in keys}):
                ks = [k for k in keys if labels[k[0]][dim] == lab]
                row = dict(tasks=len({k[0] for k in ks}), conditions=len(ks),
                           worlds=len({world_of[k[0]] for k in ks}), policies={})
                for p in pols:
                    pol = data[p]
                    cell = dict(reached=sum(pol[k] == 'reached' for k in ks),
                                collision=sum(pol[k] == 'collision' for k in ks),
                                time_limit=sum(pol[k] == 'time_limit' for k in ks))
                    if p != 'P0':
                        cell['gain_vs_P0'] = sum(pol[k] == 'reached' and data['P0'][k] != 'reached' for k in ks)
                        cell['loss_vs_P0'] = sum(pol[k] != 'reached' and data['P0'][k] == 'reached' for k in ks)
                        point, ci = bootstrap(ks, world_of, data['P0'], pol)
                        cell['delta_pp_vs_P0'] = point
                        cell['ci95_pp'] = ci
                    row['policies'][p] = cell
                table[lab] = row
            result[split]['dims'][dim] = table
    json.dump(result, open(out_path, 'w'), ensure_ascii=False, indent=1)
    for split in result:
        print('##', split, {p: c.get('reached') for p, c in result[split]['totals'].items()})
        for dim, table in result[split]['dims'].items():
            print(' #', dim)
            for lab, row in table.items():
                cells = []
                for p, c in row['policies'].items():
                    if p == 'P0':
                        cells.append('P0 %d' % c['reached'])
                    else:
                        ci = c['ci95_pp']
                        cells.append('%s %d(+%d/-%d)%s' % (p, c['reached'], c['gain_vs_P0'], c['loss_vs_P0'],
                                     '' if ci is None else '[%.0f,%.0f]' % tuple(ci)))
                print('   %s t%d c%d w%d |' % (lab, row['tasks'], row['conditions'], row['worlds']), ' '.join(cells))


if __name__ == '__main__':
    main(*sys.argv[1:5])
