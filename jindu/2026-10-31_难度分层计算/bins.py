"""Fix stratum boundaries from environment metrics only (spec §3-§5, §8).

Reads task_metrics.json (no policy outcomes) and writes bins.json plus
validation numbers. Must be run, and boundaries registered, before
stratify.py joins any policy outcome.
Usage: python bins.py task_metrics.json bins.json
"""
import json
import sys

import numpy as np
from scipy.stats import spearmanr


def tertiles(values):
    q1, q2 = np.quantile(values, [1 / 3, 2 / 3])
    return [float(q1), float(q2)]


def density_rule(values):
    values = np.asarray(values)
    zero = float(np.mean(values == 0.0))
    if zero >= 1 / 3:
        positive = values[values > 0]
        return dict(rule='zero_bin_plus_median', zero_fraction=zero,
                    edges=[0.0, float(np.median(positive))], labels=['无障碍', '低', '高'])
    return dict(rule='tertiles', zero_fraction=zero, edges=tertiles(values), labels=['低', '中', '高'])


def assign_density(rho, spec):
    if spec['rule'] == 'zero_bin_plus_median':
        if rho == 0.0:
            return '无障碍'
        return '低' if rho <= spec['edges'][1] else '高'
    a, b = spec['edges']
    return '低' if rho <= a else '中' if rho <= b else '高'


def main(metrics_path, out_path):
    rows = json.load(open(metrics_path))
    spec = {}
    for w in ('1.0', '2.0'):
        spec['rho_' + w] = density_rule([r['rho_' + w] for r in rows])
    reach_b = [r['b_band_m'] for r in rows if r['b_band_m'] is not None]
    spec['b_band_m'] = dict(rule='tertiles_reachable_in_band', edges=tertiles(reach_b), labels=['窄', '中', '宽'])
    detour = [r['kappa'] for r in rows if r['kappa'] is not None and r['kappa'] > 1.0]
    spec['kappa'] = dict(rule='one_bin_plus_median_of_kappa_gt_1', median=float(np.median(detour)),
                         labels=['无需绕行', '小绕行', '大绕行'])
    spec['unreachable_b_band'] = sum(r['b_band_m'] is None for r in rows)
    spec['unreachable_kappa'] = sum(r['kappa'] is None for r in rows)
    spec['n_tasks'] = len(rows)

    # validation (spec §8)
    bad = [r['task_id'] for r in rows if r['blocked'] and r['kappa'] is not None and r['kappa'] <= 1.0]
    pairs = [(r['rho_1.0'], r['b_band_m']) for r in rows if r['b_band_m'] is not None]
    rho_b = spearmanr([p[0] for p in pairs], [p[1] for p in pairs]).statistic
    pairs = [(r['rho_1.0'], r['kappa']) for r in rows if r['kappa'] is not None]
    rho_k = spearmanr([p[0] for p in pairs], [p[1] for p in pairs]).statistic
    pairs = [(r['b_band_m'], r['kappa']) for r in rows if r['kappa'] is not None and r['b_band_m'] is not None]
    b_k = spearmanr([p[0] for p in pairs], [p[1] for p in pairs]).statistic
    spec['validation'] = dict(
        blocked_with_kappa_le_1=bad,
        spearman_rho1_vs_b_band=float(rho_b), spearman_rho1_vs_kappa=float(rho_k),
        spearman_b_band_vs_kappa=float(b_k),
        flags={r['task_id']: r['flags'] for r in rows if r['flags']},
        degenerate_corridors=[r['task_id'] for r in rows if r['corridor_degenerate']])

    labels = {}
    for r in rows:
        k = r['kappa']
        labels[r['task_id']] = dict(
            density_1=assign_density(r['rho_1.0'], spec['rho_1.0']),
            density_2=assign_density(r['rho_2.0'], spec['rho_2.0']),
            bottleneck=('带内不可达' if r['b_band_m'] is None else
                        '窄' if r['b_band_m'] <= spec['b_band_m']['edges'][0] else
                        '中' if r['b_band_m'] <= spec['b_band_m']['edges'][1] else '宽'),
            detour=('不可达' if k is None else '无需绕行' if k <= 1.0 else
                    '小绕行' if k <= spec['kappa']['median'] else '大绕行'))
    spec['labels'] = labels
    json.dump(spec, open(out_path, 'w'), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in spec.items() if k != 'labels'}, ensure_ascii=False, indent=1))
    for dim in ('density_1', 'density_2', 'bottleneck', 'detour'):
        for split in ('dev420', 'test12'):
            counts = {}
            for r in rows:
                if r['split'] == split:
                    lab = labels[r['task_id']][dim]
                    counts[lab] = counts.get(lab, 0) + 1
            print(dim, split, counts)


if __name__ == '__main__':
    main(*sys.argv[1:3])
