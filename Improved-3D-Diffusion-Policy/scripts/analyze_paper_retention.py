"""Paired retention from paper-1.5m adjudicated episode JSONL; world intervals.

Input paper_safe_arrival must come from first-event trajectory adjudication,
never from the training environment's 0.5m safe_arrival flag.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def summarize(records, reference, treatment):
    groups = defaultdict(list)
    for row in records:
        if row['policy'] in (reference, treatment):
            groups[(row['task_id'], row['policy'])].append(row)
    tasks = sorted({task for task, _ in groups})
    paired = []
    for task in tasks:
        a, b = groups[(task, reference)], groups[(task, treatment)]
        if len(a) != 5 or len(b) != 5:
            raise ValueError(f'{task}: requires five repeats of both policies')
        if {r['repeat'] for r in a} != {r['repeat'] for r in b} or len({r['repeat'] for r in a}) != 5:
            raise ValueError(f'{task}: unmatched or duplicate repeats')
        worlds = {r['world_id'] for r in a + b}
        if len(worlds) != 1:
            raise ValueError(f'{task}: world mismatch')
        paired.append((next(iter(worlds)), sum(bool(r['paper_safe_arrival']) for r in a) >= 3,
                       sum(bool(r['paper_safe_arrival']) for r in b) >= 3))
    counts = dict(retained=sum(a and b for _, a, b in paired),
                  lost=sum(a and not b for _, a, b in paired),
                  new=sum(not a and b for _, a, b in paired),
                  persistent_failure=sum(not a and not b for _, a, b in paired))
    worlds = sorted({w for w, _, _ in paired})
    blocks = [np.array([[int(a), int(b)] for w, a, b in paired if w == world]) for world in worlds]
    rng = np.random.default_rng(2026102601)
    differences = []
    if len(worlds) >= 5:
        for _ in range(20000):
            sample = np.concatenate([blocks[i] for i in rng.integers(len(blocks), size=len(blocks))])
            differences.append(float(np.mean(sample[:, 1] - sample[:, 0])))
    denominator = counts['retained'] + counts['lost']
    return dict(reference=reference, treatment=treatment, tasks=len(tasks), worlds=len(worlds),
                counts=counts, retention_rate=counts['retained']/denominator if denominator else None,
                net_completed=counts['new']-counts['lost'], bootstrap_replicates=20000,
                bootstrap_seed=2026102601,
                paired_success_rate_difference_ci95=np.percentile(differences, [2.5, 97.5]).tolist() if differences else None,
                uncertainty_scope='world clusters in this evaluation bank, not training seeds')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', nargs='+', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--treatment', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--figure-prefix')
    args = parser.parse_args()
    records = [json.loads(line) for path in args.records for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    result = summarize(records, args.reference, args.treatment)
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    if args.figure_prefix:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from plot_paper_results_1m import style
        style()
        fig, ax = plt.subplots(figsize=(6.2, 3.2))
        labels = ['Retained', 'Lost', 'New', 'Persistent failure']
        bars = ax.bar(labels, list(result['counts'].values()),
                      color=['#56B4A1', '#D88A4A', '#0072B2', '#B9BEC7'])
        for bar, hatch in zip(bars, ['', '//', '..', 'xx']):
            bar.set_hatch(hatch)
        ax.set_ylabel('Tasks')
        ax.set_ylim(bottom=0)
        ax.set_title(args.reference + ' → ' + args.treatment)
        fig.tight_layout()
        for suffix in ('pdf', 'svg', 'png'):
            fig.savefig(args.figure_prefix + '.' + suffix, dpi=300, bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    main()
