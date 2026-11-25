#!/usr/bin/env python3
"""F10 step-3 report: per-arm integrity, outcome tables (1 m and raw), guidance and timing stats."""
import argparse
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.routea_paired_support.core import read_json, write_json


def arm_rows(job):
    import csv
    path = Path(job) / 'conditions.csv'
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def summarize(job, identity):
    rows = arm_rows(job)
    triples = [(r['task_id'], r['repeat']) for r in rows]
    dup = [k for k, n in Counter(triples).items() if n > 1]
    out1 = Counter(r['outcome_1m'] for r in rows)
    raw = Counter(r['raw_outcome'] for r in rows)
    decisions = sum(int(r['decisions']) for r in rows)
    guided = sum(int(r['guided_decisions']) for r in rows)
    denoise = np.asarray([float(r['denoise_median_ms']) for r in rows], float)
    return dict(
        job=str(job), policy=identity['policy'], guidance=identity['guidance'],
        setting=identity['setting'], split=identity['split'],
        conditions=len(rows), unique_conditions=len(set(triples)), duplicate_conditions=dup,
        outcome_1m={k: out1.get(k, 0) for k in ('reached', 'collision', 'time_limit')},
        raw_done_reason={k: raw.get(k, 0) for k in ('reached', 'collision', 'time_limit')},
        decisions=decisions, guided_decisions=guided,
        guided_fraction=(guided / decisions if decisions else None),
        conditions_with_any_guidance=sum(int(r['guided_decisions']) > 0 for r in rows),
        delta_exec_max=max(float(r['delta_exec_max']) for r in rows),
        delta_exec_sum=sum(float(r['delta_exec_sum']) for r in rows),
        denoise_median_ms_across_conditions=dict(
            median=float(np.median(denoise)), p05=float(np.quantile(denoise, .05)),
            p95=float(np.quantile(denoise, .95))),
        bank_sha256=identity['bank_sha256'],
        guidance_identity=identity.get('guidance'),
        runtime_identity=identity['native_runtime_identity'],
        state_sha256=identity['state_sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--write', required=True)
    args = parser.parse_args()
    run = Path(args.output)
    arms = {}
    for job in sorted((run / 'runs').iterdir()):
        if not (job / 'identity.json').is_file():
            continue
        arms[job.name] = summarize(job, read_json(job / 'identity.json'))
    rc = {p.stem.split('_', 1)[1]: p.read_text().strip()
          for p in sorted((run / 'execution').glob('S[14]_*.rc'))}
    for name, item in arms.items():
        item['rc'] = rc.get(name)
    integrity = dict(arms={k: dict(conditions=v['conditions'],
                                   unique=v['unique_conditions'],
                                   duplicates=len(v['duplicate_conditions']),
                                   rc=v['rc']) for k, v in arms.items()},
                     all_rc_zero=all(v['rc'] == '0' for v in arms.values()) if arms else False)
    write_json(Path(args.write), dict(arms=arms, integrity=integrity))
    print(json.dumps(integrity, ensure_ascii=False, indent=1))
    for name, v in arms.items():
        print(name, v['outcome_1m'], 'raw', v['raw_done_reason'],
              'guided %.3f' % (v['guided_fraction'] or 0))


if __name__ == '__main__':
    main()
