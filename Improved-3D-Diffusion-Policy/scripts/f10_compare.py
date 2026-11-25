#!/usr/bin/env python3
"""F10 T1/T4/T5 helper: compare two committed job trees condition by condition.

Compares (a) whole trace SHA256, (b) raw done_reason and 1 m outcome, (c) the first
decision's full_action_16x11 bitwise, (d) the per-decision action seeds.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.routea_paired_support.core import file_hash, read_json, write_json
from scripts.f7_analysis import outcome_1m


def load(job):
    from scripts.routea_paired_support.episodes import load_commit
    job = Path(job)
    return job, {p.stem: p for p in (job / 'commits').glob('*.json')}, load_commit


def outputs_equal(a, b):
    """Bitwise comparison of every recorded decision output."""
    if len(a) != len(b):
        return False, float('inf')
    fields = ('full_action_16x11', 'policy_action_8x11', 'executed_action_4x11',
              'candidate_h4_path', 'executed_path', 'planned_policy_path')
    worst = 0.
    for x, y in zip(a, b):
        for name in fields:
            if name not in x or name not in y:
                continue
            left = np.asarray(x[name], np.float32)
            right = np.asarray(y[name], np.float32)
            if left.shape != right.shape or not np.array_equal(left, right):
                if left.shape == right.shape:
                    worst = max(worst, float(np.abs(left.astype(np.float64) -
                                                   right.astype(np.float64)).max()))
                else:
                    worst = float('inf')
                return False, worst
        for name in ('action_seed', 'decision'):
            if x[name] != y[name]:
                return False, worst
        if not np.array_equal(np.asarray(x['post_state']['position'], np.float32),
                              np.asarray(y['post_state']['position'], np.float32)):
            return False, float('inf')
    return True, worst


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--new', required=True)
    parser.add_argument('--ref', required=True)
    parser.add_argument('--write', required=True)
    parser.add_argument('--label', default='')
    args = parser.parse_args()
    new_job, new_commits, load_commit = load(args.new)
    ref_job, ref_commits, _ = load(args.ref)
    keys = sorted(set(new_commits) & set(ref_commits))
    rows = []
    for key in keys:
        nh, ntrace, nend = load_commit(new_job, new_commits[key])
        rh, rtrace, rend = load_commit(ref_job, ref_commits[key])
        goal = np.asarray(nh['task']['goal_pos'], np.float32)
        equal, worst = outputs_equal(ntrace, rtrace)
        rows.append(dict(
            key=key, task_id=nh['task_id'], repeat=nh['repeat'],
            trace_sha_equal=read_json(new_commits[key])['sha256'] == read_json(ref_commits[key])['sha256'],
            new_trace_sha256=read_json(new_commits[key])['sha256'],
            ref_trace_sha256=read_json(ref_commits[key])['sha256'],
            outcome_1m_new=outcome_1m(ntrace, goal, nend['done_reason']),
            outcome_1m_ref=outcome_1m(rtrace, goal, rend['done_reason']),
            raw_new=nend['done_reason'], raw_ref=rend['done_reason'],
            decisions_new=len(ntrace), decisions_ref=len(rtrace),
            action_seeds_equal=[t['action_seed'] for t in ntrace] == [t['action_seed'] for t in rtrace],
            decision_outputs_bitwise_equal=bool(equal),
            max_abs_output_difference=worst))
    summary = dict(label=args.label, new=str(Path(args.new).resolve()),
                   ref=str(Path(args.ref).resolve()), conditions=len(rows),
                   trace_sha_equal=sum(r['trace_sha_equal'] for r in rows),
                   outcome_1m_equal=sum(r['outcome_1m_new'] == r['outcome_1m_ref'] for r in rows),
                   raw_equal=sum(r['raw_new'] == r['raw_ref'] for r in rows),
                   action_seeds_equal=sum(r['action_seeds_equal'] for r in rows),
                   decision_outputs_bitwise_equal=sum(r['decision_outputs_bitwise_equal'] for r in rows),
                   new_reached_1m=sum(r['outcome_1m_new'] == 'reached' for r in rows),
                   ref_reached_1m=sum(r['outcome_1m_ref'] == 'reached' for r in rows))
    write_json(Path(args.write), dict(summary=summary, rows=rows))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
