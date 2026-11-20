"""External policy receipt: strict rollout + per-field repeatability envelope.

Original formal FAIL reports are immutable. `passed` here means accepted under
the user-authorized 2026-11-08 policy, NOT bitwise equality. Never trains.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assess(baseline, d1, d2):
    failures, bounds = [], {}
    if d1['formal_passed']:
        failures.append('D1 is bitwise equal: alternative-policy case 3 does not apply')
    if not all(r['rollout_passed'] and r['rollout_structural_passed'] for r in (baseline, d1, d2)):
        failures.append('rollout mismatch')
    for group in ('tensor_deltas', 'update_scalar_deltas'):
        maps = [{r['path']: r for r in report[group]} for report in (baseline, d1, d2)]
        if not maps[0] or any(len(m) != len(r[group]) for m, r in zip(maps, (baseline, d1, d2))) or not (maps[0].keys() == maps[1].keys() == maps[2].keys()):
            failures.append(group + ': missing/duplicate paths')
            continue
        maximum = 0.
        for path in maps[0]:
            rows = [m[path] for m in maps]
            values = [r.get('max_abs') for r in rows]
            if any(v is None or not math.isfinite(v) or v < 0 for v in values):
                failures.append(path + ': invalid magnitude')
                continue
            if group == 'tensor_deltas' and any(r['status'] != 'finite' for r in rows):
                failures.append(path + ': tensor structure/nonfinite')
                continue
            reference = max(values[1:])
            if values[0] > 10. * reference:
                failures.append(path + ': baseline exceeds 10x same-source repetition')
            if reference:
                maximum = max(maximum, values[0] / reference)
        bounds[group] = dict(paths=len(maps[0]), max_baseline_to_repeat_ratio=maximum)
    return dict(passed=not failures, failures=failures, bounds=bounds,
                multiplier=10., granularity='each tensor and each update scalar path',
                zero_reference_rule='baseline must also be zero')


def load_comparison(folder):
    folder = Path(folder)
    formal_path, summary_path = folder/'formal_comparison.json', folder/'diagnostic_summary.json'
    formal, summary = read(formal_path), read(summary_path)
    if summary['formal_report_sha256'] != digest(formal_path) or summary['formal_passed'] != formal['passed']:
        raise ValueError('diagnostic summary is not bound to formal comparison')
    for item in formal['inputs'].values():
        if digest(item['path']) != item['sha256']:
            raise ValueError('comparison input changed: ' + item['path'])
    return formal, summary, dict(formal=dict(path=str(formal_path.resolve()), sha256=digest(formal_path)),
                                summary=dict(path=str(summary_path.resolve()), sha256=digest(summary_path)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'prepared', 'baseline', 'output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--d1')
    p.add_argument('--d2')
    args = p.parse_args()
    sys.path.insert(0, str(Path(args.repo).resolve()))
    from scripts.routea_f6_support import validated_prepared, TRANSITION_FIELDS
    from scripts.f6_u129_matched_rollout import IDENT, _identical
    frozen = validated_prepared(args.prepared)
    base, bs, evidence = load_comparison(args.baseline)
    if base['prepared'] != frozen:
        raise ValueError('baseline is not the formal run preparation')
    records = [(base, bs)]
    proof = dict(baseline=evidence)
    policy = dict(passed=True, policy='strict_bitwise')
    if not base['passed']:
        if not args.d1 or not args.d2:
            raise ValueError('failed baseline requires same-control D1 and D2')
        one, os, proof['D1'] = load_comparison(args.d1)
        two, ts, proof['D2'] = load_comparison(args.d2)
        if one['control'] != base['control'] or two['control'] != base['control']:
            raise ValueError('different control seed')
        for kind in ('metrics', 'state'):
            if one['inputs']['old_'+kind] != base['inputs']['old_'+kind] or two['inputs']['old_'+kind] != base['inputs']['new_'+kind]:
                raise ValueError('D1/D2 comparison references differ from baseline')
        records += [(one, os), (two, ts)]
        policy = assess(bs, os, ts)
        policy['policy'] = 'user_20261108_per_field_10x_repeatability'
    # Recheck the actual saved rollout fields, not only summary booleans.
    for formal, summary in records:
        scalar_paths = {r['path'] for r in summary['update_scalar_deltas']}
        for difference in formal['computational_differences']:
            path, kind = difference.rsplit(': ', 1)
            if (kind in ('keys', 'length', 'type', 'tensor type', 'tensor dtype/shape')
                    or (path.startswith('state') and kind != 'tensor bytes')
                    or (path.startswith('updates') and path not in scalar_paths)):
                raise ValueError('non-numeric or structural change is not covered: ' + difference)
        if any(r.get('status') != 'finite' for r in summary['tensor_deltas']) or any(
                r.get('max_abs') is None or not math.isfinite(r['max_abs'])
                for r in summary['update_scalar_deltas']):
            raise ValueError('nonfinite or structurally different training values')
        a, b = [read(formal['inputs'][side+'_metrics']['path'])['rollout_audit']['transition_records']
                for side in ('old', 'new')]
        aa, bb = [{r['transition_id']: r for r in rows} for rows in (a, b)]
        if len(a) != 1024 or len(b) != 1024 or len(aa) != 1024 or aa.keys() != bb.keys():
            raise ValueError('rollout structure differs')
        for tid in aa:
            for field in set(TRANSITION_FIELDS) | set(IDENT):
                if field not in aa[tid] or field not in bb[tid] or not _identical(aa[tid][field], bb[tid][field]):
                    raise ValueError('rollout mismatch: '+tid+'.'+field)
    result = copy.deepcopy(base)
    result.update(passed=policy['passed'], strict_bitwise_passed=base['passed'],
                  acceptance_policy=policy, acceptance_evidence=proof,
                  conclusion='训练更新按逐项重复性界限验收；rollout逐位一致；评价复现另行核验')
    # Preserve the schema consumed by existing seal/G-A4, but explicitly retain
    # the original differences and strict verdict; this is a NEW policy receipt.
    with Path(args.output).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    if not result['passed']:
        raise ValueError('repeatability envelope failed; see new receipt')
    print('TRAINING_POLICY_ACCEPTED strict_bitwise_passed=' + str(base['passed']))


if __name__ == '__main__':
    main()
