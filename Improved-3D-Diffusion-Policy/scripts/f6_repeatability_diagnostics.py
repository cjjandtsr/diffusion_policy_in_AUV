"""External D1/D2 organizer and numeric reporter; never trains or seals gates.

Use --repo to import the unchanged frozen F6 comparator. Run this helper outside
BASE/ROOT; the two training commands use their original entrypoints.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage(args):
    from omegaconf import OmegaConf
    from scripts.routea_f6_support import validated_prepared
    frozen = validated_prepared(args.f6_prepared)
    old = read(Path(args.f5_prepared) / 'prepared.json')
    output, weights = Path(args.output).resolve(), Path(args.weights).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if weights.exists():
        raise FileExistsError(weights)
    for name, original in (('D1', old), ('D2', frozen)):
        folder = output / name
        folder.mkdir()
        item = original['configs'][args.control]
        if digest(item['path']) != item['sha256']:
            raise ValueError('original registered config changed')
        config = OmegaConf.to_container(OmegaConf.load(item['path']), resolve=True)
        config['output_dir'] = str(weights / name / args.control)
        config['result_dir'] = str(folder / 'metrics')
        path = folder / (args.control + '.yaml')
        OmegaConf.save(OmegaConf.create(config), path)
        entry = dict(path=str(path), sha256=digest(path))
        for label, template in (('run_prepared', original), ('compare_prepared', frozen)):
            target = folder / label
            target.mkdir()
            prepared = copy.deepcopy(template)
            prepared['configs'][args.control] = entry
            write(target / 'prepared.json', prepared)
        write(folder / 'resolved_config.json', config)
    write(output / 'DIAGNOSTIC_ONLY.json', dict(
        purpose='D1/D2 repeatability only; weights forbidden for formal continuation',
        control=args.control, stop_update=129, source_comparator_root=str(Path(args.repo).resolve()),
        f5_prepared_sha256=digest(Path(args.f5_prepared) / 'prepared.json'),
        f6_prepared_sha256=digest(Path(args.f6_prepared) / 'prepared.json')))


def numeric_deltas(a, b, path='$'):
    """Per-tensor maxima and per-scalar differences; old side is the reference."""
    import torch
    tensors, scalars = [], []
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        row = dict(path=path, old_dtype=str(a.dtype), new_dtype=str(b.dtype),
                   old_shape=list(a.shape), new_shape=list(b.shape))
        if a.shape != b.shape or a.dtype != b.dtype:
            row['status'] = 'dtype_or_shape_mismatch'
        else:
            x, y = a.detach().cpu().double(), b.detach().cpu().double()
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                row['status'] = 'nonfinite'
            else:
                delta = (y - x).abs()
                nonzero = x != 0
                zero_changes = int(((x == 0) & (delta != 0)).sum())
                row.update(status='finite', max_abs=float(delta.max()) if delta.numel() else 0.,
                           max_relative=('infinity' if zero_changes else
                               float((delta[nonzero] / x[nonzero].abs()).max()) if nonzero.any() else 0.),
                           zero_reference_changed_elements=zero_changes)
        tensors.append(row)
    elif isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(a.keys() & b.keys(), key=str):
            t, s = numeric_deltas(a[key], b[key], path + '.' + str(key))
            tensors.extend(t)
            scalars.extend(s)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        for index, (x, y) in enumerate(zip(a, b)):
            t, s = numeric_deltas(x, y, path + '[' + str(index) + ']')
            tensors.extend(t)
            scalars.extend(s)
    elif type(a) in (int, float) and type(b) in (int, float):
        import math
        if not math.isfinite(a) or not math.isfinite(b):
            scalars.append(dict(path=path, status='nonfinite'))
        else:
            delta = abs(b - a)
            scalars.append(dict(path=path, old=a, new=b, max_abs=delta,
                max_relative=delta / abs(a) if a else ('infinity' if delta else 0.)))
    return tensors, scalars


def compare(args):
    from scripts.routea_f6_support import compare_training, exact_differences, verified_state, TRANSITION_FIELDS
    from scripts.f6_u129_matched_rollout import IDENT, _identical
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    formal = output / 'formal_comparison.json'
    call = SimpleNamespace(**vars(args))
    call.output = str(formal)
    try:
        compare_training(call)
    except ValueError:
        # Expected numerical FAIL must not prevent collecting D1/D2 diagnostics.
        # Identity/input failures produce no completed comparison and propagate.
        if not formal.exists() or read(formal).get('passed') is not False:
            raise
    result = read(formal)
    old, new = read(args.old_metrics), read(args.new_metrics)
    a = old['rollout_audit']['transition_records']
    b = new['rollout_audit']['transition_records']
    ids_a, ids_b = [r['transition_id'] for r in a], [r['transition_id'] for r in b]
    structural = len(a) == len(b) == 1024 and len(set(ids_a)) == 1024 and set(ids_a) == set(ids_b)
    fields = tuple(dict.fromkeys(TRANSITION_FIELDS + IDENT))
    rollout_diffs = []
    for x, y in zip(sorted(a, key=lambda r: r['transition_id']), sorted(b, key=lambda r: r['transition_id'])):
        for field in fields:
            if field not in x or field not in y or not _identical(x[field], y[field]):
                rollout_diffs.append(dict(transition=x['transition_id'], field=field))
    update_diffs = [exact_differences(x, y) for x, y in zip(old['updates'], new['updates'])]
    first = next((i for i, differences in enumerate(update_diffs) if differences), None)
    _, left = verified_state(args.old_checkpoint)
    _, right = verified_state(args.new_checkpoint)
    tensors, _ = numeric_deltas(left, right, 'state')
    _, updates = numeric_deltas(old['updates'], new['updates'], 'updates')
    write(output / 'diagnostic_summary.json', dict(
        diagnostic_only=True, formal_passed=result['passed'],
        formal_report_sha256=digest(formal), formal_difference_count=len(result['computational_differences']),
        rollout_fields=list(fields), rollout_structural_passed=structural,
        rollout_passed=structural and not rollout_diffs, rollout_differences=rollout_diffs,
        first_differing_update_index=first, first_differing_submission=None if first is None else first+1,
        per_update_difference_counts=[len(x) for x in update_diffs], tensor_deltas=tensors,
        update_scalar_deltas=updates,
        relative_definition='abs(new-old)/abs(old); zero/zero=0; nonzero/zero=infinity',
        note='No alternative gate enabled; original FAIL retained. Exit 0 means report collection completed.'))
    print('DIAGNOSTIC_REPORT_COMPLETE formal_passed=' + str(result['passed']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', required=True, help='unchanged frozen F6 ROOT, for comparator imports')
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('stage')
    for name in ('f5-prepared', 'f6-prepared', 'output', 'weights'):
        a.add_argument('--'+name, required=True)
    a.add_argument('--control', choices=['P9', 'P11'], default='P9')
    a = sub.add_parser('compare')
    for name in ('prepared', 'old-metrics', 'new-metrics', 'old-checkpoint', 'new-checkpoint', 'output'):
        a.add_argument('--'+name, required=True)
    a.add_argument('--control', choices=['P9', 'P11'], default='P9')
    args = p.parse_args()
    sys.path.insert(0, str(Path(args.repo).resolve()))
    globals()[args.command](args)


if __name__ == '__main__':
    main()
