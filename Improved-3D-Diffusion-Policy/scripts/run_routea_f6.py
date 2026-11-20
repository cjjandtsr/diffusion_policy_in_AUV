#!/usr/bin/env python3
"""Prepare F6 and reproduction controls; run the unchanged training pipeline."""
import argparse
import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_routea_a2 import digest, read, write, verify_saved_source

RUNS = {'P12': 202610311, 'P13': 202610312}


def make_config(receipt, receipt_path, reference, output, weights, run_id):
    c = copy.deepcopy(receipt['training_config'])
    c['run'].update(seed=RUNS[run_id], checkpoint_every_updates=8,
        checkpoint_updates=sorted(set(c['run']['checkpoint_updates']) | {129}))
    c['output_dir'] = str(Path(weights).resolve() / run_id)
    c['result_dir'] = str(Path(output).resolve() / run_id)
    c['a2'] = dict(arm='T', parent_checkpoint=receipt['parent_checkpoint'],
        parent_receipt=str(Path(receipt_path).resolve()), parent_receipt_sha256=digest(receipt_path),
        parent_source_root=receipt['source_root'], reference_bank=str(Path(reference).resolve()),
        reference_sha256=digest(reference), keep_seed=2026102301)
    c['reward_version'] = 'v4'
    c['f6'] = dict(run_id=run_id, keep_coefficient=.01061006, potential=dict(K=10., R=2.))
    return c


def validate_treatment_control(treatment, control):
    expected = copy.deepcopy(treatment)
    registration = expected.pop('f6')
    paired = {'P12': 'P9', 'P13': 'P11'}[registration['run_id']]
    expected['f5'] = dict(run_id=paired, keep_coefficient=registration['keep_coefficient'])
    expected['reward_version'] = 'v2'
    for key in ('output_dir', 'result_dir'):
        expected.pop(key, None)
    original = {k: v for k, v in control.items() if k not in ('output_dir', 'result_dir')}
    if expected != original:
        raise ValueError('F6 differs from same-seed control beyond registered reward/packaging')


def source_rows(root):
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.identity import _source_rows
    rows = _source_rows(root)
    names = ['scripts/run_routea_f6.py', 'scripts/routea_f6_support.py', 'scripts/analyze_routea_f6.py',
             'scripts/f6_u129_matched_rollout.py', 'scripts/run_routea_paired_comparison.py']
    support = Path(root)/'scripts/routea_paired_support'
    names += [p.relative_to(root).as_posix() for pattern in ('*.py', '*.json') for p in sorted(support.glob(pattern))]
    for name in names:
        if name not in {r['relative_path'] for r in rows}:
            rows.append(dict(relative_path=name, sha256=digest(Path(root) / name)))
    return rows


def prepare(args):
    import torch
    from omegaconf import OmegaConf
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.config_schema_v2 import validate_v2_config
    from diffusion_policy_3d.rl.dppo_uav_rebuild.checkpointing.a2_fork import validate_child_config
    receipt = read(args.receipt)
    verify_saved_source(receipt)
    if digest(Path(receipt['parent_checkpoint']) / 'state.pt') != receipt['state_sha256']:
        raise ValueError('parent weight differs from receipt')
    reference = Path(args.reference)
    manifest = read(reference.parent / 'reference_manifest.json')
    if digest(reference) != manifest['sha256']:
        raise ValueError('reference hash differs')
    bank = torch.load(reference, map_location='cpu', weights_only=True)
    if (bank.get('schema') != 'routea_a2_train_success_v1' or len(bank['actions']) != 549
            or bank.get('parent_receipt_sha256') != digest(args.receipt)):
        raise ValueError('reference provenance/size differs')
    output, weights = Path(args.output).resolve(), Path(args.weights).resolve()
    configs = {}
    for name in RUNS:
        c = make_config(receipt, args.receipt, reference, output/'training', weights, name)
        validate_v2_config(OmegaConf.create(c))
        validate_child_config(receipt['training_config'], c)
        configs[name] = c
    # Reproduction controls retain original registration and computational config.
    for name, path in [('P9', args.p9_config), ('P11', args.p11_config)]:
        c = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        if c.get('f5', {}).get('run_id') != name or c['reward_version'] != 'v2':
            raise ValueError('wrong original reproduction control')
        c['output_dir'], c['result_dir'] = str(weights/'reproduction'/name), str(output/'reproduction'/name)
        validate_v2_config(OmegaConf.create(c))
        validate_child_config(receipt['training_config'], c)
        configs[name] = c
    for treatment, control in (('P12','P9'), ('P13','P11')):
        validate_treatment_control(configs[treatment], configs[control])
    for c in configs.values():
        if Path(c['output_dir']).exists(): raise FileExistsError(c['output_dir'])
    output.mkdir(parents=True, exist_ok=False)
    paths = {}
    for name, c in configs.items():
        path = output / (name+'.yaml')
        OmegaConf.save(OmegaConf.create(c), path)
        paths[name] = dict(path=str(path), sha256=digest(path))
    write(output/'resolved_configs.json', {
        name: OmegaConf.to_container(OmegaConf.create(c), resolve=True)
        for name, c in configs.items()})
    write(output/'prepared.json', dict(schema='routea_f6_prepared_v1', configs=paths,
        files=source_rows(Path(args.repo).resolve()), parent_receipt_sha256=digest(args.receipt),
        reference_sha256=digest(reference), stop_update=192))


def verify_first_update(config, checkpoint):
    import math
    import torch
    from omegaconf import OmegaConf
    from scripts.routea_paired_support.core import verify_checkpoint
    h, sums = read(checkpoint/'header.json'), read(checkpoint/'SHA256.json')
    verify_checkpoint(checkpoint, dict(runtime_identity_hash=h['runtime_identity_hash'],
        header_sha256=sums['header.json'], state_sha256=sums['state.pt'], update=129))
    state = torch.load(checkpoint/'state.pt', map_location='cpu', weights_only=True)
    if state['training_config'] != OmegaConf.to_container(config, resolve=True):
        raise ValueError('u129 configuration differs')
    if state['schedules']['selective_keep']['coefficient'] != .01061006:
        raise ValueError('fixed keep coefficient changed')
    counters = state['counters']
    if (counters['env_steps'] != 129*1024 or counters['committed_updates'] != 129*20
            or counters['rejected_updates'] != 0):
        raise ValueError('u129 actual saved training budget differs')
    m = read(Path(config.result_dir)/'metrics_update_000129.json')
    rows = m['rollout_audit']['transition_records']
    if (len(m['updates']) != 20 or any(x['status'] != 'committed' for x in m['updates'])
            or len(rows) != 1024 or m['formal_training']['hard_stop_reason'] is not None
            or m['formal_training']['committed_optimizer_steps'] != 20
            or m['formal_training']['rejected_optimizer_steps'] != 0):
        raise ValueError('u129 requires 20/20 commits and 1024 transitions without hard stop')
    if config.reward_version == 'v4':
        if m['rollout_audit']['reward_schema_version'] != 'v4': raise ValueError('not v4 audit')
        for row in rows:
            f = row['reward_terms']['goal_potential']
            if not math.isfinite(f) or not -10. <= f <= 9.9+1e-9:
                raise ValueError('v4 shaping missing/nonfinite/out of bounds')


def run(args):
    from omegaconf import OmegaConf
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.pipeline import run as train
    prepared = read(Path(args.prepared)/'prepared.json')
    root = Path(args.repo).resolve()
    if source_rows(root) != prepared['files']: raise ValueError('prepared sources changed')
    item = prepared['configs'][args.run_id]
    if digest(item['path']) != item['sha256']: raise ValueError('registered config changed')
    c = OmegaConf.load(item['path'])
    if args.run_id in ('P9', 'P11') and args.stop_after_update != 129:
        raise ValueError('reproduction controls stop at u129')
    if args.stop_after_update == 192:
        expected = Path(c.output_dir)/'checkpoint_update_000129'
        if not args.resume or Path(args.resume).resolve() != expected.resolve():
            raise ValueError('u192 requires this run checked u129')
        verify_first_update(c, expected)
        from scripts.routea_f6_support import require_reproduction_gates, verify_matched_rollout
        require_reproduction_gates(args.gates, prepared)
        verify_matched_rollout(c, prepared, args.gates, expected)
    elif args.resume:
        raise ValueError('u129 must fork from registered parent')
    # Before the first F6 GPU update, both old-control reproduction gates are mandatory.
    if args.run_id in RUNS:
        from scripts.routea_f6_support import require_reproduction_gates
        require_reproduction_gates(args.gates, prepared)
    train(c, root, resume_from=args.resume, stop_after_update=args.stop_after_update)
    if args.stop_after_update == 129:
        checkpoint = Path(c.output_dir)/'checkpoint_update_000129'
        verify_first_update(c, checkpoint)
        if args.run_id in RUNS:
            from scripts.routea_f6_support import verify_matched_rollout
            verify_matched_rollout(c, prepared, args.gates, checkpoint)
        print('F6_U129_CONTRACT_PASS' if args.run_id in RUNS else 'F3_CONTROL_U129_CONTRACT_PASS')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('prepare')
    for key in ('repo', 'receipt', 'reference', 'p9-config', 'p11-config', 'output', 'weights'):
        a.add_argument('--'+key, required=True)
    a = sub.add_parser('run')
    for key in ('repo', 'prepared', 'run-id'): a.add_argument('--'+key, required=True)
    a.add_argument('--stop-after-update', type=int, choices=[129, 192], required=True)
    a.add_argument('--resume')
    a.add_argument('--gates', help='completed reproduction-gates JSON; mandatory for F6')
    args = p.parse_args()
    globals()[args.command](args)


if __name__ == '__main__': main()
