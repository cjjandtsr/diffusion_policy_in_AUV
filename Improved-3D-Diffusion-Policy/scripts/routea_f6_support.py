#!/usr/bin/env python3
"""F6 exact reproduction checks and explicit evaluation-only identity migration."""
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import struct
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_routea_a2 import digest, read, write

TRANSITION_FIELDS = ('episode_id', 'reset_seed', 'task_seed', 'action_chunk',
    'position_before_xyz', 'position_after_xyz', 'reward', 'reward_terms',
    'value', 'next_value', 'advantage_raw', 'return_target')

MIGRATION_LABEL = 'F6_G_A6_evaluation_copy'
MIGRATION_CONCLUSION = '计算内容与原迹逐位一致'
F6_MIGRATION_SOURCE_ALLOWLIST = frozenset((
    'diffusion_policy_3d/env/reward_v4.py', 'diffusion_policy_3d/env/kinematic_uav_env.py',
    *('diffusion_policy_3d/rl/dppo_uav_rebuild/'+p for p in (
        'rewards/v4/__init__.py', 'rewards/v4/facade.py', 'rewards/select.py',
        'environments/kinematic_adapter.py', 'entrypoints/build_system.py',
        'entrypoints/identity.py', 'entrypoints/pipeline.py', 'entrypoints/config_schema_v2.py',
        'entrypoints/a2_support.py', 'audit/training_audit.py', 'checkpointing/a2_fork.py',
        'checkpointing/restore.py', 'deployment/loader.py')),
    'scripts/run_routea_f6.py', 'scripts/f6_u129_matched_rollout.py',
    'scripts/routea_f6_support.py', 'scripts/analyze_routea_f6.py',
    'scripts/analyze_routea_f5.py', 'scripts/run_routea_a2.py',
    'scripts/run_routea_xk_evaluation.py', 'scripts/routea_paired_support/runtime.py',
    'scripts/routea_paired_support/episodes.py'))


def migration_source_changes(original, current):
    before = {r['relative_path']: r['sha256'] for r in original}
    after = {r['relative_path']: r['sha256'] for r in current}
    changes = [dict(relative_path=p, before_sha256=before.get(p), after_sha256=after.get(p))
               for p in sorted(set(before) | set(after)) if before.get(p) != after.get(p)]
    bad = [r['relative_path'] for r in changes if r['relative_path'] not in F6_MIGRATION_SOURCE_ALLOWLIST]
    if bad: raise ValueError('migration source outside F6 whitelist: '+repr(bad))
    return changes


def checkpoint_file_hashes(path):
    return {n: digest(Path(path)/n) for n in ('state.pt','header.json','SHA256.json','COMMITTED.json')}


def exact_differences(a, b, path='$'):
    """JSON scalars are compared as binary64; tensors by dtype, shape, bytes."""
    import torch
    differences = []
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor): return [path+': tensor type']
        if a.dtype != b.dtype or a.shape != b.shape: return [path+': tensor dtype/shape']
        x, y = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
        if not torch.equal(x.reshape(-1).view(torch.uint8), y.reshape(-1).view(torch.uint8)):
            return [path+': tensor bytes']
    elif type(a) is not type(b): return [path+': type']
    elif isinstance(a, dict):
        if set(a) != set(b): differences.append(path+': keys')
        for key in sorted(set(a) & set(b), key=str):
            differences.extend(exact_differences(a[key], b[key], path+'.'+str(key)))
    elif isinstance(a, (tuple, list)):
        if len(a) != len(b): differences.append(path+': length')
        for i, (x, y) in enumerate(zip(a, b)):
            differences.extend(exact_differences(x, y, path+'['+str(i)+']'))
    elif isinstance(a, float):
        if struct.pack('!d', a) != struct.pack('!d', b): differences.append(path+': float bits')
    elif a != b: differences.append(path+': value')
    return differences


def verified_state(path):
    import torch
    from scripts.routea_paired_support.core import verify_checkpoint
    path = Path(path)
    h, sums = read(path/'header.json'), read(path/'SHA256.json')
    verify_checkpoint(path, dict(runtime_identity_hash=h['runtime_identity_hash'],
        header_sha256=sums['header.json'], state_sha256=sums['state.pt'], update=h['update_index']))
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.identity import _canonical_hash
    state = torch.load(path/'state.pt', map_location='cpu', weights_only=True)
    if _canonical_hash(state['training_config']) != h['runtime_identity']['config_hash']:
        raise ValueError('checkpoint config hash mismatch')
    if _canonical_hash(h['runtime_identity']) != h['runtime_identity_hash']:
        raise ValueError('checkpoint runtime payload mismatch')
    return h, state


def validated_prepared(path):
    from scripts.run_routea_f6 import source_rows
    prepared = read(Path(path)/'prepared.json')
    if source_rows(Path(__file__).resolve().parents[1]) != prepared['files']:
        raise ValueError('reproduction source differs from prepared closure')
    for item in prepared['configs'].values():
        if digest(item['path']) != item['sha256']: raise ValueError('prepared configuration changed')
    return prepared


def compare_training(args):
    prepared = validated_prepared(args.prepared)
    a, b = read(args.old_metrics), read(args.new_metrics)
    diffs = []
    old, new = a['rollout_audit']['transition_records'], b['rollout_audit']['transition_records']
    if len(old) != 1024 or len(new) != 1024: raise ValueError('u129 requires 1024 transitions')
    for i, (x, y) in enumerate(zip(old, new)):
        for field in TRANSITION_FIELDS:
            if field not in x or field not in y: raise ValueError('missing required transition field: '+field)
            diffs += exact_differences(x[field], y[field], 'transition['+str(i)+'].'+field)
    if len(a['updates']) != 20 or len(b['updates']) != 20: raise ValueError('u129 requires 20 attempts')
    diffs += exact_differences(a['updates'], b['updates'], 'updates')
    ah, ast = verified_state(args.old_checkpoint)
    bh, bst = verified_state(args.new_checkpoint)
    if ah['update_index'] != 129 or bh['update_index'] != 129: raise ValueError('need actual u129 weights')
    if args.control not in ('P9','P11') or any(st['training_config'].get('f5', {}).get('run_id') != args.control for st in (ast,bst)):
        raise ValueError('wrong reproduction control checkpoint')
    from omegaconf import OmegaConf
    expected = OmegaConf.to_container(OmegaConf.load(prepared['configs'][args.control]['path']), resolve=True)
    if bst['training_config'] != expected:
        raise ValueError('new control is not the prepared configuration')
    # Include optimizer steps, all tensor states, RNG, sampler, schedules and buffer.
    numeric_a = {k: v for k, v in ast.items() if k != 'training_config'}
    numeric_b = {k: v for k, v in bst.items() if k != 'training_config'}
    diffs += exact_differences(numeric_a, numeric_b, 'state')
    ca, cb = copy.deepcopy(ast['training_config']), copy.deepcopy(bst['training_config'])
    wrappers = []
    for key in ('output_dir', 'result_dir'):
        wrappers.append(dict(field='training_config.'+key, old=ca.pop(key, None), new=cb.pop(key, None),
            reason='separate reproduction output, no computational change'))
    diffs += exact_differences(ca, cb, 'training_config')
    for key in ('runtime_identity', 'runtime_identity_hash'):
        wrappers.append(dict(field='header.'+key, old=ah[key], new=bh[key],
            reason='new source closure and reproduction output configuration; original identity validated'))
    wrappers += [dict(field='metrics.'+k, old=a.get(k), new=b.get(k),
        reason='source/config/checkpoint packaging under separately validated runtime')
        for k in ('runtime_identity_hash', 'environment_provenance', 'checkpoint', 'checkpoint_expected_path')]
    for i, (x, y) in enumerate(zip(old, new)):
        for key in set(x) | set(y):
            if key in TRANSITION_FIELDS: continue
            differences = exact_differences(x.get(key), y.get(key))
            if differences:
                # Only explicit identity fields are allowed to vary beyond the computational fields.
                if key not in ('checkpoint_identity_hash','config_hash','runtime_identity_hash','source_set_hash'):
                    diffs += ['transition['+str(i)+'].'+key+v for v in differences]
                else:
                    wrappers.append(dict(field='transition['+str(i)+'].'+key, old=x.get(key), new=y.get(key),
                        reason='new source/config runtime identity packaging'))
    report = dict(schema='routea_f6_training_reproduction_v1', control=args.control,
        passed=not diffs, prepared=prepared, computational_differences=diffs, packaging_differences=wrappers,
        inputs={k: dict(path=str(Path(v).resolve()), sha256=digest(v)) for k, v in
            [('old_metrics', args.old_metrics), ('new_metrics', args.new_metrics),
             ('old_state', Path(args.old_checkpoint)/'state.pt'), ('new_state', Path(args.new_checkpoint)/'state.pt')]})
    write(args.output, report)
    if diffs: raise ValueError('training reproduction differs; see report')


def compare_evaluation(args):
    from scripts.routea_paired_support.episodes import load_commit
    from scripts.analyze_routea_f5 import check_bank
    prepared = validated_prepared(args.prepared)
    migration = read(args.migration)
    native = read(migration['native_validation'])
    if (migration.get('passed') is not True or migration['source_files'] != prepared['files']
            or digest(migration['native_validation']) != migration['native_validation_sha256']
            or migration.get('original_checkpoint_unchanged') is not True
            or migration['original_checkpoint_files_before'] != migration['original_checkpoint_files_after']):
        raise ValueError('evaluation migration not bound to prepared sources/native validation')
    bank = read(args.bank)
    check_bank(bank, 'dev420')
    diffs, packaging = [], []
    jobs = [Path(args.old), Path(args.new)]
    expected = {e['key'] for e in bank['episodes']}
    for job, label in zip(jobs, ('P9', MIGRATION_LABEL)):
        if {p.stem for p in (job/'commits').glob('*.json')} != expected:
            raise ValueError('reproduction requires complete dev420')
        p = read(job/'policy_provenance.json')
        if (p['bank_hash'] != bank['content_hash'] or p['policy'] != label
                or p['actual_use_base_policy'] is not False
                or p['evaluation_sampling_protocol']['clip_protocol']['override'] != 2.):
            raise ValueError('wrong original/copy label or not paired #E2')
    pa, pb = [read(j/'policy_provenance.json') for j in jobs]
    if (pa['checkpoint_state_sha256'] != native['state_sha256']
            or pb['training_runtime_identity'] != migration['new_header']['runtime_identity_hash']
            or pb.get('evaluation_only') is not True
            or pb.get('purpose') != 'F6_G_A6_reproduction'):
        raise ValueError('evaluation reproduction checkpoints do not match migration')
    diffs += exact_differences(read(jobs[0]/'policy_provenance.json')['evaluation_sampling_protocol'],
                               read(jobs[1]/'policy_provenance.json')['evaluation_sampling_protocol'], 'sampling_protocol')
    for e in bank['episodes']:
        records = [load_commit(job, job/'commits'/(e['key']+'.json')) for job in jobs]
        for h, rows, end in records:
            if h['initial_hash'] != e['initial_hash'] or h['bank_hash'] != bank['content_hash']:
                raise ValueError('evaluation initial pairing mismatch')
        for index in (0, 2):
            x, y = records[0][index], records[1][index]
            keys = ('initial_hash', 'task_id', 'repeat', 'reset_seed') if index == 0 else ('done_reason', 'decisions')
            for key in keys:
                if key not in x or key not in y: raise ValueError('required evaluation field missing: '+key)
                diffs += exact_differences(x[key], y[key], e['key']+'.'+key)
        x, y = records[0][1], records[1][1]
        if len(x) != len(y): diffs.append(e['key']+': decision count')
        for i, (r, s) in enumerate(zip(x, y)):
            for key in ('full_action_16x11', 'policy_action_8x11', 'executed_path'):
                if key not in r or key not in s: raise ValueError('evaluation field missing: '+key)
                diffs += exact_differences(r[key], s[key], e['key']+'.decision'+str(i)+'.'+key)
        # Only the mandatory distinct copy label is packaging in the episode summary.
        ends = [dict(r[2]) for r in records]
        for end, label in zip(ends, ('P9', MIGRATION_LABEL)):
            if end.pop('policy') != label: raise ValueError('wrong original/copy summary label')
        computations = [(records[k][1], ends[k]) for k in (0,1)]
        diffs += exact_differences(computations[0], computations[1], e['key']+'.trajectory')
        ha, hb = (dict(records[k][0]) for k in (0, 1))
        if ha.pop('policy') != 'P9' or hb.pop('policy') != MIGRATION_LABEL:
            raise ValueError('copy must not be labeled P9')
        ia, ib = ha.pop('policy_identity_hash'), hb.pop('policy_identity_hash')
        diffs += exact_differences(ha, hb, e['key']+'.header')
        packaging.append(dict(episode=e['key'], field='policy_identity_hash', old=ia, new=ib,
            computational_trajectory_hash_old=__import__('hashlib').sha256(json.dumps(computations[0], sort_keys=True).encode()).hexdigest(),
            computational_trajectory_hash_new=__import__('hashlib').sha256(json.dumps(computations[1], sort_keys=True).encode()).hexdigest(),
            reason='explicit evaluation source identity migration; computational trajectory hashes must match'))
        # The byte hash includes the identity header; retain it as packaging evidence.
        for job in jobs:
            commit = read(job/'commits'/(e['key']+'.json'))
            packaging.append(dict(episode=e['key'], job=str(job), trace_sha256=commit['sha256'],
                reason='full trace includes validated runtime identity; actions/path compared separately'))
    write(args.output, dict(schema='routea_f6_evaluation_reproduction_v1', passed=not diffs,
        conclusion=MIGRATION_CONCLUSION if not diffs else '计算内容与原迹不一致，门禁未通过',
        prepared=prepared, computational_differences=diffs, packaging_differences=packaging, episodes=420,
        migration=dict(path=str(Path(args.migration).resolve()), sha256=digest(args.migration)),
        inputs=dict(old=str(jobs[0].resolve()), new=str(jobs[1].resolve()), bank_hash=bank['content_hash'])))
    if diffs: raise ValueError('evaluation reproduction differs; see report')


def seal_gates(args):
    prepared = validated_prepared(args.prepared)
    paths = [args.p9, args.p11, args.evaluation]
    reports = [read(p) for p in paths]
    if any(r.get('prepared') != prepared for r in reports):
        raise ValueError('stale reproduction report from different source/config preparation')
    if [r.get('control') for r in reports[:2]] != ['P9', 'P11']:
        raise ValueError('need both training controls')
    if [r.get('schema') for r in reports] != ['routea_f6_training_reproduction_v1']*2 + ['routea_f6_evaluation_reproduction_v1']:
        raise ValueError('wrong reproduction report types')
    for control, report in zip(('P9','P11'), reports):
        from omegaconf import OmegaConf
        expected = OmegaConf.load(prepared['configs'][control]['path'])
        if Path(report['inputs']['new_metrics']['path']).resolve() != (Path(expected.result_dir)/'metrics_update_000129.json').resolve():
            raise ValueError('training report not from prepared reproduction run')
        for item in report['inputs'].values():
            if digest(item['path']) != item['sha256']: raise ValueError('reproduction input changed')
    if any(r.get('passed') is not True for r in reports): raise ValueError('reproduction failed')
    write(args.output, dict(schema='routea_f6_reproduction_gates_v1', prepared=prepared,
        reports=[dict(path=str(Path(p).resolve()), sha256=digest(p)) for p in paths]))


def require_reproduction_gates(path, prepared):
    if not path: raise ValueError('both old-control reproduction gates required before F6')
    gate = read(path)
    if gate['schema'] != 'routea_f6_reproduction_gates_v1' or gate['prepared'] != prepared:
        raise ValueError('reproduction gates belong to different prepared sources/configs')
    if len(gate['reports']) != 3: raise ValueError('three reproduction reports required')
    for r in gate['reports']:
        if digest(r['path']) != r['sha256'] or read(r['path'])['passed'] is not True:
            raise ValueError('reproduction report missing/changed/failed')
        if read(r['path']).get('prepared') != prepared:
            raise ValueError('stale reproduction report')
    return gate


def verify_matched_rollout(config, prepared, gates_path, checkpoint):
    """Run G-A4 against the original same-seed control, after u129 and before u192."""
    from scripts.f6_u129_matched_rollout import compare_rollouts
    gate = require_reproduction_gates(gates_path, prepared)
    control = {'P12': 'P9', 'P13': 'P11'}[str(config.f6.run_id)]
    reports = [read(item['path']) for item in gate['reports']]
    report = next(r for r in reports if r.get('control') == control)
    original = report['inputs']['old_metrics']
    if digest(original['path']) != original['sha256']:
        raise ValueError('G-A4 original control metrics changed')
    treatment_path = Path(config.result_dir)/'metrics_update_000129.json'
    treatment, reference = read(treatment_path), read(original['path'])
    if (treatment['rollout_audit']['reward_schema_version'] != 'v4'
            or reference['rollout_audit']['reward_schema_version'] != 'v2'):
        raise ValueError('G-A4 requires F6 v4 and matched F3 v2')
    rows = treatment['rollout_audit']['transition_records']
    result = compare_rollouts(rows, reference['rollout_audit']['transition_records'])
    if any(row.get('run_seed') != int(config.run.seed) for row in rows):
        result['passed'] = False
        result['mismatch_examples'].append('rollout run_seed differs from registered F6 seed')
    result.update(schema='routea_f6_ga4_v1', run_id=str(config.f6.run_id), control=control,
        prepared=prepared, treatment_metrics_sha256=digest(treatment_path),
        control_metrics=original, checkpoint_state_sha256=digest(Path(checkpoint)/'state.pt'),
        probe_source_sha256=digest(Path(__file__).with_name('f6_u129_matched_rollout.py')))
    output = Path(config.result_dir)/'g_a4_u129.json'
    if not output.exists(): write(output, result)
    elif read(output) != result:
        raise ValueError('G-A4 recheck differs from saved report; do not continue')
    if not result['passed']:
        raise ValueError('G-A4 FAIL; no continuation: '+ '; '.join(result['mismatch_examples']))
    print('G_A4_MATCHED_ROLLOUT_PASS', str(config.f6.run_id), control)
    return result


def validate_native(args):
    from scripts.run_routea_xk_evaluation import native_system
    _, _, header = native_system(argparse.Namespace(source_root=args.source_root,
        checkpoint=args.checkpoint, repo=args.repo, expected_update=192))
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.identity import _source_rows
    prepared = read(Path(args.prepared)/'prepared.json')
    names = {r['relative_path'] for r in prepared['files']} | {r['relative_path'] for r in _source_rows(Path(args.source_root))}
    original_sources = [dict(relative_path=n, sha256=digest(Path(args.source_root)/n)
                            if (Path(args.source_root)/n).is_file() else None) for n in sorted(names)]
    write(args.output, dict(schema='routea_f6_native_validation_v1', source_files=original_sources,
        checkpoint=str(Path(args.checkpoint).resolve()), state_sha256=digest(Path(args.checkpoint)/'state.pt'),
        header=header, passed=True))


def rebind_evaluation(args):
    """Validate original native load in a fresh process, then migrate packaging explicitly.

    This checkpoint copy is only for the P9 evaluation reproduction gate, never training.
    All original numeric payload (including optimizer/RNG) stays bytewise equal by tensor.
    """
    import torch
    from omegaconf import OmegaConf
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.build_system import build_system
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.identity import build_runtime_identity
    from diffusion_policy_3d.rl.dppo_uav_rebuild.checkpointing.schema import CheckpointHeader
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.io_safety import atomic_publish_directory
    from scripts.run_routea_f6 import source_rows
    prepared = validated_prepared(args.prepared)
    original_hashes_before = checkpoint_file_hashes(args.checkpoint)
    report = Path(args.report)
    native_report = report.with_name(report.stem+'_native.json')
    subprocess.run([sys.executable, __file__, 'validate-native', '--repo', args.repo,
        '--source-root', args.source_root, '--checkpoint', args.checkpoint, '--prepared', args.prepared,
        '--output', str(native_report)], check=True)
    changed_sources = migration_source_changes(read(native_report)['source_files'], prepared['files'])
    old_h, state = verified_state(args.checkpoint)
    c = OmegaConf.create(state['training_config'])
    if c.reward_version != 'v2' or c.f5.run_id != 'P9' or old_h['update_index'] != 192:
        raise ValueError('only original P9 u192 evaluation migration allowed')
    root = Path(args.repo).resolve()
    system = build_system(c, root)
    identity = build_runtime_identity(c, root, system, Path(c.admission_manifest))
    new_h = CheckpointHeader(2, asdict(identity), identity.content_hash, old_h['lora_identity_hash'], 192)
    if system.lora_bundle.identity.content_hash != old_h['lora_identity_hash']:
        raise ValueError('LoRA target differs during evaluation migration')
    for k, v in old_h['runtime_identity'].items():
        if k != 'source_set_hash' and getattr(identity, k) != (tuple(v) if k == 'caveats' else v):
            raise ValueError('evaluation migration changes non-source runtime field: '+k)
    destination = Path(args.output).resolve()
    if destination.exists(): raise FileExistsError(destination)
    temporary = destination.with_name('.'+destination.name+'_publishing')
    temporary.mkdir(parents=True, exist_ok=False)
    (temporary/'header.json').write_text(new_h.to_json(), encoding='utf-8')
    marked_state = dict(state, evaluation_only=dict(purpose='F6_G_A6_reproduction',
        policy_label=MIGRATION_LABEL, original_runtime_identity_hash=old_h['runtime_identity_hash']))
    torch.save(marked_state, temporary/'state.pt')
    sums = {n: digest(temporary/n) for n in ('header.json', 'state.pt')}
    write(temporary/'SHA256.json', sums)
    atomic_publish_directory(temporary, destination, dict(schema_version=2,
        header_schema_hash=new_h.schema_hash, files=sums, status='committed'))
    _, migrated = verified_state(destination)
    marker = migrated.pop('evaluation_only')
    differences = exact_differences(state, migrated)
    if differences: raise ValueError('evaluation migration modified numeric payload')
    original_hashes_after = checkpoint_file_hashes(args.checkpoint)
    if original_hashes_after != original_hashes_before:
        raise ValueError('original checkpoint changed during migration')
    write(report, dict(schema='routea_f6_evaluation_identity_migration_v1', passed=True,
        old_header=old_h, new_header=asdict(new_h), native_validation=str(native_report.resolve()),
        native_validation_sha256=digest(native_report), source_files=prepared['files'],
        computational_differences=differences, evaluation_only=marker,
        original_checkpoint_files_before=original_hashes_before,
        original_checkpoint_files_after=original_hashes_after,
        original_checkpoint_unchanged=True, source_changes=changed_sources,
        source_whitelist=sorted(F6_MIGRATION_SOURCE_ALLOWLIST),
        use='evaluation reproduction only; not P9; no training, deployment or reported result',
        reason='strict original native validation followed by explicit source identity migration; original untouched'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for name, keys in [('compare-training', ('prepared','control','old-metrics','new-metrics','old-checkpoint','new-checkpoint','output')),
        ('compare-evaluation', ('prepared','migration','old','new','bank','output')), ('seal-gates', ('prepared','p9','p11','evaluation','output')),
        ('validate-native', ('prepared','repo','source-root','checkpoint','output')),
        ('rebind-evaluation', ('prepared','repo','source-root','checkpoint','output','report'))]:
        a = sub.add_parser(name)
        for key in keys: a.add_argument('--'+key, required=True)
    args = p.parse_args()
    globals()[args.command.replace('-', '_')](args)


if __name__ == '__main__': main()
