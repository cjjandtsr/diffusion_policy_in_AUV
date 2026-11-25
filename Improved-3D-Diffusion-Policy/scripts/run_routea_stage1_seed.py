"""G17: fresh P14 using the P2 pipeline, registered seed admission and shared E2 evaluation."""
import argparse
import copy
from dataclasses import asdict
import importlib
import json
import math
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_routea_f3_resume import canonical, digest, read, write


def changed(a, b, prefix=''):
    if isinstance(a, dict) and isinstance(b, dict):
        return [p for k in sorted(set(a) | set(b))
                for p in changed(a.get(k), b.get(k), prefix+'.'+k if prefix else k)]
    return [] if a == b else [prefix]


def first_five(metric):
    keys = ('world_id', 'task_seed', 'reset_seed', 'position_before_xyz', 'goal_xyz', 'action_chunk')
    return [{k: r[k] for k in keys} for r in metric['rollout_audit']['transition_records'][:5]]


def gate_metric(metric, update):
    assert metric['rollout_audit']['transitions'] == 1024
    assert len(metric['updates']) == 20 and all(u['status'] == 'committed' for u in metric['updates'])
    f = metric['formal_training']
    assert f['mode'] == 'joint' and f['hard_stop_reason'] is None
    assert math.isclose(f['actor_learning_rate'], 2e-5-(update-1)/255*1e-5, rel_tol=0, abs_tol=1e-16)


def verify_source(plan):
    source = Path(plan['source_root'])
    assert all(digest(source/r['relative_path']) == r['sha256'] for r in plan['source_files'])
    return source


def configuration(plan, probe=False):
    assert canonical(plan['reference_config']) == plan['reference_inputs']['resolved_config_sha256']
    c = copy.deepcopy(plan['reference_config'])
    c['run']['seed'] = 202611051
    c['output_dir'] = plan['probe_weights'] if probe else plan['weights']
    c['result_dir'] = str(Path(plan['run'])/('probe_training' if probe else 'training'))
    assert changed(plan['reference_config'], c) == ['output_dir', 'result_dir', 'run.seed']
    assert c['run']['updates'] == 256 and not c.get('a2') and not c.get('task_bank')
    return c


def equivalence_configuration(plan, output, weights):
    assert canonical(plan['reference_config']) == plan['reference_inputs']['resolved_config_sha256']
    c = copy.deepcopy(plan['reference_config'])
    assert c['run']['seed'] == 202609203 and c['run']['updates'] == 256
    c['output_dir'] = str(Path(weights).resolve())
    c['result_dir'] = str(Path(output).resolve()/'training')
    assert changed(plan['reference_config'], c) == ['output_dir', 'result_dir']
    assert not c.get('a2') and not c.get('task_bank')
    return c


def metric_differences(left, right, path=()):
    if type(left) is not type(right):
        yield path, left, right
    elif isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                yield path+(key,), dict(present=key in left, value=left.get(key)), dict(present=key in right, value=right.get(key))
            else:
                yield from metric_differences(left[key], right[key], path+(key,))
    elif isinstance(left, list):
        if len(left) != len(right):
            yield path+('length',), len(left), len(right)
        for index, (a, b) in enumerate(zip(left, right)):
            yield from metric_differences(a, b, path+(index,))
    elif left != right:
        yield path, left, right


def metadata_difference(path):
    if path in (('checkpoint',), ('checkpoint_expected_path',), ('runtime_identity_hash',)):
        return True
    timing = {'peak_cuda_memory_bytes', 'rollout_seconds', 'rollout_transitions_per_second',
              'total_seconds', 'transition_credit_audit_seconds', 'update_seconds'}
    if len(path) == 2 and path[0] == 'formal_training' and path[1] in timing:
        return True
    return (path[:2] == ('rollout_audit', 'transition_records')
            and path[-1] in ('runtime_identity_hash', 'config_hash', 'source_set_hash'))


def compare_u1_metrics(reference, actual, differences_path=None):
    keys = ('transition_id', 'task_id', 'world_id', 'episode_id', 'decision_index',
            'task_seed', 'reset_seed', 'position_before_xyz', 'goal_xyz', 'action_chunk')
    def samples(metric, episodes):
        selected, seen = [], set()
        for row in metric['rollout_audit']['transition_records']:
            if episodes and row['episode_id'] in seen:
                continue
            seen.add(row['episode_id'])
            selected.append({k: row[k] for k in keys})
            if len(selected) == 5:
                break
        return selected
    report = dict(numeric_and_structure_equal=True, full_json_equal=True,
                  data_difference_count=0, metadata_difference_count=0,
                  difference_examples=[], first_data_difference=None, float_tolerance=0,
                  comparison='parsed JSON values/types/structure; no tensor-state or file-byte equivalence claim')
    for name, episodes in (('first_five_transitions', False), ('first_five_episodes', True)):
        a, b = samples(reference, episodes), samples(actual, episodes)
        report[name] = dict(reference=a, actual=b, equal=a == b, count_reference=len(a), count_actual=len(b))
    stream = None
    try:
        if differences_path is not None:
            stream = Path(differences_path).open('x', encoding='utf-8')
        for path, old, new in metric_differences(reference, actual):
            metadata = (metadata_difference(path) and type(old) is type(new)
                        and not isinstance(old, (dict, list)))
            category = 'metadata' if metadata else 'data'
            report[category+'_difference_count'] += 1
            row = dict(path=list(path), category=category, reference=old, actual=new)
            if category == 'data' and report['first_data_difference'] is None:
                report['first_data_difference'] = row
            if stream is not None:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
            if len(report['difference_examples']) < 50:
                report['difference_examples'].append(row)
    finally:
        if stream is not None:
            stream.close()
    report['numeric_and_structure_equal'] = report['data_difference_count'] == 0
    report['full_json_equal'] = report['data_difference_count'] + report['metadata_difference_count'] == 0
    return report


def native_pipeline(plan):
    source = verify_source(plan)
    sys.path.insert(0, str(source))
    # The launcher imported only stdlib helpers from ROOT scripts. All computation
    # helpers subsequently imported by the native pipeline must resolve to native_a1.
    import scripts
    scripts.__path__ = [str(source/'scripts')]
    identity = importlib.import_module('diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.identity')
    builder = importlib.import_module('diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.build_system')
    pipeline = importlib.import_module('diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.pipeline')
    for module in (identity, builder, pipeline):
        Path(module.__file__).resolve().relative_to(source.resolve())
    # Same native-code/live-asset separation as the accepted G3 native_system.
    rows, symbols = identity._source_rows, builder.load_official_symbols
    assert rows(source) == plan['source_files']
    identity._source_rows = lambda unused: rows(source)
    builder.load_official_symbols = lambda unused: symbols(source/'third_party/dppo_official_cc7234a')
    provenance = pipeline.build_environment_provenance
    pipeline.build_environment_provenance = lambda unused, **kw: provenance(source, **kw)
    return pipeline, identity


def preflight(plan):
    from omegaconf import OmegaConf
    source = verify_source(plan)
    root, out = Path(plan['repo']), Path(plan['run'])
    out.mkdir(parents=True, exist_ok=False)
    for p in (plan['weights'], plan['probe_weights'], out/'training', out/'probe_training'):
        assert not Path(p).exists(), 'fresh run requires unused output: '+str(p)
    bc = root/plan['reference_config']['checkpoint']
    assert digest(bc) == plan['reference_inputs']['bc_checkpoint_sha256']
    pipeline, identity = native_pipeline(plan)
    config = OmegaConf.create(configuration(plan))
    # Run the actual native gate here, before a preflight success can be reported.
    pipeline._validate_run_config(config)
    manifest = identity.build_executed_source_manifest(root, config, Path(config.admission_manifest))
    expected = copy.deepcopy(plan['reference_inputs'])
    actual = copy.deepcopy(manifest['inputs'])
    expected.pop('resolved_config_sha256')
    actual.pop('resolved_config_sha256')
    assert actual == expected, 'P2 behavior-bearing assets differ'
    from diffusion_policy_3d.rl.dppo_uav.integration import manifest_train_entries
    worlds = sorted({e['world_uid'] for e in manifest_train_entries(config.admission_manifest)})
    assert len(worlds) == 21
    eval_worlds = set()
    for spec in plan['evaluation']['banks'].values():
        bank = next((Path(p) for p in spec['paths'] if Path(p).is_file()
                     and digest(p) == spec['file_sha256']), None)
        assert bank is not None, 'missing exact evaluation bank'
        payload = read(bank)
        assert payload['content_hash'] == spec['content_hash']
        eval_worlds.update(payload['worlds'])
    assert not set(worlds) & eval_worlds, 'evaluation world ID in training sampler'
    for label, row in plan['references'].items():
        assert digest(row['path']) == row['sha256'], label
        target = out/'references'/(label+'.json')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row['path'], target)
        assert digest(target) == row['sha256']
    assert pipeline._execution_last_update(config, 1, 128) == 128
    assert pipeline._actor_lr_for_update(config, 1) == 2e-5
    assert math.isclose(pipeline._actor_lr_for_update(config, 128), 1.5019607843137257e-5, abs_tol=1e-16)
    write(out/'resolved_config.json', configuration(plan))
    for row in plan['source_files']:
        target = out/'source_closure'/row['relative_path']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/row['relative_path'], target)
    write(out/'preflight.json', dict(plan_sha256=digest(ARGS.plan), source_files=plan['source_files'],
          launcher_sha256=digest(__file__),
          source_root=str(source), config_diff=['output_dir', 'result_dir', 'run.seed'],
          training_worlds=worlds, training_worlds_hash=canonical(worlds),
          reference_training_worlds_hash=canonical(worlds),
          basis='reference pool reconstructed from byte-identical P2 admission manifest using the same original factory; not inferred from sampled episodes',
          evaluation_world_id_overlap=[], geometry_disjointness_claimed=False,
          input_manifest=manifest, parent_file_sha256=digest(bc)))
    print('G17_PREFLIGHT_PASS', flush=True)


def train(plan, probe):
    from omegaconf import OmegaConf
    root, out = Path(plan['repo']), Path(plan['run'])
    report = read(out/'preflight.json')
    assert report['plan_sha256'] == digest(ARGS.plan)
    assert report['launcher_sha256'] == digest(__file__)
    config = OmegaConf.create(configuration(plan, probe))
    for path in (config.output_dir, config.result_dir):
        assert not Path(path).exists(), 'do not resume or overwrite '+str(path)
    pipeline, identity = native_pipeline(plan)
    # Observe the real strict BC loader and actual published metrics; do not change updates or RNG.
    build, publish = pipeline.build_system, pipeline._write_json_once
    def load(c, repo):
        bc = root/c.checkpoint
        assert digest(bc) == plan['reference_inputs']['bc_checkpoint_sha256']
        system = build(c, repo)
        assert system.bc_runtime.identity.checkpoint_sha256 == plan['reference_inputs']['bc_checkpoint_sha256']
        write(out/('probe_parent_load.json' if probe else 'parent_load.json'),
              dict(checkpoint=str(bc), file_sha256=digest(bc), identity=asdict(system.bc_runtime.identity),
                   initialization='fresh_from_P0', resume_from=None))
        return system
    def save(path, payload):
        publish(path, payload)
        if Path(path).name == 'metrics_update_000001.json':
            gate_metric(payload, 1)
            five = first_five(payload)
            old = plan['p2_first_five']
            assert five != old, 'new seed did not change first five rollout records'
            if not probe:
                prior = read(out/'probe_training/metrics_update_000001.json')
                assert five == first_five(prior), 'same-seed fresh restart mismatch'
                write(out/'u1_gate.json', dict(status='passed', P2=old, P14=five,
                      same_seed_probe_identical=True, compared='world/task/reset seeds, position, goal, executed action'))
                print('G17_U1_PASS', flush=True)
    pipeline.build_system, pipeline._write_json_once = load, save
    # Separate diagnostic u1 and fresh formal u1--128. No resume at u1.
    pipeline.run(config, root, resume_from=None, stop_after_update=1 if probe else 128)
    assert verify_source(plan)
    if probe:
        print('G17_PROBE_PASS', flush=True)
        return
    from scripts.routea_paired_support.core import verify_checkpoint
    for u in range(1, 129):
        metric = read(Path(config.result_dir)/('metrics_update_%06d.json' % u))
        gate_metric(metric, u)
    checkpoints = []
    for u in [0, 1]+list(range(8, 129, 8)):
        path = Path(config.output_dir)/('checkpoint_update_%06d' % u)
        h, sums = read(path/'header.json'), read(path/'SHA256.json')
        row = dict(update=u, runtime_identity_hash=h['runtime_identity_hash'],
                   state_sha256=sums['state.pt'], header_sha256=sums['header.json'])
        verify_checkpoint(path, row)
        assert h['runtime_identity']['config_hash'] == canonical(configuration(plan))
        destination = out/'weights_evidence'/path.name
        destination.mkdir(parents=True, exist_ok=False)
        for name in ('header.json', 'SHA256.json', 'COMMITTED.json'):
            shutil.copy2(path/name, destination/name)
        checkpoints.append(row)
    write(out/'training_complete.json', dict(status='passed', policy='P14', updates=128,
          configured_updates=256, automatic_training_stop=False, checkpoints=checkpoints,
          source_unchanged=True, checkpoint=str(path), **checkpoints[-1]))
    evaluation = copy.deepcopy(plan['evaluation'])
    evaluation['models'] = [dict(policy='P14', source_root=plan['source_root'],
                                checkpoint=str(path), **checkpoints[-1])]
    write(out/'evaluation_plan.json', evaluation)
    print('G17_TRAIN_PASS', 128, flush=True)


def equivalence(plan):
    from omegaconf import OmegaConf
    root, out, weights = Path(plan['repo']), Path(ARGS.output), Path(ARGS.weights)
    reference_path = Path(ARGS.reference_metrics)
    reference_sha = '971dd4efd1ffb37b068bc47c67709ce04ec44da08fe07741261680782fbe68d3'
    assert digest(reference_path) == reference_sha, 'requires original P2 u1 metric bytes'
    reference = read(reference_path)
    gate_metric(reference, 1)
    assert all(r['run_seed'] == 202609203 for r in reference['rollout_audit']['transition_records'])
    assert not out.exists() and not weights.exists(), 'use fresh diagnostic directories'
    assert plan['source_delta']['after_sha256'] == '86f9f1a4a9882d42c999879413f00f1ba612a7b0d704f8b34ce148878aa89b2e'
    pipeline, identity = native_pipeline(plan)
    config = OmegaConf.create(equivalence_configuration(plan, out, weights))
    pipeline._validate_run_config(config)
    assert pipeline._execution_last_update(config, 1, 1) == 1
    assert digest(root/config.checkpoint) == plan['reference_inputs']['bc_checkpoint_sha256']
    manifest = identity.build_executed_source_manifest(root, config, Path(config.admission_manifest))
    before, after = copy.deepcopy(plan['reference_inputs']), copy.deepcopy(manifest['inputs'])
    before.pop('resolved_config_sha256')
    after.pop('resolved_config_sha256')
    assert before == after, 'P2 behavior-bearing assets changed'
    out.mkdir(parents=True, exist_ok=False)
    write(out/'resolved_config.json', OmegaConf.to_container(config, resolve=True))
    write(out/'preflight.json', dict(plan_sha256=digest(ARGS.plan), launcher_sha256=digest(__file__),
          reference_metrics_sha256=reference_sha, source_root=plan['source_root'], source_manifest=manifest,
          seed=202609203, configured_updates=256, stop_after_update=1, purpose='diagnostic only'))
    shutil.copy2(reference_path, out/'reference_metrics.json')
    shutil.copy2(__file__, out/'launcher.py')
    build = pipeline.build_system
    def load(c, repo):
        system = build(c, repo)
        assert system.bc_runtime.identity.checkpoint_sha256 == plan['reference_inputs']['bc_checkpoint_sha256']
        write(out/'parent_load.json', dict(checkpoint=str(root/c.checkpoint),
              file_sha256=digest(root/c.checkpoint), identity=asdict(system.bc_runtime.identity),
              initialization='fresh_from_P0', resume_from=None))
        return system
    pipeline.build_system = load
    # Original native training sampling and RNG; no E2 override or deterministic-mode change.
    pipeline.run(config, root, resume_from=None, stop_after_update=1)
    verify_source(plan)
    actual = read(out/'training/metrics_update_000001.json')
    gate_metric(actual, 1)
    assert all(r['run_seed'] == 202609203 for r in actual['rollout_audit']['transition_records'])
    report = compare_u1_metrics(reference, actual, out/'all_metric_differences.jsonl')
    report.update(reference_metrics_sha256=reference_sha,
                  actual_metrics_sha256=digest(out/'training/metrics_update_000001.json'))
    write(out/'comparison.json', report)
    for u in (0, 1):
        ckpt = weights/('checkpoint_update_%06d' % u)
        target = out/'weights_evidence'/ckpt.name
        target.mkdir(parents=True, exist_ok=False)
        for name in ('header.json', 'SHA256.json', 'COMMITTED.json'):
            shutil.copy2(ckpt/name, target/name)
    passed = (report['numeric_and_structure_equal'] and report['first_five_transitions']['equal']
              and report['first_five_episodes']['equal'])
    write(out/'completion.json', dict(status='passed' if passed else 'different', updates=1,
          transitions=1024, committed=20, scope='metrics/collection equivalence only',
          source_unchanged=True, data_difference_count=report['data_difference_count'],
          metadata_difference_count=report['metadata_difference_count']))
    if not passed:
        raise ValueError('u1 comparison differs; preserve all evidence, do not rerun or relax tolerance')
    print('G17_U1_EQUIVALENCE_PASS', flush=True)


def main():
    global ARGS
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['preflight', 'probe', 'train', 'equivalence'])
    p.add_argument('--plan', required=True)
    p.add_argument('--output')
    p.add_argument('--weights')
    p.add_argument('--reference-metrics')
    ARGS = p.parse_args()
    extra = (ARGS.output, ARGS.weights, ARGS.reference_metrics)
    if ARGS.phase == 'equivalence' and not all(extra):
        p.error('equivalence requires --output, --weights and --reference-metrics')
    if ARGS.phase != 'equivalence' and any(extra):
        p.error('extra diagnostic paths are only for equivalence')
    plan = read(ARGS.plan)
    if ARGS.phase == 'equivalence':
        equivalence(plan)
    elif ARGS.phase == 'preflight':
        preflight(plan)
    else:
        train(plan, ARGS.phase == 'probe')


if __name__ == '__main__':
    main()
