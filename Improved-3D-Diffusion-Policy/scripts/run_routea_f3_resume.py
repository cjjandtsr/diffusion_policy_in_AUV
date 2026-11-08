"""G16: inspect original F5 state, then resume the unchanged native pipeline."""
import argparse
import ast
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
import sys


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def inventory(roots):
    return {str(p.resolve()): digest(p) for root in roots
            for p in sorted(Path(root).rglob('*')) if p.is_file()}


def same(a, b):
    import numpy as np
    import torch
    if torch.is_tensor(a) and torch.is_tensor(b):
        return a.dtype == b.dtype and torch.equal(a.detach().cpu(), b.detach().cpu())
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def state_keys(state):
    names = ['actor_optimizer', 'critic_optimizer', 'schedules', 'task_sampler', 'counters']
    return dict(top=sorted(state), nested={k: sorted(map(str, state[k]))
                if isinstance(state[k], dict) else None for k in names})


def gate193(metrics, config, history, restored):
    assert restored, 'training state was not verified after restore'
    formal, updates = metrics['formal_training'], metrics['updates']
    expected_lr = 2e-5 - 192 / 255 * 1e-5
    assert math.isclose(formal['actor_learning_rate'], expected_lr, rel_tol=0, abs_tol=1e-16)
    assert len(updates) == 20 and all(u['status'] == 'committed' for u in updates)
    assert metrics['rollout_audit']['transitions'] == 1024
    records = metrics['rollout_audit']['transition_records']
    assert len(records) == 1024 and all(r['task_bank']['source'] == 'original' for r in records)
    keep = [u['pre_update']['keep_diagnostics'] for u in updates]
    assert all(k['coefficient'] == .01061006 and math.isfinite(k['keep_loss']) for k in keep)
    assert any(k['keep_loss'] != 0 for k in keep)
    stats = formal['optimizer_metric_summary']
    for key in ('approx_kl', 'ratio_mean', 'value_loss'):
        assert all(math.isfinite(stats[key][x]) for x in ('min', 'max', 'mean'))
    assert stats['approx_kl']['max'] <= float(config.gates.max_approx_kl)
    assert stats['ratio_mean']['min'] >= float(config.gates.ratio_mean_min)
    assert stats['ratio_mean']['max'] <= float(config.gates.ratio_mean_max)
    ranges = {key: dict(min=min(h['formal_training']['optimizer_metric_summary'][key]['min'] for h in history),
                       max=max(h['formal_training']['optimizer_metric_summary'][key]['max'] for h in history))
              for key in ('approx_kl', 'ratio_mean', 'value_loss')}
    return dict(status='passed', learning_rate=dict(expected=expected_lr, actual=formal['actor_learning_rate']),
                keep_coefficient=.01061006, keep_loss_range=[min(k['keep_loss'] for k in keep),
                                                          max(k['keep_loss'] for k in keep)],
                transitions=1024, committed=20, current=stats, historical_u185_u192=ranges,
                judgement='Exact restored-state equality and existing numerical gates are hard gates; '
                          'historical loss ranges are descriptive, not new early-stop thresholds.')


def preflight(args, plan, root):
    import torch
    from omegaconf import OmegaConf
    from scripts.routea_paired_support.core import verify_checkpoint
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.identity import _source_rows
    prepared = read(plan['prepared'])
    assert _source_rows(root) == prepared['files'], 'original F5 source closure changed'
    result = dict(source_files=prepared['files'], arms={})
    for name, arm in plan['arms'].items():
        assert digest(arm['config']) == arm['config_sha256']
        c = OmegaConf.load(arm['config'])
        config = OmegaConf.to_container(c, resolve=True)
        assert canonical(config) == arm['config_hash']
        assert c.run.updates == 256 and c.reward_version == 'v2' and not c.get('task_bank')
        assert digest(c.a2.reference_bank) == arm['reference_sha256']
        assert digest(Path(c.result_dir)/'r1_plan_probe.json') == arm['probe_sha256']
        assert not (Path(c.result_dir)/'run.lock').exists(), 'another writer or unresolved stale lock'
        keys = {}
        for update in (129, 192):
            ckpt = Path(c.output_dir)/('checkpoint_update_%06d' % update)
            verify_checkpoint(ckpt, arm['checkpoints'][str(update)])
            state = torch.load(ckpt/'state.pt', map_location='cpu', weights_only=True)
            assert not state.get('evaluation_only')
            assert canonical(state['training_config']) == arm['config_hash']
            assert state['actor_optimizer']['state'] and state['critic_optimizer']['state']
            assert state['task_sampler'] and state['schedules']['selective_keep']['coefficient'] == .01061006
            keys[str(update)] = state_keys(state)
            del state
        assert keys['129'] == keys['192'], 'u129/u192 state field sets differ'
        baseline = inventory([c.output_dir, c.result_dir])
        assert not any('metrics_update_%06d.json' % u in p or 'checkpoint_update_%06d' % u in p
                       for p in baseline for u in range(193, 257)), 'continuation artifacts already exist'
        result['arms'][name] = dict(state_keys=keys, existing_files=baseline,
                                  config=config, source='original F5 prepared configuration')
    for row in prepared['files']:
        source = root/row['relative_path']
        target = Path(args.output)/'source_closure'/row['relative_path']
        target.parent.mkdir(parents=True, exist_ok=True)
        assert not target.exists()
        shutil.copy2(source, target)
        assert digest(target) == row['sha256']
    write(Path(args.output)/'preflight.json', result)
    print('G16_PREFLIGHT_PASS', flush=True)


def train(args, plan, root):
    import torch
    from omegaconf import OmegaConf
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints import pipeline
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.identity import _source_rows
    report = read(Path(args.output)/'preflight.json')
    arm = plan['arms'][args.arm]
    before = report['arms'][args.arm]
    assert _source_rows(root) == report['source_files']
    assert all(Path(p).is_file() and digest(p) == h for p, h in before['existing_files'].items())
    assert digest(arm['config']) == arm['config_sha256']
    config = OmegaConf.load(arm['config'])
    assert canonical(OmegaConf.to_container(config, resolve=True)) == arm['config_hash']
    checkpoint = Path(config.output_dir)/'checkpoint_update_000192'
    assert digest(checkpoint/'state.pt') == arm['checkpoints']['192']['state_sha256']
    saved = torch.load(checkpoint/'state.pt', map_location='cpu', weights_only=True)
    history = [read(Path(config.result_dir)/('metrics_update_%06d.json' % u)) for u in range(185, 193)]
    restored = []
    original_restore, original_write = pipeline.restore_checkpoint, pipeline._write_json_once

    def restore(*pos, **kw):
        header = original_restore(*pos, **kw)
        actual = dict(lora=kw['lora_bundle'].adapter_state_dict(), critic=kw['critic'].state_dict(),
                      actor_optimizer=kw['actor_optimizer'].state_dict(),
                      critic_optimizer=kw['critic_optimizer'].state_dict(),
                      counters=vars(kw['counters']), task_sampler=kw['task_sampler'].state_dict(),
                      schedules={k: v.state_dict() for k, v in kw['schedules'].items()})
        checks = {k: same(actual[k], saved[k]) for k in actual}
        assert all(checks.values()), 'restored state differs: ' + str(checks)
        assert header.runtime_identity['config_hash'] == arm['config_hash']
        assert header.runtime_identity_hash == arm['runtime_identity_hash'] and header.update_index == 192
        write(Path(args.output)/args.arm/'restored_state.json', dict(checks=checks,
              runtime_identity_hash=arm['runtime_identity_hash'], config_hash=arm['config_hash']))
        restored.append(True)
        return header

    def publish(path, payload):
        original_write(path, payload)
        if Path(path).name == 'metrics_update_000193.json':
            gate = gate193(payload, config, history, bool(restored))
            write(Path(args.output)/args.arm/'u193_gate.json', gate)
            print('G16_U193_PASS', args.arm, flush=True)

    pipeline.restore_checkpoint, pipeline._write_json_once = restore, publish
    try:
        pipeline.run(config, root, resume_from=str(checkpoint), stop_after_update=256)
        assert (Path(args.output)/args.arm/'u193_gate.json').is_file()
        for update in range(193, 257):
            metric_path = Path(config.result_dir)/('metrics_update_%06d.json' % update)
            m = read(metric_path)
            assert math.isclose(m['formal_training']['actor_learning_rate'],
                                2e-5 - (update - 1)/255*1e-5, rel_tol=0, abs_tol=1e-16)
            assert len(m['updates']) == 20 and all(u['status'] == 'committed' for u in m['updates'])
            assert m['rollout_audit']['transitions'] == 1024
            target = Path(args.output)/'metrics'/args.arm/metric_path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            assert not target.exists()
            shutil.copy2(metric_path, target)
            assert digest(target) == digest(metric_path)
        from scripts.routea_paired_support.core import verify_checkpoint
        for update in range(200, 257, 8):
            ckpt = Path(config.output_dir)/('checkpoint_update_%06d' % update)
            sums = read(ckpt/'SHA256.json')
            verify_checkpoint(ckpt, dict(runtime_identity_hash=arm['runtime_identity_hash'], update=update,
                              state_sha256=sums['state.pt'], header_sha256=sums['header.json']))
            destination = Path(args.output)/'weights_evidence'/args.arm/ckpt.name
            destination.mkdir(parents=True, exist_ok=False)
            for name in ('header.json', 'SHA256.json', 'COMMITTED.json'):
                shutil.copy2(ckpt/name, destination/name)
                assert digest(ckpt/name) == digest(destination/name)
        with (Path(args.output)/args.arm/'curve_u129_u256.csv').open('x', newline='', encoding='utf-8') as stream:
            curve = csv.writer(stream)
            curve.writerow(['update', 'actor_lr', 'committed', 'transitions', 'policy_loss',
                            'value_loss', 'approx_kl', 'ratio_mean'])
            for update in range(129, 257):
                m = read(Path(config.result_dir)/('metrics_update_%06d.json' % update))
                f = m['formal_training']
                curve.writerow([update, f['actor_learning_rate'], f['committed_optimizer_steps'],
                                m['rollout_audit']['transitions']] + [f['optimizer_metric_summary'][k]['mean']
                                for k in ('policy_loss', 'value_loss', 'approx_kl', 'ratio_mean')])
    finally:
        pipeline.restore_checkpoint, pipeline._write_json_once = original_restore, original_write
        current = inventory([config.output_dir, config.result_dir])
        unchanged = {p: current.get(p) == h for p, h in before['existing_files'].items()}
        added = sorted(current.keys() - before['existing_files'].keys())
        for path in added:
            relative = Path(path).relative_to(Path(config.output_dir).resolve()) if (
                Path(config.output_dir).resolve() in Path(path).parents) else Path(path).relative_to(Path(config.result_dir).resolve())
            match = re.fullmatch(r'(?:checkpoint_update_|metrics_update_)(\d{6})(?:\.json)?', relative.parts[0])
            assert match and 193 <= int(match.group(1)) <= 256, 'unexpected new artifact: ' + path
        source_unchanged = _source_rows(root) == report['source_files']
        write(Path(args.output)/args.arm/'original_files_unchanged.json',
              dict(files=unchanged, added_files=added, source_unchanged=source_unchanged))
        assert all(unchanged.values()) and source_unchanged, 'pre-existing evidence or closure changed'
    write(Path(args.output)/args.arm/'training_complete.json', dict(status='passed', updates=64,
          update_range=[193, 256], automatic_training_stop=False,
          checkpoint=str(Path(config.output_dir)/'checkpoint_update_000256'),
          state_sha256=sums['state.pt'], runtime_identity_hash=arm['runtime_identity_hash'],
          config_hash=arm['config_hash']))
    print('G16_TRAIN_PASS', args.arm, 64, flush=True)


def bank_repeats(bank, spec):
    if bank['schema'] == 'routea_train_domain_paired_dev_v1':
        assert spec['split'] == 'rl_train' and bank['domain'] == 'rl_train dev, fresh tasks'
        repeats = 5
    elif bank['schema'] == 'routea_fixed_split_v1':
        assert bank['split'] == spec['split']
        repeats = bank['repeats']
        assert repeats in (1, 5)
    else:
        raise ValueError('unsupported frozen bank schema: ' + bank['schema'])
    tasks = {t['task_id'] for t in bank['tasks']}
    assert len(tasks) == len(bank['tasks'])
    actual = {(e['task_id'], e['repeat']) for e in bank['episodes']}
    expected = {(task, repeat) for task in tasks for repeat in range(1, repeats + 1)}
    assert len(actual) == len(bank['episodes']) == spec['episodes'] and actual == expected
    return repeats


def evaluate(args, plan, root):
    from scripts.routea_paired_support.core import verify_checkpoint
    out = Path(args.output)
    if 'training_preflight' in plan:
        from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.identity import _source_rows
        assert digest(plan['training_preflight']) == plan['training_preflight_sha256']
        expected_sources = read(plan['training_preflight'])['source_files']
        closure_ok = lambda: _source_rows(root) == expected_sources
    else:
        # Other compatible endpoints use the same worker/audit, with their actual native closure.
        expected_sources = plan['source_files']
        source_root = Path(plan['source_root'])
        closure_ok = lambda: all(digest(source_root/r['relative_path']) == r['sha256']
                                 for r in expected_sources)
    assert closure_ok(), 'native source closure changed'
    total = sum(b['episodes'] for b in plan['banks'].values()) * len(plan['models'])
    assert total == plan.get('expected_episodes', 2080)
    for relative, expected in plan['scripts'].items():
        assert digest(root/relative) == expected, relative
    validator = root/'scripts/routea_paired_support/episodes.py'
    before = validator.read_bytes()
    nodes = [n for n in ast.parse(before.decode()).body if isinstance(n, ast.FunctionDef)
             and n.name == 'validate_rows']
    assert len(nodes) == 1
    node, lines = nodes[0], before.splitlines(keepends=True)
    body = b''.join(lines[node.lineno-1:node.end_lineno])
    assert hashlib.sha256(body).hexdigest() == '94dc969320322fcaf2317f2dfe958275ca4ad5fba44a6b72a88d8fad8f79f627'
    old = b'goal = np.asarray(header["task"]["goal_pos"], np.float64)'
    new = old.replace(b'float64', b'float32')
    assert body.count(old) == 1
    after = b''.join(lines[:node.lineno-1]) + body.replace(old, new) + b''.join(lines[node.end_lineno:])
    resolved = {}
    for dataset, spec in plan['banks'].items():
        choices = [Path(p) for p in spec['paths'] if Path(p).is_file()]
        assert choices, 'missing frozen bank: ' + dataset
        bank_path = next((p for p in choices if digest(p) == spec['file_sha256']), None)
        assert bank_path is not None, 'bank file identity mismatch: ' + dataset
        bank = read(bank_path)
        assert canonical({k: v for k, v in bank.items() if k != 'content_hash'}) == spec['content_hash'] == bank['content_hash']
        repeats = bank_repeats(bank, spec)
        for e in bank['episodes']:
            assert digest(bank_path.parent/e['initial_npz']) == e['initial_npz_sha256']
        for w in bank['worlds'].values():
            assert digest(w['obstacle_path']) == w['obstacle_sha256'], 'restore missing asset without changing bank'
        resolved[dataset] = dict(path=str(bank_path), repeats=repeats, schema=bank['schema'])
    for model in plan['models']:
        verify_checkpoint(model['checkpoint'], model)
    write(out/'resolved_inputs.json', dict(banks=resolved, plan_sha256=digest(args.plan),
                                         original_validator_sha256=hashlib.sha256(before).hexdigest()))
    sources = [root/p for p in plan['scripts']]
    sources += sorted((root/'scripts/routea_paired_support').glob('*.py'))
    snapshots = {}
    for source in sorted(set(sources)):
        relative = source.relative_to(root)
        target = out/'source'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assert digest(target) == digest(source)
        snapshots[relative.as_posix()] = digest(target)
    write(out/'source_snapshot.json', snapshots)
    (out/'validator').mkdir()
    (out/'validator/episodes.before.py').write_bytes(before)
    (out/'validator/episodes.after.py').write_bytes(after)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPYCACHEPREFIX=str(out/'unused_bytecode'))
    restorations, completed = [], []

    def command(label, argv):
        stem = out/'execution'/label
        write(stem.with_suffix('.command.json'), dict(argv=argv,
              bytecode_policy='fresh source compilation; no bytecode writes'))
        with stem.with_suffix('.log').open('x', encoding='utf-8') as log:
            rc = subprocess.run(argv, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        stem.with_suffix('.rc').write_text(str(rc)+'\n', encoding='utf-8')
        if rc:
            raise RuntimeError(label + ' failed; preserve evidence')

    try:
        for dataset, spec in plan['banks'].items():
            bank_path = resolved[dataset]['path']
            for model in plan['models']:
                name = model['policy']
                job = out/'full'/dataset/name
                assert validator.read_bytes() == before
                command(dataset+'_'+name+'_worker', [sys.executable,
                    str(root/'scripts/run_routea_goal_precision.py'), 'worker', '--repo', str(root),
                    '--current-root', str(root), '--source-root', model.get('source_root', str(root)),
                    '--checkpoint', model['checkpoint'], '--expected-update', str(model['update']), '--policy', name,
                    '--bank', bank_path, '--split', spec['split'], '--worlds', spec['worlds'],
                    '--output', str(job), '--repeats', str(resolved[dataset]['repeats']),
                    '--clip', '2.0', '--min-std'] + ['0.1']*11)
                # Worker and audit are separate processes. Only audit sees the temporary disk edit.
                try:
                    validator.write_bytes(after)
                    # The worker sets use_base_policy = (policy == 'bc'); the audit must be told
                    # the same expectation, otherwise it defaults to False and rejects the base arm.
                    command(dataset+'_'+name+'_audit', [sys.executable,
                        str(root/'scripts/audit_routea_paper_eval.py'), '--job', str(job),
                        '--bank', bank_path, '--state-sha256', model['state_sha256'],
                        '--output', str(out/'packages'/dataset/(name+'.json'))]
                        + (['--base'] if name == 'bc' else []))
                finally:
                    assert validator.read_bytes() in (before, after), 'concurrent validator edit; do not overwrite'
                    validator.write_bytes(before)
                    restorations.append(dict(dataset=dataset, policy=name, restored=validator.read_bytes() == before,
                                             restored_sha256=digest(validator)))
                package = read(out/'packages'/dataset/(name+'.json'))
                assert package['conditions'] == spec['episodes'] and package['bank_hash'] == spec['content_hash']
                assert digest(bank_path) == spec['file_sha256']
                completed.append(dict(dataset=dataset, policy=name, episodes=spec['episodes']))
                print(plan.get('label', 'G16')+'_E2_JOB_PASS', dataset, name, spec['episodes'], flush=True)
        assert sum(x['episodes'] for x in completed) == total
    finally:
        source_ok = (closure_ok()
                     and all(digest(root/p) == h for p, h in snapshots.items()))
        write(out/'restorations.json', dict(jobs=restorations, source_unchanged=source_ok,
              original_validator_restored=validator.read_bytes() == before))
        assert source_ok and validator.read_bytes() == before
    write(out/'completion.json', dict(episodes=total, jobs=completed, scope='engineering execution only'))
    print(plan.get('label', 'G16')+'_E2_COMPLETE', total, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['preflight', 'train', 'evaluate'])
    p.add_argument('--plan', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--arm', choices=['P9', 'P11'])
    args = p.parse_args()
    plan = read(args.plan)
    root = Path(plan['repo']).resolve()
    sys.path.insert(0, str(root))
    if args.phase == 'train' and args.arm is None:
        p.error('train requires --arm')
    globals()[args.phase](args, plan, root)


if __name__ == '__main__':
    main()
