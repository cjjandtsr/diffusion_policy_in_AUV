"""P6 evaluation-only E3 and x0 collision guidance; never trains or edits native sources."""
import argparse
from collections import Counter, defaultdict
import csv
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.routea_paired_support.core import canonical_hash, file_hash, read_json, write_json

P6_SHA = 'a91c9f428c5749b8ca507168ea83f17eeb88cce4b4a03c6aeaea197b214bdd43'
BLOCKED_SHA = '32cbc908b61c899a9ad2d55978857de3eb6dfa752182411be7dcf7b12bd2ab3d'
GRID = [(eta, con) for con in (0, 1) for eta in (.05, .1, .2)]
OVERLAYS = ['diffusion_policy_3d.env.reward_v2_support.environment',
            'diffusion_policy_3d.env.kinematic_uav_env',
            'scripts.evaluate_dppo_uav_formal_endpoint',
            'scripts.routea_paired_support.core', 'scripts.routea_paired_support.runtime',
            'scripts.routea_paired_support.episodes']


def csv_write(path, rows):
    with Path(path).open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_identity():
    names = [n.replace('.', '/') + '.py' for n in OVERLAYS]
    names += ['scripts/run_routea_e3g_probe.py', 'scripts/routea_denoising_guidance.py']
    names += ['scripts/run_routea_xk_evaluation.py', 'scripts/f7_analysis.py',
              'third_party/dppo_official_cc7234a/uav_idp3/e5_runtime.py',
              'third_party/dppo_official_cc7234a/model/diffusion/diffusion_vpg.py',
              'diffusion_policy_3d/env/traj_utils.py',
              'diffusion_policy_3d/env/reward_v2_support/first_event.py',
              'diffusion_policy_3d/rl/dppo_uav_rebuild/engine/contracts.py']
    return {n: file_hash(ROOT / n) for n in names}


def load_plan(output):
    path = Path(output).resolve()
    plan = read_json(path / 'request.json')
    assert plan['evaluation_sources'] == source_identity(), 'evaluation code changed after prepare'
    for name, item in plan['banks'].items():
        assert file_hash(item['path']) == item['sha256'], name + ' bank changed'
    assert file_hash(plan['blocked']) == BLOCKED_SHA
    return path, plan


def prepare(args):
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    receipt = read_json(Path(args.diag_run) / 'weights_verified.json')['specs']['P6']
    assert receipt['state_sha256'] == P6_SHA and receipt['update'] == 256
    banks = {}
    for split, path in [('train', Path(args.diag_run) / 'bank.json'),
                        ('dev', Path(args.dev_bank)), ('val', Path(args.val_bank))]:
        bank = read_json(path)
        assert canonical_hash({k: v for k, v in bank.items() if k != 'content_hash'}) == bank['content_hash']
        assert len(bank['episodes']) == {'train': 425, 'dev': 420, 'val': 80}[split]
        assert len({(e['task_id'], e['repeat']) for e in bank['episodes']}) == len(bank['episodes'])
        banks[split] = dict(path=str(path.resolve()), sha256=file_hash(path))
    assert file_hash(args.blocked) == BLOCKED_SHA
    write_json(out / 'request.json', dict(schema='routea_e3g_v1', p6=receipt, banks=banks,
        blocked=str(Path(args.blocked).resolve()), evaluation_sources=source_identity(),
        native_source=str(Path(args.source_root).resolve()), checkpoint=str(Path(args.checkpoint).resolve()),
        repo=str(Path(args.repo).resolve()), grid=GRID, baseline_dev=str(Path(args.baseline_dev).resolve()),
        baseline_val=str(Path(args.baseline_val).resolve()), full_run_requires_user_schedule=True,
        criteria=dict(e3_collision_delta=-15, e3_dev_arrival_delta=0, e3_val_arrival_delta=-4,
                      guided_blocked_arrivals=49, guided_blocked_solved=10,
                      guided_passable_arrivals=237, guided_collision=123)))
    print('E3G_PREPARED', {k: len(read_json(v['path'])['episodes']) for k, v in banks.items()})


def overlay_modules():
    """Restore identity first; overlay only evaluation environment/recording afterwards."""
    identities = {}
    for name in OVERLAYS:
        path = ROOT / (name.replace('.', '/') + '.py')
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        parent, leaf = name.rsplit('.', 1)
        setattr(sys.modules[parent], leaf, module)
        identities[name] = dict(file=str(path), sha256=file_hash(path))
    return identities


def runtime(args, plan):
    from scripts.run_routea_xk_evaluation import native_system, verify_source
    verify_source(plan['native_source'], dict(files=plan['p6']['sources']))
    assert file_hash(Path(plan['checkpoint']) / 'state.pt') == P6_SHA
    system, config, header = native_system(SimpleNamespace(source_root=plan['native_source'],
        repo=plan['repo'], checkpoint=plan['checkpoint'], expected_update=256,
        policy='P6', evaluation_reproduction=False))
    assert header['runtime_identity_hash'] == plan['p6']['runtime_identity_hash']
    overlays = overlay_modules()
    return system, config, header, overlays


def bank_maps(plan, split):
    item = plan['banks'][split]
    bank = read_json(item['path'])
    for world in bank['worlds'].values():
        if not Path(world['obstacle_path']).is_file():
            world['obstacle_path'] = str(Path(item['path']).parent / world['obstacle_copy'])
        assert file_hash(world['obstacle_path']) == world['obstacle_sha256']
    return bank, {t['task_id']: t for t in bank['tasks']}


def job_name(arm, split, eta=None, con=None):
    return ('grid_c%d_eta%s' % (con, str(eta))) if arm == 'grid' else arm + '_' + split


def worker(args):
    out, plan = load_plan(args.output)
    if not args.limit:
        assert args.full_run_approved, 'full runs require user scheduling after the 20-episode timing report'
        assert read_json(out / 'regression_pass.json')['passed']
    system, config, header, overlays = runtime(args, plan)
    from scripts.evaluate_dppo_uav_formal_endpoint import _E5Policy
    from scripts.routea_paired_support.runtime import make_adapter
    from scripts.routea_paired_support.episodes import record_episode
    from scripts.run_routea_xk_evaluation import load_override
    from scripts.routea_denoising_guidance import Guidance, ProbePolicy, install
    bank, tasks = bank_maps(plan, args.split)
    if bank.get('schema') == 'routea_fixed_split_v1' and bank.get('repeats') == 1:
        from scripts import evaluate_dppo_uav_formal_endpoint as endpoint
        from scripts.routea_paired_support import episodes
        episodes.action_seed = lambda task_id, repeat, decision: endpoint._action_seed(task_id, decision)
    eta, con = args.eta, args.consistency
    if args.arm in ('G0', 'G1'):
        selected = read_json(out / 'selection.json')
        assert selected['request_sha256'] == file_hash(out / 'request.json')
        eta, con = selected[args.arm]['eta'], selected[args.arm]['consistency']
    if args.arm == 'grid':
        assert args.split == 'train' and (eta, con) in GRID
    assert args.arm != 'E3' or args.split in ('dev', 'val')
    name = job_name(args.arm, args.split, eta, con)
    job = out / ('smoke' if args.limit else 'runs') / name
    job.mkdir(parents=True, exist_ok=False)
    runtime_file = Path(inspect.getfile(type(system.official_runtime))).resolve()
    identity = dict(type=type(system.official_runtime).__qualname__,
        module=type(system.official_runtime).__module__, file=str(runtime_file),
        source_sha256=file_hash(runtime_file), actual_use_base_policy=False,
        state_sha256=file_hash(Path(plan['checkpoint']) / 'state.pt'),
        native_runtime_identity=header['runtime_identity_hash'], overlays=overlays,
        bank_sha256=plan['banks'][args.split]['sha256'], arm=args.arm, split=args.split,
        eta=eta, consistency=con, smoke=bool(args.limit), evaluation_sources=source_identity())
    base = _E5Policy(system, use_base_policy=False, identity_hash=header['runtime_identity_hash'])
    scope = load_override(ROOT)(system.official_runtime, denoised_clip_value=2.,
                                min_std_vector=[.1] * 11, sampling_mode='deterministic')
    with scope as protocol:
        identity['sampling_protocol'] = protocol
        forward_file = Path(inspect.getfile(system.official_runtime.forward)).resolve()
        identity['bound_forward'] = dict(file=str(forward_file), sha256=file_hash(forward_file))
        forward = system.official_runtime.forward
        if getattr(forward, '__closure__', None):
            closure = dict(zip(forward.__code__.co_freevars, forward.__closure__))
            if 'forward' in closure:
                inner_file = Path(inspect.getfile(closure['forward'].cell_contents)).resolve()
                identity['wrapped_native_forward'] = dict(file=str(inner_file), sha256=file_hash(inner_file))
        guide = None
        if args.arm in ('grid', 'G0', 'G1'):
            guide = Guidance(system.bc_runtime.policy.normalizer['action'].unnormalize, eta, con)
            identity['guidance'] = install(system.official_runtime, guide)
        write_json(job / 'identity.json', identity)
        print('E3G_RUNTIME', json.dumps(identity, ensure_ascii=False), flush=True)
        for i, entry in enumerate(bank['episodes'][:args.limit or None]):
            task = tasks[entry['task_id']]
            adapter = make_adapter(system, config, bank['worlds'][task['world_id']], task['role'],
                                   evaluation_protocol='E3' if args.arm == 'E3' else 'E2')
            assert adapter.environment.exec_noise_scale == 0 and not adapter.environment.disturb_enable
            policy = base if args.arm == 'E2' else ProbePolicy(base, adapter.environment, guide)
            try:
                record_episode(job, 'P6' if args.arm == 'E2' else args.arm, policy, adapter, task, entry,
                               bank['worlds'][task['world_id']], bank['content_hash'])
            finally:
                adapter.environment.close()
            print('E3G_EPISODE', name, i + 1, flush=True)
    rows = collect(job, bank)
    csv_write(job / 'conditions.csv', rows)
    write_json(job / 'summary.json', dict(conditions=len(rows), reached=sum(r['outcome'] == 'reached' for r in rows),
                                         outcomes=dict(Counter(r['outcome'] for r in rows))))


def collect(job, bank):
    from scripts.routea_paired_support.episodes import load_commit
    from scripts.f7_analysis import outcome_1m
    rows = []
    commits = {p.stem: p for p in (job / 'commits').glob('*.json')}
    expected = {e['key'] for e in bank['episodes']}
    tasks = {t['task_id']: t for t in bank['tasks']}
    identity = read_json(job / 'identity.json') if (job / 'identity.json').is_file() else None
    expected_policy = ('P6' if identity is None or identity['arm'] == 'E2' else identity['arm'])
    expected_runtime = (identity['native_runtime_identity'] if identity else
                        'd987235a0cbb5ad395bbd6e7979bdb8d7e0cff1303b0611cb462c7d554bbd61f')
    assert set(commits) <= expected
    for e in bank['episodes']:
        if e['key'] not in commits:
            continue
        h, trace, end = load_commit(job, commits[e['key']])
        assert h['policy'] == expected_policy and h['policy_identity_hash'] == expected_runtime
        assert h.get('evaluation_protocol', 'E2') == ('E3' if expected_policy == 'E3' else 'E2')
        assert h['task_id'] == e['task_id'] and h['repeat'] == e['repeat']
        assert h['initial_hash'] == e['initial_hash'] and not h['use_base_policy']
        assert h['bank_hash'] == bank['content_hash'] and h['task'] == tasks[e['task_id']]['payload']
        assert h['reset_seed'] == e['reset_seed']
        world = bank['worlds'][tasks[e['task_id']]['world_id']]
        assert h['geometry'] == world['geometry'] and h['obstacle_sha256'] == world['obstacle_sha256']
        assert h['spawn_offset'] == world['spawn_offset'] and h['world_id'] == world['world_id']
        from scripts.evaluate_dppo_uav_formal_endpoint import _action_seed
        for t in trace:
            seed = (_action_seed(e['task_id'], t['decision'])
                    if bank.get('schema') == 'routea_fixed_split_v1' and bank.get('repeats') == 1
                    else int(canonical_hash(['policy-sample-v1', e['task_id'], e['repeat'], t['decision']])[:8], 16))
            assert t['action_seed'] == seed
        goal = np.asarray(h['task']['goal_pos'], np.float32)
        outcome = outcome_1m(trace, goal, end['done_reason'])
        line = goal - np.asarray(h['task']['start_pos'], np.float32)
        # Signed horizontal departure; exact zero positions are skipped.
        signs = []
        for t in trace:
            d = np.asarray(t['post_state']['position']) - h['task']['start_pos']
            value = line[0] * d[1] - line[1] * d[0]
            if value != 0:
                signs.append(np.sign(value))
        metrics = [t['probe_metrics'] for t in trace if 'probe_metrics' in t]
        receipt = read_json(commits[e['key']])
        rows.append(dict(task_id=e['task_id'], repeat=e['repeat'], outcome=outcome,
            raw_outcome=end['done_reason'], decisions=end['decisions'],
            min_goal_distance_m=end['trajectory_metrics']['min_goal_distance_m'],
            lateral_sign_changes=sum(a != b for a, b in zip(signs, signs[1:])),
            inference_seconds=(sum(m['inference_seconds'] for m in metrics) / len(metrics)) if metrics else '',
            guided_decision_fraction=(sum(m['guided'] for m in metrics) / len(metrics)) if metrics else '',
            trace=str(job / receipt['trace']), trace_sha256=receipt['sha256']))
    return rows


def select(args):
    out, plan = load_plan(args.output)
    bank = read_json(plan['banks']['train']['path'])
    settings = []
    for eta, con in GRID:
        job = out / 'runs' / job_name('grid', 'train', eta, con)
        rows = collect(job, bank)
        assert len(rows) == 425
        identity = read_json(job / 'identity.json')
        assert identity['eta'] == eta and identity['consistency'] == con and not identity['smoke']
        assert identity['state_sha256'] == P6_SHA and identity['evaluation_sources'] == plan['evaluation_sources']
        settings.append(dict(eta=eta, consistency=con, reached=sum(r['outcome'] == 'reached' for r in rows),
                             identity_sha256=file_hash(job / 'identity.json')))
    selected = dict(request_sha256=file_hash(out / 'request.json'), grid=settings)
    for con in (0, 1):
        selected['G' + str(con)] = min((r for r in settings if r['consistency'] == con),
                                      key=lambda r: (-r['reached'], r['eta']))
    write_json(out / 'selection.json', selected)


def e3_verdict(collision_gain, dev_gain, val_gain):
    if collision_gain > -15:
        return '执行周期在本探针下不是主要杠杆'
    return '执行周期是有效杠杆' if dev_gain >= 0 and val_gain >= -4 else '不确定'


def guided_verdict(blocked_arrivals, blocked_solved, passable_arrivals, collisions):
    if blocked_arrivals < 49:
        return '本探针下无效'
    return ('引导有效' if blocked_solved >= 10 and passable_arrivals >= 237 and collisions <= 123
            else '不确定')


def regression(args):
    out, plan = load_plan(args.output)
    # Each native load must run in a fresh process: module identities are frozen.
    import subprocess
    for split in ('dev', 'val'):
        command = [sys.executable, str(Path(__file__).resolve()), 'worker', '--output', str(out),
                   '--arm', 'E2', '--split', split, '--limit', '1']
        execution = out / 'execution'
        execution.mkdir(exist_ok=True)
        write_json(execution / ('regression_' + split + '.command.json'), dict(argv=command))
        with (execution / ('regression_' + split + '.log')).open('x', encoding='utf-8') as stream:
            run = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        (execution / ('regression_' + split + '.rc')).write_text(str(run.returncode) + '\n', encoding='utf-8')
        assert run.returncode == 0
        entry = read_json(plan['banks'][split]['path'])['episodes'][0]
        original = read_json(Path(plan['baseline_' + split]) / 'commits' / (entry['key'] + '.json'))
        job = out / 'smoke' / job_name('E2', split)
        new = read_json(job / 'commits' / (entry['key'] + '.json'))
        assert original['sha256'] == new['sha256'], 'E2 native JSONL byte regression failed'
    write_json(out / 'regression_pass.json', dict(passed=True, conditions=2, comparison='whole JSONL SHA256'))


def summarize(args, verify=False):
    out, plan = load_plan(args.output)
    blocked = read_json(plan['blocked'])
    blocked = set(blocked) if isinstance(blocked, list) else set(blocked)
    assert len(blocked) == 33
    all_rows = []
    result = {}
    for split in ('dev', 'val'):
        bank = read_json(plan['banks'][split]['path'])
        n = 420 if split == 'dev' else 80
        for arm in ('E2', 'E3', 'G0', 'G1'):
            job = Path(plan['baseline_' + split]) if arm == 'E2' else out / 'runs' / job_name(arm, split)
            rows = collect(job, bank)
            assert len(rows) == n, (arm, split, len(rows))
            for r in rows:
                all_rows.append(dict(arm=arm, split=split, **r))
            counts = Counter(r['outcome'] for r in rows)
            summary = dict(conditions=n, reached=counts['reached'], collision=counts['collision'])
            if split == 'dev':
                task_success = Counter(r['task_id'] for r in rows if r['task_id'] in blocked and r['outcome'] == 'reached')
                assert sum(r['task_id'] in blocked for r in rows) == 165
                summary.update(blocked_arrivals=sum(task_success.values()),
                               blocked_solved=sum(v >= 3 for v in task_success.values()),
                               passable_arrivals=sum(r['outcome'] == 'reached' and r['task_id'] not in blocked for r in rows))
            result[arm + '_' + split] = summary
    base = result['E2_dev']
    assert (base['blocked_arrivals'], base['blocked_solved'], base['passable_arrivals'], base['collision']) == (34, 7, 247, 138)
    result['E3_verdict'] = e3_verdict(result['E3_dev']['collision'] - base['collision'],
        result['E3_dev']['reached'] - base['reached'], result['E3_val']['reached'] - result['E2_val']['reached'])
    for arm in ('G0', 'G1'):
        s = result[arm + '_dev']
        result[arm + '_verdict'] = guided_verdict(s['blocked_arrivals'], s['blocked_solved'],
                                                s['passable_arrivals'], s['collision'])
    if verify:
        assert read_json(out / 'summary.json') == result, 'summary differs from committed traces'
        with (out / 'conditions.csv').open(encoding='utf-8-sig', newline='') as stream:
            actual = list(csv.DictReader(stream))
        expected = [{k: str(v) for k, v in row.items()} for row in all_rows]
        assert actual == expected, 'condition export differs from committed traces'
    else:
        csv_write(out / 'conditions.csv', all_rows)
        write_json(out / 'summary.json', result)


def audit(args):
    out, plan = load_plan(args.output)
    assert read_json(out / 'regression_pass.json')['passed']
    selected = read_json(out / 'selection.json')
    assert selected['request_sha256'] == file_hash(out / 'request.json')
    total = 0
    for eta, con in GRID:
        job = out / 'runs' / job_name('grid', 'train', eta, con)
        bank = read_json(plan['banks']['train']['path'])
        assert len(collect(job, bank)) == 425
        assert selected['grid'][GRID.index((eta, con))]['identity_sha256'] == file_hash(job / 'identity.json')
        total += 425
    for arm in ('E3', 'G0', 'G1'):
        for split, expected in [('dev', 420), ('val', 80)]:
            job = out / 'runs' / job_name(arm, split)
            bank = read_json(plan['banks'][split]['path'])
            assert len(collect(job, bank)) == expected
            identity = read_json(job / 'identity.json')
            assert identity['state_sha256'] == P6_SHA and not identity['actual_use_base_policy']
            assert identity['evaluation_sources'] == plan['evaluation_sources']
            assert not identity['smoke'] and identity['bank_sha256'] == plan['banks'][split]['sha256']
            if arm in ('G0', 'G1'):
                assert identity['eta'] == selected[arm]['eta'] and identity['consistency'] == selected[arm]['consistency']
            total += expected
    assert total == 4050
    summarize(args, verify=True)
    rc = list((out / 'execution').glob('*.rc'))
    assert len(rc) >= 14 and all(p.read_text().strip() == '0' for p in rc)
    write_json(out / 'audit.json', dict(passed=True, new_conditions=total, rc_files=len(rc),
                                      selection_sha256=file_hash(out / 'selection.json')))
    files = sorted(p for p in out.rglob('*') if p.is_file())
    with (out / 'MANIFEST.sha256').open('x', encoding='utf-8', newline='\n') as stream:
        for path in files:
            stream.write(file_hash(path) + '  ' + path.relative_to(out).as_posix() + '\n')


def run(args):
    """Execution/logging convenience only; all calculations use the same worker."""
    import subprocess
    import time
    out, plan = load_plan(args.output)
    assert read_json(out / 'regression_pass.json')['passed']
    jobs = []
    if args.stage in ('smoke', 'grid'):
        jobs = [('grid', 'train', eta, con) for eta, con in GRID]
        if args.stage == 'smoke':
            jobs.insert(0, ('E3', 'dev', None, None))
    elif args.stage == 'guided-smoke':
        jobs = [(arm, 'dev', None, None) for arm in ('G0', 'G1')]
    elif args.stage == 'formal':
        jobs = [(arm, split, None, None) for arm in ('E3', 'G0', 'G1') for split in ('dev', 'val')]
    smoke = args.stage in ('smoke', 'guided-smoke')
    if not smoke:
        assert args.full_run_approved, 'user must approve full evaluation after timing'
    if args.stage == 'grid':
        for eta, con in GRID:
            assert read_json(out / 'smoke' / job_name('grid', 'train', eta, con) / 'summary.json')['conditions'] == 20
    if args.stage == 'formal':
        assert read_json(out / 'smoke/E3_dev/summary.json')['conditions'] == 20
        for arm in ('G0', 'G1'):
            assert read_json(out / 'smoke' / (arm + '_dev') / 'summary.json')['conditions'] == 20
    durations = []
    for arm, split, eta, con in jobs:
        command = [sys.executable, str(Path(__file__).resolve()), 'worker', '--output', str(out),
                   '--arm', arm, '--split', split]
        if eta is not None:
            command += ['--eta', str(eta), '--consistency', str(con)]
        command += ['--limit', '20'] if smoke else ['--full-run-approved']
        execution = out / 'execution'
        execution.mkdir(exist_ok=True)
        name = args.stage + '_' + job_name(arm, split, eta, con)
        write_json(execution / (name + '.command.json'), dict(argv=command))
        start = time.perf_counter()
        with (execution / (name + '.log')).open('x', encoding='utf-8') as stream:
            process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        elapsed = time.perf_counter() - start
        with (execution / (name + '.rc')).open('x', encoding='utf-8') as stream:
            stream.write(str(process.returncode) + '\n')
        durations.append(dict(job=name, seconds=elapsed, rc=process.returncode))
        assert process.returncode == 0, 'preserve failed run; see ' + name
    write_json(out / (args.stage + '_timing.json'), dict(jobs=durations, smoke=smoke,
        note='Process wall time includes strict model restore; smoke results never enter formal denominators.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    for name in ('output', 'diag-run', 'dev-bank', 'val-bank', 'blocked', 'source-root',
                 'checkpoint', 'repo', 'baseline-dev', 'baseline-val'):
        p.add_argument('--' + name, required=True)
    p = sub.add_parser('worker')
    p.add_argument('--output', required=True)
    p.add_argument('--arm', choices=['E2', 'E3', 'grid', 'G0', 'G1'], required=True)
    p.add_argument('--split', choices=['train', 'dev', 'val'], required=True)
    p.add_argument('--eta', type=float)
    p.add_argument('--consistency', type=int)
    p.add_argument('--limit', type=int, choices=[1, 20])
    p.add_argument('--full-run-approved', action='store_true')
    p = sub.add_parser('run')
    p.add_argument('--output', required=True)
    p.add_argument('--stage', choices=['smoke', 'grid', 'guided-smoke', 'formal'], required=True)
    p.add_argument('--full-run-approved', action='store_true')
    for command in ('select', 'summarize', 'regression', 'audit'):
        sub.add_parser(command).add_argument('--output', required=True)
    args = parser.parse_args()
    with torch.jit.optimized_execution(False):
        globals()[args.command](args)


if __name__ == '__main__':
    main()
