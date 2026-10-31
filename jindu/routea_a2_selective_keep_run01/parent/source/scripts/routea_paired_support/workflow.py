"""Evaluate compatible checkpoints using the same frozen task bank."""
from pathlib import Path
import numpy as np
from .core import read_json, write_json
from .episodes import action_seed, ambiguous_events, load_commit, record_episode
from .runtime import common_system, evaluation_sources, load_actor, make_adapter
from .reference import check_episode_identity


def save(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError('existing result has different inputs: ' + str(path))
    else:
        write_json(path, value)


def check_preflight(path, root, cat):
    report = read_json(path)
    if (report.get('status') != 'pass' or report['catalogue_hash'] != cat['content_hash']
            or report['evaluation_sources'] != evaluation_sources(Path(root))):
        raise ValueError('preflight refers to different code or weights')
    return report


def selected_entries(bank, mode):
    if mode == 'full':
        return bank['episodes']
    if mode == 'anchors':
        return [e for e in bank['episodes'] if e['repeat'] == 1]
    if mode != 'smoke':
        raise ValueError('unknown evaluation mode')
    roles = {t['role'] for t in bank['tasks']}
    chosen = {next(t['task_id'] for t in bank['tasks'] if t['role'] == r) for r in roles}
    return [e for e in bank['episodes'] if e['task_id'] in chosen and e['repeat'] == 1]


def policy_order(cat):
    return ['bc'] + sorted(n for n in cat['policies'] if n != 'bc')


def active_policies(cat, mode):
    if mode == 'anchors':
        return ['bc']
    return policy_order(cat) if mode == 'full' else [n for n in policy_order(cat) if n != 'bc']


def run_protocol(root, cat, bank, mode, reference):
    names = active_policies(cat, mode)
    return {'schema': 'routea_paired_run_v3', 'mode': mode,
            'terminate_at_space_limits': False,
            'bank_hash': bank['content_hash'], 'catalogue_hash': cat['content_hash'],
            'evaluation_sources': evaluation_sources(root), 'reference': str(Path(reference).resolve()),
            'policies': names, 'expected_episodes': len(names) * len(selected_entries(bank, mode)),
            'reward_version_for_all_evaluation': 'v2', 'jit_optimized_execution': False}


def original_bc(reference, entry):
    job = Path(reference) / 'full/bc'
    return load_commit(job, job / 'commits' / (entry['key'] + '.json'))


def check_matrix(root, cat, bank_path, out, mode, reference=None):
    bank, out = read_json(bank_path), Path(out)
    protocol = read_json(out / 'protocol.json')
    reference = reference or protocol['reference']
    if protocol != run_protocol(root, cat, bank, mode, reference):
        raise ValueError('evaluation protocol differs')
    entries = selected_entries(bank, mode)
    tasks = {t['task_id']: t for t in bank['tasks']}
    summaries, partial = [], []
    for name in active_policies(cat, mode):
        job = out / name
        if {p.stem for p in (job / 'commits').glob('*.json')} != {e['key'] for e in entries}:
            raise ValueError('incomplete/extra task matrix: ' + name)
        used = set()
        for e in entries:
            t = tasks[e['task_id']]
            baseline = (load_commit(out/'bc',out/'bc/commits'/(e['key']+'.json'))
                        if bank.get('schema')=='routea_paper_test_v1' else original_bc(reference, e))
            trace = load_commit(job, job / 'commits' / (e['key'] + '.json'))
            check_episode_identity(trace, e, t, bank['worlds'][t['world_id']], bank, cat, name, baseline)
            if ambiguous_events(trace[0], trace[1]):
                raise ValueError('ambiguous first event; return raw evidence')
            summaries.append({**trace[2], 'ambiguous_first_event': False})
            used.add(read_json(job / 'commits' / (e['key'] + '.json'))['trace'])
        partial.extend(p.relative_to(out).as_posix() for p in (job / 'traces').glob('*.jsonl')
                       if p.relative_to(job).as_posix() not in used)
    result = {'status': 'pass', 'bank_hash': bank['content_hash'], 'mode': mode,
              'episodes': len(summaries), 'environment_conditions': len(entries),
              'policy_count': len(active_policies(cat, mode)), 'invalid_completed_episodes': 0,
              'ambiguous_first_event_episodes': 0, 'preserved_partial_attempts': partial,
              'records': summaries}
    save(out / 'matrix_check.json', {k: v for k, v in result.items() if k != 'records'})
    return result


def run(root, cat, bank_path, out, mode, resume=False, reference=None):
    from scripts.evaluate_dppo_uav_formal_endpoint import _E5Policy, reset_fixed_episode
    bank, out, reference = read_json(bank_path), Path(out), Path(reference)
    if out.exists() and not resume:
        raise FileExistsError('evaluation exists; use a new output or same-input --resume')
    out.mkdir(parents=True, exist_ok=True)
    save(out / 'protocol.json', run_protocol(root, cat, bank, mode, reference))
    system, c = common_system(root, cat)
    entries = selected_entries(bank, mode)
    tasks = {t['task_id']: t for t in bank['tasks']}
    for name in active_policies(cat, mode):
        d = cat['policies'][name]
        if name != 'bc':
            load_actor(root, system, d)
        identity = cat['common']['bc_identity_hash'] if name == 'bc' else d['runtime_identity_hash']
        policy = _E5Policy(system, use_base_policy=d['use_base_policy'], identity_hash=identity)
        job = out / name
        (job / 'commits').mkdir(parents=True, exist_ok=True)
        adapters = {}
        if name != 'bc':
            e, t = entries[0], tasks[entries[0]['task_id']]
            adapter = make_adapter(system, c, bank['worlds'][t['world_id']], t['role'])
            obs, ctx = reset_fixed_episode(adapter.environment, t['payload'], seed=e['reset_seed'],
                                           reset_callable=adapter.reset)
            base = _E5Policy(system, use_base_policy=True, identity_hash=cat['common']['bc_identity_hash'])
            sampled = base.predict(obs, ctx, action_seed=action_seed(e['task_id'], e['repeat'], 0))
            baseline = (load_commit(out/'bc',out/'bc/commits'/(e['key']+'.json'))
                        if bank.get('schema')=='routea_paper_test_v1' else original_bc(reference, e))
            expected = baseline[1][0]['full_action_16x11']
            if not np.array_equal(sampled.full[0].detach().cpu().numpy(), np.asarray(expected, np.float32)):
                raise ValueError('BC branch changed after loading ' + name)
            save(job / 'bc_disabled_branch_check.json', {'status': 'pass', 'update': d['update'],
                 'state_sha256': d['state_sha256'], 'actual_use_base_policy': True})
            adapters[t['world_id']] = adapter
        for index, e in enumerate(entries, 1):
            if (job / 'commits' / (e['key'] + '.json')).exists():
                continue
            t = tasks[e['task_id']]
            world = bank['worlds'][t['world_id']]
            if t['world_id'] not in adapters:
                adapters[t['world_id']] = make_adapter(system, c, world, t['role'])
            s = record_episode(job, name, policy, adapters[t['world_id']], t, e, world, bank['content_hash'])
            print('RAW_EPISODE', mode, name, index, len(entries), s['done_reason'], flush=True)
    return check_matrix(root, cat, bank_path, out, mode, reference)


def comparison_records(root, cat, bank, out, reference):
    """Compare only newly collected policies under the corrected event rule."""
    records, index, jobs, models = [], [], {}, {}
    for name, d in cat['policies'].items():
        jobs[name] = Path(out) / 'full' / name
        models[name] = d
    for e in bank['episodes']:
        row = {'episode_key': e['key'], 'task_id': e['task_id'], 'repeat': e['repeat'], 'models': {}}
        for name, job in jobs.items():
            commit = read_json(job / 'commits' / (e['key'] + '.json'))
            s = commit['summary']
            if s['bank_hash'] != bank['content_hash'] or s['initial_hash'] != e['initial_hash']:
                raise ValueError('comparison records refer to different tasks')
            records.append({**s, 'ambiguous_first_event': False})
            d = models[name]
            row['models'][name] = {'source_root': str(job.resolve()),
                                  'commit': 'commits/' + e['key'] + '.json',
                                  'trace_sha256': commit['sha256'], 'update': d.get('update'),
                                  'state_sha256': d.get('state_sha256', cat['common']['bc_file_sha256'])}
        index.append(row)
    return records, index, jobs, models


def finish(root, cat, bank, out, reference, plots=False):
    from .analysis import tables, plot_tasks
    out = Path(out)
    records, index, jobs, models = comparison_records(root, cat, bank, out, reference)
    save(out / 'paired_index.json', {'bank_hash': bank['content_hash'],
         'terminate_at_space_limits': False,
         'conditions': len(bank['episodes']), 'models': models, 'records': index})
    statistics = tables(records)
    if bank.get('schema')=='routea_paper_test_v1':
        from .paper import paper_statistics
        statistics = paper_statistics(records,bank,statistics)
    save(out / 'comparison_statistics.json', statistics)
    if plots:
        plot_tasks(bank, jobs, out / 'executed_trajectory_figures')
    return {'status': 'complete', 'bank_hash': bank['content_hash'], 'weights_included': False,
            'terminate_at_space_limits': False,
            'anchors': 0,
            'smoke': (0 if bank.get('schema')=='routea_paper_test_v1' else
                      len(selected_entries(bank, 'smoke')) * len(active_policies(cat, 'smoke'))),
            'full': len(bank['episodes']) * len(active_policies(cat, 'full')),
            'reference_full': len(records) - len(bank['episodes']) * len(active_policies(cat, 'full')),
            'comparison_full': len(records)}


def package_result(out, attempt, completion):
    """One return inventory. Logs are closed before this function is called."""
    import hashlib
    import json
    from .core import file_hash, _publish_generated
    out = Path(out)
    marker = out / ('run_complete_' + attempt + '.json')
    manifest = out / ('pack_sha256_' + attempt + '.txt')
    files = [p for p in sorted(out.rglob('*')) if p.is_file() and p.name != '.controller.lock']
    if any(p.suffix in ('.pt', '.pth', '.ckpt', '.safetensors') for p in files):
        raise ValueError('weights do not belong in experiment results')
    raw = (json.dumps(completion, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)+'\n').encode()
    hashes = {p.relative_to(out).as_posix(): file_hash(p) for p in files}
    hashes[marker.name] = hashlib.sha256(raw).hexdigest()
    content = ''.join(h+'  '+name+'\n' for name, h in sorted(hashes.items())).encode()
    _publish_generated(manifest, lambda stream: stream.write(content))
    if any(file_hash(p) != hashes[p.relative_to(out).as_posix()] for p in files):
        raise ValueError('results changed during return inventory creation')
    write_json(marker, completion)
    print('RETURN_COMPLETE', len(hashes), str(manifest), flush=True)
