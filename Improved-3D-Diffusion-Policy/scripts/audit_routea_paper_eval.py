"""Engineering verification and compact episode package for one common-worker job."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.routea_paired_support.core import canonical_hash, file_hash, read_json
from scripts.routea_paired_support.episodes import load_commit


def nearest_goal_time(transitions, goal):
    """Earliest closest point on the actual prefix, using physical segment durations."""
    goal = np.asarray(goal, dtype=np.float32).astype(float)
    best, best_time, elapsed = float('inf'), None, 0.
    for row in transitions:
        path = np.asarray(row['executed_path'], dtype=float)
        duration = row['executed_time_s']
        for index, (start, end) in enumerate(zip(path[:-1], path[1:])):
            delta = end - start
            norm = float(delta @ delta)
            fraction = 0. if norm == 0. else float(np.clip((goal - start) @ delta / norm, 0., 1.))
            distance = float(np.linalg.norm(start + fraction * delta - goal))
            segment_time = .3 if index < len(path) - 2 else duration - .3 * index
            if distance < best:
                best, best_time = distance, elapsed + .3 * index + fraction * segment_time
        elapsed += duration
    return best, best_time


def task_category(task, world):
    from scripts.routea_paired_support.task_geometry_metrics import corridor_density
    from scripts.analyze_routea_f5 import labels
    geometry_task = dict(start=task['payload']['start_pos'], goal=task['payload']['goal_pos'],
                         geometry=world['geometry'])
    rho = corridor_density(geometry_task, 1.)['rho']
    category = labels({'rho_1.0': rho, 'rho_2.0': 0., 'b_band_m': None,
                       'kappa': None, 'band_rho': 0., 'blocked': False})['w1']
    return dict(category=category, rho_1_0=rho)


def audit(job, bank_path, expected_state, expected_base, control=None):
    job, bank_path = Path(job), Path(bank_path)
    bank = read_json(bank_path)
    if canonical_hash({k: v for k, v in bank.items() if k != 'content_hash'}) != bank['content_hash']:
        raise ValueError('bank content hash mismatch')
    provenance = read_json(job / 'policy_provenance.json')
    protocol = provenance['evaluation_sampling_protocol']
    if (protocol['clip_protocol']['override'] != 2.
            or protocol['sampling_mode'] != 'deterministic'
            or provenance['actual_use_base_policy'] is not expected_base
            or provenance['checkpoint_state_sha256'] != expected_state
            or provenance['bank_hash'] != bank['content_hash']):
        raise ValueError('policy identity/branch/E2 mismatch')
    for field in ('min_sampling_denoising_std', 'min_logprob_denoising_std'):
        values = protocol[field]
        if len(values) != 11 or any(abs(v - .1) > 1e-6 for v in values):
            raise ValueError('wrong E2 noise floor: ' + field)
    entries = {e['key']: e for e in bank['episodes']}
    if len(entries) != len(bank['episodes']):
        raise ValueError('duplicate bank episode')
    commits = sorted((job / 'commits').glob('*.json'))
    if {p.stem for p in commits} != set(entries):
        raise ValueError('commit set differs from frozen bank')
    summary = read_json(job / 'summary.json')
    reported = {(r['task_id'], r['repeat']): r for r in summary['records']}
    if len(reported) != len(entries) or len(summary['records']) != len(entries):
        raise ValueError('summary duplicates or count mismatch')
    tasks = {t['task_id']: t for t in bank['tasks']}
    records, categories = [], {}
    reference = None
    if control:
        old = read_json(Path(control) / 'summary.json')['records']
        reference = {(r['task_id'], r['repeat']): r['done_reason'] for r in old}
        if len(reference) != len(old) or set(reference) != set(reported):
            raise ValueError('control condition set differs')
    for commit in commits:
        header, transitions, record = load_commit(job, commit)
        entry = entries[commit.stem]
        key = (record['task_id'], record['repeat'])
        task = tasks[entry['task_id']]
        world = bank['worlds'][task['world_id']]
        if (reported[key] != record or header['initial_hash'] != entry['initial_hash']
                or header['bank_hash'] != bank['content_hash']
                or header['use_base_policy'] is not expected_base
                or header['task_id'] != entry['task_id'] or header['repeat'] != entry['repeat']
                or header.get('terminate_at_space_limits', True)
                or header['task'] != task['payload'] or header['world_id'] != task['world_id']
                or header['role'] != task['role'] or header['geometry'] != world['geometry']
                or header['spawn_offset'] != world['spawn_offset']
                or header['obstacle_sha256'] != world['obstacle_sha256']
                or header['reset_seed'] != entry['reset_seed']
                or header['policy'] != provenance['policy']
                or (not expected_base and header['policy_identity_hash'] != provenance['training_runtime_identity'])):
            raise ValueError('summary/pairing/branch/space contract mismatch: ' + commit.stem)
        if reference is not None and record['done_reason'] != reference[key]:
            raise ValueError('deterministic control outcome differs: ' + commit.stem)
        raw = read_json(commit)
        source = dict(commit=str(commit.resolve()), commit_sha256=file_hash(commit),
                      trace=raw['trace'], trace_sha256=raw['sha256'])
        last = transitions[-1]
        if record['task_id'] not in categories:
            categories[record['task_id']] = task_category(task, world)
        distance, seconds = nearest_goal_time(transitions, task['payload']['goal_pos'])
        if abs(distance - record['trajectory_metrics']['min_goal_distance_m']) > 1e-6:
            raise ValueError('closest point recomputation differs from recorded metric')
        records.append(dict(record, world_id=header['world_id'], protocol='E2',
                            **categories[record['task_id']], nearest_goal_distance_m=distance,
                            nearest_goal_time_s=seconds,
                            event_segment=last['event_segment'], event_fraction=last['event_fraction'],
                            source=source))
    return dict(schema='routea_paper_episode_package_v1_1', status='passed',
                scope='commit/hash, paired identity, saved actions/path and physical first-event recomputation',
                bank_hash=bank['content_hash'], bank_file_sha256=file_hash(bank_path),
                provenance=provenance, provenance_sha256=file_hash(job / 'policy_provenance.json'),
                summary_sha256=file_hash(job / 'summary.json'), script_sha256=file_hash(__file__),
                conditions=len(records), outcomes=dict(Counter(r['done_reason'] for r in records)),
                control_outcomes_identical=None if reference is None else len(records), records=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True)
    parser.add_argument('--bank', required=True)
    parser.add_argument('--state-sha256', required=True)
    parser.add_argument('--base', action='store_true')
    parser.add_argument('--control')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    package = audit(args.job, args.bank, args.state_sha256, args.base, args.control)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(package, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print('PAPER_EVAL_AUDIT_PASS', package['conditions'], json.dumps(package['outcomes']))


if __name__ == '__main__':
    main()
