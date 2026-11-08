"""Check the C4 z_range correction without resampling or changing task selection."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def load_bank(path):
    bank = json.loads(path.read_text(encoding='utf-8'))
    assert canonical({k: v for k, v in bank.items() if k != 'content_hash'}) == bank['content_hash']
    assert len(bank['worlds']) == 20 and len(bank['tasks']) == 60 and len(bank['episodes']) == 300
    for task in bank['tasks']:
        assert canonical(task['payload']) == task['task_id']
    tasks = {t['task_id']: t for t in bank['tasks']}
    assert len(tasks) == 60
    assert len({e['key'] for e in bank['episodes']}) == 300
    assert {(e['task_id'], e['repeat']) for e in bank['episodes']} == {
        (tid, r) for tid in tasks for r in range(1, 6)}
    for e in bank['episodes']:
        assert e['key'] == canonical([e['task_id'], e['repeat']])[:24]
        seed = (int(e['task_id'][:8], 16) if e['repeat'] == 1 else
                int(canonical(['xk-fixed-reset-v1', e['task_id'], e['repeat']])[:8], 16))
        assert e['reset_seed'] == seed
        raw = path.parent / e['initial_npz']
        assert hashlib.sha256(raw.read_bytes()).hexdigest() == e['initial_npz_sha256']
        h = hashlib.sha256()
        with np.load(raw, allow_pickle=False) as obs:
            assert set(obs.files) == {'agent_pos', 'point_cloud'}
            for key in sorted(obs.files):
                a = np.ascontiguousarray(obs[key])
                assert a.dtype == np.float32 and np.isfinite(a).all()
                assert a.shape == ({'agent_pos': (8, 9), 'point_cloud': (8, 4096, 3)}[key])
                h.update(key.encode())
                h.update(json.dumps(list(a.shape)).encode())
                h.update(b'float32')
                h.update(a.tobytes())
        assert h.hexdigest() == e['observation_hash']
        assert canonical([tasks[e['task_id']]['payload'], seed, h.hexdigest()]) == e['initial_hash']
    for world in bank['worlds'].values():
        assert hashlib.sha256((path.parent / world['obstacle_copy']).read_bytes()).hexdigest() == world['obstacle_sha256']
    return bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old', type=Path, required=True)
    parser.add_argument('--new', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    old, new = load_bank(args.old), load_bank(args.new)
    assert old['content_hash'] == 'd05dc8e57dd69c2ffd31ec0bdefe99c8a0eda1e779f9e06802f5c6dc0130eae8'
    assert old.keys() == new.keys()
    for key in old.keys() - {'worlds', 'tasks', 'episodes', 'content_hash'}:
        assert old[key] == new[key], key
    assert old['worlds'].keys() == new['worlds'].keys()
    changes = []
    for uid, a in old['worlds'].items():
        b = new['worlds'][uid]
        assert {k: v for k, v in a.items() if k != 'obstacle_path'} == {
            k: v for k, v in b.items() if k != 'obstacle_path'}, uid
        changes.append(dict(kind='world_output_path', world=uid,
                            old=a['obstacle_path'], new=b['obstacle_path']))
    mapping = {}
    for a, b in zip(old['tasks'], new['tasks']):
        assert a['payload']['z_range'] == [1.3, 4.7] and b['payload']['z_range'] == [1.0, 5.0]
        expected = dict(a, task_id=b['task_id'], payload=dict(a['payload'], z_range=[1.0, 5.0]))
        assert expected == b, 'task changed beyond z_range and derived id'
        mapping[a['task_id']] = b['task_id']
        changes.append(dict(kind='task', old=a, new=b))
    for a, b in zip(old['episodes'], new['episodes']):
        assert b['task_id'] == mapping[a['task_id']] and b['repeat'] == a['repeat']
        assert a.keys() == b.keys()
        changes.append(dict(kind='episode', repeat=a['repeat'], fields={
            k: {'old': a[k], 'new': b[k]} for k in a if a[k] != b[k]}))
    result = dict(status='PASS', old_content_hash=old['content_hash'],
                  new_content_hash=new['content_hash'], changes=changes,
                  scope='Task payloads differ only in z_range; ids/seeds/observations are rederived. '
                        'All episode field differences are listed; rendering metadata differences require review.')
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('REFREEZE_TASKS_PASS', 60, 300, new['content_hash'])


if __name__ == '__main__':
    main()
