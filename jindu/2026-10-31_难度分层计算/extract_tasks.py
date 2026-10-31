"""Extract task geometry (start, goal, obstacles) for dev420 and test12.

Only environment information is written: no outcome of any policy.
Usage: python extract_tasks.py <dev_traces_dir> <test12_traces_dir> <out.json>
"""
import glob
import json
import sys


def load(trace_dir, split):
    tasks = {}
    for path in sorted(glob.glob(trace_dir + '/*.jsonl')):
        with open(path) as handle:
            first = json.loads(handle.readline())
        if first.get('type') != 'episode_start':
            raise ValueError('first record is not episode_start: ' + path)
        task = first['task']
        row = dict(
            task_id=first['task_id'], split=split, role=first['role'],
            world=task.get('world_name') or first.get('world_id'),
            start=[float(v) for v in task['start_pos']],
            goal=[float(v) for v in task['goal_pos']],
            start_yaw=float(task.get('start_yaw', 0.0)),
            geometry=first['geometry'])
        old = tasks.get(row['task_id'])
        if old is not None:
            if (old['start'], old['goal'], old['geometry']) != (row['start'], row['goal'], row['geometry']):
                raise ValueError('inconsistent geometry across repeats: ' + row['task_id'])
            continue
        tasks[row['task_id']] = row
    return tasks


def main(dev_dir, test_dir, out):
    dev = load(dev_dir, 'dev420')
    test = load(test_dir, 'test12')
    if len(dev) != 84 or len(test) != 48:
        raise ValueError('unexpected task counts: dev %d test %d' % (len(dev), len(test)))
    rows = sorted(list(dev.values()) + list(test.values()), key=lambda r: (r['split'], r['task_id']))
    json.dump(rows, open(out, 'w'))
    print('tasks', len(rows), 'dev', len(dev), 'test12', len(test))


if __name__ == '__main__':
    main(*sys.argv[1:4])
