#!/usr/bin/env python3
"""Select published task IDs from frozen banks without sampling new inputs."""
import argparse
import copy
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.routea_paired_support.core import canonical_hash, read_json, write_json
from scripts.routea_paired_support.ego_workflow import recorded_sources
from scripts.routea_testext_task_bank import bank_check, verify_arrays


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank', type=Path, action='append', required=True)
    p.add_argument('--tasks', type=Path, required=True, help='published tasks_186.csv')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    with a.tasks.open(encoding='utf-8-sig', newline='') as stream:
        ids = [r['task_id'] for r in csv.DictReader(stream)]
    if len(ids) != 186 or len(set(ids)) != 186:
        raise ValueError('expected the published 186 unique tasks')
    a.out.mkdir(parents=True, exist_ok=False)
    tasks, episodes, worlds, sources, origins = {}, {}, {}, None, []
    for path in a.bank:
        bank = bank_check(path)
        current = recorded_sources(bank, path)
        physics = lambda receipt: {k: v for k, v in receipt.items()
            if k.startswith('diffusion_policy_3d/env/') or
            k == 'diffusion_policy_3d/model/common/synth_pointcloud.py'}
        if sources is not None and physics(current) != physics(sources):
            raise ValueError('input banks have different environment/observation receipts')
        sources = current
        origins.append(dict(path=str(path.resolve()), content_hash=bank['content_hash'],
                            evaluation_sources=current))
        for task in bank['tasks']:
            uid = task['task_id']
            if uid not in ids:
                continue
            if uid in tasks and tasks[uid] != task:
                raise ValueError('conflicting task payload')
            tasks[uid] = task
            world = copy.deepcopy(bank['worlds'][task['world_id']])
            world['obstacle_copy'] = str((path.parent/world['obstacle_copy']).resolve())
            worlds[task['world_id']] = world
        for entry in bank['episodes']:
            if entry['task_id'] not in ids:
                continue
            entry = copy.deepcopy(entry)
            entry['initial_npz'] = str((path.parent/entry['initial_npz']).resolve())
            if entry['key'] in episodes and episodes[entry['key']] != entry:
                raise ValueError('duplicate episode from different input banks')
            episodes[entry['key']] = entry
    if set(tasks) != set(ids) or len(episodes) != 930 or len(worlds) != 40:
        raise ValueError('published task/world/repeat matrix is incomplete')
    bank = dict(schema='paper_186_frozen_v1', tasks=[tasks[i] for i in ids],
                episodes=list(episodes.values()), worlds=worlds,
                evaluation_sources=sources, origins=origins)
    bank['content_hash'] = canonical_hash(bank)
    verify_arrays(bank, a.out)
    write_json(a.out/'bank.json', bank)


if __name__ == '__main__':
    main()
