"""Geometry-only A.1 task selection; not authorization to start ROS trials."""
import argparse
from collections import Counter
from pathlib import Path
import random
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.routea_testext_task_bank import bank_check, category_of
from scripts.run_routea_f3_resume import canonical,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--test-bank',required=True)
    p.add_argument('--freeze',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    freeze=Path(a.freeze);index=bank_check(freeze/'O_prime_index.json')
    banks=[bank_check(a.test_bank),bank_check(freeze/'bank/bank.json')]
    assert banks[0]['content_hash']==index['test_bank_hash']
    assert banks[1]['content_hash']==index['extension_v2_hash']
    pools={c:{} for c in ('zero','low','high')}
    for bank in banks:
        for task in bank['tasks']:
            world=bank['worlds'][task['world_id']];category=category_of(task,world)
            pools[category].setdefault(task['world_id'],[]).append(dict(task=task,world=world,category=category))
    selected=[]
    for category,target in [('zero',12),('low',18),('high',18)]:
        worlds=sorted(pools[category]);random.Random(2026110501).shuffle(worlds)
        counts=Counter()
        for repeat in range(2):
            for world in worlds:
                if sum(counts.values())==target:break
                tasks=sorted(pools[category][world],key=lambda x:x['task']['task_id'])
                if repeat<len(tasks):
                    selected.append(tasks[repeat]);counts[world]+=1
        assert sum(counts.values())==target and max(counts.values())<=2, 'A.1 geometric quota unavailable'
    assert len(selected)==48 and len({x['task']['task_id'] for x in selected})==48
    rng=random.Random(2026110301)
    task_order=list(range(48));rng.shuffle(task_order)
    schedule=[]
    for repeat in (1,2):
        for i in task_order:
            methods=['P0','P6','EGO'];rng.shuffle(methods)
            for method in methods:
                schedule.append(dict(task_id=selected[i]['task']['task_id'],repeat=repeat,method=method))
    result=dict(schema='routea_gazebo_A1_v1',O_prime_index_hash=index['content_hash'],
        tasks=selected,schedule=schedule,repeats=2,flights=288,
        selection_seed=2026110501,order_seed=2026110301,
        category_counts=dict(zero=12,low=18,high=18),requires_Claude_task_review=True,
        execution_started=False)
    result['content_hash']=canonical(result)
    write(a.output,result)
    print('GAZEBO_TASKS_SELECTED',48,288,result['content_hash'])


if __name__=='__main__':
    main()
