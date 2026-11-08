"""Read-only final-paper curve and continuous executed-path clearance exports."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_routea_f3_resume import read, write, digest, canonical

POLICY_NAMES = {'P0':{'bc'}, 'P2':{'u128','P2'}, 'P4':{'C','P4'}, 'P6':{'T','P6'}}


def csv_write(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert records, 'empty export'
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def curves(args):
    plan = read(args.plan)
    records = []
    for policy, spec in plan['curves'].items():
        for u in range(spec['first'], spec['last']+1):
            path = Path(spec['path'])/('metrics_update_%06d.json' % u)
            m = read(path)
            audit, formal = m['rollout_audit'], m['formal_training']
            # Only completed rollout episodes count; no invented outcome for a batch boundary.
            outcomes = [r['terminal_facts']['done_reason'] for r in audit['transition_records']
                        if r['terminal_facts']['done_reason'] != 'none']
            assert len(outcomes) == audit['episodes']['completed']
            keep = [x['pre_update'].get('keep_diagnostics') for x in m['updates']]
            losses = [x['keep_loss'] for x in keep if x is not None]
            records.append(dict(policy=policy, update=u, source=str(path), source_sha256=digest(path),
                reached=outcomes.count('reached'), collision=outcomes.count('collision')+outcomes.count('ground_crash'),
                obstacle_collision=outcomes.count('collision'), ground_crash=outcomes.count('ground_crash'),
                other_terminal=sum(x not in ('reached','collision','ground_crash','time_limit') for x in outcomes),
                raw_terminal_counts=json.dumps({x:outcomes.count(x) for x in sorted(set(outcomes))},sort_keys=True),
                time_limit=outcomes.count('time_limit'), completed_episodes=len(outcomes),
                transitions=audit['transitions'],
                reward_learning_targets_mean=audit['learning_targets']['rewards']['mean'],
                reward_total_mean=audit['reward']['total']['mean'],
                approx_kl_mean=formal['optimizer_metric_summary']['approx_kl']['mean'],
                keep_loss_mean=sum(losses)/len(losses) if losses else '',
                actor_learning_rate=formal['actor_learning_rate']))
    csv_write(Path(args.output)/'training_curves.csv', records)
    print('FINAL_CURVES_EXPORTED', len(records))


def segment_clearance(a, b, obstacles, bound=math.inf):
    from scipy.optimize import minimize_scalar
    from diffusion_policy_3d.env.collision_geom import signed_clearance, GeometryPurpose
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    length = float(np.linalg.norm(b-a))
    best = bound
    for obs in obstacles:
        def distance(t):
            return signed_clearance(a+t*(b-a), obs,
                purpose=GeometryPurpose.AUDIT_REFERENCE, body_radius_m=.1)
        ends = [distance(0.), distance(1.)]
        # Signed distance is 1-Lipschitz; skip only with a conservative lower bound.
        if min(ends)-length >= best:
            continue
        result = minimize_scalar(distance, bounds=(0.,1.), method='bounded', options={'xatol':1e-10})
        assert result.success and math.isfinite(result.fun)
        best = min(best, *ends, float(result.fun))
    return best


def clearance(args):
    from diffusion_policy_3d.env.collision_geom import ObstacleSpec
    # Read and check complete committed traces. No ROOT validator disk mutation.
    from scripts.routea_paired_support import episodes
    import inspect
    source = inspect.getsource(episodes.validate_rows)
    old = 'goal = np.asarray(header["task"]["goal_pos"], np.float64)'
    if old in source:
        assert source.count(old) == 1
        namespace = dict(episodes.__dict__)
        exec(compile(source.replace(old, old.replace('float64','float32')), '<export-goal-precision>', 'exec'), namespace)
        episodes.validate_rows = namespace['validate_rows']
    plan = read(args.plan)
    index = read(args.index)
    assert canonical({k:v for k,v in index.items() if k!='content_hash'}) == index['content_hash']
    expected = {e['key']:e for e in index['conditions']}
    records, seen = [], set()
    for spec in plan['clearance_jobs']:
        job = Path(spec['job'])
        bank = read(spec['bank'])
        assert canonical({k:v for k,v in bank.items() if k!='content_hash'}) == bank['content_hash']
        tasks = {t['task_id']:t for t in bank['tasks']}
        for commit in sorted((job/'commits').glob('*.json')):
            header, transitions, summary = episodes.load_commit(job, commit)
            key = commit.stem
            assert key in expected and header['initial_hash'] == expected[key]['initial_hash']
            assert header['bank_hash'] == bank['content_hash']
            assert header['bank_hash'] == expected[key]['source_bank_hash']
            assert header['use_base_policy'] == (spec['policy'] == 'P0')
            assert header['policy'] in POLICY_NAMES[spec['policy']]
            pair = (spec['policy'], key)
            assert pair not in seen
            seen.add(pair)
            world = bank['worlds'][tasks[summary['task_id']]['world_id']]
            assert world['geometry'] == header['geometry']
            assert world['obstacle_sha256'] == header['obstacle_sha256']
            obstacles = [ObstacleSpec.from_dict(o) for o in world['geometry']]
            best = math.inf
            for row in transitions:
                path = row['executed_path']
                for a,b in zip(path,path[1:]):
                    best = segment_clearance(a,b,obstacles,best)
            assert math.isfinite(best)
            records.append(dict(policy=spec['policy'],key=key,task_id=summary['task_id'],repeat=summary['repeat'],
                world_id=header['world_id'],done_reason=summary['done_reason'],clearance_m=best,
                body_radius_m=.1, initial_hash=header['initial_hash'], source_bank_hash=header['bank_hash'],
                trace_sha256=read(commit)['sha256'], method='continuous segment minimum of signed 3D primitive distance; bounded convex minimization xatol=1e-10'))
    assert seen == {(policy,key) for policy in ('P0','P2','P4','P6') for key in expected}
    csv_write(Path(args.output)/'clearance.csv', records)
    print('FINAL_CLEARANCE_EXPORTED', len(records))


def trajectories(args):
    selection = read(args.selection)['tasks']
    assert len(selection) == 3 and {x['category'] for x in selection} == {'zero','low','high'}
    wanted = {(x['task_id'], x['repeat']):x for x in selection}
    assert len(wanted) == 3
    plan = read(args.plan)
    exported = set()
    from scripts.routea_testext_task_bank import category_of
    for spec in plan['clearance_jobs']:
        if spec['policy'] not in ('P0','P6'):
            continue
        job, bank = Path(spec['job']), read(spec['bank'])
        tasks = {t['task_id']:t for t in bank['tasks']}
        for commit in sorted((job/'commits').glob('*.json')):
            value = read(commit)
            summary = value['summary']
            pair = (summary['task_id'],summary['repeat'])
            if pair not in wanted:
                continue
            task = tasks[pair[0]]
            world = bank['worlds'][task['world_id']]
            assert category_of(task,world) == wanted[pair]['category']
            trace = job/value['trace']
            assert digest(trace) == value['sha256']
            rows = [json.loads(line) for line in trace.read_text(encoding='utf-8').splitlines()]
            assert rows[0]['policy'] in POLICY_NAMES[spec['policy']]
            assert rows[0]['use_base_policy'] == (spec['policy'] == 'P0')
            assert canonical({k:v for k,v in bank.items() if k!='content_hash'}) == bank['content_hash']
            assert rows[0]['bank_hash'] == bank['content_hash']
            assert rows[0]['geometry'] == world['geometry']
            assert rows[0]['obstacle_sha256'] == world['obstacle_sha256']
            key = (spec['policy'],*pair)
            assert key not in exported
            exported.add(key)
            write(Path(args.output)/(spec['policy']+'_'+commit.stem+'.json'),
                dict(task=task,geometry=world['geometry'],header=rows[0],summary=summary,
                     executed_segments=[r['executed_path'] for r in rows if r['type']=='transition'],
                     source_trace_sha256=digest(trace)))
    assert exported == {(policy,*pair) for policy in ('P0','P6') for pair in wanted}
    print('FINAL_TRAJECTORIES_EXPORTED', 6)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['curves','clearance','trajectories'])
    p.add_argument('--plan', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--index')
    p.add_argument('--selection', help='three tasks selected by Claude: task_id/repeat/category')
    args = p.parse_args()
    if args.phase == 'clearance' and not args.index:
        p.error('clearance requires --index')
    if args.phase == 'trajectories' and not args.selection:
        p.error('trajectories requires --selection')
    globals()[args.phase](args)


if __name__ == '__main__':
    main()
