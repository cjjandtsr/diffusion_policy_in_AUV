"""EGO collection in the existing evaluator; immutable old results are inputs."""
from pathlib import Path
import copy
import json
import time

import numpy as np

from .bank import check_bank
from .core import canonical_hash, file_hash, read_json, write_json
from .episodes import load_commit, record_episode
from .runtime import common_system, make_adapter, resolve_catalogue, evaluation_sources
from .workflow import save


def recorded_sources(bank, bank_path):
    if 'evaluation_sources' in bank:
        return bank['evaluation_sources']
    if bank.get('schema') != 'routea_paper_test_v1':
        raise ValueError('unknown bank source receipt layout')
    receipt = read_json(bank_path.parent/'inputs.json')
    if receipt['spec_hash'] != bank['spec_hash']:
        raise ValueError('source receipt belongs to a different paper specification')
    return receipt['evaluation_sources']


def selected_tasks(bank, per_group):
    from scripts.analyze_routea_trajectory_behavior import Scene
    counts = {True:0, False:0}
    chosen = set()
    for task in bank['tasks']:
        scene = Scene(bank['worlds'][task['world_id']]['geometry'])
        blocked = not scene.clear(task['payload']['start_pos'],task['payload']['goal_pos'])
        if counts[blocked] < per_group:
            chosen.add(task['task_id'])
            counts[blocked] += 1
    if len(chosen) != 2*per_group:
        raise ValueError('bank lacks the requested blocked/clear task subset')
    return chosen


def regression(root, args, bank, bank_path, cat, out):
    from .runtime import load_actor
    from scripts.evaluate_dppo_uav_formal_endpoint import _E5Policy
    if not args.checkpoint or not args.compare_existing or args.ego_smoke:
        raise ValueError('regression needs RL checkpoint and existing BC/RL jobs; no --ego-smoke')
    chosen = selected_tasks(bank,4)
    tasks = {t['task_id']:t for t in bank['tasks']}
    entries = [e for e in bank['episodes'] if e['task_id'] in chosen]
    baselines = {}
    for job in args.compare_existing:
        h,_,_ = load_commit(job,job/'commits'/(entries[0]['key']+'.json'))
        baselines[h['policy']] = job
    if set(baselines) != set(cat['policies']):
        raise ValueError('regression checkpoints and baseline policy names differ')
    system, config = common_system(root,cat)
    save(out/'selection.json',dict(regression=True,keys=[e['key'] for e in entries]))
    checked = []
    for name,d in cat['policies'].items():
        if name != 'bc': load_actor(root,system,d)
        identity = cat['common']['bc_identity_hash'] if name == 'bc' else d['runtime_identity_hash']
        policy = _E5Policy(system,use_base_policy=d['use_base_policy'],identity_hash=identity)
        job = out/'regression'/name
        (job/'commits').mkdir(parents=True,exist_ok=True)
        adapters = {}
        # Preserve the original traversal's environment creation order, even
        # for worlds without a selected regression task.
        for entry in bank['episodes']:
            task = tasks[entry['task_id']]
            world = copy.deepcopy(bank['worlds'][task['world_id']])
            world['obstacle_path'] = str((bank_path.parent/world['obstacle_copy']).resolve())
            if task['world_id'] not in adapters:
                adapters[task['world_id']] = make_adapter(system,config,world,task['role'])
            if task['task_id'] not in chosen: continue
            cp = job/'commits'/(entry['key']+'.json')
            if not cp.exists():
                record_episode(job,name,policy,adapters[task['world_id']],task,entry,world,bank['content_hash'])
            h,rows,end = load_commit(job,cp)
            old_h,old_rows,old_end = load_commit(baselines[name],baselines[name]/'commits'/(entry['key']+'.json'))
            check_pair(old_h,old_end,entry,task,bank)
            equal = (end['done_reason']==old_end['done_reason'] and len(rows)==len(old_rows)
                and all(np.array_equal(np.asarray(a[k]),np.asarray(b[k]))
                        for a,b in zip(rows,old_rows)
                        for k in ('full_action_16x11','executed_path')))
            checked.append(dict(policy=name,key=entry['key'],exact=equal))
            save(out/'regression_checks'/(name+'_'+entry['key']+'.json'),checked[-1])
            if not equal: raise ValueError('BC/RL regression differs: '+name+' '+entry['key'])
    save(out/'regression_check.json',dict(records=checked))
    save(out/'complete.json',dict(status='regression_pass',episodes=len(checked)))
    return 0


def check_pair(header, summary, entry, task, bank):
    if (header['task_id'] != entry['task_id'] or header['repeat'] != entry['repeat']
            or header['task'] != task['payload'] or header['bank_hash'] != bank['content_hash']
            or header['initial_hash'] != entry['initial_hash']
            or header['reset_seed'] != entry['reset_seed']
            or summary['initial_hash'] != entry['initial_hash']
            or summary['policy'] != header['policy']
            or summary['bank_hash'] != bank['content_hash']
            or header['obstacle_sha256'] != bank['worlds'][task['world_id']]['obstacle_sha256']
            or header.get('terminate_at_space_limits', True)):
        raise ValueError('EGO comparison has different task/initial state/physics')


def export_record(header, rows, summary, commit, job, task_number):
    from scripts.analyze_routea_trajectory_behavior import Scene
    scene = Scene(header['geometry'])
    points = np.asarray([header['task']['start_pos']] +
                        [p for row in rows for p in row['executed_path'][1:]], np.float64)
    start, goal = points[0], np.asarray(header['task']['goal_pos'])
    distance = float(np.linalg.norm(goal-start))
    calls = [u for row in rows for u in row.get('ego', {}).get('updates', [])]
    latencies = [u['planning_seconds'] for u in calls]
    return dict(policy=header['policy'], task_id=header['task_id'], task=task_number,
        repeat=header['repeat'], world=header['world_id'], role=header['role'],
        family=header['world_id'].split('__')[-2],
        blocked=not scene.clear(start, goal), outcome=summary['done_reason'],
        success=summary['safe_arrival'], decisions=summary['decisions'],
        duration=summary['physical_time_s'], D0=distance,
        **summary['trajectory_metrics'],
        normalized_best_gain=(distance-summary['trajectory_metrics']['min_goal_distance_m'])/distance,
        crossed_goal_plane=bool(np.any((points-start)@(goal-start) >= distance*distance)),
        raw_source=str((job/commit['trace']).resolve()), raw_sha256=commit['sha256'],
        planning_update_seconds_median=float(np.median(latencies)) if latencies else None,
        planning_update_seconds_p95=float(np.percentile(latencies, 95)) if latencies else None,
        trajectory_count=max((u['trajectory_count'] for u in calls), default=0),
        failed_plans=max((u['failed_plans'] for u in calls), default=0))


def run_ego(root, args):
    import torch
    from .ego import EgoPolicy
    from .analysis import tables
    if args.task_bank is None or args.ego_launch is None or args.out is None:
        raise ValueError('EGO requires --task-bank, --ego-launch and --out')
    if args.paper or args.recovery_only or args.recovery_bank or args.mechanism_baseline:
        raise ValueError('EGO reads an already frozen bank; do not combine with paper/recovery generation')
    bank_path = args.task_bank.resolve()
    bank = check_bank(bank_path)
    from .reference import source_compatible
    for name, expected in recorded_sources(bank, bank_path).items():
        if (name.startswith('diffusion_policy_3d/env/') or
                name == 'diffusion_policy_3d/model/common/synth_pointcloud.py'):
            if not source_compatible(name,file_hash(root/name),expected,True):
                raise ValueError('environment/observation source differs from bank: '+name)
    out = args.out.resolve()
    out.relative_to(Path('/home/cjj/AUV/program'))
    if out == Path('/home/cjj/AUV/program'):
        raise ValueError('use an experiment subdirectory')
    if out.exists() and not args.resume:
        raise FileExistsError('use new output or --resume')
    out.mkdir(parents=True, exist_ok=True)
    cat = resolve_catalogue(root, args.checkpoint or [])
    ego = EgoPolicy(args.ego_launch, args.ego_map_size, out/'logs/ego')
    # Hash the deployed ROS source tree, not just a launch include name.
    planner = args.ego_launch.resolve().parents[2]
    if not (planner/'plan_manage').is_dir():
        raise ValueError('launch must be inside the deployed planner/plan_manage/launch directory')
    ros_sources = {str(p.relative_to(planner)): file_hash(p) for p in sorted(planner.rglob('*'))
                   if p.is_file() and p.suffix in ('.cpp','.h','.hpp','.xml','.launch','.srv','.txt')}
    ego.identity_hash = canonical_hash(dict(adapter=ego.identity_hash, ros_sources=ros_sources))
    binaries = {}
    if not args.ego_regression:
        import subprocess
        import roslib.packages
        actual_package = Path(subprocess.check_output(['rospack','find','ego_planner'],text=True).strip()).resolve()
        if actual_package != args.ego_launch.resolve().parents[1]:
            raise ValueError('rospack resolves a different ego_planner workspace')
        nodes = roslib.packages.find_node('ego_planner','ego_planner_node') or []
        if len(nodes) != 1:
            raise ValueError('expected one built ego_planner_node')
        node = Path(nodes[0]).resolve()
        binaries[str(node)] = file_hash(node)
        for library in ('plan_env','bspline_opt','path_searching'):
            path = node.parents[1]/('lib'+library+'.so')
            if not path.is_file():
                raise FileNotFoundError('missing catkin library: '+str(path))
            binaries[str(path)] = file_hash(path)
        ego.identity_hash = canonical_hash(dict(source=ego.identity_hash,binaries=binaries))
    protocol = dict(bank_hash=bank['content_hash'], policy='ego', identity=ego.identity_hash,
        catalogue_hash=cat['content_hash'],
        execution='common_H8_H4_0.3s', expected_episodes=len(bank['episodes']),
        evaluation_sources=evaluation_sources(root), ros_sources=ros_sources,
        binaries=binaries,
        map_size=list(args.ego_map_size), repeat_policy='all_five_frozen_reset_seeds',
        compare_existing=[str(p.resolve()) for p in (args.compare_existing or [])])
    if (out/'protocol.json').exists() and read_json(out/'protocol.json') != protocol:
        raise ValueError('resume requires identical sources, binaries, bank and comparisons')
    save(out/'protocol.json', protocol)
    if args.ego_regression:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        with torch.jit.optimized_execution(False):
            return regression(root,args,bank,bank_path,cat,out)
    tasks = {t['task_id']:t for t in bank['tasks']}
    entries = bank['episodes']
    if args.ego_smoke:
        chosen = selected_tasks(bank,2)
        entries = [e for e in entries if e['task_id'] in chosen and e['repeat'] == 1]
    save(out/'selection.json', dict(smoke=args.ego_smoke, keys=[e['key'] for e in entries]))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    job = out/'full/ego'
    (job/'commits').mkdir(parents=True, exist_ok=True)
    with torch.jit.optimized_execution(False):
        system, config = common_system(root, cat)
        adapters = {}
        for index, entry in enumerate(entries, 1):
            if (job/'commits'/(entry['key']+'.json')).exists():
                continue
            task = tasks[entry['task_id']]
            world = copy.deepcopy(bank['worlds'][task['world_id']])
            world['obstacle_path'] = str((bank_path.parent/world['obstacle_copy']).resolve())
            if task['world_id'] not in adapters:
                adapters[task['world_id']] = make_adapter(system, config, world, task['role'])
            try:
                record_episode(job, 'ego', ego, adapters[task['world_id']], task, entry, world, bank['content_hash'])
            finally:
                ego.close()
            print('EGO_EPISODE',index,len(entries),entry['key'],flush=True)
    summaries, records = [], []
    if {p.stem for p in (job/'commits').glob('*.json')} != {e['key'] for e in entries}:
        raise ValueError('EGO completed matrix has missing or extra episodes')
    task_numbers = {t['task_id']:i for i,t in enumerate(bank['tasks'])}
    jobs = [job] + [p.resolve() for p in (args.compare_existing or [])]
    policy_names = set()
    for existing in jobs:
        name = None
        for entry in entries:
            path = existing/'commits'/(entry['key']+'.json')
            header, rows, summary = load_commit(existing, path)
            check_pair(header, summary, entry, tasks[entry['task_id']], bank)
            if name is not None and header['policy'] != name:
                raise ValueError('mixed policies in one job')
            name = header['policy']
            if existing == job and (header.get('policy_kind') != 'ego' or
                    header['policy_identity_hash'] != ego.identity_hash):
                raise ValueError('wrong EGO identity')
            summaries.append(summary)
            records.append(export_record(header, rows, summary, read_json(path), existing,
                                         task_numbers[entry['task_id']]))
        if name in policy_names:
            raise ValueError('duplicate comparison policy')
        policy_names.add(name)
    save(out/'comparison_statistics.json', tables(summaries))
    save(out/'b3_ego_records.json', dict(records=records,
        metric_scope='common trace_metrics; full executed polyline',
        timing_scope='synchronous observation/FSM service CPU time, not model-only inference'))
    ego_rows = [r for r in records if r['policy']=='ego']
    lines = ['# EGO common H4 evaluation', '', 'Smoke: '+str(args.ego_smoke),
             'Valid episodes: '+str(len(ego_rows)), '', '|Outcome|Count|','|---|---:|']
    from collections import Counter
    lines += ['|{}|{}|'.format(k,v) for k,v in sorted(Counter(r['outcome'] for r in ego_rows).items())]
    lines += ['', 'Historical BC/RL results were read, not rerun.',
              'EGO uses the common H4 commitment; this is not native high-rate Gazebo execution.']
    (out/'b3_ego.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(out/'complete.json',dict(status='smoke_complete' if args.ego_smoke else 'complete',
                                      episodes=len(ego_rows), bank_hash=bank['content_hash']))
    return 0
