"""Frozen, finite paper evaluation; no actor-dependent task selection."""
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .core import canonical_hash, file_hash, read_json, observation_hash, write_npz, ROLES


SCOPE = ("New tasks in a historically evaluated, BC-admitted domain; shared training "
         "templates; 11 different geometries and one exact training-geometry control. "
         "Not a blind test or unseen-structure generalization.")


def validate_models(cat):
    if set(cat['policies']) != {'bc', 'rl_2f3f9914db78_u32'}:
        raise ValueError('paper budget requires only E5 BC and frozen fresh203 u32')
    if cat['policies']['rl_2f3f9914db78_u32']['update'] != 32:
        raise ValueError('paper checkpoint update differs')


def geometry_hash(document):
    objects = [{k: v for k, v in o.items() if k != 'name'} for o in document['obstacles']]
    return canonical_hash(sorted(objects, key=canonical_hash))


def read_spec(path):
    spec = read_json(path)
    if canonical_hash({k:v for k,v in spec.items() if k != 'content_hash'}) != spec['content_hash']:
        raise ValueError('paper input specification changed')
    worlds = spec['worlds']
    if len(worlds) != 12 or len({w['world_id'] for w in worlds}) != 12:
        raise ValueError('paper world set differs')
    if [sum(w['role']==r for w in worlds) for r in ROLES] != [1,9,1,1]:
        raise ValueError('paper role allocation differs')
    duplicates = 0
    for w in worlds:
        actual = geometry_hash(w['document'])
        matches = sorted(uid for uid,h in spec['training_geometry_hashes'].items() if h==actual)
        if actual != w['geometry_hash'] or matches != w['exact_train_matches']:
            raise ValueError('paper geometry provenance differs')
        duplicates += bool(matches)
    if duplicates != 1:
        raise ValueError('paper geometry audit changed')
    return spec


def select_tasks(env, world_id, role, candidates=128):
    """First blocked candidate plus first distinct ordinary valid candidate."""
    from diffusion_policy_3d.env.collision_geom import segment_obstacle_collision
    env._rng = np.random.default_rng(int(canonical_hash(['paper-task-v1',world_id,20261020])[:8],16))
    choices = []
    for index in range(candidates):
        start, yaw, goal = env._sample_start_goal(role)
        blocked = segment_obstacle_collision(start,goal,env.obstacles,.1) is not None
        payload = dict(episode_id='paper-v1-'+world_id+'-'+str(index),
                       world_name=world_id,start_pos=start.tolist(),goal_pos=goal.tolist(),
                       start_yaw=float(yaw),z_range=[float(env.z_min),float(env.z_max)],
                       disturb_enable=False,role=role)
        choices.append(dict(payload=payload,blocked=blocked,candidate_index=index,
                            astar_diagnostic=env._last_astar_diagnostic))
        first_blocked = next((x for x in choices if x['blocked']),None)
        if first_blocked is not None:
            ordinary = next((x for x in choices if x['payload']['start_pos'] != first_blocked['payload']['start_pos']
                             or x['payload']['goal_pos'] != first_blocked['payload']['goal_pos']),None)
            if ordinary is not None:
                first_blocked = dict(first_blocked,selection='first_geometrically_blocked')
                ordinary = dict(ordinary,selection='first_distinct_valid_ordinary')
                return [first_blocked,ordinary]
    raise ValueError('finite candidate budget lacks blocked/distinct task: '+world_id)


def prepare_bank(root, cat, out):
    from scripts.evaluate_dppo_uav_formal_endpoint import reset_fixed_episode
    from .bank import check_bank, episode_key
    from .runtime import common_system, make_adapter, metadata, evaluation_sources
    from .workflow import save
    validate_models(cat)
    out = Path(out)
    spec = read_spec(Path(root)/'scripts/routea_paired_support/paper_bank_spec_v1.json')
    receipt = dict(spec_hash=spec['content_hash'],catalogue_hash=cat['content_hash'],
                   evaluation_sources=evaluation_sources(root),candidate_budget_per_world=128,
                   selection='first blocked + first distinct ordinary; geometry only',
                   seed_namespace='paper-task-v1/20261020')
    save(out/'inputs.json',receipt)
    if (out/'bank.json').exists():
        return check_bank(out/'bank.json')
    system,c = common_system(root,cat)
    bank = dict(schema='routea_paper_test_v1',scope=SCOPE,spec_hash=spec['content_hash'],
                worlds={},tasks=[],episodes=[],role_world_counts=dict(zip(ROLES,(1,9,1,1))))
    # Complete geometry-only selection for every world before any actor runs.
    for w in spec['worlds']:
        uid = w['world_id']
        obstacle = out/'obstacles'/(uid+'.json')
        save(obstacle,w['document'])
        world = dict(world_id=uid,role=w['role'],obstacle_path=str(obstacle.resolve()),
                     obstacle_copy=obstacle.relative_to(out).as_posix(),obstacle_sha256=file_hash(obstacle),
                     original_obstacle_sha256=w['original_sha256'],geometry_hash=w['geometry_hash'],
                     exact_train_matches=w['exact_train_matches'])
        adapter = make_adapter(system,c,world,w['role'])
        env = adapter.environment
        env._set_world(uid,str(obstacle.resolve()))
        world.update(geometry=[asdict(o) for o in env.obstacles],spawn_offset=env._spawn_offset.tolist())
        bank['worlds'][uid]=world
        selected = select_tasks(env,uid,w['role'])
        save(out/'selection'/(uid+'.json'),selected)
        for chosen in selected:
            task_id = canonical_hash(['paper-task-v1',uid,chosen['payload']])[:24]
            task = dict(chosen,task_id=task_id,world_id=uid,role=w['role'])
            bank['tasks'].append(task)
            for repeat in range(1,6):
                seed=int(canonical_hash(['paper-reset-v1',task_id,repeat])[:8],16)
                obs,ctx = reset_fixed_episode(env,task['payload'],seed=seed,reset_callable=adapter.reset)
                obs = {k:np.asarray(v,dtype=np.float32) for k,v in obs.items()}
                key=episode_key(task_id,repeat)
                npz=out/'initial'/(key+'.npz')
                if npz.exists():
                    with np.load(npz,allow_pickle=False) as saved:
                        if observation_hash(dict(saved))!=observation_hash(obs):
                            raise ValueError('partial bank reset is not reproducible')
                else:
                    write_npz(npz,obs)
                bank['episodes'].append(dict(key=key,task_id=task_id,repeat=repeat,reset_seed=seed,
                    observation_hash=observation_hash(obs),
                    initial_hash=canonical_hash([task['payload'],seed,observation_hash(obs)]),
                    initial_npz=npz.relative_to(out).as_posix(),initial_npz_sha256=file_hash(npz),
                    initial_pointcloud_metadata=metadata(ctx)))
        print('PAPER_BANK_WORLD',uid,flush=True)
    bank['content_hash']=canonical_hash(bank)
    save(out/'bank.json',bank)
    return check_bank(out/'bank.json')


def paper_statistics(records,bank,statistics):
    from .core import paired_summary
    treatment='rl_2f3f9914db78_u32'
    blocked={t['task_id'] for t in bank['tasks'] if t['blocked']}
    different={uid for uid,w in bank['worlds'].items() if not w['exact_train_matches']}
    subsets={
        'different_geometry_tasks':[r for r in records if r['world_id'] in different],
        'different_geometry_blocked':[r for r in records if r['world_id'] in different and r['task_id'] in blocked],
        'seen_geometry_control':[r for r in records if r['world_id'] not in different],
        'all_blocked_descriptive':[r for r in records if r['task_id'] in blocked]}
    statistics['scope']=SCOPE
    statistics['geometry_strata']={name:paired_summary(rows,treatment,'bc') for name,rows in subsets.items()}
    # Remove inherited development-domain wording also inside paired summaries.
    def label(value):
        if isinstance(value,dict):
            for k,v in value.items():
                if k=='inference_scope':value[k]=SCOPE
                else:label(v)
        elif isinstance(value,list):
            for v in value:label(v)
    label(statistics)
    statistics['primary_endpoint']='safe_arrival; different_geometry_tasks'
    statistics['task_selection']='geometry only; ordinary does not imply unblocked'
    return statistics
