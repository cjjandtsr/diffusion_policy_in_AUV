"""Approved A1 first-H4 crossover and static-goal response, evaluation only."""
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from .core import canonical_hash, file_hash, read_json, write_json
from .recovery import (read_bank, replay_state, clone_snapshot, paired_seed,
                       predict_with_noise, action_record, continue_flight)

ANGLES = (-135, -90, -45, 0, 45, 90, 135, 180)


def assert_diagonal(expected, actual, path='flight'):
    """Exact recursive comparison; matching outcome alone is not sufficient."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            raise ValueError('diagonal keys differ: '+path)
        for key in expected:
            assert_diagonal(expected[key], actual[key], path+'.'+key)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise ValueError('diagonal length differs: '+path)
        for i, (a, b) in enumerate(zip(expected, actual)):
            assert_diagonal(a, b, path+'[%d]'%i)
    elif expected != actual:
        raise ValueError('diagonal value differs: '+path)


def _json_value(value):
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    return value


def snapshot_digest(snapshot):
    adapter, obs, ctx = snapshot
    env = adapter.environment
    names = ('_drone_pos', '_drone_vel', '_drone_yaw', 'goal_w', '_reward_history',
             '_episode_start_world', 'episode_step', 'max_episode_steps',
             '_episode_rng_seed', '_observation_frame_index', '_episode_identity_hash',
             '_done_reason', '_pc_history', '_state_history', '_pc_meta_history',
             '_last_path', '_last_reward_terms', '_last_reward_metrics')
    value = dict(environment={k: getattr(env, k, None) for k in names},
                 rng=env._rng.bit_generator.state, observation=obs, context=ctx,
                 reward_state=adapter.reward_state)
    return canonical_hash(_json_value(value))


def fork_after_first(snapshot, first_sample):
    adapter, _, _ = clone_snapshot(snapshot)
    full = np.asarray(first_sample['full_action_16x11'], np.float32)
    obs, _, terminated, truncated, ctx = adapter.step(SimpleNamespace(policy=full[2:10]))
    return (adapter, obs, ctx), ctx['done_reason'] if terminated or truncated else None


def retarget_observation(env, observation, goal):
    """Change only target channels for every frame; no environment/reward step."""
    obs = {key: np.array(value, copy=True) for key, value in observation.items()}
    target = env._world_to_odom(np.asarray(goal, np.float32))
    for row in obs['agent_pos']:
        yaw = float(np.arctan2(row[7], row[8]))
        row[3:7] = env._build_state(row[:3], yaw, target)[3:7]
    return obs


def goal_response_metrics(path, goal):
    path, goal = np.asarray(path, dtype=np.float64), np.asarray(goal, dtype=np.float64)
    delta = goal-path[0]
    distance = float(np.linalg.norm(delta))
    if distance <= .5:
        raise ValueError('goal probe starts inside arrival ball')
    result = {'h4_target_projection_m': float(np.dot(path[4]-path[0], delta/distance))}
    for h in (4, 8):
        if len(path) <= h:
            continue
        a, d = path[:h], np.diff(path[:h+1], axis=0)
        t = np.clip(np.sum((goal-a)*d,axis=1)/np.maximum(np.sum(d*d,axis=1),1e-30),0,1)
        minimum = float(np.linalg.norm(a+t[:,None]*d-goal,axis=1).min())
        result['h%d_min_goal_center_distance_m'%h] = minimum
        result['h%d_min_goal_ball_distance_m'%h] = max(0.,minimum-.5)
    return result


def candidate_goals(env):
    """Same original distance/elevation; bearings relative to current yaw."""
    from diffusion_policy_3d.env.collision_geom import segment_obstacle_collision
    pos, old = env._drone_pos, env.goal_w
    radius = float(np.linalg.norm(old[:2]-pos[:2]))
    xmin,xmax,ymin,ymax = env._curriculum_xy_range()
    zmin,zmax = env._curriculum_z_range()
    candidates = [('original', None, old.copy())]
    for angle in ANGLES:
        theta = env._drone_yaw + np.deg2rad(angle)
        candidates.append(('bearing_%+d'%angle, angle,
                           np.array([pos[0]+radius*np.cos(theta),pos[1]+radius*np.sin(theta),old[2]],np.float32)))
    result=[]
    for label, angle, goal in candidates:
        reasons=[]
        if not (xmin<=goal[0]<=xmax and ymin<=goal[1]<=ymax and zmin<=goal[2]<=zmax and goal[2]>.2):
            reasons.append('outside_task_sampling_bounds')
        if np.linalg.norm(goal-pos)<=.5:
            reasons.append('inside_arrival_ball')
        history=np.asarray(env._state_history,dtype=np.float64)[-3:,:3]+env._spawn_offset
        if len(history)>1:
            a,d=history[:-1],np.diff(history,axis=0)
            t=np.clip(np.sum((goal-a)*d,axis=1)/np.maximum(np.sum(d*d,axis=1),1e-30),0,1)
            if np.linalg.norm(a+t[:,None]*d-goal,axis=1).min()<=.5:
                reasons.append('arrival_in_observed_history')
        if segment_obstacle_collision(goal,goal,env.obstacles,env.collision_thresh) is not None:
            reasons.append('goal_occupied')
        result.append(dict(label=label,bearing_deg=angle,goal=goal.tolist(),
            valid=not reasons,rejection_reasons=reasons,
            direct_clear=segment_obstacle_collision(pos,goal,env.obstacles,env.collision_thresh) is None,
            sampling_bounds=[xmin,xmax,ymin,ymax,zmin,zmax]))
    return result


def path_clearance(path, env):
    """Conservative signed clearance lower bound using 5 cm segment sampling."""
    from diffusion_policy_3d.env.collision_geom import signed_clearance, GeometryPurpose
    path=np.asarray(path,dtype=np.float64)
    best=float('inf')
    for a,b in zip(path[:-1],path[1:]):
        length=float(np.linalg.norm(b-a))
        n=max(1,int(np.ceil(length/.05)))
        for t in np.linspace(0.,1.,n+1):
            p=a+t*(b-a)
            clearance=min([float(p[2]-.2)]+[
                signed_clearance(p,o,purpose=GeometryPurpose.PHYSICAL_COLLISION,
                                 body_radius_m=env.collision_thresh) for o in env.obstacles])
            best=min(best,clearance-length/(2*n))
    return float(best)


def summarize_cells(rows):
    states=[]
    for sid in sorted({r['state_id'] for r in rows}):
        selected=[r for r in rows if r['state_id']==sid]
        means={k:float(np.mean([r['outcomes'][k]=='reached' for r in selected])) for k in 'ABCD'}
        states.append(dict(state_id=sid,task_id=selected[0]['task_id'],group=selected[0]['group'],
            draws=selected,arrival_fraction=means,
            response_contrast=((means['A']-means['B'])+(means['C']-means['D']))/2,
            source_contrast=((means['A']-means['C'])+(means['B']-means['D']))/2,
            interaction=means['A']-means['B']-means['C']+means['D']))
    tasks=[]
    for task in sorted({s['task_id'] for s in states}):
        selected=[s for s in states if s['task_id']==task]
        tasks.append(dict(task_id=task,state_ids=[s['state_id'] for s in selected],
            arrival_fraction={k:float(np.mean([s['arrival_fraction'][k] for s in selected])) for k in 'ABCD'}))
    return dict(state_count=len(states),task_count=len(tasks),states=states,tasks=tasks,
        task_equal_arrival_fraction={k:float(np.mean([t['arrival_fraction'][k] for t in tasks])) for k in 'ABCD'},
        interpretation='descriptive conditional diagnosis, not overall SR or independent draw inference')


def run_mechanism(root, cat, task_bank, frozen_path, baseline_path, out):
    """All diagonal gates precede any crossover or target intervention."""
    from .runtime import common_system, load_actor
    from .workflow import policy_order
    from scripts.evaluate_dppo_uav_formal_endpoint import _E5Policy
    bank, baseline, out = read_bank(frozen_path), Path(baseline_path), Path(out)
    if len(bank['states'])!=18 or bank['continuation_draws']!=[0,1,2] or bank['max_h4_cycles']!=12:
        raise ValueError('mechanism requires approved 18-state/3-draw/12-cycle bank')
    rl=[name for name in policy_order(cat) if name!='bc']
    if len(rl)!=2 or sorted(cat['policies'][n]['update'] for n in rl)!=[128,256]:
        raise ValueError('mechanism requires BC and exactly u128/u256')
    if len({cat['policies'][n]['runtime_identity_hash'] for n in rl})!=1:
        raise ValueError('crossed checkpoints must belong to the same training arm')
    rl.sort(key=lambda n:cat['policies'][n]['update'])
    names=['bc']+rl
    protocol=read_json(baseline/'protocol.json')
    original_models=read_json(baseline.parent/'models.json')
    if original_models['common']['bc_file_sha256']!=cat['common']['bc_file_sha256']:
        raise ValueError('baseline BC weight differs')
    for name in rl:
        if original_models['policies'][name]['state_sha256']!=cat['policies'][name]['state_sha256']:
            raise ValueError('baseline RL weight differs')
    if protocol['bank_hash']!=bank['content_hash']:
        raise ValueError('baseline state bank differs')
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'protocol.json',dict(schema='a1_cross_goal_v1',bank_hash=bank['content_hash'],
        catalogue_hash=cat['content_hash'],fork_after_h4=1,draws=[0,1,2],
        maximum_total_cycles=12,angles_deg=list(ANGLES),baseline=str(baseline),
        diagonal_comparison='exact all fields per cycle',unit='state/task, draws nested',
        sampling='unchanged deterministic DDIM with paired random initial noise',
        target_scope='open-loop static-goal response only; no reward or success-rate claim'))
    write_json(out/'frozen_bank.json',bank)
    system,config=common_system(root,cat)
    snapshots={s['id']:replay_state(system,config,task_bank,s) for s in bank['states']}
    old={}; receipts=[]; forks={}; signatures={}
    def policy_for(name):
        d=cat['policies'][name]
        if name!='bc':load_actor(root,system,d)
        identity=cat['common']['bc_identity_hash'] if name=='bc' else d['runtime_identity_hash']
        return _E5Policy(system,use_base_policy=d['use_base_policy'],identity_hash=identity)
    # Recompute initial sample too: do not bypass generation with archived actions.
    for name in names:
        policy=policy_for(name)
        for state in bank['states']:
            sid=state['id']; snapshot=snapshots[sid]
            source=baseline/name/(sid+'.json')
            record=read_json(source); old[name,sid]=record
            if record['state_id']!=sid or record['policy']!=name or record['actual_use_base_policy']!=policy.use_base_policy:
                raise ValueError('baseline branch/state mismatch')
            expected_identity=cat['common']['bc_identity_hash'] if name=='bc' else cat['policies'][name]['runtime_identity_hash']
            if record['policy_identity']!=expected_identity:
                raise ValueError('baseline actor identity mismatch')
            receipts.append(dict(path=str(source),sha256=file_hash(source)))
            for draw in bank['continuation_draws']:
                adapter,obs,ctx=snapshot
                bundle,noise=predict_with_noise(policy,obs,ctx,paired_seed(sid,draw))
                sample=record['samples'][draw]
                actual=action_record(bundle,adapter.environment,adapter.environment.goal_w)
                actual.update(draw=draw,seed=paired_seed(sid,draw),
                    policy_input_hash=policy.last_provenance.policy_input_hash,
                    initial_noise_hash=noise,observation_provenance=asdict(policy.last_provenance))
                flight=continue_flight(policy,snapshot,sid,draw,actual)
                expected=next(f for f in record['continuations'] if f['draw']==draw)
                path=out/'diagonal'/name/(sid+'_%d.json'%draw)
                write_json(path,dict(sample=actual,continuation=flight))
                try:
                    assert_diagonal(sample,actual,'sample')
                    assert_diagonal(expected,flight)
                except ValueError as error:
                    write_json(out/'diagonal_failure.json',dict(policy=name,state_id=sid,draw=draw,error=str(error)))
                    raise
                if name!='bc':
                    fork,reason=fork_after_first(snapshot,sample)
                    digest=snapshot_digest(fork)
                    if digest!=snapshot_digest(clone_snapshot(fork)):
                        raise ValueError('deep clone changed complete fork state')
                    forks[name,sid,draw]=(fork,reason)
                    signatures[name,sid,draw]=digest
            print('DIAGONAL_PASS',name,sid,flush=True)
    write_json(out/'diagonal_gate.json',dict(status='pass',flights=162,source_files=receipts))
    crossed=[]
    # Each opposite continuation uses the complete same source snapshot and the
    # original absolute cycle seeds. Diagonal suffix is already independently reproduced.
    for target in rl:
        policy=policy_for(target)
        source=next(n for n in rl if n!=target)
        for state in bank['states']:
            sid=state['id']
            for draw in bank['continuation_draws']:
                fork,reason=forks[source,sid,draw]
                if snapshot_digest(fork)!=signatures[source,sid,draw]:
                    raise ValueError('fork mutated between paired continuations')
                flight=None if reason else continue_flight(policy,fork,sid,draw,None,maximum=11,cycle_start=1)
                if flight and flight['rows']:
                    expected=next(f for f in old[source,sid]['continuations'] if f['draw']==draw)['rows'][1]
                    for key in ('seed','initial_noise_hash','observation_provenance','pre_state','decision','pointcloud_metadata_before'):
                        assert_diagonal(expected[key],flight['rows'][0][key],'paired_fork.'+key)
                if snapshot_digest(fork)!=signatures[source,sid,draw]:
                    raise ValueError('continuation mutated source snapshot')
                result=dict(state_id=sid,task_id=state['header']['task_id'],group=state['group'],draw=draw,
                    source_policy=source,continuation_policy=target,fork_digest=signatures[source,sid,draw],
                    absorbed=bool(reason),outcome=reason if reason else flight['outcome'],continuation=flight)
                write_json(out/'crossed'/target/(sid+'_%d.json'%draw),result)
                crossed.append({k:v for k,v in result.items() if k!='continuation'})
    goals={s['id']:candidate_goals(snapshots[s['id']][0].environment) for s in bank['states']}
    write_json(out/'goal_candidates.json',goals)
    goal_count=0
    for name in names:
        policy=policy_for(name)
        for state in bank['states']:
            sid=state['id']; adapter,obs,ctx=snapshots[sid]; env=adapter.environment
            results=[]
            for candidate in goals[sid]:
                if not candidate['valid']:
                    results.append(dict(candidate=candidate,samples=[])); continue
                goal=np.asarray(candidate['goal'],np.float32)
                newobs=obs if candidate['label']=='original' else retarget_observation(env,obs,goal)
                samples=[]
                for draw in bank['continuation_draws']:
                    bundle,noise=predict_with_noise(policy,newobs,ctx,paired_seed(sid,draw))
                    row=action_record(bundle,env,goal)
                    row.update(goal_response_metrics(row['planned_future_path'],goal))
                    row.update(draw=draw,seed=paired_seed(sid,draw),initial_noise_hash=noise,
                        observation_provenance=asdict(policy.last_provenance),
                        h4_clearance_lower_bound_m=path_clearance(row['planned_future_path'][:5],env),
                        h8_clearance_lower_bound_m=path_clearance(row['planned_future_path'][:9],env))
                    if noise!=old[name,sid]['samples'][draw]['initial_noise_hash']:
                        raise ValueError('goal intervention changed paired initial noise')
                    samples.append(row); goal_count+=1
                results.append(dict(candidate=candidate,samples=samples,
                    agent_pos_history=np.asarray(newobs['agent_pos']).tolist()))
            write_json(out/'goals'/name/(sid+'.json'),dict(policy=name,state_id=sid,
                task_id=state['header']['task_id'],group=state['group'],results=results))
            print('GOAL_RESPONSE',name,sid,flush=True)
    summary=dict(status='complete',diagonal_flights=162,crossed_cases=len(crossed),
        absorbed_crossed=sum(r['absorbed'] for r in crossed),goal_samples=goal_count,
        crossed_records=crossed,independent_unit='state/task; no binomial CI over draws')
    cross_index={(r['source_policy'],r['state_id'],r['draw']):r for r in crossed}
    cells=[]
    for state in bank['states']:
        sid=state['id']
        for draw in bank['continuation_draws']:
            a=next(f for f in old[rl[0],sid]['continuations'] if f['draw']==draw)
            d=next(f for f in old[rl[1],sid]['continuations'] if f['draw']==draw)
            b,c=cross_index[rl[0],sid,draw],cross_index[rl[1],sid,draw]
            cells.append(dict(state_id=sid,task_id=state['header']['task_id'],group=state['group'],draw=draw,
                outcomes=dict(A=a['outcome'],B=b['outcome'],C=c['outcome'],D=d['outcome']),
                source_absorbed=dict(u128=b['absorbed'],u256=c['absorbed'])))
    write_json(out/'crossed_cells.json',summarize_cells(cells))
    if len(crossed)!=108 or goal_count!=9*sum(c['valid'] for items in goals.values() for c in items):
        raise ValueError('mechanism output denominator mismatch')
    write_json(out/'summary.json',summary)
    return {k:v for k,v in summary.items() if k!='crossed_records'}
