"""Reconstruct P6 training prefixes for F7 calibration; never use evaluation traces."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_routea_a2 import read, write, digest
from diffusion_policy_3d.env.reward_v5 import approach_cost
from diffusion_policy_3d.env.reward_v2_support.reward import space_running_cost
from diffusion_policy_3d.env.reward_v2_support.first_event import locate_first_event
from diffusion_policy_3d.env.collision_geom import load_obstacles


def reset_rng(record):
    from diffusion_policy_3d.env.kinematic_uav_env import KinematicUAVEnv
    # Replay only the original reset draws; no point cloud or diagnostic A* needed.
    env=KinematicUAVEnv.__new__(KinematicUAVEnv)
    env._rng=np.random.default_rng(record['reset_seed'])
    env.curriculum_stage='C2';env.z_min=1.;env.z_max=5.;env.mid_state_pos_scale=.5
    for _ in range(50):
        start,yaw,goal=env._draw_start_goal_candidate(record['role'])
        if env._rng.random()<.3:start,yaw=env._perturb_to_mid_state(start,yaw)
        # NumPy 2 scalar promotion can shift the reset perturbation by one float32
        # ULP versus P6's NumPy 1. This match only locates the RNG state; subsequent
        # trajectory reconstruction starts from the recorded position and is exact.
        if (np.allclose(start,record['position_before_xyz'],atol=1e-6,rtol=0.)
                and np.array_equal(goal,np.asarray(record['goal_xyz'],np.float32))):
            return env._rng
    raise ValueError('recorded reset not reproducible from reset_seed')


def reconstruct(record, obstacles, start, rng):
    p=np.asarray(record['position_before_xyz'],np.float32)
    path=[p.copy()]
    for delta in np.asarray(record['action_chunk'],np.float32)[:4,:3]:
        p=p+delta;path.append(p.copy())
    path=np.asarray(path);goal=np.asarray(record['goal_xyz'],np.float32)
    event=locate_first_event(path[:4],goal,obstacles,.1,5.,terminate_at_space_limits=False)
    if event is None:
        # P6 config: exec_noise_scale=0, disturb_enable=True, probability=.2, scale=.3.
        if rng.random()<.2:path[4]+=rng.normal(0,.3,3).astype(np.float32)
        event=locate_first_event(path,goal,obstacles,.1,5.,terminate_at_space_limits=False)
    if event is not None:
        path=np.vstack((path[:event.segment_index+1],event.position.astype(np.float32)))
    dt=np.full(len(path)-1,.3)
    if event is not None:dt[-1]*=event.fraction
    assert np.array_equal(path[-1],np.asarray(record['position_after_xyz'],np.float32)),'reconstructed endpoint differs'
    assert abs(float(dt.sum())-record['reward_metrics']['space_elapsed_s'])<1e-12,'elapsed differs'
    reason=record['terminal_facts']['done_reason']
    actual='none' if event is None else event.kind
    assert actual==('none' if reason=='time_limit' else reason), 'reconstructed first event differs'
    space=space_running_cost(path,start,record['goal_xyz'],dt,1.,72.)
    assert abs(space-record['reward_terms']['space'])<1e-7,'reconstructed space cost differs'
    cost,c=approach_cost(path,obstacles,dt,2.)
    return dict(executed_path=path.tolist(),segment_durations=dt.tolist(),
                approach_cbf_w2=cost,point_clearances=c.tolist(),
                endpoint_source='float32 world deltas plus replayed reset/disturbance RNG',
                verification='independent endpoint, first event, elapsed and space integral matched')


def select_weight(success):
    return 2. if success['count']>=50 and success['median']>=-2. and success['signed_p10']>=-8. else 1.


def export(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    geometry={};episodes={};rngs={};failures=[];sources=[];seen=set();total=nonzero=0
    fs=[Path(args.metrics)/('metrics_update_%06d.json'%u) for u in range(129,257)]
    with (out/'reconstructed_transitions.jsonl').open('x',encoding='utf8') as stream:
        for p in fs:
            m=read(p);sources.append(dict(path=str(p),sha256=digest(p)))
            assert m['runtime_identity_hash']=='d987235a0cbb5ad395bbd6e7979bdb8d7e0cff1303b0611cb462c7d554bbd61f'
            for r in m['rollout_audit']['transition_records']:
                total+=1;tid=r['transition_id'];assert tid not in seen;seen.add(tid)
                eid=r['episode_id']
                if eid not in episodes:
                    # Remaining-horizon context is one exactly at reset, including mid-state reset.
                    episodes[eid]=dict(start=r['position_before_xyz'],complete_start=r['critic_episode_context'][-1]==1.,
                        valid=True,cost=0.,outcome='fragment',decisions=0)
                    rngs[eid]=reset_rng(r)
                e=episodes[eid];e['decisions']+=1
                world=r['world_id']
                if world not in geometry:
                    gp=Path(args.worlds)/(world+'.world.obstacles.json')
                    geometry[world]=load_obstacles(str(gp))
                    sources.append(dict(path=str(gp),sha256=digest(gp)))
                try:
                    details=reconstruct(r,geometry[world],e['start'],rngs[eid])
                except (AssertionError,KeyError,ValueError) as exc:
                    e['valid']=False;failures.append(dict(transition_id=tid,error=str(exc)));continue
                e['cost']+=details['approach_cbf_w2'];nonzero+=details['approach_cbf_w2']!=0.
                e['outcome']=r['terminal_facts']['done_reason']
                stream.write(json.dumps(dict(transition_id=tid,episode_id=eid,world_id=world,
                    outcome=e['outcome'],reconstructed=True,**details),allow_nan=False)+'\n')
    complete=[dict(episode_id=k,**v) for k,v in episodes.items() if v['valid'] and v['complete_start'] and v['outcome'] not in ('none','fragment')]
    stats={}
    for reason in ('reached','collision','ground_crash','time_limit'):
        a=[e['cost'] for e in complete if e['outcome']==reason]
        stats[reason]=dict(count=len(a),median=float(np.median(a)) if a else None,
            signed_p10=float(np.quantile(a,.10)) if a else None,
            signed_p90=float(np.quantile(a,.90)) if a else None)
    s=stats['reached']
    selected=select_weight(s)
    report=dict(reconstruction_passed=len(failures)/total<=.01,total_decisions=total,
        rejected_decisions=len(failures),failures=failures,stats=stats,selected_weight=selected,
        quantile_rule='signed reward p10 >= -8; F7 v1.3, user confirmed 2026-11-09',
        nonzero_decision_fraction=nonzero/(total-len(failures)) if total>len(failures) else None,
        incomplete_or_rejected_episodes=len(episodes)-len(complete),sources=sources,
        interpretation='reconstructed training-domain evidence; Claude confirms w before prepare')
    write(out/'complete_episodes.json',complete);write(out/'calibration.json',report)
    print(json.dumps({k:report[k] for k in ('reconstruction_passed','total_decisions','rejected_decisions','stats','selected_weight')}))
    if not report['reconstruction_passed']:raise RuntimeError('more than 1% reconstruction rejected')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('metrics','worlds','output'):p.add_argument('--'+k,required=True)
    export(p.parse_args())
