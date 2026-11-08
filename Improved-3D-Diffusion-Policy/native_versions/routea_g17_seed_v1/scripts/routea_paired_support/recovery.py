"""Frozen-state diagnostics used by the shared paired evaluator, never training."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .core import canonical_hash, file_hash, read_json, write_json, write_npz


def paired_seed(state_id, draw, cycle=0):
    return int(canonical_hash(['recovery-v1', state_id, int(draw), int(cycle)])[:8], 16)


def clone_snapshot(snapshot):
    return copy.deepcopy(snapshot)


def continuation_limit(adapter, maximum=12):
    env = adapter.environment
    remaining = max(0, env.max_episode_steps-env.episode_step)
    return remaining if maximum is None else min(maximum, remaining)


def recovery_counts(states, policies, draws=50, continuations=5, remaining=None, maximum=12):
    return dict(states=states, policies=policies, open_loop_samples=states*policies*draws,
                continuations=states*policies*continuations,
                max_continuation_decisions=(states*maximum if remaining is None else sum(remaining))*policies*continuations)


def check_replayed_state(env, expected, actual_provenance, expected_provenance):
    for key, actual in [('position',env._drone_pos),('velocity',env._drone_vel),('yaw',env._drone_yaw)]:
        if not np.allclose(actual, expected[key], atol=1e-6, rtol=0):
            raise ValueError('frozen state replay mismatch: '+key)
    for key in ('raw_history_hash','normalized_history_hash','policy_input_hash'):
        if actual_provenance[key] != expected_provenance[key]:
            raise ValueError('frozen three-frame/input replay mismatch: '+key)


def verify_pairing(reference, records):
    if len(reference)!=len(records):
        raise ValueError('different diagnostic draw counts')
    for a,b in zip(reference,records):
        for key in ('draw','seed','policy_input_hash','initial_noise_hash'):
            if a[key]!=b[key]:
                raise ValueError('same-state pairing mismatch: '+key)


def predict_with_noise(policy, observation, context, seed):
    from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.observation_provenance import canonical_array_hash
    captured=[]
    def capture(module,inputs,output):
        captured.append(canonical_array_hash(output.chains[:,0]))
    hook=policy.system.official_runtime.register_forward_hook(capture)
    try:
        bundle=policy.predict(observation,context,action_seed=seed)
    finally:
        hook.remove()
    if len(captured)!=1:raise ValueError('expected one actual diffusion chain')
    return bundle,captured[0]


def read_bank(path):
    bank = read_json(path)
    if canonical_hash({k:v for k,v in bank.items() if k!='content_hash'})!=bank['content_hash']:
        raise ValueError('recovery bank hash mismatch')
    if bank['schema'] not in ('routea_recovery_v1','routea_a1_mechanism_v1'):
        raise ValueError('unknown recovery bank')
    if bank['schema']=='routea_recovery_v1' and len(bank['states'])!=24:
        raise ValueError('expected frozen 24-state recovery bank')
    if not bank['states'] or not 1 <= bank['draws'] <= 50:
        raise ValueError('invalid recovery state or draw count')
    if (not bank['continuation_draws'] or len(set(bank['continuation_draws']))!=len(bank['continuation_draws'])
            or any(not 0 <= i < bank['draws'] for i in bank['continuation_draws'])):
        raise ValueError('invalid recovery continuation draws')
    if not 1 <= bank['max_h4_cycles'] <= 60:
        raise ValueError('invalid recovery continuation budget')
    return bank


def obstacle_witness(geometry, position, yaw):
    """Positive witness on 192 real camera pixels; a miss remains unknown."""
    import math
    import torch
    from diffusion_policy_3d.model.common.synth_pointcloud import (
        SynthPointCloud,DepthCameraIntrinsics,intersect_rays_box,
        intersect_rays_cylinder,intersect_rays_ground)
    renderer=SynthPointCloud.__new__(SynthPointCloud)
    renderer.device=torch.device('cpu')
    renderer.cam=DepthCameraIntrinsics()
    renderer._build_geometry({'obstacles':geometry})
    u,v=np.meshgrid(np.linspace(0,319,16).astype(int),np.linspace(0,239,12).astype(int))
    rays=torch.tensor(np.column_stack(((u.ravel()-160)/renderer.cam.fx,
        (v.ravel()-120)/renderer.cam.fy,np.ones(192))),dtype=torch.float32)
    rays=rays/torch.linalg.norm(rays,dim=1,keepdim=True)
    origin,rotation=renderer._cam_to_world_pose(np.asarray(position),yaw,math.radians(8))
    origins=origin[None].expand(192,-1)
    directions=rays@rotation.T
    nearest=torch.full((192,),float('inf'))
    if renderer.n_boxes:
        nearest=torch.minimum(nearest,intersect_rays_box(origins,directions,
            renderer.box_center,renderer.box_half_ext,renderer.box_yaw).min(dim=1).values)
    if renderer.n_cylinders:
        nearest=torch.minimum(nearest,intersect_rays_cylinder(origins,directions,
            renderer.cyl_center,renderer.cyl_radius,renderer.cyl_z_min,renderer.cyl_z_max).min(dim=1).values)
    ground=intersect_rays_ground(origins,directions,0.)
    valid=(nearest>=.05)&(nearest<=10)&(nearest<ground)
    return int(valid.sum())


def freeze_bank(behavior_path, output_path):
    """Prepare a small, immutable replay input from already accepted traces."""
    import hashlib
    import torch
    from scripts.analyze_routea_trajectory_behavior import Scene
    torch.set_num_threads(2)
    data=json.loads(Path(behavior_path).read_text(encoding='utf-8'))
    policies=('bc','rl_2f3f9914db78_u32','rl_2f3f9914db78_u384')
    sources=[r for r in data['records'] if r['policy'] in policies]
    raw={}
    def get_raw(r):
        if r['source'] not in raw:
            b=Path(r['source']).read_bytes()
            raw[r['source']]=([json.loads(s) for s in b.splitlines()],hashlib.sha256(b).hexdigest())
        return raw[r['source']]
    def make(r,decision,group):
        objects,digest=get_raw(r)
        h=objects[0]; row=objects[decision+1]
        if row['decision']!=decision or row['done_reason']!='none':
            raise ValueError('anchor must precede a nonterminal action')
        key=canonical_hash([r['policy'],r['task_id'],r['repeat'],decision])[:16]
        return dict(id=key,group=group,source_policy=r['policy'],source_task=r['task'],
            source_repeat=r['repeat'],source_outcome=r['outcome'],source_trace=r['source'],
            source_trace_sha256=digest,decision=decision,header=h,
            pre_state=row['pre_state'],provenance=row['observation_provenance'],
            expected_full_action=row['full_action_16x11'],source_action_seed=row['action_seed'],
            prefix_actions=[x['policy_action_8x11'] for x in objects[1:decision+1]],
            labels=r['cycles'][decision])
    states=[]; used=set()
    # These six predeclared witnesses were inspected before seeing A1 results.
    for task,policy,decision,label in ((8,policies[2],23,'delayed_after_near_goal'),
            (56,policies[2],22,'delayed_lost_early_ability'),(49,policies[2],2,'delayed_ground'),
            (76,policies[2],4,'safe_tail_arrival_cancelled'),(6,policies[2],25,'whole_plan_failure'),
            (30,policies[1],41,'successful_bypass_control')):
        r=next(r for r in sources if r['task']==task and r['policy']==policy and r['repeat']==1)
        s=make(r,decision,'mechanism/'+label); states.append(s); used.add(s['id'])
    for scene in ('post_bypass','ground_direct','visible_clear'):
        for success in (True,False):
            chosen_tasks=set(); chosen_sources=[]; pool=[]
            for r in sources:
                if r['success']!=success:continue
                for c in r['cycles']:
                    i=c['decision']
                    if not (4<=i<=48 and i<r['decisions']-1 and c['clear'] and c['distance']>=2):continue
                    if scene=='post_bypass' and i!=r['first_passed_initial_blockers']:continue
                    if scene=='ground_direct' and not (r['initial_clear'] and c['ground_history_certificate']):continue
                    if scene=='visible_clear' and c['empty_view_certificate']:continue
                    pool.append((r,i)); break
            pool.sort(key=lambda x:canonical_hash([scene,success,x[0]['task_id'],x[0]['policy'],x[0]['repeat']]))
            for slot in range(3):
                preferred=policies[slot]
                ordered=sorted(pool,key=lambda x:x[0]['policy']!=preferred)
                selected=None
                for r,i in ordered:
                    if r['task_id'] in chosen_tasks:continue
                    objects,_=get_raw(r); h=objects[0]
                    if Scene(h['geometry']).geometry_mismatches:continue
                    s=make(r,i,scene+('/success' if success else '/failure'))
                    if s['id'] in used:continue
                    witness=obstacle_witness(h['geometry'],s['pre_state']['position'],s['pre_state']['yaw']) if scene=='visible_clear' else None
                    if scene=='visible_clear' and not witness:continue
                    s['obstacle_ray_witness_count']=witness
                    selected=s; chosen_tasks.add(r['task_id']); chosen_sources.append(r['policy']); break
                if selected is None:raise ValueError('insufficient distinct tasks for '+scene)
                states.append(selected); used.add(selected['id'])
    bank=dict(schema='routea_recovery_v1',date='2026-10-18',states=states,
        selection='18 scene/outcome stratified states (3 distinct tasks per cell) plus 6 predeclared mechanism witnesses; source-balanced where available; no A1 outcome selection',
        visibility='ground: conservative latest-3 certificate; visible: positive exact-pixel ray witness, before downsampling; unknown is not visible',
        source_behavior_sha256=file_hash(behavior_path),draws=50,continuation_draws=list(range(5)),max_h4_cycles=12)
    bank['content_hash']=canonical_hash(bank)
    # Source preparation runs on Windows as well; no hard-link publication is
    # needed for a generated input that is verified before delivery.
    with Path(output_path).open('x',encoding='utf-8') as stream:
        json.dump(bank,stream,ensure_ascii=False,sort_keys=True,indent=2)
    return bank


def freeze_a1_mechanism_bank(behavior_path, trace_root, output_path, u128_policy, u256_policy):
    """Freeze 6 failed, 6 early-success and 6 BC-success states from A1 raw traces.

    This is a diagnostic, outcome-stratified bank; its draws are never a
    population success-rate estimate. ``trace_root`` is the returned A1
    evaluation directory on Windows, while each source path stays in WSL form.
    """
    import hashlib
    data = json.loads(Path(behavior_path).read_text(encoding='utf-8'))
    records = data['records']
    by_key = {(r['task_id'],r['repeat'],r['policy']):r for r in records}
    if len(by_key)!=len(records):
        raise ValueError('duplicate A1 behavior identity')
    if {u128_policy,u256_policy,'bc'}-set(r['policy'] for r in records):
        raise ValueError('requested A1 policy missing from behavior')
    root = Path(trace_root)
    cache = {}

    def read_trace(r):
        source = r['source']
        if source not in cache:
            marker = '/full/'
            if marker not in source:
                raise ValueError('unexpected A1 source trace path')
            relative = source.split(marker,1)[1]
            local = root/'full'/relative
            raw = local.read_bytes()
            cache[source] = ([json.loads(line) for line in raw.splitlines()],
                             hashlib.sha256(raw).hexdigest())
        return cache[source]

    def eligible_cycle(r, group):
        # The last row can end in success/collision and is not a replay anchor.
        options = [c for c in r['cycles'][:-1] if c['decision']<=48 and c['clear'] and
                   .5 < c['distance'] <= (1. if group=='u256_near_goal_timeout' else 2.)]
        if not options:
            return None
        return options[0] if group=='u256_near_goal_timeout' else options[-1]

    def candidates(group):
        if group=='u256_near_goal_timeout':
            pool = [r for r in records if r['policy']==u256_policy and r['outcome']=='time_limit'
                    and by_key[(r['task_id'],r['repeat'],u128_policy)]['outcome']=='reached']
        elif group=='u128_success_control':
            pool = [r for r in records if r['policy']==u128_policy and r['outcome']=='reached'
                    and by_key[(r['task_id'],r['repeat'],u256_policy)]['outcome']=='time_limit']
        else:
            pool = [r for r in records if r['policy']=='bc' and r['outcome']=='reached']
        options = [(r,eligible_cycle(r,group)) for r in pool]
        options = [(r,c) for r,c in options if c is not None]
        return sorted(options,key=lambda rc:canonical_hash(['a1-mechanism-v1',group,
                            rc[0]['task_id'],rc[0]['repeat'],rc[1]['decision']]))

    states = []
    for group in ('u256_near_goal_timeout','u128_success_control','bc_success_control'):
        used_tasks = set()
        for r,c in candidates(group):
            if r['task_id'] in used_tasks:
                continue
            objects,digest = read_trace(r)
            decision = c['decision']
            header,row = objects[0],objects[decision+1]
            if row['decision']!=decision or row['done_reason']!='none':
                raise ValueError('selected A1 anchor is terminal or misaligned')
            state_id = canonical_hash(['a1-mechanism-v1',r['policy'],r['task_id'],r['repeat'],decision])[:16]
            states.append(dict(id=state_id,group=group,source_policy=r['policy'],
                source_task=r['task'],source_repeat=r['repeat'],source_outcome=r['outcome'],
                source_trace=r['source'],source_trace_sha256=digest,decision=decision,
                header=header,pre_state=row['pre_state'],provenance=row['observation_provenance'],
                expected_full_action=row['full_action_16x11'],source_action_seed=row['action_seed'],
                prefix_actions=[x['policy_action_8x11'] for x in objects[1:decision+1]],
                labels=c))
            used_tasks.add(r['task_id'])
            if len(used_tasks)==6:
                break
        if len(used_tasks)!=6:
            raise ValueError('fewer than six distinct tasks for '+group)
    bank = dict(schema='routea_a1_mechanism_v1',date='2026-10-21',states=states,
        selection='A1 outcome-stratified diagnosis: 6 distinct tasks in each failure/positive-control group; fixed content-hash order',
        source_behavior_sha256=file_hash(behavior_path),draws=20,
        continuation_draws=[0,1,2],max_h4_cycles=12)
    bank['content_hash']=canonical_hash(bank)
    with Path(output_path).open('x',encoding='utf-8') as stream:
        json.dump(bank,stream,ensure_ascii=False,sort_keys=True,indent=2)
    return bank


def replay_state(system, config, task_bank, state):
    from .runtime import make_adapter
    from scripts.evaluate_dppo_uav_formal_endpoint import reset_fixed_episode
    header = state['header']
    if header['bank_hash']!=task_bank['content_hash']:
        raise ValueError('recovery/endpoint task banks differ')
    world = task_bank['worlds'][header['world_id']]
    if world['obstacle_sha256']!=header['obstacle_sha256']:
        raise ValueError('recovery geometry differs')
    adapter = make_adapter(system, config, world, header['role'])
    obs,ctx = reset_fixed_episode(adapter.environment, header['task'],
                                  seed=header['reset_seed'], reset_callable=adapter.reset)
    for action in state['prefix_actions']:
        obs,_,terminated,truncated,ctx = adapter.step(SimpleNamespace(policy=np.asarray(action,np.float32)))
        if terminated or truncated:
            raise ValueError('replay terminated before frozen anchor')
    encoded = system.observation_adapter.encode(obs,ctx['pointcloud_meta_history'],
        episode_identity_hash=ctx['episode_identity_hash'],decision_index=ctx['decision_index'])
    system.observation_adapter.verify(encoded)
    check_replayed_state(adapter.environment,state['pre_state'],asdict(encoded.provenance),state['provenance'])
    if adapter.environment.episode_step!=state['decision']:
        raise ValueError('replay changed remaining task budget')
    return adapter,obs,ctx


def action_record(bundle, env, goal):
    from diffusion_policy_3d.env.traj_utils import action_to_world
    from diffusion_policy_3d.env.reward_v2_support.first_event import locate_first_event
    from scripts.analyze_routea_trajectory_behavior import plan_features
    value = bundle.full
    full = np.asarray(value.detach().cpu() if hasattr(value,'detach') else value,np.float32).reshape(16,11)
    world = action_to_world(full[2:],env._drone_pos,env._drone_vel,env._drone_yaw)
    path = np.vstack((env._drone_pos,world[:,:3]))
    facts = plan_features(path,goal,world[:,9],env._drone_vel,env._drone_yaw)
    for horizon in (4,8):
        event = locate_first_event(path[:horizon+1],goal,env.obstacles,.1,env.z_max,
                                   terminate_at_space_limits=False)
        facts['h%d_first_event'%horizon] = None if event is None else dict(
            kind=event.kind,segment_index=event.segment_index,fraction=event.fraction)
    return dict(full_action_16x11=full.tolist(),planned_future_path=path.tolist(),metrics=facts)


def continue_flight(policy, snapshot, state_id, draw, first_sample, maximum=12, cycle_start=0):
    adapter,obs,ctx = clone_snapshot(snapshot)
    env = adapter.environment
    goal = env.goal_w.copy()
    distance = float(np.linalg.norm(goal-env._drone_pos))
    rows, outcome = [], 'window_end'
    for cycle in range(cycle_start, cycle_start + continuation_limit(adapter, maximum)):
        seed = paired_seed(state_id,draw,cycle)
        if cycle==0 and first_sample is not None:
            full = np.asarray(first_sample['full_action_16x11'],np.float32)
            bundle = SimpleNamespace(full=full,policy=full[2:10])
            noise_hash=first_sample['initial_noise_hash']
            provenance=first_sample['observation_provenance']
        else:
            bundle,noise_hash = predict_with_noise(policy,obs,ctx,seed)
            provenance=asdict(policy.last_provenance)
        row = action_record(bundle,env,goal)
        row.update(cycle=cycle,decision=env.episode_step,seed=seed,
                   initial_noise_hash=noise_hash,observation_provenance=provenance,
                   source_sample_draw=draw if cycle==0 else None,
                   pointcloud_metadata_before=[asdict(m) for m in ctx['pointcloud_meta_history']],
                   pre_state=dict(position=env._drone_pos.tolist(),velocity=env._drone_vel.tolist(),yaw=float(env._drone_yaw)))
        obs,reward,terminated,truncated,ctx = adapter.step(bundle)
        row.update(executed_path=np.asarray(ctx['executed_path']).tolist(),
                   done_reason=ctx['done_reason'],executed_time_s=ctx['executed_time_s'],
                   reward=float(reward))
        rows.append(row)
        if terminated or truncated:
            outcome=ctx['done_reason']
            break
    return dict(draw=draw,outcome=outcome,initial_distance=distance,
                final_distance=float(np.linalg.norm(goal-env._drone_pos)),
                net_progress=distance-float(np.linalg.norm(goal-env._drone_pos)),
                remaining_decisions=env.max_episode_steps-env.episode_step,rows=rows)


def run_recovery(root, cat, task_bank, frozen_path, out, full_remaining=False):
    from .runtime import common_system,load_actor
    from .workflow import save,policy_order
    from scripts.evaluate_dppo_uav_formal_endpoint import _E5Policy
    import time
    bank, out = read_bank(frozen_path),Path(out)
    if full_remaining:
        from .paper import validate_models
        validate_models(cat)
    maximum = None if full_remaining else bank['max_h4_cycles']
    remaining = [60-s['decision'] for s in bank['states']] if full_remaining else None
    counts = recovery_counts(len(bank['states']),len(cat['policies']),bank['draws'],
                             len(bank['continuation_draws']),remaining=remaining,maximum=bank['max_h4_cycles'])
    out.mkdir(parents=True,exist_ok=True)
    save(out/'protocol.json',dict(schema='same_state_recovery_v1',bank_hash=bank['content_hash'],
        catalogue_hash=cat['content_hash'],draws=bank['draws'],continuation_draws=bank['continuation_draws'],
        maximum_h4_cycles=maximum,task_budget='original remaining decisions',
        main_endpoint=full_remaining,counts=counts))
    save(out/'frozen_bank.json',bank)
    system,config = common_system(root,cat)
    # Rebuild once per state, with original actions and observation RNG; clones
    # retain reward history, sensor history, task clock and environment RNG.
    snapshots = {}
    started = time.monotonic()
    for state in bank['states']:
        snapshot = replay_state(system,config,task_bank,state)
        snapshots[state['id']] = snapshot
        path = out/'observations'/(state['id']+'.npz')
        if not path.exists():write_npz(path,snapshot[1])
    all_summaries = []
    for name in policy_order(cat):
        descriptor=cat['policies'][name]
        if name!='bc':
            load_actor(root,system,descriptor)
            anchor=bank['states'][0]
            _,obs,ctx=snapshots[anchor['id']]
            base=_E5Policy(system,use_base_policy=True,identity_hash=cat['common']['bc_identity_hash'])
            bundle,noise=predict_with_noise(base,obs,ctx,paired_seed(anchor['id'],0))
            expected=read_json(out/'bc'/(anchor['id']+'.json'))['samples'][0]
            actual=np.asarray(bundle.full.detach().cpu(),np.float32).reshape(16,11)
            if (not np.array_equal(actual,np.asarray(expected['full_action_16x11'],np.float32))
                    or noise!=expected['initial_noise_hash']):
                raise ValueError('BC branch changed after loading '+name)
            save(out/name/'bc_disabled_branch_check.json',dict(status='pass',
                actual_use_base_policy=True,state_sha256=descriptor['state_sha256']))
        identity=cat['common']['bc_identity_hash'] if name=='bc' else descriptor['runtime_identity_hash']
        policy=_E5Policy(system,use_base_policy=descriptor['use_base_policy'],identity_hash=identity)
        for state in bank['states']:
            path=out/name/(state['id']+'.json')
            if path.exists():
                result=read_json(path)
            else:
                adapter,obs,ctx=snapshots[state['id']]
                samples=[]
                for draw in range(bank['draws']):
                    seed=paired_seed(state['id'],draw)
                    bundle,noise_hash=predict_with_noise(policy,obs,ctx,seed)
                    row=action_record(bundle,adapter.environment,adapter.environment.goal_w)
                    row.update(draw=draw,seed=seed,policy_input_hash=policy.last_provenance.policy_input_hash,
                               initial_noise_hash=noise_hash,observation_provenance=asdict(policy.last_provenance))
                    samples.append(row)
                flights=[continue_flight(policy,snapshots[state['id']],state['id'],i,samples[i],maximum)
                         for i in bank['continuation_draws']]
                result=dict(policy=name,actual_use_base_policy=policy.use_base_policy,
                    policy_identity=identity,state_id=state['id'],samples=samples,continuations=flights)
                write_json(path,result)
            if name!='bc':verify_pairing(read_json(out/'bc'/(state['id']+'.json'))['samples'],result['samples'])
            if (len(result['samples'])!=bank['draws'] or len(result['continuations'])!=len(bank['continuation_draws'])
                    or result['state_id']!=state['id'] or result['policy']!=name):
                raise ValueError('incomplete or mismatched recovery record')
            if result['policy_identity']!=identity or result['actual_use_base_policy']!=policy.use_base_policy:
                raise ValueError('saved recovery result belongs to a different actor branch')
            if full_remaining and any(f['outcome']=='window_end' or
                    len(f['rows'])>60-state['decision'] for f in result['continuations']):
                raise ValueError('paper continuation must terminate within original task budget')
            all_summaries.append(dict(policy=name,state_id=state['id'],group=state['group'],
                mean_h4_progress=float(np.mean([s['metrics']['h4_progress'] for s in result['samples']])),
                continuation_outcomes=[f['outcome'] for f in result['continuations']],
                continuation_progress=[f['net_progress'] for f in result['continuations']],
                continuation_scope='original_remaining_deadline' if full_remaining else '12_cycle_window'))
            if not full_remaining:
                all_summaries[-1]['short_window_outcomes']=all_summaries[-1]['continuation_outcomes']
                all_summaries[-1]['short_window_progress']=all_summaries[-1]['continuation_progress']
            print('RECOVERY',name,state['id'],'seconds',round(time.monotonic()-started,1),flush=True)
    summary=dict(status='complete',counts=counts,
                 elapsed_seconds=time.monotonic()-started,records=all_summaries)
    # Timings are per invocation; do not bind resume to a prior duration.
    if not (out/'summary.json').exists():write_json(out/'summary.json',summary)
    return {k:v for k,v in summary.items() if k!='records'}


def analyze_new_evaluation(task_bank, out):
    from scripts.analyze_routea_trajectory_behavior import Scene,episode,summarize,replanning_episode
    from .workflow import save
    out=Path(out)
    records=[]; replans=[]
    numbers={t['task_id']:i+1 for i,t in enumerate(task_bank['tasks'])}
    for job in sorted((out/'full').iterdir()):
        if not (job/'commits').is_dir():continue
        for commit in sorted((job/'commits').glob('*.json')):
            receipt=read_json(commit)
            trace=job/receipt['trace']
            objects=[json.loads(s) for s in trace.read_text(encoding='utf-8').splitlines()]
            h,rows,end=objects[0],objects[1:-1],objects[-1]
            r=episode(h,rows,end,Scene(h['geometry']),numbers[h['task_id']])
            r['source']=str(trace)
            q=replanning_episode(r)
            records.append(r)
            replans.append({k:v for k,v in q.items() if k!='cycles'})
    expected=len(task_bank['episodes'])*len([p for p in (out/'full').iterdir() if (p/'commits').is_dir()])
    if len(records)!=expected:raise ValueError('behavior analysis denominator mismatch')
    result=dict(n=len(records),cycles=sum(r['decisions'] for r in records),
        summary=summarize(records),records=records,replanning=replans,
        task_numbering='one-based order in frozen task bank; task_id is authoritative')
    destination=out/'behavior.json'
    if not destination.exists():write_json(destination,result)
    return dict(status='complete',episodes=len(records),cycles=result['cycles'])


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Prepare frozen inputs; policy evaluation stays in run_routea_paired_comparison.py')
    parser.add_argument('--freeze-behavior',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    bank=freeze_bank(args.freeze_behavior,args.out)
    print('FROZEN',len(bank['states']),bank['content_hash'])
