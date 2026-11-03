#!/usr/bin/env python3
"""Independent test12 and X/K diagnostics, outside historical source identities."""
import argparse
from collections import Counter
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def test12_contract():
    return dict(worlds=12, tasks=48, repeats=5, policies=['bc','u128','C','T'], episodes=960)


def select_c2(rows):
    counts = {}
    for name in ('baseline','z03','z06','clip2'):
        counts[name] = {}
        for group in ('V','P'):
            values = [r for r in rows if r['variant']==name and r['group']==group]
            if not values:
                raise ValueError('incomplete C2 matrix: '+name+'/'+group)
            c = Counter(r['outcome'] for r in values)
            counts[name][group] = dict(c, n=len(values),
                arrival_rate=c['reached']/len(values), collision_rate=c['collision']/len(values))
    for group in ('V','P'):
        if len({counts[n][group]['n'] for n in counts})!=1:
            raise ValueError('unequal paired C2 denominator')
    baseline = counts['baseline']
    eligible = [name for name in ('z03','z06','clip2') if
        counts[name]['P']['collision_rate'] <= baseline['P']['collision_rate']+.02+1e-12 and
        counts[name]['P']['arrival_rate'] >= baseline['P']['arrival_rate']-.05-1e-12 and
        counts[name]['V']['arrival_rate'] >= baseline['V']['arrival_rate']+.05-1e-12]
    eligible.sort(key=lambda n: (-counts[n]['V']['arrival_rate'], n=='clip2', ('z03','z06','clip2').index(n)))
    return dict(selected=eligible[0] if eligible else None, counts=counts,
        denominator='all fixed continuations, including window_end',
        scope='conditioned development diagnostic, not population SR or factor causality')


def choose_c2_anchors(candidates):
    selected=[]
    used=set()
    positive_tasks=set()
    for group,number in [('V',40),('P_narrow',10),('P_approach',10)]:
        tasks=Counter()
        pool=sorted((c for c in candidates if c['group']==group),
            key=lambda c:hashlib.sha256(('xk-c2-v1/'+c['id']).encode()).hexdigest())
        for candidate in pool:
            if candidate['id'] in used or tasks[candidate['task_id']] >= (3 if group=='V' else 1):
                continue
            if group!='V' and candidate['task_id'] in positive_tasks:continue
            selected.append(candidate); used.add(candidate['id']); tasks[candidate['task_id']]+=1
            if group!='V':positive_tasks.add(candidate['task_id'])
            if sum(s['group']==group for s in selected)==number: break
        if sum(s['group']==group for s in selected)!=number:
            raise ValueError('insufficient fixed C2 candidates: '+group)
    return selected


def closest_path_point(path,goal):
    import numpy as np
    path=np.asarray(path,dtype=np.float64);goal=np.asarray(goal,dtype=np.float64)
    a=path[:-1];delta=path[1:]-a
    lengths=np.sum(delta*delta,axis=1)
    fraction=np.divide(np.sum((goal-a)*delta,axis=1),lengths,
        out=np.zeros(len(lengths)),where=lengths>0).clip(0,1)
    points=a+fraction[:,None]*delta
    return points[np.argmin(np.linalg.norm(points-goal,axis=1))]


def read_accepted_trace(job,commit):
    """Read already accepted evidence without repeating the full physical audit."""
    receipt=read(commit);trace=Path(job)/receipt['trace']
    if digest(trace)!=receipt['sha256']:raise ValueError('accepted trace bytes changed')
    objects=[json.loads(s) for s in trace.read_text(encoding='utf-8').splitlines()]
    if objects[0]['type']!='episode_start' or objects[-1]!=receipt['summary']:
        raise ValueError('accepted trace/commit mismatch')
    return objects[0],objects[1:-1],objects[-1]


def freeze_c2(args):
    """Outcome-conditioned bank from existing A2 raw evidence, before variant sampling."""
    sys.path.insert(0,str(Path(args.repo).resolve()))
    import numpy as np
    from scripts.routea_paired_support.core import canonical_hash
    from diffusion_policy_3d.env.collision_geom import ObstacleSpec,segment_obstacle_collision
    dev=Path(args.dev); candidates=[]
    policies=[args.u128_policy,args.control_policy,args.treatment_policy]
    for policy in policies:
        job=dev/'full'/policy
        for commit in sorted((job/'commits').glob('*.json')):
            header,rows,end=read_accepted_trace(job,commit)
            if end['done_reason'] not in ('time_limit','reached'): continue
            goal=np.asarray(header['task']['goal_pos'])
            points=np.concatenate([np.asarray(r['executed_path']) for r in rows])
            nearest=closest_path_point(points,goal)
            vertical=(end['done_reason']=='time_limit' and np.linalg.norm(nearest-goal)<1 and goal[2]-nearest[2]>.3)
            if not vertical and (policy!=args.control_policy or end['done_reason']!='reached'):continue
            obstacles=[ObstacleSpec(**o) for o in header['geometry']]
            trace=job/read(commit)['trace']
            for row in rows:
                position=np.asarray(row['pre_state']['position'])
                distance=float(np.linalg.norm(position-goal))
                if distance<=.5:continue
                if vertical:
                    if distance>=2 or goal[2]-position[2]<=.3:continue
                    group='V'
                else:
                    path=np.asarray(row['executed_path'])
                    narrow=any(segment_obstacle_collision(a,b,obstacles,.6) is not None for a,b in zip(path[:-1],path[1:]))
                    group='P_narrow' if narrow else 'P_approach'
                    if not narrow and distance>=2:continue
                decision=row['decision']
                sid=canonical_hash(['xk-c2-v1',policy,header['task_id'],header['repeat'],decision])[:20]
                candidates.append(dict(id=sid,group=group,task_id=header['task_id'],
                    source_policy=policy,source_task=header['task_id'],source_repeat=header['repeat'],
                    source_outcome=end['done_reason'],source_trace=str(trace.resolve()),source_trace_sha256=digest(trace),
                    decision=decision,header=header,pre_state=row['pre_state'],provenance=row['observation_provenance'],
                    expected_full_action=row['full_action_16x11'],source_action_seed=row['action_seed'],
                    prefix_actions=[r['policy_action_8x11'] for r in rows[:decision]],
                    selection=dict(distance_m=distance,goal_above_m=float(goal[2]-position[2]),
                        episode_closest_m=float(np.linalg.norm(nearest-goal)),
                        continuous_gap_criterion='existing segment_obstacle_collision threshold 0.6m')))
    bank=dict(schema='routea_xk_c2_v1',states=choose_c2_anchors(candidates),draws=5,max_h4_cycles=12,
        selection='40 V states task cap3; 10 distinct narrow P and 10 ordinary approach P; hash order',
        source_dev=str(dev.resolve()),policies=policies,candidate_counts=dict(Counter(c['group'] for c in candidates)))
    bank['content_hash']=canonical_hash(bank)
    write(args.output,bank)


def c2(args):
    system,config,header=native_system(args)
    import numpy as np
    import torch
    from dataclasses import asdict
    from scripts.evaluate_dppo_uav_formal_endpoint import _E5Policy
    from scripts.routea_paired_support.core import canonical_hash
    from scripts.routea_paired_support.recovery import replay_state,predict_with_noise,action_record,continue_flight,paired_seed
    bank=read(args.bank); tasks=read(args.task_bank)
    if canonical_hash({k:v for k,v in bank.items() if k!='content_hash'})!=bank['content_hash']:
        raise ValueError('C2 frozen bank differs')
    if Counter(s['group'] for s in bank['states'])!={'V':40,'P_narrow':10,'P_approach':10}:
        raise ValueError('C2 state counts differ')
    out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
    policy=_E5Policy(system,use_base_policy=False,identity_hash=header['runtime_identity_hash'])
    snapshots={s['id']:replay_state(system,config,tasks,s) for s in bank['states']}
    variants=[('baseline',.1,1.),('z03',.3,1.),('z06',.6,1.),('clip2',.1,2.)]
    override=load_override(args.current_root)
    records=[]; timing=[]; draws=5
    write(out/'frozen_bank.json',bank)
    for name,zfloor,clip in variants:
        started=time.monotonic(); floors=[.1]*11;floors[2]=zfloor
        with override(system.official_runtime,min_std_vector=floors,denoised_clip_value=clip,sampling_mode='random') as protocol, torch.jit.optimized_execution(False):
            write(out/name/'protocol.json',dict(training_identity=header['runtime_identity_hash'],
                checkpoint_state_sha256=digest(Path(args.checkpoint)/'state.pt'),evaluation_sampling_protocol=protocol,
                bank_hash=bank['content_hash'],paired_seed_rule='recovery-v1/state/draw/cycle',
                original_remaining_deadline=True,max_cycles=12,draw_prefix=draws))
            for state in bank['states']:
                adapter,obs,ctx=snapshots[state['id']]
                for draw in range(draws):
                    seed=paired_seed(state['id'],draw)
                    bundle,noise=predict_with_noise(policy,obs,ctx,seed)
                    first=action_record(bundle,adapter.environment,adapter.environment.goal_w)
                    first.update(draw=draw,seed=seed,initial_noise_hash=noise,observation_provenance=asdict(policy.last_provenance))
                    input_hash=policy.last_provenance.policy_input_hash
                    flight=continue_flight(policy,snapshots[state['id']],state['id'],draw,first,12)
                    executed_z=[]; generated_z=[]
                    for row in flight['rows']:
                        full=torch.tensor(row['full_action_16x11'],dtype=torch.float32,device=system.bc_runtime.device)
                        normalized=system.bc_runtime.policy.normalizer['action'].normalize(full).detach().cpu().numpy()
                        generated_z.extend(normalized[2:6,2].tolist())
                        # Actual partial terminal segment is normalized after scaling the physical delta.
                        for index,delta in enumerate(np.diff(np.asarray(row['executed_path']),axis=0)):
                            value=full[2+index].clone();value[:3]=torch.as_tensor(delta,dtype=value.dtype,device=value.device)
                            executed_z.append(float(system.bc_runtime.policy.normalizer['action'].normalize(value)[2].cpu()))
                    flight.update(state_id=state['id'],group=state['group'],variant=name,
                        normalized_executed_delta_z=executed_z,normalized_generated_h4_delta_z=generated_z,
                        actual_use_base_policy=False,training_identity=header['runtime_identity_hash'],
                        initial_noise_hash=noise,policy_input_hash=input_hash)
                    write(out/name/(state['id']+'_draw'+str(draw)+'.json'),flight)
                    records.append(dict(variant=name,state_id=state['id'],draw=draw,
                        group='V' if state['group']=='V' else 'P',outcome=flight['outcome'],
                        normalized_executed_delta_z=executed_z,initial_noise_hash=noise))
            seconds=time.monotonic()-started
            timing.append(dict(variant=name,seconds=seconds,draws=draws))
        if name=='baseline' and seconds*4>7200:
            draws=3
            # All outcomes remain raw evidence; selection uses predeclared prefix0:3 for ALL variants.
        print('C2_VARIANT',name,seconds,'selection_draw_prefix',draws,flush=True)
    selected=[r for r in records if r['draw']<draws]
    index={(r['variant'],r['state_id'],r['draw']):r for r in selected}
    for r in selected:
        if r['initial_noise_hash']!=index[('baseline',r['state_id'],r['draw'])]['initial_noise_hash']:
            raise ValueError('C2 initial noise pairing differs')
    result=select_c2(selected)
    chosen=result['selected']
    floors=[.1]*11
    if chosen in ('z03','z06'):floors[2]=.3 if chosen=='z03' else .6
    result['selected_protocol']=(None if chosen is None else
        dict(min_std_vector=floors,denoised_clip_value=2. if chosen=='clip2' else 1.))
    for name,_,_ in variants:
        values=np.asarray([v for r in selected if r['variant']==name for v in r['normalized_executed_delta_z']])
        result['counts'][name]['executed_z']=dict(n=len(values),quantiles=np.quantile(values,[0,.01,.5,.99,1]).tolist(),
            abs_gt1_rate=float(np.mean(np.abs(values)>1)))
    result.update(draws=draws,continuations=len(selected),raw_continuations=len(records),timing=timing,
        timing_rule='baseline full5 projected total >7200s => all variants selection uses draw prefix0:3')
    write(out/'selection.json',result)


def snapshot_source(source, target, vendor_fallback=None):
    """Native source bytes; relative BC/data inputs still reference live verified assets."""
    source, target = Path(source).resolve(), Path(target).resolve()
    target.mkdir(parents=True, exist_ok=False)
    for directory in ('diffusion_policy_3d','scripts','third_party'):
        if (source/directory).exists():
            shutil.copytree(source/directory, target/directory,
                ignore=shutil.ignore_patterns('__pycache__','*.pyc','.venv*'))
    # These three small behavior inputs are members of _source_rows, despite
    # living under data. The model weights remain only at their original path.
    for name in ('fixture_manifest_v2.json','frozen_density_v1.json','calibration_report_v2.json'):
        original=source/'data/reward_v17'/name
        if original.exists():
            dest=target/'data/reward_v17'/name
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(original,dest)
    if vendor_fallback is not None:
        # A1's receipt saved its uav overlay, not every unchanged upstream file.
        for path in Path(vendor_fallback).rglob('*'):
            relative = path.relative_to(vendor_fallback)
            dest = target/'third_party'/relative
            if path.is_file() and not dest.exists() and '__pycache__' not in path.parts:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path,dest)
    return target


def verify_source(source, manifest):
    for row in manifest['files']:
        if digest(Path(source)/row['relative_path'])!=row['sha256']:
            raise ValueError('native source mismatch: '+row['relative_path'])


def native_system(args):
    """Full restore under the actual native source; no identity bypass or actor-only substitute."""
    source = Path(args.source_root).resolve()
    sys.path.insert(0,str(source))
    import torch
    from omegaconf import OmegaConf
    import importlib
    identity=importlib.import_module('diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.identity')
    builder=importlib.import_module('diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.build_system')
    Path(builder.__file__).resolve().relative_to(source)
    # Assets remain under their original repository paths. Hash the actual
    # imported native code and load the actual native vendor; never reuse a
    # claimed old manifest over new implementations.
    source_rows=identity._source_rows
    identity._source_rows=lambda unused:source_rows(source)
    original_symbols=builder.load_official_symbols
    builder.load_official_symbols=lambda unused:original_symbols(source/'third_party/dppo_official_cc7234a')
    from diffusion_policy_3d.rl.dppo_uav_rebuild.deployment.loader import load_deployment_policy
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.identity import _canonical_hash
    state = torch.load(Path(args.checkpoint)/'state.pt', map_location='cpu', weights_only=True)
    header = read(Path(args.checkpoint)/'header.json')
    if _canonical_hash(state['training_config'])!=header['runtime_identity']['config_hash']:
        raise ValueError('saved config identity differs')
    config = OmegaConf.create(state['training_config'])
    deployment = load_deployment_policy(config,Path(args.checkpoint),Path(args.repo).resolve())
    if deployment.checkpoint_header.update_index!=args.expected_update:
        raise ValueError('wrong checkpoint update')
    deployment.system.official_runtime.eval()
    deployment.system.bc_runtime.policy.eval()
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    return deployment.system,config,header


def prepare_fixed_bank(system, config, source, split, world_root, output, repeats=5):
    from dataclasses import asdict
    import numpy as np
    from scripts import evaluate_dppo_uav_formal_endpoint as endpoint
    from scripts.routea_paired_support.core import canonical_hash, observation_hash, write_npz
    from scripts.routea_paired_support.bank import episode_key
    from scripts.routea_paired_support.runtime import make_adapter, metadata
    from diffusion_policy_3d.rl.dppo_uav.integration import manifest_eval_entries
    endpoint.BENCHMARK_SPLITS['xk-fixed']=(split,)
    fixed=endpoint._load_fixed_bank(Path(source)/'data/eval_configs','xk-fixed')
    endpoint._validate_bank_sources(fixed,manifest_eval_entries(str(config.admission_manifest)))
    out=Path(output)
    bank=dict(schema='routea_fixed_split_v1', split=split, repeats=repeats, worlds={},tasks=[],episodes=[])
    for _,uid,item in fixed:
        role=item.payload['role']
        if uid not in bank['worlds']:
            obstacle=endpoint._resolve_obstacle(Path(world_root),split,uid)
            copied=out/'obstacles'/(uid+'.json')
            copied.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(obstacle,copied)
            bank['worlds'][uid]=dict(world_id=uid,role=role,obstacle_path=str(copied.resolve()),
                obstacle_copy=copied.relative_to(out).as_posix(),obstacle_sha256=digest(copied))
        world=bank['worlds'][uid]
        adapter=make_adapter(system,config,world,role)
        adapter.environment._set_world(uid,world['obstacle_path'])
        world.update(geometry=[asdict(o) for o in adapter.environment.obstacles],
                     spawn_offset=adapter.environment._spawn_offset.tolist())
        task=dict(task_id=item.config_hash,world_id=uid,role=role,payload=item.payload)
        bank['tasks'].append(task)
        for repeat in range(1,repeats+1):
            seed=(item.seed if repeat==1 else int(canonical_hash(['xk-fixed-reset-v1',item.config_hash,repeat])[:8],16))
            obs,ctx=endpoint.reset_fixed_episode(adapter.environment,item.payload,seed=seed,reset_callable=adapter.reset)
            obs={k:np.asarray(v,dtype=np.float32) for k,v in obs.items()}
            key=episode_key(item.config_hash,repeat)
            npz=out/'initial'/(key+'.npz')
            write_npz(npz,obs)
            bank['episodes'].append(dict(key=key,task_id=item.config_hash,repeat=repeat,reset_seed=seed,
                observation_hash=observation_hash(obs),initial_hash=canonical_hash([item.payload,seed,observation_hash(obs)]),
                initial_npz=npz.relative_to(out).as_posix(),initial_npz_sha256=digest(npz),initial_pointcloud_metadata=metadata(ctx)))
        adapter.environment.close()
    bank['content_hash']=canonical_hash(bank)
    write(out/'bank.json',bank)
    return bank


def load_override(current_root):
    path=Path(current_root)/'third_party/dppo_official_cc7234a/uav_idp3/e5_runtime.py'
    spec=importlib.util.spec_from_file_location('xk_current_e5_protocol',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.sampling_protocol_override


def worker(args):
    import torch
    system,config,header=native_system(args)
    from scripts.evaluate_dppo_uav_formal_endpoint import _E5Policy
    from scripts.routea_paired_support.core import canonical_hash
    from scripts.routea_paired_support.episodes import record_episode,load_commit
    from scripts.routea_paired_support.runtime import make_adapter
    out=Path(args.output)
    bank_path=Path(args.bank)
    bank=(read(bank_path) if bank_path.exists() else prepare_fixed_bank(
        system,config,Path(args.repo).resolve(),args.split,args.worlds,bank_path.parent,args.repeats))
    if canonical_hash({k:v for k,v in bank.items() if k!='content_hash'})!=bank['content_hash']:
        raise ValueError('fixed bank content changed')
    if bank.get('schema')=='routea_fixed_split_v1' and bank.get('repeats')==1:
        from scripts import evaluate_dppo_uav_formal_endpoint as endpoint
        from scripts.routea_paired_support import episodes
        episodes.action_seed=lambda task_id,repeat,decision:endpoint._action_seed(task_id,decision)
    policy=_E5Policy(system,use_base_policy=args.policy=='bc',identity_hash=(
        system.bc_runtime.identity.content_hash if args.policy=='bc' else header['runtime_identity_hash']))
    from contextlib import nullcontext
    scope=(nullcontext(dict(sampling_mode='deterministic',clip_protocol='native',min_std=.1))
        if args.clip is None and args.min_std is None else load_override(args.current_root)(
            system.official_runtime,denoised_clip_value=args.clip,min_std_vector=args.min_std,sampling_mode='deterministic'))
    with scope as protocol, torch.jit.optimized_execution(False):
        write(out/'policy_provenance.json',dict(policy=args.policy,actual_use_base_policy=policy.use_base_policy,
            training_runtime_identity=header['runtime_identity_hash'],checkpoint_state_sha256=digest(Path(args.checkpoint)/'state.pt'),
            native_source_root=str(Path(args.source_root).resolve()),evaluation_wrapper_sha256=digest(__file__),
            bank_hash=bank['content_hash'],evaluation_sampling_protocol=protocol,
            note='evaluation overrides applied only after strict native identity restore'))
        tasks={t['task_id']:t for t in bank['tasks']}
        for index,e in enumerate(bank['episodes'],1):
            t=tasks[e['task_id']]; world=bank['worlds'][t['world_id']]
            adapter=make_adapter(system,config,world,t['role'])
            try:
                record_episode(out,args.policy,policy,adapter,t,e,world,bank['content_hash'])
            finally:
                adapter.environment.close()
            print('FIXED',args.policy,index,len(bank['episodes']),flush=True)
    records=[]
    for e in bank['episodes']:
        _,_,summary=load_commit(out,out/'commits'/(e['key']+'.json'))
        records.append(summary)
    write(out/'summary.json',dict(episodes=len(records),records=records,outcomes=dict(Counter(r['done_reason'] for r in records))))


def test12(args):
    if sys.version_info[:3]!=(3,8,20):raise RuntimeError('use fixed idp3 Python3.8.20')
    root=Path(args.repo).resolve(); out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    # Run this command before copying any changed training/vendor code.
    sys.path.insert(0,str(root))
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.identity import _source_rows
    rows=_source_rows(root)
    source=snapshot_source(root,out/'native_a2')
    write(out/'native_a2_manifest.json',dict(files=rows))
    verify_source(source,dict(files=rows))
    receipt=read(args.parent_receipt)
    verify_source(receipt['source_root'],receipt['source_manifest'])
    a1=snapshot_source(receipt['source_root'],out/'native_a1',root/'third_party')
    verify_source(a1,receipt['source_manifest'])
    write(out/'request.json',dict(contract=test12_contract(),checkpoints=dict(u128=args.u128,C=args.control,T=args.treatment),
        parent_receipt_sha256=digest(args.parent_receipt),wrapper_sha256=digest(__file__),
        order=['bc','u128','C','T'],test12_results_must_not_change_X=True,
        scope='independent fixed rl_test geometry verification; lightweight kinematic'))
    for name,path,src,update in [('bc',args.control,source,256),('u128',args.u128,a1,128),
                                ('C',args.control,source,256),('T',args.treatment,source,256)]:
        command=[sys.executable,str(Path(__file__).resolve()),'worker','--source-root',str(src),
            '--repo',str(root),
            '--checkpoint',str(path),'--expected-update',str(update),'--policy',name,
            '--bank',str(out/'bank/bank.json'),'--split','rl_test','--worlds',args.worlds,'--output',str(out/'full'/name)]
        log=out/'execution'/(name+'.log'); log.parent.mkdir(exist_ok=True)
        write(log.with_suffix('.command.json'),dict(argv=command))
        with log.open('x',encoding='utf-8') as stream:
            result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        log.with_suffix('.rc').write_text(str(result.returncode)+'\n',encoding='utf-8')
        if result.returncode: raise RuntimeError('test12 failed: '+name+'; preserve all evidence')
    summaries={name:read(out/'full'/name/'summary.json') for name in test12_contract()['policies']}
    if any(v['episodes']!=240 for v in summaries.values()): raise ValueError('test12 denominator differs')
    bank=read(out/'bank/bank.json')
    if len(bank['worlds'])!=12 or len(bank['tasks'])!=48: raise ValueError('wrong test12 world/task count')
    write(out/'completion.json',dict(status='complete',episodes=960,
        outcomes={k:v['outcomes'] for k,v in summaries.items()},bank_hash=bank['content_hash']))
    rows=[r for result in summaries.values() for r in result['records']]
    write(out/'independent_statistics.json',dict(scope='fixed rl_test; results must not tune X',
        comparisons=[paired_cluster(rows,a,b) for a,b in [('u128','bc'),('C','bc'),('T','bc'),('T','C')]]))


def paired_cluster(rows, treatment='X', reference='K'):
    import numpy as np
    keyed={name:{(r['task_id'],r['repeat']):r for r in rows if r['policy']==name}
           for name in (treatment,reference)}
    if not keyed[treatment] or set(keyed[treatment])!=set(keyed[reference]):
        raise ValueError('incomplete E1 paired comparison')
    if any(len(keyed[n])!=sum(r['policy']==n for r in rows) for n in keyed):
        raise ValueError('duplicate E1 condition')
    worlds={}; changes=[]
    for key,a in sorted(keyed[treatment].items()):
        b=keyed[reference][key]
        if a['world_id']!=b['world_id']:raise ValueError('paired worlds differ')
        delta=int(a['safe_arrival'])-int(b['safe_arrival'])
        worlds.setdefault(a['world_id'],[]).append(delta)
        changes.append(dict(task_id=key[0],repeat=key[1],world_id=a['world_id'],
            delta=delta,treatment_outcome=a['done_reason'],reference_outcome=b['done_reason']))
    cluster_sum=np.asarray([sum(v) for _,v in sorted(worlds.items())],dtype=float)
    cluster_n=np.asarray([len(v) for _,v in sorted(worlds.items())],dtype=float)
    rng=np.random.default_rng(2026102601)
    sampled=rng.integers(0,len(worlds),size=(20000,len(worlds)))
    boot=cluster_sum[sampled].sum(axis=1)/cluster_n[sampled].sum(axis=1)
    return dict(treatment=treatment,reference=reference,pairs=len(changes),worlds=len(worlds),
        bootstrap_draws=20000,bootstrap_seed=2026102601,
        ci95_delta_pp=(np.quantile(boot,[.025,.975])*100).tolist(),
        delta_pp=100*sum(r['delta'] for r in changes)/len(changes),
        rescued=sum(r['delta']>0 for r in changes),lost=sum(r['delta']<0 for r in changes),
        conditions=changes)


def matrix(args):
    """E1 workers: explicit native source/checkpoint plan, same bank and inference protocol."""
    plan=read(args.plan); out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    names=[p['policy'] for p in plan['models']]
    if len(names)!=len(set(names)):raise ValueError('duplicate matrix policy')
    bank=Path(args.bank).resolve()
    write(out/'plan.json',dict(plan=plan,plan_sha256=digest(args.plan),evaluation_wrapper_sha256=digest(__file__)))
    for item in plan['models']:
        command=[sys.executable,str(Path(__file__).resolve()),'worker','--repo',args.repo,
            '--source-root',item['source_root'],'--checkpoint',item['checkpoint'],
            '--expected-update',str(item['update']),'--policy',item['policy'],
            '--bank',str(bank),'--split',args.split,'--worlds',args.worlds,
            '--output',str(out/'full'/item['policy']),'--current-root',args.repo,
            '--repeats','1',
            '--clip',str(plan.get('denoised_clip_value',1.)),
            '--min-std']+[str(v) for v in plan.get('min_std_vector',[.1]*11)]
        log=out/'execution'/(item['policy']+'.log');log.parent.mkdir(exist_ok=True)
        write(log.with_suffix('.command.json'),dict(argv=command))
        with log.open('x',encoding='utf-8') as stream:
            result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        log.with_suffix('.rc').write_text(str(result.returncode)+'\n',encoding='utf-8')
        if result.returncode:raise RuntimeError('matrix worker failed: '+item['policy'])
    rows=[r for name in names for r in read(out/'full'/name/'summary.json')['records']]
    write(out/'completion.json',dict(episodes=len(rows),policies=names,bank_hash=read(bank)['content_hash']))


def analyze_e1(args):
    sys.path.insert(0,str(Path(args.repo).resolve()))
    import numpy as np
    from scripts.routea_paired_support.episodes import load_commit
    from diffusion_policy_3d.env.collision_geom import ObstacleSpec,segment_obstacle_collision
    out=Path(args.evaluation); bank=read(args.bank)
    names=[p.name for p in (out/'full').iterdir() if (p/'summary.json').is_file()]
    expected={e['key'] for e in bank['episodes']}
    rows=[]
    for name in names:
        job=out/'full'/name
        if {p.stem for p in (job/'commits').glob('*.json')}!=expected:
            raise ValueError('E1 matrix incomplete or extra: '+name)
        verified=[]
        for entry in bank['episodes']:
            h,_,end=load_commit(job,job/'commits'/(entry['key']+'.json'))
            if (h['bank_hash']!=bank['content_hash'] or h['initial_hash']!=entry['initial_hash']
                    or h['task_id']!=entry['task_id'] or h['repeat']!=entry['repeat']
                    or h['policy']!=name or h['use_base_policy']!=(name=='bc')):
                raise ValueError('E1 raw condition/branch pairing differs')
            verified.append(end)
        if verified!=read(job/'summary.json')['records']:
            raise ValueError('E1 summary differs from verified raw traces')
        rows.extend(verified)
    primary=paired_cluster(rows,'X','K')
    tasks={t['task_id']:t for t in bank['tasks']}
    strata={name:set() for name in ('goal_lt3_blocked','goal_lt3_clear','goal_ge3','goal_ge3p5')}
    for tid,t in tasks.items():
        goal=t['payload']['goal_pos'];world=bank['worlds'][t['world_id']]
        obstacles=[ObstacleSpec(**o) for o in world['geometry']]
        if goal[2]>=3:strata['goal_ge3'].add(tid)
        if goal[2]>=3.5:strata['goal_ge3p5'].add(tid)
        if goal[2]<3:
            blocked=segment_obstacle_collision(t['payload']['start_pos'],goal,obstacles,.1) is not None
            strata['goal_lt3_blocked' if blocked else 'goal_lt3_clear'].add(tid)
    layers={k:(paired_cluster([r for r in rows if r['task_id'] in ids]) if ids else None) for k,ids in strata.items()}
    outcomes={n:dict(Counter(r['done_reason'] for r in rows if r['policy']==n)) for n in names}
    task_results={tid:{name:dict(successes=sum(r['safe_arrival'] for r in rows
        if r['task_id']==tid and r['policy']==name),
        repetitions=sum(r['task_id']==tid and r['policy']==name for r in rows))
        for name in names} for tid in tasks}
    retention={}; double_failure=[]
    indexed={n:{(r['task_id'],r['repeat']):r for r in rows if r['policy']==n} for n in names}
    for reference in ('bc','u128'):
        if reference not in indexed:continue
        good={k for k,r in indexed[reference].items() if r['safe_arrival']}
        retention[reference]={name:dict(kept=sum(indexed[name][k]['safe_arrival'] for k in good),
            lost=[list(k) for k in sorted(good) if not indexed[name][k]['safe_arrival']],denominator=len(good)) for name in ('K','X')}
    if {'bc','u128'}<=set(indexed):
        for k,r in indexed['bc'].items():
            if not r['safe_arrival'] and not indexed['u128'][k]['safe_arrival']:
                double_failure.append(dict(task_id=k[0],repeat=k[1],
                    K=indexed['K'][k]['safe_arrival'],X=indexed['X'][k]['safe_arrival']))
    below={}
    for name in names:
        cases=[]
        for commit in sorted((out/'full'/name/'commits').glob('*.json')):
            h,transitions,end=load_commit(out/'full'/name,commit)
            if end['done_reason']!='time_limit':continue
            goal=np.asarray(h['task']['goal_pos'])
            points=np.concatenate([np.asarray(r['executed_path']) for r in transitions])
            p=closest_path_point(points,goal)
            if np.linalg.norm(p-goal)<1 and goal[2]-p[2]>.3:
                cases.append(dict(task_id=h['task_id'],repeat=h['repeat']))
        below[name]=cases
    xcollision=outcomes['X'].get('collision',0); kcollision=outcomes['K'].get('collision',0)
    ci=primary['ci95_delta_pp']; net=primary['rescued']-primary['lost']
    verdict=('do_not_adopt' if net<0 or xcollision>kcollision+10 else
        'combined_effect_supported' if net>0 and ci[0]>0 and xcollision<=kcollision+5 else
        'difference_not_resolved')
    if args.trend_only:verdict='descriptive_trend_only_no_checkpoint_selection'
    monitoring=None
    if args.x_metrics or args.k_metrics:
        if not args.x_metrics or not args.k_metrics:raise ValueError('both monitoring files required')
        monitors={name:read(path)['drift_monitor'] for name,path in [('X',args.x_metrics),('K',args.k_metrics)]}
        if monitors['X']['bank_sha256']!=monitors['K']['bank_sha256']:
            raise ValueError('monitoring banks differ')
        monitoring=dict(values=monitors,deployment_interface_review_required=
            monitors['X']['cum_inert_norm']>1.5*monitors['K']['cum_inert_norm'])
    historical=None
    if args.historical_c:
        job=Path(args.historical_c); old=[]
        if {p.stem for p in (job/'commits').glob('*.json')}!=expected:
            raise ValueError('historical C denominator differs')
        for entry in bank['episodes']:
            h,_,end=load_commit(job,job/'commits'/(entry['key']+'.json'))
            if h['bank_hash']!=bank['content_hash'] or h['initial_hash']!=entry['initial_hash']:
                raise ValueError('historical C conditions differ')
            old.append(dict(end,policy='C'))
        historical=paired_cluster([r for r in rows if r['policy']=='K']+old,'K','C')
    write(args.output,dict(primary=primary,strata=layers,outcomes=outcomes,retention=retention,
        both_bc_u128_failed=double_failure,below_goal_timeouts=below,verdict=verdict,
        task_results=task_results,monitoring=monitoring,historical_K_minus_C=historical,
        scope='single X/K seed; dev/val evidence; no factor-causal claim',
        task_level_conditions_in_primary=True))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('test12')
    for key in ('repo','parent-receipt','u128','control','treatment','worlds','output'):
        a.add_argument('--'+key,required=True)
    a=sub.add_parser('worker')
    for key in ('repo','source-root','checkpoint','policy','bank','split','worlds','output'):
        a.add_argument('--'+key,required=True)
    a.add_argument('--expected-update',required=True,type=int)
    a.add_argument('--repeats',type=int,choices=(1,5),default=5)
    a.add_argument('--current-root')
    a.add_argument('--clip',type=float)
    a.add_argument('--min-std',type=float,nargs=11)
    a=sub.add_parser('freeze-c2')
    for key in ('repo','dev','u128-policy','control-policy','treatment-policy','output'):
        a.add_argument('--'+key,required=True)
    a=sub.add_parser('c2')
    for key in ('repo','source-root','current-root','checkpoint','bank','task-bank','output'):
        a.add_argument('--'+key,required=True)
    a.add_argument('--expected-update',type=int,default=256)
    a=sub.add_parser('matrix')
    for key in ('repo','plan','bank','worlds','output'):a.add_argument('--'+key,required=True)
    a.add_argument('--split',choices=('rl_val','rl_test'),default='rl_val')
    a=sub.add_parser('analyze-e1')
    for key in ('repo','evaluation','bank','output'):a.add_argument('--'+key,required=True)
    a.add_argument('--trend-only',action='store_true')
    for key in ('x-metrics','k-metrics','historical-c'):a.add_argument('--'+key)
    return p


def main(argv=None):
    import torch
    args=parser().parse_args(argv)
    # Establish the rendering mode before native loading, bank creation or replay.
    # A later context cannot undo fusion cached by an earlier scripted ray call.
    with torch.jit.optimized_execution(False):
        globals()[args.command.replace('-','_')](args)


if __name__=='__main__':
    main()
