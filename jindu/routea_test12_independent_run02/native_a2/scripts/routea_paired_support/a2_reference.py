"""Fixed training-task paired collection using the shared E5/environment adapters."""
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from .core import canonical_hash, file_hash, observation_hash, write_json, write_npz


def select_prefixes(n):
    return np.linspace(0,n-1,min(8,n)).round().astype(int).tolist() if n else []


def check_execution(full, before, info):
    path=np.vstack((before,np.asarray(before)+np.cumsum(full[2:6,:3],axis=0)))
    segment,fraction=info['event_segment'],info['event_fraction']
    if segment is None:
        expected=path
    else:
        if segment not in range(4) or fraction is None or not 0<=fraction<=1:
            raise ValueError('invalid physical prefix event')
        point=path[segment]+fraction*(path[segment+1]-path[segment])
        expected=np.vstack((path[:segment+1],point))
    actual=np.asarray(info['executed_path'])
    if actual.shape!=expected.shape or not np.allclose(actual,expected,atol=2e-5,rtol=0):
        raise ValueError('actual prefix differs from clean plan; no valid supervision')


def task_fingerprint(world, payload):
    return canonical_hash([world, payload['start_pos'],payload['goal_pos'],payload['start_yaw']])


def collect_reference(system, config, *, output, parent_receipt_sha256, exclude_bank, seed=20261023):
    from omegaconf import OmegaConf
    from scripts.evaluate_dppo_uav_formal_endpoint import _E5Policy, reset_fixed_episode
    from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.factory import RouteAEnvironmentFactory
    from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.selective_keep import executed_weights, paired_inventory
    out=Path(output); out.mkdir(parents=True,exist_ok=False)
    excluded=json.loads(Path(exclude_bank).read_text(encoding='utf-8'))
    forbidden={task_fingerprint(t['world_id'],t['payload']) for t in excluded['tasks']}
    factory=RouteAEnvironmentFactory(manifest_path=config.admission_manifest,worlds_train_dir=config.worlds_train,
        reward=system.reward,device=str(system.bc_runtime.device),base_seed=seed,
        max_episode_steps=60,disturb_enable=False,exec_noise_scale=0.,mid_state_init_ratio=0.,
        max_reset_attempts=int(config.environment.max_reset_attempts),
        reward_config=OmegaConf.to_container(config.reward,resolve=True),terminate_at_space_limits=False,
        space_cost_budget=float(config.task_protocol.space_cost_budget))
    tasks=sorted(factory.tasks,key=lambda t:t.world_id)
    if len(tasks)!=21 or len({t.world_id for t in tasks})!=21:
        raise ValueError('reference requires exactly the admitted 21 training worlds')
    frozen=[]; seen=set()
    for task in tasks:
        if not task.training_eligible or task.historical_eval_only:
            raise ValueError('nontraining world in reference collection')
        for repeat in range(6):
            reset_seed=(task.seed+repeat*1000003)%(2**32)
            adapter=factory.make(task); adapter.reset(seed=reset_seed); env=adapter.environment
            payload=dict(episode_id=repeat,world_name=task.world_id,role=task.role,
                start_pos=env._drone_pos.tolist(),goal_pos=env.goal_w.tolist(),start_yaw=float(env._drone_yaw),
                z_range=[float(env.z_min),float(env.z_max)],disturb_enable=False)
            fp=task_fingerprint(task.world_id,payload)
            if fp in forbidden or fp in seen: raise ValueError('reference task duplicates evaluation/another task')
            seen.add(fp)
            frozen.append(dict(task_id=fp,world_id=task.world_id,payload=payload,reset_seed=reset_seed,
                training_task_identity=task.identity_hash,obstacle_sha256=file_hash(task.obstacle_json)))
    write_json(out/'tasks.json',dict(schema='a2_training_tasks_v1',tasks=frozen,
        seed=seed,exclude_bank_sha256=file_hash(exclude_bank),parent_receipt_sha256=parent_receipt_sha256))
    system.official_runtime.eval(); system.bc_runtime.policy.eval()
    samples=[]; episodes=[]
    by_world={t.world_id:t for t in tasks}
    for item in frozen:
        initial_hash=None
        for source in ('bc','u128'):
            policy=_E5Policy(system,use_base_policy=source=='bc',identity_hash=parent_receipt_sha256)
            adapter=factory.make(by_world[item['world_id']]); env=adapter.environment
            obs,context=reset_fixed_episode(env,item['payload'],seed=item['reset_seed'],reset_callable=adapter.reset)
            current=observation_hash(obs)
            if initial_hash is not None and current!=initial_hash: raise ValueError('unpaired initial observations')
            initial_hash=current
            episode_id=item['task_id']+'_'+source
            trace_path=out/(episode_id+'.jsonl')
            candidates=[]; full_only=0; rows=[]
            with trace_path.open('x',encoding='utf-8') as stream:
                stream.write(json.dumps(dict(type='start',task=item,source=source,initial_hash=current,
                    geometry=[asdict(o) for o in env.obstacles],scope='training-origin reference'))+'\n')
                for decision in range(60):
                    captured={}
                    def capture(module,args,kwargs,result):
                        captured['condition']=kwargs['cond']['global_cond'].detach().cpu()[0].clone()
                        captured['action']=result.trajectories.detach().cpu()[0].clone()
                    hook=system.official_runtime.register_forward_hook(capture,with_kwargs=True)
                    before=env._drone_pos.copy()
                    raw_obs={k:np.asarray(v).copy() for k,v in obs.items()}
                    action_seed=int(canonical_hash(['a2-reference',item['task_id'],decision])[:8],16)
                    try: bundle=policy.predict(obs,context,action_seed=action_seed)
                    finally: hook.remove()
                    obs,reward,terminated,truncated,context=adapter.step(bundle)
                    full=bundle.full[0].detach().cpu().numpy()
                    check_execution(full,before,context)
                    reason=context['done_reason']
                    row=dict(type='transition',decision=decision,action_seed=action_seed,
                        before=before.tolist(),full_action=full.tolist(),done_reason=reason,
                        event_segment=context['event_segment'],event_fraction=context['event_fraction'],
                        executed_path=np.asarray(context['executed_path']).tolist(),reward=float(reward),
                        provenance=asdict(policy.last_provenance))
                    stream.write(json.dumps(row,allow_nan=False)+'\n'); stream.flush()
                    rows.append(row)
                    if reason in ('none','reached'):
                        w=executed_weights(reason,context['event_segment'],context['event_fraction'])
                        full_only+=int(float(w.sum())==16.)
                        if w.sum()>0:
                            candidates.append(dict(action=captured['action'],condition=captured['condition'],
                                weights=w,raw=raw_obs,decision=decision,terminal=reason=='reached'))
                    if terminated or truncated: break
                else: raise ValueError('reference episode did not terminate within budget')
                reached=reason=='reached'
                chosen=select_prefixes(len(candidates)) if reached else []
                summary=dict(task_id=item['task_id'],source=source,reached=reached,samples=len(chosen),
                    decisions=len(rows),available_prefixes=len(candidates) if reached else 0,
                    full_only_available=full_only if reached else 0,
                    terminal_included=bool(chosen and candidates[chosen[-1]]['terminal']),done_reason=reason)
                stream.write(json.dumps(dict(type='end',**summary))+'\n')
            episodes.append(summary)
            for index in chosen:
                candidate=candidates[index]
                name=episode_id+'_'+str(candidate['decision'])+'.npz'
                write_npz(out/'observations'/name,candidate.pop('raw'))
                samples.append(dict(task_id=item['task_id'],source=source,observation=name,
                    trace=trace_path.name,trace_sha256=file_hash(trace_path),
                    observation_sha256=file_hash(out/'observations'/name),**candidate))
            print(json.dumps(dict(event='a2_reference_episode',completed=len(episodes),**summary)),flush=True)
    inventory=paired_inventory(episodes)
    inventory.update(episodes=episodes,selected_prefixes=len(samples),tasks=len(frozen),
        auxiliary_draw_budget=81920,
        full_only_zero_success_episodes=sum(r['reached'] and not r['full_only_available'] for r in episodes),
        terminal_included=sum(r['terminal_included'] for r in episodes),
        coverage_warnings=[k+' has fewer than five independent available tasks' for k,v in inventory['available_tasks'].items() if v<5])
    groups={}
    for s in samples:
        groups.setdefault(s['task_id'],{}).setdefault(s['source'],0)
        groups[s['task_id']][s['source']]+=1
    repeats=[81920./len(groups)/len(groups[s['task_id']])/groups[s['task_id']][s['source']] for s in samples]
    inventory['expected_draws_per_prefix']=dict(min=min(repeats),median=float(np.median(repeats)),max=max(repeats)) if repeats else None
    write_json(out/'inventory.json',inventory)
    if not samples: raise ValueError('no successful reference data; inventory retained')
    bank=dict(schema='routea_a2_train_success_v1',parent_receipt_sha256=parent_receipt_sha256,
        actions=torch.stack([s['action'] for s in samples]),conditions=torch.stack([s['condition'] for s in samples]),
        weights=torch.stack([s['weights'] for s in samples]),task_ids=[s['task_id'] for s in samples],
        sources=[s['source'] for s in samples],categories=[inventory['categories'][s['task_id']] for s in samples],
        provenance=[{k:s[k] for k in ('trace','observation','trace_sha256','observation_sha256','decision','terminal')} for s in samples],
        inventory_sha256=file_hash(out/'inventory.json'),tasks_sha256=file_hash(out/'tasks.json'),
        collector_sha256=file_hash(__file__))
    torch.save(bank,out/'reference.pt')
    write_json(out/'reference_manifest.json',dict(sha256=file_hash(out/'reference.pt'),samples=len(samples),
        inventory_sha256=bank['inventory_sha256'],tasks_sha256=bank['tasks_sha256']))
    return inventory
