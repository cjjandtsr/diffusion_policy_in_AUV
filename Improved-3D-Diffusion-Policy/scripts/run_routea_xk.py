#!/usr/bin/env python3
"""Prepare and execute X/K using the canonical pipeline and explicit parent fork."""
import argparse
import copy
import json
from pathlib import Path
import sys


def stratified_indices(items, count=256):
    """Round-robin roles, evenly spaced within each frozen parent role."""
    groups = {}
    for i, item in enumerate(items): groups.setdefault(item['role'], []).append(i)
    if len(items) < count: raise ValueError('parent buffer has fewer than 256 conditions')
    roles = sorted(groups); allocated = dict.fromkeys(roles, 0)
    while sum(allocated.values()) < count:
        for role in roles:
            if allocated[role] < len(groups[role]) and sum(allocated.values()) < count:
                allocated[role] += 1
    return [indices[(j*len(indices))//allocated[role]]
            for role, indices in sorted(groups.items()) for j in range(allocated[role])]


def make_config(receipt, receipt_path, reference, bank, output, weights, arm, *, scale=1., clip=.1,
                sampling=None):
    from scripts.run_routea_a2 import digest
    c = copy.deepcopy(receipt['training_config'])
    c['run'].update(seed=202610261, checkpoint_every_updates=8,
        checkpoint_updates=sorted(set(c['run']['checkpoint_updates'])|{129}))
    c['output_dir'] = str(Path(weights).resolve()/arm)
    c['result_dir'] = str(Path(output).resolve()/arm)
    c['a2'] = dict(arm='C', parent_checkpoint=receipt['parent_checkpoint'],
        parent_receipt=str(Path(receipt_path).resolve()), parent_receipt_sha256=digest(receipt_path),
        parent_source_root=receipt['source_root'], reference_bank=str(Path(reference).resolve()),
        reference_sha256=digest(reference), keep_seed=2026102301)
    c['xk'] = dict(arm=arm, actor_lr_scale=scale if arm=='X' else 1.)
    d = c['dppo']
    d.update(ppo_inert_mask=dict(enabled=arm=='X',channels=[3,4,5,6,7,8,10],scope='final_step'),
        drift_monitor=dict(enabled=True,bank_path=str(Path(bank).resolve()),
                           bank_sha256=digest(bank),every_commits=5),
        drift_gate=dict(enabled=False),
        inert_anchor=dict(enabled=False,coefficient=0.,target_ratio=.05,batch_size=128,seed=202610262),
        self_imitation=dict(enabled=arm=='X',window_updates=8,capacity=2048,
                            target_ratio=.1,batch_size=32,seed=202610263),
        clip_ploss_coef=clip if arm=='X' else .1)
    if arm=='X' and sampling:
        d['min_std_vector'] = list(sampling['min_std_vector'])
        d['denoised_clip_value'] = float(sampling['denoised_clip_value'])
    return c


def prepare(args):
    import torch
    from omegaconf import OmegaConf
    from scripts.run_routea_a2 import digest, read, write, verify_saved_source
    receipt = read(args.receipt); verify_saved_source(receipt)
    parent = Path(receipt['parent_checkpoint'])/'state.pt'
    if digest(parent)!=receipt['state_sha256']: raise ValueError('parent state hash differs')
    reference = Path(args.reference)
    if digest(reference)!=read(reference.parent/'reference_manifest.json')['sha256']:
        raise ValueError('reference manifest differs')
    bank_ref = torch.load(reference,map_location='cpu',weights_only=True)
    if len(bank_ref['actions'])!=549: raise ValueError('expected existing 549-state A2 bank')
    state = torch.load(parent,map_location='cpu',weights_only=True)
    items = state['rollout_buffer']['items']; indices = stratified_indices(items)
    output = Path(args.output).resolve(); output.mkdir(parents=True,exist_ok=False)
    bank = output/'drift_probe_bank_v1.pt'
    torch.save(dict(schema='routea_drift_probe_v1',
        conditions={k:torch.stack([items[i]['condition'][k] for i in indices])
                    for k in ('state','global_cond')},
        seeds=[2026102700+i for i in range(256)], indices=indices,
        roles=[items[i]['role'] for i in indices],parent_state_sha256=digest(parent)),bank)
    # Original protocol is enforced by DriftMonitor when the physical parent baseline is sampled.
    c = make_config(receipt,args.receipt,reference,bank,output/'c1_run',args.weights,'K')
    c['xk']['c1_gradient_output'] = str(output/'c1')
    OmegaConf.save(OmegaConf.create(c), output/'C1.yaml')
    write(output/'inputs.json',dict(receipt=str(Path(args.receipt).resolve()),
        receipt_sha256=digest(args.receipt),reference=str(reference.resolve()),
        reference_sha256=digest(reference),bank=str(bank),bank_sha256=digest(bank),
        seed=202610261, test12_excluded_from_development=True))


def short_clip(x_rows, k_rows):
    import statistics
    def values(rows):
        selected = [r for r in rows if 129<=int(r['update_index'])<=136]
        if sorted(int(r['update_index']) for r in selected)!=list(range(129,137)):
            raise ValueError('short comparison needs each global update 129..136 exactly once')
        return [float(r['drift_monitor']['step_h4_rms_m']) for r in selected]
    x,k = values(x_rows),values(k_rows)
    return .05 if statistics.median(x)>2*statistics.median(k) else .1


def paired_short_gate(x_dir,k_dir):
    import torch
    from scripts.run_routea_a2 import read,digest,assert_equal_state
    from scripts.routea_paired_support.core import verify_checkpoint
    states=[]; metrics=[]; hashes={}
    for arm,directory in [('X',x_dir),('K',k_dir)]:
        path=Path(directory)/'metrics_update_000129.json'; metric=read(path)
        checkpoint=Path(metric['checkpoint']); header=read(checkpoint/'header.json')
        sums=read(checkpoint/'SHA256.json')
        verify_checkpoint(checkpoint,dict(runtime_identity_hash=header['runtime_identity_hash'],
            header_sha256=sums['header.json'],state_sha256=sums['state.pt'],update=129))
        state=torch.load(checkpoint/'state.pt',map_location='cpu',weights_only=True)
        if state['training_config']['xk']['arm']!=arm: raise ValueError('short arm mismatch')
        if state['training_config']['run']['seed']!=202610261: raise ValueError('short seed mismatch')
        states.append(state); metrics.append(metric)
        hashes[arm]=dict(state_sha256=digest(checkpoint/'state.pt'),
            metrics={p.name:digest(p) for p in sorted(Path(directory).glob('metrics_update_*.json'))})
    left,right=copy.deepcopy(states[0]['rollout_buffer']),copy.deepcopy(states[1]['rollout_buffer'])
    for buffer in (left,right):
        for item in buffer['items']:
            for key in ('task_id','episode_id','event_segment','event_fraction'):
                item.pop(key,None)
    assert_equal_state(left,right,'short_u129_same_collection_and_raw_GAE')
    assert_equal_state(metrics[0]['formal_training']['pair_schedule_audits'],
                       metrics[1]['formal_training']['pair_schedule_audits'],'short_pair_order')
    assert_equal_state(states[0]['training_config']['a2'],states[1]['training_config']['a2'],'short_parent')
    assert_equal_state(states[0]['schedules']['drift_monitor'],states[1]['schedules']['drift_monitor'],
                       'same_original_protocol_parent_baseline')
    if states[0]['schedules']['selective_keep']['coefficient']<=0:
        raise ValueError('short X did not calibrate T5')
    return hashes


def configure(args):
    from omegaconf import OmegaConf
    from scripts.run_routea_a2 import digest, read, write
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.config_schema_v2 import validate_v2_config
    inputs = read(Path(args.inputs)/'inputs.json')
    for name in ('receipt','reference','bank'):
        if digest(inputs[name])!=inputs[name+'_sha256']: raise ValueError(name+' input changed')
    c1 = read(Path(args.inputs)/'c1/c1.json')
    scale = float(c1['actor_lr_scale']); clip=.1; sampling=None
    decision = dict(stage=args.stage,c1_sha256=digest(Path(args.inputs)/'c1/c1.json'))
    if args.stage=='main':
        if not args.short_x or not args.short_k or not args.c2:
            raise ValueError('main requires both eight-update metric files and frozen C2 decision')
        rows=lambda p:[dict(read(f), update_index=int(f.stem.rsplit('_',1)[1]))
                       for f in sorted(Path(p).glob('metrics_update_*.json'))]
        clip=short_clip(rows(args.short_x),rows(args.short_k))
        c2=read(args.c2)
        sampling=c2['selected_protocol']
        decision.update(short_pair_gate=paired_short_gate(args.short_x,args.short_k),
                        c2_sha256=digest(args.c2))
    output=Path(args.output).resolve(); output.mkdir(parents=True,exist_ok=False)
    receipt=read(inputs['receipt'])
    for arm in ('X','K'):
        c=make_config(receipt,inputs['receipt'],inputs['reference'],inputs['bank'],output,args.weights,
                      arm,scale=scale,clip=clip,sampling=sampling)
        cfg=OmegaConf.create(c); validate_v2_config(cfg); OmegaConf.save(cfg,output/(arm+'.yaml'))
    write(output/'decision.json',dict(**decision,actor_lr_scale=scale,ppo_clip=clip,
        sampling=sampling,T5_enabled_X=True,T4_enabled=False, endpoint=256,
        configs={arm:digest(output/(arm+'.yaml')) for arm in ('X','K')}))


def train(args):
    from omegaconf import OmegaConf
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.pipeline import run
    run(OmegaConf.load(args.config),Path(args.repo).resolve(),
        resume_from=Path(args.resume) if args.resume else None,stop_after_update=args.stop)


def export(args):
    import shutil
    import torch
    from scripts.run_routea_a2 import digest,read,write
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    for arm in ('X','K'):
        for checkpoint in sorted((Path(args.weights)/arm).glob('checkpoint_update_*')):
            target=output/arm/checkpoint.name;target.mkdir(parents=True)
            for name in ('header.json','SHA256.json','COMMITTED.json'):
                shutil.copy2(checkpoint/name,target/name)
            observed=digest(checkpoint/'state.pt')
            if observed!=read(checkpoint/'SHA256.json')['state.pt']:
                raise ValueError('export checkpoint hash differs')
            write(target/'state_hash.json',dict(original_path=str(checkpoint/'state.pt'),sha256=observed))
            if checkpoint.name in ('checkpoint_update_000129','checkpoint_update_000256'):
                state=torch.load(checkpoint/'state.pt',map_location='cpu',weights_only=True)
                torch.save(state['schedules'],target/'auxiliary_schedules.pt')


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--repo',required=True)
    sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare')
    for name in ('receipt','reference','output','weights'): a.add_argument('--'+name,required=True)
    a.set_defaults(func=prepare)
    a=sub.add_parser('configure')
    for name in ('inputs','output','weights'): a.add_argument('--'+name,required=True)
    a.add_argument('--stage',choices=['short','main'],required=True)
    for name in ('short-x','short-k','c2'): a.add_argument('--'+name)
    a.set_defaults(func=configure)
    a=sub.add_parser('export')
    for name in ('weights','output'):a.add_argument('--'+name,required=True)
    a.set_defaults(func=export)
    a=sub.add_parser('train'); a.add_argument('--config',required=True)
    a.add_argument('--resume'); a.add_argument('--stop',type=int); a.set_defaults(func=train)
    args=p.parse_args(); sys.path.insert(0,str(Path(args.repo).resolve())); args.func(args)


if __name__=='__main__': main()
