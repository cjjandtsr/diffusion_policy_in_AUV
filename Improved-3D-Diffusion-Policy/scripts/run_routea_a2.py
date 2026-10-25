#!/usr/bin/env python3
"""A2 preparation/continuity orchestration; training remains in the shared pipeline.

prepare-parent must run against the intact A1 checkout BEFORE copying A2 files.
old-replay imports the saved A1 source and only redirects output/checkpoint saving.
"""
import argparse
import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n')


def assert_equal_state(left,right,path='state'):
    import torch
    import numpy as np
    if isinstance(left,torch.Tensor):
        if not isinstance(right,torch.Tensor) or left.dtype!=right.dtype or not torch.equal(left.cpu(),right.cpu()):
            raise ValueError('continuity mismatch: '+path)
    elif isinstance(left,np.ndarray):
        if not isinstance(right,np.ndarray) or left.dtype!=right.dtype or not np.array_equal(left,right):
            raise ValueError('continuity mismatch: '+path)
    elif isinstance(left,dict):
        if not isinstance(right,dict) or set(left)!=set(right): raise ValueError('continuity keys: '+path)
        for k in left: assert_equal_state(left[k],right[k],path+'.'+str(k))
    elif isinstance(left,(tuple,list)):
        if not isinstance(right,(tuple,list)) or len(left)!=len(right): raise ValueError('continuity length: '+path)
        for i,(a,b) in enumerate(zip(left,right)): assert_equal_state(a,b,path+'.'+str(i))
    elif left!=right:
        raise ValueError('continuity mismatch: '+path)


def prepare_parent(args):
    import torch
    from omegaconf import OmegaConf
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.build_system import build_system
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.identity import build_runtime_identity,build_executed_source_manifest,_canonical_hash
    from diffusion_policy_3d.rl.dppo_uav_rebuild.checkpointing.restore import restore_checkpoint
    root=Path(args.repo).resolve(); cfg=OmegaConf.load(args.config)
    system=build_system(cfg,root)
    identity=build_runtime_identity(cfg,root,system,Path(cfg.admission_manifest))
    header=restore_checkpoint(args.parent,runtime_identity=identity,lora_identity=system.lora_bundle.identity,
        lora_bundle=system.lora_bundle,critic=system.critic,actor_optimizer=system.actor_optimizer,
        critic_optimizer=system.critic_optimizer,counters=system.trainer.counters,map_location='cpu')
    if header.update_index!=128 or int(cfg.dppo.logical_batch_pairs)!=2560:
        raise ValueError('parent is not B2560 u128')
    state=torch.load(Path(args.parent)/'state.pt',map_location='cpu',weights_only=True)
    if _canonical_hash(state['training_config'])!=identity.config_hash:
        raise ValueError('saved parent configuration mismatch')
    out=Path(args.output).resolve(); out.mkdir(parents=True,exist_ok=False)
    source=out/'source'
    # This is the one necessary immutable A1 source for the actual continuity replay.
    for directory in ('diffusion_policy_3d','scripts'):
        shutil.copytree(root/directory,source/directory,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    closure=build_executed_source_manifest(root,cfg,Path(cfg.admission_manifest))
    for row in closure['files']:
        dst=source/row['relative_path']; dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists(): shutil.copy2(root/row['relative_path'],dst)
        if digest(dst)!=row['sha256']: raise ValueError('parent source copy mismatch')
    probe=read(Path(cfg.result_dir)/'r1_plan_probe.json')
    if probe['runtime_identity_hash']!=identity.content_hash: raise ValueError('parent probe mismatch')
    receipt=dict(schema='a2_parent_v1',runtime_identity_hash=identity.content_hash,
        runtime_identity=header.runtime_identity,parent_checkpoint=str(Path(args.parent).resolve()),
        state_sha256=digest(Path(args.parent)/'state.pt'),source_root=str(source),
        training_config=state['training_config'],source_manifest=closure,r1_probe=probe,
        actor_names=[n for n,_ in system.lora_bundle.named_adapter_parameters()],
        critic_names=[n for n,_ in system.critic.named_parameters()],
        preparation_script_sha256=digest(__file__))
    write(out/'receipt.json',receipt)
    print(json.dumps(dict(status='PARENT_VERIFIED',receipt=str(out/'receipt.json'),sha256=digest(out/'receipt.json'))))


def verify_saved_source(receipt):
    source=Path(receipt['source_root'])
    for row in receipt['source_manifest']['files']:
        if digest(source/row['relative_path'])!=row['sha256']:
            raise ValueError('saved A1 source mismatch: '+row['relative_path'])


def old_replay(args):
    from omegaconf import OmegaConf
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints import pipeline,identity
    r=read(args.receipt); verify_saved_source(r)
    try:
        Path(pipeline.__file__).resolve().relative_to(Path(r['source_root']).resolve())
    except ValueError:
        raise ValueError('old replay did not import saved A1 pipeline')
    cfg=OmegaConf.create(r['training_config']); root=Path(args.repo).resolve()
    # Inputs remain live and verified by A1 build_runtime_identity. Source rows
    # come from the verified snapshot that this process actually imports.
    identity._source_rows=lambda unused: copy.deepcopy(r['source_manifest']['files'])
    old_resolve=pipeline._resolve
    redirects={str(cfg.output_dir):Path(args.weights).resolve(),str(cfg.result_dir):Path(args.output).resolve()}
    for p in redirects.values():
        if p.exists(): raise FileExistsError('old replay requires fresh output: '+str(p))
    def resolve(repo,value):
        return redirects[str(value)] if str(value) in redirects else old_resolve(repo,value)
    pipeline._resolve=resolve
    original=pipeline._should_save_checkpoint
    pipeline._should_save_checkpoint=lambda c,u: int(u)==129 or original(c,u)
    write(Path(args.output)/'r1_plan_probe.json',r['r1_probe'])
    write(Path(args.output)/'replay_instrumentation.json',dict(source_root=r['source_root'],
        receipt_sha256=digest(args.receipt),wrapper_sha256=digest(__file__),
        changes=['output routing','additional u129 checkpoint saving'],training_semantics='unchanged'))
    pipeline.run(cfg,root,resume_from=Path(r['parent_checkpoint']),stop_after_update=129)


def configure(args):
    from omegaconf import OmegaConf
    r=read(args.receipt); ref=Path(args.reference).resolve(); inventory=read(ref.parent/'inventory.json')
    manifest=read(ref.parent/'reference_manifest.json')
    if digest(ref)!=manifest['sha256']: raise ValueError('reference manifest mismatch')
    if not inventory['available_tasks']['rl_only']:
        raise ValueError('no RL-only reference task; return inventory for scientific decision')
    out=Path(args.output).resolve(); out.mkdir(parents=True,exist_ok=False)
    for arm in ('C','T'):
        c=copy.deepcopy(r['training_config'])
        c['run']['checkpoint_updates']=sorted(set(c['run']['checkpoint_updates'])|{129})
        c['output_dir']=str(Path(args.weights).resolve()/arm)
        c['result_dir']=str(out/arm)
        c['a2']=dict(arm=arm,parent_checkpoint=r['parent_checkpoint'],
            parent_receipt=str(Path(args.receipt).resolve()),parent_receipt_sha256=digest(args.receipt),
            parent_source_root=r['source_root'],reference_bank=str(ref),reference_sha256=digest(ref),keep_seed=2026102301)
        OmegaConf.save(OmegaConf.create(c),out/(arm+'.yaml'))
    write(out/'plan_receipt.json',dict(reference_sha256=digest(ref),reference_inventory=inventory,
        main_update=256,first_gate_update=129,additional_updates=128,
        config_sha256={arm:digest(out/(arm+'.yaml')) for arm in ('C','T')}))


def collect(args):
    import torch
    from omegaconf import OmegaConf
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.build_system import build_system
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.identity import build_runtime_identity
    from diffusion_policy_3d.rl.dppo_uav_rebuild.checkpointing.a2_fork import fork_parent
    from scripts.routea_paired_support.a2_reference import collect_reference
    r=read(args.receipt); cfg=OmegaConf.create(r['training_config']); root=Path(args.repo).resolve()
    system=build_system(cfg,root)
    cfg.a2=dict(parent_checkpoint=r['parent_checkpoint'],parent_receipt=str(Path(args.receipt).resolve()),
                parent_receipt_sha256=digest(args.receipt),parent_source_root=r['source_root'])
    identity=build_runtime_identity(cfg,root,system,Path(cfg.admission_manifest))
    fork_parent(cfg,root,system,None,identity)
    collect_reference(system,cfg,output=args.output,parent_receipt_sha256=digest(args.receipt),
                      exclude_bank=args.exclude_bank)


def gate(args):
    import torch
    def load(p):
        from scripts.routea_paired_support.core import verify_checkpoint
        root=Path(p); h=read(root/'header.json'); sums=read(root/'SHA256.json')
        verify_checkpoint(root,dict(runtime_identity_hash=h['runtime_identity_hash'],
            header_sha256=sums['header.json'],state_sha256=sums['state.pt'],update=129))
        if h['update_index']!=129: raise ValueError('gate requires u129')
        return torch.load(root/'state.pt',map_location='cpu',weights_only=True)
    old,c,t=(load(p) for p in (args.old_checkpoint,args.control,args.treatment))
    from omegaconf import OmegaConf
    configs={'C':args.control_config,'T':args.treatment_config}
    for arm,state in [('C',c),('T',t)]:
        assert_equal_state(state['training_config'],OmegaConf.to_container(OmegaConf.load(configs[arm]),resolve=True),arm+'_config')
        spec=state['training_config']['a2']
        if spec['arm']!=arm: raise ValueError('wrong arm')
        if digest(spec['reference_bank'])!=spec['reference_sha256']: raise ValueError('reference changed')
    for key in ('parent_receipt_sha256','reference_sha256','keep_seed'):
        assert_equal_state(c['training_config']['a2'][key],t['training_config']['a2'][key],key)
    keys=('lora','critic','actor_optimizer','critic_optimizer','counters','rollout_buffer','task_sampler',
          'rng_python','rng_numpy_legacy','rng_cpu','rng_cuda')
    for key in keys: assert_equal_state(old[key],c[key],key)
    assert_equal_state(c['rollout_buffer'],t['rollout_buffer'],'C_T_first_rollout_raw_GAE')
    om,cm,tm=(read(p) for p in (args.old_metrics,args.control_metrics,args.treatment_metrics))
    for m in (cm,tm):
        assert_equal_state(om['formal_training']['pair_schedule_audits'],m['formal_training']['pair_schedule_audits'],'pair_order')
    # None-valued new diagnostic field is instrumentation, not an objective change.
    cu=copy.deepcopy(cm['updates'])
    for u in cu:
        if u.get('pre_update',{}).get('keep_diagnostics') is None: u['pre_update'].pop('keep_diagnostics',None)
    assert_equal_state(om['updates'],cu,'actual_optimizer_updates')
    inv=read(args.inventory)
    if inv['tasks']!=126 or len(inv['episodes'])!=252: raise ValueError('reference count gate')
    if not inv['available_tasks']['rl_only']:
        raise ValueError('reference has no RL-only successful task; cannot test new-capability retention')
    coefficient=t['schedules']['selective_keep']['coefficient']
    if not coefficient>0 or c['schedules']['selective_keep']['coefficient']!=0:
        raise ValueError('C/T keep coefficient contract')
    if all(torch.equal(c['lora'][k],t['lora'][k]) for k in c['lora']):
        raise ValueError('positive keep produced no actor difference')
    write(args.output,dict(status='PASS',scope='A1->A2 first outer update exact continuity and paired rollout',
        checkpoints={name:dict(path=str(Path(p).resolve()),sha256=digest(Path(p)/'state.pt'),
                              config_sha256=digest(configs[name]) if name in configs else None)
                     for name,p in [('old',args.old_checkpoint),('C',args.control),('T',args.treatment)]},
        inventory_sha256=digest(args.inventory),coverage_warnings=inv['coverage_warnings'],
        coefficient=coefficient,warning='single-seed bounded intervention, coverage counts are not proof of generalization'))


def continuation_checkpoint(receipt, arm, config):
    gate_receipt=read(receipt)
    if gate_receipt['status']!='PASS': raise ValueError('first-update gate has not passed')
    item=gate_receipt['checkpoints'][arm]; checkpoint=Path(item['path'])
    if digest(checkpoint/'state.pt')!=item['sha256'] or digest(config)!=item['config_sha256']:
        raise ValueError('approved checkpoint/config changed')
    return checkpoint


def continue_training(args):
    from omegaconf import OmegaConf
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.pipeline import run
    checkpoint=continuation_checkpoint(args.gate,args.arm,args.config)
    run(OmegaConf.load(args.config),Path(args.repo).resolve(),resume_from=checkpoint)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare-parent')
    for name in ('repo','config','parent','output'): a.add_argument('--'+name,required=True)
    a=sub.add_parser('old-replay')
    for name in ('repo','receipt','output','weights'): a.add_argument('--'+name,required=True)
    a=sub.add_parser('collect')
    for name in ('repo','receipt','output','exclude-bank'): a.add_argument('--'+name,required=True)
    a=sub.add_parser('configure')
    for name in ('repo','receipt','reference','output','weights'): a.add_argument('--'+name,required=True)
    a=sub.add_parser('gate')
    for name in ('repo','old-checkpoint','control','treatment','old-metrics','control-metrics','treatment-metrics','control-config','treatment-config','inventory','output'):
        a.add_argument('--'+name,required=True)
    a=sub.add_parser('continue-training')
    for name in ('repo','gate','arm','config'): a.add_argument('--'+name,required=True)
    args=p.parse_args(argv)
    sys.path.insert(0,str(Path(args.repo).resolve()))
    if args.command=='old-replay': sys.path.insert(0,read(args.receipt)['source_root'])
    globals()[args.command.replace('-','_')](args)
    return 0


if __name__=='__main__': raise SystemExit(main())
