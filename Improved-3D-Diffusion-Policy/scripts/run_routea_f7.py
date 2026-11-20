"""F7 v1.2 orchestration over the existing training pipeline; no training recipe fork."""
import argparse
import copy
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_routea_a2 import read, write, digest, verify_saved_source


def parent(args):
    # Load the original P6 builder before importing any current computational module.
    import importlib
    import torch
    from omegaconf import OmegaConf
    source=Path(args.source_root).resolve()
    sys.path.insert(0,str(source))
    import scripts
    scripts.__path__=[str(source/'scripts')]
    prefix='diffusion_policy_3d.rl.dppo_uav_rebuild.'
    identity=importlib.import_module(prefix+'entrypoints.identity')
    builder=importlib.import_module(prefix+'entrypoints.build_system')
    restore=importlib.import_module(prefix+'checkpointing.restore').restore_checkpoint
    Path(identity.__file__).resolve().relative_to(source)
    rows=identity._source_rows(source)
    identity._source_rows=lambda unused:rows
    symbols=builder.load_official_symbols
    builder.load_official_symbols=lambda unused:symbols(source/'third_party/dppo_official_cc7234a')
    checkpoint=Path(args.checkpoint)
    expected='a91c9f428c5749b8ca507168ea83f17eeb88cce4b4a03c6aeaea197b214bdd43'
    if digest(checkpoint/'state.pt')!=expected: raise ValueError('not registered P6 u256')
    state=torch.load(checkpoint/'state.pt',map_location='cpu',weights_only=True)
    c=OmegaConf.create(state['training_config'])
    system=builder.build_system(c,Path(args.repo))
    runtime=identity.build_runtime_identity(c,Path(args.repo),system,Path(c.admission_manifest))
    h=restore(checkpoint,runtime_identity=runtime,lora_identity=system.lora_bundle.identity,
        lora_bundle=system.lora_bundle,critic=system.critic,actor_optimizer=system.actor_optimizer,
        critic_optimizer=system.critic_optimizer,counters=system.trainer.counters,map_location='cpu')
    assert h.update_index==256 and runtime.content_hash=='d987235a0cbb5ad395bbd6e7979bdb8d7e0cff1303b0611cb462c7d554bbd61f'
    closure=identity.build_executed_source_manifest(Path(args.repo),c,Path(c.admission_manifest))
    probe=read(Path(args.metrics)/'r1_plan_probe.json')
    assert probe['runtime_identity_hash']==runtime.content_hash
    write(args.output,dict(schema='f7_parent_v1',parent_checkpoint=str(checkpoint.resolve()),
        state_sha256=expected,runtime_identity_hash=runtime.content_hash,runtime_identity=h.runtime_identity,
        training_config=state['training_config'],source_manifest=closure,source_root=str(source),r1_probe=probe,
        actor_names=[n for n,_ in system.lora_bundle.named_adapter_parameters()],
        critic_names=[n for n,_ in system.critic.named_parameters()]))
    print('F7_P6_ACTUAL_RESTORE_PASS')


def make_config(receipt, receipt_path, reference, output, weights, run_id, weight):
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.f7_support import RUNS, KEEP
    seed,adaptive=RUNS[run_id]
    c=copy.deepcopy(receipt['training_config'])
    original_reference_parent=c['a2']['parent_receipt_sha256']
    c['run'].update(seed=seed,updates=320,checkpoint_every_updates=8,
        checkpoint_updates=sorted({0,1,129,257}|set(range(8,321,8))))
    c['optimizer'].update(actor_lr=1e-5,actor_lr_end=1e-5,critic_lr=1e-3)
    c['output_dir']=str(Path(weights).resolve()/run_id)
    c['result_dir']=str(Path(output).resolve()/'training'/run_id)
    c['a2'].update(parent_checkpoint=receipt['parent_checkpoint'],
        parent_receipt=str(Path(receipt_path).resolve()),parent_receipt_sha256=digest(receipt_path),
        parent_source_root=receipt['source_root'],reference_bank=str(Path(reference).resolve()))
    c['reward_version']='v5' if adaptive else 'v2'
    c['f7']=dict(run_id=run_id,approach_weight=weight,adaptive_lr=adaptive,
        keep_coefficient=KEEP,reference_parent_receipt_sha256=original_reference_parent)
    return c


def sources(root):
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.identity import _source_rows
    rows=_source_rows(root)
    names=['scripts/run_routea_f7.py','scripts/f7_analysis.py','scripts/f7_reconstruct.py']
    names += [p.relative_to(root).as_posix() for p in (root/'scripts/routea_paired_support').glob('*.py')]
    for name in names:
        if name not in {r['relative_path'] for r in rows}:
            rows.append(dict(relative_path=name,sha256=digest(root/name)))
    return sorted(rows,key=lambda r:r['relative_path'])


def prepare(args):
    import torch
    from omegaconf import OmegaConf
    from diffusion_policy_3d.rl.dppo_uav_rebuild.checkpointing.a2_fork import validate_child_config
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.f7_support import RUNS, REFERENCE_SHA
    r=read(args.receipt);verify_saved_source(r)
    assert digest(Path(r['parent_checkpoint'])/'state.pt')==r['state_sha256']
    assert digest(args.reference)==REFERENCE_SHA
    b=torch.load(args.reference,map_location='cpu',weights_only=True)
    assert len(b['actions'])==549 and b['parent_receipt_sha256']==r['training_config']['a2']['parent_receipt_sha256']
    calibration=read(args.calibration)
    from scripts.f7_reconstruct import select_weight
    selected=select_weight(calibration['stats']['reached'])
    assert selected==args.weight,'F7 v1.3 signed-p10 calibration weight differs'
    assert calibration['reconstruction_passed'] and calibration['selected_weight']==args.weight
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    configs={}
    for name in RUNS:
        c=make_config(r,args.receipt,args.reference,out,args.weights,name,args.weight)
        validate_child_config(r['training_config'],c)
        p=out/(name+'.yaml');OmegaConf.save(OmegaConf.create(c),p)
        configs[name]=dict(path=str(p.resolve()),sha256=digest(p))
    write(out/'prepared.json',dict(configs=configs,files=sources(Path(args.repo).resolve()),
        calibration=str(Path(args.calibration).resolve()),calibration_sha256=digest(args.calibration),
        weight=args.weight,claim='prepared only; A-level review and real u257 pairing remain required'))


def run(args):
    from omegaconf import OmegaConf
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.pipeline import run as train
    from scripts.f7_analysis import matched_rollout
    prepared=read(Path(args.prepared)/'prepared.json');root=Path(args.repo).resolve()
    assert sources(root)==prepared['files'],'prepared source changed'
    assert digest(prepared['calibration'])==prepared['calibration_sha256']
    item=prepared['configs'][args.run_id];assert digest(item['path'])==item['sha256']
    c=OmegaConf.load(item['path'])
    resume=None
    if args.stop_after_update==320:
        matched_rollout(args.prepared,{'P16':'P18','P17':'P19','P18':'P18','P19':'P19'}[args.run_id])
        resume=Path(args.resume) if args.resume else Path(c.output_dir)/'checkpoint_update_000257'
        assert resume.resolve().parent==Path(c.output_dir).resolve()
        assert 257<=read(resume/'header.json')['update_index']<320
    elif args.resume:raise ValueError('u257 must fork P6')
    train(c,root,resume_from=resume,stop_after_update=args.stop_after_update)
    m=read(Path(c.result_dir)/('metrics_update_%06d.json'%args.stop_after_update))
    assert len(m['updates'])==20 and all(u['status']=='committed' for u in m['updates'])
    assert m['formal_training']['hard_stop_reason'] is None
    from scripts.routea_f6_support import verified_state
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.f7_support import KEEP
    checkpoint=Path(c.output_dir)/('checkpoint_update_%06d'%args.stop_after_update)
    _,state=verified_state(checkpoint)
    assert state['training_config']==OmegaConf.to_container(c,resolve=True)
    assert state['schedules']['selective_keep']['coefficient']==KEEP
    assert state['schedules']['f7_actor_lr']==m['formal_training']['f7_actor_lr']
    assert state['counters']['committed_updates']==args.stop_after_update*20
    assert state['counters']['env_steps']==args.stop_after_update*1024
    print('F7_TRAINING_STAGE_COMPLETE',args.run_id,args.stop_after_update)


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('parent')
    for k in ('repo','source-root','checkpoint','metrics','output'):a.add_argument('--'+k,required=True)
    a=s.add_parser('prepare')
    for k in ('repo','receipt','reference','output','weights','calibration'):a.add_argument('--'+k,required=True)
    a.add_argument('--weight',type=float,choices=(1.,2.),required=True)
    a=s.add_parser('run')
    for k in ('repo','prepared','run-id'):a.add_argument('--'+k,required=True)
    a.add_argument('--stop-after-update',type=int,choices=(257,320),required=True)
    a.add_argument('--resume',help='same arm committed checkpoint >=u257, only for continuation')
    args=p.parse_args();globals()[args.command](args)


if __name__=='__main__':main()
