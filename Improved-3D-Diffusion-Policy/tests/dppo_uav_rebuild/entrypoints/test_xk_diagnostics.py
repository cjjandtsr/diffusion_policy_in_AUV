from types import SimpleNamespace
import copy
import hashlib
import json
import pytest
import torch

from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.xk_diagnostics import actor_gradient, lr_scale_for_ratio
from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.official_loss_components import InertMask
from scripts.run_routea_xk import paired_short_gate, stratified_indices, short_clip


def test_frozen_gradient_microbatch_equivalence_and_no_submission():
    parameter=torch.nn.Parameter(torch.arange(11).float()/100.)
    runtime=SimpleNamespace(gamma_denoising=.9,ft_denoising_steps=10,
        clip_ploss_coef_base=.001,clip_ploss_coef=.1,clip_ploss_coef_rate=3.,clip_vloss_coef=None,
        critic=lambda cond:torch.zeros(len(cond['state'])))
    runtime.get_logprobs_subsample=lambda cond,prev,nxt,inds,get_ent: (
        parameter[None,None,:].expand(len(inds),16,11),torch.ones(len(inds)))
    trainer=SimpleNamespace(runtime=runtime,actor_parameters=[parameter],
        config=SimpleNamespace(reward_start=2,reward_horizon=8,entropy_loss_coefficient=0.))
    n=23
    batch=SimpleNamespace(condition={'state':torch.zeros(n,576),'global_cond':torch.zeros(n,576)},
        chains_prev=torch.zeros(n,16,11),chains_next=torch.zeros(n,16,11),
        denoising_inds=torch.arange(n)%10,returns=torch.zeros(n),oldvalues=torch.zeros(n),
        advantages=torch.linspace(-1,1,n),oldlogprobs=torch.zeros(n,16,11))
    before=parameter.detach().clone()
    all_mask=InertMask(True,scope='all_steps')
    g=actor_gradient(trainer,batch,all_mask,7)
    torch.testing.assert_close(g,actor_gradient(trainer,batch,all_mask,n),atol=1e-7,rtol=1e-5)
    assert torch.equal(parameter,before) and parameter.grad is None
    assert torch.count_nonzero(g[[3,4,5,6,7,8,10]])==0
    assert torch.count_nonzero(g[[0,1,2,9]])==4
    assert not torch.equal(g,actor_gradient(trainer,batch,InertMask(),7))
    for mask,expected in ((InertMask(),.05),(InertMask(True),.12/11),
                          (InertMask(True,scope='all_steps'),.12/11)):
        final=[]
        actor_gradient(trainer,batch,mask,7,final_logratios=final)
        observed=torch.cat(final)
        assert len(observed)==2
        torch.testing.assert_close(observed,torch.full((2,),expected),atol=1e-8,rtol=1e-6)


def test_short_clip_requires_complete_pair_and_fixed_rule():
    def rows(v): return [dict(update_index=u,drift_monitor={'step_h4_rms_m':v}) for u in range(129,137)]
    assert short_clip(rows(3),rows(1))==.05
    assert short_clip(rows(2),rows(1))==.1
    with pytest.raises(ValueError): short_clip(rows(3)[:-1],rows(1))
    assert lr_scale_for_ratio(.5)==1.
    assert lr_scale_for_ratio(2.)==.5
    with pytest.raises(ValueError): lr_scale_for_ratio(float('nan'))


def test_role_stratification_is_fixed_unique_and_uses_minority():
    items=[{'role':'a'}]*1000+[{'role':'b'}]*24
    chosen=stratified_indices(items)
    assert len(chosen)==len(set(chosen))==256
    assert sum(items[i]['role']=='b' for i in chosen)==24
    assert chosen==stratified_indices(items)


def _short_pair(tmp_path, mutation=None):
    from diffusion_policy_3d.rl.dppo_uav_rebuild.checkpointing.save import save_checkpoint
    from diffusion_policy_3d.rl.dppo_uav_rebuild.checkpointing.schema import CheckpointHeader
    from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.rollout_buffer import OnPolicyRolloutBuffer
    from scripts.routea_paired_support.core import canonical_hash

    buffer=OnPolicyRolloutBuffer(2,outer_update_id=129)
    for i in range(2):
        buffer.add(condition={'state':torch.ones(576),'global_cond':torch.zeros(576)},
            chain=torch.ones(11,16,11)*i,old_logprobs=torch.zeros(10,16,11),
            reward=1.,value=.2,next_value=0.,terminated=i==1,truncated=False,
            role='dense_recovery',outcome='reached' if i==1 else 'none')
    buffer.finalize()
    paths=[]
    for arm in ('X','K'):
        directory=tmp_path/arm; directory.mkdir()
        checkpoint=directory/'checkpoint_update_000129'
        rollout=copy.deepcopy(buffer.state_dict())
        if arm=='X':
            for i,item in enumerate(rollout['items']):
                item.update(task_id='train',episode_id='ep0',event_segment=2 if i==1 else None,
                            event_fraction=.5 if i==1 else None)
        config={'xk':{'arm':arm},'run':{'seed':202610261},
                'a2':{'parent_checkpoint':'fixed_u128','parent_receipt_sha256':'a'*64}}
        drift={'bank_sha256':'b'*64,'parent_full':torch.zeros(256,16,11)}
        coefficient=.1
        audits=[{'pass_id':i,'pair_permutation_sha256':str(i)*64} for i in range(5)]
        if arm=='K':
            if mutation=='gae': rollout['advantages'][0]+=1.
            if mutation=='chain': rollout['items'][0]['chain'][0,0,0]+=1.
            if mutation=='pair': audits[0]['pair_permutation_sha256']='f'*64
            if mutation=='drift': drift['parent_full'][0,0,0]=1.
            if mutation=='parent': config['a2']['parent_checkpoint']='different_u128'
            if mutation=='seed': config['run']['seed']+=1
        if arm=='X' and mutation=='t5': coefficient=0.
        if arm=='X' and mutation=='arm': config['xk']['arm']='K'
        identity={'arm':arm,'seed':202610261}
        critic=torch.nn.Linear(1,1)
        parameter=torch.nn.Parameter(torch.tensor(1.))
        save_checkpoint(checkpoint,header=CheckpointHeader(2,identity,canonical_hash(identity),'c'*64,129),
            lora_bundle=SimpleNamespace(adapter_state_dict=lambda:{'adapter':parameter.detach()}),
            critic=critic,actor_optimizer=torch.optim.Adam([parameter]),
            critic_optimizer=torch.optim.Adam(critic.parameters()),counters=SimpleNamespace(committed_updates=20),
            rollout_buffer=SimpleNamespace(state_dict=lambda:rollout),training_config=config,
            schedules={'drift_monitor':SimpleNamespace(state_dict=lambda:drift),
                       **({'selective_keep':SimpleNamespace(state_dict=lambda:{'coefficient':coefficient})}
                          if arm=='X' else {})})
        # These fixtures model WSL-produced checkpoints: Windows text-mode
        # saving otherwise changes header LF to CRLF before the file digest.
        header_path=checkpoint/'header.json'
        header_path.write_bytes(header_path.read_bytes().replace(b'\r\n',b'\n'))
        sums={name:hashlib.sha256((checkpoint/name).read_bytes()).hexdigest()
              for name in ('header.json','state.pt')}
        (checkpoint/'SHA256.json').write_bytes(json.dumps(sums).encode())
        (checkpoint/'COMMITTED.json').write_bytes(json.dumps(dict(schema_version=2,
            status='committed',files=sums,header_schema_hash=sums['header.json'])).encode())
        for update in range(129,137):
            metric={'checkpoint':str(checkpoint),'drift_monitor':{'step_h4_rms_m':.2},
                    'formal_training':{'pair_schedule_audits':audits}}
            (directory/('metrics_update_%06d.json'%update)).write_text(json.dumps(metric),encoding='utf-8')
        paths.append(directory)
    return paths


def test_paired_short_gate_reads_actual_checkpoints_and_hashes_every_metric(tmp_path):
    paths=_short_pair(tmp_path)
    before={str(p):p.read_bytes() for directory in paths for p in directory.rglob('*') if p.is_file()}
    receipt=paired_short_gate(*paths)
    for arm,directory in zip(('X','K'),paths):
        assert receipt[arm]['state_sha256']==hashlib.sha256((directory/'checkpoint_update_000129/state.pt').read_bytes()).hexdigest()
        assert len(receipt[arm]['metrics'])==8
        for name,digest in receipt[arm]['metrics'].items():
            assert digest==hashlib.sha256((directory/name).read_bytes()).hexdigest()
    assert all(p.read_bytes()==before[str(p)] for directory in paths for p in directory.rglob('*') if p.is_file())


@pytest.mark.parametrize('mutation,reason',[
    ('gae','raw_GAE'),('chain','raw_GAE'),('pair','short_pair_order'),
    ('drift','parent_baseline'),('parent','short_parent'),('seed','seed mismatch'),
    ('t5','calibrate T5'),('arm','arm mismatch')])
def test_paired_short_gate_rejects_protocol_or_collection_mismatch(tmp_path,mutation,reason):
    with pytest.raises(ValueError,match=reason):
        paired_short_gate(*_short_pair(tmp_path,mutation))


def test_paired_short_gate_rejects_checkpoint_byte_tampering(tmp_path):
    paths=_short_pair(tmp_path)
    state=paths[0]/'checkpoint_update_000129/state.pt'
    state.write_bytes(state.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='checkpoint bytes changed'):
        paired_short_gate(*paths)


def test_export_keeps_weights_in_place_and_exports_complete_auxiliary_state(tmp_path):
    from scripts.run_routea_xk import export
    paths=_short_pair(tmp_path)
    output=tmp_path/'evidence'
    export(SimpleNamespace(weights=tmp_path,output=output))
    for arm,path in zip(('X','K'),paths):
        target=output/arm/'checkpoint_update_000129'
        assert not (target/'state.pt').exists()
        original=torch.load(path/'checkpoint_update_000129/state.pt',weights_only=True)
        saved=torch.load(target/'auxiliary_schedules.pt',weights_only=True)
        torch.testing.assert_close(saved['drift_monitor']['parent_full'],original['schedules']['drift_monitor']['parent_full'])
        assert (target/'state_hash.json').exists()
