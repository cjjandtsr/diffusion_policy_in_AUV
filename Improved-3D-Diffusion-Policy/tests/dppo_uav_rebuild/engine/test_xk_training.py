from types import SimpleNamespace

import pytest
import torch

from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.official_loss_components import (
    AdvantageStats, compute_official_components, InertMask,
)
from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.self_imitation import RollingSelfImitation
from tests.dppo_uav_rebuild.engine.test_logical_accumulation_v2 import make_trainer, batch
from tests.dppo_uav_rebuild.engine.test_selective_keep import bank


def test_mask_keeps_88_denominator_and_only_final_step_gradients():
    trainer, runtime = make_trainer()
    values = torch.nn.Parameter(torch.full((10,16,11),.02,dtype=torch.float64))
    runtime.get_logprobs_subsample = lambda obs,p,n,k,get_ent: (values[k], torch.ones(len(k),11))
    b=batch(10)
    stats=AdvantageStats.from_tensor(b.advantages)
    old=compute_official_components(runtime,b,stats,reward_horizon=8)
    disabled=compute_official_components(runtime,b,stats,reward_horizon=8,inert_mask=InertMask(False))
    assert torch.equal(old.policy,disabled.policy)
    assert torch.equal(old.ratio,disabled.ratio)
    assert torch.equal(torch.autograd.grad(old.policy.mean(),values,retain_graph=True)[0],
                       torch.autograd.grad(disabled.policy.mean(),values,retain_graph=True)[0])
    result=compute_official_components(runtime,b,stats,reward_horizon=8,inert_mask=InertMask(True))
    assert torch.equal(result.ratio[:9],old.ratio[:9])
    assert result.ratio[9].item()==pytest.approx(torch.exp(torch.tensor(.02*32/88)).item())
    result.logratio.sum().backward()
    assert values.grad[9,:,3:9].sum()==0
    assert values.grad[9,:,10].sum()==0
    assert values.grad[0,2:10,3:9].abs().sum()>0


def test_rolling_bank_uses_only_known_success_segments_and_fraction():
    keep=RollingSelfImitation(bank(),coefficient=0.,seed=3,window_updates=8,capacity=2048,initial_update=128)
    def row(i,reason='none',cut=False):
        return dict(condition={'state':torch.ones(576),'global_cond':torch.ones(576)},
                    chain=torch.ones(11,16,11)*i,outcome=reason,terminated=reason!='none',
                    truncated=cut,transition_id=str(i),task_id='a',episode_id='ep',
                    event_segment=1 if reason=='reached' else None,
                    event_fraction=.5 if reason=='reached' else None)
    # A collector cut cannot be assigned the success of the following episode.
    buffer=SimpleNamespace(_items=[row(0,cut=True),row(1),row(2,'reached'),row(3,'collision')])
    keep.ingest(buffer,129)
    assert len(keep.actions)==5
    assert keep.actions[-2,0,0]==1
    assert keep.weights[-1,3,0]==.25 and keep.weights[-1,3,9]==0
    saved=keep.state_dict()
    draw=keep.draw_indices()
    clone=RollingSelfImitation(bank(),coefficient=0.,seed=999,initial_update=128)
    clone.load_state_dict(saved)
    assert torch.equal(draw,clone.draw_indices())
    clone.ingest(SimpleNamespace(_items=[]),136)
    assert len(clone.actions)==2
    tampered=dict(saved,content_sha256='0'*64)
    with pytest.raises(ValueError,match='hash'): clone.load_state_dict(tampered)


def test_real_buffer_metadata_survives_checkpoint_and_success_collection():
    from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.rollout_buffer import OnPolicyRolloutBuffer
    buffer=OnPolicyRolloutBuffer(2,outer_update_id=129)
    for i in range(2):
        buffer.add(condition={'state':torch.ones(576),'global_cond':torch.ones(576)},
            chain=torch.ones(11,16,11)*i,old_logprobs=torch.zeros(10,16,11),
            reward=1.,value=0.,next_value=0.,terminated=i==1,truncated=False,
            outcome='reached' if i==1 else 'none',execution_metadata=dict(task_id='train',
                episode_id='actual_episode',event_segment=2 if i==1 else None,
                event_fraction=.5 if i==1 else None))
    buffer.finalize()
    restored=OnPolicyRolloutBuffer(2,outer_update_id=129)
    restored.load_state_dict(buffer.state_dict())
    keep=RollingSelfImitation(bank(),coefficient=.1,seed=3)
    keep.ingest(restored,129)
    assert len(keep.records)==5
    assert keep.records[-1]['episode_id']=='actual_episode'
    assert keep.weights[-1,4,0]==.25
    assert keep.weights[-1,4,9]==0


def _real_e5():
    import importlib
    from pathlib import Path
    from diffusers import DDIMScheduler
    from diffusion_policy_3d.rl.dppo_uav_rebuild.official_core.symbols import load_official_symbols
    load_official_symbols(Path(__file__).resolve().parents[3]/'third_party/dppo_official_cc7234a')
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.value=torch.nn.Parameter(torch.tensor(.25))
        def forward(self,*,sample,timestep,local_cond,global_cond):
            return self.value.expand_as(sample).clone()
    class Critic(torch.nn.Linear):
        def forward(self,condition): return super().forward(condition['state'])
    return importlib.import_module('uav_idp3.e5_runtime').E5PPODiffusion(
        model=Model(),scheduler=DDIMScheduler(num_train_timesteps=100,prediction_type='sample',clip_sample=True),
        critic=Critic(576,1),device='cpu',num_inference_steps=10,ft_denoising_steps=10)


def test_real_e5_mask_and_rolling_keep_backward():
    runtime=_real_e5()
    condition={'state':torch.ones(2,576),'global_cond':torch.ones(2,576)}
    with torch.no_grad():
        sample=runtime(cond=condition,deterministic=False,return_chain=True)
        old=runtime.get_logprobs(condition,sample.chains).reshape(2,10,16,11)
    indices=torch.tensor([0,9]); rows=torch.arange(2)
    b=SimpleNamespace(condition=condition,chains_prev=sample.chains[rows,indices],
        chains_next=sample.chains[rows,indices+1],denoising_inds=indices,
        returns=torch.ones(2),oldvalues=torch.zeros(2),advantages=torch.tensor([1.,2.]),
        oldlogprobs=old[rows,indices])
    parts=compute_official_components(runtime,b,AdvantageStats.from_tensor(b.advantages),
        reward_horizon=8,inert_mask=InertMask(True))
    assert torch.equal(parts.ratio,torch.ones(2))
    keep=RollingSelfImitation(bank(),coefficient=.1,seed=3)
    loss=keep.loss(runtime,device='cpu')+parts.policy.mean()
    loss.backward()
    assert torch.isfinite(runtime.actor_ft.model.value.grad)
    keep.ingest(SimpleNamespace(_items=[]),136)
    assert len(keep.records)==0
    runtime.actor_ft.model.value.grad=None
    keep.loss(runtime,device='cpu').backward()
    assert runtime.actor_ft.model.value.grad==0


def test_real_drift_probe_restores_rng_protocol_and_saved_baseline(tmp_path):
    import hashlib
    import random
    import numpy as np
    from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.drift_monitor import DriftMonitor
    runtime=_real_e5()
    class ActionNormalizer:
        def unnormalize(self,value): return value*2.
        def get_input_stats(self): return {'std':torch.ones(11)*2.}
    path=tmp_path/'bank.pt'
    torch.save(dict(schema='routea_drift_probe_v1',conditions={'state':torch.ones(256,576),
        'global_cond':torch.ones(256,576)},seeds=list(range(256))),path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    monitor=DriftMonitor(runtime,{'action':ActionNormalizer()},path,digest)
    protocol=runtime.sampling_protocol()
    rng=torch.get_rng_state().clone(); python_rng=random.getstate(); numpy_rng=np.random.get_state()
    monitor.begin()
    result=monitor.finish()
    assert result['step_h4_rms_m']==0 and result['cum_inert_norm']==0
    assert torch.equal(rng,torch.get_rng_state()) and random.getstate()==python_rng
    assert np.array_equal(np.random.get_state()[1],numpy_rng[1])
    assert runtime.sampling_protocol()==protocol
    restored=DriftMonitor(runtime,{'action':ActionNormalizer()},path,digest)
    restored.load_state_dict(monitor.state_dict())
    assert torch.equal(monitor.parent_full,restored.parent_full)
    with pytest.raises(ValueError,match='hash'):
        restored.load_state_dict(dict(monitor.state_dict(),bank_sha256='0'*64))


def test_rolling_keep_zero_noop_and_rejected_update_restores_rng():
    from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.trainer import TrainerConfig
    from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.update_transaction import UpdateRejected
    from tests.dppo_uav_rebuild.engine.test_selective_keep import attach_keep
    left,rl=make_trainer(); right,rr=make_trainer()
    rr.critic.load_state_dict(rl.critic.state_dict())
    attach_keep(right,rr,0.)
    right.keep=RollingSelfImitation(bank(),coefficient=0.,seed=3)
    state=right.keep.state_dict()
    left.update_logical_batch(batch(),microbatch_pairs=128)
    right.update_logical_batch(batch(),microbatch_pairs=128)
    assert torch.equal(rl.actor_parameter,rr.actor_parameter)
    assert torch.equal(state['rng'],right.keep.generator.get_state())
    right.keep.coefficient=.1
    right.config=TrainerConfig(max_approx_kl=-1.)
    saved=right.keep.state_dict()
    with pytest.raises(UpdateRejected): right.update_logical_batch(batch(),microbatch_pairs=128)
    assert right.keep.state_dict()['content_sha256']==saved['content_sha256']
    assert torch.equal(right.keep.generator.get_state(),saved['rng'])


def test_anchor_actor_only_and_zero_noop_with_rng_roundtrip():
    from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.inert_anchor import InertAnchor,calibrate_anchor
    left,rl=make_trainer(); right,rr=make_trainer()
    rr.critic.load_state_dict(rl.critic.state_dict())
    for r in (rl,rr):
        r.predict_epsilon=False
        r.ddim_t=torch.arange(10)
        r.actor=lambda x,t,c: torch.zeros_like(x)
        r.actor_ft=lambda x,t,c,_r=r: _r.actor_parameter.expand_as(x)
    right.inert_anchor=InertAnchor(coefficient=0.,seed=73)
    saved=right.inert_anchor.state_dict()
    left.update_logical_batch(batch(),microbatch_pairs=128)
    right.update_logical_batch(batch(),microbatch_pairs=128)
    assert torch.equal(rl.actor_parameter,rr.actor_parameter)
    assert torch.equal(saved['rng'],right.inert_anchor.generator.get_state())
    calibration=calibrate_anchor(right,[batch(128)]*4,right.inert_anchor)
    assert calibration['coefficient']>0
    right.inert_anchor.coefficient=calibration['coefficient']
    right.update_logical_batch(batch(),microbatch_pairs=128)
    assert not torch.equal(rl.actor_parameter,rr.actor_parameter)


def test_actual_checkpoint_restores_bank_content_rng_and_adam(tmp_path):
    from tests.dppo_uav_rebuild.checkpointing.test_checkpoint_roundtrip import _components
    from diffusion_policy_3d.rl.dppo_uav_rebuild.checkpointing.save import save_checkpoint
    from diffusion_policy_3d.rl.dppo_uav_rebuild.checkpointing.restore import restore_checkpoint
    from diffusion_policy_3d.rl.dppo_uav_rebuild.checkpointing.schema import CheckpointHeader,SCHEMA_VERSION
    lora,critic,actor_opt,critic_opt,counters=_components()
    keep=RollingSelfImitation(bank(),coefficient=.123,seed=73)
    (lora.value**2).backward(); actor_opt.step()
    saved_adam=actor_opt.state_dict()['state'][0]['exp_avg'].clone()
    save_checkpoint(tmp_path/'checkpoint',header=CheckpointHeader(SCHEMA_VERSION,{},'a'*64,'b'*64,129),
        lora_bundle=lora,critic=critic,actor_optimizer=actor_opt,critic_optimizer=critic_opt,
        counters=counters,schedules={'selective_keep':keep})
    expected=keep.draw_indices(); expected_rng=torch.rand(4)
    keep.records=[]; keep._rebuild()
    actor_opt.state[lora.value]['exp_avg'].zero_()
    restore_checkpoint(tmp_path/'checkpoint',runtime_identity=SimpleNamespace(content_hash='a'*64),
        lora_identity=SimpleNamespace(content_hash='b'*64),lora_bundle=lora,critic=critic,
        actor_optimizer=actor_opt,critic_optimizer=critic_opt,counters=counters,
        schedules={'selective_keep':keep})
    assert len(keep.records)==3 and keep.coefficient==.123
    assert torch.equal(expected,keep.draw_indices())
    assert torch.equal(expected_rng,torch.rand(4))
    assert torch.equal(actor_opt.state[lora.value]['exp_avg'],saved_adam)


@pytest.mark.parametrize('arm',['X','K'])
def test_xk_schema_and_global_learning_rate(arm):
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.config_schema_v2 import validate_v2_config
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.pipeline import _actor_lr_for_update
    from tests.dppo_uav_rebuild.entrypoints.test_common_protocol_wiring import common_config
    config=common_config()
    config.a2=dict(arm='C',parent_checkpoint='parent',parent_receipt='receipt',
        parent_receipt_sha256='a'*64,parent_source_root='old',reference_bank='bank',
        reference_sha256='b'*64,keep_seed=73)
    config.xk=dict(arm=arm,actor_lr_scale=.8 if arm=='X' else 1.)
    config.run.seed=202610261
    config.run.checkpoint_updates=sorted(list(config.run.checkpoint_updates)+[129])
    config.dppo.ppo_inert_mask=dict(enabled=arm=='X',channels=[3,4,5,6,7,8,10],scope='final_step')
    config.dppo.self_imitation=dict(enabled=arm=='X',window_updates=8,capacity=2048,
        target_ratio=.1,batch_size=32,seed=73)
    config.dppo.drift_monitor=dict(enabled=True,bank_path='drift',bank_sha256='c'*64,every_commits=5)
    config.dppo.drift_gate=dict(enabled=False)
    config.dppo.inert_anchor=dict(enabled=False,coefficient=0.,target_ratio=.05,batch_size=128,seed=73)
    if arm=='X': config.dppo.clip_ploss_coef=.05
    validate_v2_config(config)
    expected=(2e-5+(129-1)/255*(1e-5-2e-5))*config.xk.actor_lr_scale
    assert _actor_lr_for_update(config,129)==pytest.approx(expected)
    assert _actor_lr_for_update(config,256)==pytest.approx(1e-5*config.xk.actor_lr_scale)
    config.dppo.self_imitation.enabled=arm!='X'
    with pytest.raises(ValueError,match='T1 and T5'): validate_v2_config(config)
