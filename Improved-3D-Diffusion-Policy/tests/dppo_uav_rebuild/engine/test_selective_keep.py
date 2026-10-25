"""A2 execution support, FT gradients and calibration use actual PPO semantics."""
from types import SimpleNamespace

import pytest
import torch

from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.selective_keep import (
    executed_weights, paired_inventory, SelectiveKeep, calibrate_keep,
)
from tests.dppo_uav_rebuild.engine.test_logical_accumulation_v2 import make_trainer, batch


def test_terminal_inside_segment_never_supervises_future_or_partial_yaw():
    w = executed_weights('reached', 1, .5)
    assert w.shape == (16, 11)
    assert w[2, [0, 1, 2, 9]].tolist() == [1., 1., 1., 1.]
    assert w[3, :3].tolist() == [.25, .25, .25]
    assert w[3, 9] == 0
    assert w[4:].sum() == 0
    assert w[:2].sum() == 0
    assert w[:, 3:9].sum() == 0
    assert w[:, 10].sum() == 0


def test_full_prefix_and_zero_time_events():
    assert executed_weights('none').sum() == 16
    assert executed_weights('reached', 0, 0).sum() == 0
    assert executed_weights('reached', 3, 1).sum() == 16
    with pytest.raises(ValueError):
        executed_weights('collision', 0, .5)


def test_pairing_counts_tasks_once_and_does_not_hide_missing_samples():
    rows = [dict(task_id=t, source=s, reached=ok, samples=n)
            for t, pair in [('a', (True, True)), ('b', (False, True)),
                            ('c', (True, False)), ('d', (False, False))]
            for s, ok, n in [('bc', pair[0], int(pair[0])*2),
                              ('u128', pair[1], int(pair[1])*3)]]
    inv = paired_inventory(rows)
    assert inv['task_counts'] == dict(both_success=1, bc_only=1, rl_only=1, neither=1)
    assert inv['sampling_weights'] == dict(both_success=1/3, bc_only=1/3, rl_only=1/3)
    with pytest.raises(ValueError):
        paired_inventory(rows[:-1])


def bank():
    return dict(actions=torch.ones(3, 16, 11), conditions=torch.ones(3, 576),
                weights=torch.stack([executed_weights('none')]*3),
                task_ids=['a','b','c'], sources=['bc','bc','u128'],
                categories=['both_success','bc_only','rl_only'])


def runtime():
    class Actor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.x0 = torch.nn.Parameter(torch.zeros(16, 11))
        def forward(self, x, t, cond):
            assert cond['global_cond'].shape[-1] == 576
            return self.x0.expand_as(x)
    return SimpleNamespace(actor_ft=Actor(), ddim_t=torch.tensor([9, 3]),
                           ft_denoising_steps=2, ddim_alphas=torch.tensor([.1,.7]),
                           predict_epsilon=False)


@pytest.mark.parametrize('ft_steps', [2, 10])
def test_real_e5_keep_uses_matching_ddim_time_and_alpha_without_full_schedule(ft_steps):
    import importlib
    from pathlib import Path
    from diffusers import DDIMScheduler
    from diffusion_policy_3d.rl.dppo_uav_rebuild.official_core.symbols import load_official_symbols
    load_official_symbols(Path(__file__).resolve().parents[3]/'third_party/dppo_official_cc7234a')
    adapter=importlib.import_module('uav_idp3.e5_runtime')

    class Predictor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.value=torch.nn.Parameter(torch.tensor(.25))
            self.inputs=None
        def forward(self, *, sample, timestep, local_cond, global_cond):
            self.inputs=(sample.detach().clone(),timestep.detach().clone(),global_cond.detach().clone())
            return self.value.expand_as(sample)

    scheduler=DDIMScheduler(num_train_timesteps=100,prediction_type='sample',clip_sample=False)
    model=Predictor()
    r=adapter.E5PPODiffusion(model=model,scheduler=scheduler,critic=torch.nn.Linear(576,1),
                           device='cpu',num_inference_steps=10,ft_denoising_steps=ft_steps)
    assert not hasattr(r,'alphas_cumprod')
    k=SelectiveKeep(bank(),coefficient=.1,seed=73)
    state=k.state_dict()
    indices=k.draw_indices()
    # Independently use the native scheduler's full time-indexed schedule.
    positions=torch.randint(10-ft_steps,10,(len(indices),),generator=k.generator)
    expected_t=scheduler.timesteps[positions]
    noise=torch.randn((len(indices),16,11),generator=k.generator)
    alpha=scheduler.alphas_cumprod[expected_t].reshape(-1,1,1)
    expected_xt=alpha.sqrt()*bank()['actions'][indices]+(1-alpha).sqrt()*noise
    k.load_state_dict(state)
    global_rng=torch.get_rng_state().clone()
    loss=k.loss(r,device='cpu')
    loss.backward()
    actual_xt,actual_t,actual_cond=model.inputs
    assert torch.equal(actual_t,expected_t)
    torch.testing.assert_close(actual_xt,expected_xt,rtol=0,atol=0)
    assert torch.equal(actual_cond,bank()['conditions'][indices])
    assert torch.equal(global_rng,torch.get_rng_state())
    assert loss.item()==pytest.approx(.75**2)
    assert model.value.grad.item()==pytest.approx(-1.5)


def test_keep_only_updates_executed_ft_channels_and_preserves_global_rng():
    r = runtime()
    k = SelectiveKeep(bank(), coefficient=.1, seed=73)
    state = torch.get_rng_state().clone()
    loss = k.loss(r, device='cpu')
    loss.backward()
    assert torch.equal(state, torch.get_rng_state())
    assert loss.item() == pytest.approx(1)
    g = r.actor_ft.x0.grad
    assert g[2:6, [0,1,2,9]].abs().sum() > 0
    assert g[:2].sum() == 0 and g[6:].sum() == 0
    assert g[:,3:9].sum() == 0 and g[:,10].sum() == 0


def test_reference_rng_resumes_exactly():
    k = SelectiveKeep(bank(), coefficient=.1, seed=73)
    saved = k.state_dict()
    first = k.draw_indices()
    k.load_state_dict(saved)
    assert torch.equal(first, k.draw_indices())


def test_calibration_uses_clipped_actual_ppo_and_leaves_parameters_rng_grads():
    trainer, r = make_trainer()
    r.actor_ft = lambda x,t,c: r.actor_parameter.expand_as(x)
    r.ddim_t = torch.tensor([9,3]); r.ddim_alphas = torch.tensor([.1,.7])
    r.predict_epsilon = False
    k = SelectiveKeep(bank(), coefficient=0, seed=73)
    b = batch(128)
    # The actual logprob clipping zeros the actor gradient above its +2 ceiling.
    with torch.no_grad(): r.actor_parameter.fill_(3.)
    state = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match='gradient'):
        calibrate_keep(trainer, [b]*4, k)
    assert torch.equal(state, torch.get_rng_state())
    assert r.actor_parameter.grad is None

    assert r.actor_parameter.item() == 3.
    with torch.no_grad(): r.actor_parameter.fill_(.02)
    result = calibrate_keep(trainer, [b]*4, k)
    assert result['coefficient'] > 0
    assert len(result['batches']) == 4
    assert trainer.counters.committed_updates == 0
    assert r.actor_parameter.grad is None


def attach_keep(trainer, r, coefficient):
    r.actor_ft = lambda x,t,c: r.actor_parameter.expand_as(x)
    r.ddim_t = torch.tensor([9,3]); r.ddim_alphas = torch.tensor([.1,.7])
    r.predict_epsilon = False
    trainer.keep = SelectiveKeep(bank(), coefficient=coefficient, seed=73)


def test_keep_added_once_per_logical_batch_and_no_extra_critic_update():
    direct, rd = make_trainer(); accum, ra = make_trainer(); control, rc = make_trainer()
    for r in (ra,rc): r.critic.load_state_dict(rd.critic.state_dict())
    attach_keep(direct,rd,.1); attach_keep(accum,ra,.1)
    md = direct.update_logical_batch(batch(),microbatch_pairs=257)
    ma = accum.update_logical_batch(batch(),microbatch_pairs=32)
    control.update_logical_batch(batch(),microbatch_pairs=32)
    assert md.keep_diagnostics['coefficient'] == .1
    assert ma.keep_diagnostics['keep_loss'] > 0
    torch.testing.assert_close(rd.actor_parameter,ra.actor_parameter)
    for x,y in zip(ra.critic.parameters(),rc.critic.parameters()): torch.testing.assert_close(x,y)
    assert direct.counters.committed_updates == accum.counters.committed_updates == 1


def test_zero_keep_is_exact_noop_and_rejection_restores_reference_rng():
    from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.trainer import TrainerConfig
    from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.update_transaction import UpdateRejected
    a, ra = make_trainer(); b, rb = make_trainer()
    rb.critic.load_state_dict(ra.critic.state_dict())
    attach_keep(b,rb,0.)
    rng = b.keep.generator.get_state().clone()
    a.update_logical_batch(batch(),microbatch_pairs=128)
    b.update_logical_batch(batch(),microbatch_pairs=128)
    assert torch.equal(ra.actor_parameter,rb.actor_parameter)
    assert torch.equal(rng,b.keep.generator.get_state())
    b.keep.coefficient = .1
    b.config = TrainerConfig(max_approx_kl=-1.)
    rng = b.keep.generator.get_state().clone()
    with pytest.raises(UpdateRejected): b.update_logical_batch(batch(),microbatch_pairs=128)
    assert torch.equal(rng,b.keep.generator.get_state())
