import copy
import pytest
from diffusion_policy_3d.rl.dppo_uav_rebuild.checkpointing.a2_fork import validate_child_config


def test_fork_allows_only_keep_and_output_identity_changes():
    p = dict(output_dir='old',result_dir='old_results', run=dict(updates=256,seed=3),
             optimizer=dict(actor_lr=2e-5), reward=dict(r_reach=100))
    c = copy.deepcopy(p); c.update(output_dir='C',result_dir='C_results',a2={'arm':'C'})
    validate_child_config(p,c)
    c['optimizer']['actor_lr']=1e-5
    with pytest.raises(ValueError): validate_child_config(p,c)


def test_fork_cannot_change_seed_budget_or_reward():
    p = dict(run=dict(updates=256,seed=3),reward=dict(r_reach=100))
    for key in ('updates','seed'):
        c=copy.deepcopy(p); c['run'][key]+=1
        with pytest.raises(ValueError): validate_child_config(p,c)
    c=copy.deepcopy(p); c['reward']['r_reach']=0
    with pytest.raises(ValueError): validate_child_config(p,c)


def test_xk_allows_only_declared_experiment_changes():
    p = dict(run=dict(updates=256, seed=3, checkpoint_every_updates=32),
             dppo=dict(denoised_clip_value=1., min_sampling_std=.1),
             reward=dict(r_reach=100))
    c = copy.deepcopy(p)
    c['xk'] = dict(arm='X', actor_lr_scale=.8)
    c['run'].update(seed=202610261, checkpoint_every_updates=8)
    c['dppo'].update(ppo_inert_mask=dict(enabled=True), self_imitation=dict(enabled=True),
                     min_std_vector=[.1]*11, denoised_clip_value=2.)
    validate_child_config(p,c)
    for section, key, value in [('run','updates',384), ('reward','r_reach',0),
                                ('dppo','min_sampling_std',.3)]:
        bad = copy.deepcopy(c); bad[section][key] = value
        with pytest.raises(ValueError): validate_child_config(p,bad)
