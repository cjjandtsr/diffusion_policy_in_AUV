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
