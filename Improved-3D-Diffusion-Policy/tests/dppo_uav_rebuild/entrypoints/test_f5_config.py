import copy
import pytest
from omegaconf import OmegaConf
from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.config_schema_v2 import validate_v2_config
from diffusion_policy_3d.rl.dppo_uav_rebuild.checkpointing.a2_fork import validate_child_config


def config(run_id='P8'):
    c = OmegaConf.load('diffusion_policy_3d/rl/dppo_uav_rebuild/config/train_e5_dppo_common_b2560_seed202609203_v1.yaml')
    c.environment.max_no_ray_hit_fraction = .9
    c.environment.pointcloud_calibration_sha256 = 'a' * 64
    c.run.seed = 202610311 if run_id in ('P8', 'P9') else 202610312
    c.run.checkpoint_updates = sorted(set(c.run.checkpoint_updates) | {129})
    c.a2 = dict(arm='T', parent_checkpoint='parent', parent_receipt='receipt',
                parent_receipt_sha256='a'*64, parent_source_root='source',
                reference_bank='reference', reference_sha256='b'*64, keep_seed=2026102301)
    c.f5 = dict(run_id=run_id, keep_coefficient=.01061006)
    if run_id in ('P8', 'P10'):
        c.reward_version = 'v3'
        c.task_bank = dict(enabled=True, path='bank.json', sha256='c'*64,
                           seed=int(c.run.seed)+17,
                           mixture=dict(original=.15, L0=.10, L1=.15,
                                        L23_random=.30, L23_near=.20, L4=.10))
    return c


@pytest.mark.parametrize('run_id', ['P8','P9','P10','P11'])
def test_registered_configs_keep_global_schedule(run_id):
    c = config(run_id)
    assert validate_v2_config(c).logical_batch_pairs == 2560
    assert c.run.updates == 256


@pytest.mark.parametrize('field,value', [('run.seed',202610261),('run.updates',192),
                                      ('f5.keep_coefficient',0),('dppo.clip_ploss_coef',.05)])
def test_f5_rejects_unregistered_changes(field, value):
    c = config()
    OmegaConf.update(c, field, value)
    with pytest.raises(ValueError): validate_v2_config(c)


def test_f5_fork_changes_only_registered_distribution_reward_and_seed():
    child = OmegaConf.to_container(config(), resolve=True)
    parent = copy.deepcopy(child)
    parent.pop('f5'); parent.pop('task_bank'); parent.pop('a2')
    parent['reward_version'] = 'v2'; parent['run']['seed'] = 202609203
    validate_child_config(parent, child)
    child['reward']['r_reach'] = 101
    with pytest.raises(ValueError): validate_child_config(parent, child)
