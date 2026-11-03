from types import SimpleNamespace

import pytest

from scripts.routea_paired_support import runtime


@pytest.mark.parametrize('version', ['v2', 'v3'])
def test_common_environment_accepts_reward_only_extension(monkeypatch, version):
    from scripts import evaluate_dppo_uav_formal_endpoint as endpoint
    env = SimpleNamespace(reward_version=version, n_exec_steps=4, rl_action_steps=8,
                          traj_dt=.3, max_episode_steps=60, collision_thresh=.1)
    adapter = SimpleNamespace(environment=env)
    monkeypatch.setattr(runtime, 'file_hash', lambda path: 'geometry')
    monkeypatch.setattr(endpoint, '_build_adapter', lambda *args: adapter)
    world = dict(obstacle_path='world.json', obstacle_sha256='geometry', world_id='train')
    assert runtime.make_adapter(None, None, world, 'dense_recovery') is adapter
    assert env.disturb_enable is False and env.terminate_at_space_limits is False


@pytest.mark.parametrize('field,value', [('reward_version', 'v17'), ('n_exec_steps', 8),
    ('rl_action_steps', 4), ('traj_dt', .6), ('max_episode_steps', 61), ('collision_thresh', .2)])
def test_common_environment_keeps_physical_contract(monkeypatch, field, value):
    from scripts import evaluate_dppo_uav_formal_endpoint as endpoint
    env = SimpleNamespace(reward_version='v3', n_exec_steps=4, rl_action_steps=8,
                          traj_dt=.3, max_episode_steps=60, collision_thresh=.1)
    setattr(env, field, value)
    monkeypatch.setattr(runtime, 'file_hash', lambda path: 'geometry')
    monkeypatch.setattr(endpoint, '_build_adapter', lambda *args: SimpleNamespace(environment=env))
    with pytest.raises(ValueError, match='common environment'):
        runtime.make_adapter(None, None, dict(obstacle_path='world.json',
            obstacle_sha256='geometry', world_id='train'), 'dense_recovery')
