"""Cost belongs to actual elapsed H4 motion, including fractional first events."""

from unittest.mock import patch

import numpy as np
import pytest

from diffusion_policy_3d.env import kinematic_uav_env as kenv
from diffusion_policy_3d.env.reward_v2_support import reward as reward_module
from diffusion_policy_3d.rl.dppo_uav_rebuild.rewards.v2.facade import RewardV2Facade
from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.kinematic_adapter import KinematicAdapter
from test_reward_v2_integration import PointCloud


def test_space_cost_zero_on_segment_bounded_off_segment_and_uses_duration():
    assert hasattr(reward_module, "space_running_cost")
    cost = reward_module.space_running_cost
    start, goal = np.array([0., 0., 2.]), np.array([10., 0., 2.])
    assert cost([start, [5., 0., 2.], goal], start, goal, [.3, .3], 1., 72.) == 0.
    path = np.array([[0., 5., 2.], [5., 5., 2.], [10., 5., 2.]])
    full = cost(path, start, goal, [.3, .3], 1., 72.)
    partial = cost(path, start, goal, [.3, .075], 1., 72.)
    assert -.6 / 72. < full < partial < 0.
    assert cost(path, start, goal, [36., 36.], 1., 72.) >= -1.
    assert cost(path, start, goal, [.3, .3], 0., 72.) == 0.  # Ablation only.


@pytest.mark.parametrize("dx,segment,fraction", [(3., 0, .5), (.4, 3, .75)])
def test_environment_fractional_event_cost_excludes_tail(dx, segment, fraction):
    pc = PointCloud()
    with patch.object(kenv, "load_obstacles", return_value=[]), patch.object(
        kenv.KinematicUAVEnv, "_load_spawn_offset", staticmethod(lambda _: np.zeros(3))):
        env = kenv.KinematicUAVEnv(synth_pc=pc, goal_world=np.array([2., 0., 2.]),
                                  reward_version="v2", disturb_enable=False, seed=1,
                                  terminate_at_space_limits=False, space_cost_budget=1.)
    env._drone_pos = np.array([0., 0., 2.], np.float32)
    env._episode_start_world = np.array([0., 2., 2.], np.float32)
    env._episode_identity_hash = "e" * 64
    env._observation_frame_index = 7
    env._episode_rng_seed = 1
    initial = float(np.linalg.norm(env.goal_w - env._episode_start_world))
    env._reward_history = dict(initial_distance=initial, best_distance=2.)
    facade = RewardV2Facade(space_cost_budget=1.)
    adapter = KinematicAdapter(env, facade, facade.initial_state())
    action = np.zeros((8, 11), np.float32)
    action[:, 0] = dx
    action[4:, 1] = 100.  # Unexecuted tail must not pay space cost.
    _, total, terminated, truncated, info = adapter.step(type("Action", (), {"policy": action})())
    assert terminated and not truncated and info["done_reason"] == "reached"
    assert info["event_segment"] == segment
    assert info["event_fraction"] == pytest.approx(fraction, abs=1e-6)
    durations = [.3] * segment + [.3 * info["event_fraction"]]
    expected = reward_module.space_running_cost(info["executed_path"], env._episode_start_world,
                                               env.goal_w, durations, 1., 72.)
    assert info["reward_terms"]["space"] == pytest.approx(expected)
    assert info["reward_metrics"]["space_elapsed_s"] == pytest.approx(sum(durations))
    assert total == pytest.approx(sum(info["reward_terms"].values()))


def test_training_open_space_runs_to_real_deadline():
    pc = PointCloud()
    with patch.object(kenv, "load_obstacles", return_value=[]), patch.object(
        kenv.KinematicUAVEnv, "_load_spawn_offset", staticmethod(lambda _: np.zeros(3))):
        env = kenv.KinematicUAVEnv(synth_pc=pc, goal_world=np.array([0., 0., 2.]),
                                  reward_version="v2", disturb_enable=False, seed=1,
                                  terminate_at_space_limits=False, space_cost_budget=1.)
    env._drone_pos = np.array([29., 0., 5.], np.float32)
    env._episode_start_world = env._drone_pos.copy()
    env._episode_identity_hash = "e" * 64
    env._observation_frame_index = 7
    env._episode_rng_seed = 1
    initial = float(np.linalg.norm(env.goal_w - env._drone_pos))
    env._reward_history = dict(initial_distance=initial, best_distance=initial)
    env.episode_step = 59
    action = np.zeros((8, 11), np.float32)
    action[:, 0], action[:, 2] = 1., .5
    _, _, terminated, truncated, info = env.step(action)
    assert not terminated and truncated and info["done_reason"] == "time_limit"
    assert info["reward_terms"]["terminal"] == 0.
    assert -1.2 / 72. < info["reward_terms"]["space"] < 0.
    np.testing.assert_allclose(env._drone_pos, [33., 0., 7.])
