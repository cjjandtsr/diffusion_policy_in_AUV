"""Real env/adapter transitions; only world I/O and point-cloud cost are replaced."""

from types import SimpleNamespace

import numpy as np
import pytest

from diffusion_policy_3d.env import kinematic_uav_env as kenv
from diffusion_policy_3d.env.reward_v17 import resolve_config
from diffusion_policy_3d.env.reward_v17_support.density import build_density
from diffusion_policy_3d.rl.dppo_uav_rebuild.rewards.v17.facade import RewardV17Facade
from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.kinematic_adapter import KinematicAdapter


class PointCloud:
    def __init__(self):
        self.calls = 0

    def synthesize(self, *args, num_points=4096, **kwargs):
        self.calls += 1
        return np.zeros((num_points, 3), np.float32)

    def generate_with_metadata(
        self, *args, num_points=4096, episode_identity_hash, decision_index, **kwargs
    ):
        from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.observation_provenance import (
            PointCloudGenerationMeta,
            canonical_array_hash,
        )
        self.calls += 1
        points = np.ones((num_points, 3), np.float32)
        return points, PointCloudGenerationMeta(
            total_rays=num_points,
            valid_hit_rays=num_points,
            no_ray_hit_fraction=0.0,
            used_far_plane_fallback=False,
            finite=True,
            episode_identity_hash=episode_identity_hash,
            decision_index=decision_index,
            raw_array_hash=canonical_array_hash(points),
        )


def adapter(monkeypatch, start, goal):
    cfg = resolve_config(dict(gamma_pbrs=.99, d_ref=.7, s_f=.4, w_low=2.5, w_high=1., w_climb=.1))
    den = build_density([dict(episode_id="demo", split="bc_demo", source="frozen_bc", start=[0, 0, 1],
                             goal=[10, 0, 1], initial_yaw=0, ground_z=0, positions=[[0, 0, 1]])], dataset_sha256="a" * 64)
    monkeypatch.setattr(kenv, "load_obstacles", lambda _: [])
    monkeypatch.setattr(kenv.KinematicUAVEnv, "_load_spawn_offset", staticmethod(lambda _: np.zeros(3)))
    pc = PointCloud()
    env = kenv.KinematicUAVEnv(synth_pc=pc, goal_world=np.array(goal, np.float32),
                              reward_version="v17", reward_v17_cfg=cfg, offline_density=den,
                              disturb_enable=False, mid_state_init_ratio=0., seed=1)
    env._drone_pos = np.array(start, np.float32)
    env._episode_start_world = env._drone_pos.copy()
    env._episode_identity_hash = "e" * 64
    env._observation_frame_index = 7
    facade = RewardV17Facade()
    state = facade.initial_state(cfg)
    env._reward_history = state.history
    return KinematicAdapter(env, facade, state, cfg), pc


def test_goal_terminal_keeps_progress_pbrs_and_new_history(monkeypatch):
    a, _ = adapter(monkeypatch, [0, 0, 1], [1, 0, 1])
    action = np.zeros((8, 11), np.float32)
    action[:4, 0] = .2
    original = action.copy()
    _, reward, terminated, truncated, info = a.step(SimpleNamespace(policy=action))
    assert terminated and not truncated
    assert info["reward_terms"]["terminal"] == 100
    assert info["reward_terms"]["progress"] > 0
    assert info["reward_terms"]["pbrs_free"] < 0
    assert reward == pytest.approx(sum(info["reward_terms"].values()))
    assert a.reward_state.history["episode_steps"] == 1
    np.testing.assert_array_equal(action, original)


def test_goal_with_point_four_vertical_error_uses_single_3d_arrival_ball(monkeypatch):
    """Catches the v1.7 environment retaining the legacy abs(dz)<0.3 gate."""

    a, _ = adapter(monkeypatch, [0, 0, 1.0], [0, 0, 1.4])
    action = np.zeros((8, 11), np.float32)
    _, _, terminated, truncated, info = a.step(SimpleNamespace(policy=action))
    assert terminated is True and truncated is False
    assert info["transition_facts"].done_reason == "reached"


def test_high_altitude_and_intermediate_ground_are_distinct_failures(monkeypatch):
    a, _ = adapter(monkeypatch, [0, 0, 100], [10, 0, 100])
    action = np.zeros((8, 11), np.float32)
    action[:4, 0] = .1
    _, _, done, _, info = a.step(SimpleNamespace(policy=action))
    assert done is True
    assert info["transition_facts"].done_reason == "altitude_exceed"
    a, _ = adapter(monkeypatch, [0, 0, 1], [10, 0, 1])
    action[:4, 2] = [-1.2, 1.2, 0, 0]
    _, _, done, _, info = a.step(SimpleNamespace(policy=action))
    assert done and info["transition_facts"].done_reason == "ground_crash"
    assert info["transition_facts"].raw_done_reason == "crash_ground"
    assert info["reward_terms"]["support_escrow"] == 0


def test_nan_is_terminal_before_pointcloud_and_observation_is_not_fabricated(monkeypatch):
    a, pc = adapter(monkeypatch, [0, 0, 1], [10, 0, 1])
    action = np.zeros((8, 11), np.float32)
    action[0, 0] = np.nan
    obs, reward, done, truncated, info = a.step(SimpleNamespace(policy=action))
    assert obs is None and done and not truncated and reward <= -50
    assert pc.calls == 0
    assert not info["reward_metrics"]["training_usable"]
    assert info["transition_facts"].raw_done_reason == "invalid_state"
