import unittest
from unittest.mock import patch

import numpy as np

from diffusion_policy_3d.env import kinematic_uav_env as kenv
from diffusion_policy_3d.rl.dppo_uav_rebuild.rewards.v2.facade import RewardV2Facade
from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.kinematic_adapter import KinematicAdapter


class PointCloud:
    def __init__(self):
        self.calls = 0

    def generate_with_metadata(self, *args, num_points=4096,
                               episode_identity_hash, decision_index, **kwargs):
        from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.observation_provenance import (
            PointCloudGenerationMeta, canonical_array_hash)
        self.calls += 1
        points = np.ones((num_points, 3), np.float32)
        return points, PointCloudGenerationMeta(
            total_rays=num_points, valid_hit_rays=num_points, no_ray_hit_fraction=0.,
            used_far_plane_fallback=False, finite=True,
            episode_identity_hash=episode_identity_hash, decision_index=decision_index,
            raw_array_hash=canonical_array_hash(points))


class RewardV2IntegrationTest(unittest.TestCase):
    def test_evaluation_continues_beyond_space_limits_then_reaches(self):
        pc = PointCloud()
        with patch.object(kenv, "load_obstacles", return_value=[]), patch.object(
            kenv.KinematicUAVEnv, "_load_spawn_offset", staticmethod(lambda _: np.zeros(3))):
            env = kenv.KinematicUAVEnv(
                synth_pc=pc, goal_world=np.array([34., 0., 7.], np.float32),
                reward_version="v2", disturb_enable=False, seed=1)
        env._drone_pos = np.array([29., 0., 5.], np.float32)
        env._episode_identity_hash = "e" * 64
        env._observation_frame_index = 7
        env._episode_rng_seed = 1
        initial = float(np.linalg.norm(env.goal_w - env._drone_pos))
        env._reward_history = dict(initial_distance=initial, best_distance=initial)
        facade = RewardV2Facade()
        adapter = KinematicAdapter(env, facade, facade.initial_state())
        from scripts.routea_paired_support.runtime import make_adapter
        with patch("scripts.evaluate_dppo_uav_formal_endpoint._build_adapter", return_value=adapter), patch(
            "scripts.routea_paired_support.runtime.file_hash", return_value="fixture"):
            adapter = make_adapter(None, None, {"world_id": 4, "obstacle_path": "fixture.json",
                                               "obstacle_sha256": "fixture"}, "anticipatory")
        self.assertFalse(env.terminate_at_space_limits)
        action = np.zeros((8, 11), np.float32)
        action[:, 0] = 1.
        action[:, 2] = .5
        _, _, terminated, truncated, info = adapter.step(
            type("Action", (), {"policy": action})())
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["done_reason"], "none")
        np.testing.assert_allclose(env._drone_pos, [33., 0., 7.])
        self.assertEqual(info["reward_terms"]["terminal"], 0.)
        action[:, 2] = 0.
        _, _, terminated, truncated, info = adapter.step(
            type("Action", (), {"policy": action})())
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["done_reason"], "reached")
        self.assertEqual(info["reward_terms"]["terminal"], 100.)
        np.testing.assert_allclose(env._drone_pos, [33.5, 0., 7.])

    def test_no_event_executes_exact_h4_and_keeps_tail_unexecuted(self):
        pc = PointCloud()
        with patch.object(kenv, "load_obstacles", return_value=[]), patch.object(
            kenv.KinematicUAVEnv, "_load_spawn_offset", staticmethod(lambda _: np.zeros(3))):
            env = kenv.KinematicUAVEnv(
                synth_pc=pc, goal_world=np.array([20., 0., 1.], np.float32),
                reward_version="v2", disturb_enable=False, seed=1)
        env._drone_pos = np.array([0., 0., 1.], np.float32)
        env._episode_identity_hash = "e" * 64
        env._observation_frame_index = 7
        env._episode_rng_seed = 1
        env._reward_history = dict(initial_distance=20., best_distance=20.)
        facade = RewardV2Facade()
        adapter = KinematicAdapter(env, facade, facade.initial_state())
        action = np.zeros((8, 11), np.float32)
        action[:, 0] = 1.
        _, reward, terminated, truncated, info = adapter.step(
            type("Action", (), {"policy": action})())
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["done_reason"], "none")
        self.assertEqual(info["executed_path"].shape, (5, 3))
        self.assertEqual(info["event_fraction"], None)
        self.assertEqual(pc.calls, 4)
        self.assertAlmostEqual(float(env._drone_pos[0]), 4., delta=1e-6)
        self.assertAlmostEqual(info["executed_time_s"], 1.2, delta=1e-6)
        self.assertAlmostEqual(info["reward_terms"]["newbest"], 2., delta=1e-6)
        self.assertAlmostEqual(reward, sum(info["reward_terms"].values()))

    def test_reset_then_h2_entry_uses_one_fraction_transaction(self):
        pc = PointCloud()
        with patch.object(kenv, "load_obstacles", return_value=[]), patch.object(
            kenv.KinematicUAVEnv, "_load_spawn_offset", staticmethod(lambda _: np.zeros(3))):
            env = kenv.KinematicUAVEnv(
                synth_pc=pc, goal_world=np.array([2., 0., 1.], np.float32),
                reward_version="v2", disturb_enable=False, mid_state_init_ratio=0., seed=1)
        facade = RewardV2Facade()
        adapter = KinematicAdapter(env, facade, facade.initial_state())
        env._last_astar_diagnostic = {}
        with patch.object(env, "_prepare_reset_candidate", return_value=(
            np.array([0., 0., 1.], np.float32), 0.,
            np.array([2., 0., 1.], np.float32))):
            obs, context = adapter.reset(seed=7)
        self.assertEqual(env._reward_history, dict(initial_distance=2., best_distance=2.))
        self.assertEqual(context["decision_index"], 7)
        self.assertEqual(len(context["pointcloud_meta_history"]), 8)
        action = np.zeros((8, 11), np.float32)
        action[0, 0] = 1.
        action[1, 0] = 2.
        action[0, 9] = .4
        action[1, 9] = .4
        action[0, 3] = 1.
        action[1, 3] = 3.
        obs, reward, terminated, truncated, info = adapter.step(
            type("Action", (), {"policy": action})())
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["event_segment"], 1)
        self.assertAlmostEqual(info["event_fraction"], .25, delta=1e-6)
        self.assertAlmostEqual(float(env._drone_pos[0]), 1.5, delta=1e-6)
        self.assertAlmostEqual(float(env._drone_vel[0]), 1.5, delta=1e-6)
        self.assertAlmostEqual(env._drone_yaw, .5, delta=1e-6)
        self.assertAlmostEqual(info["executed_time_s"], .375, delta=1e-6)
        self.assertEqual(info["executed_path"].shape, (3, 3))
        self.assertEqual(pc.calls, 3)
        self.assertEqual(info["decision_index"], 9)
        self.assertEqual(len(info["pointcloud_meta_history"]), 8)
        self.assertEqual(info["reward_metrics"]["best_after"], .5)
        self.assertAlmostEqual(reward, sum(info["reward_terms"].values()))

    def test_h1_entry_commits_fraction_and_stops_observation_frames(self):
        pc = PointCloud()
        with patch.object(kenv, "load_obstacles", return_value=[]), patch.object(
            kenv.KinematicUAVEnv, "_load_spawn_offset", staticmethod(lambda _: np.zeros(3))):
            env = kenv.KinematicUAVEnv(
                synth_pc=pc, goal_world=np.array([2., 0., 1.], np.float32),
                reward_version="v2", disturb_enable=False, mid_state_init_ratio=0., seed=1)
        env._drone_pos = np.array([0., 0., 1.], np.float32)
        env._episode_start_world = env._drone_pos.copy()
        env._episode_identity_hash = "e" * 64
        env._observation_frame_index = 7
        env._episode_rng_seed = 1
        facade = RewardV2Facade()
        state = facade.initial_state()
        env._reward_history = dict(initial_distance=2., best_distance=2.)
        adapter = KinematicAdapter(env, facade, state)
        action = np.zeros((8, 11), np.float32)
        action[0, 0] = 3.
        obs, reward, terminated, truncated, info = adapter.step(type("Action", (), {"policy": action})())
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["transition_facts"].done_reason, "reached")
        self.assertAlmostEqual(float(info["executed_path"][-1, 0]), 1.5, delta=1e-6)
        self.assertAlmostEqual(info["event_fraction"], .5, delta=1e-6)
        self.assertEqual(pc.calls, 1)
        self.assertAlmostEqual(reward, sum(info["reward_terms"].values()))
        self.assertIsNotNone(obs)


if __name__ == "__main__":
    unittest.main()
