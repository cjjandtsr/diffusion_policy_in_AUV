import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf

from diffusion_policy_3d.env.collision_geom import ObstacleSpec
from diffusion_policy_3d.env import kinematic_uav_env as kenv
from diffusion_policy_3d.env.reward_v2_support.reward import compute_reward_v2
from diffusion_policy_3d.env.reward_v3 import compute_reward_v3
from diffusion_policy_3d.rl.dppo_uav_rebuild.rewards.select import select_reward
from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.kinematic_adapter import KinematicAdapter
from tests.dppo_uav_rebuild.rewards.test_reward_v2_integration import PointCloud


class RewardV3Test(unittest.TestCase):
    def setUp(self):
        self.wall = ObstacleSpec('wall', 'box', 2., 0., 2.,
                                 size_x=1., size_y=20., size_z=4.)
        self.goal = [10., 0., 2.]
        self.planned = np.array([[0., 0., 2.]] * 9)

    def reward(self, path, obstacles=None):
        return compute_reward_v3(path, self.planned, self.goal,
                                 [self.wall] if obstacles is None else obstacles,
                                 10., 10., 'none')

    def test_close_formula_and_v2_terms_unchanged(self):
        path = [[.5, 0., 2.], [.8, 0., 2.], [1., 0., 2.]]
        total, terms, metrics, best = self.reward(path)
        old, old_terms, old_metrics, old_best = compute_reward_v2(
            path, self.planned, self.goal, [self.wall], 10., 10., 'none')
        expected = -4. * (.3 * ((1.2-.7)/1.1)**2 + .2 * ((1.2-.5)/1.1)**2)
        self.assertAlmostEqual(terms['risk_close'], expected)
        self.assertAlmostEqual(total, old+expected)
        self.assertEqual({k: terms[k] for k in old_terms}, old_terms)
        self.assertEqual({k: metrics[k] for k in old_metrics}, old_metrics)
        self.assertEqual(best, old_best)
        self.assertAlmostEqual(metrics['closing_m'], .5)
        self.assertAlmostEqual(metrics['min_clearance_m'], .5)

    def test_retreat_constant_clearance_and_no_obstacles_have_zero_cost(self):
        for path in ([[1., 0., 2.], [.5, 0., 2.]],
                     [[1., 0., 2.], [1., 1., 2.]]):
            self.assertEqual(self.reward(path)[1]['risk_close'], 0.)
        # Ground is deliberately excluded from the extra term.
        self.assertEqual(self.reward([[0., 0., 2.], [0., 0., .2]], [])[1]['risk_close'], 0.)

    def test_cost_is_capped_per_decision(self):
        result = self.reward([[0., 0., 2.], [1.4, 0., 2.]])
        self.assertEqual(result[1]['risk_close'], -2.)

    def test_select_and_fractional_collision_use_actual_path_once(self):
        pc = PointCloud()
        config = OmegaConf.create({'reward_version': 'v3',
                                  'task_protocol': {'space_cost_budget': 1.}})
        facade = select_reward(config, '.')
        self.assertEqual(facade.version, 'v3')
        with patch.object(kenv, 'load_obstacles', return_value=[self.wall]), patch.object(
                kenv.KinematicUAVEnv, '_load_spawn_offset', staticmethod(lambda _: np.zeros(3))):
            env = kenv.KinematicUAVEnv(synth_pc=pc, goal_world=np.array(self.goal, np.float32),
                                      reward_version='v3', disturb_enable=False,
                                      space_cost_budget=1., terminate_at_space_limits=False, seed=1)
        env._drone_pos = np.array([.5, 0., 2.], np.float32)
        env._episode_start_world = env._drone_pos.copy()
        env._episode_identity_hash = 'e'*64
        env._observation_frame_index = 7
        env._episode_rng_seed = 1
        env._reward_history = dict(initial_distance=9.5, best_distance=9.5)
        action = np.zeros((8, 11), np.float32)
        action[:, 0] = 3.
        adapter = KinematicAdapter(env, facade, facade.initial_state())
        _, total, terminated, truncated, info = adapter.step(type('Action', (), {'policy': action})())
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info['done_reason'], 'collision')
        self.assertEqual(info['reward_version'], 'v3')
        self.assertAlmostEqual(info['executed_path'][-1, 0], 1.4, places=5)
        self.assertAlmostEqual(info['reward_metrics']['closing_m'], .9, places=5)
        self.assertAlmostEqual(info['reward_metrics']['min_clearance_m'], .1, places=5)
        self.assertEqual(info['reward_terms']['risk_close'], -2.)
        self.assertEqual(total, sum(info['reward_terms'].values()))
        self.assertEqual(pc.calls, 1)


if __name__ == '__main__':
    unittest.main()
