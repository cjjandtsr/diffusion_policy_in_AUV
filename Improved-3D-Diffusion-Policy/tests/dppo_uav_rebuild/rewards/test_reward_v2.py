import numpy as np
import unittest

from diffusion_policy_3d.env.collision_geom import ObstacleSpec
from diffusion_policy_3d.env.reward_v2_support.first_event import locate_first_event
from diffusion_policy_3d.env.reward_v2_support.reward import compute_reward_v2
from diffusion_policy_3d.env.reward_v2_support.reward import _obstacle_clearances
from diffusion_policy_3d.env.reward_v17_support.geometry import segment_clearances


class RewardV2ContractTest(unittest.TestCase):
  def test_batched_clearance_matches_individual_segments(self):
    path = np.array([[0., 0., 1.], [1., .2, 1.], [2., .1, 2.],
                     [3., .4, 1.], [4., 0., 1.]])
    solids = [
      ObstacleSpec("c", "cylinder", 1.5, 0., 1., radius=.3, height=2.),
      ObstacleSpec("b", "box", 3., 0., 1., yaw=.4,
                   size_x=.7, size_y=.7, size_z=2.),
    ]
    sphere = ObstacleSpec("s", "sphere", 2.5, .2, 1.5, radius=.2)
    expected = []
    for start, end in zip(path[:-1], path[1:]):
      solid = float(segment_clearances(start[None], end[None], solids).min())
      delta = end - start
      centre = np.array([sphere.x, sphere.y, sphere.z])
      t = np.clip(np.dot(centre - start, delta) / np.dot(delta, delta), 0., 1.)
      ball = np.linalg.norm(start + t * delta - centre) - sphere.radius
      expected.append(min(solid, ball))
    np.testing.assert_allclose(_obstacle_clearances(path, solids + [sphere]),
                               expected, rtol=0., atol=1e-12)

  def test_first_event_prefers_collision_before_goal_in_same_segment(self):
    path = np.array([[0., 0., 1.], [10., 0., 1.]])
    obstacle = ObstacleSpec("early", "cylinder", 3., 0., 1., radius=.5, height=2.)
    event = locate_first_event(path, [7., 0., 1.], [obstacle], .1, 5.)
    self.assertEqual(event.kind, "collision")
    self.assertEqual(event.segment_index, 0)
    self.assertAlmostEqual(event.fraction, .24, delta=1e-6)


  def test_first_event_stops_at_goal_before_later_collision(self):
    path = np.array([[0., 0., 1.], [10., 0., 1.]])
    obstacle = ObstacleSpec("late", "cylinder", 9., 0., 1., radius=.5, height=2.)
    event = locate_first_event(path, [3., 0., 1.], [obstacle], .1, 5.)
    self.assertEqual(event.kind, "reached")
    self.assertAlmostEqual(event.fraction, .25, delta=1e-6)


  def test_first_event_detects_between_waypoint_sphere_obstacle(self):
    path = np.array([[0., 0., 1.], [10., 0., 1.]])
    obstacle = ObstacleSpec("ball", "sphere", 3., 0., 1., radius=.5)
    event = locate_first_event(path, [8., 0., 1.], [obstacle], .1, 5.)
    self.assertEqual(event.kind, "collision")
    self.assertAlmostEqual(event.fraction, .24, delta=1e-6)

  def test_same_fraction_physical_failure_wins(self):
    path = np.array([[0., 0., 1.], [10., 0., 1.]])
    obstacle = ObstacleSpec("tie", "cylinder", 3.1, 0., 1., radius=.5, height=2.)
    event = locate_first_event(path, [3., 0., 1.], [obstacle], .1, 5.)
    self.assertEqual(event.kind, "collision")
    self.assertAlmostEqual(event.fraction, .25, delta=1e-6)

  def test_ground_bounds_and_altitude_intersections(self):
    cases = (
      ([[0., 0., 1.], [0., 0., 0.]], "ground_crash", .8),
      ([[29., 0., 1.], [31., 0., 1.]], "out_of_bounds", .5),
      ([[0., 0., 5.], [0., 0., 7.]], "altitude_exceed", .5),
    )
    for path, kind, fraction in cases:
      with self.subTest(kind=kind):
        event = locate_first_event(path, [20., 20., 1.], [], .1, 5.)
        self.assertEqual(event.kind, kind)
        self.assertAlmostEqual(event.fraction, fraction, delta=1e-6)

  def test_evaluation_can_reach_beyond_old_space_limits(self):
    cases = (
      ([[0., 0., 5.5], [0., 0., 7.5]], [0., 0., 7.4], [0., 0., 6.9]),
      ([[29.5, 0., 2.], [32.5, 0., 2.]], [32., 0., 2.], [31.5, 0., 2.]),
    )
    for path, goal, position in cases:
      with self.subTest(goal=goal):
        event = locate_first_event(path, goal, [], .1, 5.,
                                   terminate_at_space_limits=False)
        self.assertEqual(event.kind, "reached")
        np.testing.assert_allclose(event.position, position, atol=1e-6)

  def test_evaluation_keeps_real_collisions_beyond_old_limits(self):
    path = [[29., 0., 7.], [35., 0., 7.]]
    obstacle = ObstacleSpec("high_ball", "sphere", 32., 0., 7., radius=.5)
    event = locate_first_event(path, [34., 0., 7.], [obstacle], .1, 5.,
                               terminate_at_space_limits=False)
    self.assertEqual(event.kind, "collision")
    self.assertAlmostEqual(event.position[0], 31.4, delta=1e-6)

  def test_rotated_box_first_contact(self):
    path = np.array([[0., 0., 1.], [10., 0., 1.]])
    box = ObstacleSpec("rotated", "box", 3., 0., 1., yaw=np.pi / 4,
                       size_x=1., size_y=1., size_z=2.)
    event = locate_first_event(path, [9., 0., 1.], [box], .1, 5.)
    self.assertEqual(event.kind, "collision")
    self.assertAlmostEqual(event.fraction, (3. - np.sqrt(.5) - .1) / 10., delta=1e-6)


  def test_newbest_cannot_be_reearned_after_backtracking(self):
    first = np.array([[0., 0., 1.], [2., 0., 1.]])
    back = np.array([[2., 0., 1.], [1., 0., 1.]])
    repeat = np.array([[1., 0., 1.], [2., 0., 1.]])
    planned = np.array([[0., 0., 1.]] * 9)
    goal = np.array([10., 0., 1.])
    values = []
    best = 10.
    for path in (first, back, repeat):
        total, terms, metrics, best = compute_reward_v2(
            path, planned, goal, [], best, 10., "none", .1, 0.)
        values.append(terms["newbest"])
        self.assertAlmostEqual(total, sum(terms.values()))
        self.assertTrue(0 <= metrics["risk"] <= 1)
    np.testing.assert_allclose(values, [2., 0., 0.])


  def test_risk_uses_max_of_prefix_and_tail_not_sum(self):
    executed = np.array([[0., 0., .3], [1., 0., .3]])
    planned = np.array([[0., 0., .3], [1., 0., .3], [2., 0., .3],
                        [3., 0., .3], [4., 0., .3], [5., 0., .3],
                        [6., 0., .3], [7., 0., .3], [8., 0., .3]])
    _, _, metrics, _ = compute_reward_v2(
        executed, planned, [20., 0., .3], [], 20., 20., "reached", .1, 0.)
    self.assertAlmostEqual(metrics["risk"], metrics["risk_exec"])
    self.assertEqual(metrics["risk_tail"], 0.)

  def test_tail_risk_provenance_and_timeout(self):
    executed = np.array([[0., 0., 1.], [1., 0., 1.], [2., 0., 1.],
                         [3., 0., 1.], [4., 0., 1.]])
    planned = np.array([[float(i), 0., 1.] for i in range(9)])
    obstacle = ObstacleSpec("tail", "sphere", 6., 0., 1., radius=.3)
    _, terms, metrics, _ = compute_reward_v2(
        executed, planned, [20., 0., 1.], [obstacle], 20., 20., "none")
    self.assertGreater(metrics["risk_tail"], metrics["risk_exec"])
    self.assertEqual(metrics["risk_source"], "tail")
    self.assertEqual(metrics["risk_type"], "obstacle")
    self.assertIn(metrics["argmax_segment"], (6, 7))
    self.assertEqual(terms["terminal"], 0.)
    _, timeout_terms, timeout_metrics, _ = compute_reward_v2(
        executed, planned, [20., 0., 1.], [obstacle], 20., 20., "time_limit")
    self.assertEqual(timeout_terms["terminal"], 0.)
    self.assertEqual(timeout_metrics["risk_tail"], 0.)

  def test_worst_late_success_outscores_best_timeout_bound(self):
    gamma = .99
    q = gamma ** 59
    s = sum(gamma ** t for t in range(60))
    self.assertGreater(100 * q - s - (1 - q) * 10, 0.)


if __name__ == "__main__":
    unittest.main()
