"""Acceptance probes for the paper supplement's sampling and first-event contract."""
import unittest
import importlib.util

from scripts.routea_testext_task_bank import select_tasks, draw_candidate
from scripts.routea_gazebo_offline import classify_odom, check_sampling
from scripts.audit_routea_paper_eval import nearest_goal_time
from scripts.collect_routea_paper_inventory import is_evidence_export
from scripts.probe_routea_paper_checkpoints import validate_plan


class PaperSupplementTests(unittest.TestCase):
    def test_freeze_enters_common_render_mode_before_loading(self):
        import sys
        import torch
        from unittest.mock import patch
        from scripts import routea_testext_task_bank as entry
        observed = []
        argv = ['bank', 'freeze', '--candidates', 'unused', '--repo', 'unused',
                '--source-root', 'unused', '--checkpoint', 'unused', '--output', 'unused',
                '--expected-update', '256']
        with torch.jit.optimized_execution(True), patch.object(sys, 'argv', argv), \
                patch.object(entry, 'freeze', lambda args: observed.append(torch._C._get_graph_executor_optimize())):
            entry.main()
        self.assertEqual(observed, [False])

    def test_h3_rejects_an_incomplete_model_plan(self):
        with self.assertRaises(ValueError):
            validate_plan(dict(models=[], c5_models=[]))

    def test_inventory_excludes_its_prior_exports(self):
        from pathlib import Path
        self.assertTrue(is_evidence_export(Path('/results/routea_paper_prereq_run01/evidence/history/plan.json')))
        self.assertTrue(is_evidence_export(Path('/results/routea_paper_prereq_run02/evidence/history/admission.json')))
        self.assertFalse(is_evidence_export(Path('/results/routea_training/run/input_snapshot/admission.json')))

    def test_nearest_goal_time_uses_continuous_executed_prefix(self):
        rows = [dict(executed_path=[[0., 0., 2.], [2., 0., 2.]], executed_time_s=.3)]
        distance, seconds = nearest_goal_time(rows, [1., 1., 2.])
        self.assertAlmostEqual(distance, 1.)
        self.assertAlmostEqual(seconds, .15)
        rows = [dict(executed_path=[[0., 0., 2.], [1.2, 0., 2.]], executed_time_s=.12)]
        self.assertAlmostEqual(nearest_goal_time(rows, [1., 1., 2.])[1], .1)

    @unittest.skipUnless(importlib.util.find_spec('gymnasium'), 'requires existing idp3 gymnasium')
    def test_real_admission_rejects_overlap_and_accepts_clear_endpoints(self):
        from scripts.routea_task_bank import admit
        from diffusion_policy_3d.env.collision_geom import ObstacleSpec
        box = ObstacleSpec(name='box', shape='box', x=0., y=0., z=2.,
                           size_x=1., size_y=1., size_z=1.)
        self.assertFalse(admit([0., 0., 2.], [3., 0., 2.], [box], 'anticipatory'))
        self.assertTrue(admit([-3., 0., 2.], [3., 0., 2.], [box], 'anticipatory'))

    def test_selection_is_reproducible_and_respects_world_caps(self):
        rows = [dict(world='w%02d' % w, category=c, candidate=i)
                for w in range(30) for c in ('zero', 'low', 'high') for i in range(4)]
        chosen = select_tasks(rows)
        self.assertEqual(chosen, select_tasks(rows))
        self.assertEqual(len(chosen), 60)
        for world in {r['world'] for r in chosen}:
            subset = [r for r in chosen if r['world'] == world]
            self.assertLessEqual(len(subset), 4)
            for category in ('zero', 'low', 'high'):
                self.assertLessEqual(sum(r['category'] == category for r in subset), 2)

    def test_candidate_distribution_and_repeatability(self):
        import random
        first = draw_candidate(random.Random(20261103))
        self.assertEqual(first, draw_candidate(random.Random(20261103)))
        start, goal, yaw = first
        self.assertTrue(all(-14 <= x <= 14 for x in start[:2] + goal[:2]))
        self.assertTrue(1.3 <= start[2] <= 4.7)
        self.assertTrue(1.5 <= goal[2] <= 4.5)

    def test_first_arrival_before_later_collision(self):
        from diffusion_policy_3d.env.collision_geom import ObstacleSpec
        obs = [ObstacleSpec(name='wall', shape='box', x=2., y=0., z=2.,
                            size_x=.2, size_y=2., size_z=2.)]
        result = classify_odom([0., 1.], [[0., 0., 2.], [3., 0., 2.]], [1., 0., 2.], obs)
        self.assertEqual(result['done_reason'], 'reached')
        self.assertAlmostEqual(result['event_time_s'], 1 / 6, places=5)
        result = classify_odom([0., 1.], [[0., 0., 2.], [3., 0., 2.]], [3., 0., 2.], obs)
        self.assertEqual(result['done_reason'], 'collision')

    def test_deadline_clips_path_and_requires_complete_timeout(self):
        self.assertEqual(classify_odom([0., 73.], [[0., 0., 2.], [73., 0., 2.]],
                                       [73., 0., 2.], [])['done_reason'], 'time_limit')
        with self.assertRaises(ValueError):
            classify_odom([0., 2.], [[0., 0., 2.], [0., 0., 2.]], [3., 0., 2.], [])

    def test_collision_tie_and_safe_grazing(self):
        from diffusion_policy_3d.env.collision_geom import ObstacleSpec
        sphere = ObstacleSpec(name='tie', shape='sphere', x=2., y=0., z=2., radius=.4)
        result = classify_odom([0., 1.], [[0., 0., 2.], [3., 0., 2.]], [2., 0., 2.], [sphere])
        self.assertEqual(result['done_reason'], 'collision')
        box = ObstacleSpec(name='graze', shape='box', x=1., y=0., z=2.,
                           size_x=1., size_y=1., size_z=1.)
        result = classify_odom([0., 1.], [[0., .61, 2.], [3., .61, 2.]], [3., .61, 2.], [box])
        self.assertEqual(result['done_reason'], 'reached')

    def test_odometry_sampling_gate(self):
        self.assertGreaterEqual(check_sampling([0., .02, .04])['min_rate_hz'], 30)
        with self.assertRaises(ValueError):
            check_sampling([0., .02, .1])


if __name__ == '__main__':
    unittest.main()
