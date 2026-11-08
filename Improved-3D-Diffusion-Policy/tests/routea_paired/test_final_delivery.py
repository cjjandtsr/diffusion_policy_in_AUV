"""Small CPU contracts for the final paper delivery; no fake training pass."""
import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.run_routea_final_stage2 import calibration_check
from scripts.run_routea_stage1_seed import compare_u1_metrics
from scripts.routea_testext_task_bank import merge_banks
from scripts.export_routea_final_data import segment_clearance
from scripts.prepare_routea_final_evaluation import bind_bank
from scripts.run_routea_f3_resume import canonical


class FinalContracts(unittest.TestCase):
    def test_approved_index_rejects_another_self_consistent_bank(self):
        entry = dict(key='key',task_id='task',repeat=1,initial_hash='obs')
        bank = dict(episodes=[entry])
        bank['content_hash'] = canonical(bank)
        index = dict(conditions=[dict(entry,component='added',source_bank_hash=bank['content_hash'])])
        report = dict(incremental_hash=bank['content_hash'])
        bind_bank('added',bank,index,report)
        wrong = dict(episodes=[dict(entry,initial_hash='other')])
        wrong['content_hash'] = canonical(wrong)
        with self.assertRaises(AssertionError):
            bind_bank('added',wrong,index,report)

    def test_continuous_clearance_checks_segment_interior(self):
        from diffusion_policy_3d.env.collision_geom import ObstacleSpec
        sphere = ObstacleSpec(name='s',shape='sphere',x=0,y=0,z=0,radius=1)
        self.assertAlmostEqual(segment_clearance([-2,2,0],[2,2,0],[sphere]), .9, places=8)
        box = ObstacleSpec(name='b',shape='box',x=0,y=0,z=0,size_x=2,size_y=2,size_z=2)
        self.assertAlmostEqual(segment_clearance([-2,0,0],[2,0,0],[box]), -1.1, places=7)

    def test_calibration_uses_ratio_of_medians(self):
        value = dict(batches=[dict(ppo_norm=a, keep_norm=b) for a,b in [(1,2),(2,8),(3,4),(4,6)]],
                     coefficient=.05)
        self.assertEqual(calibration_check(value)['expected'], .05)
        value['coefficient'] = .1
        with self.assertRaises(AssertionError):
            calibration_check(value)

    def test_equivalence_rejects_tiny_numeric_change(self):
        row = {k: 1 for k in ('transition_id','task_id','world_id','episode_id','decision_index',
                             'task_seed','reset_seed','position_before_xyz','goal_xyz','action_chunk')}
        metric = dict(rollout_audit=dict(transition_records=[row]*5), loss=.1)
        altered = copy.deepcopy(metric)
        altered['loss'] += 1e-12
        self.assertEqual(compare_u1_metrics(metric, altered)['data_difference_count'], 1)
        altered = copy.deepcopy(metric)
        metric['runtime_identity_hash'], altered['runtime_identity_hash'] = 'old','new'
        self.assertTrue(compare_u1_metrics(metric, altered)['numeric_and_structure_equal'])

    def test_union_keeps_original_rows(self):
        old = dict(worlds={}, tasks=[dict(task_id='old')], episodes=[dict(key='old')],
                   candidate_content_hash='a', content_hash='old')
        added = dict(worlds={}, tasks=[dict(task_id='new')], episodes=[dict(key='new')],
                     candidate_content_hash='b', content_hash='new')
        original = copy.deepcopy(old)
        merged = merge_banks(old, added)
        self.assertEqual(merged['episodes'][0], original['episodes'][0])
        self.assertEqual(old, original)
        added['episodes'][0]['key'] = 'old'
        with self.assertRaises(ValueError):
            merge_banks(old, added)


if __name__ == '__main__':
    unittest.main()
