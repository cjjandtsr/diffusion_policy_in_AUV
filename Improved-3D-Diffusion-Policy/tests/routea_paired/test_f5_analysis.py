import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('f5_analysis', ROOT / 'scripts/analyze_routea_f5.py')


class F5AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(Path(SPEC.origin).is_file(), 'F5 analysis is not implemented')
        self.module = importlib.util.module_from_spec(SPEC)
        SPEC.loader.exec_module(self.module)

    def test_collision_increase_never_automatically_passes(self):
        values = dict(high_arrival_delta=8, dev_arrival_delta=-10,
                      zero_band_loss_delta=8, val_arrival_delta=-4,
                      dev_collision_delta=1, high_collision_delta=0)
        self.assertEqual(self.module.pair_verdict(values)['status'], 'manual_review')
        values['dev_collision_delta'] = 0
        self.assertEqual(self.module.pair_verdict(values)['status'], 'meets_adoption_criteria')
        values['high_arrival_delta'] = 7
        self.assertEqual(self.module.pair_verdict(values)['status'], 'criteria_not_met')

    def test_two_seed_rule_keeps_engineering_and_uncertainty_distinct(self):
        self.assertEqual(self.module.combined_verdict(['meets_adoption_criteria'] * 2),
                         'engineering_criteria_met_both_seeds')
        self.assertEqual(self.module.combined_verdict(['criteria_not_met'] * 2),
                         'adoption_criteria_not_met_both_seeds')
        self.assertEqual(self.module.combined_verdict(['meets_adoption_criteria', 'criteria_not_met']),
                         'inconclusive_between_seeds')

    def test_fewer_than_five_worlds_has_counts_without_confidence_interval(self):
        rows = []
        for world in range(4):
            for name, success in [('P9', False), ('P8', True)]:
                rows.append(dict(policy=name, task_id=str(world), world_id=str(world), repeat=1,
                                 safe_arrival=success, done_reason='reached' if success else 'time_limit'))
        result = self.module.compare(rows, 'P8', 'P9')
        self.assertEqual(result['rescued'], 4)
        self.assertIsNone(result['ci95_delta_pp'])
        with self.assertRaises(ValueError):
            self.module.compare(rows + [rows[0]], 'P8', 'P9')

    def test_monitor_keeps_unfinished_boundary_in_exposure_denominator(self):
        rows = []
        for eid, reason, boundary in [('a', 'reached', False), ('b', 'none', True)]:
            rows.append(dict(episode_id=eid, decision_index=0, world_id='w',
                             terminated=reason == 'reached', truncated=False,
                             collection_boundary=boundary,
                             terminal_facts=dict(done_reason=reason),
                             task_bank=dict(source='L0', tier='L0', kind='random', bank_task_id=eid)))
        result = self.module.summarize_training_records(rows)
        counts = result['by_source']['L0']
        self.assertEqual(counts['started'], 2)
        self.assertEqual(counts['completed'], 1)
        self.assertEqual(counts['collection_cut'], 1)
        self.assertEqual(counts['transitions'], 2)
        self.assertEqual(counts['success_rate_completed'], 1.)
        self.assertEqual(counts['success_rate_started'], .5)


if __name__ == '__main__':
    unittest.main()
