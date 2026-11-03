import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('xk_eval', ROOT/'scripts/run_routea_xk_evaluation.py')
module = importlib.util.module_from_spec(spec)
if spec.loader is not None and spec.origin and Path(spec.origin).exists():
    spec.loader.exec_module(module)


class EvaluationProtocolTests(unittest.TestCase):
    def test_test12_is_four_frozen_policies_and_five_draws(self):
        self.assertTrue(hasattr(module, 'test12_contract'), 'test12 protocol not implemented')
        self.assertEqual(module.test12_contract(), {'worlds':12, 'tasks':48, 'repeats':5,
                         'policies':['bc','u128','C','T'], 'episodes':960})

    def test_window_end_stays_in_selection_denominator(self):
        self.assertTrue(hasattr(module, 'select_c2'), 'C2 selection not implemented')
        rows=[]
        for variant in ('baseline','z03','z06','clip2'):
            for group,n in [('V',200),('P',100)]:
                reached = 20 if variant=='baseline' else 30
                rows.extend(dict(variant=variant,group=group,outcome='reached' if i<reached else 'window_end')
                            for i in range(n))
        result=module.select_c2(rows)
        self.assertEqual(result['counts']['baseline']['P']['n'],100)
        self.assertEqual(result['counts']['z03']['V']['window_end'],170)
        self.assertEqual(result['selected'],'z03')

    def test_collision_safety_gate_is_absolute_percentage_points(self):
        self.assertTrue(hasattr(module, 'select_c2'), 'C2 selection not implemented')
        rows=[]
        for variant in ('baseline','z03','z06','clip2'):
            for group,n in [('V',200),('P',100)]:
                for i in range(n):
                    outcome='reached' if i<(10 if variant=='baseline' else 40) else 'window_end'
                    if group=='P' and variant!='baseline' and i>=97: outcome='collision'
                    rows.append(dict(variant=variant,group=group,outcome=outcome))
        self.assertIsNone(module.select_c2(rows)['selected'])

    def test_vertical_bank_caps_task_reuse_and_keeps_positive_strata(self):
        self.assertTrue(hasattr(module,'choose_c2_anchors'),'bank selection not implemented')
        candidates=[dict(id=str(i),group='V',task_id=str(i//5)) for i in range(100)]
        candidates += [dict(id='p'+str(i),group='P_narrow',task_id='p'+str(i)) for i in range(20)]
        candidates += [dict(id='g'+str(i),group='P_approach',task_id='g'+str(i)) for i in range(20)]
        states=module.choose_c2_anchors(candidates)
        from collections import Counter
        self.assertEqual(Counter(s['group'] for s in states),{'V':40,'P_narrow':10,'P_approach':10})
        self.assertLessEqual(max(Counter(s['task_id'] for s in states if s['group']=='V').values()),3)

    def test_cluster_statistics_use_twenty_thousand_fixed_world_draws(self):
        self.assertTrue(hasattr(module,'paired_cluster'),'E1 cluster analysis not implemented')
        rows=[]
        for world in range(21):
            for repeat in range(1,6):
                for name,success in [('K',False),('X',True)]:
                    rows.append(dict(policy=name,world_id=str(world),task_id=str(world),repeat=repeat,
                                     safe_arrival=success,done_reason='reached' if success else 'time_limit'))
        result=module.paired_cluster(rows,'X','K')
        self.assertEqual(result['bootstrap_draws'],20000)
        self.assertEqual(result['worlds'],21)
        self.assertEqual(result['ci95_delta_pp'],[100.,100.])
        self.assertEqual(result['rescued'],105)

    def test_nearest_point_checks_continuous_executed_segments(self):
        self.assertTrue(hasattr(module,'closest_path_point'),'continuous nearest point not implemented')
        import numpy as np
        np.testing.assert_allclose(module.closest_path_point([[-2,0,0],[2,0,0]],[0,0,.4]),[0,0,0])

    def test_native_snapshot_keeps_behavior_identity_inputs(self):
        import tempfile
        temp=ROOT/'.test_tmp_xk';temp.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(temp)) as directory:
            source=Path(directory)/'original';target=Path(directory)/'snapshot'
            artifacts=source/'data/reward_v17';artifacts.mkdir(parents=True)
            for name in ('fixture_manifest_v2.json','frozen_density_v1.json','calibration_report_v2.json'):
                (artifacts/name).write_bytes(b'{"verified":true}')
            module.snapshot_source(source,target)
            self.assertTrue((target/'data/reward_v17/fixture_manifest_v2.json').exists(),
                            'native closure dropped data/reward_v17 identity input')
            self.assertEqual((target/'data/reward_v17/fixture_manifest_v2.json').read_bytes(),b'{"verified":true}')

    def test_unequal_strata_world_sizes_bootstrap_task_weighted_endpoint(self):
        rows=[]
        for world in range(21):
            for repeat in range(1,2 if world==0 else 10):
                for name in ('K','X'):
                    rows.append(dict(policy=name,world_id=str(world),task_id=str(world),repeat=repeat,
                        safe_arrival=name=='X' and world==0,done_reason='time_limit'))
        result=module.paired_cluster(rows)
        self.assertAlmostEqual(result['delta_pp'],100/181)
        self.assertLess(result['ci95_delta_pp'][1],5.)


if __name__=='__main__': unittest.main()
