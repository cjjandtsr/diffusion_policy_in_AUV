"""Approved paper matrix: no old-bank relabeling or extra trajectories."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.routea_paired_support import recovery, bank


class PaperProtocolTests(unittest.TestCase):
    def test_full_remaining_budget_does_not_reset_clock(self):
        adapter = SimpleNamespace(environment=SimpleNamespace(max_episode_steps=60, episode_step=23))
        self.assertEqual(recovery.continuation_limit(adapter, None), 37)
        self.assertEqual(recovery.continuation_limit(adapter), 12)

    def test_contract_counts_explicitly_distinguish_new_bank(self):
        self.assertEqual(bank.bank_contract('routea_paper_test_v1'), (12, 24, 120, 2))
        self.assertEqual(bank.bank_contract('legacy'), (21, 84, 420, 4))
        with self.assertRaises(ValueError):
            bank.bank_contract('anything')

    def test_paper_recovery_counts_use_each_original_deadline(self):
        counts = recovery.recovery_counts(2, 2, remaining=[3, 37])
        self.assertEqual(counts['max_continuation_decisions'], 400)
        self.assertEqual(counts['continuations'], 20)

    def test_paper_requires_exact_models(self):
        from scripts.routea_paired_support.paper import validate_models
        cat = {'policies': {'bc': {}, 'rl_2f3f9914db78_u32': {'update': 32}}}
        validate_models(cat)
        with self.assertRaises(ValueError):
            validate_models({'policies': {'bc': {}, 'rl_2f3f9914db78_u384': {'update': 384}}})

    def test_shipped_geometries_and_selection_are_finite_and_reproducible(self):
        import json
        import tempfile
        from scripts.routea_paired_support.paper import read_spec,select_tasks
        from scripts.evaluate_dppo_uav_formal_endpoint import canonical_episode_config
        from diffusion_policy_3d.env.kinematic_uav_env import KinematicUAVEnv
        root=Path(__file__).resolve().parents[2]
        spec=read_spec(root/'scripts/routea_paired_support/paper_bank_spec_v1.json')
        selected=[]
        with tempfile.TemporaryDirectory() as d:
            for world in spec['worlds']:
                p=Path(d)/(world['world_id']+'.json')
                p.write_text(json.dumps(world['document']),encoding='utf-8')
                env=KinematicUAVEnv(world_pool=[(world['world_id'],str(p))],
                    synth_pc_factory=lambda _:None,episode_role=world['role'],
                    mid_state_init_ratio=0,disturb_enable=False)
                env._set_world(world['world_id'],str(p))
                a=select_tasks(env,world['world_id'],world['role'])
                b=select_tasks(env,world['world_id'],world['role'])
                self.assertEqual(a,b)
                self.assertTrue(a[0]['blocked'])
                self.assertNotEqual(a[0]['candidate_index'],a[1]['candidate_index'])
                for task in a:
                    canonical_episode_config(task['payload'])
                selected.extend(a)
        self.assertEqual(len(selected),24)
        self.assertEqual(sum(x['blocked'] for x in selected),13)

    def test_paper_statistics_excludes_seen_geometry_from_primary_stratum(self):
        from scripts.routea_paired_support.paper import paper_statistics
        records=[]
        for task,world in [('t1','new'),('t2','seen')]:
            for policy in ['bc','rl_2f3f9914db78_u32']:
                records.append(dict(task_id=task,world_id=world,policy=policy,repeat=1,
                    role='dense_recovery',initial_hash=task,status='valid',
                    safe_arrival=(policy!='bc' if world=='new' else policy=='bc')))
        b=dict(tasks=[dict(task_id=t,blocked=True) for t in ['t1','t2']],
               worlds={'new':{'exact_train_matches':[]},'seen':{'exact_train_matches':['train']}})
        result=paper_statistics(records,b,{'comparisons':[{'inference_scope':'old'}]})
        self.assertEqual(result['geometry_strata']['different_geometry_tasks']['net'],1)
        self.assertEqual(result['geometry_strata']['seen_geometry_control']['net'],-1)
        self.assertNotEqual(result['comparisons'][0]['inference_scope'],'old')


if __name__ == '__main__':
    unittest.main()
