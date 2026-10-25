"""Descriptive analysis serialization and return-pack preservation."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


class Analysis(unittest.TestCase):
    def test_comparison_denominator_contains_new_baseline_and_requested_models(self):
        from scripts.routea_paired_support.core import write_json
        from scripts.routea_paired_support.workflow import finish
        bank = {'content_hash': 'b', 'tasks': [{'task_id': 't', 'role': 'dense_recovery'}],
                'episodes': [{'key': 'e%d' % i, 'task_id': 't', 'repeat': i,
                              'initial_hash': 'h%d' % i} for i in (1, 2)]}
        base = {'use_base_policy': True}
        model = {'use_base_policy': False, 'update': 32, 'state_sha256': 'a' * 64}
        cat = {'policies': {'bc': base, 'new32': model},
               'common': {'bc_file_sha256': 'c' * 64}}
        with tempfile.TemporaryDirectory() as d:
            reference, out = Path(d) / 'reference', Path(d) / 'out'
            for name, root in [('bc', reference), ('old256', reference), ('bc', out), ('new32', out)]:
                for e in bank['episodes']:
                    summary = {'policy': name, 'task_id': 't', 'world_id': 'w',
                               'role': 'dense_recovery', 'repeat': e['repeat'], 'status': 'valid',
                               'safe_arrival': name == 'new32',
                               'done_reason': 'reached' if name == 'new32' else 'collision',
                               'decisions': 1, 'physical_time_s': .3,
                               'trajectory_metrics': {'path_length_m': 1.},
                               'bank_hash': 'b', 'initial_hash': e['initial_hash']}
                    write_json(root / 'full' / name / 'commits' / (e['key'] + '.json'),
                               {'sha256': 'f' * 64, 'summary': summary})
            completion = finish(ROOT, cat, bank, out, reference)
            self.assertEqual(completion['full'], 4)
            self.assertEqual(completion['reference_full'], 0)
            self.assertEqual(completion['comparison_full'], 4)
            self.assertEqual(completion['anchors'], 0)

    def test_seeds_and_roles_remain_separate_and_json_serializable(self):
        import json
        from scripts.routea_paired_support.core import ROLES
        from scripts.routea_paired_support.analysis import tables
        policies=["bc"]+[f+"_seed"+str(s) for f in ("a0","r1")
                         for s in (202609201,202609202,202609203)]
        records=[]
        for role in ROLES:
            for repeat in (1,2):
                for policy in policies:
                    success=policy.startswith("r1")
                    records.append({"policy":policy,"task_id":role,"world_id":role,
                        "role":role,"repeat":repeat,"status":"valid","safe_arrival":success,
                        "done_reason":"reached" if success else "collision","decisions":1,
                        "physical_time_s":.3,"initial_hash":role+str(repeat),
                        "trajectory_metrics":{"path_length_m":1.},"ambiguous_first_event":False})
        result=tables(records)
        self.assertEqual(len(result["comparisons"]),21)
        self.assertEqual(len(result["role_policy"]),28)
        self.assertEqual(len(result["episode_metrics"]),56)
        self.assertEqual(result["comparisons"][1]["overall_descriptive"]["tasks"],4)
        self.assertFalse(result["paper_effect_threshold_frozen"])
        json.dumps(result,allow_nan=False)

    def test_return_inventory_preserves_attempts_and_excludes_weights(self):
        from scripts.routea_paired_support.core import file_hash
        from scripts.routea_paired_support.workflow import package_result
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            (out / "raw.jsonl").write_text("original raw\\n", encoding="utf-8")
            for number in (1, 2):
                (out / ("attempt%d.log" % number)).write_text("closed log\\n", encoding="utf-8")
                package_result(out, "%03d" % number, {"status": "complete", "full": 420})
            lines = (out / "pack_sha256_002.txt").read_text(encoding="utf-8").splitlines()
            for line in lines:
                expected, name = line.split("  ", 1)
                self.assertEqual(file_hash(out / name), expected)
            self.assertTrue((out / "pack_sha256_001.txt").exists())
            self.assertEqual(len(list(out.glob("run_complete_*.json"))), 2)
            (out / "state.pt").write_bytes(b"weight")
            with self.assertRaises(ValueError):
                package_result(out, "003", {"status": "complete"})
            self.assertFalse((out / "run_complete_003.json").exists())


if __name__=="__main__":
    unittest.main()
