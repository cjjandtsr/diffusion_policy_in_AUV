"""Read-only evaluation contracts; runnable with Python unittest."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class Contracts(unittest.TestCase):
    def test_model_count_and_modes_are_derived_from_inputs(self):
        from scripts.routea_paired_support.workflow import run_protocol, selected_entries
        from unittest.mock import patch
        bank = {'content_hash': 'b', 'tasks': [{'task_id': 'a', 'role': 'dense'},
                {'task_id': 'b', 'role': 'support'}], 'episodes': [
                    {'task_id': task, 'repeat': repeat} for task in ('a', 'b') for repeat in range(1, 6)]}
        cat = {'content_hash': 'c', 'policies': {'bc': {}, 'x': {}, 'y': {}, 'z': {}}}
        with patch('scripts.routea_paired_support.workflow.evaluation_sources', return_value={}):
            self.assertEqual(run_protocol(ROOT, cat, bank, 'full', ROOT)['expected_episodes'], 40)
            self.assertEqual(run_protocol(ROOT, cat, bank, 'smoke', ROOT)['expected_episodes'], 6)
            self.assertEqual(run_protocol(ROOT, cat, bank, 'anchors', ROOT)['expected_episodes'], 2)

    def test_comparison_uses_new_bc_and_rl_traces_only(self):
        from scripts.routea_paired_support.workflow import comparison_records
        from scripts.routea_paired_support.core import write_json
        bank = {'content_hash': 'bank', 'episodes': [
            {'key': 'one', 'task_id': 'task', 'repeat': 1, 'initial_hash': 'initial'}]}
        cat = {'common': {'bc_file_sha256': 'base'},
               'policies': {'bc': {'use_base_policy': True},
                            'rl': {'update': 256, 'state_sha256': 'actor'}}}
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'new'
            for name in cat['policies']:
                p = out / 'full' / name / 'commits' / 'one.json'
                write_json(p, {'sha256': name, 'summary': {
                    'bank_hash': 'bank', 'initial_hash': 'initial', 'policy': name}})
            records, index, jobs, models = comparison_records(
                ROOT, cat, bank, out, Path(directory) / 'old-unused')
            self.assertEqual(set(models), {'bc', 'rl'})
            self.assertEqual(len(records), 2)
            self.assertEqual(set(index[0]['models']), {'bc', 'rl'})
            self.assertEqual(jobs['bc'], out / 'full/bc')

    def test_shared_reader_checks_full_planned_tail_and_observation_identity(self):
        import copy
        from scripts.routea_paired_support.core import read_json
        from scripts.routea_paired_support.episodes import load_commit
        from scripts.routea_paired_support.reference import check_episode_identity
        from scripts.routea_paired_support.runtime import catalogue
        ref = ROOT.parents[1] / 'wsl_snapshot/diagnostics/routea_bc_a0_r1_paired_dev_20261010_py38fix_run02'
        if not ref.exists():
            self.skipTest('Windows historical trace fixture; WSL verifies its real bank during execution')
        bank = read_json(ref / 'bank/bank.json')
        e = bank['episodes'][0]
        t = next(t for t in bank['tasks'] if t['task_id'] == e['task_id'])
        original = load_commit(ref / 'full/bc', ref / 'full/bc/commits' / (e['key'] + '.json'))
        cat = catalogue(ROOT)
        check_episode_identity(original, e, t, bank['worlds'][t['world_id']], bank, cat, 'bc', original)
        for field in ('tail', 'observation'):
            changed = copy.deepcopy(original)
            if field == 'tail':
                changed[1][0]['planned_policy_path'][-1][0] += .01
            else:
                changed[1][0]['observation_provenance']['policy_input_hash'] = '0' * 64
            with self.assertRaises(ValueError):
                check_episode_identity(changed, e, t, bank['worlds'][t['world_id']], bank, cat, 'bc', original)

    def test_weight_address_selects_update_without_stage_specific_code(self):
        from scripts.routea_paired_support.runtime import resolve_catalogue
        from scripts.routea_paired_support.core import read_json, write_json, canonical_hash, file_hash
        from scripts.routea_paired_support.runtime import catalogue
        base = catalogue(ROOT)
        original = base['policies']['r1_seed202609201']
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for update in (32, 256):
                p = Path(d) / ('checkpoint_update_%06d' % update)
                p.mkdir()
                header = {'schema_version': 2, 'update_index': update,
                          'runtime_identity': original['runtime_identity'],
                          'runtime_identity_hash': original['runtime_identity_hash'],
                          'lora_identity_hash': base['common']['lora_identity_hash']}
                write_json(p / 'header.json', header)
                (p / 'state.pt').write_bytes(('actor at %d' % update).encode())
                sums = {n: file_hash(p / n) for n in ('header.json', 'state.pt')}
                write_json(p / 'SHA256.json', sums)
                write_json(p / 'COMMITTED.json', {'schema_version': 2, 'status': 'committed',
                    'files': sums, 'header_schema_hash': sums['header.json']})
                paths.append(p)
            cat = resolve_catalogue(ROOT, paths)
            self.assertEqual(set(cat['policies']), {'bc', 'r1_seed202609201_u32', 'r1_seed202609201_u256'})
            self.assertEqual([cat['policies'][n]['update'] for n in sorted(cat['policies']) if n != 'bc'], [256, 32])
            self.assertNotEqual(cat['policies']['r1_seed202609201_u32']['state_sha256'],
                                cat['policies']['r1_seed202609201_u256']['state_sha256'])

    def core(self):
        try:
            from scripts.routea_paired_support import core
        except ImportError as exc:
            self.fail("paired evaluation core is not implemented: " + str(exc))
        return core

    def fixture(self, directory):
        c = self.core()
        root = Path(directory)
        identity = {"bc_identity_hash": "a" * 64}
        h = {"update_index": 256, "runtime_identity": identity,
             "runtime_identity_hash": c.canonical_hash(identity),
             "lora_identity_hash": "b" * 64, "schema_version": 2}
        (root / "header.json").write_text(json.dumps(h), encoding="utf-8")
        (root / "state.pt").write_bytes(b"serialized actor fixture")
        sums = {k: c.file_hash(root / k) for k in ("header.json", "state.pt")}
        (root / "SHA256.json").write_text(json.dumps(sums), encoding="utf-8")
        (root / "COMMITTED.json").write_text(json.dumps({
            "status": "committed", "schema_version": 2,
            "files": sums, "header_schema_hash": sums["header.json"]}), encoding="utf-8")
        return {"header_sha256": sums["header.json"], "state_sha256": sums["state.pt"],
                "runtime_identity_hash": h["runtime_identity_hash"], "update": 256}

    def test_changed_state_is_rejected_before_loading(self):
        c = self.core()
        with tempfile.TemporaryDirectory() as d:
            e = self.fixture(d)
            self.assertEqual(c.verify_checkpoint(Path(d), e)["update_index"], 256)
            (Path(d) / "state.pt").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                c.verify_checkpoint(Path(d), e)

    def test_wrong_training_identity_is_rejected(self):
        c = self.core()
        with tempfile.TemporaryDirectory() as d:
            e = self.fixture(d)
            e["runtime_identity_hash"] = "0" * 64
            with self.assertRaises(ValueError):
                c.verify_checkpoint(Path(d), e)

    def test_unknown_manifest_target_is_rejected(self):
        c = self.core()
        with tempfile.TemporaryDirectory() as d:
            e = self.fixture(d)
            p = Path(d) / "SHA256.json"
            sums = json.loads(p.read_text())
            sums["../outside"] = "0" * 64
            p.write_text(json.dumps(sums))
            with self.assertRaises(ValueError):
                c.verify_checkpoint(Path(d), e)

    def test_duplicate_json_and_nonfinite_are_rejected(self):
        c = self.core()
        with self.assertRaises(ValueError):
            c.parse_json('{"x":1,"x":2}')
        with self.assertRaises(ValueError):
            c.parse_json('{"x":NaN}')

    def test_interrupted_publish_never_creates_a_completed_filename(self):
        c=self.core()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"commit.json"
            def interrupted(stream):
                stream.write(b'{"partial":')
                raise RuntimeError("interrupted")
            with self.assertRaises(RuntimeError):
                c._publish_generated(p,interrupted)
            self.assertFalse(p.exists())
            self.assertEqual(len(list(Path(d).glob(".*.partial_*"))),1)
            c.write_json(p,{"complete":True})
            with self.assertRaises(FileExistsError):
                c.write_json(p,{"replacement":True})
            self.assertEqual(c.read_json(p),{"complete":True})

    def test_changed_physical_event_source_invalidates_preflight(self):
        import shutil
        from scripts.routea_paired_support.runtime import evaluation_sources
        from scripts.routea_paired_support.workflow import check_preflight
        c=self.core()
        sources=evaluation_sources(ROOT)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in sources:
                target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(ROOT/name,target)
            report=root/"preflight.json"
            c.write_json(report,{"status":"pass","catalogue_hash":"fixture",
                                 "evaluation_sources":evaluation_sources(root)})
            check_preflight(report,root,{"content_hash":"fixture"})
            changed=root/"diffusion_policy_3d/env/reward_v2_support/first_event.py"
            with changed.open("a",encoding="utf-8") as stream:
                stream.write("\n# changed fixture physics\n")
            with self.assertRaises(ValueError):
                check_preflight(report,root,{"content_hash":"fixture"})

    def test_trace_uses_continuous_segments(self):
        c = self.core()
        metrics = c.trace_metrics([[0, 0, 1], [2, 0, 1]], [1, 0, 1])
        self.assertAlmostEqual(metrics["min_goal_distance_m"], 0.)
        self.assertAlmostEqual(metrics["path_length_m"], 2.)

    def rows(self):
        return [{"task_id": "t", "repeat": r, "world_id": "w", "role": "dense_recovery",
                 "policy": policy, "safe_arrival": policy == "r1", "initial_hash": "same",
                 "status": "valid"} for r in (1, 2, 3) for policy in ("bc", "r1")]

    def test_repeats_do_not_become_independent_tasks(self):
        s = self.core().paired_summary(self.rows(), "r1", "bc")
        self.assertEqual((s["pairs"], s["tasks"], s["worlds"]), (3, 1, 1))
        self.assertEqual((s["rescue"], s["break"], s["net"]), (3, 0, 3))

    def test_summary_numbers_can_be_saved_as_plain_json(self):
        summary = self.core().paired_summary(self.rows(), "r1", "bc")
        self.assertIsInstance(summary["improved_tasks"], int)
        json.dumps(summary, allow_nan=False)

    def test_missing_pair_and_different_initial_observation_are_rejected(self):
        c = self.core()
        with self.assertRaises(ValueError):
            c.paired_summary(self.rows()[:-1], "r1", "bc")
        rows = self.rows()
        rows[0]["initial_hash"] = "different"
        with self.assertRaises(ValueError):
            c.paired_summary(rows, "r1", "bc")

    def test_invalid_record_cannot_be_silently_dropped(self):
        rows = self.rows()
        rows[0]["status"] = "invalid"
        with self.assertRaises(ValueError):
            self.core().paired_summary(rows, "r1", "bc")

    def test_action_tail_is_not_the_executed_prefix(self):
        import numpy as np
        c = self.core()
        full = np.zeros((16, 11), np.float32)
        full[6:10, 0] = 100.
        _, policy, executed = c.action_views(full, full[2:10], full[2:6])
        self.assertEqual(float(executed[:, 0].sum()), 0.)
        self.assertEqual(float(policy[:, 0].sum()), 400.)
        with self.assertRaises(ValueError):
            c.action_views(full, full[2:10], full[3:7])

    def test_first_physical_event_uses_h4_and_failure_wins_tie(self):
        import numpy as np
        from diffusion_policy_3d.env.reward_v2_support.first_event import locate_first_event
        from diffusion_policy_3d.env.collision_geom import ObstacleSpec
        goal = np.array([2., 0., 1.])
        self.assertIsNone(locate_first_event([[0, 0, 1], [.2, 0, 1]], goal, [], .1, 5.))
        sphere = ObstacleSpec("s", "sphere", 2., 0., 1., radius=.4)
        event = locate_first_event([[0, 0, 1], [3, 0, 1]], goal, [sphere], .1, 5.)
        self.assertEqual(event.kind, "collision")
        event = locate_first_event([[0, 0, 1], [3, 0, 1]], goal, [], .1, 5.)
        self.assertEqual(event.kind, "reached")
        self.assertAlmostEqual(event.fraction, .5)

    def test_actual_lora_load_does_not_change_disabled_bc_branch(self):
        import torch
        from torch import nn
        from diffusion_policy_3d.rl.dppo_uav_rebuild.actor_ddpm.lora import install_lora
        from scripts.routea_paired_support.runtime import validate_actor
        model = nn.Sequential(nn.Linear(3, 2)).float()
        bundle = install_lora(model, rank=2, alpha=2., expected_linear=1, expected_conv1d=0)
        x = torch.tensor([[1., 2., 3.]])
        before = model(x).detach().clone()
        tensors = {k: torch.ones_like(v) for k, v in bundle.adapter_state_dict().items()}
        validate_actor(bundle, tensors)
        bundle.load_adapter_state_dict(tensors)
        self.assertTrue(torch.equal(before, model(x)))
        with bundle.mode(True):
            self.assertFalse(torch.equal(before, model(x)))
        tensors[next(iter(tensors))] = torch.tensor(float("nan"))
        with self.assertRaises(ValueError):
            validate_actor(bundle, tensors)

    def test_partial_trace_cannot_be_counted_as_complete_episode(self):
        try:
            from scripts.routea_paired_support.episodes import load_commit
        except ImportError as exc:
            self.fail("trace commit validation is not implemented: " + str(exc))
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "trace.jsonl").write_text('{"type":"episode_start"}\n', encoding="utf-8")
            c = self.core()
            c.write_json(root / "commit.json", {"trace": "trace.jsonl", "sha256": c.file_hash(root / "trace.jsonl"), "summary": {}})
            with self.assertRaises(ValueError):
                load_commit(root, root / "commit.json")

    def test_valid_collision_trace_keeps_environment_reason(self):
        import numpy as np
        from dataclasses import asdict
        from diffusion_policy_3d.env.collision_geom import ObstacleSpec
        from diffusion_policy_3d.env.reward_v2_support.first_event import locate_first_event
        from scripts.routea_paired_support.episodes import validate_rows
        obstacle = ObstacleSpec("s", "sphere", 2.4, 0., 1., radius=.2)
        candidate = np.array([[0,0,1],[1,0,1],[2,0,1],[3,0,1],[4,0,1]], np.float32)
        goal = [8.,0.,1.]
        event = locate_first_event(candidate, goal, [obstacle], .1, 5.)
        actual = np.vstack((candidate[:event.segment_index+1], event.position.astype(np.float32)))
        full = np.zeros((16,11), np.float32)
        full[2:10,0] = 1.
        row = {"decision":0,"full_action_16x11":full.tolist(),"policy_action_8x11":full[2:10].tolist(),
               "executed_action_4x11":full[2:6].tolist(),"pre_state":{"position":[0,0,1],"velocity":[0,0,0],"yaw":0.},
               "candidate_h4_path":candidate.tolist(),"done_reason":"collision","executed_path":actual.tolist(),
               "event_segment":event.segment_index,"event_fraction":event.fraction,
               "executed_time_s":(event.segment_index+event.fraction)*.3,
               "post_state":{"position":actual[-1].tolist()}}
        header = {"geometry":[asdict(obstacle)],"task":{"start_pos":[0,0,1],"goal_pos":goal,"z_range":[1,5]}}
        summary = {"status":"valid","done_reason":"collision","safe_arrival":False,"decisions":1,
                   "physical_time_s":row["executed_time_s"],"trajectory_metrics":self.core().trace_metrics(actual,goal)}
        try:
            validate_rows(header,[row],summary)
        except ValueError as exc:
            self.fail("valid physical collision must be retained: " + str(exc))


if __name__ == "__main__":
    unittest.main()
