"""The validation diagnostic must not quietly consume the held-out test split."""
import ast
from dataclasses import asdict
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts import evaluate_dppo_uav_formal_endpoint as endpoint
from scripts import run_routea_val20_endpoint as val20
from scripts.formal_endpoint_eval_core import PairedEpisodeRecord


class ValOnlyProtocolTests(TestCase):
    def _record(self, distance, *, done='reached', raw='reached', flags=()):
        return dict(
            task_id='edge', config_id='edge', config_hash='a'*64,
            obstacle_hash='b'*64, source_manifest_hash='c'*64,
            policy_identity_hash='d'*64, world_id='w0', split='rl_val',
            role='anticipatory', seed=1, policy='rl', done_reason=done,
            raw_done_reason=raw, episode_return=0., onset_triggered=False,
            onset_first_step=None, future_clearance_reward_sum=0.,
            d_soft_h_min=1., d_soft_h_steps=1, support_cells=0,
            terminal_xyz=(distance, 0., 0.), min_xyz=(distance, 0., 0.),
            max_xyz=(distance, 0., 0.), action_dz_mean=0., action_dz_min=0.,
            action_dz_max=0., trap_recovery_count=0, loop_count=0,
            side_flip_count_max=0, steps=1, goal_xyz=(0., 0., 0.),
            final_distance_3d=distance, final_error_xy=distance,
            final_error_z=0., terminal_safety_flags=flags)

    def test_boundary_arrival_keeps_authoritative_reached_and_records_delta(self):
        fields = self._record(.5000000757)
        with self.assertRaisesRegex(ValueError, 'reached must mean'):
            PairedEpisodeRecord(**fields)
        val20.Val20EpisodeRecord.boundary_events.clear()
        record = val20.Val20EpisodeRecord(**fields)
        self.assertTrue(record.success)
        self.assertEqual(asdict(record)['done_reason'], 'reached')
        self.assertEqual(len(val20.Val20EpisodeRecord.boundary_events), 1)

    def test_replayed_wsl_boundary_coordinates_are_within_rounding_allowance(self):
        terminal = (3.3557403087615967, -8.706485748291016, 3.920018196105957)
        goal = (3.4705851078033447, -8.3926420211792, 4.291922569274902)
        delta = [a - b for a, b in zip(terminal, goal)]
        distance = sum(value * value for value in delta) ** .5
        fields = self._record(distance)
        fields.update(terminal_xyz=terminal, goal_xyz=goal,
                      min_xyz=terminal, max_xyz=terminal,
                      final_error_xy=(delta[0]**2 + delta[1]**2)**.5,
                      final_error_z=abs(delta[2]))
        val20.Val20EpisodeRecord.boundary_events.clear()
        self.assertTrue(val20.Val20EpisodeRecord(**fields).success)
        event = val20.Val20EpisodeRecord.boundary_events[0]
        self.assertLess(event['excess_m'], event['rounding_allowance_m'])

    def test_boundary_exception_is_narrow(self):
        val20.Val20EpisodeRecord.boundary_events.clear()
        self.assertTrue(val20.Val20EpisodeRecord(**self._record(.5)).success)
        self.assertEqual(val20.Val20EpisodeRecord.boundary_events, [])
        for fields in (
            self._record(.500002),
            self._record(.5000000757, raw='collision'),
            self._record(.5000000757, flags=('collision',)),
            self._record(.499999, done='time_limit', raw='time_limit'),
        ):
            with self.subTest(fields=fields):
                with self.assertRaises(ValueError):
                    val20.Val20EpisodeRecord(**fields)
        self.assertEqual(val20.Val20EpisodeRecord.boundary_events, [])

    def test_fixed_val_bank_has_only_20_worlds_and_80_episodes(self):
        root = Path(__file__).resolve().parents[2]
        val20.install_val_split()
        bank = endpoint._load_fixed_bank(root / 'data/eval_configs', 'routea-val')
        self.assertEqual(len(bank), 80)
        self.assertEqual(len({world for _, world, _ in bank}), 20)
        self.assertEqual({split for split, _, _ in bank}, {'rl_val'})

    def test_val_command_needs_no_test_world_path(self):
        root = Path(__file__).resolve().parents[2]
        with TemporaryDirectory(dir=str(root)) as temporary:
            output = Path(temporary) / 'val'
            command = ['--config', 'config.yaml', '--rl-checkpoint', 'weights',
                       '--eval-config-dir', 'data/eval_configs', '--benchmark', 'routea-val',
                       '--worlds-val', 'worlds/val', '--output-dir', str(output),
                       '--expected-update', '128']
            def fake_run(args):
                self.assertIs(endpoint.PairedEpisodeRecord, val20.Val20EpisodeRecord)
                output.mkdir()
                (output / 'evaluation_provenance.json').write_text('{}', encoding='utf-8')
                return 0
            with patch.object(endpoint, 'run', side_effect=fake_run) as run:
                self.assertEqual(val20.main(command), 0)
            self.assertIsNone(run.call_args.args[0].worlds_test)
            audit = json.loads((output / 'arrival_boundary_audit.json').read_text(encoding='utf-8'))
            self.assertEqual(audit['accepted_boundary_count'], 0)
            self.assertEqual(audit['minimum_rounding_tolerance_m'], 1e-6)

    def test_val_runner_is_outside_checkpoint_source_closure(self):
        root = Path(__file__).resolve().parents[2]
        source = (root / 'diffusion_policy_3d/rl/dppo_uav_rebuild/entrypoints/identity.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        paths = next(ast.literal_eval(node.value) for node in tree.body
                     if isinstance(node, ast.Assign) and any(
                         isinstance(target, ast.Name) and target.id == 'CLI_SOURCE_PATHS'
                         for target in node.targets))
        self.assertIn('scripts/evaluate_dppo_uav_formal_endpoint.py', paths)
        self.assertNotIn('scripts/run_routea_val20_endpoint.py', paths)

