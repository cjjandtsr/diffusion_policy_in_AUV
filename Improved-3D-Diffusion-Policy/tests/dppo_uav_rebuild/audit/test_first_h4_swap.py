import json
import unittest
from types import SimpleNamespace

import numpy as np

from scripts.routea_causal_diagnostics.first_h4_swap import (
    ARM_SPECS,
    classify_swap,
    first_actions,
    select_matched_tasks,
)
from scripts.routea_causal_diagnostics.contracts import DiagnosticFailure
from scripts.run_routea_first_h4_swap import (
    _action_bundle, _executed_numpy, _verify_baseline_path, execute_four_arms,
)
import scripts.run_routea_first_h4_swap as swap_script


def _entry(task, role, outcome, distance, repeat=1):
    return {
        "task_id": task,
        "repeat": repeat,
        "role": role,
        "outcome_class": outcome,
        "k_div": 0,
        "episode_config": {
            "start_pos": [0.0, 0.0, 0.0],
            "goal_pos": [distance, 0.0, 0.0],
        },
    }


def _transition(task, repeat, lane, value):
    action = np.zeros((8, 11), dtype=np.float32)
    action[:, 0] = value
    return {
        "task_id": task,
        "repeat": repeat,
        "lane": lane,
        "decision_index": 0,
        "policy_action_8x11": action.tolist(),
        "executed_action_4x11": action[:4].tolist(),
        "executed_h4_path_5x3": [[0.0, 0.0, 0.0]] * 5,
        "action_seed": 7,
        "policy_input_hash": "hash",
    }


class FirstH4SwapTests(unittest.TestCase):
    def test_first_input_failure_identifies_earliest_available_hash_layer(self):
        self.assertTrue(hasattr(swap_script, "_assert_first_input"))
        expected = {
            "raw_frame_hash": "a" * 64,
            "normalized_history_hash": "b" * 64,
            "policy_input_hash": "c" * 64,
            "episode_identity_hash": "d" * 64,
            "action_seed": 17,
        }
        for actual_raw, actual_normalized, layer in (
            ("e" * 64, "f" * 64, "RAW_FRAME"),
            ("a" * 64, "f" * 64, "NORMALIZED_HISTORY_OR_UPSTREAM"),
            ("a" * 64, "b" * 64, "POLICY_INPUT"),
        ):
            provenance = SimpleNamespace(
                raw_frame_hash=actual_raw,
                raw_history_hash="1" * 64,
                normalizer_identity_hash="2" * 64,
                normalized_history_hash=actual_normalized,
                policy_input_hash="f" * 64,
                episode_identity_hash="d" * 64,
            )
            with self.assertRaises(DiagnosticFailure) as caught:
                swap_script._assert_first_input(
                    "env_012_obs22__cfg00", "bc_bc", expected, provenance,
                )
            self.assertEqual(caught.exception.status, "SNAPSHOT_REPLAY_MISMATCH")
            self.assertEqual(caught.exception.exit_code, 10)
            detail = json.loads(caught.exception.detail)
            self.assertEqual(detail["first_mismatch_layer"], layer)
            self.assertEqual(detail["expected"]["policy_input_hash"], "c" * 64)
            self.assertEqual(detail["actual"]["policy_input_hash"], "f" * 64)
            self.assertEqual(detail["actual"]["raw_history_hash"], "1" * 64)
            self.assertEqual(detail["action_seed"], 17)
        provenance.policy_input_hash = "c" * 64
        swap_script._assert_first_input("env_012_obs22__cfg00", "bc_bc", expected, provenance)

    def test_select_matches_each_rescue_to_distinct_nearest_break_in_same_role(self):
        targets = [
            _entry("rescue-a", "anticipatory", "rescue", 10, 2),
            _entry("rescue-a", "anticipatory", "rescue", 10, 1),
            _entry("rescue-b", "dense_recovery", "rescue", 20),
            _entry("break-far", "anticipatory", "break", 18),
            _entry("break-near", "anticipatory", "break", 11),
            _entry("break-dense", "dense_recovery", "break", 19),
        ]
        pairs = select_matched_tasks(targets)
        self.assertEqual(
            [(p["rescue"]["task_id"], p["break"]["task_id"]) for p in pairs],
            [("rescue-a", "break-near"), ("rescue-b", "break-dense")],
        )
        self.assertEqual(pairs[0]["rescue"]["repeat"], 1)

    def test_first_actions_rejects_bad_h4_contract(self):
        rows = [_transition("task", 1, "u0", 0.1), _transition("task", 1, "seed3-u32", 0.2)]
        rows[1]["executed_action_4x11"][0][0] = 9.0
        with self.assertRaisesRegex(ValueError, "executed H4"):
            first_actions(rows, "task", 1)

    def test_first_actions_rejects_different_initial_observation(self):
        rows = [_transition("task", 1, "u0", 0.1), _transition("task", 1, "seed3-u32", 0.2)]
        rows[1]["policy_input_hash"] = "different"
        with self.assertRaisesRegex(ValueError, "policy_input_hash"):
            first_actions(rows, "task", 1)

    def test_arm_donor_and_follow_policy_are_not_interchanged(self):
        self.assertEqual(
            ARM_SPECS,
            {
                "bc_bc": ("u0", "u0"),
                "rl_rl": ("seed3-u32", "seed3-u32"),
                "bc_rl": ("u0", "seed3-u32"),
                "rl_bc": ("seed3-u32", "u0"),
            },
        )

    def test_swap_classifies_both_donor_and_both_follow_outcomes(self):
        self.assertEqual(classify_swap(True, False, True, False), "FIRST_H4_DOMINANT")
        self.assertEqual(classify_swap(True, False, False, True), "DOWNSTREAM_POLICY_DOMINANT")
        self.assertEqual(classify_swap(True, False, True, True), "MIXED")
        with self.assertRaisesRegex(ValueError, "opposite"):
            classify_swap(True, True, True, True)

    def test_baseline_mismatch_stops_before_any_cross_arm(self):
        entry = _entry("break", "anticipatory", "break", 10)
        entry["bc_done_reason"] = "reached"
        entry["seed3_done_reason"] = "time_limit"
        entry["expected_bc_summary"] = {"done_reason": "reached", "steps": 2, "terminal_xyz": [1.0, 0.0, 0.0]}
        entry["expected_seed3_summary"] = {"done_reason": "time_limit", "steps": 2, "terminal_xyz": [1.0, 0.0, 0.0]}
        calls = []

        def run_arm(_entry, arm, donor, follow):
            calls.append((arm, donor, follow))
            return {"done_reason": "time_limit", "decision_count": 2, "terminal_xyz": [1.0, 0.0, 0.0], "transitions": []}

        with self.assertRaisesRegex(ValueError, "baseline"):
            execute_four_arms(entry, run_arm)
        self.assertEqual(calls, [("bc_bc", "u0", "u0")])

    def test_cross_arms_swap_only_first_action_donor(self):
        entry = _entry("rescue", "dense_recovery", "rescue", 10)
        entry["bc_done_reason"] = "collision"
        entry["seed3_done_reason"] = "reached"
        entry["expected_bc_summary"] = {"done_reason": "collision", "steps": 2, "terminal_xyz": [1.0, 0.0, 0.0]}
        entry["expected_seed3_summary"] = {"done_reason": "reached", "steps": 2, "terminal_xyz": [1.0, 0.0, 0.0]}
        calls = []

        def run_arm(_entry, arm, donor, follow):
            calls.append((arm, donor, follow))
            return {
                "done_reason": "reached" if follow == "seed3-u32" else "collision",
                "decision_count": 2, "terminal_xyz": [1.0, 0.0, 0.0],
                "transitions": [],
            }

        result = execute_four_arms(entry, run_arm)
        self.assertEqual(
            calls,
            [
                ("bc_bc", "u0", "u0"),
                ("rl_rl", "seed3-u32", "seed3-u32"),
                ("bc_rl", "u0", "seed3-u32"),
                ("rl_bc", "seed3-u32", "u0"),
            ],
        )
        self.assertEqual(result["swap_pattern"], "DOWNSTREAM_POLICY_DOMINANT")

    def test_same_failure_bit_but_wrong_reason_stops_cross_arms(self):
        entry = _entry("break", "anticipatory", "break", 10)
        entry["bc_done_reason"] = "reached"
        entry["seed3_done_reason"] = "collision"
        entry["expected_bc_summary"] = {"done_reason": "reached", "steps": 2, "terminal_xyz": [1.0, 0.0, 0.0]}
        entry["expected_seed3_summary"] = {"done_reason": "collision", "steps": 2, "terminal_xyz": [1.0, 0.0, 0.0]}
        calls = []

        def run_arm(_entry, arm, donor, follow):
            calls.append(arm)
            return {
                "done_reason": "reached" if arm == "bc_bc" else "time_limit",
                "decision_count": 2, "terminal_xyz": [1.0, 0.0, 0.0], "transitions": [],
            }

        with self.assertRaisesRegex(ValueError, "baseline"):
            execute_four_arms(entry, run_arm)
        self.assertEqual(calls, ["bc_bc", "rl_rl"])

    def test_same_reason_but_wrong_terminal_position_stops_cross_arms(self):
        entry = _entry("break", "anticipatory", "break", 10)
        entry["bc_done_reason"] = "reached"
        entry["seed3_done_reason"] = "collision"
        entry["expected_bc_summary"] = {"done_reason": "reached", "steps": 2, "terminal_xyz": [1.0, 0.0, 0.0]}

        def run_arm(_entry, arm, donor, follow):
            return {"done_reason": "reached", "decision_count": 2, "terminal_xyz": [1.1, 0.0, 0.0], "transitions": []}

        with self.assertRaisesRegex(ValueError, "baseline"):
            execute_four_arms(entry, run_arm)

    def test_frozen_numpy_bundle_and_live_torch_bundle_share_h4_view(self):
        import torch

        frozen = np.arange(88, dtype=np.float32).reshape(8, 11) / 100
        bundle = _action_bundle(frozen)
        np.testing.assert_array_equal(_executed_numpy(bundle), frozen[:4])
        live = type("LiveAction", (), {"rollout_executed": torch.from_numpy(frozen[:4][None].copy()).requires_grad_(True)})()
        np.testing.assert_array_equal(_executed_numpy(live), frozen[:4])

    def test_baseline_path_rejects_second_decision_drift(self):
        rows = [
            {"decision_index": 0, "executed_h4_path_5x3": [[0.0, 0.0, 0.0]] * 5},
            {"decision_index": 1, "executed_h4_path_5x3": [[1.0, 0.0, 0.0]] * 5},
        ]
        _verify_baseline_path(np.asarray([[1.0, 0.0, 0.0]] * 5, dtype=np.float32), rows, 1)
        with self.assertRaisesRegex(ValueError, "baseline path"):
            _verify_baseline_path(np.asarray([[2.0, 0.0, 0.0]] * 5, dtype=np.float32), rows, 1)


if __name__ == "__main__":
    unittest.main()
