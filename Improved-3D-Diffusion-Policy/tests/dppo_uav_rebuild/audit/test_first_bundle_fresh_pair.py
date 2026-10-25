"""Contracts for the one-run fresh first-bundle 2x2 diagnostic."""

import unittest

import numpy as np

from scripts.routea_causal_diagnostics.first_bundle_fresh_pair import (
    classify_task,
    count_classes,
    decompose_first_actions,
    factorial_effects,
    horizontal_vertical_metrics,
    safe_arrival,
    summarize_action_deltas,
    summarize_factorial_effects,
    trajectory_metrics,
)
from scripts.run_routea_first_bundle_fresh_pair import (
    _compare_snapshots,
    _typed,
)


class FreshPairContracts(unittest.TestCase):
    def test_four_informative_patterns_and_denominators(self):
        controls = {"bb": True, "rr": False}
        expected = {
            (True, False): "FIRST_BUNDLE_DOMINANT",
            (False, True): "DOWNSTREAM_POLICY_DOMINANT",
            (True, True): "EITHER_SUCCESS_COMPONENT_SUFFICIENT",
            (False, False): "BOTH_SUCCESS_COMPONENTS_REQUIRED",
        }
        labels = []
        for (br, rb), label in expected.items():
            actual = classify_task(dict(controls, br=br, rb=rb))
            self.assertEqual(actual, label)
            labels.append(actual)
        labels += [classify_task({"bb": True, "rr": True, "br": False, "rb": True}),
                   classify_task({"bb": None, "rr": False, "br": False, "rb": False}),
                   classify_task(controls, pair_valid=False)]
        self.assertEqual(count_classes(labels)["pair_valid"], 6)
        self.assertEqual(count_classes(labels)["endpoint_determinate"], 5)
        self.assertEqual(count_classes(labels)["informative_task_count"], 4)
        self.assertEqual(count_classes(labels)["invalid"], 1)

    def test_action_decomposition_keeps_tail_and_dimension_names(self):
        bc = np.zeros((8, 11), dtype=np.float32)
        rl = bc.copy()
        rl[0, 2] = 2.0
        rl[4, 9] = -3.0
        result = decompose_first_actions(bc, rl, np.zeros(3), np.array([1., 0., 1.]))
        self.assertEqual(result["delta_8x11"][0][2], 2.0)
        self.assertEqual(result["waypoint_11d_l2"][0], 2.0)
        self.assertEqual(result["waypoint_11d_l2"][4], 3.0)
        self.assertEqual(result["executed_by_dimension"]["dpos_world_z"], [2.0, 0.0, 0.0, 0.0])
        self.assertEqual(result["executed_dimension_mean"]["dpos_world_z"], 0.5)
        self.assertEqual(result["executed_dimension_median"]["dpos_world_z"], 0.0)
        self.assertEqual(result["tail_by_dimension"]["dyaw_cw"], [-3.0, 0.0, 0.0, 0.0])

    def test_factorial_and_horizontal_vertical_separation(self):
        effects = factorial_effects({"bb": 1.0, "br": 1.0, "rb": 0.0, "rr": 0.0})
        self.assertEqual(effects, {"h4": 1.0, "downstream": 0.0, "interaction": 0.0})
        metrics = horizontal_vertical_metrics(
            np.array([[0., 0., 1.], [2., 3., 5.]]),
            np.array([10., 0., 1.]), 0.0,
        )
        self.assertEqual(metrics["horizontal_goal_parallel_end_m"], 2.0)
        self.assertEqual(metrics["horizontal_lateral_end_m"], 3.0)
        self.assertEqual(metrics["vertical_end_m"], 4.0)

    def test_continuous_entry_precedes_later_failure_but_same_h4_is_ambiguous(self):
        path = [[0., 0., 1.], [0.5, 0., 1.], [1., 0., 1.],
                [1.5, 0., 1.], [2., 0., 1.]]
        rows = [{"executed_path": path, "physical_failure": False,
                 "done_reason": "none", "clearance_m": 0.8},
                {"executed_path": path, "physical_failure": True,
                 "done_reason": "collision", "clearance_m": -0.1}]
        result = safe_arrival(rows, np.array([1., 0., 1.]))
        self.assertIs(result["success"], True)
        self.assertEqual(result["entry"]["decision"], 0)
        rows[0]["physical_failure"] = True
        self.assertIsNone(safe_arrival(rows, np.array([1., 0., 1.]))["success"])
        self.assertEqual(trajectory_metrics(rows, np.array([1., 0., 1.]), 0.0)["net_goal_progress_m"], 0.0)

    def test_typed_state_detects_dtype_and_field_mismatch(self):
        a = {"position": _typed(np.array([1., 2.], dtype=np.float32)), "step": _typed(1)}
        b = {"position": _typed(np.array([1., 2.], dtype=np.float64)), "step": _typed(1)}
        self.assertEqual(_compare_snapshots(a, a), {})
        self.assertIn("position", _compare_snapshots(a, b))

    def test_task_level_delta_strata_and_effect_signs(self):
        delta = np.zeros((8, 11)).tolist()
        delta[0][2] = 1.0
        tasks = [
            {"task_id": "b1", "fresh_control_label": "fresh_break",
             "action_decomposition": {"delta_8x11": delta},
             "factorial_effects": {"net_goal_progress_m": {"h4": 1., "downstream": 0., "interaction": 0.}}},
            {"task_id": "b2", "fresh_control_label": "fresh_break",
             "action_decomposition": {"delta_8x11": delta},
             "factorial_effects": {"net_goal_progress_m": {"h4": 2., "downstream": 0., "interaction": 0.}}},
            {"task_id": "r1", "fresh_control_label": "fresh_rescue",
             "action_decomposition": {"delta_8x11": np.negative(delta).tolist()},
             "factorial_effects": {"net_goal_progress_m": {"h4": -1., "downstream": 0., "interaction": 0.}}},
        ]
        actions = summarize_action_deltas(tasks)
        self.assertEqual(actions["earliest_repeated_executed_waypoint"], 1)
        self.assertEqual(actions["by_stratum"]["fresh_break"]["waypoint_dimension"][0][2]["positive"], 2)
        effects = summarize_factorial_effects(tasks)
        self.assertEqual(effects["net_goal_progress_m"]["h4"]["denominator"], 3)
        self.assertEqual(effects["net_goal_progress_m"]["h4"]["median"], 1.0)


if __name__ == "__main__":
    unittest.main()
