import unittest

import numpy as np

from scripts.routea_causal_diagnostics.action_feedback import (
    audit_action_feedback,
    decompose_action_pair,
    summarize_b1_progression,
)


def _action(executed_dx=0.0, tail_dx=0.0, yaw=0.0):
    value = np.zeros((8, 11), dtype=np.float32)
    value[:4, 0] = float(executed_dx)
    value[4:, 0] = float(tail_dx)
    value[:4, 9] = float(yaw)
    return value.tolist()


def _transition(task, repeat, lane, decision, action, position_after):
    return {
        "schema": "routea_d05b_transition_v1",
        "task_id": task,
        "repeat": repeat,
        "lane": lane,
        "outcome_class": "break",
        "role": "anticipatory",
        "decision_index": decision,
        "action_seed": 1000 + decision,
        "policy_input_hash": "same" if decision == 0 else lane + str(decision),
        "position_before_xyz": [float(decision), 0.0, 0.0],
        "position_after_xyz": list(position_after),
        "goal_xyz": [10.0, 0.0, 0.0],
        "policy_action_8x11": action,
        "executed_action_4x11": np.asarray(action, dtype=np.float32)[:4].tolist(),
    }


class ActionChunkFeedbackTests(unittest.TestCase):
    def test_decomposition_keeps_executed_and_tail_separate(self):
        base = _action()
        rl = _action(executed_dx=0.25, tail_dx=0.5, yaw=0.1)
        result = decompose_action_pair(
            base,
            rl,
            position_before=[0.0, 0.0, 0.0],
            goal=[10.0, 0.0, 0.0],
        )
        self.assertAlmostEqual(result["executed_dpos_l2"], 0.5)
        self.assertAlmostEqual(result["tail_dpos_l2"], 1.0)
        self.assertAlmostEqual(result["executed_endpoint_delta_m"], 1.0)
        self.assertAlmostEqual(result["goal_parallel_delta_m"], 1.0)
        self.assertAlmostEqual(result["yaw_sum_delta_rad"], 0.4)
        self.assertEqual(len(result["waypoints"]), 8)
        self.assertTrue(result["waypoints"][3]["executed"])
        self.assertFalse(result["waypoints"][4]["executed"])
        waypoint = result["waypoints"][0]
        reconstructed = [
            waypoint["delta_dpos_x"],
            waypoint["delta_dpos_y"],
            waypoint["delta_dpos_z"],
            waypoint["delta_dvel_x"],
            waypoint["delta_dvel_y"],
            waypoint["delta_dvel_z"],
            waypoint["delta_acc_x"],
            waypoint["delta_acc_y"],
            waypoint["delta_acc_z"],
            waypoint["delta_dyaw"],
            waypoint["delta_yaw_rate"],
        ]
        np.testing.assert_allclose(reconstructed, np.asarray(rl)[0] - np.asarray(base)[0])

    def test_decomposition_rejects_wrong_shape_before_indexing(self):
        with self.assertRaisesRegex(ValueError, "8, 11"):
            decompose_action_pair(
                np.zeros((4, 11), dtype=np.float32),
                np.zeros((8, 11), dtype=np.float32),
                position_before=[0.0, 0.0, 0.0],
                goal=[1.0, 0.0, 0.0],
            )

    def test_audit_uses_bc_null_and_fixed_zero_to_five_feedback_endpoint(self):
        task = "task-a"
        selected = {
            "task_id": task,
            "repeat": 1,
            "k_div": 0,
            "decision_count": 6,
            "outcome_class": "break",
            "role": "anticipatory",
            "episode_config": {
                "start_pos": [0.0, 0.0, 0.0],
                "goal_pos": [10.0, 0.0, 0.0],
            },
        }
        rows = []
        for decision in range(6):
            rows.append(
                _transition(
                    task,
                    1,
                    "u0",
                    decision,
                    _action(),
                    [float(decision + 1), 0.0, 0.0],
                )
            )
            rows.append(
                _transition(
                    task,
                    1,
                    "seed3-u32",
                    decision,
                    _action(executed_dx=0.1 * (decision + 1)),
                    [float(decision + 1), 0.1 * (decision + 1), 0.0],
                )
            )
        null = {
            task: {
                str(index): {
                    "h4_path": {"threshold": 0.001},
                    "endpoint": {"threshold": 0.001},
                }
                for index in range(6)
            }
        }
        result = audit_action_feedback([selected], rows, null)
        self.assertEqual(result["summary"]["repeat_count"], 1)
        self.assertEqual(result["summary"]["task_count"], 1)
        self.assertEqual(result["repeat_rows"][0]["primary_feedback_endpoint"], 5)
        self.assertGreater(result["repeat_rows"][0]["feedback_position_gap_growth_m"], 0.0)
        self.assertTrue(result["repeat_rows"][0]["executed_h4_material_vs_bc_null"])
        self.assertEqual(
            {row["relative_decision"] for row in result["progression_rows"]},
            set(range(6)),
        )

    def test_audit_rejects_rng_pairing_mismatch(self):
        selected = {
            "task_id": "task-a",
            "repeat": 1,
            "k_div": 0,
            "decision_count": 1,
            "outcome_class": "break",
            "role": "anticipatory",
            "episode_config": {
                "start_pos": [0.0, 0.0, 0.0],
                "goal_pos": [1.0, 0.0, 0.0],
            },
        }
        left = _transition("task-a", 1, "u0", 0, _action(), [0.0, 0.0, 0.0])
        right = _transition(
            "task-a", 1, "seed3-u32", 0, _action(), [0.0, 0.0, 0.0]
        )
        right["action_seed"] = 999
        null = {
            "task-a": {
                "0": {
                    "h4_path": {"threshold": 0.0},
                    "endpoint": {"threshold": 0.0},
                }
            }
        }
        with self.assertRaisesRegex(ValueError, "action_seed"):
            audit_action_feedback([selected], [left, right], null)

    def test_late_bc_null_may_be_missing_but_divergence_null_is_required(self):
        selected = {
            "task_id": "task-a",
            "repeat": 1,
            "k_div": 0,
            "decision_count": 6,
            "outcome_class": "break",
            "role": "anticipatory",
            "episode_config": {
                "start_pos": [0.0, 0.0, 0.0],
                "goal_pos": [10.0, 0.0, 0.0],
            },
        }
        rows = []
        for decision in range(6):
            rows.append(
                _transition(
                    "task-a", 1, "u0", decision, _action(), [decision + 1.0, 0.0, 0.0]
                )
            )
            rows.append(
                _transition(
                    "task-a",
                    1,
                    "seed3-u32",
                    decision,
                    _action(executed_dx=0.1),
                    [decision + 1.0, 0.1 * (decision + 1), 0.0],
                )
            )
        only_divergence_null = {
            "task-a": {
                "0": {
                    "h4_path": {"threshold": 0.001},
                    "endpoint": {"threshold": 0.001},
                }
            }
        }
        result = audit_action_feedback([selected], rows, only_divergence_null)
        self.assertIsNone(
            result["repeat_rows"][0]["feedback_endpoint_normalized_minus_null"]
        )
        self.assertGreater(
            result["repeat_rows"][0]["feedback_position_gap_growth_m"], 0.0
        )
        self.assertFalse(result["task_rows"][0]["fixed_0_to_5_feedback_supported"])
        self.assertEqual(
            result["task_rows"][0]["fixed_0_to_5_feedback_status"],
            "INDETERMINATE_MISSING_ENDPOINT_NULL",
        )

        with self.assertRaisesRegex(ValueError, "missing BC null threshold"):
            audit_action_feedback([selected], rows, {"task-a": {}})

    def test_feedback_support_requires_fixed_endpoint_above_bc_null(self):
        selected = {
            "task_id": "task-a",
            "repeat": 1,
            "k_div": 0,
            "decision_count": 6,
            "outcome_class": "break",
            "role": "anticipatory",
            "episode_config": {
                "start_pos": [0.0, 0.0, 0.0],
                "goal_pos": [10.0, 0.0, 0.0],
            },
        }
        rows = []
        for decision in range(6):
            rows.append(
                _transition(
                    "task-a", 1, "u0", decision, _action(), [decision + 1.0, 0.0, 0.0]
                )
            )
            rows.append(
                _transition(
                    "task-a",
                    1,
                    "seed3-u32",
                    decision,
                    _action(executed_dx=0.1),
                    [decision + 1.0, 0.1 * (decision + 1), 0.0],
                )
            )
        null = {
            "task-a": {
                str(index): {
                    "h4_path": {"threshold": 0.001},
                    "endpoint": {"threshold": 1.0 if index == 5 else 0.001},
                }
                for index in range(6)
            }
        }
        result = audit_action_feedback([selected], rows, null)
        self.assertGreater(
            result["repeat_rows"][0]["feedback_position_gap_growth_m"], 0.0
        )
        self.assertLess(
            result["repeat_rows"][0]["feedback_endpoint_normalized_minus_null"], 0.0
        )
        self.assertFalse(result["task_rows"][0]["fixed_0_to_5_feedback_supported"])
        self.assertEqual(
            result["task_rows"][0]["fixed_0_to_5_feedback_status"],
            "NO_SUPPORT_ENDPOINT_WITHIN_BC_NULL",
        )

    def test_b1_summary_keeps_all_three_seeds_and_updates(self):
        rows = []
        for seed in (1, 2, 3):
            for update in (8, 16, 24, 32):
                rows.append(
                    {
                        "checkpoint": "seed{}-u{}".format(seed, update),
                        "update": str(update),
                        "stratum": "S1_SEED1_RESCUE",
                        "task_id": "task-a",
                        "snapshot_id": "snapshot-a",
                        "action_h4_l2": str(seed * update / 100.0),
                        "goal_parallel_m": str(seed / 10.0),
                        "goal_perpendicular_m": str(-seed / 20.0),
                        "vertical_m": "0.0",
                        "yaw_total_variation_rad": str(update / 1000.0),
                        "path_shape_norm_m": "0.01",
                    }
                )
        summary = summarize_b1_progression(rows)
        overall = [row for row in summary if row["stratum"] == "ALL"]
        self.assertEqual(len(overall), 12)
        self.assertEqual({row["seed"] for row in overall}, {1, 2, 3})
        self.assertEqual({row["update"] for row in overall}, {8, 16, 24, 32})

    def test_route_candidate_is_driven_by_break_tasks_not_rescue_controls(self):
        selected_rows = []
        transitions = []
        nulls = {}
        for index in range(2):
            task = "rescue-{}".format(index)
            selected_rows.append(
                {
                    "task_id": task,
                    "repeat": 1,
                    "k_div": 0,
                    "decision_count": 1,
                    "outcome_class": "rescue",
                    "role": "anticipatory",
                    "episode_config": {
                        "start_pos": [0.0, 0.0, 0.0],
                        "goal_pos": [10.0, 0.0, 0.0],
                    },
                }
            )
            transitions.append(
                _transition(task, 1, "u0", 0, _action(), [1.0, 0.0, 0.0])
            )
            transitions.append(
                _transition(
                    task,
                    1,
                    "seed3-u32",
                    0,
                    _action(executed_dx=0.2),
                    [1.0, 0.2, 0.0],
                )
            )
            nulls[task] = {
                "0": {
                    "h4_path": {"threshold": 0.001},
                    "endpoint": {"threshold": 0.001},
                }
            }
        result = audit_action_feedback(selected_rows, transitions, nulls)
        self.assertEqual(
            result["summary"]["route_candidate"],
            "ACTION_CHUNK_ROUTE_NOT_DETERMINATE",
        )
        self.assertEqual(result["summary"]["rescue_executed_h4_material_task_count"], 2)
        self.assertEqual(result["summary"]["break_executed_h4_material_task_count"], 0)

    def test_b1_metrics_are_collapsed_to_tasks_before_lane_statistics(self):
        rows = []
        for seed in (1, 2, 3):
            for update in (8, 16, 24, 32):
                for task, copies, value in (("task-a", 3, 0.0), ("task-b", 1, 10.0)):
                    for copy_index in range(copies):
                        rows.append(
                            {
                                "checkpoint": "seed{}-u{}".format(seed, update),
                                "update": str(update),
                                "stratum": "S1_SEED1_RESCUE",
                                "task_id": task,
                                "snapshot_id": "{}-{}".format(task, copy_index),
                                "action_h4_l2": str(value),
                                "goal_parallel_m": "0.0",
                                "goal_perpendicular_m": "0.0",
                                "vertical_m": "0.0",
                                "yaw_total_variation_rad": "0.0",
                                "path_shape_norm_m": "0.0",
                            }
                        )
        summary = summarize_b1_progression(rows)
        lane = next(
            row
            for row in summary
            if row["seed"] == 1 and row["update"] == 8 and row["stratum"] == "ALL"
        )
        self.assertEqual(lane["snapshot_count"], 4)
        self.assertEqual(lane["task_count"], 2)
        self.assertAlmostEqual(lane["action_h4_l2_task_mean"], 5.0)
        self.assertAlmostEqual(lane["action_h4_l2_task_median"], 5.0)

    def test_b1_rejects_incomparable_snapshot_banks(self):
        rows = []
        for seed in (1, 2, 3):
            for update in (8, 16, 24, 32):
                rows.append(
                    {
                        "checkpoint": "seed{}-u{}".format(seed, update),
                        "update": str(update),
                        "stratum": "S1_SEED1_RESCUE",
                        "task_id": "task-a",
                        "snapshot_id": (
                            "wrong" if seed == 3 and update == 32 else "snapshot-a"
                        ),
                        "action_h4_l2": "0.1",
                        "goal_parallel_m": "0.0",
                        "goal_perpendicular_m": "0.0",
                        "vertical_m": "0.0",
                        "yaw_total_variation_rad": "0.0",
                        "path_shape_norm_m": "0.0",
                    }
                )
        with self.assertRaisesRegex(ValueError, "snapshot bank"):
            summarize_b1_progression(rows)


if __name__ == "__main__":
    unittest.main()
