from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from diffusion_policy_3d.env.traj_utils import action_to_world
from scripts import probe_routea_goal_sensitivity as probe_module
from scripts.probe_routea_goal_sensitivity import _frozen_plan_action
from scripts.routea_causal_diagnostics import failure_state as failure_state_module
from scripts.routea_causal_diagnostics.failure_state import (
    BOTH_BELOW_FUNCTIONAL_REFERENCE,
    GOAL_RESPONSE_PRESERVED,
    RL_RETENTION_OR_DIRECTION_FAILURE,
    anchor_decisions,
    build_failure_summary,
    build_bc_repeat_null,
    build_functional_reference,
    classify_failure_snapshot,
    exact_progression_test,
    endpoint_identity_audit,
    extract_decision_features,
    find_first_persistent_divergence,
    response_direction_metrics,
    serialize_bc_repeat_null,
    support_summary,
)
from scripts.routea_causal_diagnostics.contracts import (
    DiagnosticFailure,
    write_numeric_npz,
    write_strict_json,
)
from scripts.routea_causal_diagnostics.reporting import (
    D1F_MINIMUM_FINAL_OUTPUTS,
    D1F_SELECTION_BOUND_FILES,
    d1f_selection_identity,
    file_sha256,
    finalize_d1f_outputs,
    publish_stage_manifest,
    write_csv_rows,
    write_jsonl,
)


SOURCE_MANIFEST = {
    "schema": "routea_causal_diagnostic_source_manifest_v1",
    "content_sha256": "a" * 64,
    "file_count": 1,
    "files": [{"path": "scripts/fake.py", "bytes": 1, "sha256": "b" * 64}],
}


class FrozenTrajectoryReplayTests(unittest.TestCase):
    def test_runtime_gate_rejects_one_ulp_even_when_allclose_passes(self):
        check = getattr(probe_module, "_assert_bitwise_trajectory_match", None)
        self.assertIsNotNone(check, "missing runtime bitwise trajectory gate")
        if check is None:
            return
        frozen_planned = np.zeros((1, 14), dtype=np.float64)
        rebuilt_planned = frozen_planned.copy()
        rebuilt_planned[0, 12] = np.nextafter(np.float64(0.0), np.float64(1.0))
        frozen_executed = np.zeros((1, 6), dtype=np.float64)
        self.assertTrue(
            np.allclose(rebuilt_planned, frozen_planned, rtol=1e-6, atol=1e-6)
        )

        with self.assertRaises(DiagnosticFailure) as caught:
            check(
                task_id="task",
                rebuilt_planned=rebuilt_planned,
                frozen_planned=frozen_planned,
                rebuilt_executed=frozen_executed,
                frozen_executed=frozen_executed,
            )

        self.assertEqual(caught.exception.status, "SNAPSHOT_REPLAY_MISMATCH")
        self.assertEqual(caught.exception.exit_code, 10)

    def test_real_yaw_counterexample_round_trips_bitwise(self):
        """Catch one-ULP yaw drift that changes the latest-3 normalized history."""

        before = {
            "position": np.zeros(3, dtype=np.float32),
            "velocity_reference": np.zeros(3, dtype=np.float32),
            "yaw": 0.13593871891498566,
        }
        planned = np.zeros((8, 11), dtype=np.float32)
        planned[:, 9] = np.asarray(
            [
                0.11403390765190125,
                0.011591341346502304,
                -0.08387592434883118,
                -0.18065477907657623,
                -0.2659946382045746,
                -0.34429970383644104,
                -0.39108723402023315,
                -0.40956050157546997,
            ],
            dtype=np.float32,
        )
        rows = np.column_stack(
            (
                np.ones(8),
                np.arange(8),
                np.arange(1, 9) * 0.3,
                planned,
            )
        )

        action = _frozen_plan_action(
            rows,
            decision_step=1,
            before=before,
            exact_bitwise=True,
        )
        rebuilt = action_to_world(
            action.policy[0],
            before["position"],
            before["velocity_reference"],
            before["yaw"],
        )

        np.testing.assert_array_equal(
            rebuilt[:, 9].view(np.uint32),
            planned[:, 9].view(np.uint32),
        )

    def test_frozen_plan_rejects_any_non_bitwise_position_inverse(self):
        before = {
            "position": np.asarray([1.0e8, 0.0, 0.0], dtype=np.float32),
            "velocity_reference": np.zeros(3, dtype=np.float32),
            "yaw": 0.0,
        }
        planned = np.zeros((8, 11), dtype=np.float32)
        planned[:, 0] = 1.0
        rows = np.column_stack(
            (
                np.ones(8),
                np.arange(8),
                np.arange(1, 9) * 0.3,
                planned,
            )
        )

        with self.assertRaisesRegex(ValueError, "bitwise exact"):
            _frozen_plan_action(
                rows,
                decision_step=1,
                before=before,
                exact_bitwise=True,
            )

    def test_b0_legacy_inverse_preserves_tolerance_contract(self):
        before = {
            "position": np.asarray([1.0e8, 0.0, 0.0], dtype=np.float32),
            "velocity_reference": np.zeros(3, dtype=np.float32),
            "yaw": 0.0,
        }
        planned = np.zeros((8, 11), dtype=np.float32)
        planned[:, 0] = 1.0
        rows = np.column_stack(
            (
                np.ones(8),
                np.arange(8),
                np.arange(1, 9) * 0.3,
                planned,
            )
        )

        action = _frozen_plan_action(rows, decision_step=1, before=before)

        self.assertEqual(action.policy.shape, (1, 8, 11))


def _feature(value: float) -> dict[str, float]:
    return {
        "endpoint": value,
        "h4_path": value,
        "goal_parallel": value,
        "goal_perpendicular": value,
        "vertical": value,
        "yaw": value,
    }


class DivergenceContractTests(unittest.TestCase):
    def test_replay_identity_diagnostics_separates_trajectory_and_encoder_layers(self):
        diagnose = getattr(failure_state_module, "build_replay_identity_diagnostics", None)
        self.assertIsNotNone(diagnose, "missing replay identity layer diagnostics")
        if diagnose is None:
            return
        frozen = np.zeros((2, 14), dtype=np.float64)
        rebuilt = frozen.copy()
        rebuilt[1, 12] = np.nextafter(np.float64(0.0), np.float64(1.0))
        result = diagnose(
            task_id="task",
            expected_observation={
                "last_raw_frame_hash": "a" * 64,
                "last_normalized_history_hash": "b" * 64,
                "last_policy_input_hash": "c" * 64,
            },
            actual_provenance=SimpleNamespace(
                raw_frame_hash="a" * 64,
                raw_history_hash="d" * 64,
                normalizer_identity_hash="e" * 64,
                normalized_history_hash="b" * 64,
                policy_input_hash="f" * 64,
                episode_identity_hash="1" * 64,
                decision_index=99,
            ),
            rebuilt_planned=rebuilt,
            frozen_planned=frozen,
        )
        self.assertEqual(result["first_mismatch_layer"], "ENCODER_CONDITION")
        self.assertTrue(result["comparisons"]["raw_frame_exact"])
        self.assertTrue(result["comparisons"]["normalized_history_exact"])
        self.assertFalse(result["comparisons"]["policy_input_exact"])
        self.assertFalse(result["trajectory_roundtrip"]["bitwise_exact"])
        self.assertEqual(result["trajectory_roundtrip"]["first_mismatch"]["column"], 12)

    def test_endpoint_population_excludes_continuous_only_breaks(self):
        def parent(task, bc_reached, seed3_reached, bc_done, seed3_done):
            bc_path, seed3_path = f"{task}-bc.csv", f"{task}-s3.csv"
            return (
                SimpleNamespace(
                    task_id=task, repeat=1, env_reset_seed=1, world="w", split="s", role="r",
                    bc_reached=bc_reached, seed3_reached=seed3_reached,
                    trace_paths=(bc_path, "s1", "s2", seed3_path),
                    bc_trace_sha256="a" * 64,
                    rl_trace_sha256=("1" * 64, "2" * 64, "3" * 64),
                ),
                {bc_path: bc_done, seed3_path: seed3_done},
            )
        main, reasons_main = parent("main", True, False, "reached", "collision")
        continuous_only, reasons_extra = parent("extra", True, False, "timeout", "collision")
        stable, reasons_stable = parent("stable", True, True, "reached", "reached")
        identities, audit = endpoint_identity_audit(
            [main, continuous_only, stable],
            {**reasons_main, **reasons_extra, **reasons_stable},
        )
        self.assertEqual([item.task_id for item in identities], ["main"])
        self.assertEqual(audit["endpoint_break_repeat_count"], 1)
        self.assertEqual(audit["continuous_break_repeat_count"], 2)
        self.assertEqual(audit["continuous_only_repeat_count"], 1)

    def test_extract_decision_features_ignores_unexecuted_h8_tail(self):
        rows = []
        for decision in (1, 2):
            for waypoint in range(8):
                x = float((decision - 1) * 4 + waypoint + 1)
                rows.append([decision, waypoint, 0.3 * (waypoint + 1), x, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1 * x, 0.0])
        planned = np.asarray(rows, dtype=np.float64)
        altered = planned.copy()
        altered[np.isin(altered[:, 1], [4, 5, 6, 7]), 3:14] += 1000.0
        first = extract_decision_features(planned, start=[0, 0, 0], goal=[10, 0, 0])
        second = extract_decision_features(altered, start=[0, 0, 0], goal=[10, 0, 0])
        self.assertEqual(len(first), 2)
        for key in ("endpoint", "h4_path", "goal_parallel", "goal_perpendicular", "vertical", "yaw"):
            np.testing.assert_allclose(first[0][key], second[0][key])

    def test_bc_null_requires_four_repeats_and_uses_pairwise_q95(self):
        repeats = {
            repeat: [_feature(float(repeat - 1)), _feature(float(repeat - 1) * 2.0)]
            for repeat in range(1, 5)
        }
        null = build_bc_repeat_null(repeats)
        self.assertEqual(null[0]["endpoint"]["pair_count"], 6)
        self.assertEqual(null[0]["endpoint"]["threshold"], 3.0)
        with self.assertRaisesRegex(ValueError, "at least four"):
            build_bc_repeat_null({key: value for key, value in repeats.items() if key < 4})

    def test_bc_null_serialization_converts_decision_keys_only_at_json_boundary(self):
        payload = serialize_bc_repeat_null({"task": {0: {"endpoint": {"threshold": 1.0}}}})
        self.assertEqual(payload, {"task": {"0": {"endpoint": {"threshold": 1.0}}}})

    def test_first_divergence_requires_same_metric_for_two_decisions(self):
        null = {
            index: {
                name: {"threshold": 1.0, "pair_count": 10}
                for name in _feature(0.0)
            }
            for index in range(4)
        }
        bc = [_feature(0.0) for _ in range(4)]
        rl = [_feature(0.0), _feature(2.0), _feature(2.5), _feature(0.0)]
        self.assertEqual(find_first_persistent_divergence(bc, rl, null), 1)
        rl[2] = _feature(0.0)
        self.assertIsNone(find_first_persistent_divergence(bc, rl, null))

    def test_anchor_dedup_and_no_divergence_contract(self):
        anchors = anchor_decisions(5, decision_count=7, closest_index=6)
        self.assertEqual(anchors[4], ("pre_divergence",))
        self.assertEqual(anchors[5], ("divergence",))
        self.assertEqual(
            anchors[6],
            ("post_divergence_1", "closest_goal", "preterminal"),
        )
        self.assertEqual(
            anchor_decisions(None, decision_count=3, closest_index=2),
            {2: ("closest_goal", "preterminal")},
        )


class FunctionalDirectionTests(unittest.TestCase):
    def test_functional_floor_is_not_measurement_floor(self):
        rows = [
            {"bearing_norm": 0.01 + index * 0.01, "vertical_norm": 0.02 + index * 0.01,
             "combined_norm": 0.03 + index * 0.01, "abs_bearing_yaw_delta": 0.004 + index * 0.001}
            for index in range(8)
        ]
        reference = build_functional_reference(rows, measurement_floor=0.0)
        self.assertEqual(reference["channels"]["bearing_norm"]["functional_floor"], 0.01)
        self.assertFalse(0.005 >= reference["channels"]["bearing_norm"]["functional_floor"])
        self.assertTrue(reference["valid"])

    def test_reference_rejects_fewer_than_eight_or_nonpositive_floor(self):
        row = {"bearing_norm": 1.0, "vertical_norm": 1.0, "combined_norm": 1.0,
               "abs_bearing_yaw_delta": 1.0}
        self.assertFalse(build_functional_reference([row] * 7, 0.0)["valid"])
        zero = dict(row, vertical_norm=0.0)
        self.assertFalse(build_functional_reference([zero] * 8, 0.0)["valid"])

    def test_direction_metrics_detect_equal_norm_opposite_direction_and_assignment(self):
        plus_goal = np.asarray([2.0, 0.0, 0.0])
        minus_goal = np.asarray([-2.0, 0.0, 0.0])
        metrics = response_direction_metrics(
            bc_plus=np.asarray([1.0, 0.0, 0.0]),
            bc_minus=np.asarray([-1.0, 0.0, 0.0]),
            rl_plus=np.asarray([-1.0, 0.0, 0.0]),
            rl_minus=np.asarray([1.0, 0.0, 0.0]),
            plus_goal=plus_goal,
            minus_goal=minus_goal,
            bc_yaw_delta=0.2,
            rl_yaw_delta=-0.2,
            bc_floor=0.5,
            rl_floor=0.5,
            yaw_floor=0.1,
        )
        self.assertAlmostEqual(metrics["cosine"], -1.0)
        self.assertGreater(metrics["bc_assignment_margin"], 0.0)
        self.assertLess(metrics["rl_assignment_margin"], 0.0)
        self.assertFalse(metrics["yaw_sign_consistent"])

    def test_near_zero_response_has_null_cosine(self):
        metrics = response_direction_metrics(
            bc_plus=np.zeros(3), bc_minus=np.zeros(3),
            rl_plus=np.ones(3), rl_minus=np.zeros(3),
            plus_goal=np.ones(3), minus_goal=-np.ones(3),
            bc_yaw_delta=0.0, rl_yaw_delta=1.0,
            bc_floor=0.1, rl_floor=0.1, yaw_floor=0.1,
        )
        self.assertIsNone(metrics["cosine"])

    def test_exact_zero_response_has_null_cosine_even_with_zero_measurement_floor(self):
        metrics = response_direction_metrics(
            bc_plus=np.zeros(3), bc_minus=np.zeros(3),
            rl_plus=np.zeros(3), rl_minus=np.zeros(3),
            plus_goal=np.ones(3), minus_goal=-np.ones(3),
            bc_yaw_delta=0.0, rl_yaw_delta=0.0,
            bc_floor=0.0, rl_floor=0.0, yaw_floor=0.0,
        )
        self.assertIsNone(metrics["cosine"])

    def test_response_norm_uses_registered_perturbation_denominator(self):
        metrics = response_direction_metrics(
            bc_plus=np.asarray([0.3, 0.0, 0.0]), bc_minus=np.zeros(3),
            rl_plus=np.asarray([0.6, 0.0, 0.0]), rl_minus=np.zeros(3),
            plus_goal=np.asarray([1.0, 0.0, 0.0]), minus_goal=np.asarray([-1.0, 0.0, 0.0]),
            bc_yaw_delta=0.0, rl_yaw_delta=0.0,
            bc_floor=0.1, rl_floor=0.1, yaw_floor=0.1,
            response_denominator=0.6,
        )
        self.assertAlmostEqual(metrics["bc_norm"], 0.5)
        self.assertAlmostEqual(metrics["rl_norm"], 1.0)

    def test_snapshot_classification_uses_amplitude_direction_margin_and_yaw(self):
        floors = {"bearing_norm": 0.1, "vertical_norm": 0.1, "abs_bearing_yaw_delta": 0.05}
        both_below = {
            "bearing": {"bc_norm": 0.05, "rl_norm": 0.04},
            "vertical": {"bc_norm": 0.05, "rl_norm": 0.04},
        }
        self.assertEqual(classify_failure_snapshot(both_below, floors), BOTH_BELOW_FUNCTIONAL_REFERENCE)
        failure = {
            "bearing": {"bc_norm": 0.2, "rl_norm": 0.2, "bc_assignment_margin": 1.0,
                        "rl_assignment_margin": -1.0, "cosine": -1.0,
                        "bc_yaw_delta": 0.1, "yaw_sign_consistent": False},
            "vertical": {"bc_norm": 0.2, "rl_norm": 0.2, "bc_assignment_margin": 1.0,
                         "rl_assignment_margin": 1.0, "cosine": 1.0},
        }
        self.assertEqual(classify_failure_snapshot(failure, floors), RL_RETENTION_OR_DIRECTION_FAILURE)
        preserved = {
            "bearing": {"bc_norm": 0.2, "rl_norm": 0.2, "bc_assignment_margin": 1.0,
                        "rl_assignment_margin": 1.0, "cosine": 1.0,
                        "bc_yaw_delta": 0.1, "yaw_sign_consistent": True},
            "vertical": {"bc_norm": 0.2, "rl_norm": 0.2, "bc_assignment_margin": 1.0,
                         "rl_assignment_margin": 1.0, "cosine": 1.0},
        }
        self.assertEqual(classify_failure_snapshot(preserved, floors), GOAL_RESPONSE_PRESERVED)

    def test_preserved_in_one_channel_and_below_in_other_is_mixed(self):
        floors = {"bearing_norm": 0.1, "vertical_norm": 0.1, "abs_bearing_yaw_delta": 0.05}
        channels = {
            "bearing": {"bc_norm": 0.2, "rl_norm": 0.2, "bc_assignment_margin": 1.0,
                        "rl_assignment_margin": 1.0, "cosine": 1.0,
                        "bc_yaw_delta": 0.1, "yaw_sign_consistent": True},
            "vertical": {"bc_norm": 0.05, "rl_norm": 0.04},
        }
        self.assertEqual(
            classify_failure_snapshot(channels, floors),
            "BC_RECOVERY_REFERENCE_INVALID_OR_MIXED",
        )


class TaskInferenceTests(unittest.TestCase):
    def test_two_of_forty_one_is_repeated_support_not_primary(self):
        summary = support_summary(2, 41)
        self.assertEqual(summary["support_level"], "REPEATED_SUPPORT")
        self.assertLess(summary["wilson_95_lower"], 0.5)

    def test_twenty_eight_of_forty_one_crosses_primary_wilson_gate(self):
        summary = support_summary(28, 41)
        self.assertEqual(summary["support_level"], "PRIMARY_ROUTE_SUPPORTED")
        self.assertAlmostEqual(summary["wilson_95_lower"], 0.530158, places=5)

    def test_progression_is_fixed_to_pre_divergence_and_preterminal(self):
        rows = [
            {"task_id": "a", "stage": "pre_divergence", "mechanism": False},
            {"task_id": "a", "stage": "closest_goal", "mechanism": False},
            {"task_id": "a", "stage": "preterminal", "mechanism": True},
            {"task_id": "b", "stage": "pre_divergence", "mechanism": True},
            {"task_id": "b", "stage": "closest_goal", "mechanism": False},
            {"task_id": "b", "stage": "preterminal", "mechanism": False},
            {"task_id": "c", "stage": "pre_divergence", "mechanism": False},
            {"task_id": "c", "stage": "closest_goal", "mechanism": True},
            {"task_id": "d", "stage": "closest_goal", "mechanism": True},
        ]
        result = exact_progression_test(rows)
        self.assertEqual(result["pre_stage"], "pre_divergence")
        self.assertEqual(result["late_stage"], "preterminal")
        self.assertEqual(result["paired_task_count"], 2)
        self.assertEqual(result["pre_normal_to_late_failure"], 1)
        self.assertEqual(result["pre_failure_to_late_normal"], 1)
        self.assertEqual(result["missing_fixed_endpoint_task_count"], 2)
        self.assertEqual(result["intermediate_stages_role"], "DESCRIPTIVE_ONLY")
        self.assertTrue(math.isclose(result["one_sided_exact_p"], 0.75))

    def test_indeterminate_fixed_endpoint_reduces_progression_pair_denominator(self):
        rows = [
            {
                "task_id": "indeterminate-before",
                "stage": "pre_divergence",
                "mechanism": False,
                "determinate": False,
            },
            {
                "task_id": "indeterminate-before",
                "stage": "preterminal",
                "mechanism": True,
                "determinate": True,
            },
            {
                "task_id": "fully-determinate",
                "stage": "pre_divergence",
                "mechanism": False,
                "determinate": True,
            },
            {
                "task_id": "fully-determinate",
                "stage": "preterminal",
                "mechanism": True,
                "determinate": True,
            },
        ]
        result = exact_progression_test(rows)
        self.assertEqual(result["paired_task_count"], 1)
        self.assertEqual(result["missing_fixed_endpoint_task_count"], 1)
        self.assertEqual(result["pre_normal_to_late_failure"], 1)

    def test_indeterminate_tasks_are_excluded_from_coverage_denominators(self):
        classifications = (
            [GOAL_RESPONSE_PRESERVED] * 28
            + [RL_RETENTION_OR_DIRECTION_FAILURE] * 9
            + ["BC_RECOVERY_REFERENCE_INVALID_OR_MIXED"] * 4
        )
        rows = [
            {
                "task_id": f"t{index}",
                "classification": classification,
                "anchor_labels": ["preterminal"],
                "split": "rl_test",
                "role": "dense_recovery",
                "goal_distance": 1.0,
            }
            for index, classification in enumerate(classifications)
        ]
        summary = build_failure_summary(rows, shared_primary_mechanisms=())
        preserved = summary["overall_coverage"]["GOAL_CONTROL_PRESERVED"]
        retention = summary["overall_coverage"]["RL_RETENTION_FAILURE"]
        stage = summary["stage_coverage"]["GOAL_CONTROL_PRESERVED"]["preterminal"]

        self.assertEqual(summary["independent_task_count"], 41)
        self.assertEqual(summary["determinate_task_count"], 37)
        self.assertEqual(summary["indeterminate_task_count"], 4)
        self.assertEqual(preserved["determinate_task_denominator"], 37)
        self.assertEqual(preserved["mechanism_task_numerator"], 28)
        self.assertAlmostEqual(preserved["coverage_fraction"], 28 / 37)
        self.assertEqual(retention["determinate_task_denominator"], 37)
        self.assertEqual(stage["determinate_task_denominator"], 37)
        self.assertEqual(
            summary["verdict"]["status"],
            "GOAL_CONTROL_PRESERVED_ON_FAILURE_STATES",
        )

    def test_overall_route_requires_primary_coverage_and_preserves_boundaries(self):
        rows = []
        for index in range(41):
            rows.append({
                "task_id": f"t{index}",
                "classification": (
                    RL_RETENTION_OR_DIRECTION_FAILURE if index < 28 else GOAL_RESPONSE_PRESERVED
                ),
                "anchor_labels": ["preterminal"],
                "split": "rl_test",
                "role": "dense_recovery",
                "goal_distance": 1.0,
            })
        summary = build_failure_summary(rows, shared_primary_mechanisms=())
        self.assertEqual(summary["verdict"]["status"], "STATE_DEPENDENT_GOAL_CONTROL_FAILURE")
        self.assertEqual(summary["verdict"]["route"], "CAPABILITY_RETENTION_AND_RL_UPDATE_CONSTRAINTS")
        self.assertEqual(summary["overall_coverage"]["RL_RETENTION_FAILURE"]["mechanism_task_numerator"], 28)
        self.assertIn("start retraining", " ".join(summary["verdict"]["forbidden_actions"]))

    def test_shared_primary_same_mechanism_forces_inconclusive(self):
        rows = [
            {"task_id": f"t{index}", "classification": RL_RETENTION_OR_DIRECTION_FAILURE,
             "anchor_labels": ["preterminal"], "split": "rl_test", "role": "r", "goal_distance": 1.0}
            for index in range(3)
        ]
        summary = build_failure_summary(rows, shared_primary_mechanisms=("RL_RETENTION_FAILURE",))
        self.assertEqual(summary["verdict"]["status"], "INCONCLUSIVE_MIXED_FAILURE_STATE_RESPONSE")

    def test_repeats_snapshots_and_overlapping_labels_do_not_expand_task_n(self):
        rows = [
            {"task_id": "a", "classification": RL_RETENTION_OR_DIRECTION_FAILURE,
             "anchor_labels": ["divergence", "closest_goal"], "split": "x", "role": "r", "goal_distance": 1.0},
            {"task_id": "a", "classification": RL_RETENTION_OR_DIRECTION_FAILURE,
             "anchor_labels": ["preterminal"], "split": "x", "role": "r", "goal_distance": 0.5},
            {"task_id": "b", "classification": GOAL_RESPONSE_PRESERVED,
             "anchor_labels": ["preterminal"], "split": "x", "role": "r", "goal_distance": 0.5},
        ]
        summary = build_failure_summary(rows)
        self.assertEqual(summary["independent_task_count"], 2)
        self.assertEqual(
            summary["overall_coverage"]["RL_RETENTION_FAILURE"]["mechanism_task_numerator"], 1
        )

    def test_snapshot_count_cannot_overweight_one_repeat_in_task_collapse(self):
        rows = [
            {"task_id": "a", "repeat": 1, "classification": RL_RETENTION_OR_DIRECTION_FAILURE,
             "anchor_labels": ["preterminal"], "split": "x", "role": "r", "goal_distance": 1.0}
            for _ in range(5)
        ]
        rows.append(
            {"task_id": "a", "repeat": 2, "classification": GOAL_RESPONSE_PRESERVED,
             "anchor_labels": ["preterminal"], "split": "x", "role": "r", "goal_distance": 1.0}
        )
        summary = build_failure_summary(rows)
        self.assertEqual(
            summary["task_classification"]["a"],
            "BC_RECOVERY_REFERENCE_INVALID_OR_MIXED",
        )

    def test_intermediate_stage_cannot_drive_primary_route(self):
        rows = []
        for index in range(41):
            rows.extend(
                [
                    {"task_id": f"t{index}", "repeat": 1,
                     "classification": RL_RETENTION_OR_DIRECTION_FAILURE,
                     "anchor_labels": ["closest_goal"], "split": "x", "role": "r", "goal_distance": 1.0},
                    {"task_id": f"t{index}", "repeat": 1,
                     "classification": GOAL_RESPONSE_PRESERVED,
                     "anchor_labels": ["preterminal"], "split": "x", "role": "r", "goal_distance": 0.5},
                ]
            )
        summary = build_failure_summary(rows)
        self.assertEqual(summary["primary_coverage_stage"], "preterminal")
        self.assertEqual(summary["verdict"]["status"], "GOAL_CONTROL_PRESERVED_ON_FAILURE_STATES")
        self.assertEqual(
            summary["stage_coverage"]["RL_RETENTION_FAILURE"]["closest_goal"]["mechanism_task_numerator"],
            41,
        )


class D1FCliSurfaceTests(unittest.TestCase):
    def test_all_d1f_stage_commands_have_help(self):
        repo_root = Path(__file__).parents[3]
        script = repo_root / "scripts" / "probe_routea_goal_sensitivity.py"
        for command in (
            "select-d1f",
            "capture-d1f",
            "run-d1f",
            "finalize-d1f",
            "reaudit-d1f",
        ):
            result = subprocess.run(
                [sys.executable, str(script), command, "--help"],
                cwd=repo_root,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("--output-dir", result.stdout)

    def test_reaudit_d1f_recomputes_from_verified_parent_without_forward(self):
        repo_root = Path(__file__).parents[3]
        script = repo_root / "scripts" / "probe_routea_goal_sensitivity.py"

        def channel(*, functional: bool = True, retained: bool = True):
            if not functional:
                return {
                    "bc_norm": 0.05,
                    "rl_norm": 0.04,
                    "bc_assignment_margin": 1.0,
                    "rl_assignment_margin": 1.0,
                    "cosine": 1.0,
                    "bc_yaw_delta": 0.0,
                    "rl_yaw_delta": 0.0,
                    "yaw_sign_consistent": None,
                }
            return {
                "bc_norm": 0.2,
                "rl_norm": 0.2,
                "bc_assignment_margin": 1.0,
                "rl_assignment_margin": 1.0 if retained else -1.0,
                "cosine": 1.0 if retained else -1.0,
                "bc_yaw_delta": 0.2,
                "rl_yaw_delta": 0.2 if retained else -0.2,
                "yaw_sign_consistent": retained,
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "parent-final"
            output = root / "reaudit"
            parent.mkdir()
            floors = {
                "bearing_norm": 0.1,
                "vertical_norm": 0.1,
                "combined_norm": 0.1,
                "abs_bearing_yaw_delta": 0.05,
            }
            reference = {
                "valid": True,
                "shared_primary_mechanisms": [],
                "channels": {
                    name: {"functional_floor": value}
                    for name, value in floors.items()
                },
            }
            write_strict_json(parent / "shared_goal_response_reference.json", reference)

            classifications = (["preserved"] * 28 + ["retention"] * 9 + ["mixed"] * 4)
            jacobian_rows = []
            for index, classification in enumerate(classifications):
                bearing = channel(retained=classification != "retention")
                vertical = channel(functional=classification != "mixed")
                jacobian_rows.append(
                    {
                        "snapshot_id": f"s{index}",
                        "task_id": f"t{index}",
                        "repeat": 1,
                        "decision_index": 1,
                        "anchor_labels": ["preterminal"],
                        "split": "rl_test",
                        "role": "dense_recovery",
                        "goal_distance": 1.0,
                        "relative_goal_distance": 0.1,
                        "distance_band": "closure",
                        "bearing": bearing,
                        "vertical": vertical,
                        "bc_combined_norm": 0.2,
                        "seed3_combined_norm": 0.2,
                        "classification": (
                            GOAL_RESPONSE_PRESERVED
                            if classification == "preserved"
                            else (
                                RL_RETENTION_OR_DIRECTION_FAILURE
                                if classification == "retention"
                                else "BC_RECOVERY_REFERENCE_INVALID_OR_MIXED"
                            )
                        ),
                    }
                )
            write_csv_rows(parent / "failure_state_jacobians.csv", jacobian_rows)
            parent_identity = {
                "run_lineage_id": "d1f-parent-lineage",
                "diagnostic_source_sha256": SOURCE_MANIFEST["content_sha256"],
            }
            write_strict_json(parent / "identity.json", parent_identity)
            write_strict_json(
                parent / "failure_state_summary.json",
                {
                    "input_identity": {
                        **parent_identity,
                        "shared_reference_sha256": file_sha256(
                            parent / "shared_goal_response_reference.json"
                        ),
                    }
                },
            )
            publish_stage_manifest(parent, SOURCE_MANIFEST)

            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "reaudit-d1f",
                    "--final-dir",
                    str(parent),
                    "--output-dir",
                    str(output),
                ],
                cwd=repo_root,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            summary = json.loads(
                (output / "failure_state_summary.json").read_text(encoding="utf-8")
            )
            identity = json.loads(
                (output / "reaudit_identity.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["determinate_task_count"], 37)
            self.assertEqual(summary["indeterminate_task_count"], 4)
            self.assertEqual(
                summary["overall_coverage"]["GOAL_CONTROL_PRESERVED"][
                    "determinate_task_denominator"
                ],
                37,
            )
            self.assertEqual(
                summary["verdict"]["status"],
                "GOAL_CONTROL_PRESERVED_ON_FAILURE_STATES",
            )
            self.assertEqual(identity["run_lineage_id"], "d1f-parent-lineage")
            self.assertEqual(
                identity["parent_diagnostic_source_sha256"],
                SOURCE_MANIFEST["content_sha256"],
            )
            self.assertTrue((output / "manifest_sha256.txt").is_file())

    def test_capture_d1f_refuses_existing_output_before_inputs(self):
        import json
        import tempfile

        repo_root = Path(__file__).parents[3]
        script = repo_root / "scripts" / "probe_routea_goal_sensitivity.py"
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "existing"
            existing.mkdir()
            result = subprocess.run(
                [sys.executable, str(script), "capture-d1f",
                 "--selection-dir", str(Path(directory) / "missing"),
                 "--config", str(Path(directory) / "missing.yaml"),
                 "--output-dir", str(existing)],
                cwd=repo_root, text=True, capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)["status"], "OUTPUT_DIR_EXISTS")

    def test_finalize_d1f_copies_complete_verified_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection, bank, forward, output = (
                root / name for name in ("selection", "bank", "forward", "final")
            )
            for path in (selection, bank, forward):
                path.mkdir()
            for name in (
                "endpoint_identity_audit.json", "bc_repeat_null.json",
                "shared_goal_response_reference.json", "failure_state_selection.json",
                "trigger_checkpoint_comparability.json",
            ):
                payload = {"schema": name}
                if name == "failure_state_selection.json":
                    payload["run_lineage_id"] = "d1f-test-lineage"
                write_strict_json(selection / name, payload)
            publish_stage_manifest(selection, SOURCE_MANIFEST)
            selection_identity = d1f_selection_identity(selection)

            for name in D1F_SELECTION_BOUND_FILES:
                shutil.copy2(selection / name, bank / name)
            write_numeric_npz(
                bank / "failure_state_bank.npz",
                {"x": np.zeros(1, np.float32)},
                max_total_bytes=4096,
            )
            write_strict_json(
                bank / "failure_state_bank_manifest.json",
                {
                    "entry_count": 1,
                    "selection_manifest_sha256": file_sha256(
                        selection / "failure_state_selection.json"
                    ),
                    "run_lineage_id": "d1f-test-lineage",
                    "selection_identity": selection_identity,
                },
            )
            write_strict_json(bank / "tripwire_verification.json", {"status": "READ_ONLY_VERIFIED"})
            publish_stage_manifest(bank, SOURCE_MANIFEST)

            for name in (
                "checkpoint_comparability.json", "failure_state_progression.json",
                "rng_pairing_verification.json",
                "tripwire_verification.json", "verdict.json",
            ):
                write_strict_json(forward / name, {"schema": name})
            write_strict_json(
                forward / "failure_state_summary.json",
                {
                    "input_identity": {
                        "run_lineage_id": "d1f-test-lineage",
                        "diagnostic_source_sha256": SOURCE_MANIFEST["content_sha256"],
                        "selection_identity": selection_identity,
                        "bank_stage_manifest_sha256": file_sha256(
                            bank / "manifest_sha256.txt"
                        ),
                        "bank_manifest_sha256": file_sha256(
                            bank / "failure_state_bank_manifest.json"
                        ),
                        "bank_npz_sha256": file_sha256(bank / "failure_state_bank.npz"),
                        "selection_sha256": file_sha256(
                            bank / "failure_state_selection.json"
                        ),
                        "shared_reference_sha256": file_sha256(
                            bank / "shared_goal_response_reference.json"
                        ),
                    }
                },
            )
            write_jsonl(forward / "goal_variants.jsonl", [{"row": 1}])
            for name in (
                "failure_state_sensitivity.csv", "failure_state_jacobians.csv",
                "failure_state_task_summary.csv", "failure_state_stage_coverage.csv",
                "failure_state_distance_role_coverage.csv",
            ):
                write_csv_rows(forward / name, [{"row": 1}])
            publish_stage_manifest(forward, SOURCE_MANIFEST)

            finalize_d1f_outputs(
                selection, bank, forward, output, source_manifest=SOURCE_MANIFEST
            )
            self.assertTrue(D1F_MINIMUM_FINAL_OUTPUTS <= {path.name for path in output.iterdir()})

            altered_selection = root / "altered-selection"
            shutil.copytree(selection, altered_selection)
            for generated in (
                "console.log",
                "diagnostic_source_manifest.json",
                "manifest_sha256.txt",
            ):
                (altered_selection / generated).unlink()
            write_strict_json(
                altered_selection / "shared_goal_response_reference.json",
                {"schema": "different-reference"},
            )
            publish_stage_manifest(altered_selection, SOURCE_MANIFEST)
            with self.assertRaises(DiagnosticFailure):
                finalize_d1f_outputs(
                    altered_selection,
                    bank,
                    forward,
                    root / "wrong-final",
                    source_manifest=SOURCE_MANIFEST,
                )


if __name__ == "__main__":
    unittest.main()
