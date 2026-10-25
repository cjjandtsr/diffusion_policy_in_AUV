from __future__ import annotations

import math

import numpy as np
import pytest

from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.contracts import ActionSpec
from scripts.routea_causal_diagnostics.contracts import DiagnosticFailure
from scripts.routea_causal_diagnostics.goals import (
    GOAL_VARIANTS,
    make_goal_variant,
    reconstruct_goal_history,
    verify_goal_only_mutation,
)
from scripts.routea_causal_diagnostics.metrics import (
    action_geometry,
    bearing_jacobian,
    vertical_jacobian,
)


def _rotate_z(vector, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([c * vector[0] - s * vector[1], s * vector[0] + c * vector[1], vector[2]])


def synthetic_history_with_rotating_yaw(goal=np.array([4.0, -2.0, 1.5])):
    state = np.zeros((8, 9), dtype=np.float32)
    for index in range(8):
        position = np.array([0.1 * index, -0.05 * index, 1.0 + 0.02 * index])
        yaw = -0.8 + 0.2 * index
        goal_body = _rotate_z(np.asarray(goal) - position, -yaw)
        state[index, :3] = position
        state[index, 3:6] = goal_body
        state[index, 6] = np.linalg.norm(np.asarray(goal) - position)
        state[index, 7] = np.sin(yaw)
        state[index, 8] = np.cos(yaw)
    return state


def test_goal_roundtrip_and_bearing_variants_use_each_frames_yaw():
    goal = np.array([4.0, -2.0, 1.5])
    state = synthetic_history_with_rotating_yaw(goal=goal)
    rebuilt = reconstruct_goal_history(state)
    assert rebuilt.max_pairwise_inf_m <= 1e-4
    assert np.allclose(rebuilt.goal, goal, atol=1e-5)

    plus = make_goal_variant(state, "L1-bearing+15")
    latest = state[-1, :3].astype(np.float64)
    relative = goal - latest
    expected = latest + _rotate_z(relative, math.pi / 12.0)
    assert np.allclose(reconstruct_goal_history(plus).goal, expected, atol=1e-5)


def test_inconsistent_history_fails_instead_of_averaging():
    state = synthetic_history_with_rotating_yaw()
    state[0, 3] += 1e-2
    with pytest.raises(DiagnosticFailure, match="INCONSISTENT_GOAL_HISTORY"):
        reconstruct_goal_history(state)


def test_goal_variant_changes_only_agent_pos_columns_3_to_6():
    before_state = synthetic_history_with_rotating_yaw()
    before_pc = np.arange(8 * 16 * 3, dtype=np.float32).reshape(8, 16, 3)
    after_state = make_goal_variant(before_state, "L5-z+0.3m")
    report = verify_goal_only_mutation(before_state, after_state, before_pc)
    assert report["position_exact"]
    assert report["yaw_exact"]
    assert report["point_cloud_exact"]
    assert report["non_goal_bytes_exact"]
    assert report["goal_fields_changed"]
    assert after_state.dtype == before_state.dtype


@pytest.mark.parametrize("name", GOAL_VARIANTS)
def test_all_five_goal_variants_roundtrip(name):
    variant = make_goal_variant(synthetic_history_with_rotating_yaw(), name)
    assert reconstruct_goal_history(variant).max_pairwise_inf_m <= 1e-4
    assert np.isfinite(variant).all()


def test_undefined_horizontal_bearing_fails_without_nan():
    state = synthetic_history_with_rotating_yaw()
    latest = state[-1, :3].astype(np.float64)
    state = synthetic_history_with_rotating_yaw(goal=np.array([latest[0], latest[1], 2.0]))
    with pytest.raises(DiagnosticFailure, match="UNDEFINED_HORIZONTAL_BEARING"):
        make_goal_variant(state, "L1-bearing+15")


def test_unknown_variant_is_rejected():
    with pytest.raises(DiagnosticFailure, match="UNKNOWN_GOAL_VARIANT"):
        make_goal_variant(synthetic_history_with_rotating_yaw(), "L9-invalid")


def test_point_cloud_mutation_is_rejected_with_scientific_exit_code():
    state = synthetic_history_with_rotating_yaw()
    point_cloud = np.zeros((8, 16, 3), dtype=np.float32)
    changed = point_cloud.copy()
    changed[0, 0, 0] = 1.0
    with pytest.raises(DiagnosticFailure, match="INVALID_OBSERVATION_MUTATION") as caught:
        verify_goal_only_mutation(state, state.copy(), point_cloud, changed)
    assert caught.value.exit_code == 12


def test_h4_metrics_use_full_action_2_to_6_only():
    full = np.zeros((1, 16, 11), np.float32)
    full[0, 0:2, :3] = 999.0
    full[0, 2:6, 0] = 1.0
    full[0, 6:10, 1] = 777.0
    metrics = action_geometry(
        full, np.zeros(3), np.array([10.0, 0.0, 0.0]), ActionSpec.e5()
    )
    assert metrics["action_h4"].shape == (4, 11)
    assert metrics["endpoint"].tolist() == [4.0, 0.0, 0.0]
    assert metrics["first_step_horizontal_compliance"] == pytest.approx(1.0)
    assert metrics["h4_3d_compliance"] == pytest.approx(1.0)


def test_h4_zero_horizontal_step_reports_null_compliance():
    full = np.zeros((16, 11), np.float32)
    full[2:6, 2] = 0.1
    metrics = action_geometry(
        full, np.zeros(3), np.array([1.0, 0.0, 1.0]), ActionSpec.e5()
    )
    assert metrics["first_step_horizontal_compliance"] is None
    assert metrics["endpoint"].tolist() == pytest.approx([0.0, 0.0, 0.4])


def test_central_jacobians_use_two_sided_denominator():
    plus = {"goal_parallel_m": 3.0, "endpoint": np.array([2.0, 1.0, 0.5])}
    minus = {"goal_parallel_m": 1.0, "endpoint": np.array([0.0, -1.0, -0.5])}
    bearing = bearing_jacobian(plus, minus, radians=0.5)
    vertical = vertical_jacobian(plus, minus, delta_m=0.25)
    assert bearing["goal_parallel_m"] == pytest.approx(2.0)
    assert bearing["endpoint"].tolist() == pytest.approx([2.0, 2.0, 1.0])
    assert vertical["goal_parallel_m"] == pytest.approx(4.0)
