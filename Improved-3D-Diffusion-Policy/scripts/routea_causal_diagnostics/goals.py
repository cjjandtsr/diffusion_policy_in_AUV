"""Eight-frame odom/body goal reconstruction and frozen sentinel variants."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .contracts import DiagnosticFailure, canonical_array_sha256


GOAL_VARIANTS = (
    "L0-original",
    "L1-bearing+15",
    "L2-bearing-15",
    "L5-z+0.3m",
    "L6-z-0.3m",
)


@dataclass(frozen=True)
class GoalHistory:
    goal: np.ndarray
    per_frame_goals: np.ndarray
    yaw: np.ndarray
    max_pairwise_inf_m: float


def _validate_agent_pos(agent_pos: Any) -> np.ndarray:
    array = np.asarray(agent_pos)
    if array.shape != (8, 9):
        raise DiagnosticFailure(
            "INVALID_OBSERVATION_SHAPE", f"agent_pos expected (8,9), got {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
        raise DiagnosticFailure(
            "INVALID_OBSERVATION_VALUE", f"agent_pos dtype={array.dtype} must be finite float"
        )
    return array


def _rotate_z(vectors: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    yaw = np.asarray(yaw, dtype=np.float64)
    c, s = np.cos(yaw), np.sin(yaw)
    result = vectors.copy()
    result[..., 0] = c * vectors[..., 0] - s * vectors[..., 1]
    result[..., 1] = s * vectors[..., 0] + c * vectors[..., 1]
    return result


def reconstruct_goal_history(
    agent_pos: Any, tolerance_m: float = 1e-4
) -> GoalHistory:
    state = _validate_agent_pos(agent_pos)
    state64 = state.astype(np.float64, copy=False)
    yaw_norm = np.hypot(state64[:, 7], state64[:, 8])
    if np.any(yaw_norm < 1e-8):
        raise DiagnosticFailure(
            "INVALID_YAW_ENCODING", "sin/cos yaw norm is below 1e-8"
        )
    yaw = np.arctan2(state64[:, 7], state64[:, 8])
    goals = state64[:, :3] + _rotate_z(state64[:, 3:6], yaw)
    reference = goals[-1]
    max_pairwise = float(np.max(np.abs(goals - reference[None, :])))
    if max_pairwise > float(tolerance_m):
        raise DiagnosticFailure(
            "INCONSISTENT_GOAL_HISTORY",
            f"max_pairwise_inf_m={max_pairwise:.9g} tolerance_m={tolerance_m}",
        )
    encoded_distance = state64[:, 6]
    reconstructed_distance = np.linalg.norm(reference[None, :] - state64[:, :3], axis=1)
    max_distance_error = float(np.max(np.abs(encoded_distance - reconstructed_distance)))
    if max_distance_error > float(tolerance_m):
        raise DiagnosticFailure(
            "INCONSISTENT_GOAL_HISTORY",
            f"distance_field_max_abs_m={max_distance_error:.9g} tolerance_m={tolerance_m}",
        )
    return GoalHistory(
        goal=reference.copy(),
        per_frame_goals=goals,
        yaw=yaw,
        max_pairwise_inf_m=max_pairwise,
    )


def _variant_goal(history: GoalHistory, state: np.ndarray, name: str) -> np.ndarray:
    goal = history.goal.copy()
    latest_position = state[-1, :3].astype(np.float64, copy=False)
    if name == "L0-original":
        return goal
    if name in {"L1-bearing+15", "L2-bearing-15"}:
        relative = goal - latest_position
        # The state is float32; an exactly vertical target can reconstruct with
        # a one-ULP horizontal residual.  Use the same 1e-4 m contract as the
        # eight-frame goal round-trip rather than a sub-ULP threshold.
        if float(np.linalg.norm(relative[:2])) <= 1e-4:
            raise DiagnosticFailure(
                "UNDEFINED_HORIZONTAL_BEARING",
                "latest position and goal have identical horizontal coordinates",
            )
        angle = math.pi / 12.0 if name == "L1-bearing+15" else -math.pi / 12.0
        return latest_position + _rotate_z(relative[None, :], np.array([angle]))[0]
    if name == "L5-z+0.3m":
        goal[2] += 0.3
        return goal
    if name == "L6-z-0.3m":
        goal[2] -= 0.3
        return goal
    raise DiagnosticFailure("UNKNOWN_GOAL_VARIANT", name)


def make_goal_variant(agent_pos: Any, name: str) -> np.ndarray:
    if name not in GOAL_VARIANTS:
        raise DiagnosticFailure("UNKNOWN_GOAL_VARIANT", str(name))
    state = _validate_agent_pos(agent_pos)
    history = reconstruct_goal_history(state)
    if name == "L0-original":
        return state.copy()
    target = _variant_goal(history, state, name)
    yaw = history.yaw
    target_delta_odom = target[None, :] - state[:, :3].astype(np.float64, copy=False)
    goal_body = _rotate_z(target_delta_odom, -yaw)
    result = state.copy()
    result[:, 3:6] = goal_body.astype(result.dtype, copy=False)
    result[:, 6] = np.linalg.norm(target_delta_odom, axis=1).astype(
        result.dtype, copy=False
    )
    rebuilt = reconstruct_goal_history(result)
    error = float(np.max(np.abs(rebuilt.goal - target)))
    if error > 1e-4:
        raise DiagnosticFailure(
            "GOAL_VARIANT_ROUNDTRIP_MISMATCH", f"variant={name} max_abs_m={error}"
        )
    return result


def verify_goal_only_mutation(
    original: Any,
    variant: Any,
    point_cloud: Any,
    variant_point_cloud: Any | None = None,
) -> dict[str, Any]:
    before = _validate_agent_pos(original)
    after = _validate_agent_pos(variant)
    before_pc = np.asarray(point_cloud)
    after_pc = before_pc if variant_point_cloud is None else np.asarray(variant_point_cloud)
    if before_pc.shape != after_pc.shape:
        raise DiagnosticFailure(
            "INVALID_OBSERVATION_MUTATION",
            f"point_cloud shape changed {before_pc.shape}->{after_pc.shape}",
            exit_code=12,
        )
    position_exact = before[:, :3].tobytes(order="C") == after[:, :3].tobytes(order="C")
    yaw_exact = before[:, 7:9].tobytes(order="C") == after[:, 7:9].tobytes(order="C")
    point_cloud_exact = before_pc.tobytes(order="C") == after_pc.tobytes(order="C")
    non_goal_before = np.ascontiguousarray(np.concatenate([before[:, :3], before[:, 7:9]], axis=1))
    non_goal_after = np.ascontiguousarray(np.concatenate([after[:, :3], after[:, 7:9]], axis=1))
    non_goal_bytes_exact = non_goal_before.tobytes() == non_goal_after.tobytes()
    goal_fields_changed = before[:, 3:7].tobytes(order="C") != after[:, 3:7].tobytes(order="C")
    report = {
        "position_exact": position_exact,
        "yaw_exact": yaw_exact,
        "point_cloud_exact": point_cloud_exact,
        "non_goal_bytes_exact": non_goal_bytes_exact,
        "goal_fields_changed": goal_fields_changed,
        "original_agent_pos_sha256": canonical_array_sha256(before),
        "variant_agent_pos_sha256": canonical_array_sha256(after),
        "point_cloud_sha256": canonical_array_sha256(before_pc),
    }
    if not all((position_exact, yaw_exact, point_cloud_exact, non_goal_bytes_exact)):
        raise DiagnosticFailure(
            "INVALID_OBSERVATION_MUTATION", str(report), exit_code=12
        )
    return report
