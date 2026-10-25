"""Executed-H4 action geometry and local finite-difference sensitivity metrics."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import torch

from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.contracts import ActionSpec

from .contracts import DiagnosticFailure


def _numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _cosine(left: np.ndarray, right: np.ndarray, *, tolerance: float = 1e-12) -> float | None:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= tolerance:
        return None
    return float(np.dot(left, right) / denominator)


def action_geometry(
    full_action: Any,
    latest_position: Any,
    target_goal: Any,
    action_spec: ActionSpec,
) -> dict[str, Any]:
    full = _numpy(full_action)
    try:
        _, _, executed = action_spec.split(full)
    except ValueError as exc:
        raise DiagnosticFailure("INVALID_ACTION_SHAPE", str(exc)) from exc
    executed = _numpy(executed)
    if executed.ndim == 3:
        if executed.shape[0] != 1:
            raise DiagnosticFailure(
                "INVALID_ACTION_SHAPE", f"diagnostic requires batch=1, got {executed.shape}"
            )
        executed = executed[0]
    expected = (action_spec.rollout_exec_horizon, action_spec.action_dim)
    if executed.shape != expected or not np.isfinite(executed).all():
        raise DiagnosticFailure(
            "INVALID_ACTION_SHAPE", f"executed expected={expected} actual={executed.shape}"
        )
    latest = np.asarray(latest_position, dtype=np.float64)
    goal = np.asarray(target_goal, dtype=np.float64)
    if latest.shape != (3,) or goal.shape != (3,) or not np.isfinite(latest).all() or not np.isfinite(goal).all():
        raise DiagnosticFailure(
            "INVALID_ACTION_GEOMETRY_INPUT", "latest_position and target_goal must be finite xyz"
        )
    action_h4 = np.ascontiguousarray(executed)
    deltas = action_h4[:, :3].astype(np.float64, copy=False)
    waypoints = latest[None, :] + np.cumsum(deltas, axis=0)
    endpoint = waypoints[-1]
    displacement = endpoint - latest
    target = goal - latest
    target_horizontal = target[:2]
    horizontal_norm = float(np.linalg.norm(target_horizontal))
    if horizontal_norm > 1e-12:
        axis = target_horizontal / horizontal_norm
        goal_parallel = float(np.dot(displacement[:2], axis))
        goal_perpendicular = float(
            np.dot(displacement[:2], np.array([-axis[1], axis[0]]))
        )
    else:
        goal_parallel = None
        goal_perpendicular = None
    first_horizontal = deltas[0, :2]
    first_compliance = _cosine(first_horizontal, target_horizontal)
    h4_compliance = _cosine(displacement, target)
    yaw_delta = action_h4[:, 9].astype(np.float64, copy=False)
    second_difference = np.diff(waypoints, n=2, axis=0)
    path_shape_norm = float(np.linalg.norm(second_difference))
    return {
        "action_h4": action_h4.copy(),
        "waypoints_h4": waypoints,
        "endpoint": endpoint,
        "goal_parallel_m": goal_parallel,
        "goal_perpendicular_m": goal_perpendicular,
        "vertical_m": float(displacement[2]),
        "first_step_horizontal_compliance": first_compliance,
        "h4_3d_compliance": h4_compliance,
        "yaw_sum_rad": float(yaw_delta.sum()),
        "yaw_total_variation_rad": float(np.abs(yaw_delta).sum()),
        "path_shape_norm_m": path_shape_norm,
        "traj_dt_s": float(action_spec.traj_dt),
        "action_start": int(action_spec.action_start),
        "rollout_exec_horizon": int(action_spec.rollout_exec_horizon),
    }


def _central_jacobian(
    plus_metrics: Mapping[str, Any], minus_metrics: Mapping[str, Any], denominator: float
) -> dict[str, Any]:
    if not math.isfinite(denominator) or denominator <= 0:
        raise DiagnosticFailure("INVALID_JACOBIAN_DELTA", f"denominator={denominator}")
    result: dict[str, Any] = {}
    for key in sorted(set(plus_metrics).intersection(minus_metrics)):
        plus = plus_metrics[key]
        minus = minus_metrics[key]
        if plus is None or minus is None or isinstance(plus, (bool, str)) or isinstance(minus, (bool, str)):
            continue
        try:
            plus_array = np.asarray(plus, dtype=np.float64)
            minus_array = np.asarray(minus, dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if plus_array.shape != minus_array.shape or not np.isfinite(plus_array).all() or not np.isfinite(minus_array).all():
            continue
        derivative = (plus_array - minus_array) / denominator
        result[key] = float(derivative) if derivative.ndim == 0 else derivative
    return result


def bearing_jacobian(
    plus_metrics: Mapping[str, Any],
    minus_metrics: Mapping[str, Any],
    radians: float = math.pi / 12.0,
) -> dict[str, Any]:
    return _central_jacobian(plus_metrics, minus_metrics, 2.0 * float(radians))


def vertical_jacobian(
    plus_metrics: Mapping[str, Any],
    minus_metrics: Mapping[str, Any],
    delta_m: float = 0.3,
) -> dict[str, Any]:
    return _central_jacobian(plus_metrics, minus_metrics, 2.0 * float(delta_m))

