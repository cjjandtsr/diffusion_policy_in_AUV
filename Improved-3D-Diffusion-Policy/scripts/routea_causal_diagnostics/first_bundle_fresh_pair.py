"""Pure analysis contracts for the fresh first-bundle 2x2 diagnostic.

The unit of classification is one preregistered task. No checkpoint, world,
or parent trajectory is read here.
"""

from __future__ import annotations

from collections import Counter
import math
from typing import Any, Mapping

import numpy as np

from scripts.routea_causal_diagnostics.action_feedback import decompose_action_pair


ARMS = ("bb", "br", "rb", "rr")
INFORMATIVE = (
    "FIRST_BUNDLE_DOMINANT",
    "DOWNSTREAM_POLICY_DOMINANT",
    "EITHER_SUCCESS_COMPONENT_SUFFICIENT",
    "BOTH_SUCCESS_COMPONENTS_REQUIRED",
)
DIMENSION_NAMES = (
    "dpos_world_x", "dpos_world_y", "dpos_world_z",
    "dvel_world_x", "dvel_world_y", "dvel_world_z",
    "acc_world_x", "acc_world_y", "acc_world_z",
    "dyaw_cw", "yaw_rate",
)


def classify_task(outcomes: Mapping[str, Any], *, pair_valid: bool = True) -> str:
    """Return exactly one task label, with pair validity taking precedence."""

    if not pair_valid:
        return "INVALID_PAIR"
    if set(outcomes) != set(ARMS):
        raise ValueError("four named conditions are required")
    if any(outcomes[arm] is None for arm in ARMS):
        return "AMBIGUOUS"
    if any(type(outcomes[arm]) is not bool for arm in ARMS):
        raise ValueError("determinate endpoints must be bool")
    b, br, rb, r = (outcomes[arm] for arm in ARMS)
    if b == r:
        return "UNINFORMATIVE_BINARY_CONTROL"
    if br == b and rb == r:
        return "FIRST_BUNDLE_DOMINANT"
    if br == r and rb == b:
        return "DOWNSTREAM_POLICY_DOMINANT"
    if br and rb:
        return "EITHER_SUCCESS_COMPONENT_SUFFICIENT"
    return "BOTH_SUCCESS_COMPONENTS_REQUIRED"


def count_classes(labels: list[str]) -> dict[str, int]:
    counts = Counter(labels)
    allowed = set(INFORMATIVE) | {
        "UNINFORMATIVE_BINARY_CONTROL", "AMBIGUOUS", "INVALID_PAIR"
    }
    if set(counts) - allowed:
        raise ValueError("unknown task label")
    informative = sum(counts[name] for name in INFORMATIVE)
    determinate = informative + counts["UNINFORMATIVE_BINARY_CONTROL"]
    pair_valid = determinate + counts["AMBIGUOUS"]
    return {
        "total": len(labels),
        "invalid": counts["INVALID_PAIR"],
        "pair_valid": pair_valid,
        "endpoint_determinate": determinate,
        "informative_task_count": informative,
        "uninformative": counts["UNINFORMATIVE_BINARY_CONTROL"],
        "ambiguous": counts["AMBIGUOUS"],
        **{name: counts[name] for name in INFORMATIVE},
    }


def factorial_effects(values: Mapping[str, float]) -> dict[str, float]:
    if set(values) != set(ARMS):
        raise ValueError("four named values required")
    q = {arm: float(values[arm]) for arm in ARMS}
    if not all(np.isfinite(value) for value in q.values()):
        raise ValueError("factorial values must be finite")
    return {
        "h4": ((q["bb"] + q["br"]) - (q["rb"] + q["rr"])) / 2.,
        "downstream": ((q["bb"] + q["rb"]) - (q["br"] + q["rr"])) / 2.,
        "interaction": q["bb"] - q["br"] - q["rb"] + q["rr"],
    }


def decompose_first_actions(
    bc_action: Any, rl_action: Any, position_before: Any, goal: Any
) -> dict[str, Any]:
    bc = np.asarray(bc_action)
    rl = np.asarray(rl_action)
    if bc.shape != (8, 11) or rl.shape != (8, 11):
        raise ValueError("both first bundles must have shape (8,11)")
    if not np.isfinite(bc).all() or not np.isfinite(rl).all():
        raise ValueError("first bundles must be finite")
    existing = decompose_action_pair(
        bc, rl, position_before=position_before, goal=goal
    )
    delta = rl.astype(np.float64) - bc.astype(np.float64)
    return {
        "delta_8x11": delta.tolist(),
        "waypoint_11d_l2": np.linalg.norm(delta, axis=1).tolist(),
        "executed_4x11_l2": float(np.linalg.norm(delta[:4])),
        "tail_4x11_l2": float(np.linalg.norm(delta[4:])),
        "executed_by_dimension": {
            name: delta[:4, index].tolist()
            for index, name in enumerate(DIMENSION_NAMES)
        },
        "executed_dimension_mean": {
            name: float(np.mean(delta[:4, index]))
            for index, name in enumerate(DIMENSION_NAMES)
        },
        "executed_dimension_median": {
            name: float(np.median(delta[:4, index]))
            for index, name in enumerate(DIMENSION_NAMES)
        },
        "tail_by_dimension": {
            name: delta[4:, index].tolist()
            for index, name in enumerate(DIMENSION_NAMES)
        },
        "existing_geometry": existing,
    }


def horizontal_vertical_metrics(
    positions: Any, goal: Any, initial_yaw: float
) -> dict[str, Any]:
    p = np.asarray(positions, dtype=np.float64)
    g = np.asarray(goal, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 1 or g.shape != (3,):
        raise ValueError("positions (N,3) and goal (3,) required")
    if not np.isfinite(p).all() or not np.isfinite(g).all() or not np.isfinite(initial_yaw):
        raise ValueError("geometry must be finite")
    toward = g[:2] - p[0, :2]
    norm = float(np.linalg.norm(toward))
    fallback = norm <= 1e-12
    axis = np.array([np.cos(initial_yaw), np.sin(initial_yaw)]) if fallback else toward / norm
    lateral = np.array([-axis[1], axis[0]])
    displacement = p - p[0]
    parallel = displacement[:, :2] @ axis
    sideways = displacement[:, :2] @ lateral
    vertical = displacement[:, 2]
    return {
        "horizontal_axis_yaw_fallback": bool(fallback),
        "horizontal_goal_parallel_end_m": float(parallel[-1]),
        "horizontal_lateral_end_m": float(sideways[-1]),
        "vertical_end_m": float(vertical[-1]),
        "horizontal_goal_parallel_path_m": parallel.tolist(),
        "horizontal_lateral_path_m": sideways.tolist(),
        "vertical_path_m": vertical.tolist(),
    }


def _first_sphere_entry(path: np.ndarray, goal: np.ndarray) -> tuple[int, float] | None:
    """First closed 0.5 m sphere intersection along four executed segments."""

    for segment in range(4):
        start = path[segment] - goal
        direction = path[segment + 1] - path[segment]
        a = float(direction @ direction)
        c = float(start @ start) - 0.25
        if c <= 0.0:
            return segment, 0.0
        if a == 0.0:
            continue
        b = 2.0 * float(start @ direction)
        discriminant = b * b - 4.0 * a * c
        if discriminant < 0.0:
            continue
        root = math.sqrt(max(0.0, discriminant))
        for fraction in sorted(((-b - root) / (2.0 * a), (-b + root) / (2.0 * a))):
            if 0.0 <= fraction <= 1.0:
                return segment, float(fraction)
    return None


def safe_arrival(rows: list[Mapping[str, Any]], goal: Any) -> dict[str, Any]:
    center = np.asarray(goal, dtype=np.float64)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("finite goal (3,) required")
    for decision, row in enumerate(rows):
        raw = row.get("executed_path")
        if raw is None:
            return {"success": None, "entry": None, "reason": "missing_executed_path"}
        path = np.asarray(raw, dtype=np.float64)
        if path.shape != (5, 3) or not np.isfinite(path).all():
            return {"success": None, "entry": None, "reason": "invalid_executed_path"}
        entry = _first_sphere_entry(path, center)
        failed = bool(row.get("physical_failure", False))
        if entry is not None:
            if failed:
                return {"success": None, "entry": None, "reason": "AMBIGUOUS_SAME_H4"}
            return {
                "success": True,
                "entry": {"decision": decision, "segment": entry[0], "fraction": entry[1]},
                "reason": "safe_sphere_entry",
            }
        if failed:
            return {"success": False, "entry": None, "reason": row.get("done_reason")}
    return {"success": False, "entry": None, "reason": "no_safe_entry"}


def trajectory_metrics(rows: list[Mapping[str, Any]], goal: Any, initial_yaw: float) -> dict[str, Any]:
    if not rows:
        raise ValueError("nonempty trajectory required")
    paths = [np.asarray(row["executed_path"], dtype=np.float64) for row in rows]
    if any(path.shape != (5, 3) or not np.isfinite(path).all() for path in paths):
        raise ValueError("finite executed paths (5,3) required")
    points = np.concatenate([paths[0][:1]] + [path[1:] for path in paths])
    center = np.asarray(goal, dtype=np.float64)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("finite goal (3,) required")
    initial_distance = float(np.linalg.norm(center - points[0]))
    if initial_distance <= 0.0:
        raise ValueError("initial goal direction undefined")
    axis = (center - points[0]) / initial_distance
    distances = np.linalg.norm(points - center, axis=1)
    progress = (points - points[0]) @ axis
    cross = (points - points[0]) - progress[:, None] * axis[None, :]
    segment_min = []
    for start, end in zip(points[:-1], points[1:]):
        v = end - start
        a = float(v @ v)
        fraction = 0.0 if a == 0.0 else float(np.clip(((center - start) @ v) / a, 0., 1.))
        segment_min.append(float(np.linalg.norm(start + fraction * v - center)))
    after_h4 = distances[4:]
    regression = float(np.max(after_h4 - np.minimum.accumulate(after_h4)))
    horizontal = horizontal_vertical_metrics(points, center, initial_yaw)
    clearances = [float(row["clearance_m"]) for row in rows if row.get("clearance_m") is not None]
    yaws = [float(initial_yaw)]
    for row in rows:
        if row.get("executed_yaws") is not None:
            yaws.extend(float(value) for value in row["executed_yaws"])
    yaw_accumulation = None
    if len(yaws) == len(points):
        differences = np.diff(yaws)
        yaw_accumulation = float(np.sum(np.abs((differences + np.pi) % (2. * np.pi) - np.pi)))
    return {
        "final_goal_distance_m": float(distances[-1]),
        "min_goal_distance_m": min(segment_min) if segment_min else initial_distance,
        "net_goal_progress_m": initial_distance - float(distances[-1]),
        "first_h4_goal_projection_m": float(progress[4]),
        "first_h4_min_goal_distance_m": min(segment_min[:4]),
        "post_h4_best_so_far_regression_m": regression,
        "max_cross_track_3d_m": float(np.max(np.linalg.norm(cross, axis=1))),
        "backtrack_m": float(np.sum(np.maximum(0., -np.diff(progress)))),
        "yaw_accumulation_rad": yaw_accumulation,
        "min_clearance_m": min(clearances) if clearances else None,
        **horizontal,
    }


def summarize_action_deltas(tasks: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Task-unit 8x11 signed summaries by fresh control outcome."""

    strata = ("fresh_break", "fresh_rescue", "uninformative", "ambiguous", "invalid")
    output: dict[str, Any] = {"by_stratum": {}}
    for stratum in strata:
        group = [task for task in tasks
                 if task.get("fresh_control_label") == stratum
                 and "action_decomposition" in task]
        arrays = [np.asarray(task["action_decomposition"]["delta_8x11"], dtype=np.float64)
                  for task in group]
        if any(array.shape != (8, 11) or not np.isfinite(array).all() for array in arrays):
            raise ValueError("action decomposition must contain finite 8x11 delta")
        cells = []
        for waypoint in range(8):
            row = []
            for dim, name in enumerate(DIMENSION_NAMES):
                values = [float(array[waypoint, dim]) for array in arrays]
                row.append({
                    "waypoint": waypoint + 1, "dimension": name,
                    "denominator": len(values),
                    "positive": sum(value > 0. for value in values),
                    "negative": sum(value < 0. for value in values),
                    "zero": sum(value == 0. for value in values),
                    "median": None if not values else float(np.median(values)),
                    "task_values": [
                        {"task_id": task["task_id"], "delta": value}
                        for task, value in zip(group, values)
                    ],
                })
            cells.append(row)
        output["by_stratum"][stratum] = {
            "task_count": len(group), "waypoint_dimension": cells,
        }
    candidates = []
    break_cells = output["by_stratum"]["fresh_break"]["waypoint_dimension"]
    rescue_cells = output["by_stratum"]["fresh_rescue"]["waypoint_dimension"]
    for waypoint in range(4):
        for dim in range(11):
            b, r = break_cells[waypoint][dim], rescue_cells[waypoint][dim]
            for sign in ("positive", "negative"):
                if b[sign] >= 2 and r[sign] < 2:
                    candidates.append({"waypoint": waypoint + 1, "dimension": DIMENSION_NAMES[dim],
                                       "sign": sign, "fresh_break_count": b[sign],
                                       "fresh_rescue_same_sign_count": r[sign]})
    output["repeated_delta_candidates"] = candidates
    output["earliest_repeated_executed_waypoint"] = min(
        (item["waypoint"] for item in candidates), default=None
    )
    output["candidate_is_descriptive_not_significance"] = True
    return output


def summarize_factorial_effects(tasks: list[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[tuple[str, float]]]] = {}
    for task in tasks:
        for metric, terms in task.get("factorial_effects", {}).items():
            for term in ("h4", "downstream", "interaction"):
                value = float(terms[term])
                if not np.isfinite(value):
                    raise ValueError("factorial effect must be finite")
                grouped.setdefault(metric, {}).setdefault(term, []).append((task["task_id"], value))
    result = {}
    for metric, terms in sorted(grouped.items()):
        result[metric] = {}
        for term, pairs in sorted(terms.items()):
            values = [value for _, value in pairs]
            result[metric][term] = {
                "denominator": len(values), "median": float(np.median(values)),
                "positive": sum(value > 0. for value in values),
                "negative": sum(value < 0. for value in values),
                "zero": sum(value == 0. for value in values),
                "task_values": [{"task_id": task_id, "effect": value} for task_id, value in pairs],
            }
    return result
