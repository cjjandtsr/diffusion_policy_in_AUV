"""Pure contracts for the small action-chunk and closed-loop feedback audit.

This module is deliberately read-only and model-free.  It consumes the frozen
D0.5-B paired transition rows plus the existing B1 drift table.  It does not
load a checkpoint, run an actor, change a reward, or authorize training.
"""

from __future__ import annotations

from collections import defaultdict
import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


ACTION_GROUPS = {
    "dpos": (0, 3),
    "dvel": (3, 6),
    "acc": (6, 9),
    "dyaw": (9, 10),
    "yaw_rate": (10, 11),
}
PRIMARY_FEEDBACK_START = 0
PRIMARY_FEEDBACK_END = 5


def _finite_array(value: Any, shape: Tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value)
    if tuple(array.shape) != tuple(shape):
        raise ValueError("{} must have shape {}; got {}".format(label, shape, array.shape))
    if not np.issubdtype(array.dtype, np.number) or not bool(np.isfinite(array).all()):
        raise ValueError("{} must be a finite numeric array".format(label))
    return np.ascontiguousarray(array, dtype=np.float64)


def _finite_xyz(value: Any, label: str) -> np.ndarray:
    return _finite_array(value, (3,), label)


def _l2(value: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(value, dtype=np.float64)))


def _median(values: Iterable[Any]) -> float:
    numeric = np.asarray([float(value) for value in values], dtype=np.float64)
    if numeric.size == 0 or not bool(np.isfinite(numeric).all()):
        raise ValueError("median requires finite non-empty values")
    return float(np.median(numeric))


def _quantile(values: Sequence[float], probability: float) -> float:
    numeric = np.asarray(values, dtype=np.float64)
    if numeric.size == 0 or not bool(np.isfinite(numeric).all()):
        raise ValueError("quantile requires finite non-empty values")
    return float(np.quantile(numeric, float(probability)))


def decompose_action_pair(
    u0_action: Any,
    rl_action: Any,
    *,
    position_before: Any,
    goal: Any
) -> Dict[str, Any]:
    """Decompose one paired H8 action without mixing executed H4 and tail H4."""

    u0 = _finite_array(u0_action, (8, 11), "u0_action")
    rl = _finite_array(rl_action, (8, 11), "rl_action")
    position = _finite_xyz(position_before, "position_before")
    target_goal = _finite_xyz(goal, "goal")
    delta = rl - u0
    executed = delta[:4]
    tail = delta[4:]

    target_xy = target_goal[:2] - position[:2]
    target_xy_norm = _l2(target_xy)
    executed_endpoint_delta = np.sum(executed[:, :3], axis=0)
    if target_xy_norm > 1e-12:
        axis = target_xy / target_xy_norm
        lateral_axis = np.asarray([-axis[1], axis[0]], dtype=np.float64)
        goal_parallel_delta = float(np.dot(executed_endpoint_delta[:2], axis))
        goal_perpendicular_delta = float(
            np.dot(executed_endpoint_delta[:2], lateral_axis)
        )
    else:
        goal_parallel_delta = 0.0
        goal_perpendicular_delta = 0.0

    executed_cumulative = np.cumsum(executed[:, :3], axis=0)
    tail_cumulative = np.cumsum(tail[:, :3], axis=0)
    all_cumulative = np.cumsum(delta[:, :3], axis=0)
    result = {
        "full_action_l2": _l2(delta),
        "executed_action_l2": _l2(executed),
        "tail_action_l2": _l2(tail),
        "executed_dpos_l2": _l2(executed[:, :3]),
        "tail_dpos_l2": _l2(tail[:, :3]),
        "executed_position_path_l2_m": _l2(executed_cumulative),
        "tail_position_path_l2_m": _l2(tail_cumulative),
        "executed_endpoint_delta_m": _l2(executed_endpoint_delta),
        "goal_parallel_delta_m": goal_parallel_delta,
        "goal_perpendicular_delta_m": goal_perpendicular_delta,
        "vertical_delta_m": float(executed_endpoint_delta[2]),
        "yaw_sum_delta_rad": float(np.sum(executed[:, 9])),
        "yaw_total_variation_delta_l1_rad": float(np.sum(np.abs(executed[:, 9]))),
    }
    for group, bounds in ACTION_GROUPS.items():
        start, end = bounds
        result["executed_{}_l2".format(group)] = _l2(executed[:, start:end])
        result["tail_{}_l2".format(group)] = _l2(tail[:, start:end])
        result["full_{}_l2".format(group)] = _l2(delta[:, start:end])

    waypoints = []
    for index in range(8):
        cumulative = all_cumulative[index]
        waypoints.append(
            {
                "waypoint_index": index,
                "executed": bool(index < 4),
                "delta_dpos_x": float(delta[index, 0]),
                "delta_dpos_y": float(delta[index, 1]),
                "delta_dpos_z": float(delta[index, 2]),
                "delta_dpos_l2": _l2(delta[index, :3]),
                "cumulative_delta_x": float(cumulative[0]),
                "cumulative_delta_y": float(cumulative[1]),
                "cumulative_delta_z": float(cumulative[2]),
                "cumulative_delta_l2": _l2(cumulative),
                "delta_dvel_x": float(delta[index, 3]),
                "delta_dvel_y": float(delta[index, 4]),
                "delta_dvel_z": float(delta[index, 5]),
                "delta_dvel_l2": _l2(delta[index, 3:6]),
                "delta_acc_x": float(delta[index, 6]),
                "delta_acc_y": float(delta[index, 7]),
                "delta_acc_z": float(delta[index, 8]),
                "delta_acc_l2": _l2(delta[index, 6:9]),
                "delta_dyaw": float(delta[index, 9]),
                "delta_yaw_rate": float(delta[index, 10]),
            }
        )
    result["waypoints"] = waypoints
    return result


def _null_threshold(
    nulls: Mapping[str, Any],
    task_id: str,
    decision: int,
    metric: str,
    required: bool = True,
) -> Any:
    try:
        value = nulls[task_id][str(int(decision))][metric]["threshold"]
    except (KeyError, TypeError) as exc:
        if not required:
            return None
        raise ValueError(
            "missing BC null threshold task={} decision={} metric={}".format(
                task_id, decision, metric
            )
        ) from exc
    threshold = float(value)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("BC null threshold must be finite and non-negative")
    return threshold


def _validate_transition_pair(
    left: Mapping[str, Any], right: Mapping[str, Any], identity: Tuple[str, int], decision: int
) -> None:
    for row, lane in ((left, "u0"), (right, "seed3-u32")):
        if str(row.get("task_id")) != identity[0] or int(row.get("repeat")) != identity[1]:
            raise ValueError("paired transition identity mismatch")
        if str(row.get("lane")) != lane or int(row.get("decision_index")) != decision:
            raise ValueError("paired transition lane/decision mismatch")
        policy = _finite_array(row.get("policy_action_8x11"), (8, 11), lane + " policy")
        executed = _finite_array(
            row.get("executed_action_4x11"), (4, 11), lane + " executed"
        )
        if not np.array_equal(policy[:4], executed):
            raise ValueError("executed_action_4x11 is not the exact policy H4 prefix")
    if int(left.get("action_seed")) != int(right.get("action_seed")):
        raise ValueError(
            "paired action_seed mismatch {} r{} d{}".format(
                identity[0], identity[1], decision
            )
        )


def _task_aggregate(repeat_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped = defaultdict(list)
    for row in repeat_rows:
        grouped[str(row["task_id"])].append(row)
    result = []
    for task_id in sorted(grouped):
        rows = grouped[task_id]
        h4_excess = _median(row["executed_h4_normalized_minus_null"] for row in rows)
        tail_minus_executed = _median(
            row["tail_action_l2"] - row["executed_action_l2"] for row in rows
        )
        feedback_rows = [
            row for row in rows if row["feedback_position_gap_growth_m"] is not None
        ]
        feedback_growth = (
            _median(row["feedback_position_gap_growth_m"] for row in feedback_rows)
            if feedback_rows
            else None
        )
        endpoint_values = [
            row["feedback_endpoint_normalized_minus_null"]
            for row in feedback_rows
            if row["feedback_endpoint_normalized_minus_null"] is not None
        ]
        endpoint_excess = _median(endpoint_values) if endpoint_values else None
        if not feedback_rows:
            feedback_status = "INDETERMINATE_MISSING_FIXED_ENDPOINT"
        elif endpoint_excess is None:
            feedback_status = "INDETERMINATE_MISSING_ENDPOINT_NULL"
        elif h4_excess <= 0.0:
            feedback_status = "NO_SUPPORT_INITIAL_H4_WITHIN_BC_NULL"
        elif feedback_growth is None or feedback_growth <= 0.0:
            feedback_status = "NO_SUPPORT_NO_POSITIVE_GAP_GROWTH"
        elif endpoint_excess <= 0.0:
            feedback_status = "NO_SUPPORT_ENDPOINT_WITHIN_BC_NULL"
        else:
            feedback_status = "SUPPORTED"
        result.append(
            {
                "task_id": task_id,
                "outcome_class": rows[0]["outcome_class"],
                "role": rows[0]["role"],
                "repeat_count": len(rows),
                "same_policy_input_fraction_at_divergence": float(
                    np.mean(
                        np.asarray(
                            [bool(row["same_policy_input_at_divergence"]) for row in rows],
                            dtype=np.float64,
                        )
                    )
                ),
                "median_executed_action_l2": _median(
                    row["executed_action_l2"] for row in rows
                ),
                "median_tail_action_l2": _median(row["tail_action_l2"] for row in rows),
                "median_executed_h4_normalized_minus_null": h4_excess,
                "executed_h4_material_vs_bc_null": bool(h4_excess > 0.0),
                "tail_dominant_action_drift": bool(tail_minus_executed > 0.0),
                "median_feedback_position_gap_growth_m": feedback_growth,
                "median_feedback_endpoint_normalized_minus_null": endpoint_excess,
                "fixed_0_to_5_feedback_status": feedback_status,
                "fixed_0_to_5_feedback_supported": bool(
                    feedback_status == "SUPPORTED"
                ),
            }
        )
    return result


def _progression_aggregate(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    scopes = ["ALL"] + sorted({str(row["outcome_class"]) for row in rows})
    output = []
    for scope in scopes:
        scoped = [
            row for row in rows if scope == "ALL" or str(row["outcome_class"]) == scope
        ]
        for relative in range(PRIMARY_FEEDBACK_START, PRIMARY_FEEDBACK_END + 1):
            relative_rows = [
                row for row in scoped if int(row["relative_decision"]) == relative
            ]
            by_task = defaultdict(list)
            for row in relative_rows:
                by_task[str(row["task_id"])].append(row)
            if not by_task:
                continue
            task_action = [
                _median(row["executed_action_l2"] for row in task_rows)
                for task_rows in by_task.values()
            ]
            task_gap = [
                _median(row["position_gap_after_m"] for row in task_rows)
                for task_rows in by_task.values()
            ]
            output.append(
                {
                    "scope": scope,
                    "relative_decision": relative,
                    "task_count": len(by_task),
                    "repeat_count": len(relative_rows),
                    "executed_action_l2_task_median": _median(task_action),
                    "executed_action_l2_task_q25": _quantile(task_action, 0.25),
                    "executed_action_l2_task_q75": _quantile(task_action, 0.75),
                    "position_gap_after_m_task_median": _median(task_gap),
                    "position_gap_after_m_task_q25": _quantile(task_gap, 0.25),
                    "position_gap_after_m_task_q75": _quantile(task_gap, 0.75),
                }
            )
    return output


def audit_action_feedback(
    selected_rows: Sequence[Mapping[str, Any]],
    transitions: Sequence[Mapping[str, Any]],
    bc_nulls: Mapping[str, Any],
) -> Dict[str, Any]:
    """Audit selected D0.5 pairs at k_div through the fixed k_div+5 endpoint."""

    episodes = defaultdict(dict)
    for row in transitions:
        key = (str(row.get("task_id")), int(row.get("repeat")), str(row.get("lane")))
        decision = int(row.get("decision_index"))
        if decision in episodes[key]:
            raise ValueError("duplicate transition decision {}".format((key, decision)))
        episodes[key][decision] = row

    decision_rows = []
    waypoint_rows = []
    repeat_rows = []
    for selected in selected_rows:
        task_id = str(selected["task_id"])
        repeat = int(selected["repeat"])
        identity = (task_id, repeat)
        k_div = selected.get("k_div")
        if k_div is None:
            raise ValueError("small action audit requires a determinate k_div")
        k_div = int(k_div)
        decision_count = int(selected["decision_count"])
        u0 = episodes.get((task_id, repeat, "u0"), {})
        rl = episodes.get((task_id, repeat, "seed3-u32"), {})
        if not u0 or not rl:
            raise ValueError("missing paired episode {}".format(identity))
        config = selected.get("episode_config")
        if not isinstance(config, Mapping):
            raise ValueError("selection is missing episode_config")
        start = _finite_xyz(config.get("start_pos"), "start_pos")
        goal = _finite_xyz(config.get("goal_pos"), "goal_pos")
        scale = _l2(goal - start)
        if scale <= 1e-12:
            raise ValueError("initial goal distance must be positive")

        pair_decisions = []
        for relative in range(PRIMARY_FEEDBACK_START, PRIMARY_FEEDBACK_END + 1):
            decision = k_div + relative
            if decision >= decision_count or decision not in u0 or decision not in rl:
                continue
            left = u0[decision]
            right = rl[decision]
            _validate_transition_pair(left, right, identity, decision)
            decomposition = decompose_action_pair(
                left["policy_action_8x11"],
                right["policy_action_8x11"],
                position_before=left["position_before_xyz"],
                goal=left["goal_xyz"],
            )
            h4_threshold = _null_threshold(
                bc_nulls,
                task_id,
                decision,
                "h4_path",
                required=bool(relative == PRIMARY_FEEDBACK_START),
            )
            endpoint_threshold = _null_threshold(
                bc_nulls,
                task_id,
                decision,
                "endpoint",
                required=bool(relative == PRIMARY_FEEDBACK_START),
            )
            h4_normalized = decomposition["executed_position_path_l2_m"] / scale
            left_before = _finite_xyz(left["position_before_xyz"], "u0 position_before")
            right_before = _finite_xyz(right["position_before_xyz"], "rl position_before")
            left_after = _finite_xyz(left["position_after_xyz"], "u0 position_after")
            right_after = _finite_xyz(right["position_after_xyz"], "rl position_after")
            position_gap_before = _l2(right_before - left_before)
            position_gap_after = _l2(right_after - left_after)
            flat = {
                "task_id": task_id,
                "repeat": repeat,
                "outcome_class": str(selected["outcome_class"]),
                "role": str(selected["role"]),
                "k_div": k_div,
                "decision_index": decision,
                "relative_decision": relative,
                "action_seed": int(left["action_seed"]),
                "same_policy_input": bool(
                    left.get("policy_input_hash") == right.get("policy_input_hash")
                ),
                "initial_goal_distance_m": scale,
                "bc_h4_path_null_threshold": h4_threshold,
                "bc_endpoint_null_threshold": endpoint_threshold,
                "executed_h4_path_delta_normalized": h4_normalized,
                "executed_h4_normalized_minus_null": (
                    None if h4_threshold is None else h4_normalized - h4_threshold
                ),
                "executed_h4_material_vs_bc_null": (
                    None if h4_threshold is None else bool(h4_normalized > h4_threshold)
                ),
                "position_gap_before_m": position_gap_before,
                "position_gap_after_m": position_gap_after,
                "position_gap_after_normalized": position_gap_after / scale,
                "endpoint_normalized_minus_null": (
                    None
                    if endpoint_threshold is None
                    else position_gap_after / scale - endpoint_threshold
                ),
                "endpoint_material_vs_bc_null": (
                    None
                    if endpoint_threshold is None
                    else bool(position_gap_after / scale > endpoint_threshold)
                ),
            }
            for name, value in decomposition.items():
                if name != "waypoints":
                    flat[name] = value
            decision_rows.append(flat)
            pair_decisions.append(flat)
            for waypoint in decomposition["waypoints"]:
                waypoint_rows.append(dict(flat, **waypoint))

        if not pair_decisions or int(pair_decisions[0]["relative_decision"]) != 0:
            raise ValueError("paired episode has no first-divergence action row")
        by_relative = {int(row["relative_decision"]): row for row in pair_decisions}
        first = by_relative[PRIMARY_FEEDBACK_START]
        endpoint = by_relative.get(PRIMARY_FEEDBACK_END)
        repeat_rows.append(
            {
                "task_id": task_id,
                "repeat": repeat,
                "outcome_class": str(selected["outcome_class"]),
                "role": str(selected["role"]),
                "k_div": k_div,
                "same_policy_input_at_divergence": first["same_policy_input"],
                "executed_action_l2": first["executed_action_l2"],
                "tail_action_l2": first["tail_action_l2"],
                "executed_dpos_l2": first["executed_dpos_l2"],
                "tail_dpos_l2": first["tail_dpos_l2"],
                "goal_parallel_delta_m": first["goal_parallel_delta_m"],
                "goal_perpendicular_delta_m": first["goal_perpendicular_delta_m"],
                "vertical_delta_m": first["vertical_delta_m"],
                "yaw_sum_delta_rad": first["yaw_sum_delta_rad"],
                "executed_h4_normalized_minus_null": first[
                    "executed_h4_normalized_minus_null"
                ],
                "executed_h4_material_vs_bc_null": first[
                    "executed_h4_material_vs_bc_null"
                ],
                "primary_feedback_start": PRIMARY_FEEDBACK_START,
                "primary_feedback_endpoint": (
                    PRIMARY_FEEDBACK_END if endpoint is not None else None
                ),
                "feedback_position_gap_growth_m": (
                    None
                    if endpoint is None
                    else endpoint["position_gap_after_m"]
                    - first["position_gap_after_m"]
                ),
                "feedback_endpoint_normalized_minus_null": (
                    None if endpoint is None else endpoint["endpoint_normalized_minus_null"]
                ),
            }
        )

    task_rows = _task_aggregate(repeat_rows)
    initial_tasks = sum(
        bool(row["executed_h4_material_vs_bc_null"]) for row in task_rows
    )
    feedback_tasks = sum(
        bool(row["fixed_0_to_5_feedback_supported"]) for row in task_rows
    )
    tail_tasks = sum(bool(row["tail_dominant_action_drift"]) for row in task_rows)
    break_rows = [row for row in task_rows if row["outcome_class"] == "break"]
    rescue_rows = [row for row in task_rows if row["outcome_class"] == "rescue"]
    break_initial_tasks = sum(
        bool(row["executed_h4_material_vs_bc_null"]) for row in break_rows
    )
    break_feedback_tasks = sum(
        bool(row["fixed_0_to_5_feedback_supported"]) for row in break_rows
    )
    rescue_initial_tasks = sum(
        bool(row["executed_h4_material_vs_bc_null"]) for row in rescue_rows
    )
    rescue_feedback_tasks = sum(
        bool(row["fixed_0_to_5_feedback_supported"]) for row in rescue_rows
    )
    feedback_indeterminate_tasks = sum(
        str(row["fixed_0_to_5_feedback_status"]).startswith("INDETERMINATE_")
        for row in task_rows
    )
    if break_initial_tasks >= 2 and break_feedback_tasks >= 2:
        route = "EXECUTED_H4_DRIFT_AND_FEEDBACK_AMPLIFICATION_SUPPORTED"
    elif break_initial_tasks >= 2:
        route = "EXECUTED_H4_DRIFT_SUPPORTED_FEEDBACK_NOT_REPEATED"
    elif sum(bool(row["tail_dominant_action_drift"]) for row in break_rows) >= 2:
        route = "TAIL_DOMINANT_DRIFT_D2X_CANDIDATE"
    else:
        route = "ACTION_CHUNK_ROUTE_NOT_DETERMINATE"
    summary = {
        "schema": "routea_action_chunk_feedback_summary_v1",
        "status": "ACTION_CHUNK_FEEDBACK_AUDIT_COMPLETE",
        "route_candidate": route,
        "repeat_count": len(repeat_rows),
        "task_count": len(task_rows),
        "break_task_count": len(
            {row["task_id"] for row in task_rows if row["outcome_class"] == "break"}
        ),
        "rescue_task_count": len(
            {row["task_id"] for row in task_rows if row["outcome_class"] == "rescue"}
        ),
        "executed_h4_material_task_count": initial_tasks,
        "fixed_0_to_5_feedback_supported_task_count": feedback_tasks,
        "tail_dominant_task_count": tail_tasks,
        "break_executed_h4_material_task_count": break_initial_tasks,
        "break_fixed_0_to_5_feedback_supported_task_count": break_feedback_tasks,
        "rescue_executed_h4_material_task_count": rescue_initial_tasks,
        "rescue_fixed_0_to_5_feedback_supported_task_count": rescue_feedback_tasks,
        "fixed_0_to_5_feedback_indeterminate_task_count": feedback_indeterminate_tasks,
        "route_candidate_uses_break_tasks_only": True,
        "primary_feedback_comparison": "k_div+0_to_k_div+5_only",
        "intermediate_decisions_are_descriptive_only": True,
        "mechanism_support_threshold_tasks": 2,
        "mechanism_support_is_not_population_root_cause": True,
        "late_bc_null_missing_is_indeterminate": True,
        "feedback_support_requires_initial_h4_and_endpoint_above_null_plus_positive_fixed_0_to_5_growth": True,
        "training_authorized": False,
        "reward_change_authorized": False,
        "causal_claim_allowed": False,
    }
    return {
        "summary": summary,
        "repeat_rows": repeat_rows,
        "task_rows": task_rows,
        "decision_rows": decision_rows,
        "waypoint_rows": waypoint_rows,
        "progression_rows": _progression_aggregate(decision_rows),
    }


def summarize_b1_progression(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse the existing B1 bank by seed/update and stratum at task scope."""

    pattern = re.compile(r"^seed([123])-u(8|16|24|32)$")
    metrics = (
        "action_h4_l2",
        "goal_parallel_m",
        "goal_perpendicular_m",
        "vertical_m",
        "yaw_total_variation_rad",
        "path_shape_norm_m",
    )
    parsed = []
    for row in rows:
        match = pattern.match(str(row.get("checkpoint", "")))
        if match is None:
            continue
        seed = int(match.group(1))
        update = int(match.group(2))
        if int(row.get("update")) != update:
            raise ValueError("B1 checkpoint/update mismatch")
        item = {
            "seed": seed,
            "update": update,
            "stratum": str(row.get("stratum")),
            "task_id": str(row.get("task_id")),
            "snapshot_id": str(row.get("snapshot_id")),
        }
        if not item["snapshot_id"]:
            raise ValueError("B1 row is missing snapshot_id")
        for metric in metrics:
            value = float(row.get(metric))
            if not math.isfinite(value):
                raise ValueError("B1 metric is non-finite: {}".format(metric))
            item[metric] = value
        parsed.append(item)
    expected = {(seed, update) for seed in (1, 2, 3) for update in (8, 16, 24, 32)}
    actual = {(row["seed"], row["update"]) for row in parsed}
    if actual != expected:
        raise ValueError("B1 progression is missing seed/update lanes")
    snapshot_banks = {}
    for lane in sorted(expected):
        lane_rows = [
            row for row in parsed if (row["seed"], row["update"]) == lane
        ]
        snapshot_ids = [row["snapshot_id"] for row in lane_rows]
        if len(snapshot_ids) != len(set(snapshot_ids)):
            raise ValueError("B1 lane contains duplicate snapshot_id")
        snapshot_banks[lane] = set(snapshot_ids)
    reference_lane = sorted(expected)[0]
    reference_bank = snapshot_banks[reference_lane]
    for lane in sorted(expected):
        if snapshot_banks[lane] != reference_bank:
            raise ValueError(
                "B1 seed/update lanes do not share one snapshot bank: {}".format(lane)
            )

    output = []
    strata = sorted({row["stratum"] for row in parsed})
    for seed, update in sorted(expected):
        lane = [
            row for row in parsed if row["seed"] == seed and row["update"] == update
        ]
        for stratum in ["ALL"] + strata:
            scoped = [row for row in lane if stratum == "ALL" or row["stratum"] == stratum]
            if not scoped:
                continue
            summary = {
                "seed": seed,
                "update": update,
                "checkpoint": "seed{}-u{}".format(seed, update),
                "stratum": stratum,
                "snapshot_count": len(scoped),
                "task_count": len({row["task_id"] for row in scoped}),
                "evidence_scope": "existing_B1_enriched_bank_descriptive_only",
                "statistical_unit": "task",
            }
            by_task = defaultdict(list)
            for row in scoped:
                by_task[row["task_id"]].append(row)
            for metric in metrics:
                task_values = [
                    _median(row[metric] for row in task_rows)
                    for task_rows in by_task.values()
                ]
                summary[metric + "_task_mean"] = float(np.mean(task_values))
                summary[metric + "_task_median"] = _median(task_values)
                summary[metric + "_task_q25"] = _quantile(task_values, 0.25)
                summary[metric + "_task_q75"] = _quantile(task_values, 0.75)
            output.append(summary)
    return output
