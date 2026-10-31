"""Pure contracts for the Route A D1-F failure-state diagnostic.

The functions in this module deliberately avoid model/runtime imports.  They
freeze population identity, divergence, functional-response, direction, and
task-level inference rules so the GPU-facing CLI cannot silently change the
scientific contract.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .contracts import DiagnosticFailure, load_numeric_npz, load_strict_json


DIVERGENCE_METRICS = (
    "endpoint",
    "h4_path",
    "goal_parallel",
    "goal_perpendicular",
    "vertical",
    "yaw",
)

ANCHOR_ORDER = (
    "pre_divergence",
    "divergence",
    "post_divergence_1",
    "post_divergence_2",
    "closest_goal",
    "preterminal",
)

BOTH_BELOW_FUNCTIONAL_REFERENCE = "BOTH_BELOW_FUNCTIONAL_REFERENCE"
RL_RETENTION_OR_DIRECTION_FAILURE = "RL_RETENTION_OR_DIRECTION_FAILURE"
GOAL_RESPONSE_PRESERVED = "GOAL_RESPONSE_PRESERVED"
BC_RECOVERY_REFERENCE_INVALID_OR_MIXED = "BC_RECOVERY_REFERENCE_INVALID_OR_MIXED"


@dataclass(frozen=True)
class BreakIdentity:
    task_id: str
    repeat: int
    env_reset_seed: int
    world: str
    split: str
    role: str
    bc_trace_path: str
    seed3_trace_path: str
    bc_trace_sha256: str
    seed3_trace_sha256: str
    continuous_break: bool
    bc_done_reason: str
    seed3_done_reason: str


def build_replay_identity_diagnostics(
    *,
    task_id: str,
    expected_observation: Mapping[str, Any],
    actual_provenance: Any,
    rebuilt_planned: np.ndarray,
    frozen_planned: np.ndarray,
) -> dict[str, Any]:
    """Locate an exact replay failure without weakening the replay gate."""

    rebuilt = np.asarray(rebuilt_planned)
    frozen = np.asarray(frozen_planned)
    if rebuilt.shape != frozen.shape:
        raise ValueError("rebuilt and frozen planned arrays must have the same shape")

    def provenance_field(name: str) -> Any:
        if isinstance(actual_provenance, Mapping):
            return actual_provenance.get(name)
        return getattr(actual_provenance, name, None)

    expected_hashes = {
        "raw_frame": expected_observation.get("last_raw_frame_hash"),
        "normalized_history": expected_observation.get("last_normalized_history_hash"),
        "policy_input": expected_observation.get("last_policy_input_hash"),
    }
    actual_hashes = {
        "raw_frame": provenance_field("raw_frame_hash"),
        "raw_history": provenance_field("raw_history_hash"),
        "normalizer_identity": provenance_field("normalizer_identity_hash"),
        "normalized_history": provenance_field("normalized_history_hash"),
        "policy_input": provenance_field("policy_input_hash"),
        "episode_identity": provenance_field("episode_identity_hash"),
        "decision_index": provenance_field("decision_index"),
    }
    comparisons = {
        f"{name}_exact": None if expected is None else actual_hashes[name] == expected
        for name, expected in expected_hashes.items()
    }
    first_mismatch_layer = "NONE"
    for comparison, layer in (
        ("raw_frame_exact", "RAW_FRAME"),
        ("normalized_history_exact", "NORMALIZED_HISTORY"),
        ("policy_input_exact", "ENCODER_CONDITION"),
    ):
        if comparisons[comparison] is False:
            first_mismatch_layer = layer
            break

    mismatch_indices = np.argwhere(rebuilt != frozen)
    first_mismatch = None
    if len(mismatch_indices):
        index = tuple(int(value) for value in mismatch_indices[0])
        first_mismatch = {
            "row": index[0],
            "column": index[1] if len(index) > 1 else None,
            "rebuilt": float(rebuilt[index]),
            "frozen": float(frozen[index]),
            "abs_difference": float(abs(float(rebuilt[index]) - float(frozen[index]))),
        }
    max_abs = (
        0.0
        if rebuilt.size == 0
        else float(np.max(np.abs(rebuilt.astype(np.float64) - frozen.astype(np.float64))))
    )
    return {
        "schema": "routea_d1f_replay_identity_diagnostics_v1",
        "task_id": str(task_id),
        "first_mismatch_layer": first_mismatch_layer,
        "expected_hashes": expected_hashes,
        "actual_hashes": actual_hashes,
        "comparisons": comparisons,
        "trajectory_roundtrip": {
            "bitwise_exact": bool(np.array_equal(rebuilt, frozen)),
            "mismatch_element_count": int(len(mismatch_indices)),
            "max_abs_difference": max_abs,
            "first_mismatch": first_mismatch,
        },
    }


def endpoint_identity_audit(
    parent_rows: Sequence[Any], done_reason_by_trace_path: Mapping[str, str]
) -> tuple[tuple[BreakIdentity, ...], dict[str, Any]]:
    """Select the endpoint-defined BC->seed3 breaks and audit continuous extras."""

    selected: list[BreakIdentity] = []
    continuous_keys: set[tuple[str, int]] = set()
    endpoint_keys: set[tuple[str, int]] = set()
    for row in parent_rows:
        bc_path = str(row.trace_paths[0])
        seed3_path = str(row.trace_paths[3])
        if bc_path not in done_reason_by_trace_path or seed3_path not in done_reason_by_trace_path:
            raise DiagnosticFailure(
                "ENDPOINT_IDENTITY_MISMATCH",
                f"missing done_reason for task={row.task_id} repeat={row.repeat}",
            )
        bc_done = str(done_reason_by_trace_path[bc_path])
        seed3_done = str(done_reason_by_trace_path[seed3_path])
        key = (str(row.task_id), int(row.repeat))
        continuous = bool(row.bc_reached and not row.seed3_reached)
        endpoint = bc_done == "reached" and seed3_done != "reached"
        if continuous:
            continuous_keys.add(key)
        if not endpoint:
            continue
        endpoint_keys.add(key)
        selected.append(
            BreakIdentity(
                task_id=str(row.task_id),
                repeat=int(row.repeat),
                env_reset_seed=int(row.env_reset_seed),
                world=str(row.world),
                split=str(row.split),
                role=str(row.role),
                bc_trace_path=bc_path,
                seed3_trace_path=seed3_path,
                bc_trace_sha256=str(row.bc_trace_sha256),
                seed3_trace_sha256=str(row.rl_trace_sha256[2]),
                continuous_break=continuous,
                bc_done_reason=bc_done,
                seed3_done_reason=seed3_done,
            )
        )
    selected.sort(key=lambda item: (item.task_id, item.repeat))
    continuous_only = sorted(continuous_keys - endpoint_keys)
    endpoint_only = sorted(endpoint_keys - continuous_keys)
    return tuple(selected), {
        "schema": "routea_d1f_endpoint_identity_audit_v1",
        "endpoint_break_repeat_count": len(endpoint_keys),
        "endpoint_break_task_count": len({task for task, _ in endpoint_keys}),
        "continuous_break_repeat_count": len(continuous_keys),
        "continuous_break_task_count": len({task for task, _ in continuous_keys}),
        "continuous_only_repeat_count": len(continuous_only),
        "continuous_only_identities": [
            {"task_id": task, "repeat": repeat} for task, repeat in continuous_only
        ],
        "endpoint_only_repeat_count": len(endpoint_only),
        "endpoint_only_identities": [
            {"task_id": task, "repeat": repeat} for task, repeat in endpoint_only
        ],
    }


def extract_decision_features(
    planned_rows: Any, *, start: Any, goal: Any
) -> list[dict[str, Any]]:
    """Extract the six registered divergence metrics from executed H4 only."""

    planned = np.asarray(planned_rows, dtype=np.float64)
    start_a = np.asarray(start, dtype=np.float64)
    goal_a = np.asarray(goal, dtype=np.float64)
    if planned.ndim != 2 or planned.shape[1] != 14 or not np.isfinite(planned).all():
        raise ValueError("planned table must be finite with shape (N,14)")
    if start_a.shape != (3,) or goal_a.shape != (3,):
        raise ValueError("start and goal must be xyz")
    scale = max(float(np.linalg.norm(goal_a - start_a)), 1e-12)
    features: list[dict[str, Any]] = []
    for decision_step in sorted(set(planned[:, 0].astype(int))):
        rows = planned[planned[:, 0].astype(int) == decision_step]
        rows = rows[np.argsort(rows[:, 1])]
        if len(rows) != 8 or tuple(rows[:, 1].astype(int)) != tuple(range(8)):
            raise ValueError(f"decision {decision_step} does not contain exactly k=0..7")
        executed = rows[:4]
        path = executed[:, 3:6]
        before = start_a if not features else np.asarray(features[-1]["endpoint_world"])
        endpoint = path[-1]
        displacement = endpoint - before
        target = goal_a - before
        target_horizontal = target[:2]
        horizontal_norm = float(np.linalg.norm(target_horizontal))
        if horizontal_norm > 1e-12:
            axis = target_horizontal / horizontal_norm
            parallel = float(np.dot(displacement[:2], axis)) / scale
            perpendicular = float(
                np.dot(displacement[:2], np.asarray([-axis[1], axis[0]]))
            ) / scale
        else:
            parallel = 0.0
            perpendicular = 0.0
        yaw = float(executed[-1, 12])
        features.append(
            {
                "decision_index": len(features),
                "endpoint": ((endpoint - start_a) / scale).tolist(),
                "h4_path": ((path - start_a[None, :]) / scale).reshape(-1).tolist(),
                "goal_parallel": parallel,
                "goal_perpendicular": perpendicular,
                "vertical": float(displacement[2] / scale),
                "yaw": [math.sin(yaw), math.cos(yaw)],
                "endpoint_world": endpoint.tolist(),
                "goal_distance": float(np.linalg.norm(goal_a - before)),
            }
        )
    return features


def _metric_distance(left: Any, right: Any) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape:
        raise ValueError(f"metric shapes differ: {left_array.shape} != {right_array.shape}")
    if not np.isfinite(left_array).all() or not np.isfinite(right_array).all():
        raise ValueError("metric contains a non-finite value")
    return float(np.linalg.norm(left_array - right_array))


def _upper_quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires at least one value")
    ordered = sorted(float(value) for value in values)
    index = int(math.ceil(float(probability) * len(ordered))) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def build_bc_repeat_null(
    repeat_features: Mapping[int, Sequence[Mapping[str, Any]]],
) -> dict[int, dict[str, dict[str, Any]]]:
    """Build a conservative pairwise Q0.95 null for each decision and metric."""

    if len(repeat_features) < 4:
        raise ValueError("BC null requires at least four valid repeats")
    decision_count = min(len(rows) for rows in repeat_features.values())
    if decision_count < 1:
        raise ValueError("BC null has no shared decision")
    repeats = sorted(repeat_features)
    result: dict[int, dict[str, dict[str, Any]]] = {}
    for decision in range(decision_count):
        result[decision] = {}
        for metric in DIVERGENCE_METRICS:
            distances = []
            for left_index, left_repeat in enumerate(repeats):
                for right_repeat in repeats[left_index + 1 :]:
                    try:
                        left = repeat_features[left_repeat][decision][metric]
                        right = repeat_features[right_repeat][decision][metric]
                    except KeyError as exc:
                        raise ValueError(f"missing divergence metric: {metric}") from exc
                    distances.append(_metric_distance(left, right))
            if len(distances) < 6:
                raise ValueError(
                    f"BC null decision={decision} metric={metric} has {len(distances)} pairs; expected >=6"
                )
            result[decision][metric] = {
                "threshold": _upper_quantile(distances, 0.95),
                "pair_count": len(distances),
                "pairwise_distances": sorted(distances),
            }
    return result


def serialize_bc_repeat_null(
    task_nulls: Mapping[str, Mapping[int, Mapping[str, Any]]]
) -> dict[str, dict[str, Mapping[str, Any]]]:
    """Convert numeric decision indices only at the strict-JSON boundary."""

    return {
        str(task_id): {str(decision): metrics for decision, metrics in sorted(null.items())}
        for task_id, null in sorted(task_nulls.items())
    }


def find_first_persistent_divergence(
    bc_features: Sequence[Mapping[str, Any]],
    rl_features: Sequence[Mapping[str, Any]],
    null: Mapping[int, Mapping[str, Mapping[str, Any]]],
) -> int | None:
    """Return the first decision whose same metric exceeds null twice in a row."""

    count = min(len(bc_features), len(rl_features))
    if count < 2:
        return None
    for decision in range(count - 1):
        for metric in DIVERGENCE_METRICS:
            current_threshold = float(null.get(decision, {}).get(metric, {}).get("threshold", math.inf))
            next_threshold = float(null.get(decision + 1, {}).get(metric, {}).get("threshold", math.inf))
            current = _metric_distance(bc_features[decision][metric], rl_features[decision][metric])
            following = _metric_distance(bc_features[decision + 1][metric], rl_features[decision + 1][metric])
            if current > current_threshold and following > next_threshold:
                return decision
    return None


def anchor_decisions(
    k_div: int | None,
    *,
    decision_count: int,
    closest_index: int,
) -> dict[int, tuple[str, ...]]:
    """Freeze and deduplicate anchor labels without inventing a divergence."""

    if decision_count < 1:
        raise ValueError("decision_count must be positive")
    if not 0 <= int(closest_index) < int(decision_count):
        raise ValueError("closest_index is outside the trajectory")
    labels: dict[int, list[str]] = {}

    def add(index: int, label: str) -> None:
        if 0 <= index < decision_count:
            labels.setdefault(index, []).append(label)

    if k_div is not None:
        if not 0 <= int(k_div) < decision_count:
            raise ValueError("k_div is outside the trajectory")
        add(int(k_div) - 1, "pre_divergence")
        add(int(k_div), "divergence")
        add(int(k_div) + 1, "post_divergence_1")
        add(int(k_div) + 2, "post_divergence_2")
    add(int(closest_index), "closest_goal")
    add(decision_count - 1, "preterminal")
    order = {name: index for index, name in enumerate(ANCHOR_ORDER)}
    return {
        index: tuple(sorted(set(names), key=order.__getitem__))
        for index, names in sorted(labels.items())
    }


def build_functional_reference(
    rows: Sequence[Mapping[str, Any]], measurement_floor: float
) -> dict[str, Any]:
    """Freeze channel floors from exactly eight pre-existing shared BC states."""

    channels = (
        "bearing_norm",
        "vertical_norm",
        "combined_norm",
        "abs_bearing_yaw_delta",
    )
    channel_report: dict[str, Any] = {}
    valid = len(rows) == 8
    invalid_reasons: list[str] = []
    if len(rows) != 8:
        invalid_reasons.append(f"shared BC reference count={len(rows)}; expected 8")
    for channel in channels:
        values = []
        for row in rows:
            try:
                value = float(row[channel])
            except (KeyError, TypeError, ValueError):
                value = math.nan
            values.append(value)
        finite = len(values) == 8 and all(math.isfinite(value) for value in values)
        floor = min(values) if finite else None
        channel_valid = finite and floor is not None and floor > float(measurement_floor)
        if not channel_valid:
            invalid_reasons.append(
                f"{channel} reference invalid: floor={floor} measurement_floor={measurement_floor}"
            )
        valid = valid and channel_valid
        ordered = sorted(values) if finite else []
        channel_report[channel] = {
            "functional_floor": floor,
            "minimum": floor,
            "q25": None if not finite else float(np.quantile(ordered, 0.25)),
            "median": None if not finite else float(np.median(ordered)),
            "maximum": None if not finite else max(ordered),
            "sample_count": len(values),
            "valid": channel_valid,
        }
    return {
        "schema": "routea_d1f_functional_reference_v1",
        "measurement_floor": float(measurement_floor),
        "required_shared_bc_count": 8,
        "shared_bc_count": len(rows),
        "channels": channel_report,
        "valid": bool(valid),
        "invalid_reasons": invalid_reasons,
    }


def build_shared_goal_response_reference(
    final_dir: Path | str,
    b2_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Rebuild the functional floor and direction baseline from frozen v4 B2."""

    from .goals import make_goal_variant, reconstruct_goal_history
    from .reporting import file_sha256, verify_sha256_manifest

    final = Path(final_dir).resolve()
    b2 = final if b2_dir is None else Path(b2_dir).resolve()
    verify_sha256_manifest(final)
    verify_sha256_manifest(b2)
    summary = load_strict_json(final / "goal_sensitivity_summary.json")
    measurement_floor = float(summary["measurement_floor"]["tol_abs"])
    manifest = load_strict_json(final / "observation_bank_manifest.json")
    bank = load_numeric_npz(final / "observation_bank.npz", max_total_bytes=2 * 1024**3)
    snapshots = {str(row["snapshot_id"]): row for row in manifest["snapshots"]}
    rows_by_snapshot: dict[str, dict[str, dict[str, Mapping[str, str]]]] = {}
    csv_path = b2 / "b2_sentinel_pairs.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            snapshot_id = str(row["snapshot_id"])
            policy = str(row["policy"])
            variant = str(row["target_variant"])
            policies = rows_by_snapshot.setdefault(snapshot_id, {})
            variants = policies.setdefault(policy, {})
            if variant in variants:
                raise DiagnosticFailure(
                    "FUNCTIONAL_REFERENCE_INVALID",
                    f"duplicate B2 row: {snapshot_id}/{policy}/{variant}",
                )
            variants[variant] = row
    required_variants = {
        "L0-original",
        "L1-bearing+15",
        "L2-bearing-15",
        "L5-z+0.3m",
        "L6-z-0.3m",
    }
    if len(rows_by_snapshot) != 8:
        raise DiagnosticFailure(
            "FUNCTIONAL_REFERENCE_INVALID",
            f"B2 shared snapshot count={len(rows_by_snapshot)}; expected 8",
        )
    snapshot_reports = []
    floor_rows = []
    bearing_denominator = math.pi / 6.0
    vertical_denominator = 0.6
    for snapshot_id in sorted(rows_by_snapshot):
        if snapshot_id not in snapshots:
            raise DiagnosticFailure(
                "FUNCTIONAL_REFERENCE_INVALID", f"B2 snapshot absent from bank: {snapshot_id}"
            )
        policy_rows = rows_by_snapshot[snapshot_id]
        if set(policy_rows) != {"bc", "seed3-u32"} or any(
            set(variants) != required_variants for variants in policy_rows.values()
        ):
            raise DiagnosticFailure(
                "FUNCTIONAL_REFERENCE_INVALID", f"incomplete B2 variants: {snapshot_id}"
            )
        snapshot = snapshots[snapshot_id]
        bank_index = int(snapshot["bank_index"])
        agent_pos = bank["agent_pos"][bank_index]
        goals = {
            name: reconstruct_goal_history(make_goal_variant(agent_pos, name)).goal
            for name in required_variants
        }

        def endpoint(policy: str, variant: str) -> np.ndarray:
            try:
                value = json.loads(policy_rows[policy][variant]["endpoint"])
                array = np.asarray(value, dtype=np.float64)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise DiagnosticFailure(
                    "FUNCTIONAL_REFERENCE_INVALID",
                    f"invalid endpoint {snapshot_id}/{policy}/{variant}",
                ) from exc
            if array.shape != (3,) or not np.isfinite(array).all():
                raise DiagnosticFailure(
                    "FUNCTIONAL_REFERENCE_INVALID",
                    f"non-finite endpoint {snapshot_id}/{policy}/{variant}",
                )
            return array

        def scalar(policy: str, variant: str, field: str) -> float:
            value = float(policy_rows[policy][variant][field])
            if not math.isfinite(value):
                raise DiagnosticFailure(
                    "FUNCTIONAL_REFERENCE_INVALID",
                    f"non-finite {field} {snapshot_id}/{policy}/{variant}",
                )
            return value

        bearing = response_direction_metrics(
            bc_plus=endpoint("bc", "L1-bearing+15"),
            bc_minus=endpoint("bc", "L2-bearing-15"),
            rl_plus=endpoint("seed3-u32", "L1-bearing+15"),
            rl_minus=endpoint("seed3-u32", "L2-bearing-15"),
            plus_goal=goals["L1-bearing+15"],
            minus_goal=goals["L2-bearing-15"],
            bc_yaw_delta=scalar("bc", "L1-bearing+15", "yaw_sum_rad")
            - scalar("bc", "L2-bearing-15", "yaw_sum_rad"),
            rl_yaw_delta=scalar("seed3-u32", "L1-bearing+15", "yaw_sum_rad")
            - scalar("seed3-u32", "L2-bearing-15", "yaw_sum_rad"),
            bc_floor=0.0,
            rl_floor=0.0,
            yaw_floor=0.0,
            response_denominator=bearing_denominator,
        )
        vertical = response_direction_metrics(
            bc_plus=endpoint("bc", "L5-z+0.3m"),
            bc_minus=endpoint("bc", "L6-z-0.3m"),
            rl_plus=endpoint("seed3-u32", "L5-z+0.3m"),
            rl_minus=endpoint("seed3-u32", "L6-z-0.3m"),
            plus_goal=goals["L5-z+0.3m"],
            minus_goal=goals["L6-z-0.3m"],
            bc_yaw_delta=0.0,
            rl_yaw_delta=0.0,
            bc_floor=0.0,
            rl_floor=0.0,
            yaw_floor=math.inf,
            response_denominator=vertical_denominator,
        )
        combined = float(math.hypot(bearing["bc_norm"], vertical["bc_norm"]))
        floor_rows.append(
            {
                "bearing_norm": bearing["bc_norm"],
                "vertical_norm": vertical["bc_norm"],
                "combined_norm": combined,
                "abs_bearing_yaw_delta": abs(float(bearing["bc_yaw_delta"])),
            }
        )
        snapshot_reports.append(
            {
                "snapshot_id": snapshot_id,
                "task_id": str(snapshot["task_id"]),
                "bearing": bearing,
                "vertical": vertical,
                "bc_combined_norm": combined,
                "seed3_combined_norm": float(
                    math.hypot(bearing["rl_norm"], vertical["rl_norm"])
                ),
            }
        )
    reference = build_functional_reference(floor_rows, measurement_floor)
    floors = {
        name: float(reference["channels"][name]["functional_floor"])
        for name in (
            "bearing_norm",
            "vertical_norm",
            "combined_norm",
            "abs_bearing_yaw_delta",
        )
    }
    shared_rows = []
    for report in snapshot_reports:
        classification = classify_failure_snapshot(
            {"bearing": report["bearing"], "vertical": report["vertical"]}, floors
        )
        report["classification"] = classification
        shared_rows.append(
            {
                "task_id": report["task_id"],
                "classification": classification,
                "anchor_labels": ["preterminal"],
                "split": "shared_b2",
                "role": "shared_b2",
                "goal_distance": 0.0,
            }
        )
    shared_summary = build_failure_summary(shared_rows)
    primary = [
        name
        for name, value in shared_summary["overall_coverage"].items()
        if value["support_level"] == "PRIMARY_ROUTE_SUPPORTED"
    ]
    return {
        **reference,
        "input_identity": {
            "final_manifest_sha256": file_sha256(final / "manifest_sha256.txt"),
            "b2_manifest_sha256": file_sha256(b2 / "manifest_sha256.txt"),
            "b2_rows_sha256": file_sha256(csv_path),
            "observation_bank_sha256": file_sha256(final / "observation_bank.npz"),
        },
        "registered_denominators": {
            "bearing_radians": bearing_denominator,
            "vertical_m": vertical_denominator,
        },
        "snapshot_reports": snapshot_reports,
        "shared_state_coverage": shared_summary["overall_coverage"],
        "shared_primary_mechanisms": primary,
    }


def _assignment_margin(
    plus_endpoint: np.ndarray,
    minus_endpoint: np.ndarray,
    plus_goal: np.ndarray,
    minus_goal: np.ndarray,
) -> float:
    return 0.5 * float(
        np.linalg.norm(plus_endpoint - minus_goal)
        - np.linalg.norm(plus_endpoint - plus_goal)
        + np.linalg.norm(minus_endpoint - plus_goal)
        - np.linalg.norm(minus_endpoint - minus_goal)
    )


def response_direction_metrics(
    *,
    bc_plus: Any,
    bc_minus: Any,
    rl_plus: Any,
    rl_minus: Any,
    plus_goal: Any,
    minus_goal: Any,
    bc_yaw_delta: float,
    rl_yaw_delta: float,
    bc_floor: float,
    rl_floor: float,
    yaw_floor: float,
    response_denominator: float = 1.0,
) -> dict[str, Any]:
    arrays = [
        np.asarray(value, dtype=np.float64)
        for value in (bc_plus, bc_minus, rl_plus, rl_minus, plus_goal, minus_goal)
    ]
    if any(array.shape != (3,) or not np.isfinite(array).all() for array in arrays):
        raise ValueError("direction endpoints and goals must be finite xyz")
    bc_plus_a, bc_minus_a, rl_plus_a, rl_minus_a, plus_goal_a, minus_goal_a = arrays
    denominator = float(response_denominator)
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("response_denominator must be finite and positive")
    bc_vector = (bc_plus_a - bc_minus_a) / denominator
    rl_vector = (rl_plus_a - rl_minus_a) / denominator
    bc_norm = float(np.linalg.norm(bc_vector))
    rl_norm = float(np.linalg.norm(rl_vector))
    cosine = None
    if (
        bc_norm >= float(bc_floor)
        and rl_norm >= float(rl_floor)
        and bc_norm * rl_norm > 0.0
    ):
        cosine = float(np.dot(bc_vector, rl_vector) / (bc_norm * rl_norm))
    bc_yaw_applicable = abs(float(bc_yaw_delta)) >= float(yaw_floor)
    yaw_consistent = None
    if bc_yaw_applicable:
        yaw_consistent = bool(float(bc_yaw_delta) * float(rl_yaw_delta) > 0.0)
    return {
        "bc_response_vector": bc_vector.tolist(),
        "rl_response_vector": rl_vector.tolist(),
        "bc_norm": bc_norm,
        "rl_norm": rl_norm,
        "cosine": cosine,
        "bc_assignment_margin": _assignment_margin(
            bc_plus_a, bc_minus_a, plus_goal_a, minus_goal_a
        ),
        "rl_assignment_margin": _assignment_margin(
            rl_plus_a, rl_minus_a, plus_goal_a, minus_goal_a
        ),
        "bc_yaw_delta": float(bc_yaw_delta),
        "rl_yaw_delta": float(rl_yaw_delta),
        "bc_yaw_applicable": bc_yaw_applicable,
        "yaw_sign_consistent": yaw_consistent,
    }


def classify_failure_snapshot(
    channels: Mapping[str, Mapping[str, Any]], floors: Mapping[str, float]
) -> str:
    per_channel = []
    for channel in ("bearing", "vertical"):
        row = channels[channel]
        floor = float(floors[f"{channel}_norm"])
        bc_functional = float(row["bc_norm"]) >= floor
        rl_functional = float(row["rl_norm"]) >= floor
        if not bc_functional and not rl_functional:
            per_channel.append(BOTH_BELOW_FUNCTIONAL_REFERENCE)
            continue
        if not bc_functional or float(row.get("bc_assignment_margin", 0.0)) <= 0.0:
            per_channel.append(BC_RECOVERY_REFERENCE_INVALID_OR_MIXED)
            continue
        direction_ok = (
            rl_functional
            and float(row.get("rl_assignment_margin", 0.0)) > 0.0
            and row.get("cosine") is not None
            and float(row["cosine"]) > 0.0
        )
        if channel == "bearing" and abs(float(row.get("bc_yaw_delta", 0.0))) >= float(
            floors["abs_bearing_yaw_delta"]
        ):
            direction_ok = direction_ok and row.get("yaw_sign_consistent") is True
        per_channel.append(
            GOAL_RESPONSE_PRESERVED
            if direction_ok
            else RL_RETENTION_OR_DIRECTION_FAILURE
        )
    if RL_RETENTION_OR_DIRECTION_FAILURE in per_channel:
        return RL_RETENTION_OR_DIRECTION_FAILURE
    if BC_RECOVERY_REFERENCE_INVALID_OR_MIXED in per_channel:
        return BC_RECOVERY_REFERENCE_INVALID_OR_MIXED
    if all(value == BOTH_BELOW_FUNCTIONAL_REFERENCE for value in per_channel):
        return BOTH_BELOW_FUNCTIONAL_REFERENCE
    if all(value == GOAL_RESPONSE_PRESERVED for value in per_channel):
        return GOAL_RESPONSE_PRESERVED
    return BC_RECOVERY_REFERENCE_INVALID_OR_MIXED


def wilson_interval(numerator: int, denominator: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if denominator <= 0 or numerator < 0 or numerator > denominator:
        raise ValueError("invalid Wilson numerator/denominator")
    p = numerator / denominator
    z2 = z * z
    center = (p + z2 / (2.0 * denominator)) / (1.0 + z2 / denominator)
    half = z * math.sqrt(
        p * (1.0 - p) / denominator + z2 / (4.0 * denominator * denominator)
    ) / (1.0 + z2 / denominator)
    return max(0.0, center - half), min(1.0, center + half)


def support_summary(numerator: int, denominator: int) -> dict[str, Any]:
    if int(denominator) == 0:
        if int(numerator) != 0:
            raise ValueError("nonzero numerator with zero denominator")
        return {
            "mechanism_task_numerator": 0,
            "determinate_task_denominator": 0,
            "coverage_fraction": None,
            "wilson_95_lower": None,
            "wilson_95_upper": None,
            "support_level": "NONE",
        }
    lower, upper = wilson_interval(int(numerator), int(denominator))
    if numerator == 0:
        level = "NONE"
    elif numerator == 1:
        level = "SINGLE_TASK"
    elif lower > 0.5:
        level = "PRIMARY_ROUTE_SUPPORTED"
    else:
        level = "REPEATED_SUPPORT"
    return {
        "mechanism_task_numerator": int(numerator),
        "determinate_task_denominator": int(denominator),
        "coverage_fraction": float(numerator / denominator),
        "wilson_95_lower": lower,
        "wilson_95_upper": upper,
        "support_level": level,
    }


def _one_sided_exact_p(successes: int, failures: int) -> float:
    total = successes + failures
    if total == 0:
        return 1.0
    numerator = sum(math.comb(total, value) for value in range(successes, total + 1))
    return float(numerator / (2**total))


def exact_progression_test(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Pre-registered pre_divergence -> preterminal one-sided paired test.

    Intermediate anchors are intentionally ignored here.  Missing fixed
    endpoints reduce the paired denominator; they never trigger a fallback.
    """

    by_task: dict[str, dict[str, bool | None]] = {}
    for row in rows:
        task = by_task.setdefault(str(row["task_id"]), {})
        stage = str(row["stage"])
        if stage not in {"pre_divergence", "preterminal"}:
            continue
        task[stage] = (
            bool(row["mechanism"])
            if bool(row.get("determinate", True))
            else None
        )
    forward = 0
    reverse = 0
    paired = 0
    for task in by_task.values():
        if (
            "pre_divergence" not in task
            or "preterminal" not in task
            or task["pre_divergence"] is None
            or task["preterminal"] is None
        ):
            continue
        paired += 1
        before, after = task["pre_divergence"], task["preterminal"]
        forward += int(not before and after)
        reverse += int(before and not after)
    p_value = _one_sided_exact_p(forward, reverse)
    return {
        "schema": "routea_d1f_progression_v2",
        "pre_stage": "pre_divergence",
        "late_stage": "preterminal",
        "paired_task_count": paired,
        "pre_normal_to_late_failure": forward,
        "pre_failure_to_late_normal": reverse,
        "discordant_task_count": forward + reverse,
        "one_sided_exact_p": p_value,
        "progression_supported": bool(p_value < 0.05),
        "missing_fixed_endpoint_task_count": len(by_task) - paired,
        "fallback_stage": None,
        "intermediate_stages_role": "DESCRIPTIVE_ONLY",
    }


_CLASS_TO_MECHANISM = {
    RL_RETENTION_OR_DIRECTION_FAILURE: "RL_RETENTION_FAILURE",
    BOTH_BELOW_FUNCTIONAL_REFERENCE: "OBSERVATION_OR_SUPPORT_LIMIT",
    GOAL_RESPONSE_PRESERVED: "GOAL_CONTROL_PRESERVED",
}

_MECHANISM_ROUTE = {
    "RL_RETENTION_FAILURE": (
        "STATE_DEPENDENT_GOAL_CONTROL_FAILURE",
        "CAPABILITY_RETENTION_AND_RL_UPDATE_CONSTRAINTS",
    ),
    "OBSERVATION_OR_SUPPORT_LIMIT": (
        "OBSERVATION_OR_SUPPORT_LIMIT",
        "OBSERVATION_HISTORY_TIME_AND_MODEL_SUPPORT",
    ),
    "GOAL_CONTROL_PRESERVED": (
        "GOAL_CONTROL_PRESERVED_ON_FAILURE_STATES",
        "CREDIT_ACTION_CHUNK_AND_CLOSED_LOOP_FEEDBACK",
    ),
}


def _collapse_category(values: Sequence[str]) -> str:
    if not values:
        return BC_RECOVERY_REFERENCE_INVALID_OR_MIXED
    counts = {value: values.count(value) for value in set(values)}
    maximum = max(counts.values())
    winners = sorted(value for value, count in counts.items() if count == maximum)
    return winners[0] if len(winners) == 1 else BC_RECOVERY_REFERENCE_INVALID_OR_MIXED


def _median_or_none(values: Sequence[Any]) -> float | None:
    finite = []
    for value in values:
        if value is None:
            continue
        numeric = float(value)
        if math.isfinite(numeric):
            finite.append(numeric)
    return None if not finite else float(np.median(finite))


def _collapse_rows(
    rows: Sequence[Mapping[str, Any]],
    functional_floors: Mapping[str, float] | None,
) -> str:
    return str(_aggregate_rows(rows, functional_floors)["classification"])


def _aggregate_rows(
    rows: Sequence[Mapping[str, Any]],
    functional_floors: Mapping[str, float] | None,
) -> dict[str, Any]:
    if functional_floors is None or not all(
        isinstance(row.get("bearing"), Mapping) and isinstance(row.get("vertical"), Mapping)
        for row in rows
    ):
        return {
            "classification": _collapse_category(
                [str(row["classification"]) for row in rows]
            )
        }
    channels: dict[str, dict[str, Any]] = {}
    for channel in ("bearing", "vertical"):
        collapsed = {
            key: _median_or_none([row[channel].get(key) for row in rows])
            for key in (
                "bc_norm",
                "rl_norm",
                "bc_assignment_margin",
                "rl_assignment_margin",
                "cosine",
                "bc_yaw_delta",
                "rl_yaw_delta",
            )
        }
        if channel == "bearing":
            bc_yaw = float(collapsed["bc_yaw_delta"] or 0.0)
            rl_yaw = float(collapsed["rl_yaw_delta"] or 0.0)
            collapsed["yaw_sign_consistent"] = bool(bc_yaw * rl_yaw > 0.0)
        channels[channel] = collapsed
    return {
        "classification": classify_failure_snapshot(channels, functional_floors),
        **channels,
    }


def _stage_names(anchor_labels: Sequence[str]) -> tuple[str, ...]:
    stages = []
    for label in anchor_labels:
        if label in {"post_divergence_1", "post_divergence_2"}:
            stage = "post_divergence"
        else:
            stage = str(label)
        if stage not in stages:
            stages.append(stage)
    return tuple(stages)


def build_failure_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    shared_primary_mechanisms: Sequence[str] = (),
    functional_floors: Mapping[str, float] | None = None,
    primary_stage: str | None = "preterminal",
) -> dict[str, Any]:
    """Collapse snapshots to independent tasks and apply conservative route gates."""

    by_repeat: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    by_repeat_stage: dict[tuple[str, int, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        task_id = str(row["task_id"])
        repeat = int(row.get("repeat", 0))
        by_repeat.setdefault((task_id, repeat), []).append(row)
        for stage in _stage_names(row.get("anchor_labels", ())):
            by_repeat_stage.setdefault((task_id, repeat, stage), []).append(row)

    by_task: dict[str, list[Mapping[str, Any]]] = {}
    for (task_id, _repeat), repeat_rows in by_repeat.items():
        by_task.setdefault(task_id, []).append(
            _aggregate_rows(repeat_rows, functional_floors)
        )
    by_task_stage: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for (task_id, _repeat, stage), repeat_rows in by_repeat_stage.items():
        by_task_stage.setdefault((task_id, stage), []).append(
            _aggregate_rows(repeat_rows, functional_floors)
        )

    all_anchor_task_categories = {
        task_id: _collapse_rows(task_rows, functional_floors)
        for task_id, task_rows in sorted(by_task.items())
    }
    stage_task_categories = {
        stage: {
            task: _collapse_rows(task_rows, functional_floors)
            for (task, candidate), task_rows in by_task_stage.items()
            if candidate == stage
        }
        for stage in (
            "pre_divergence",
            "divergence",
            "post_divergence",
            "closest_goal",
            "preterminal",
        )
    }
    if primary_stage is None:
        task_categories = all_anchor_task_categories
    else:
        if primary_stage not in stage_task_categories:
            raise ValueError(f"unknown primary coverage stage: {primary_stage}")
        task_categories = stage_task_categories[primary_stage]
    mechanisms = tuple(_MECHANISM_ROUTE)
    overall: dict[str, Any] = {}
    determinate_task_categories = {
        task: category
        for task, category in task_categories.items()
        if category in _CLASS_TO_MECHANISM
    }
    denominator = len(determinate_task_categories)
    for mechanism in mechanisms:
        target_class = next(
            category for category, mapped in _CLASS_TO_MECHANISM.items() if mapped == mechanism
        )
        numerator = sum(
            category == target_class for category in determinate_task_categories.values()
        )
        overall[mechanism] = support_summary(numerator, denominator)

    stages = (
        "pre_divergence",
        "divergence",
        "post_divergence",
        "closest_goal",
        "preterminal",
    )
    stage_coverage: dict[str, Any] = {}
    progression: dict[str, Any] = {}
    for mechanism in mechanisms:
        target_class = next(
            category for category, mapped in _CLASS_TO_MECHANISM.items() if mapped == mechanism
        )
        stage_coverage[mechanism] = {}
        progression_rows = []
        for stage in stages:
            collapsed = stage_task_categories[stage]
            determinate = {
                task: category
                for task, category in collapsed.items()
                if category in _CLASS_TO_MECHANISM
            }
            numerator = sum(value == target_class for value in determinate.values())
            stage_coverage[mechanism][stage] = (
                support_summary(numerator, len(determinate)) if collapsed else None
            )
            progression_rows.extend(
                {
                    "task_id": task,
                    "stage": stage,
                    "mechanism": category == target_class,
                    "determinate": category in _CLASS_TO_MECHANISM,
                }
                for task, category in collapsed.items()
            )
        progression[mechanism] = exact_progression_test(progression_rows)

    primary = [
        mechanism
        for mechanism, report in overall.items()
        if report["support_level"] == "PRIMARY_ROUTE_SUPPORTED"
    ]
    shared_conflict = sorted(set(primary).intersection(shared_primary_mechanisms))
    if len(primary) == 1 and not shared_conflict:
        mechanism = primary[0]
        status, route = _MECHANISM_ROUTE[mechanism]
        progressed = progression[mechanism]["progression_supported"]
        evidence_strength = (
            "PRIMARY_ROUTE_SUPPORTED_WITH_PROGRESSION"
            if progressed
            else "PRIMARY_ROUTE_SUPPORTED_WITHOUT_PROGRESSION"
        )
    else:
        mechanism = None
        status = "INCONCLUSIVE_MIXED_FAILURE_STATE_RESPONSE"
        route = "NO_TRAINING_ROUTE_SELECTED"
        evidence_strength = "NO_PRIMARY_ROUTE"
    verdict = {
        "schema": "routea_d1f_verdict_v2",
        "status": status,
        "primary_mechanism": mechanism,
        "route": route,
        "evidence_strength": evidence_strength,
        "shared_primary_conflicts": shared_conflict,
        "evidence_scope": "lightweight_kinematic_network_forward",
        "next_allowed_action": (
            "design the route-specific next diagnostic or constrained training proposal; obtain approval before training"
            if mechanism is not None
            else "inspect mixed mechanisms and fixed-endpoint coverage; do not select a training route"
        ),
        "forbidden_actions": [
            "start retraining from D1-F alone",
            "claim a specific reward or critic cause",
            "claim Gazebo/PX4 closed-loop capability",
            "replace the fixed pre_divergence to preterminal progression test",
        ],
    }
    return {
        "schema": "routea_d1f_failure_summary_v2",
        "independent_task_count": len(task_categories),
        "determinate_task_count": denominator,
        "indeterminate_task_count": len(task_categories) - denominator,
        "primary_coverage_stage": primary_stage or "all_registered_anchors_descriptive",
        "task_classification": task_categories,
        "all_anchor_task_classification": all_anchor_task_categories,
        "overall_coverage": overall,
        "stage_coverage": stage_coverage,
        "progression": progression,
        "progression_contract": {
            "pre_stage": "pre_divergence",
            "late_stage": "preterminal",
            "intermediate_stages_role": "DESCRIPTIVE_ONLY",
            "fallback_allowed": False,
            "indeterminate_endpoint_policy": "EXCLUDE_FROM_PAIRED_DENOMINATOR",
        },
        "verdict": verdict,
    }
