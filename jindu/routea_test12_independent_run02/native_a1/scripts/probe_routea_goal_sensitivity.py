#!/usr/bin/env python3
"""Read-only B0→B1→B2-sentinel and D1-F diagnostics for Route A RL v1.7."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.contracts import ActionBundle, ActionSpec
from diffusion_policy_3d.env.traj_utils import action_to_world, normalize_angle
from scripts.audit_checkpoint_source_identity import build_inventory
from scripts.analyze_dppo_u32_trajectory_repeats import (
    obstacle_clearance,
    parse_trace_csv,
)
from scripts.routea_causal_diagnostics.bank import (
    STRATA,
    build_stratum_manifest,
    freeze_sentinel_subset,
    load_fixed32_parent,
    select_stratified_tasks,
)
from scripts.routea_causal_diagnostics.contracts import (
    DiagnosticFailure,
    build_diagnostic_source_manifest,
    canonical_array_sha256,
    canonical_tree_sha256,
    capture_rng_state,
    load_numeric_npz,
    load_strict_json,
    restore_rng_state,
    rng_state_sha256,
    write_numeric_npz,
    write_strict_json,
)
from scripts.routea_causal_diagnostics.goals import (
    GOAL_VARIANTS,
    make_goal_variant,
    reconstruct_goal_history,
    verify_goal_only_mutation,
)
from scripts.routea_causal_diagnostics.failure_state import (
    anchor_decisions,
    build_bc_repeat_null,
    build_failure_summary,
    build_replay_identity_diagnostics,
    build_shared_goal_response_reference,
    classify_failure_snapshot,
    endpoint_identity_audit,
    extract_decision_features,
    find_first_persistent_divergence,
    response_direction_metrics,
    serialize_bc_repeat_null,
)
from scripts.routea_causal_diagnostics.metrics import (
    action_geometry,
    bearing_jacobian,
    vertical_jacobian,
)
from scripts.routea_causal_diagnostics.reporting import (
    AtomicOutputDirectory,
    D1F_SELECTION_BOUND_FILES,
    d1f_selection_identity,
    file_sha256,
    finalize_d1f_outputs,
    finalize_outputs,
    publish_stage_manifest,
    verify_sha256_manifest,
    write_csv_rows,
    write_jsonl,
)
from scripts.routea_causal_diagnostics.runtime import (
    compare_checkpoint_contracts,
    inspect_checkpoint_contract,
    install_read_only_tripwires,
    load_read_only_system,
    paired_forward,
    validate_zero_effect_lora_state,
    verify_forward_pairing,
)


EXIT_CODES = {
    "SNAPSHOT_REPLAY_MISMATCH": 10,
    "CHECKPOINT_CONTRACT_MISMATCH": 11,
    "INVALID_NOISE_PAIRING": 12,
    "INVALID_OBSERVATION_MUTATION": 12,
    "INVALID_STATE_MUTATION": 13,
    "MEASUREMENT_INVALID": 14,
    "FUNCTIONAL_REFERENCE_INVALID": 14,
}


def _failure(status: str, detail: str, exit_code: int | None = None) -> DiagnosticFailure:
    return DiagnosticFailure(status, detail, EXIT_CODES.get(status, 2) if exit_code is None else exit_code)


def _single_line(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False), flush=True)


def _load_episode_configs(config_dir: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    from scripts.evaluate_dppo_uav_formal_endpoint import canonical_episode_config

    index: dict[str, dict[str, Any]] = {}
    sources = []
    for split in ("rl_test", "layerA"):
        path = config_dir / f"episode_configs_{split}.json"
        data = load_strict_json(path)
        if not isinstance(data, Mapping):
            raise _failure("EPISODE_CONFIG_IDENTITY_MISMATCH", f"invalid {path}")
        count = 0
        for entries in data.values():
            if not isinstance(entries, list):
                raise _failure("EPISODE_CONFIG_IDENTITY_MISMATCH", f"invalid entries in {path}")
            for payload in entries:
                fixed = canonical_episode_config(payload)
                task_id = str(payload["episode_id"])
                if task_id in index:
                    raise _failure("EPISODE_CONFIG_IDENTITY_MISMATCH", f"duplicate {task_id}")
                index[task_id] = {"payload": dict(payload), "config_hash": fixed.config_hash}
                count += 1
        sources.append(
            {"split": split, "path": str(path.resolve()), "sha256": file_sha256(path), "episodes": count}
        )
    return index, sources


def _check_new_output_first(path: Path) -> None:
    if path.resolve().exists():
        raise _failure("OUTPUT_DIR_EXISTS", str(path.resolve()))


def _run_select_b0(args: argparse.Namespace) -> int:
    _check_new_output_first(args.output_dir)
    seed_dirs = [args.fixed32_seed1_dir, args.fixed32_seed2_dir, args.fixed32_seed3_dir]
    parent = load_fixed32_parent(
        seed_dirs, {"layerA": args.layera_geometry, "rl_test": args.rl_test_geometry}
    )
    selection = select_stratified_tasks(parent, max_tasks=8)
    config_index, config_sources = _load_episode_configs(args.episode_config_dir)

    seed1_request = load_strict_json(Path(seed_dirs[0]) / "replay_request.json")
    seed1_verification = load_strict_json(Path(seed_dirs[0]) / "replay_verification.json")
    bc_checks = {
        (str(check["task_id"]), int(check["repeat"]) + 1): check
        for check in seed1_verification["checks"]
        if check["policy"] == "bc"
    }
    obstacle_sources = seed1_request["resolved_inputs"]["obstacle_sources"]
    manifest = build_stratum_manifest(
        parent,
        selection,
        parent_sources=[
            {
                "seed": index,
                "directory": str(Path(directory).resolve()),
                "request_sha256": file_sha256(Path(directory) / "replay_request.json"),
                "verification_sha256": file_sha256(Path(directory) / "replay_verification.json"),
                "csv_set_sha256": canonical_tree_sha256(
                    [
                        [path.name, file_sha256(path)]
                        for path in sorted(Path(directory).glob("*.csv"))
                    ]
                ),
            }
            for index, directory in enumerate(seed_dirs, start=1)
        ],
        geometry_sources=[
            {"world": world, **source} for world, source in sorted(obstacle_sources.items())
        ],
    )
    manifest["episode_config_sources"] = config_sources
    manifest["training_config_path"] = str(Path(args.seed1_config).resolve())
    manifest["training_config_sha256"] = file_sha256(args.seed1_config)
    for stratum in STRATA:
        enriched = []
        for item in selection[stratum]:
            if item.task_id not in config_index:
                raise _failure("EPISODE_CONFIG_IDENTITY_MISMATCH", item.task_id)
            config_entry = config_index[item.task_id]
            check = bc_checks.get((item.task_id, item.repeat))
            if check is None:
                raise _failure("PAIRED_IDENTITY_MISMATCH", f"missing BC check {item.task_id} r{item.repeat}")
            actual_hash = str(check["actual"]["config_hash"])
            if config_entry["config_hash"] != actual_hash:
                raise _failure(
                    "EPISODE_CONFIG_IDENTITY_MISMATCH",
                    f"{item.task_id}: bank={config_entry['config_hash']} verification={actual_hash}",
                )
            source = obstacle_sources[item.world]
            geometry_root = args.layera_geometry if source["split"] == "layerA" else args.rl_test_geometry
            geometry_path = Path(geometry_root) / Path(source["path"]).name
            enriched.append(
                {
                    **manifest["strata"][stratum]["selected"][len(enriched)],
                    "episode_config": config_entry["payload"],
                    "config_hash": actual_hash,
                    "bc_trace_path": item.parent.trace_paths[0],
                    "bc_trace_sha256": item.parent.bc_trace_sha256,
                    "expected_summary": check["actual"],
                    "geometry_path": str(geometry_path.resolve()),
                    "geometry_sha256": str(source["sha256"]),
                    "clear_line": bool(item.parent.clear_line),
                }
            )
        manifest["strata"][stratum]["selected"] = enriched
    source_manifest = build_diagnostic_source_manifest(REPO_ROOT)
    with AtomicOutputDirectory(args.output_dir) as staging:
        write_strict_json(staging / "stratum_manifest.json", manifest)
        publish_stage_manifest(staging, source_manifest)
    _single_line({"status": "SELECT_B0_COMPLETE", "output_dir": str(args.output_dir.resolve())})
    return 0


def _endpoint_done_reasons(seed_dirs: Sequence[Path]) -> tuple[dict[str, str], dict[tuple[str, int, str], Mapping[str, Any]]]:
    reasons: dict[str, str] = {}
    checks_by_identity: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for directory in seed_dirs:
        verification = load_strict_json(Path(directory) / "replay_verification.json")
        for check in verification["checks"]:
            path = str((Path(directory) / str(check["trajectory_file"])).resolve())
            actual = check["actual"]
            reasons[path] = str(actual["done_reason"])
            key = (
                str(check["task_id"]),
                int(check["repeat"]) + 1,
                str(check["policy"]),
            )
            if key in checks_by_identity and str(check["policy"]) == "rl":
                # RL identities repeat across seed directories.  The caller uses
                # this index only for the final (seed3) directory, replaced below.
                continue
            checks_by_identity[key] = check
    seed3_verification = load_strict_json(Path(seed_dirs[2]) / "replay_verification.json")
    for check in seed3_verification["checks"]:
        checks_by_identity[
            (str(check["task_id"]), int(check["repeat"]) + 1, str(check["policy"]))
        ] = check
    return reasons, checks_by_identity


def _run_select_d1f(args: argparse.Namespace) -> int:
    _check_new_output_first(args.output_dir)
    for directory in (args.trigger_final_dir, args.b1_dir, args.b2_dir):
        verify_sha256_manifest(directory)
    trigger = load_strict_json(Path(args.trigger_final_dir) / "verdict.json")
    if trigger.get("status") != "TRIGGER_D1F":
        raise _failure(
            "D1F_TRIGGER_MISMATCH",
            f"expected TRIGGER_D1F, got {trigger.get('status')}",
        )
    trigger_identity = load_strict_json(Path(args.trigger_final_dir) / "identity.json")
    expected_stage_manifests = trigger_identity.get("stage_manifest_sha256") or {}
    actual_stage_manifests = {
        "b1": file_sha256(Path(args.b1_dir) / "manifest_sha256.txt"),
        "b2": file_sha256(Path(args.b2_dir) / "manifest_sha256.txt"),
    }
    identity_mismatch = {
        name: {
            "expected": expected_stage_manifests.get(name),
            "actual": digest,
        }
        for name, digest in actual_stage_manifests.items()
        if expected_stage_manifests.get(name) != digest
    }
    if identity_mismatch:
        raise _failure(
            "D1F_TRIGGER_MISMATCH",
            f"trigger final does not bind supplied stages: {identity_mismatch}",
        )
    b2_summary = load_strict_json(Path(args.b2_dir) / "goal_sensitivity_summary.json")
    if b2_summary.get("status") != "SENTINEL_NORMAL_B1_NORMAL_TRIGGER_D1F":
        raise _failure(
            "D1F_TRIGGER_MISMATCH",
            f"unexpected B2 status: {b2_summary.get('status')}",
        )
    reference = build_shared_goal_response_reference(args.trigger_final_dir, args.b2_dir)
    if not reference["valid"]:
        raise _failure(
            "FUNCTIONAL_REFERENCE_INVALID", "; ".join(reference["invalid_reasons"])
        )
    combined_floor = float(reference["channels"]["combined_norm"]["functional_floor"])
    if not math.isclose(combined_floor, 0.05605613252672551, rel_tol=0.0, abs_tol=1e-15):
        raise _failure(
            "FUNCTIONAL_REFERENCE_INVALID",
            f"combined floor={combined_floor!r}; expected frozen v4 input 0.05605613252672551",
        )

    seed_dirs = [args.fixed32_seed1_dir, args.fixed32_seed2_dir, args.fixed32_seed3_dir]
    parent = load_fixed32_parent(
        seed_dirs, {"layerA": args.layera_geometry, "rl_test": args.rl_test_geometry}
    )
    done_reasons, checks = _endpoint_done_reasons(seed_dirs)
    identities, identity_audit = endpoint_identity_audit(parent, done_reasons)
    expected_counts = {
        "endpoint_break_repeat_count": 143,
        "endpoint_break_task_count": 41,
        "continuous_break_repeat_count": 148,
        "continuous_only_repeat_count": 5,
    }
    mismatches = {
        key: {"expected": expected, "actual": identity_audit.get(key)}
        for key, expected in expected_counts.items()
        if identity_audit.get(key) != expected
    }
    if mismatches:
        raise _failure("ENDPOINT_IDENTITY_MISMATCH", json.dumps(mismatches, sort_keys=True))

    config_index, config_sources = _load_episode_configs(args.episode_config_dir)
    seed1_request = load_strict_json(Path(seed_dirs[0]) / "replay_request.json")
    obstacle_sources = seed1_request["resolved_inputs"]["obstacle_sources"]
    parent_by_task: dict[str, list[Any]] = {}
    for row in parent:
        parent_by_task.setdefault(row.task_id, []).append(row)
    trace_cache: dict[str, Any] = {}

    def trace(path: str) -> Any:
        if path not in trace_cache:
            trace_cache[path] = parse_trace_csv(path)
        return trace_cache[path]

    task_nulls: dict[str, Any] = {}
    selected_rows = []
    for task_id in sorted({item.task_id for item in identities}):
        repeats = {}
        for row in sorted(parent_by_task[task_id], key=lambda item: item.repeat):
            bc_trace = trace(row.trace_paths[0])
            repeats[int(row.repeat)] = extract_decision_features(
                bc_trace.planned, start=row.start, goal=row.goal
            )
        task_nulls[task_id] = build_bc_repeat_null(repeats)

    for item in identities:
        config_entry = config_index.get(item.task_id)
        if config_entry is None:
            raise _failure("EPISODE_CONFIG_IDENTITY_MISMATCH", item.task_id)
        check = checks.get((item.task_id, item.repeat, "rl"))
        if check is None:
            raise _failure(
                "ENDPOINT_IDENTITY_MISMATCH", f"missing seed3 check {item.task_id} r{item.repeat}"
            )
        if str(check["actual"]["config_hash"]) != config_entry["config_hash"]:
            raise _failure(
                "EPISODE_CONFIG_IDENTITY_MISMATCH", f"config hash differs {item.task_id}"
            )
        bc_trace = trace(item.bc_trace_path)
        seed3_trace = trace(item.seed3_trace_path)
        start = np.asarray(seed3_trace.header["start"], dtype=np.float64)
        goal = np.asarray(seed3_trace.header["goal"], dtype=np.float64)
        bc_features = extract_decision_features(bc_trace.planned, start=start, goal=goal)
        seed3_features = extract_decision_features(seed3_trace.planned, start=start, goal=goal)
        k_div = find_first_persistent_divergence(
            bc_features, seed3_features, task_nulls[item.task_id]
        )
        closest = min(
            range(len(seed3_features)),
            key=lambda index: (seed3_features[index]["goal_distance"], index),
        )
        anchors = anchor_decisions(
            k_div, decision_count=len(seed3_features), closest_index=closest
        )
        source = obstacle_sources[item.world]
        geometry_root = (
            args.layera_geometry if source["split"] == "layerA" else args.rl_test_geometry
        )
        geometry_path = Path(geometry_root) / Path(source["path"]).name
        selected_rows.append(
            {
                **asdict(item),
                "episode_config": config_entry["payload"],
                "config_hash": config_entry["config_hash"],
                "expected_summary": check["actual"],
                "geometry_path": str(geometry_path.resolve()),
                "geometry_sha256": str(source["sha256"]),
                "k_div": k_div,
                "decision_count": len(seed3_features),
                "closest_index": closest,
                "initial_goal_distance_m": float(np.linalg.norm(goal - start)),
                "anchors": [
                    {"decision_index": index, "anchor_labels": list(labels)}
                    for index, labels in anchors.items()
                ],
            }
        )
    selection = {
        "schema": "routea_d1f_selection_v1",
        "run_lineage_id": f"d1f-{uuid.uuid4().hex}",
        "actor_rerun": False,
        "frozen_trajectory_policy": "seed3-u32",
        "trigger_status": trigger["status"],
        "endpoint_break_repeat_count": len(selected_rows),
        "endpoint_break_task_count": len({row["task_id"] for row in selected_rows}),
        "snapshot_anchor_count": sum(len(row["anchors"]) for row in selected_rows),
        "capture_config_path": str(Path(args.seed3_u32_config).resolve()),
        "capture_config_sha256": file_sha256(args.seed3_u32_config),
        "episode_config_sources": config_sources,
        "selected": selected_rows,
        "input_manifests": {
            "trigger_final": file_sha256(Path(args.trigger_final_dir) / "manifest_sha256.txt"),
            "b1": file_sha256(Path(args.b1_dir) / "manifest_sha256.txt"),
            "b2": file_sha256(Path(args.b2_dir) / "manifest_sha256.txt"),
        },
        "trigger_checkpoint_comparability_sha256": file_sha256(
            Path(args.trigger_final_dir) / "checkpoint_comparability.json"
        ),
    }
    source_manifest = build_diagnostic_source_manifest(REPO_ROOT)
    with AtomicOutputDirectory(args.output_dir) as staging:
        write_strict_json(staging / "endpoint_identity_audit.json", identity_audit)
        write_strict_json(staging / "bc_repeat_null.json", serialize_bc_repeat_null(task_nulls))
        write_strict_json(staging / "shared_goal_response_reference.json", reference)
        write_strict_json(staging / "failure_state_selection.json", selection)
        shutil.copy2(
            Path(args.trigger_final_dir) / "checkpoint_comparability.json",
            staging / "trigger_checkpoint_comparability.json",
        )
        publish_stage_manifest(staging, source_manifest)
    _single_line(
        {
            "status": "SELECT_D1F_COMPLETE",
            "endpoint_repeats": len(selected_rows),
            "tasks": len({row["task_id"] for row in selected_rows}),
            "snapshots": selection["snapshot_anchor_count"],
            "output_dir": str(args.output_dir.resolve()),
        }
    )
    return 0


def _repeat_identity(config_hash: str, repeat: int) -> tuple[int | None, int]:
    if repeat == 1:
        return None, 0
    digest = hashlib.sha256(f"{config_hash}:repeat:{repeat - 1}".encode("ascii")).hexdigest()
    return int(digest[:8], 16), int(digest[8:16], 16)


def _policy_seed(config_hash: str, decision: int, offset: int) -> int:
    base = int(hashlib.sha256(f"{config_hash}:{decision}".encode("ascii")).hexdigest()[:8], 16)
    return (base + int(offset)) & 0xFFFFFFFF


def _minimum_path_clearance(points: np.ndarray, obstacles: Sequence[Mapping[str, Any]]) -> float:
    if not obstacles:
        return float("inf")
    return min(obstacle_clearance(point, dict(obstacle)) for point in np.asarray(points) for obstacle in obstacles)


_EXACT_YAW_SEARCH_ULP_LIMIT = 16_384


def _candidate_yaw_deltas(current_yaw: float, target_yaw: np.float32):
    """Yield nearby float32 deltas that can round to one frozen yaw."""

    target = np.float32(target_yaw)
    center = np.float32(normalize_angle(float(target) - float(current_yaw)))
    yield center

    target_previous = np.nextafter(target, np.float32(-np.inf), dtype=np.float32)
    target_next = np.nextafter(target, np.float32(np.inf), dtype=np.float32)
    delta_previous = np.nextafter(center, np.float32(-np.inf), dtype=np.float32)
    delta_next = np.nextafter(center, np.float32(np.inf), dtype=np.float32)
    target_spacing = max(
        abs(float(target) - float(target_previous)),
        abs(float(target_next) - float(target)),
    )
    delta_spacings = tuple(
        spacing
        for spacing in (
            abs(float(center) - float(delta_previous)),
            abs(float(delta_next) - float(center)),
        )
        if spacing > 0.0
    )
    if not delta_spacings:
        raise ValueError("cannot enumerate finite float32 yaw deltas")
    search_radius = int(math.ceil(target_spacing / min(delta_spacings))) + 4
    if search_radius > _EXACT_YAW_SEARCH_ULP_LIMIT:
        raise ValueError(
            "exact yaw inverse exceeds the registered float32 search bound: "
            f"required_ulps={search_radius} limit={_EXACT_YAW_SEARCH_ULP_LIMIT}"
        )

    lower = center
    upper = center
    for _ in range(search_radius):
        lower = np.nextafter(lower, np.float32(-np.inf), dtype=np.float32)
        upper = np.nextafter(upper, np.float32(np.inf), dtype=np.float32)
        yield lower
        yield upper


def _exact_frozen_yaw_deltas(target_yaws: np.ndarray, initial_yaw: float) -> np.ndarray:
    """Invert an H8 yaw sequence under ``action_to_world`` bitwise semantics."""

    targets = np.ascontiguousarray(target_yaws, dtype=np.float32)
    target_bits = targets.view(np.uint32)

    def solve(index: int, current_yaw: float):
        if index == len(targets):
            return []
        for delta in _candidate_yaw_deltas(current_yaw, targets[index]):
            next_yaw = normalize_angle(float(current_yaw) + float(delta))
            if np.float32(next_yaw).view(np.uint32) != target_bits[index]:
                continue
            tail = solve(index + 1, next_yaw)
            if tail is not None:
                return [delta, *tail]
        return None

    solution = solve(0, float(initial_yaw))
    if solution is None:
        raise ValueError("frozen yaw sequence has no exact float32 delta inverse")
    return np.asarray(solution, dtype=np.float32)


def _assert_bitwise_trajectory_match(
    *,
    task_id: str,
    rebuilt_planned: np.ndarray,
    frozen_planned: np.ndarray,
    rebuilt_executed: np.ndarray,
    frozen_executed: np.ndarray,
) -> None:
    """Reject any replay table difference, including one-ULP float drift."""

    for table_name, rebuilt_value, frozen_value in (
        ("planned", rebuilt_planned, frozen_planned),
        ("executed", rebuilt_executed, frozen_executed),
    ):
        rebuilt = np.ascontiguousarray(rebuilt_value, dtype=np.float64)
        frozen = np.ascontiguousarray(frozen_value, dtype=np.float64)
        if rebuilt.shape != frozen.shape:
            raise _failure(
                "SNAPSHOT_REPLAY_MISMATCH",
                f"trajectory shape {task_id} {table_name}={rebuilt.shape}/{frozen.shape}",
            )
        mismatch = rebuilt.view(np.uint64) != frozen.view(np.uint64)
        if np.any(mismatch):
            row, column = np.argwhere(mismatch)[0]
            raise _failure(
                "SNAPSHOT_REPLAY_MISMATCH",
                "trajectory bitwise mismatch "
                f"{task_id} table={table_name} row={int(row)} column={int(column)} "
                f"count={int(np.count_nonzero(mismatch))}",
            )


def _frozen_plan_action(
    planned_rows: np.ndarray,
    *,
    decision_step: int,
    before: Mapping[str, Any],
    exact_bitwise: bool = False,
) -> ActionBundle:
    """Invert one frozen world-frame H8 plan into the unchanged 8x11 policy chunk.

    The fixed32 CSV is the authoritative trajectory that was actually collected.
    Re-running the actor in isolation is not an equivalent replay because the
    original collector evaluated each cell inside a long, ordered process.  For
    B0 we need the observations *on that frozen trajectory*, not a newly inferred
    approximation.  The legacy mode preserves B0's registered 1e-6 trajectory
    tolerance.  D1-F opts into ``exact_bitwise`` so the H8 yaw deltas are solved
    against ``action_to_world`` accumulation and all 11 stored columns are
    reproduced bitwise.
    """

    rows = np.asarray(planned_rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 14 or not np.isfinite(rows).all():
        raise ValueError("frozen planned table must be finite with shape (N,14)")
    selected = rows[np.isclose(rows[:, 0], float(decision_step), rtol=0.0, atol=0.0)]
    selected = selected[np.argsort(selected[:, 1])]
    if len(selected) != 8 or tuple(selected[:, 1].astype(int)) != tuple(range(8)):
        raise ValueError(f"decision {decision_step} does not contain exactly k=0..7")

    world = np.ascontiguousarray(selected[:, 3:14], dtype=np.float32)
    raw = np.zeros((8, 11), dtype=np.float32)
    previous_position = np.asarray(before["position"], dtype=np.float32)
    for index in range(8):
        raw[index, :3] = world[index, :3] - previous_position
        raw[index, 3:9] = world[index, 3:9]
        raw[index, 10] = world[index, 10]
        previous_position = world[index, :3]
    if exact_bitwise:
        raw[:, 9] = _exact_frozen_yaw_deltas(world[:, 9], float(before["yaw"]))
        rebuilt = action_to_world(
            raw,
            np.asarray(before["position"], dtype=np.float32),
            np.asarray(before["velocity_reference"], dtype=np.float32),
            float(before["yaw"]),
        )
        if not np.array_equal(rebuilt.view(np.uint32), world.view(np.uint32)):
            mismatch = np.argwhere(rebuilt.view(np.uint32) != world.view(np.uint32))[0]
            raise ValueError(
                "frozen plan has no bitwise exact delta inverse: "
                f"waypoint={int(mismatch[0])} column={int(mismatch[1])}"
            )
    else:
        previous_yaw = float(before["yaw"])
        for index in range(8):
            raw[index, 9] = normalize_angle(float(world[index, 9]) - previous_yaw)
            previous_yaw = float(world[index, 9])

    spec = ActionSpec.e5()
    full = np.zeros((1, spec.full_horizon, spec.action_dim), dtype=np.float32)
    full[:, spec.action_start : spec.action_start + spec.policy_horizon, :] = raw
    return ActionBundle.from_full(full, spec)


def _capture_one(
    system: Any,
    config: Any,
    selected: Mapping[str, Any],
    *,
    trajectory_policy: str = "bc",
    strict_bitwise_replay: bool = False,
) -> list[dict[str, Any]]:
    from scripts.endpoint_trajectory_trace_v2 import TraceAdapter
    from scripts.evaluate_dppo_uav_formal_endpoint import (
        _build_adapter,
        canonical_done_reason,
        canonical_episode_config,
        reset_fixed_episode,
    )

    geometry_path = Path(selected["geometry_path"])
    if file_sha256(geometry_path) != selected["geometry_sha256"]:
        raise _failure("SNAPSHOT_REPLAY_MISMATCH", f"geometry changed: {geometry_path}")
    geometry = load_strict_json(geometry_path)
    obstacles = geometry["obstacles"]
    if trajectory_policy == "bc":
        trace_path_field, trace_sha_field, header_policy = (
            "bc_trace_path",
            "bc_trace_sha256",
            "bc",
        )
    elif trajectory_policy == "seed3-u32":
        trace_path_field, trace_sha_field, header_policy = (
            "seed3_trace_path",
            "seed3_trace_sha256",
            "rl",
        )
    else:
        raise _failure("SNAPSHOT_REPLAY_MISMATCH", f"unsupported trajectory policy {trajectory_policy}")
    frozen_path = Path(selected[trace_path_field])
    if file_sha256(frozen_path) != selected[trace_sha_field]:
        raise _failure("SNAPSHOT_REPLAY_MISMATCH", f"frozen trace changed: {frozen_path}")
    frozen = parse_trace_csv(frozen_path)
    if (
        frozen.header.get("task_id") != selected["task_id"]
        or frozen.header.get("policy") != header_policy
        or int(frozen.header.get("repeat", -1)) != int(selected["repeat"])
    ):
        raise _failure(
            "SNAPSHOT_REPLAY_MISMATCH",
            f"frozen trace identity mismatch {selected['task_id']}",
        )
    fixed = canonical_episode_config(selected["episode_config"])
    repeat = int(selected["repeat"])
    derived_reset, action_offset = _repeat_identity(fixed.config_hash, repeat)
    reset_seed = int(selected["env_reset_seed"])
    if derived_reset is not None and derived_reset != reset_seed:
        raise _failure(
            "SNAPSHOT_REPLAY_MISMATCH",
            f"repeat seed mismatch {selected['task_id']}: expected={derived_reset} actual={reset_seed}",
        )
    adapter = _build_adapter(
        system,
        config,
        selected["world"],
        geometry_path,
        selected["role"],
    )
    trace = TraceAdapter(adapter, None, {"task_id": selected["task_id"]})
    observation, context = reset_fixed_episode(
        adapter.environment,
        fixed.payload,
        seed=reset_seed,
        reset_callable=trace.reset,
    )
    decisions: list[dict[str, Any]] = []
    last_provenance = None
    last_provenance_hash = None
    terminal_reason = None
    decision = 0
    try:
        while True:
            before_position = adapter.environment._drone_pos.copy()
            before_velocity = adapter.environment._drone_vel.copy()
            before_goal = adapter.environment.goal_w.copy()
            before_yaw = float(adapter.environment._drone_yaw)
            raw = {
                "agent_pos": np.asarray(observation["agent_pos"]).copy(),
                "point_cloud": np.asarray(observation["point_cloud"]).copy(),
            }
            metadata_history = [
                asdict(item) for item in context["pointcloud_meta_history"]
            ]
            episode_identity_hash = str(context["episode_identity_hash"])
            context_decision_index = int(context["decision_index"])
            seed = _policy_seed(fixed.config_hash, decision, action_offset)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            with torch.no_grad():
                encoded = system.observation_adapter.encode(
                    observation,
                    context["pointcloud_meta_history"],
                    episode_identity_hash=context["episode_identity_hash"],
                    decision_index=context["decision_index"],
                )
                system.observation_adapter.verify(encoded)
            provenance = encoded.provenance
            last_provenance = provenance
            last_provenance_hash = getattr(provenance, "policy_input_hash", None)
            try:
                action = _frozen_plan_action(
                    frozen.planned,
                    decision_step=decision + 1,
                    before={
                        "position": before_position,
                        "velocity_reference": before_velocity,
                        "yaw": before_yaw,
                    },
                    exact_bitwise=strict_bitwise_replay,
                )
            except ValueError as exc:
                raise _failure(
                    "SNAPSHOT_REPLAY_MISMATCH",
                    f"frozen plan invalid {selected['task_id']}: {exc}",
                ) from exc
            trace.policy_inputs.append(
                {"action_seed": seed, "provenance": None if provenance is None else asdict(provenance)}
            )
            next_observation, _, terminated, truncated, info = trace.step(action)
            executed = np.asarray(info.get("executed_path"), dtype=np.float64)
            decisions.append(
                {
                    "decision_index": decision,
                    "agent_pos": raw["agent_pos"],
                    "point_cloud": raw["point_cloud"],
                    "position_world": before_position,
                    "goal_world": before_goal,
                    "yaw": before_yaw,
                    "policy_seed": seed,
                    "pointcloud_meta_history": metadata_history,
                    "episode_identity_hash": episode_identity_hash,
                    "context_decision_index": context_decision_index,
                    "goal_distance": float(np.linalg.norm(before_goal - before_position)),
                    "clearance": _minimum_path_clearance(executed[1:], obstacles),
                    "policy_input_hash": last_provenance_hash,
                }
            )
            decision += 1
            if terminated or truncated:
                facts = info.get("transition_facts")
                raw_reason = getattr(facts, "raw_done_reason", None) or getattr(
                    adapter.environment, "_done_reason", None
                )
                terminal_reason = canonical_done_reason(
                    raw_reason, adapter.environment._drone_pos[2], terminated, truncated
                )
                break
            observation, context = next_observation, info
    finally:
        trace.close()

    planned = np.asarray(trace.planned_rows, dtype=np.float64)
    executed = np.asarray(trace.executed_rows, dtype=np.float64)
    if strict_bitwise_replay:
        _assert_bitwise_trajectory_match(
            task_id=selected["task_id"],
            rebuilt_planned=planned,
            frozen_planned=frozen.planned,
            rebuilt_executed=executed,
            frozen_executed=frozen.executed,
        )
    elif not (
        planned.shape == frozen.planned.shape
        and executed.shape == frozen.executed.shape
        and np.allclose(planned, frozen.planned, rtol=1e-6, atol=1e-6)
        and np.allclose(executed, frozen.executed, rtol=1e-6, atol=1e-6)
    ):
        raise _failure(
            "SNAPSHOT_REPLAY_MISMATCH",
            f"trajectory numeric mismatch {selected['task_id']}",
        )
    expected = selected["expected_summary"]
    if terminal_reason != expected["done_reason"] or decision != int(expected["steps"]):
        raise _failure(
            "SNAPSHOT_REPLAY_MISMATCH",
            f"terminal mismatch {selected['task_id']} reason={terminal_reason}/{expected['done_reason']} steps={decision}/{expected['steps']}",
        )
    expected_last = (expected.get("observation_validity_summary") or {}).get("last_policy_input_hash")
    if expected_last is not None and last_provenance_hash != expected_last:
        diagnostics = build_replay_identity_diagnostics(
            task_id=selected["task_id"],
            expected_observation=expected.get("observation_validity_summary") or {},
            actual_provenance=last_provenance,
            rebuilt_planned=planned,
            frozen_planned=frozen.planned,
        )
        raise _failure(
            "SNAPSHOT_REPLAY_MISMATCH",
            json.dumps(diagnostics, sort_keys=True, separators=(",", ":")),
        )
    if trajectory_policy == "seed3-u32":
        labels = {
            int(anchor["decision_index"]): list(anchor["anchor_labels"])
            for anchor in selected["anchors"]
        }
    else:
        early = 0
        closest = min(range(len(decisions)), key=lambda i: (decisions[i]["goal_distance"], i))
        clearance = min(range(len(decisions)), key=lambda i: (decisions[i]["clearance"], i))
        labels: dict[int, list[str]] = {}
        for label, index in (("early", early), ("closest", closest), ("clearance", clearance)):
            labels.setdefault(index, []).append(label)
    return [{**decisions[index], "anchor_labels": labels[index]} for index in sorted(labels)]


def _run_capture_b0(args: argparse.Namespace) -> int:
    _check_new_output_first(args.output_dir)
    verify_sha256_manifest(args.selection_dir)
    selection = load_strict_json(Path(args.selection_dir) / "stratum_manifest.json")
    if file_sha256(args.config) != selection["training_config_sha256"]:
        raise _failure("STAGE_INPUT_HASH_MISMATCH", "capture config differs from select-b0")
    config, system = load_read_only_system(args.config, None, REPO_ROOT)
    validate_zero_effect_lora_state(system.lora_bundle.adapter_state_dict())
    outer_rng = capture_rng_state()
    guard = install_read_only_tripwires(system)
    snapshot_rows: list[dict[str, Any]] = []
    try:
        for stratum in STRATA:
            for selected in selection["strata"][stratum]["selected"]:
                for row in _capture_one(system, config, selected):
                    snapshot_rows.append({"stratum": stratum, "selected": selected, **row})
    finally:
        restore_rng_state(outer_rng)
    tripwire = guard.verify(require_rng_unchanged=True)

    arrays: dict[str, list[Any]] = {
        name: []
        for name in (
            "agent_pos",
            "point_cloud",
            "position_world",
            "goal_world",
            "yaw",
            "decision_index",
            "task_index",
            "stratum_code",
            "anchor_mask",
            "reset_seed",
            "policy_seed",
        )
    }
    tasks = sorted({row["selected"]["task_id"] for row in snapshot_rows})
    task_index = {task: index for index, task in enumerate(tasks)}
    manifest_rows = []
    anchor_bits = {"early": 1, "closest": 2, "clearance": 4}
    for bank_index, row in enumerate(snapshot_rows):
        selected = row["selected"]
        mask = sum(anchor_bits[label] for label in row["anchor_labels"])
        arrays["agent_pos"].append(row["agent_pos"])
        arrays["point_cloud"].append(row["point_cloud"])
        arrays["position_world"].append(row["position_world"])
        arrays["goal_world"].append(row["goal_world"])
        arrays["yaw"].append(row["yaw"])
        arrays["decision_index"].append(row["decision_index"])
        arrays["task_index"].append(task_index[selected["task_id"]])
        arrays["stratum_code"].append(STRATA.index(row["stratum"]))
        arrays["anchor_mask"].append(mask)
        arrays["reset_seed"].append(selected["env_reset_seed"])
        arrays["policy_seed"].append(row["policy_seed"])
        manifest_rows.append(
            {
                "bank_index": bank_index,
                "snapshot_id": f"{row['stratum']}::{selected['task_id']}::r{selected['repeat']}::d{row['decision_index']}",
                "stratum": row["stratum"],
                "task_id": selected["task_id"],
                "repeat": selected["repeat"],
                "env_reset_seed": selected["env_reset_seed"],
                "decision_index": row["decision_index"],
                "anchor_labels": row["anchor_labels"],
                "anchor_kind": row["anchor_labels"][0],
                "anchor_mask": mask,
                "policy_seed": row["policy_seed"],
                "policy_input_hash": row["policy_input_hash"],
                "pointcloud_meta_history": row["pointcloud_meta_history"],
                "episode_identity_hash": row["episode_identity_hash"],
                "context_decision_index": row["context_decision_index"],
                "agent_pos_sha256": canonical_array_sha256(row["agent_pos"]),
                "point_cloud_sha256": canonical_array_sha256(row["point_cloud"]),
            }
        )
    if not manifest_rows:
        raise _failure("EMPTY_OBSERVATION_BANK", "no selected strata produced snapshots")
    numeric = {
        "agent_pos": np.asarray(arrays["agent_pos"]),
        "point_cloud": np.asarray(arrays["point_cloud"]),
        "position_world": np.asarray(arrays["position_world"], dtype=np.float32),
        "goal_world": np.asarray(arrays["goal_world"], dtype=np.float32),
        "yaw": np.asarray(arrays["yaw"], dtype=np.float32),
        "decision_index": np.asarray(arrays["decision_index"], dtype=np.int32),
        "task_index": np.asarray(arrays["task_index"], dtype=np.int32),
        "stratum_code": np.asarray(arrays["stratum_code"], dtype=np.int8),
        "anchor_mask": np.asarray(arrays["anchor_mask"], dtype=np.uint8),
        "reset_seed": np.asarray(arrays["reset_seed"], dtype=np.int64),
        "policy_seed": np.asarray(arrays["policy_seed"], dtype=np.int64),
    }
    sentinel_input = []
    for row in manifest_rows:
        for anchor in row["anchor_labels"]:
            sentinel_input.append({**row, "anchor_kind": anchor})
    sentinel = [asdict(entry) for entry in freeze_sentinel_subset(sentinel_input)]
    source_manifest = build_diagnostic_source_manifest(REPO_ROOT)
    with AtomicOutputDirectory(args.output_dir) as staging:
        shutil.copy2(Path(args.selection_dir) / "stratum_manifest.json", staging / "stratum_manifest.json")
        npz_manifest = write_numeric_npz(
            staging / "observation_bank.npz", numeric, max_total_bytes=2 * 1024**3
        )
        reloaded = load_numeric_npz(
            staging / "observation_bank.npz",
            expected_shapes={name: value.shape for name, value in numeric.items()},
            max_total_bytes=2 * 1024**3,
        )
        for name in numeric:
            if canonical_array_sha256(numeric[name]) != canonical_array_sha256(reloaded[name]):
                raise _failure("INVALID_NUMERIC_ARTIFACT", f"reload hash mismatch: {name}")
        write_strict_json(
            staging / "observation_bank_manifest.json",
            {
                "schema": "routea_b0_observation_bank_v1",
                "entry_count": len(manifest_rows),
                "task_count": len(tasks),
                "tasks": tasks,
                "strata": list(STRATA),
                "anchor_labels": ["early", "closest", "clearance"],
                "snapshots": manifest_rows,
                "sentinel_subset": sentinel,
                "npz": npz_manifest,
                "selection_manifest_sha256": file_sha256(Path(args.selection_dir) / "stratum_manifest.json"),
                "replay_contract": {
                    "mode": "frozen_trajectory_observation_reconstruction",
                    "trajectory_rtol": 1e-6,
                    "trajectory_atol": 1e-6,
                    "last_policy_input_hash_exact": True,
                    "source_pack_has_all_decision_hashes": False,
                    "actor_rerun": False,
                    "rationale": "fixed32 CSV is authoritative; isolated actor inference does not preserve the original collector's ordered process context",
                    "limitation": "trajectory-only fixed32 pack records only the last policy-input hash; every reconstructed planned/executed row is checked, while earlier decision hashes have no frozen comparator",
                },
            },
        )
        write_strict_json(staging / "tripwire_verification.json", tripwire)
        publish_stage_manifest(staging, source_manifest)
    _single_line({"status": "CAPTURE_B0_COMPLETE", "entries": len(manifest_rows), "output_dir": str(args.output_dir.resolve())})
    return 0


def _run_capture_d1f(args: argparse.Namespace) -> int:
    _check_new_output_first(args.output_dir)
    verify_sha256_manifest(args.selection_dir)
    source_manifest = build_diagnostic_source_manifest(REPO_ROOT)
    selection_identity = d1f_selection_identity(args.selection_dir)
    if selection_identity["diagnostic_source_sha256"] != source_manifest["content_sha256"]:
        raise _failure(
            "STAGE_INPUT_HASH_MISMATCH",
            "selection and capture diagnostic source identities differ",
        )
    selection_path = Path(args.selection_dir) / "failure_state_selection.json"
    selection = load_strict_json(selection_path)
    if file_sha256(args.config) != selection["capture_config_sha256"]:
        raise _failure("STAGE_INPUT_HASH_MISMATCH", "capture config differs from select-d1f")
    config, system = load_read_only_system(args.config, None, REPO_ROOT)
    validate_zero_effect_lora_state(system.lora_bundle.adapter_state_dict())
    immutable = [
        Path(args.selection_dir) / name
        for name in (
            "failure_state_selection.json",
            "endpoint_identity_audit.json",
            "bc_repeat_null.json",
            "shared_goal_response_reference.json",
            "trigger_checkpoint_comparability.json",
        )
    ]
    outer_rng = capture_rng_state()
    guard = install_read_only_tripwires(system, immutable_files=immutable)
    snapshot_rows = []
    try:
        for selected in selection["selected"]:
            rows = _capture_one(
                system,
                config,
                selected,
                trajectory_policy="seed3-u32",
                strict_bitwise_replay=True,
            )
            for row in rows:
                snapshot_rows.append({"selected": selected, **row})
    finally:
        restore_rng_state(outer_rng)
    tripwire = guard.verify(require_rng_unchanged=True)
    if len(snapshot_rows) != int(selection["snapshot_anchor_count"]):
        raise _failure(
            "SNAPSHOT_REPLAY_MISMATCH",
            f"captured={len(snapshot_rows)} expected={selection['snapshot_anchor_count']}",
        )

    array_names = (
        "agent_pos",
        "point_cloud",
        "position_world",
        "goal_world",
        "yaw",
        "decision_index",
        "task_index",
        "repeat",
        "anchor_mask",
        "reset_seed",
        "policy_seed",
        "goal_distance",
    )
    arrays: dict[str, list[Any]] = {name: [] for name in array_names}
    tasks = sorted({row["selected"]["task_id"] for row in snapshot_rows})
    task_index = {task: index for index, task in enumerate(tasks)}
    anchor_bits = {
        "pre_divergence": 1,
        "divergence": 2,
        "post_divergence_1": 4,
        "post_divergence_2": 8,
        "closest_goal": 16,
        "preterminal": 32,
    }
    manifest_rows = []
    seen_ids = set()
    for bank_index, row in enumerate(snapshot_rows):
        selected = row["selected"]
        mask = sum(anchor_bits[label] for label in row["anchor_labels"])
        snapshot_id = (
            f"D1F::{selected['task_id']}::r{selected['repeat']}::d{row['decision_index']}"
        )
        if snapshot_id in seen_ids:
            raise _failure("INVALID_SNAPSHOT_MANIFEST", f"duplicate snapshot_id: {snapshot_id}")
        seen_ids.add(snapshot_id)
        arrays["agent_pos"].append(row["agent_pos"])
        arrays["point_cloud"].append(row["point_cloud"])
        arrays["position_world"].append(row["position_world"])
        arrays["goal_world"].append(row["goal_world"])
        arrays["yaw"].append(row["yaw"])
        arrays["decision_index"].append(row["decision_index"])
        arrays["task_index"].append(task_index[selected["task_id"]])
        arrays["repeat"].append(selected["repeat"])
        arrays["anchor_mask"].append(mask)
        arrays["reset_seed"].append(selected["env_reset_seed"])
        arrays["policy_seed"].append(row["policy_seed"])
        arrays["goal_distance"].append(row["goal_distance"])
        manifest_rows.append(
            {
                "bank_index": bank_index,
                "snapshot_id": snapshot_id,
                "task_id": selected["task_id"],
                "repeat": selected["repeat"],
                "env_reset_seed": selected["env_reset_seed"],
                "world": selected["world"],
                "split": selected["split"],
                "role": selected["role"],
                "decision_index": row["decision_index"],
                "k_div": selected["k_div"],
                "anchor_labels": row["anchor_labels"],
                "anchor_mask": mask,
                "goal_distance": row["goal_distance"],
                "initial_goal_distance_m": selected["initial_goal_distance_m"],
                "policy_seed": row["policy_seed"],
                "policy_input_hash": row["policy_input_hash"],
                "pointcloud_meta_history": row["pointcloud_meta_history"],
                "episode_identity_hash": row["episode_identity_hash"],
                "context_decision_index": row["context_decision_index"],
                "agent_pos_sha256": canonical_array_sha256(row["agent_pos"]),
                "point_cloud_sha256": canonical_array_sha256(row["point_cloud"]),
            }
        )
    numeric = {
        "agent_pos": np.asarray(arrays["agent_pos"]),
        "point_cloud": np.asarray(arrays["point_cloud"]),
        "position_world": np.asarray(arrays["position_world"], dtype=np.float32),
        "goal_world": np.asarray(arrays["goal_world"], dtype=np.float32),
        "yaw": np.asarray(arrays["yaw"], dtype=np.float32),
        "decision_index": np.asarray(arrays["decision_index"], dtype=np.int32),
        "task_index": np.asarray(arrays["task_index"], dtype=np.int32),
        "repeat": np.asarray(arrays["repeat"], dtype=np.int8),
        "anchor_mask": np.asarray(arrays["anchor_mask"], dtype=np.uint8),
        "reset_seed": np.asarray(arrays["reset_seed"], dtype=np.int64),
        "policy_seed": np.asarray(arrays["policy_seed"], dtype=np.int64),
        "goal_distance": np.asarray(arrays["goal_distance"], dtype=np.float32),
    }
    with AtomicOutputDirectory(args.output_dir) as staging:
        for name in D1F_SELECTION_BOUND_FILES:
            shutil.copy2(Path(args.selection_dir) / name, staging / name)
        npz_manifest = write_numeric_npz(
            staging / "failure_state_bank.npz", numeric, max_total_bytes=2 * 1024**3
        )
        reloaded = load_numeric_npz(
            staging / "failure_state_bank.npz",
            expected_shapes={name: value.shape for name, value in numeric.items()},
            max_total_bytes=2 * 1024**3,
        )
        for name, value in numeric.items():
            if canonical_array_sha256(value) != canonical_array_sha256(reloaded[name]):
                raise _failure("INVALID_NUMERIC_ARTIFACT", f"reload hash mismatch: {name}")
        write_strict_json(
            staging / "failure_state_bank_manifest.json",
            {
                "schema": "routea_d1f_failure_state_bank_v1",
                "entry_count": len(manifest_rows),
                "repeat_count": selection["endpoint_break_repeat_count"],
                "task_count": len(tasks),
                "tasks": tasks,
                "anchor_labels": list(anchor_bits),
                "snapshots": manifest_rows,
                "npz": npz_manifest,
                "selection_manifest_sha256": file_sha256(selection_path),
                "run_lineage_id": selection["run_lineage_id"],
                "selection_identity": selection_identity,
                "replay_contract": {
                    "mode": "seed3_u32_frozen_trajectory_observation_reconstruction",
                    "trajectory_comparison": "bitwise_exact",
                    "full_plan_bitwise_exact": True,
                    "full_plan_compared_column_count": 11,
                    "full_plan_compared_element_count": sum(
                        int(item["decision_count"]) * 8 * 11
                        for item in selection["selected"]
                    ),
                    "full_plan_mismatch_element_count": 0,
                    "planned_table_bitwise_exact": True,
                    "executed_table_bitwise_exact": True,
                    "terminal_steps_last_policy_input_hash_exact": True,
                    "actor_rerun": False,
                },
            },
        )
        write_strict_json(staging / "tripwire_verification.json", tripwire)
        publish_stage_manifest(staging, source_manifest)
    _single_line(
        {
            "status": "CAPTURE_D1F_COMPLETE",
            "entries": len(manifest_rows),
            "tasks": len(tasks),
            "output_dir": str(args.output_dir.resolve()),
        }
    )
    return 0


def _parse_checkpoint_arg(value: str) -> tuple[str, Path, Path]:
    parts = value.split("|", 2)
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError("checkpoint must be LABEL|CONFIG_PATH|CHECKPOINT_DIR")
    return parts[0], Path(parts[1]), Path(parts[2])


def _release_system(system: Any) -> None:
    del system
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _bank_observation(
    bank: Mapping[str, np.ndarray], index: int, snapshot: Mapping[str, Any]
) -> dict[str, Any]:
    from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.observation_provenance import (
        PointCloudGenerationMeta,
    )

    metadata = tuple(
        PointCloudGenerationMeta(**item)
        for item in snapshot["pointcloud_meta_history"]
    )
    return {
        "agent_pos": bank["agent_pos"][index],
        "point_cloud": bank["point_cloud"][index],
        "pointcloud_meta_history": metadata,
        "episode_identity_hash": str(snapshot["episode_identity_hash"]),
        "decision_index": int(snapshot["context_decision_index"]),
    }


def _original_variant(observation: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [{"name": "L0-original", "agent_pos": observation["agent_pos"]}]


def _run_b1(args: argparse.Namespace) -> int:
    _check_new_output_first(args.output_dir)
    verify_sha256_manifest(args.b0_dir)
    bank_path = Path(args.b0_dir) / "observation_bank.npz"
    bank_manifest = load_strict_json(Path(args.b0_dir) / "observation_bank_manifest.json")
    bank = load_numeric_npz(bank_path, max_total_bytes=2 * 1024**3)
    expected_labels = {"u0"}.union(
        f"seed{seed}-u{update}" for seed in (1, 2, 3) for update in (8, 16, 24, 32)
    )
    checkpoint_args = {
        label: (config_path, checkpoint_path)
        for label, config_path, checkpoint_path in args.checkpoint
    }
    if len(checkpoint_args) != len(args.checkpoint):
        raise _failure("CHECKPOINT_MATRIX_MISMATCH", "duplicate checkpoint label")
    if set(checkpoint_args) != expected_labels:
        raise _failure(
            "CHECKPOINT_MATRIX_MISMATCH",
            f"missing={sorted(expected_labels-set(checkpoint_args))} extra={sorted(set(checkpoint_args)-expected_labels)}",
        )
    contracts = [inspect_checkpoint_contract("bc", args.bc_config, None, REPO_ROOT)]
    for label in sorted(checkpoint_args, key=lambda name: (name != "u0", name)):
        config_path, checkpoint_path = checkpoint_args[label]
        contracts.append(
            inspect_checkpoint_contract(label, config_path, checkpoint_path, REPO_ROOT)
        )
    comparability = compare_checkpoint_contracts(contracts)
    source_manifest = build_diagnostic_source_manifest(REPO_ROOT)
    if comparability["status"] != "COMPARABLE":
        with AtomicOutputDirectory(args.output_dir) as staging:
            write_strict_json(staging / "checkpoint_comparability.json", comparability)
            publish_stage_manifest(staging, source_manifest)
        raise _failure("CHECKPOINT_CONTRACT_MISMATCH", "see checkpoint_comparability.json")

    # Freeze the measurement floor using BC duplicate forwards before any u0 forward.
    config, bc_system = load_read_only_system(args.bc_config, None, REPO_ROOT)
    bc_guard = install_read_only_tripwires(bc_system, immutable_files=[bank_path])
    calibration_abs: list[float] = []
    calibration_rel: list[float] = []
    bc_records: dict[int, Any] = {}
    try:
        for index, snapshot in enumerate(bank_manifest["snapshots"]):
            observation = _bank_observation(bank, index, snapshot)
            first = paired_forward(
                bc_system, observation, _original_variant(observation), int(snapshot["policy_seed"])
            )[0]
            second = paired_forward(
                bc_system, observation, _original_variant(observation), int(snapshot["policy_seed"])
            )[0]
            difference = np.abs(first.full_action.astype(np.float64) - second.full_action.astype(np.float64))
            denominator = np.maximum(np.abs(first.full_action.astype(np.float64)), 1e-12)
            calibration_abs.extend(difference.reshape(-1).tolist())
            calibration_rel.extend((difference / denominator).reshape(-1).tolist())
            bc_records[index] = first
        p99_abs = float(np.quantile(calibration_abs, 0.99))
        p99_rel = float(np.quantile(calibration_rel, 0.99))
        tol_abs = 0.0 if p99_abs == 0.0 else max(1e-6, 5.0 * p99_abs)
        tol_rel = 0.0 if p99_rel == 0.0 else max(1e-6, 5.0 * p99_rel)
        thresholds = {
            "schema": "routea_b1_frozen_thresholds_v1",
            "frozen_before_u0": True,
            "calibration_bank_sha256": file_sha256(bank_path),
            "bc_duplicate_p99_abs": p99_abs,
            "bc_duplicate_p99_rel": p99_rel,
            "tol_abs": tol_abs,
            "tol_rel": tol_rel,
            "bitwise_exact": p99_abs == 0.0 and p99_rel == 0.0,
        }
        tripwire_reports = {"bc": bc_guard.verify(require_rng_unchanged=True)}
    finally:
        _release_system(bc_system)

    rows: list[dict[str, Any]] = []
    rng_groups: dict[str, dict[str, list[str]]] = {
        str(index): {"initial": [], "post": []}
        for index in range(len(bank_manifest["snapshots"]))
    }
    label_specs = [("bc", args.bc_config, None)] + [
        (label, *checkpoint_args[label]) for label in sorted(checkpoint_args, key=lambda name: (name != "u0", name))
    ]
    for label, config_path, checkpoint_path in label_specs:
        _, system = load_read_only_system(config_path, checkpoint_path, REPO_ROOT)
        guard = install_read_only_tripwires(system, immutable_files=[bank_path])
        try:
            for index, snapshot in enumerate(bank_manifest["snapshots"]):
                observation = _bank_observation(bank, index, snapshot)
                record = paired_forward(
                    system, observation, _original_variant(observation), int(snapshot["policy_seed"])
                )[0]
                rng_groups[str(index)]["initial"].append(record.initial_chain_sha256)
                rng_groups[str(index)]["post"].append(record.post_call_rng_sha256)
                baseline = bc_records[index]
                difference = record.full_action.astype(np.float64) - baseline.full_action.astype(np.float64)
                geometry = action_geometry(
                    difference,
                    np.zeros(3),
                    bank["goal_world"][index] - bank["position_world"][index],
                    ActionSpec.e5(),
                )
                abs_max = float(np.max(np.abs(difference)))
                ref = np.maximum(np.abs(baseline.full_action.astype(np.float64)), 1e-12)
                rel_max = float(np.max(np.abs(difference) / ref))
                if label == "u0" and not (abs_max <= tol_abs and rel_max <= tol_rel):
                    raise _failure(
                        "MEASUREMENT_INVALID",
                        f"u0 differs from BC at snapshot {index}: abs={abs_max} rel={rel_max}",
                    )
                rows.append(
                    {
                        "checkpoint": label,
                        "update": next(c.update for c in contracts if c.label == label),
                        "snapshot_id": snapshot["snapshot_id"],
                        "task_id": snapshot["task_id"],
                        "stratum": snapshot["stratum"],
                        "anchor_labels": snapshot["anchor_labels"],
                        "action_h4_l2": float(np.linalg.norm(difference[..., 2:6, :])),
                        "action_max_abs": abs_max,
                        "action_max_rel": rel_max,
                        "goal_parallel_m": geometry["goal_parallel_m"],
                        "goal_perpendicular_m": geometry["goal_perpendicular_m"],
                        "vertical_m": geometry["vertical_m"],
                        "yaw_sum_rad": geometry["yaw_sum_rad"],
                        "yaw_total_variation_rad": geometry["yaw_total_variation_rad"],
                        "path_shape_norm_m": geometry["path_shape_norm_m"],
                        "endpoint": geometry["endpoint"],
                        "action_sha256": canonical_array_sha256(record.full_action),
                        "chain_sha256": canonical_array_sha256(record.normalized_chain),
                        "initial_chain_sha256": record.initial_chain_sha256,
                        "post_call_rng_sha256": record.post_call_rng_sha256,
                        "input_sha256": record.input_sha256,
                    }
                )
            tripwire_reports[label] = guard.verify(require_rng_unchanged=True)
        finally:
            _release_system(system)
    invalid_rng = {}
    for index, values in rng_groups.items():
        initial = sorted(set(values["initial"]))
        post = sorted(set(values["post"]))
        if len(initial) != 1 or len(post) != 1:
            invalid_rng[index] = {"initial": initial, "post": post}
    if invalid_rng:
        raise _failure("INVALID_NOISE_PAIRING", f"cross-checkpoint initial chain mismatch: {invalid_rng}")
    rl_rows = [row for row in rows if row["checkpoint"] not in {"bc", "u0"}]
    exceeding = [
        row for row in rl_rows if row["action_max_abs"] > tol_abs and row["action_max_rel"] > tol_rel
    ]
    b1_summary = {
        "schema": "routea_b1_summary_v1",
        "route": "DESCRIPTIVE_DRIFT_ONLY",
        "reason": (
            "automatic systematic-retention verdict is forbidden: it requires break/clear task-level "
            "paired comparison against rescue and seed1/seed2 distributions"
        ),
        "rl_row_count": len(rl_rows),
        "rows_above_measurement_floor": len(exceeding),
    }
    with AtomicOutputDirectory(args.output_dir) as staging:
        write_strict_json(staging / "checkpoint_comparability.json", comparability)
        write_strict_json(staging / "thresholds.json", thresholds)
        write_csv_rows(staging / "bc_retention_drift.csv", rows)
        write_strict_json(staging / "b1_summary.json", b1_summary)
        write_strict_json(
            staging / "rng_pairing_verification.json",
            {"status": "VALID_NOISE_PAIRING", "snapshot_groups": len(rng_groups), "mismatches": {}},
        )
        write_strict_json(
            staging / "tripwire_verification.json",
            {"status": "READ_ONLY_VERIFIED", "systems": tripwire_reports},
        )
        publish_stage_manifest(staging, source_manifest)
    _single_line({"status": "RUN_B1_COMPLETE", "rows": len(rows), "output_dir": str(args.output_dir.resolve())})
    return 0


def _metrics_jsonable(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in metrics.items()
    }


def _run_b2(args: argparse.Namespace) -> int:
    _check_new_output_first(args.output_dir)
    verify_sha256_manifest(args.b0_dir)
    verify_sha256_manifest(args.b1_dir)
    comparability = load_strict_json(Path(args.b1_dir) / "checkpoint_comparability.json")
    if comparability.get("status") != "COMPARABLE":
        raise _failure("CHECKPOINT_CONTRACT_MISMATCH", "B1 comparability did not pass")
    thresholds = load_strict_json(Path(args.b1_dir) / "thresholds.json")
    b1_summary = load_strict_json(Path(args.b1_dir) / "b1_summary.json")
    bank_manifest = load_strict_json(Path(args.b0_dir) / "observation_bank_manifest.json")
    bank_path = Path(args.b0_dir) / "observation_bank.npz"
    bank = load_numeric_npz(bank_path, max_total_bytes=2 * 1024**3)
    snapshots = {row["snapshot_id"]: row for row in bank_manifest["snapshots"]}
    sentinel = bank_manifest["sentinel_subset"]
    if any(sum(item["task_id"] == x["task_id"] and item["stratum"] == x["stratum"] for x in sentinel) > 1 for item in sentinel):
        raise _failure("INVALID_SENTINEL_SUBSET", "more than one anchor per task")

    policy_specs = (
        ("bc", args.bc_config, None),
        ("seed3-u32", args.seed3_u32_config, args.seed3_u32_checkpoint),
    )
    forward_by_policy: dict[str, dict[str, Any]] = {label: {} for label, _, _ in policy_specs}
    pair_reports = []
    tripwire_reports = {}
    jsonl_rows = []
    csv_rows = []
    for label, config_path, checkpoint_path in policy_specs:
        _, system = load_read_only_system(config_path, checkpoint_path, REPO_ROOT)
        guard = install_read_only_tripwires(system, immutable_files=[bank_path])
        try:
            for entry in sentinel:
                snapshot = snapshots[entry["snapshot_id"]]
                index = int(snapshot["bank_index"])
                observation = _bank_observation(bank, index, snapshot)
                variants = []
                for name in GOAL_VARIANTS:
                    state = make_goal_variant(observation["agent_pos"], name)
                    mutation = verify_goal_only_mutation(
                        observation["agent_pos"], state, observation["point_cloud"]
                    )
                    variants.append({"name": name, "agent_pos": state, "mutation": mutation})
                records = paired_forward(
                    system, observation, variants, int(snapshot["policy_seed"])
                )
                pair_reports.append(
                    {"policy": label, "snapshot_id": snapshot["snapshot_id"], **verify_forward_pairing(records)}
                )
                metrics_by_variant = {}
                for variant, record in zip(variants, records):
                    target_goal = reconstruct_goal_history(variant["agent_pos"]).goal
                    metrics = action_geometry(
                        record.full_action,
                        bank["position_world"][index],
                        target_goal,
                        ActionSpec.e5(),
                    )
                    metrics_by_variant[record.variant] = metrics
                    row = {
                        "policy": label,
                        "snapshot_id": snapshot["snapshot_id"],
                        "task_id": snapshot["task_id"],
                        "stratum": snapshot["stratum"],
                        "anchor_kind": entry["anchor_kind"],
                        "target_variant": record.variant,
                        "initial_chain_sha256": record.initial_chain_sha256,
                        "post_call_rng_sha256": record.post_call_rng_sha256,
                        "input_sha256": record.input_sha256,
                        "non_goal_sha256": canonical_tree_sha256(
                            {
                                "position": variant["agent_pos"][:, :3],
                                "yaw": variant["agent_pos"][:, 7:9],
                                "point_cloud": observation["point_cloud"],
                            }
                        ),
                        **{
                            key: value
                            for key, value in _metrics_jsonable(metrics).items()
                            if key not in {"action_h4", "waypoints_h4"}
                        },
                    }
                    csv_rows.append(row)
                    jsonl_rows.append(
                        {
                            **row,
                            "agent_pos_sha256": canonical_array_sha256(variant["agent_pos"]),
                            "mutation": variant["mutation"],
                        }
                    )
                bearing = bearing_jacobian(
                    metrics_by_variant["L1-bearing+15"], metrics_by_variant["L2-bearing-15"]
                )
                vertical = vertical_jacobian(
                    metrics_by_variant["L5-z+0.3m"], metrics_by_variant["L6-z-0.3m"]
                )
                vector = np.concatenate(
                    [
                        np.asarray(bearing.get("endpoint", np.zeros(3))).reshape(-1),
                        np.asarray(vertical.get("endpoint", np.zeros(3))).reshape(-1),
                    ]
                )
                forward_by_policy[label][snapshot["snapshot_id"]] = {
                    "bearing": bearing,
                    "vertical": vertical,
                    "response_norm": float(np.linalg.norm(vector)),
                }
            tripwire_reports[label] = guard.verify(require_rng_unchanged=True)
        finally:
            _release_system(system)

    cross_policy_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in csv_rows:
        cross_policy_groups.setdefault(
            (str(row["snapshot_id"]), str(row["target_variant"])), []
        ).append(row)
    for identity, members in cross_policy_groups.items():
        if len(members) != 2 or {row["policy"] for row in members} != {"bc", "seed3-u32"}:
            raise _failure("INVALID_OBSERVATION_MUTATION", f"incomplete BC/seed3 pair: {identity}")
        if len({row["input_sha256"] for row in members}) != 1 or len(
            {row["non_goal_sha256"] for row in members}
        ) != 1:
            raise _failure(
                "INVALID_OBSERVATION_MUTATION",
                f"BC/seed3 observation identity differs: {identity}",
            )
        if len({row["initial_chain_sha256"] for row in members}) != 1 or len(
            {row["post_call_rng_sha256"] for row in members}
        ) != 1:
            raise _failure(
                "INVALID_NOISE_PAIRING", f"BC/seed3 RNG stream differs: {identity}"
            )

    task_rows = []
    for entry in sentinel:
        snapshot_id = entry["snapshot_id"]
        bc_response = forward_by_policy["bc"][snapshot_id]["response_norm"]
        rl_response = forward_by_policy["seed3-u32"][snapshot_id]["response_norm"]
        relative = None if bc_response < 1e-12 else rl_response / bc_response
        task_rows.append(
            {
                "snapshot_id": snapshot_id,
                "task_id": entry["task_id"],
                "stratum": entry["stratum"],
                "bc_response_norm": bc_response,
                "seed3_response_norm": rl_response,
                "absolute_delta": abs(rl_response - bc_response),
                "relative_sensitivity": relative,
            }
        )
    strata_summary = {}
    abnormal_strata = []
    for stratum in STRATA:
        members = [row for row in task_rows if row["stratum"] == stratum]
        values = [row["relative_sensitivity"] for row in members if row["relative_sensitivity"] is not None]
        median = None if not values else float(np.median(values))
        range_value = None if not values else [float(min(values)), float(max(values))]
        abnormal = (
            len({row["task_id"] for row in members}) >= 2
            and median is not None
            and median < 0.70
            and float(np.median([row["absolute_delta"] for row in members])) > float(thresholds["tol_abs"])
        )
        if abnormal:
            abnormal_strata.append(stratum)
        strata_summary[stratum] = {
            "task_count": len({row["task_id"] for row in members}),
            "median": median,
            "range": range_value,
            "ci_available": False,
            "engineering_gate_relative_lt_0p70": abnormal,
        }
    status = (
        "SENTINEL_ABNORMAL_EXPAND_FULL_B2"
        if abnormal_strata
        else (
            "SENTINEL_NORMAL_B1_DRIFT_OBJECTIVE_CREDIT_ROUTE"
            if b1_summary.get("route") == "SYSTEMATIC_DRIFT"
            else "SENTINEL_NORMAL_B1_NORMAL_TRIGGER_D1F"
        )
    )
    summary = {
        "schema": "routea_b2_sentinel_summary_v1",
        "status": status,
        "abnormal_strata": abnormal_strata,
        "strata": strata_summary,
        "task_rows": task_rows,
        "measurement_floor": thresholds,
        "interpretation_boundary": (
            "relative_sensitivity<0.70 is a secondary engineering expansion gate; this sentinel "
            "does not prove a training repair or a causal reward mechanism"
        ),
    }
    source_manifest = build_diagnostic_source_manifest(REPO_ROOT)
    with AtomicOutputDirectory(args.output_dir) as staging:
        write_jsonl(staging / "goal_variants.jsonl", jsonl_rows)
        write_csv_rows(staging / "b2_sentinel_pairs.csv", csv_rows)
        write_strict_json(staging / "goal_sensitivity_summary.json", summary)
        write_strict_json(
            staging / "rng_pairing_verification.json",
            {"status": "VALID_NOISE_PAIRING", "groups": pair_reports},
        )
        write_strict_json(
            staging / "tripwire_verification.json",
            {"status": "READ_ONLY_VERIFIED", "systems": tripwire_reports},
        )
        publish_stage_manifest(staging, source_manifest)
    _single_line({"status": "RUN_B2_SENTINEL_COMPLETE", "route": status, "output_dir": str(args.output_dir.resolve())})
    return 0


def _run_d1f(args: argparse.Namespace) -> int:
    _check_new_output_first(args.output_dir)
    verify_sha256_manifest(args.bank_dir)
    source_manifest = build_diagnostic_source_manifest(REPO_ROOT)
    bank_dir = Path(args.bank_dir)
    bank_path = bank_dir / "failure_state_bank.npz"
    bank_manifest = load_strict_json(bank_dir / "failure_state_bank_manifest.json")
    reference = load_strict_json(bank_dir / "shared_goal_response_reference.json")
    if not reference.get("valid"):
        raise _failure("FUNCTIONAL_REFERENCE_INVALID", "shared B2 reference is invalid")
    bank = load_numeric_npz(bank_path, max_total_bytes=2 * 1024**3)
    actual_selection_sha = file_sha256(bank_dir / "failure_state_selection.json")
    if bank_manifest.get("selection_manifest_sha256") != actual_selection_sha:
        raise _failure(
            "STAGE_INPUT_HASH_MISMATCH",
            "failure-state bank is not bound to its copied selection",
        )
    bank_source = load_strict_json(bank_dir / "diagnostic_source_manifest.json")
    selection_identity = bank_manifest.get("selection_identity") or {}
    copied_selection = load_strict_json(bank_dir / "failure_state_selection.json")
    if (
        bank_source.get("content_sha256") != source_manifest["content_sha256"]
        or selection_identity.get("diagnostic_source_sha256")
        != source_manifest["content_sha256"]
        or bank_manifest.get("run_lineage_id") != copied_selection.get("run_lineage_id")
        or selection_identity.get("run_lineage_id") != copied_selection.get("run_lineage_id")
        or selection_identity.get("files")
        != {
            filename: file_sha256(bank_dir / filename)
            for filename in D1F_SELECTION_BOUND_FILES
        }
    ):
        raise _failure(
            "STAGE_INPUT_HASH_MISMATCH",
            "failure-state bank selection/source/run lineage is inconsistent",
        )
    if int(bank_manifest["entry_count"]) != len(bank_manifest["snapshots"]):
        raise _failure("INVALID_SNAPSHOT_MANIFEST", "bank entry count differs from snapshots")

    bc_contract = inspect_checkpoint_contract("bc", args.bc_config, None, REPO_ROOT)
    u0_contract = replace(bc_contract, label="u0")
    seed3_contract = inspect_checkpoint_contract(
        "seed3-u32", args.seed3_u32_config, args.seed3_u32_checkpoint, REPO_ROOT
    )
    comparability = compare_checkpoint_contracts(
        [bc_contract, u0_contract, seed3_contract]
    )
    trigger_comparability = load_strict_json(
        bank_dir / "trigger_checkpoint_comparability.json"
    )
    trigger_contracts = {
        str(item["label"]): item for item in trigger_comparability.get("contracts", [])
    }
    current_contracts = {
        "bc": {
            "source_set_hash": bc_contract.source_set_hash,
            "actor_state_sha256": canonical_tree_sha256(bc_contract.actor_state),
        },
        "seed3-u32": {
            "source_set_hash": seed3_contract.source_set_hash,
            "actor_state_sha256": canonical_tree_sha256(seed3_contract.actor_state),
        },
    }
    prior_mismatch = {
        label: {
            "expected": trigger_contracts.get(label),
            "actual": current,
        }
        for label, current in current_contracts.items()
        if trigger_contracts.get(label, {}).get("source_set_hash")
        != current["source_set_hash"]
        or trigger_contracts.get(label, {}).get("actor_state_sha256")
        != current["actor_state_sha256"]
    }
    if prior_mismatch:
        comparability["status"] = "CHECKPOINT_CONTRACT_MISMATCH"
        comparability["prior_v4_identity_mismatch"] = prior_mismatch
    if comparability["status"] != "COMPARABLE":
        with AtomicOutputDirectory(args.output_dir) as staging:
            write_strict_json(staging / "checkpoint_comparability.json", comparability)
            publish_stage_manifest(staging, source_manifest)
        raise _failure("CHECKPOINT_CONTRACT_MISMATCH", "see checkpoint_comparability.json")

    policy_specs = (
        ("bc", args.bc_config, None),
        ("seed3-u32", args.seed3_u32_config, args.seed3_u32_checkpoint),
    )
    forward_metrics: dict[str, dict[str, dict[str, Any]]] = {}
    pair_reports = []
    tripwire_reports = {}
    sensitivity_rows = []
    jsonl_rows = []
    for label, config_path, checkpoint_path in policy_specs:
        _, system = load_read_only_system(config_path, checkpoint_path, REPO_ROOT)
        guard = install_read_only_tripwires(system, immutable_files=[bank_path])
        try:
            for snapshot in bank_manifest["snapshots"]:
                index = int(snapshot["bank_index"])
                observation = _bank_observation(bank, index, snapshot)
                variants = []
                goals = {}
                for name in GOAL_VARIANTS:
                    state = make_goal_variant(observation["agent_pos"], name)
                    mutation = verify_goal_only_mutation(
                        observation["agent_pos"], state, observation["point_cloud"]
                    )
                    goals[name] = reconstruct_goal_history(state).goal
                    variants.append({"name": name, "agent_pos": state, "mutation": mutation})
                records = paired_forward(
                    system, observation, variants, int(snapshot["policy_seed"])
                )
                pair_reports.append(
                    {
                        "policy": label,
                        "snapshot_id": snapshot["snapshot_id"],
                        **verify_forward_pairing(records),
                    }
                )
                metrics_by_variant = {}
                for variant, record in zip(variants, records):
                    metrics = action_geometry(
                        record.full_action,
                        bank["position_world"][index],
                        goals[record.variant],
                        ActionSpec.e5(),
                    )
                    metrics_by_variant[record.variant] = metrics
                    row = {
                        "policy": label,
                        "snapshot_id": snapshot["snapshot_id"],
                        "task_id": snapshot["task_id"],
                        "repeat": snapshot["repeat"],
                        "decision_index": snapshot["decision_index"],
                        "anchor_labels": snapshot["anchor_labels"],
                        "target_variant": record.variant,
                        "initial_chain_sha256": record.initial_chain_sha256,
                        "post_call_rng_sha256": record.post_call_rng_sha256,
                        "input_sha256": record.input_sha256,
                        "non_goal_sha256": canonical_tree_sha256(
                            {
                                "position": variant["agent_pos"][:, :3],
                                "yaw": variant["agent_pos"][:, 7:9],
                                "point_cloud": observation["point_cloud"],
                            }
                        ),
                        **{
                            key: value
                            for key, value in _metrics_jsonable(metrics).items()
                            if key not in {"action_h4", "waypoints_h4"}
                        },
                    }
                    sensitivity_rows.append(row)
                    jsonl_rows.append(
                        {
                            **row,
                            "agent_pos_sha256": canonical_array_sha256(variant["agent_pos"]),
                            "mutation": variant["mutation"],
                        }
                    )
                forward_metrics.setdefault(snapshot["snapshot_id"], {})[label] = {
                    "metrics": metrics_by_variant,
                    "goals": goals,
                }
            tripwire_reports[label] = guard.verify(require_rng_unchanged=True)
        finally:
            _release_system(system)

    paired_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in sensitivity_rows:
        paired_groups.setdefault(
            (str(row["snapshot_id"]), str(row["target_variant"])), []
        ).append(row)
    for identity, members in paired_groups.items():
        if len(members) != 2 or {row["policy"] for row in members} != {"bc", "seed3-u32"}:
            raise _failure("INVALID_OBSERVATION_MUTATION", f"incomplete pair: {identity}")
        for field, status in (
            ("input_sha256", "INVALID_OBSERVATION_MUTATION"),
            ("non_goal_sha256", "INVALID_OBSERVATION_MUTATION"),
            ("initial_chain_sha256", "INVALID_NOISE_PAIRING"),
            ("post_call_rng_sha256", "INVALID_NOISE_PAIRING"),
        ):
            if len({row[field] for row in members}) != 1:
                raise _failure(status, f"{field} differs: {identity}")

    floors = {
        name: float(reference["channels"][name]["functional_floor"])
        for name in (
            "bearing_norm",
            "vertical_norm",
            "combined_norm",
            "abs_bearing_yaw_delta",
        )
    }
    jacobian_rows = []
    for snapshot in bank_manifest["snapshots"]:
        snapshot_id = snapshot["snapshot_id"]
        policies = forward_metrics[snapshot_id]

        def endpoint(policy: str, variant: str) -> np.ndarray:
            return np.asarray(
                policies[policy]["metrics"][variant]["endpoint"], dtype=np.float64
            )

        def scalar(policy: str, variant: str, key: str) -> float:
            return float(policies[policy]["metrics"][variant][key])

        goals = policies["bc"]["goals"]
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
            bc_floor=floors["bearing_norm"],
            rl_floor=floors["bearing_norm"],
            yaw_floor=floors["abs_bearing_yaw_delta"],
            response_denominator=math.pi / 6.0,
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
            bc_floor=floors["vertical_norm"],
            rl_floor=floors["vertical_norm"],
            yaw_floor=math.inf,
            response_denominator=0.6,
        )
        bearing["bc_goal_parallel_delta"] = (
            scalar("bc", "L1-bearing+15", "goal_parallel_m")
            - scalar("bc", "L2-bearing-15", "goal_parallel_m")
        ) / (math.pi / 6.0)
        bearing["rl_goal_parallel_delta"] = (
            scalar("seed3-u32", "L1-bearing+15", "goal_parallel_m")
            - scalar("seed3-u32", "L2-bearing-15", "goal_parallel_m")
        ) / (math.pi / 6.0)
        bearing["bc_goal_perpendicular_delta"] = (
            scalar("bc", "L1-bearing+15", "goal_perpendicular_m")
            - scalar("bc", "L2-bearing-15", "goal_perpendicular_m")
        ) / (math.pi / 6.0)
        bearing["rl_goal_perpendicular_delta"] = (
            scalar("seed3-u32", "L1-bearing+15", "goal_perpendicular_m")
            - scalar("seed3-u32", "L2-bearing-15", "goal_perpendicular_m")
        ) / (math.pi / 6.0)
        classification = classify_failure_snapshot(
            {"bearing": bearing, "vertical": vertical}, floors
        )
        relative_distance = float(snapshot["goal_distance"]) / max(
            float(snapshot["initial_goal_distance_m"]), 1e-12
        )
        distance_band = (
            "closure" if relative_distance <= 1.0 / 3.0 else (
                "middle" if relative_distance <= 2.0 / 3.0 else "route"
            )
        )
        jacobian_rows.append(
            {
                "snapshot_id": snapshot_id,
                "task_id": snapshot["task_id"],
                "repeat": snapshot["repeat"],
                "decision_index": snapshot["decision_index"],
                "anchor_labels": snapshot["anchor_labels"],
                "split": snapshot["split"],
                "role": snapshot["role"],
                "goal_distance": snapshot["goal_distance"],
                "relative_goal_distance": relative_distance,
                "distance_band": distance_band,
                "bearing": bearing,
                "vertical": vertical,
                "bc_combined_norm": float(math.hypot(bearing["bc_norm"], vertical["bc_norm"])),
                "seed3_combined_norm": float(math.hypot(bearing["rl_norm"], vertical["rl_norm"])),
                "classification": classification,
            }
        )

    summary = build_failure_summary(
        jacobian_rows,
        shared_primary_mechanisms=reference.get("shared_primary_mechanisms", ()),
        functional_floors=floors,
    )
    summary["input_identity"] = {
        "run_lineage_id": bank_manifest["run_lineage_id"],
        "diagnostic_source_sha256": source_manifest["content_sha256"],
        "selection_identity": selection_identity,
        "bank_stage_manifest_sha256": file_sha256(bank_dir / "manifest_sha256.txt"),
        "bank_manifest_sha256": file_sha256(
            bank_dir / "failure_state_bank_manifest.json"
        ),
        "bank_npz_sha256": file_sha256(bank_path),
        "selection_sha256": actual_selection_sha,
        "shared_reference_sha256": file_sha256(
            bank_dir / "shared_goal_response_reference.json"
        ),
    }
    task_rows, stage_rows, distance_role_rows = _failure_summary_tables(
        jacobian_rows, summary, floors
    )
    with AtomicOutputDirectory(args.output_dir) as staging:
        write_strict_json(staging / "checkpoint_comparability.json", comparability)
        write_jsonl(staging / "goal_variants.jsonl", jsonl_rows)
        write_csv_rows(staging / "failure_state_sensitivity.csv", sensitivity_rows)
        write_csv_rows(staging / "failure_state_jacobians.csv", jacobian_rows)
        write_csv_rows(staging / "failure_state_task_summary.csv", task_rows)
        write_csv_rows(staging / "failure_state_stage_coverage.csv", stage_rows)
        write_csv_rows(staging / "failure_state_distance_role_coverage.csv", distance_role_rows)
        write_strict_json(staging / "failure_state_progression.json", summary["progression"])
        write_strict_json(staging / "failure_state_summary.json", summary)
        write_strict_json(staging / "verdict.json", summary["verdict"])
        write_strict_json(
            staging / "rng_pairing_verification.json",
            {"status": "VALID_NOISE_PAIRING", "groups": pair_reports},
        )
        write_strict_json(
            staging / "tripwire_verification.json",
            {"status": "READ_ONLY_VERIFIED", "systems": tripwire_reports},
        )
        publish_stage_manifest(staging, source_manifest)
    _single_line(
        {
            "status": "RUN_D1F_COMPLETE",
            "snapshots": len(jacobian_rows),
            "tasks": summary["independent_task_count"],
            "verdict": summary["verdict"]["status"],
            "output_dir": str(args.output_dir.resolve()),
        }
    )
    return 0


def _failure_summary_tables(
    jacobian_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    floors: Mapping[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    task_rows = [
        {"task_id": task, "classification": classification}
        for task, classification in summary["task_classification"].items()
    ]
    stage_rows = []
    for mechanism, stages in summary["stage_coverage"].items():
        for stage, report in stages.items():
            stage_rows.append(
                {"mechanism": mechanism, "stage": stage, **(report or {
                    "mechanism_task_numerator": 0,
                    "determinate_task_denominator": 0,
                    "coverage_fraction": None,
                    "wilson_95_lower": None,
                    "wilson_95_upper": None,
                    "support_level": "NONE",
                })}
            )
    distance_role_rows = []
    for dimension in ("distance_band", "split", "role"):
        for value in sorted({str(row[dimension]) for row in jacobian_rows}):
            members = [row for row in jacobian_rows if str(row[dimension]) == value]
            subgroup = build_failure_summary(
                members, functional_floors=floors, primary_stage=None
            )
            for mechanism, report in subgroup["overall_coverage"].items():
                distance_role_rows.append(
                    {"dimension": dimension, "value": value, "mechanism": mechanism, **report}
                )
    return task_rows, stage_rows, distance_role_rows


def _load_reaudit_jacobians(path: Path, floors: Mapping[str, float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, raw in enumerate(csv.DictReader(handle), start=2):
            try:
                row = dict(raw)
                row["repeat"] = int(row["repeat"])
                row["decision_index"] = int(row["decision_index"])
                for field in (
                    "goal_distance",
                    "relative_goal_distance",
                    "bc_combined_norm",
                    "seed3_combined_norm",
                ):
                    row[field] = float(row[field])
                row["anchor_labels"] = json.loads(row["anchor_labels"])
                row["bearing"] = json.loads(row["bearing"])
                row["vertical"] = json.loads(row["vertical"])
                if not isinstance(row["anchor_labels"], list):
                    raise ValueError("anchor_labels is not a list")
                if not isinstance(row["bearing"], Mapping) or not isinstance(
                    row["vertical"], Mapping
                ):
                    raise ValueError("bearing/vertical is not an object")
                recomputed = classify_failure_snapshot(
                    {"bearing": row["bearing"], "vertical": row["vertical"]}, floors
                )
                if row["classification"] != recomputed:
                    raise ValueError(
                        f"stored classification {row['classification']} != {recomputed}"
                    )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise _failure(
                    "INVALID_REAUDIT_INPUT",
                    f"failure_state_jacobians.csv line {line_number}: {exc}",
                ) from exc
            rows.append(row)
    if not rows:
        raise _failure("INVALID_REAUDIT_INPUT", "failure_state_jacobians.csv is empty")
    return rows


def _run_reaudit_d1f(args: argparse.Namespace) -> int:
    """Recompute only D1-F statistical artifacts from a verified final package."""

    _check_new_output_first(args.output_dir)
    final_dir = Path(args.final_dir).resolve()
    verified_files = verify_sha256_manifest(final_dir)
    required = {
        "diagnostic_source_manifest.json",
        "failure_state_jacobians.csv",
        "failure_state_summary.json",
        "identity.json",
        "shared_goal_response_reference.json",
    }
    missing = sorted(required - set(verified_files))
    if missing:
        raise _failure("INVALID_REAUDIT_INPUT", f"parent manifest omits {missing}")

    parent_source = load_strict_json(final_dir / "diagnostic_source_manifest.json")
    parent_summary = load_strict_json(final_dir / "failure_state_summary.json")
    parent_identity = load_strict_json(final_dir / "identity.json")
    reference = load_strict_json(final_dir / "shared_goal_response_reference.json")
    if not reference.get("valid"):
        raise _failure("FUNCTIONAL_REFERENCE_INVALID", "parent shared reference is invalid")
    source_sha = str(parent_source.get("content_sha256", ""))
    run_lineage_id = str(parent_identity.get("run_lineage_id", ""))
    summary_identity = parent_summary.get("input_identity") or {}
    reference_sha = file_sha256(final_dir / "shared_goal_response_reference.json")
    if (
        not source_sha
        or not run_lineage_id
        or parent_identity.get("diagnostic_source_sha256") != source_sha
        or summary_identity.get("diagnostic_source_sha256") != source_sha
        or summary_identity.get("run_lineage_id") != run_lineage_id
        or summary_identity.get("shared_reference_sha256") != reference_sha
    ):
        raise _failure(
            "STAGE_INPUT_HASH_MISMATCH",
            "parent final source/run-lineage/shared-reference identity is inconsistent",
        )

    floors = {
        name: float(reference["channels"][name]["functional_floor"])
        for name in (
            "bearing_norm",
            "vertical_norm",
            "combined_norm",
            "abs_bearing_yaw_delta",
        )
    }
    jacobian_path = final_dir / "failure_state_jacobians.csv"
    jacobian_rows = _load_reaudit_jacobians(jacobian_path, floors)
    summary = build_failure_summary(
        jacobian_rows,
        shared_primary_mechanisms=reference.get("shared_primary_mechanisms", ()),
        functional_floors=floors,
    )
    current_source = build_diagnostic_source_manifest(REPO_ROOT)
    reaudit_identity = {
        "schema": "routea_d1f_reaudit_identity_v1",
        "mode": "READ_ONLY_STATISTICAL_RECOMPUTATION",
        "run_lineage_id": run_lineage_id,
        "parent_diagnostic_source_sha256": source_sha,
        "reaudit_diagnostic_source_sha256": current_source["content_sha256"],
        "parent_final_manifest_sha256": file_sha256(final_dir / "manifest_sha256.txt"),
        "parent_failure_state_jacobians_sha256": file_sha256(jacobian_path),
        "parent_shared_reference_sha256": reference_sha,
        "parent_failure_state_summary_sha256": file_sha256(
            final_dir / "failure_state_summary.json"
        ),
        "parent_stage_files": verified_files,
        "forward_rerun": False,
        "model_loaded": False,
        "training_performed": False,
    }
    summary["input_identity"] = reaudit_identity
    summary["reaudit_contract"] = {
        "parent_artifacts_read_only": True,
        "raw_forward_values_reused": True,
        "statistical_aggregation_only": True,
    }
    task_rows, stage_rows, distance_role_rows = _failure_summary_tables(
        jacobian_rows, summary, floors
    )
    with AtomicOutputDirectory(args.output_dir) as staging:
        write_csv_rows(staging / "failure_state_task_summary.csv", task_rows)
        write_csv_rows(staging / "failure_state_stage_coverage.csv", stage_rows)
        write_csv_rows(
            staging / "failure_state_distance_role_coverage.csv", distance_role_rows
        )
        write_strict_json(staging / "failure_state_progression.json", summary["progression"])
        write_strict_json(staging / "failure_state_summary.json", summary)
        write_strict_json(staging / "verdict.json", summary["verdict"])
        write_strict_json(staging / "reaudit_identity.json", reaudit_identity)
        publish_stage_manifest(staging, current_source)
    _single_line(
        {
            "status": "READ_ONLY_REAUDIT_D1F_COMPLETE",
            "snapshots": len(jacobian_rows),
            "tasks": summary["independent_task_count"],
            "determinate_tasks": summary["determinate_task_count"],
            "verdict": summary["verdict"]["status"],
            "output_dir": str(args.output_dir.resolve()),
        }
    )
    return 0


def _run_finalize(args: argparse.Namespace) -> int:
    _check_new_output_first(args.output_dir)
    output = finalize_outputs(
        args.b0_dir,
        args.b1_dir,
        args.b2_dir,
        args.output_dir,
        source_manifest=build_diagnostic_source_manifest(REPO_ROOT),
    )
    _single_line({"status": "FINALIZE_COMPLETE", "output_dir": str(output)})
    return 0


def _run_finalize_d1f(args: argparse.Namespace) -> int:
    _check_new_output_first(args.output_dir)
    output = finalize_d1f_outputs(
        args.selection_dir,
        args.bank_dir,
        args.forward_dir,
        args.output_dir,
        source_manifest=build_diagnostic_source_manifest(REPO_ROOT),
    )
    _single_line({"status": "FINALIZE_D1F_COMPLETE", "output_dir": str(output)})
    return 0


def _self_check() -> int:
    before_source = build_inventory(REPO_ROOT)[1]
    rng = capture_rng_state()
    expected = (random.random(), np.random.random(), torch.rand(2))
    restore_rng_state(rng)
    actual = (random.random(), np.random.random(), torch.rand(2))
    rng_restored = expected[0] == actual[0] and expected[1] == actual[1] and torch.equal(expected[2], actual[2])
    restore_rng_state(rng)
    state = np.zeros((8, 9), dtype=np.float32)
    goal = np.array([3.0, -1.0, 1.5])
    for index in range(8):
        position = np.array([index * 0.1, 0.0, 1.0])
        yaw = index * 0.1
        delta = goal - position
        c, s = math.cos(-yaw), math.sin(-yaw)
        body = np.array([c * delta[0] - s * delta[1], s * delta[0] + c * delta[1], delta[2]])
        state[index, :3] = position
        state[index, 3:6] = body
        state[index, 6] = np.linalg.norm(delta)
        state[index, 7:9] = (math.sin(yaw), math.cos(yaw))
    roundtrip = reconstruct_goal_history(state).max_pairwise_inf_m
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "check.npz"
        source = {"x": np.arange(6, dtype=np.float32).reshape(2, 3)}
        write_numeric_npz(path, source, max_total_bytes=1024)
        loaded = load_numeric_npz(path, expected_shapes={"x": (2, 3)}, max_total_bytes=1024)
        npz_reload = canonical_array_sha256(source["x"]) == canonical_array_sha256(loaded["x"])
    after_source = build_inventory(REPO_ROOT)[1]
    payload = {
        "status": "SELF_CHECK_PASS",
        "goal_roundtrip_max_abs": roundtrip,
        "rng_restored": rng_restored,
        "numeric_npz_reload": npz_reload,
        "training_source_set_unchanged": before_source == after_source,
        "training_source_set_hash": before_source,
        "diagnostic_source_sha256": build_diagnostic_source_manifest(REPO_ROOT)["content_sha256"],
    }
    if not (
        roundtrip <= 1e-4
        and rng_restored
        and npz_reload
        and before_source == after_source
    ):
        payload["status"] = "SELF_CHECK_FAIL"
        _single_line(payload)
        return 2
    _single_line(payload)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    subparsers = parser.add_subparsers(dest="command")

    select = subparsers.add_parser("select-b0")
    for seed in (1, 2, 3):
        select.add_argument(f"--fixed32-seed{seed}-dir", type=Path, required=True)
    select.add_argument("--layera-geometry", type=Path, required=True)
    select.add_argument("--rl-test-geometry", type=Path, required=True)
    select.add_argument("--episode-config-dir", type=Path, default=REPO_ROOT / "data" / "eval_configs")
    select.add_argument("--seed1-config", type=Path, required=True)
    select.add_argument("--output-dir", type=Path, required=True)

    capture = subparsers.add_parser("capture-b0")
    capture.add_argument("--selection-dir", type=Path, required=True)
    capture.add_argument("--config", type=Path, required=True)
    capture.add_argument("--output-dir", type=Path, required=True)

    b1 = subparsers.add_parser("run-b1")
    b1.add_argument("--b0-dir", type=Path, required=True)
    b1.add_argument("--bc-config", type=Path, required=True)
    b1.add_argument("--checkpoint", action="append", type=_parse_checkpoint_arg, required=True)
    b1.add_argument("--output-dir", type=Path, required=True)

    b2 = subparsers.add_parser("run-b2-sentinel")
    b2.add_argument("--b0-dir", type=Path, required=True)
    b2.add_argument("--b1-dir", type=Path, required=True)
    b2.add_argument("--bc-config", type=Path, required=True)
    b2.add_argument("--seed3-u32-config", type=Path, required=True)
    b2.add_argument("--seed3-u32-checkpoint", type=Path, required=True)
    b2.add_argument("--output-dir", type=Path, required=True)

    final = subparsers.add_parser("finalize")
    final.add_argument("--b0-dir", type=Path, required=True)
    final.add_argument("--b1-dir", type=Path, required=True)
    final.add_argument("--b2-dir", type=Path, required=True)
    final.add_argument("--output-dir", type=Path, required=True)

    select_d1f = subparsers.add_parser("select-d1f")
    select_d1f.add_argument("--trigger-final-dir", type=Path, required=True)
    select_d1f.add_argument("--b1-dir", type=Path, required=True)
    select_d1f.add_argument("--b2-dir", type=Path, required=True)
    for seed in (1, 2, 3):
        select_d1f.add_argument(f"--fixed32-seed{seed}-dir", type=Path, required=True)
    select_d1f.add_argument("--layera-geometry", type=Path, required=True)
    select_d1f.add_argument("--rl-test-geometry", type=Path, required=True)
    select_d1f.add_argument("--episode-config-dir", type=Path, default=REPO_ROOT / "data" / "eval_configs")
    select_d1f.add_argument("--seed3-u32-config", type=Path, required=True)
    select_d1f.add_argument("--output-dir", type=Path, required=True)

    capture_d1f = subparsers.add_parser("capture-d1f")
    capture_d1f.add_argument("--selection-dir", type=Path, required=True)
    capture_d1f.add_argument("--config", type=Path, required=True)
    capture_d1f.add_argument("--output-dir", type=Path, required=True)

    run_d1f = subparsers.add_parser("run-d1f")
    run_d1f.add_argument("--bank-dir", type=Path, required=True)
    run_d1f.add_argument("--bc-config", type=Path, required=True)
    run_d1f.add_argument("--seed3-u32-config", type=Path, required=True)
    run_d1f.add_argument("--seed3-u32-checkpoint", type=Path, required=True)
    run_d1f.add_argument("--output-dir", type=Path, required=True)

    finalize_d1f = subparsers.add_parser("finalize-d1f")
    finalize_d1f.add_argument("--selection-dir", type=Path, required=True)
    finalize_d1f.add_argument("--bank-dir", type=Path, required=True)
    finalize_d1f.add_argument("--forward-dir", type=Path, required=True)
    finalize_d1f.add_argument("--output-dir", type=Path, required=True)

    reaudit_d1f = subparsers.add_parser("reaudit-d1f")
    reaudit_d1f.add_argument("--final-dir", type=Path, required=True)
    reaudit_d1f.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.self_check:
            if args.command is not None:
                raise _failure("CLI_USAGE_ERROR", "--self-check cannot be combined with a command")
            return _self_check()
        handlers = {
            "select-b0": _run_select_b0,
            "capture-b0": _run_capture_b0,
            "run-b1": _run_b1,
            "run-b2-sentinel": _run_b2,
            "finalize": _run_finalize,
            "select-d1f": _run_select_d1f,
            "capture-d1f": _run_capture_d1f,
            "run-d1f": _run_d1f,
            "finalize-d1f": _run_finalize_d1f,
            "reaudit-d1f": _run_reaudit_d1f,
        }
        if args.command not in handlers:
            parser.error("a command or --self-check is required")
        return handlers[args.command](args)
    except DiagnosticFailure as exc:
        _single_line(
            {
                "status": exc.status,
                "detail": exc.detail,
                "exit_code": exc.exit_code,
            }
        )
        return exc.exit_code
    except Exception as exc:
        _single_line(
            {
                "status": "UNEXPECTED_ERROR",
                "error_type": type(exc).__name__,
                "detail": str(exc),
                "exit_code": 2,
            }
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
