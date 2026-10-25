#!/usr/bin/env python3
"""Read-only Route A D0.5-B paired first-divergence reward/credit diagnosis.

The CLI has four immutable stages:

``select-d05b`` binds endpoint break/rescue identities and BC-repeat nulls;
``capture-d05b`` replays frozen actions through production reward and critics;
``analyze-d05b`` extracts the registered divergence windows and task evidence;
``finalize-d05b`` publishes the bounded verdict without authorizing training.

No stage samples a new actor action, calls backward/optimizer/trainer update, or
modifies a checkpoint, reward configuration, fixed32 artifact, or world asset.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import uuid

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusion_policy_3d.env.reward_v17 import TERM_NAMES  # noqa: E402
from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.gae import compute_gae  # noqa: E402
from scripts.routea_causal_diagnostics.credit_divergence import (  # noqa: E402
    CONTEXT_GATING_SUSPECTED,
    INCONCLUSIVE,
    REWARD_GROUPS,
    build_d05_verdict,
    classify_endpoint_pair,
    classify_transition_signal,
    reward_group_totals,
    s2_normalize_advantages,
    trace_window_indices,
    validate_reward_trace_row,
)
from scripts.routea_causal_diagnostics.bank import (  # noqa: E402
    load_fixed32_parent,
    parse_trace_csv,
)
from scripts.routea_causal_diagnostics.contracts import (  # noqa: E402
    DiagnosticFailure,
    canonical_array_sha256,
    canonical_tree_sha256,
    capture_rng_state,
    load_strict_json,
    restore_rng_state,
    write_strict_json,
)
from scripts.routea_causal_diagnostics.failure_state import (  # noqa: E402
    build_bc_repeat_null,
    extract_decision_features,
    find_first_persistent_divergence,
    serialize_bc_repeat_null,
)
from scripts.routea_causal_diagnostics.reporting import (  # noqa: E402
    AtomicOutputDirectory,
    file_sha256,
    publish_stage_manifest,
    verify_sha256_manifest,
    write_csv_rows,
    write_jsonl,
)
from scripts.routea_causal_diagnostics.runtime import (  # noqa: E402
    install_read_only_tripwires,
    load_read_only_system,
    validate_zero_effect_lora_state,
)


EXIT_USAGE = 2
EXIT_REPLAY = 10
EXIT_CHECKPOINT = 11
EXIT_READ_ONLY = 13
D05_EXPECTED_TARGET_REPEATS = 160
D05_EXPECTED_FILLER_REPEATS = 480

D05_SELECTION_FILES = (
    "selection.json",
    "paired_episode_outcomes.csv",
    "bc_repeat_null.json",
    "input_identity.json",
)

D05_CAPTURE_FILES = (
    "selection.json",
    "paired_episode_outcomes.csv",
    "bc_repeat_null.json",
    "input_identity.json",
    "paired_trace_transitions.jsonl",
    "credit_population.jsonl",
    "credit_batch_audit.json",
    "tripwire_verification.json",
)

D05_ANALYSIS_FILES = (
    "first_divergence_windows.jsonl",
    "reward_credit_transition_table.csv",
    "reward_credit_group_summary.csv",
    "paired_episode_outcomes.csv",
    "bc_repeat_null.json",
    "safe_arrival_substep_labels.csv",
    "task_mechanisms.csv",
    "analysis_summary.json",
)


def _failure(status: str, detail: str, exit_code: int = EXIT_USAGE) -> DiagnosticFailure:
    return DiagnosticFailure(status, detail, exit_code=exit_code)


def _single_line(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise _failure(
                    "INVALID_JSON_ARTIFACT", "{}:{}: {}".format(path, line_number, exc)
                )
            if not isinstance(value, dict):
                raise _failure(
                    "INVALID_JSON_ARTIFACT", "{}:{} is not an object".format(path, line_number)
                )
            rows.append(value)
    return rows


def _d05_source_manifest() -> Dict[str, Any]:
    paths = [
        Path(__file__).resolve(),
        (REPO_ROOT / "scripts" / "routea_causal_diagnostics" / "credit_divergence.py").resolve(),
    ]
    records = []
    for path in paths:
        payload = path.read_bytes()
        records.append(
            {
                "path": path.relative_to(REPO_ROOT).as_posix(),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    records.sort(key=lambda row: row["path"])
    return {
        "schema": "routea_d05b_source_manifest_v1",
        "content_sha256": canonical_tree_sha256(
            [[row["path"], row["sha256"], row["bytes"]] for row in records]
        ),
        "file_count": len(records),
        "files": records,
    }


def _check_new_output(path: Path) -> None:
    if Path(path).exists():
        raise _failure("OUTPUT_DIR_EXISTS", str(Path(path).resolve()))


def _load_episode_configs(config_dir: Path) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    index = {}
    sources = []
    for path in sorted(Path(config_dir).glob("*.json")):
        payload = load_strict_json(path)
        if not isinstance(payload, Mapping):
            continue
        sources.append({"path": str(path.resolve()), "sha256": file_sha256(path)})
        for world_rows in payload.values():
            if not isinstance(world_rows, list):
                continue
            for row in world_rows:
                if not isinstance(row, Mapping) or "episode_id" not in row:
                    continue
                episode_id = str(row["episode_id"])
                encoded = json.dumps(
                    dict(row), sort_keys=True, separators=(",", ":"), ensure_ascii=False
                ).encode("utf-8")
                record = {
                    "payload": dict(row),
                    "config_hash": hashlib.sha256(encoded).hexdigest(),
                    "source_path": str(path.resolve()),
                }
                if episode_id in index and index[episode_id]["config_hash"] != record["config_hash"]:
                    raise _failure("EPISODE_CONFIG_IDENTITY_MISMATCH", episode_id)
                index[episode_id] = record
    if not index:
        raise _failure("EPISODE_CONFIG_IDENTITY_MISMATCH", "no episode configs found")
    return index, sources


def _endpoint_summaries(
    seed_dirs: Sequence[Path],
) -> Tuple[Dict[str, str], Dict[Tuple[str, int, str], Mapping[str, Any]]]:
    # Reuse the already-tested fixed32 parser, but keep its private helper out
    # of this CLI's public contract.
    from scripts.probe_routea_goal_sensitivity import _endpoint_done_reasons

    return _endpoint_done_reasons(seed_dirs)


def _is_sha256(value: Any) -> bool:
    token = str(value)
    return len(token) == 64 and all(character in "0123456789abcdef" for character in token)


def _validate_endpoint_actual(
    actual: Any, *, task_id: str, policy: str, done_reason: str
) -> Dict[str, Any]:
    if not isinstance(actual, Mapping):
        raise _failure(
            "ENDPOINT_SUMMARY_CONTRACT_MISMATCH",
            "normalized endpoint summary must be an object for {} {}".format(
                task_id, policy
            ),
        )
    normalized = dict(actual)
    if normalized.get("task_id") != task_id or normalized.get("policy") != policy:
        raise _failure(
            "ENDPOINT_SUMMARY_CONTRACT_MISMATCH",
            "endpoint identity differs for {} {}".format(task_id, policy),
        )
    if normalized.get("done_reason") != done_reason:
        raise _failure(
            "ENDPOINT_SUMMARY_CONTRACT_MISMATCH",
            "endpoint reason differs for {} {}: {}/{}".format(
                task_id, policy, normalized.get("done_reason"), done_reason
            ),
        )
    steps = normalized.get("steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise _failure(
            "ENDPOINT_SUMMARY_CONTRACT_MISMATCH",
            "endpoint steps must be a positive integer for {} {}".format(task_id, policy),
        )
    observation = normalized.get("observation_validity_summary")
    if not isinstance(observation, Mapping) or not _is_sha256(
        observation.get("last_policy_input_hash")
    ):
        raise _failure(
            "ENDPOINT_SUMMARY_CONTRACT_MISMATCH",
            "endpoint last policy input hash is invalid for {} {}".format(
                task_id, policy
            ),
        )
    return normalized


def _normalize_endpoint_check(
    check: Any, *, task_id: str, repeat: int, policy: str, done_reason: str
) -> Dict[str, Any]:
    """Validate one replay-verification wrapper and retain its actual summary."""

    if not isinstance(check, Mapping) or not isinstance(check.get("actual"), Mapping):
        raise _failure(
            "ENDPOINT_SUMMARY_CONTRACT_MISMATCH",
            "replay verification must contain actual for {} {}".format(task_id, policy),
        )
    if check.get("task_id") != task_id or check.get("policy") != policy:
        raise _failure(
            "ENDPOINT_SUMMARY_CONTRACT_MISMATCH",
            "replay verification identity differs for {} {}".format(task_id, policy),
        )
    wrapper_repeat = check.get("repeat")
    if (
        isinstance(wrapper_repeat, bool)
        or not isinstance(wrapper_repeat, int)
        or wrapper_repeat + 1 != repeat
    ):
        raise _failure(
            "ENDPOINT_SUMMARY_CONTRACT_MISMATCH",
            "replay verification repeat differs for {} {}: wrapper={}/selection={}".format(
                task_id, policy, wrapper_repeat, repeat
            ),
        )
    actual = _validate_endpoint_actual(
        check["actual"], task_id=task_id, policy=policy, done_reason=done_reason
    )
    steps = int(actual["steps"])
    expected_counts = {"planned_rows": steps * 8, "executed_rows": steps * 4 + 1}
    count_mismatch = {
        name: {"expected": expected, "actual": check.get(name)}
        for name, expected in expected_counts.items()
        if check.get(name) != expected
    }
    if count_mismatch:
        raise _failure(
            "ENDPOINT_SUMMARY_CONTRACT_MISMATCH",
            "endpoint row counts differ for {} {}: {}".format(
                task_id, policy, json.dumps(count_mismatch, sort_keys=True)
            ),
        )
    return actual


def _selected_expected_summary(
    selected: Mapping[str, Any], lane: str
) -> Dict[str, Any]:
    if lane == "u0":
        field, policy, reason_field = "expected_bc_summary", "bc", "bc_done_reason"
    elif lane == "seed3-u32":
        field, policy, reason_field = (
            "expected_seed3_summary",
            "rl",
            "seed3_done_reason",
        )
    else:
        raise ValueError("unsupported D0.5 lane: {}".format(lane))
    summary = selected.get(field)
    if isinstance(summary, Mapping) and "actual" in summary:
        raise _failure(
            "ENDPOINT_SUMMARY_CONTRACT_MISMATCH",
            "{} must contain a normalized endpoint summary, not verification wrapper".format(
                field
            ),
        )
    return _validate_endpoint_actual(
        summary,
        task_id=str(selected.get("task_id")),
        policy=policy,
        done_reason=str(selected.get(reason_field)),
    )


def _validate_capture_selection(selection: Mapping[str, Any]) -> int:
    targets = selection.get("targets")
    if not isinstance(targets, list) or not targets:
        raise _failure("D05B_POPULATION_EMPTY", "selection contains no targets")
    fillers = selection.get("fillers")
    if not isinstance(fillers, list) or not fillers:
        raise _failure("D05B_FILLER_EMPTY", "selection contains no fillers")
    identities = set()
    summary_count = 0
    for collection_name, entries in (("target", targets), ("filler", fillers)):
        for selected in entries:
            if not isinstance(selected, Mapping):
                raise _failure(
                    "ENDPOINT_SUMMARY_CONTRACT_MISMATCH",
                    "{} entry is not an object".format(collection_name),
                )
            repeat = selected.get("repeat")
            if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 1:
                raise _failure(
                    "ENDPOINT_SUMMARY_CONTRACT_MISMATCH",
                    "{} repeat must be a positive integer".format(collection_name),
                )
            identity = (str(selected.get("task_id")), repeat)
            if identity in identities:
                raise _failure(
                    "PAIRED_IDENTITY_MISMATCH",
                    "duplicate {} {}".format(collection_name, identity),
                )
            identities.add(identity)
            for lane in ("u0", "seed3-u32"):
                _selected_expected_summary(selected, lane)
                summary_count += 1
    if selection.get("target_repeat_count") != len(targets):
        raise _failure(
            "D05B_CAPTURE_CARDINALITY_MISMATCH",
            "selection target count differs from declared count",
        )
    if selection.get("filler_repeat_count") != len(fillers):
        raise _failure(
            "D05B_CAPTURE_CARDINALITY_MISMATCH",
            "selection filler count differs from declared count",
        )
    if (
        len(targets) != D05_EXPECTED_TARGET_REPEATS
        or len(fillers) != D05_EXPECTED_FILLER_REPEATS
    ):
        raise _failure(
            "D05B_CAPTURE_CARDINALITY_MISMATCH",
            "selection fixed cardinality differs: targets={}/{} fillers={}/{}".format(
                len(targets),
                D05_EXPECTED_TARGET_REPEATS,
                len(fillers),
                D05_EXPECTED_FILLER_REPEATS,
            ),
        )
    return summary_count


def _geometry_source(
    parent_request: Mapping[str, Any],
    world: str,
    layera_geometry: Path,
    rl_test_geometry: Path,
) -> Tuple[Path, str, str]:
    source = parent_request["resolved_inputs"]["obstacle_sources"][world]
    root = layera_geometry if source["split"] == "layerA" else rl_test_geometry
    path = Path(root) / Path(source["path"]).name
    if not path.is_file() or file_sha256(path) != source["sha256"]:
        raise _failure("STAGE_INPUT_HASH_MISMATCH", "geometry changed: {}".format(path))
    return path.resolve(), str(source["sha256"]), str(source["split"])


def _parent_entry(
    row: Any,
    *,
    config_entry: Mapping[str, Any],
    geometry_path: Path,
    geometry_sha256: str,
    outcome_class: str,
    done_reasons: Mapping[str, str],
    summaries: Mapping[Tuple[str, int, str], Mapping[str, Any]],
    task_null: Mapping[int, Mapping[str, Any]],
) -> Dict[str, Any]:
    bc_trace = parse_trace_csv(row.trace_paths[0])
    rl_trace = parse_trace_csv(row.trace_paths[3])
    start = np.asarray(rl_trace.header["start"], dtype=np.float64)
    goal = np.asarray(rl_trace.header["goal"], dtype=np.float64)
    bc_features = extract_decision_features(bc_trace.planned, start=start, goal=goal)
    rl_features = extract_decision_features(rl_trace.planned, start=start, goal=goal)
    k_div = find_first_persistent_divergence(bc_features, rl_features, task_null)
    bc_summary = summaries.get((row.task_id, int(row.repeat), "bc"))
    rl_summary = summaries.get((row.task_id, int(row.repeat), "rl"))
    if bc_summary is None or rl_summary is None:
        raise _failure(
            "PAIRED_IDENTITY_MISMATCH",
            "missing endpoint summary {} r{}".format(row.task_id, row.repeat),
        )
    bc_reason = str(done_reasons[str(row.trace_paths[0])])
    rl_reason = str(done_reasons[str(row.trace_paths[3])])
    bc_expected = _normalize_endpoint_check(
        bc_summary,
        task_id=str(row.task_id),
        repeat=int(row.repeat),
        policy="bc",
        done_reason=bc_reason,
    )
    rl_expected = _normalize_endpoint_check(
        rl_summary,
        task_id=str(row.task_id),
        repeat=int(row.repeat),
        policy="rl",
        done_reason=rl_reason,
    )
    return {
        "task_id": str(row.task_id),
        "repeat": int(row.repeat),
        "env_reset_seed": int(row.env_reset_seed),
        "world": str(row.world),
        "split": str(row.split),
        "role": str(row.role),
        "outcome_class": outcome_class,
        "bc_done_reason": bc_reason,
        "seed3_done_reason": rl_reason,
        "bc_trace_path": str(Path(row.trace_paths[0]).resolve()),
        "seed3_trace_path": str(Path(row.trace_paths[3]).resolve()),
        "bc_trace_sha256": str(row.bc_trace_sha256),
        "seed3_trace_sha256": str(row.rl_trace_sha256[2]),
        "episode_config": dict(config_entry["payload"]),
        "config_hash": str(config_entry["config_hash"]),
        "geometry_path": str(geometry_path),
        "geometry_sha256": geometry_sha256,
        "expected_bc_summary": bc_expected,
        "expected_seed3_summary": rl_expected,
        "k_div": k_div,
        "decision_count": min(len(bc_features), len(rl_features)),
        "window_indices": list(
            trace_window_indices(k_div, decision_count=min(len(bc_features), len(rl_features)))
        ),
    }


def _run_select(args: argparse.Namespace) -> int:
    _check_new_output(args.output_dir)
    verify_sha256_manifest(args.d1f_final_dir)
    verify_sha256_manifest(args.d1f_reaudit_dir)
    final_verdict = load_strict_json(Path(args.d1f_final_dir) / "verdict.json")
    reaudit_verdict = load_strict_json(Path(args.d1f_reaudit_dir) / "verdict.json")
    reaudit_identity = load_strict_json(
        Path(args.d1f_reaudit_dir) / "reaudit_identity.json"
    )
    if final_verdict.get("status") != "GOAL_CONTROL_PRESERVED_ON_FAILURE_STATES":
        raise _failure(
            "D05B_TRIGGER_MISMATCH",
            "expected GOAL_CONTROL_PRESERVED_ON_FAILURE_STATES, got {}".format(
                final_verdict.get("status")
            ),
        )
    if reaudit_verdict.get("status") != "GOAL_CONTROL_PRESERVED_ON_FAILURE_STATES":
        raise _failure(
            "D05B_TRIGGER_MISMATCH",
            "v2a reaudit verdict is {}".format(reaudit_verdict.get("status")),
        )
    final_selection = load_strict_json(Path(args.d1f_final_dir) / "failure_state_selection.json")
    parent_manifest_hash = file_sha256(Path(args.d1f_final_dir) / "manifest_sha256.txt")
    if reaudit_identity.get("parent_final_manifest_sha256") != parent_manifest_hash:
        raise _failure(
            "D05B_TRIGGER_MISMATCH",
            "v2a reaudit does not bind the supplied D1-F final directory",
        )
    if reaudit_identity.get("run_lineage_id") != final_selection.get("run_lineage_id"):
        raise _failure("D05B_TRIGGER_MISMATCH", "v2/v2a run lineage differs")
    expected_breaks = {
        (str(row["task_id"]), int(row["repeat"])) for row in final_selection["selected"]
    }

    seed_dirs = [args.fixed32_seed1_dir, args.fixed32_seed2_dir, args.fixed32_seed3_dir]
    parent = load_fixed32_parent(
        seed_dirs,
        {"layerA": args.layera_geometry, "rl_test": args.rl_test_geometry},
    )
    done_reasons, summaries = _endpoint_summaries(seed_dirs)
    configs, config_sources = _load_episode_configs(args.episode_config_dir)
    request = load_strict_json(Path(args.fixed32_seed1_dir) / "replay_request.json")

    repeats_by_task = {}
    for row in parent:
        bc = parse_trace_csv(row.trace_paths[0])
        start = np.asarray(bc.header["start"], dtype=np.float64)
        goal = np.asarray(bc.header["goal"], dtype=np.float64)
        repeats_by_task.setdefault(str(row.task_id), {})[int(row.repeat)] = extract_decision_features(
            bc.planned, start=start, goal=goal
        )
    task_nulls = {
        task_id: build_bc_repeat_null(repeats)
        for task_id, repeats in sorted(repeats_by_task.items())
    }

    targets = []
    fillers = []
    outcome_rows = []
    actual_breaks = set()
    for row in parent:
        bc_reason = done_reasons[str(row.trace_paths[0])]
        rl_reason = done_reasons[str(row.trace_paths[3])]
        outcome = classify_endpoint_pair(bc_reason, rl_reason)
        if outcome == "break":
            actual_breaks.add((str(row.task_id), int(row.repeat)))
        config_entry = configs.get(str(row.task_id))
        if config_entry is None:
            raise _failure("EPISODE_CONFIG_IDENTITY_MISMATCH", str(row.task_id))
        geometry_path, geometry_hash, split = _geometry_source(
            request, str(row.world), args.layera_geometry, args.rl_test_geometry
        )
        if split != str(row.split):
            raise _failure("PAIRED_IDENTITY_MISMATCH", "split differs for {}".format(row.task_id))
        entry = _parent_entry(
            row,
            config_entry=config_entry,
            geometry_path=geometry_path,
            geometry_sha256=geometry_hash,
            outcome_class=outcome,
            done_reasons=done_reasons,
            summaries=summaries,
            task_null=task_nulls[str(row.task_id)],
        )
        outcome_rows.append(
            {
                "task_id": entry["task_id"],
                "repeat": entry["repeat"],
                "role": entry["role"],
                "split": entry["split"],
                "bc_done_reason": bc_reason,
                "seed3_done_reason": rl_reason,
                "outcome_class": outcome,
                "k_div": entry["k_div"],
            }
        )
        if outcome in ("break", "rescue"):
            targets.append(entry)
        else:
            fillers.append(entry)
    if actual_breaks != expected_breaks:
        missing = sorted(expected_breaks - actual_breaks)
        extra = sorted(actual_breaks - expected_breaks)
        raise _failure(
            "PAIRED_IDENTITY_MISMATCH",
            "D1-F break set differs: missing={} extra={}".format(missing[:5], extra[:5]),
        )
    if not any(row["outcome_class"] == "rescue" for row in targets):
        raise _failure("D05B_POPULATION_EMPTY", "seed3 fixed32 contains no endpoint rescue")
    if not fillers:
        raise _failure("D05B_FILLER_EMPTY", "no stable fixed32 episode is available")

    targets.sort(key=lambda row: (row["task_id"], row["repeat"]))
    fillers.sort(
        key=lambda row: hashlib.sha256(
            "{}::{}".format(row["task_id"], row["repeat"]).encode("utf-8")
        ).hexdigest()
    )
    lineage = "d05b-{}".format(uuid.uuid4().hex)
    selection = {
        "schema": "routea_d05b_selection_v1",
        "run_lineage_id": lineage,
        "trigger_status": final_verdict["status"],
        "actor_rerun": False,
        "trajectory_source": "frozen_fixed32_action_replay",
        "target_repeat_count": len(targets),
        "target_task_count": len({row["task_id"] for row in targets}),
        "break_repeat_count": sum(row["outcome_class"] == "break" for row in targets),
        "rescue_repeat_count": sum(row["outcome_class"] == "rescue" for row in targets),
        "filler_repeat_count": len(fillers),
        "targets": targets,
        "fillers": fillers,
        "normalization_contract": {
            "mode": "full_joint_accum_batch1280",
            "transitions_per_diagnostic_buffer": 1024,
            "denoising_steps": 10,
            "logical_batch_pairs": 1280,
            "microbatch_pairs": 128,
            "target_episode_boundary": "natural terminal/truncated only",
            "collection_boundary_allowed_on": "filler only",
        },
    }
    identity = {
        "schema": "routea_d05b_input_identity_v1",
        "run_lineage_id": lineage,
        "d1f_final_manifest_sha256": parent_manifest_hash,
        "d1f_final_verdict_sha256": file_sha256(Path(args.d1f_final_dir) / "verdict.json"),
        "d1f_reaudit_manifest_sha256": file_sha256(
            Path(args.d1f_reaudit_dir) / "manifest_sha256.txt"
        ),
        "d1f_reaudit_identity_sha256": file_sha256(
            Path(args.d1f_reaudit_dir) / "reaudit_identity.json"
        ),
        "fixed32_identity": {
            "seed{}".format(index): {
                "replay_request_sha256": file_sha256(Path(path) / "replay_request.json"),
                "replay_verification_sha256": file_sha256(
                    Path(path) / "replay_verification.json"
                ),
                "trajectory_count": 1280,
                "trajectory_hash_binding": "replay_verification.checks[].trajectory_sha256",
            }
            for index, path in enumerate(seed_dirs, start=1)
        },
        "episode_config_sources": config_sources,
        "d05_source": _d05_source_manifest(),
    }
    with AtomicOutputDirectory(args.output_dir) as staging:
        write_strict_json(staging / "selection.json", selection)
        write_strict_json(staging / "bc_repeat_null.json", serialize_bc_repeat_null(task_nulls))
        write_strict_json(staging / "input_identity.json", identity)
        write_csv_rows(staging / "paired_episode_outcomes.csv", outcome_rows)
        publish_stage_manifest(staging, _d05_source_manifest())
    _single_line(
        {
            "status": "SELECT_D05B_COMPLETE",
            "targets": len(targets),
            "tasks": len({row["task_id"] for row in targets}),
            "breaks": selection["break_repeat_count"],
            "rescues": selection["rescue_repeat_count"],
            "output_dir": str(Path(args.output_dir).resolve()),
        }
    )
    return 0


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return _jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _failure("INVALID_JSON_ARTIFACT", "non-finite float")
        return value
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    raise _failure("INVALID_JSON_ARTIFACT", "unsupported value {}".format(type(value).__name__))


def _angle_error(goal: np.ndarray, position: np.ndarray, yaw: float) -> float:
    delta = np.asarray(goal, dtype=np.float64)[:2] - np.asarray(position, dtype=np.float64)[:2]
    if float(np.linalg.norm(delta)) <= 1e-12:
        return 0.0
    target = math.atan2(float(delta[1]), float(delta[0]))
    return float(math.atan2(math.sin(target - yaw), math.cos(target - yaw)))


def _first_sphere_entry(
    executed_path: Any, goal: Any, radius: float = 0.5
) -> Tuple[bool, Optional[int], Optional[float]]:
    points = np.asarray(executed_path, dtype=np.float64)
    center = np.asarray(goal, dtype=np.float64)
    if points.shape != (5, 3):
        raise ValueError("executed H4 path must have shape (5,3)")
    if center.shape != (3,) or not np.isfinite(points).all() or not np.isfinite(center).all():
        raise ValueError("goal-sphere inputs must be finite")
    r2 = float(radius) ** 2
    for segment in range(len(points) - 1):
        start = points[segment] - center
        end = points[segment + 1] - center
        if float(np.dot(start, start)) <= r2:
            return True, segment, 0.0
        direction = end - start
        a = float(np.dot(direction, direction))
        if a <= 0.0:
            continue
        b = 2.0 * float(np.dot(start, direction))
        c = float(np.dot(start, start)) - r2
        discriminant = b * b - 4.0 * a * c
        if discriminant < 0.0:
            continue
        root = math.sqrt(max(0.0, discriminant))
        candidates = sorted(((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)))
        for fraction in candidates:
            if 0.0 <= fraction <= 1.0:
                return True, segment, float(fraction)
    return False, None, None


def _action_view_to_list(
    value: Any, *, expected_shape: Tuple[int, int], label: str
) -> List[List[float]]:
    """Serialize a float32 action view without assuming Torch or NumPy storage."""

    if torch.is_tensor(value):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if tuple(array.shape) != tuple(expected_shape):
        raise _failure(
            "ACTION_CONTRACT_MISMATCH",
            "{} shape {} != {}".format(label, tuple(array.shape), expected_shape),
            EXIT_REPLAY,
        )
    if array.dtype != np.dtype(np.float32) or not bool(np.isfinite(array).all()):
        raise _failure(
            "ACTION_CONTRACT_MISMATCH",
            "{} must be finite float32; got {}".format(label, array.dtype),
            EXIT_REPLAY,
        )
    return np.ascontiguousarray(array).tolist()


def _policy_action_8x11(action: Any) -> List[List[float]]:
    """Serialize only the policy horizon, never the authoritative 16-step tensor."""

    return _action_view_to_list(
        action.policy[0], expected_shape=(8, 11), label="policy action"
    )


def _executed_action_4x11(action: Any) -> List[List[float]]:
    """Serialize the executed H4 action view with the same backend-neutral contract."""

    return _action_view_to_list(
        action.rollout_executed[0],
        expected_shape=(4, 11),
        label="rollout executed action",
    )


def _collect_frozen_episode(
    system: Any,
    config: Any,
    selected: Mapping[str, Any],
    *,
    lane: str,
) -> List[Dict[str, Any]]:
    """Replay one frozen fixed32 trajectory and collect production reward/critic facts."""

    from scripts.endpoint_trajectory_trace_v2 import TraceAdapter
    from scripts.evaluate_dppo_uav_formal_endpoint import (
        _build_adapter,
        canonical_done_reason,
        canonical_episode_config,
        reset_fixed_episode,
    )
    from scripts.probe_routea_goal_sensitivity import (
        _assert_bitwise_trajectory_match,
        _frozen_plan_action,
        _policy_seed,
        _repeat_identity,
    )

    if lane == "u0":
        path_field, hash_field, header_policy = "bc_trace_path", "bc_trace_sha256", "bc"
    elif lane == "seed3-u32":
        path_field, hash_field, header_policy = (
            "seed3_trace_path",
            "seed3_trace_sha256",
            "rl",
        )
    else:
        raise ValueError("unsupported D0.5 lane: {}".format(lane))
    expected = _selected_expected_summary(selected, lane)

    geometry_path = Path(selected["geometry_path"])
    frozen_path = Path(selected[path_field])
    if file_sha256(geometry_path) != selected["geometry_sha256"]:
        raise _failure("SNAPSHOT_REPLAY_MISMATCH", "geometry changed: {}".format(geometry_path), EXIT_REPLAY)
    if file_sha256(frozen_path) != selected[hash_field]:
        raise _failure("SNAPSHOT_REPLAY_MISMATCH", "trace changed: {}".format(frozen_path), EXIT_REPLAY)
    frozen = parse_trace_csv(frozen_path)
    if (
        frozen.header.get("task_id") != selected["task_id"]
        or frozen.header.get("policy") != header_policy
        or int(frozen.header.get("repeat", -1)) != int(selected["repeat"])
    ):
        raise _failure(
            "SNAPSHOT_REPLAY_MISMATCH",
            "frozen identity mismatch {} {}".format(selected["task_id"], lane),
            EXIT_REPLAY,
        )

    fixed = canonical_episode_config(selected["episode_config"])
    derived_reset, action_offset = _repeat_identity(fixed.config_hash, int(selected["repeat"]))
    reset_seed = int(selected["env_reset_seed"])
    if derived_reset is not None and int(derived_reset) != reset_seed:
        raise _failure(
            "SNAPSHOT_REPLAY_MISMATCH",
            "repeat/reset mismatch {} r{}".format(selected["task_id"], selected["repeat"]),
            EXIT_REPLAY,
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
    rows = []
    terminal_reason = None
    last_provenance_hash = None
    decision = 0
    try:
        while True:
            before_position = np.asarray(adapter.environment._drone_pos, dtype=np.float64).copy()
            before_velocity = np.asarray(adapter.environment._drone_vel, dtype=np.float64).copy()
            before_yaw = float(adapter.environment._drone_yaw)
            goal = np.asarray(adapter.environment.goal_w, dtype=np.float64).copy()
            action_seed = _policy_seed(fixed.config_hash, decision, action_offset)
            torch.manual_seed(int(action_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(action_seed))
            with torch.no_grad():
                encoded = system.observation_adapter.encode(
                    observation,
                    context["pointcloud_meta_history"],
                    episode_identity_hash=context["episode_identity_hash"],
                    decision_index=context["decision_index"],
                )
                system.observation_adapter.verify(encoded)
                value = float(system.critic(encoded.condition).reshape(-1)[0].item())
            provenance = encoded.provenance
            last_provenance_hash = getattr(provenance, "policy_input_hash", None)
            current_meta_history = context.get("pointcloud_meta_history", ())
            current_last_meta = current_meta_history[-1] if current_meta_history else None
            action = _frozen_plan_action(
                frozen.planned,
                decision_step=decision + 1,
                before={
                    "position": before_position,
                    "velocity_reference": before_velocity,
                    "yaw": before_yaw,
                },
                exact_bitwise=True,
            )
            trace.policy_inputs.append(
                {
                    "action_seed": int(action_seed),
                    "provenance": None if provenance is None else asdict(provenance),
                }
            )
            next_observation, reward, terminated, truncated, info = trace.step(action)
            next_encoded = system.observation_adapter.encode(
                next_observation,
                info["pointcloud_meta_history"],
                episode_identity_hash=info["episode_identity_hash"],
                decision_index=info["decision_index"],
            )
            system.observation_adapter.verify(next_encoded)
            with torch.no_grad():
                next_value = (
                    0.0
                    if terminated
                    else float(system.critic(next_encoded.condition).reshape(-1)[0].item())
                )
            reward_contract = validate_reward_trace_row(
                {"reward": reward, "reward_terms": info.get("reward_terms")}
            )
            after_position = np.asarray(adapter.environment._drone_pos, dtype=np.float64).copy()
            after_yaw = float(adapter.environment._drone_yaw)
            executed_path = np.asarray(info.get("executed_path"), dtype=np.float64)
            entered, segment, fraction = _first_sphere_entry(executed_path, goal, radius=0.5)
            facts = info.get("transition_facts")
            physical_failure = bool(getattr(facts, "physical_failure", False))
            metrics = {
                str(key): _jsonable(metric)
                for key, metric in info.get("reward_metrics", {}).items()
                if metric is None or isinstance(metric, (bool, int, float, str, np.generic))
            }
            rows.append(
                {
                    "schema": "routea_d05b_transition_v1",
                    "lane": lane,
                    "target": True,
                    "task_id": selected["task_id"],
                    "repeat": int(selected["repeat"]),
                    "outcome_class": selected["outcome_class"],
                    "role": selected["role"],
                    "split": selected["split"],
                    "world": selected["world"],
                    "env_reset_seed": reset_seed,
                    "decision_index": decision,
                    "action_seed": int(action_seed),
                    "episode_identity_hash": str(context["episode_identity_hash"]),
                    "policy_input_hash": getattr(provenance, "policy_input_hash", None),
                    "raw_frame_hash": getattr(provenance, "raw_frame_hash", None),
                    "normalized_history_hash": getattr(provenance, "normalized_history_hash", None),
                    "condition_sha256": canonical_tree_sha256(encoded.condition),
                    "pointcloud_sha256": canonical_array_sha256(observation["point_cloud"]),
                    "pointcloud_raw_frame_sha256": getattr(
                        current_last_meta, "raw_array_hash", None
                    ),
                    "position_before_xyz": before_position.tolist(),
                    "position_after_xyz": after_position.tolist(),
                    "yaw_before": before_yaw,
                    "yaw_after": after_yaw,
                    "goal_xyz": goal.tolist(),
                    "goal_distance_before_3d": float(np.linalg.norm(goal - before_position)),
                    "goal_distance_after_3d": float(np.linalg.norm(goal - after_position)),
                    "heading_error_before": _angle_error(goal, before_position, before_yaw),
                    "heading_error_after": _angle_error(goal, after_position, after_yaw),
                    "executed_displacement_xyz": (after_position - before_position).tolist(),
                    "policy_action_8x11": _policy_action_8x11(action),
                    "executed_action_4x11": _executed_action_4x11(action),
                    "executed_h4_path_5x3": executed_path.tolist(),
                    "reward": reward_contract["reward"],
                    "reward_terms": reward_contract["reward_terms"],
                    "reward_groups": reward_group_totals(reward_contract["reward_terms"]),
                    "reward_metrics": metrics,
                    "reward_sum_error_abs": reward_contract["reward_sum_error_abs"],
                    "value": value,
                    "next_value": next_value,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "collection_boundary": False,
                    "terminal_facts": _jsonable(facts) if facts is not None else None,
                    "static_clearance_m": metrics.get("d_min"),
                    "entered_goal_sphere_during_executed_H4": bool(entered),
                    "entered_goal_sphere_safe": bool(entered and not physical_failure),
                    "goal_sphere_first_segment": segment,
                    "goal_sphere_first_fraction": fraction,
                }
            )
            decision += 1
            if terminated or truncated:
                raw_reason = getattr(facts, "raw_done_reason", None) or getattr(
                    adapter.environment, "_done_reason", None
                )
                terminal_reason = canonical_done_reason(
                    raw_reason,
                    adapter.environment._drone_pos[2],
                    terminated,
                    truncated,
                )
                break
            observation, context = next_observation, info
    finally:
        trace.close()

    _assert_bitwise_trajectory_match(
        task_id="{}::{}".format(selected["task_id"], lane),
        rebuilt_planned=np.asarray(trace.planned_rows, dtype=np.float64),
        frozen_planned=frozen.planned,
        rebuilt_executed=np.asarray(trace.executed_rows, dtype=np.float64),
        frozen_executed=frozen.executed,
    )
    if terminal_reason != expected["done_reason"] or decision != int(expected["steps"]):
        raise _failure(
            "SNAPSHOT_REPLAY_MISMATCH",
            "terminal mismatch {} {} reason={}/{} steps={}/{}".format(
                selected["task_id"],
                lane,
                terminal_reason,
                expected["done_reason"],
                decision,
                expected["steps"],
            ),
            EXIT_REPLAY,
        )
    expected_last = (expected.get("observation_validity_summary") or {}).get(
        "last_policy_input_hash"
    )
    if expected_last is not None and expected_last != last_provenance_hash:
        raise _failure(
            "SNAPSHOT_REPLAY_MISMATCH",
            "last policy input mismatch {} {}".format(selected["task_id"], lane),
            EXIT_REPLAY,
        )
    return rows


def _credit_schedule_seed(lineage: str, lane: str, batch_index: int) -> int:
    payload = "{}::{}::{}".format(lineage, lane, int(batch_index)).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def _copy_filler_prefix(
    episode: Sequence[Mapping[str, Any]],
    *,
    count: int,
    filler_instance: int,
) -> List[Dict[str, Any]]:
    if count < 1 or count > len(episode):
        raise ValueError("invalid filler prefix length")
    rows = []
    for index, source in enumerate(episode[:count]):
        row = dict(source)
        row["target"] = False
        row["filler_instance"] = int(filler_instance)
        row["source_task_id"] = row["task_id"]
        row["task_id"] = "FILLER::{}::{}".format(row["task_id"], filler_instance)
        rows.append(row)
    if count < len(episode):
        rows[-1]["collection_boundary"] = True
        rows[-1]["terminated"] = False
        rows[-1]["truncated"] = False
        terminal = dict(rows[-1].get("terminal_facts") or {})
        terminal["diagnostic_collection_boundary"] = True
        rows[-1]["terminal_facts"] = terminal
    return rows


def _build_credit_batches(
    target_episodes: Sequence[Sequence[Mapping[str, Any]]],
    *,
    filler_entries: Sequence[Mapping[str, Any]],
    filler_loader: Any,
    lineage: str,
    lane: str,
    gamma: float,
    gae_lambda: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not target_episodes:
        raise _failure("D05B_POPULATION_EMPTY", "no target episodes for {}".format(lane))
    if not filler_entries:
        raise _failure("D05B_FILLER_EMPTY", "no filler entries")
    packed = []
    current = []
    for episode in target_episodes:
        if len(episode) > 1024:
            raise _failure("D05B_EPISODE_TOO_LONG", "{} transition episode".format(len(episode)))
        if current and len(current) + len(episode) > 1024:
            packed.append(current)
            current = []
        current.extend(dict(row) for row in episode)
    if current:
        packed.append(current)

    filler_cache = {}
    filler_cursor = 0
    filler_instance = 0
    population = []
    audits = []
    for batch_index, batch_targets in enumerate(packed):
        batch = list(batch_targets)
        while len(batch) < 1024:
            entry = filler_entries[filler_cursor % len(filler_entries)]
            filler_cursor += 1
            key = (str(entry["task_id"]), int(entry["repeat"]))
            if key not in filler_cache:
                filler_cache[key] = filler_loader(entry)
            source_episode = filler_cache[key]
            take = min(1024 - len(batch), len(source_episode))
            filler_instance += 1
            batch.extend(
                _copy_filler_prefix(
                    source_episode,
                    count=take,
                    filler_instance=filler_instance,
                )
            )
        if len(batch) != 1024:
            raise RuntimeError("diagnostic credit batch cardinality drift")
        for transition_index, row in enumerate(batch):
            row["credit_batch_index"] = batch_index
            row["transition_index"] = transition_index
            row["transition_id"] = "{}::b{}::t{}".format(lane, batch_index, transition_index)

        rewards = torch.tensor([row["reward"] for row in batch], dtype=torch.float32)
        values = torch.tensor([row["value"] for row in batch], dtype=torch.float32)
        next_values = torch.tensor([row["next_value"] for row in batch], dtype=torch.float32)
        terminated = torch.tensor([row["terminated"] for row in batch], dtype=torch.bool)
        effective_truncated = torch.tensor(
            [bool(row["truncated"] or row["collection_boundary"]) for row in batch],
            dtype=torch.bool,
        )
        advantages, returns = compute_gae(
            rewards,
            values,
            next_values,
            terminated,
            effective_truncated,
            float(gamma),
            float(gae_lambda),
        )
        schedule_seed = _credit_schedule_seed(lineage, lane, batch_index)
        normalized = s2_normalize_advantages(
            advantages.tolist(),
            [row["transition_id"] for row in batch],
            schedule_seed,
        )
        for index, row in enumerate(batch):
            bootstrap = 0.0 if row["terminated"] else 1.0
            row["td_delta"] = float(
                rewards[index] + float(gamma) * next_values[index] * bootstrap - values[index]
            )
            row["raw_advantage"] = float(advantages[index])
            row["return_target"] = float(returns[index])
            exposures = normalized["by_transition"][row["transition_id"]]
            row["normalized_advantage_s2_by_k"] = exposures
            row["normalized_advantage_s2_mean"] = float(
                np.mean([item["normalized_advantage"] for item in exposures], dtype=np.float64)
            )
            row["normalized_advantage_s2_positive_fraction"] = float(
                np.mean([item["normalized_advantage"] > 0.0 for item in exposures])
            )
        target_count = sum(bool(row["target"]) for row in batch)
        filler_count = 1024 - target_count
        audit = dict(normalized["audit"])
        audit.update(
            {
                "lane": lane,
                "credit_batch_index": batch_index,
                "target_transition_count": target_count,
                "filler_transition_count": filler_count,
                "target_collection_boundary_count": sum(
                    bool(row["target"] and row["collection_boundary"]) for row in batch
                ),
                "filler_collection_boundary_count": sum(
                    bool((not row["target"]) and row["collection_boundary"]) for row in batch
                ),
                "logical_batches": normalized["logical_batches"],
            }
        )
        if audit["target_collection_boundary_count"] != 0:
            raise _failure(
                "D05B_TARGET_BOUNDARY_VIOLATION",
                "target episode was cut by collection boundary",
            )
        audits.append(audit)
        population.extend(batch)
    return population, audits


def _run_capture(args: argparse.Namespace) -> int:
    _check_new_output(args.output_dir)
    verify_sha256_manifest(args.selection_dir)
    selection = load_strict_json(Path(args.selection_dir) / "selection.json")
    identity = load_strict_json(Path(args.selection_dir) / "input_identity.json")
    if identity["d05_source"]["content_sha256"] != _d05_source_manifest()["content_sha256"]:
        raise _failure("STAGE_INPUT_HASH_MISMATCH", "selection and capture source differ")
    endpoint_summary_count = _validate_capture_selection(selection)

    u0_config, u0_system = load_read_only_system(args.u0_config, args.u0_checkpoint, REPO_ROOT)
    u32_config, u32_system = load_read_only_system(args.u32_config, args.u32_checkpoint, REPO_ROOT)
    validate_zero_effect_lora_state(u0_system.lora_bundle.adapter_state_dict())
    gamma_u0 = float(u0_config.dppo.gamma_environment)
    gamma_u32 = float(u32_config.dppo.gamma_environment)
    lam_u0 = float(u0_config.dppo.gae_lambda)
    lam_u32 = float(u32_config.dppo.gae_lambda)
    if (gamma_u0, lam_u0) != (gamma_u32, lam_u32):
        raise _failure(
            "CHECKPOINT_CONTRACT_MISMATCH",
            "u0/u32 gamma or GAE lambda differs",
            EXIT_CHECKPOINT,
        )

    immutable = [Path(args.selection_dir) / name for name in D05_SELECTION_FILES]
    outer_rng = capture_rng_state()
    guards = {
        "u0": install_read_only_tripwires(u0_system, immutable_files=immutable),
        "seed3-u32": install_read_only_tripwires(u32_system, immutable_files=immutable),
    }
    systems = {"u0": (u0_system, u0_config), "seed3-u32": (u32_system, u32_config)}
    population_rows = []
    batch_audits = []
    try:
        for lane in ("u0", "seed3-u32"):
            system, config = systems[lane]
            target_episodes = [
                _collect_frozen_episode(system, config, selected, lane=lane)
                for selected in selection["targets"]
            ]

            def load_filler(entry: Mapping[str, Any], _system=system, _config=config, _lane=lane):
                return _collect_frozen_episode(_system, _config, entry, lane=_lane)

            rows, audits = _build_credit_batches(
                target_episodes,
                filler_entries=selection["fillers"],
                filler_loader=load_filler,
                lineage=selection["run_lineage_id"],
                lane=lane,
                gamma=gamma_u0,
                gae_lambda=lam_u0,
            )
            population_rows.extend(rows)
            batch_audits.extend(audits)
    finally:
        restore_rng_state(outer_rng)

    tripwire = {
        "schema": "routea_d05b_tripwire_v1",
        "status": "READ_ONLY_VERIFIED",
        "lanes": {
            lane: guard.verify(require_rng_unchanged=True) for lane, guard in guards.items()
        },
    }
    target_rows = [row for row in population_rows if row["target"]]
    expected_target_rows = sum(
        int(_selected_expected_summary(selected, lane)["steps"])
        for lane in ("u0", "seed3-u32")
        for selected in selection["targets"]
    )
    if len(target_rows) != expected_target_rows:
        raise _failure(
            "D05B_CAPTURE_CARDINALITY_MISMATCH",
            "target transitions={} expected={}".format(len(target_rows), expected_target_rows),
        )
    with AtomicOutputDirectory(args.output_dir) as staging:
        for name in D05_SELECTION_FILES:
            shutil.copy2(Path(args.selection_dir) / name, staging / name)
        write_jsonl(staging / "paired_trace_transitions.jsonl", target_rows)
        write_jsonl(staging / "credit_population.jsonl", population_rows)
        write_strict_json(
            staging / "credit_batch_audit.json",
            {
                "schema": "routea_d05b_credit_batch_audit_v1",
                "run_lineage_id": selection["run_lineage_id"],
                "target_transition_count": len(target_rows),
                "population_transition_count": len(population_rows),
                "validated_endpoint_summary_count": endpoint_summary_count,
                "batch_count": len(batch_audits),
                "batches": batch_audits,
                "formal_normalization": "S2 full_joint_accum_batch1280",
                "probe_normalization_used": False,
                "actor_rerun": False,
            },
        )
        write_strict_json(staging / "tripwire_verification.json", tripwire)
        publish_stage_manifest(staging, _d05_source_manifest())
    _single_line(
        {
            "status": "CAPTURE_D05B_COMPLETE",
            "target_transitions": len(target_rows),
            "population_transitions": len(population_rows),
            "credit_batches": len(batch_audits),
            "output_dir": str(Path(args.output_dir).resolve()),
        }
    )
    return 0


def _episode_key(row: Mapping[str, Any]) -> Tuple[str, int, str]:
    return str(row["task_id"]), int(row["repeat"]), str(row["lane"])


def _mean_or_none(values: Iterable[Any]) -> Optional[float]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(np.asarray(finite, dtype=np.float64))) if finite else None


def _run_analyze(args: argparse.Namespace) -> int:
    _check_new_output(args.output_dir)
    verify_sha256_manifest(args.capture_dir)
    source = load_strict_json(Path(args.capture_dir) / "diagnostic_source_manifest.json")
    if source.get("content_sha256") != _d05_source_manifest()["content_sha256"]:
        raise _failure("STAGE_INPUT_HASH_MISMATCH", "capture and analysis source differ")
    selection = load_strict_json(Path(args.capture_dir) / "selection.json")
    _validate_capture_selection(selection)
    rows = _read_jsonl(Path(args.capture_dir) / "paired_trace_transitions.jsonl")
    if not rows:
        raise _failure("D05B_POPULATION_EMPTY", "capture contains no target transitions")
    by_episode = {}
    max_reward_sum_error = 0.0
    for row in rows:
        validated = validate_reward_trace_row(row)
        max_reward_sum_error = max(max_reward_sum_error, validated["reward_sum_error_abs"])
        key = _episode_key(row)
        by_episode.setdefault(key, []).append(row)
    for key, episode in by_episode.items():
        episode.sort(key=lambda row: int(row["decision_index"]))
        if [int(row["decision_index"]) for row in episode] != list(range(len(episode))):
            raise _failure("PAIRED_IDENTITY_MISMATCH", "non-contiguous decisions: {}".format(key))

    windows = []
    transition_table = []
    safe_labels = []
    repeat_evidence = []
    no_divergence = 0
    for selected in selection["targets"]:
        identity = (str(selected["task_id"]), int(selected["repeat"]))
        u0 = by_episode.get((identity[0], identity[1], "u0"))
        u32 = by_episode.get((identity[0], identity[1], "seed3-u32"))
        if u0 is None or u32 is None:
            raise _failure("PAIRED_IDENTITY_MISMATCH", "missing paired episode {}".format(identity))
        decision_count = int(selected["decision_count"])
        if len(u0) != int(_selected_expected_summary(selected, "u0")["steps"]):
            raise _failure("PAIRED_IDENTITY_MISMATCH", "u0 length differs {}".format(identity))
        if len(u32) != int(_selected_expected_summary(selected, "seed3-u32")["steps"]):
            raise _failure("PAIRED_IDENTITY_MISMATCH", "u32 length differs {}".format(identity))
        common = min(len(u0), len(u32), decision_count)
        for decision in range(common):
            if int(u0[decision]["action_seed"]) != int(u32[decision]["action_seed"]):
                raise _failure(
                    "RNG_PAIRING_MISMATCH",
                    "{} r{} d{}".format(identity[0], identity[1], decision),
                )
        k_div = selected.get("k_div")
        indices = trace_window_indices(k_div, decision_count=common)
        if not indices:
            no_divergence += 1
            repeat_evidence.append(
                {
                    "task_id": identity[0],
                    "repeat": identity[1],
                    "outcome_class": selected["outcome_class"],
                    "mechanism": INCONCLUSIVE,
                    "reason": "no_persistent_divergence",
                    "dominant_positive_group": None,
                }
            )
            continue
        k = int(k_div)
        divergence_row = u32[k]
        normalized = float(divergence_row["normalized_advantage_s2_mean"])
        mechanism = classify_transition_signal(
            float(divergence_row["reward"]),
            float(divergence_row["raw_advantage"]),
            normalized,
        )
        groups = {name: float(value) for name, value in divergence_row["reward_groups"].items()}
        dominant = max(groups, key=lambda name: (groups[name], name))
        if groups[dominant] <= 0.0:
            dominant = None
            if mechanism == "REWARD_OBJECTIVE_SUSPECTED":
                mechanism = INCONCLUSIVE
        deltas = {
            name: float(divergence_row["reward_groups"][name])
            - float(u0[k]["reward_groups"][name])
            for name in REWARD_GROUPS
        }
        repeat_evidence.append(
            {
                "task_id": identity[0],
                "repeat": identity[1],
                "outcome_class": selected["outcome_class"],
                "mechanism": mechanism,
                "reason": "first_persistent_divergence_sign_chain",
                "dominant_positive_group": dominant,
                "dominant_group_delta_vs_u0": None if dominant is None else deltas[dominant],
                "reward": float(divergence_row["reward"]),
                "raw_advantage": float(divergence_row["raw_advantage"]),
                "normalized_advantage_s2_mean": normalized,
                "normalized_advantage_s2_positive_fraction": float(
                    divergence_row["normalized_advantage_s2_positive_fraction"]
                ),
                "k_div": k,
            }
        )
        window_rows = []
        for decision in indices:
            for lane, episode in (("u0", u0), ("seed3-u32", u32)):
                row = episode[decision]
                relative = decision - k
                flat = {
                    "task_id": identity[0],
                    "repeat": identity[1],
                    "role": selected["role"],
                    "split": selected["split"],
                    "outcome_class": selected["outcome_class"],
                    "lane": lane,
                    "decision_index": decision,
                    "relative_to_divergence": relative,
                    "k_div": k,
                    "reward": row["reward"],
                    "value": row["value"],
                    "next_value": row["next_value"],
                    "td_delta": row["td_delta"],
                    "raw_advantage": row["raw_advantage"],
                    "normalized_advantage_s2_mean": row["normalized_advantage_s2_mean"],
                    "normalized_advantage_s2_positive_fraction": row[
                        "normalized_advantage_s2_positive_fraction"
                    ],
                    "goal_distance_before_3d": row["goal_distance_before_3d"],
                    "goal_distance_after_3d": row["goal_distance_after_3d"],
                    "static_clearance_m": row["static_clearance_m"],
                    "heading_error_before": row["heading_error_before"],
                    "heading_error_after": row["heading_error_after"],
                    "executed_displacement_xyz": row["executed_displacement_xyz"],
                    "action_seed": row["action_seed"],
                    "policy_input_hash": row["policy_input_hash"],
                    "credit_batch_index": row["credit_batch_index"],
                }
                for name in TERM_NAMES:
                    flat["term_{}".format(name)] = row["reward_terms"][name]
                for name in REWARD_GROUPS:
                    flat["group_{}".format(name)] = row["reward_groups"][name]
                transition_table.append(flat)
                safe_labels.append(
                    {
                        "task_id": identity[0],
                        "repeat": identity[1],
                        "outcome_class": selected["outcome_class"],
                        "lane": lane,
                        "decision_index": decision,
                        "entered_goal_sphere_during_executed_H4": row[
                            "entered_goal_sphere_during_executed_H4"
                        ],
                        "entered_goal_sphere_safe": row["entered_goal_sphere_safe"],
                        "first_segment": row["goal_sphere_first_segment"],
                        "first_fraction": row["goal_sphere_first_fraction"],
                    }
                )
                window_rows.append(row)
        windows.append(
            {
                "schema": "routea_d05b_first_divergence_window_v1",
                "task_id": identity[0],
                "repeat": identity[1],
                "outcome_class": selected["outcome_class"],
                "role": selected["role"],
                "split": selected["split"],
                "k_div": k,
                "window_start": indices[0],
                "window_end": indices[-1],
                "threshold_source": "same-task five-BC-repeat pairwise Q0.95",
                "mechanism": mechanism,
                "dominant_positive_group": dominant,
                "group_delta_vs_u0_at_divergence": deltas,
                "transitions": window_rows,
            }
        )

    # Context-gating requires the same rising group in both outcome classes.
    # Use unique tasks, not repeat counts, and require two tasks on each side.
    group_tasks = {
        group: {"break": set(), "rescue": set()} for group in REWARD_GROUPS
    }
    for row in repeat_evidence:
        group = row.get("dominant_positive_group")
        outcome = row.get("outcome_class")
        if (
            group in group_tasks
            and outcome in ("break", "rescue")
            and float(row.get("dominant_group_delta_vs_u0") or 0.0) > 0.0
        ):
            group_tasks[group][outcome].add(row["task_id"])
    context_groups = [
        group
        for group, outcomes in group_tasks.items()
        if len(outcomes["break"]) >= 2 and len(outcomes["rescue"]) >= 2
    ]
    if context_groups:
        involved = set()
        for group in context_groups:
            involved.update(group_tasks[group]["break"])
            involved.update(group_tasks[group]["rescue"])
        for row in repeat_evidence:
            if row["task_id"] in involved and row.get("dominant_positive_group") in context_groups:
                row["mechanism"] = CONTEXT_GATING_SUSPECTED
                row["reason"] = "same_rising_group_in_break_and_rescue_tasks"

    # First-divergence reward-group table is descriptive and task-counted.
    group_summary = []
    for outcome in ("break", "rescue"):
        outcome_rows = [row for row in repeat_evidence if row["outcome_class"] == outcome]
        for group in REWARD_GROUPS:
            selected_rows = [row for row in outcome_rows if row.get("dominant_positive_group") == group]
            group_summary.append(
                {
                    "outcome_class": outcome,
                    "reward_group": group,
                    "repeat_count": len(selected_rows),
                    "task_count": len({row["task_id"] for row in selected_rows}),
                    "positive_delta_task_count": len(group_tasks[group][outcome]),
                    "mean_delta_vs_u0": _mean_or_none(
                        row.get("dominant_group_delta_vs_u0") for row in selected_rows
                    ),
                }
            )

    preview = build_d05_verdict(repeat_evidence)
    summary = {
        "schema": "routea_d05b_analysis_summary_v1",
        "run_lineage_id": selection["run_lineage_id"],
        "target_repeat_count": selection["target_repeat_count"],
        "target_task_count": selection["target_task_count"],
        "window_repeat_count": len(windows),
        "window_task_count": len({row["task_id"] for row in windows}),
        "no_divergence_repeat_count": no_divergence,
        "max_reward_sum_error_abs": max_reward_sum_error,
        "rng_pairing_checked_decisions": sum(
            min(
                len(by_episode[(selected["task_id"], int(selected["repeat"]), "u0")]),
                len(by_episode[(selected["task_id"], int(selected["repeat"]), "seed3-u32")]),
                int(selected["decision_count"]),
            )
            for selected in selection["targets"]
        ),
        "context_gating_groups": context_groups,
        "task_level_preview": preview,
        "evidence_scope": "paired_first_divergence_correlation",
        "causal_claim_allowed": False,
    }
    with AtomicOutputDirectory(args.output_dir) as staging:
        write_jsonl(staging / "first_divergence_windows.jsonl", windows)
        write_csv_rows(staging / "reward_credit_transition_table.csv", transition_table)
        write_csv_rows(staging / "reward_credit_group_summary.csv", group_summary)
        shutil.copy2(
            Path(args.capture_dir) / "paired_episode_outcomes.csv",
            staging / "paired_episode_outcomes.csv",
        )
        shutil.copy2(Path(args.capture_dir) / "bc_repeat_null.json", staging / "bc_repeat_null.json")
        write_csv_rows(staging / "safe_arrival_substep_labels.csv", safe_labels)
        write_csv_rows(staging / "task_mechanisms.csv", repeat_evidence)
        write_strict_json(staging / "analysis_summary.json", summary)
        publish_stage_manifest(staging, _d05_source_manifest())
    _single_line(
        {
            "status": "ANALYZE_D05B_COMPLETE",
            "windows": len(windows),
            "tasks": summary["window_task_count"],
            "no_divergence": no_divergence,
            "preview": preview["status"],
            "output_dir": str(Path(args.output_dir).resolve()),
        }
    )
    return 0


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    import csv

    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _run_finalize(args: argparse.Namespace) -> int:
    _check_new_output(args.output_dir)
    verify_sha256_manifest(args.capture_dir)
    verify_sha256_manifest(args.analysis_dir)
    capture_source = load_strict_json(Path(args.capture_dir) / "diagnostic_source_manifest.json")
    analysis_source = load_strict_json(Path(args.analysis_dir) / "diagnostic_source_manifest.json")
    current_source = _d05_source_manifest()
    source_ids = {
        capture_source.get("content_sha256"),
        analysis_source.get("content_sha256"),
        current_source.get("content_sha256"),
    }
    if len(source_ids) != 1:
        raise _failure("STAGE_INPUT_HASH_MISMATCH", "capture/analysis/finalize source differs")
    capture_selection = load_strict_json(Path(args.capture_dir) / "selection.json")
    summary = load_strict_json(Path(args.analysis_dir) / "analysis_summary.json")
    if capture_selection["run_lineage_id"] != summary["run_lineage_id"]:
        raise _failure("STAGE_INPUT_HASH_MISMATCH", "run lineage differs")
    mechanism_rows = _read_csv_rows(Path(args.analysis_dir) / "task_mechanisms.csv")
    verdict = build_d05_verdict(mechanism_rows)
    verdict.update(
        {
            "run_lineage_id": capture_selection["run_lineage_id"],
            "break_repeat_count": capture_selection["break_repeat_count"],
            "rescue_repeat_count": capture_selection["rescue_repeat_count"],
            "window_task_count": summary["window_task_count"],
            "no_divergence_repeat_count": summary["no_divergence_repeat_count"],
            "training_authorized": False,
            "reward_change_authorized": False,
            "boundary": (
                "D0.5-B aligns reward and credit with first divergence; it does not prove "
                "a reward term caused training behavior"
            ),
        }
    )
    identity = {
        "schema": "routea_d05b_final_identity_v1",
        "run_lineage_id": capture_selection["run_lineage_id"],
        "capture_manifest_sha256": file_sha256(Path(args.capture_dir) / "manifest_sha256.txt"),
        "analysis_manifest_sha256": file_sha256(Path(args.analysis_dir) / "manifest_sha256.txt"),
        "selection_sha256": file_sha256(Path(args.capture_dir) / "selection.json"),
        "credit_batch_audit_sha256": file_sha256(
            Path(args.capture_dir) / "credit_batch_audit.json"
        ),
        "diagnostic_source_sha256": current_source["content_sha256"],
        "actor_rerun": False,
        "model_forward_for_actor": False,
        "critic_forward": True,
        "training_performed": False,
    }
    previous = {
        "schema": "routea_d05b_existing_credit_boundary_v1",
        "status": "D05A_PREVIOUSLY_COMPLETED_NOT_RERUN",
        "reason": "design v1.4 E1 forbids repeating the completed Level A 15x1024 audit",
        "this_stage_uses_probe_normalization": False,
        "this_stage_formal_normalization": "S2 full_joint_accum_batch1280",
    }
    with AtomicOutputDirectory(args.output_dir) as staging:
        for name in D05_ANALYSIS_FILES:
            shutil.copy2(Path(args.analysis_dir) / name, staging / name)
        for name in (
            "selection.json",
            "input_identity.json",
            "credit_batch_audit.json",
            "tripwire_verification.json",
        ):
            shutil.copy2(Path(args.capture_dir) / name, staging / name)
        write_strict_json(staging / "identity.json", identity)
        write_strict_json(staging / "existing_credit_identity.json", previous)
        write_strict_json(staging / "d05_verdict.json", verdict)
        publish_stage_manifest(staging, current_source)
    _single_line(
        {
            "status": "FINALIZE_D05B_COMPLETE",
            "verdict": verdict["status"],
            "determinate_tasks": verdict["determinate_task_count"],
            "training_authorized": False,
            "output_dir": str(Path(args.output_dir).resolve()),
        }
    )
    return 0


def _self_check() -> int:
    from scripts.probe_routea_goal_sensitivity import (
        _frozen_plan_action,
        build_inventory,
    )

    before_training_hash = build_inventory(REPO_ROOT)[1]
    rng = capture_rng_state()
    expected = (random.random(), np.random.random(), torch.rand(3))
    restore_rng_state(rng)
    actual = (random.random(), np.random.random(), torch.rand(3))
    rng_restored = (
        expected[0] == actual[0]
        and expected[1] == actual[1]
        and torch.equal(expected[2], actual[2])
    )
    restore_rng_state(rng)
    terms = {name: 0.0 for name in TERM_NAMES}
    terms["progress"] = 1.0
    reward_check = validate_reward_trace_row({"reward": 1.0, "reward_terms": terms})
    normalized = s2_normalize_advantages(
        [float(index) for index in range(1024)],
        ["t{}".format(index) for index in range(1024)],
        17,
    )
    entered, segment, fraction = _first_sphere_entry(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.5, 0.0, 0.0],
        ],
        [1.0, 0.0, 0.0],
        radius=0.5,
    )
    self_check_plan = np.zeros((8, 11), dtype=np.float32)
    self_check_rows = np.column_stack(
        (
            np.ones(8),
            np.arange(8),
            np.arange(1, 9) * 0.3,
            self_check_plan,
        )
    )
    frozen_action = _frozen_plan_action(
        self_check_rows,
        decision_step=1,
        before={
            "position": np.zeros(3, dtype=np.float32),
            "velocity_reference": np.zeros(3, dtype=np.float32),
            "yaw": 0.0,
        },
        exact_bitwise=True,
    )
    frozen_numpy_action_serialization = bool(
        np.asarray(_policy_action_8x11(frozen_action)).shape == (8, 11)
        and np.asarray(_executed_action_4x11(frozen_action)).shape == (4, 11)
    )
    endpoint_check = {
        "task_id": "self_check_task",
        "repeat": 0,
        "policy": "bc",
        "planned_rows": 104,
        "executed_rows": 53,
        "actual": {
            "task_id": "self_check_task",
            "policy": "bc",
            "done_reason": "reached",
            "steps": 13,
            "observation_validity_summary": {
                "last_policy_input_hash": "0" * 64,
            },
        },
    }
    normalized_endpoint = _normalize_endpoint_check(
        endpoint_check,
        task_id="self_check_task",
        repeat=1,
        policy="bc",
        done_reason="reached",
    )
    endpoint_summary_normalization = bool(
        normalized_endpoint == endpoint_check["actual"]
        and _selected_expected_summary(
            {
                "task_id": "self_check_task",
                "bc_done_reason": "reached",
                "expected_bc_summary": normalized_endpoint,
            },
            "u0",
        )
        == normalized_endpoint
    )
    after_training_hash = build_inventory(REPO_ROOT)[1]
    passed = bool(
        rng_restored
        and reward_check["reward_sum_error_abs"] == 0.0
        and normalized["audit"]["logical_batch_count"] == 8
        and normalized["audit"]["unique_pair_count"] == 10240
        and entered
        and segment == 0
        and abs(float(fraction) - 0.25) <= 1e-12
        and frozen_numpy_action_serialization
        and endpoint_summary_normalization
        and before_training_hash == after_training_hash
    )
    payload = {
        "status": "SELF_CHECK_PASS" if passed else "SELF_CHECK_FAIL",
        "rng_restored": rng_restored,
        "reward_sum_error_abs": reward_check["reward_sum_error_abs"],
        "s2_pair_count": normalized["audit"]["pair_count"],
        "s2_logical_batch_count": normalized["audit"]["logical_batch_count"],
        "goal_sphere_fraction": fraction,
        "frozen_numpy_action_serialization": frozen_numpy_action_serialization,
        "endpoint_summary_normalization": endpoint_summary_normalization,
        "training_source_set_unchanged": before_training_hash == after_training_hash,
        "training_source_set_hash": before_training_hash,
        "diagnostic_source_sha256": _d05_source_manifest()["content_sha256"],
    }
    _single_line(payload)
    return 0 if passed else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    subparsers = parser.add_subparsers(dest="command")

    select = subparsers.add_parser("select-d05b")
    select.add_argument("--d1f-final-dir", type=Path, required=True)
    select.add_argument("--d1f-reaudit-dir", type=Path, required=True)
    for seed in (1, 2, 3):
        select.add_argument("--fixed32-seed{}-dir".format(seed), type=Path, required=True)
    select.add_argument("--layera-geometry", type=Path, required=True)
    select.add_argument("--rl-test-geometry", type=Path, required=True)
    select.add_argument(
        "--episode-config-dir", type=Path, default=REPO_ROOT / "data" / "eval_configs"
    )
    select.add_argument("--output-dir", type=Path, required=True)

    capture = subparsers.add_parser("capture-d05b")
    capture.add_argument("--selection-dir", type=Path, required=True)
    capture.add_argument("--u0-config", type=Path, required=True)
    capture.add_argument("--u0-checkpoint", type=Path, required=True)
    capture.add_argument("--u32-config", type=Path, required=True)
    capture.add_argument("--u32-checkpoint", type=Path, required=True)
    capture.add_argument("--output-dir", type=Path, required=True)

    analyze = subparsers.add_parser("analyze-d05b")
    analyze.add_argument("--capture-dir", type=Path, required=True)
    analyze.add_argument("--output-dir", type=Path, required=True)

    finalize = subparsers.add_parser("finalize-d05b")
    finalize.add_argument("--capture-dir", type=Path, required=True)
    finalize.add_argument("--analysis-dir", type=Path, required=True)
    finalize.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.self_check:
            if args.command is not None:
                raise _failure("CLI_USAGE_ERROR", "--self-check cannot be combined with a command")
            return _self_check()
        handlers = {
            "select-d05b": _run_select,
            "capture-d05b": _run_capture,
            "analyze-d05b": _run_analyze,
            "finalize-d05b": _run_finalize,
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
        return int(exc.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
