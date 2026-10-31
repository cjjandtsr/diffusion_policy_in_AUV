#!/usr/bin/env python3
"""Read-only small action-chunk and closed-loop feedback audit.

The program reuses accepted D0.5-B paired H8/H4 actions and the existing B1
three-seed update bank.  It performs no capture, model load, forward pass,
training, reward change, checkpoint write, or world modification.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.routea_causal_diagnostics.action_feedback import (  # noqa: E402
    audit_action_feedback,
    decompose_action_pair,
    summarize_b1_progression,
)


EXIT_USAGE = 2


class AuditFailure(RuntimeError):
    def __init__(self, status: str, detail: str, exit_code: int = EXIT_USAGE):
        super().__init__("{}: {}".format(status, detail))
        self.status = str(status)
        self.detail = str(detail)
        self.exit_code = int(exit_code)


def _single_line(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False))


def _file_sha256(path: Any) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_manifest(root: Any) -> Dict[str, str]:
    directory = Path(root).resolve()
    manifest = directory / "manifest_sha256.txt"
    if not manifest.is_file():
        raise AuditFailure("STAGE_INPUT_HASH_MISMATCH", "missing {}".format(manifest))
    verified = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise AuditFailure("STAGE_INPUT_HASH_MISMATCH", "invalid manifest line")
        expected, relative = parts
        if relative in verified:
            raise AuditFailure(
                "STAGE_INPUT_HASH_MISMATCH",
                "duplicate manifest path {}".format(relative),
            )
        path = (directory / relative).resolve()
        if directory not in path.parents or not path.is_file():
            raise AuditFailure(
                "STAGE_INPUT_HASH_MISMATCH", "invalid or missing {}".format(relative)
            )
        actual = _file_sha256(path)
        if actual != expected.lower():
            raise AuditFailure(
                "STAGE_INPUT_HASH_MISMATCH",
                "{} expected={} actual={}".format(relative, expected, actual),
            )
        verified[relative] = actual
    if not verified:
        raise AuditFailure("STAGE_INPUT_HASH_MISMATCH", "manifest is empty")
    return verified


def _require_manifest_members(
    verified: Mapping[str, str], required: Iterable[str], label: str
) -> None:
    missing = sorted(set(required) - set(verified))
    if missing:
        raise AuditFailure(
            "STAGE_INPUT_HASH_MISMATCH",
            "{} manifest does not seal {}".format(label, missing),
        )


def _load_json(path: Any) -> Any:
    target = Path(path)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditFailure("INVALID_JSON_ARTIFACT", "{}: {}".format(target, exc)) from exc
    return value


def _read_jsonl(path: Any) -> List[Dict[str, Any]]:
    target = Path(path)
    rows = []
    try:
        with target.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("row is not an object")
                rows.append(value)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise AuditFailure(
            "INVALID_JSON_ARTIFACT", "{}: {}".format(target, exc)
        ) from exc
    return rows


def _read_csv(path: Any) -> List[Dict[str, str]]:
    target = Path(path)
    try:
        with target.open("r", encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))
    except (OSError, csv.Error) as exc:
        raise AuditFailure("INVALID_CSV_ARTIFACT", "{}: {}".format(target, exc)) from exc


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AuditFailure("INVALID_OUTPUT_ARTIFACT", "non-finite float")
        return value
    raise AuditFailure(
        "INVALID_OUTPUT_ARTIFACT", "unsupported type {}".format(type(value).__name__)
    )


def _write_json(path: Any, value: Any) -> None:
    Path(path).write_text(
        json.dumps(
            _json_ready(value),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (dict, list, tuple, np.ndarray)):
        return json.dumps(
            _json_ready(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    if isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise AuditFailure("INVALID_OUTPUT_ARTIFACT", "non-finite CSV value")
        return value
    raise AuditFailure("INVALID_OUTPUT_ARTIFACT", "unsupported CSV value")


def _write_csv(path: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    target = Path(path)
    fieldnames = sorted({name for row in rows for name in row})
    with target.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row.get(name)) for name in fieldnames})


def _write_manifest(root: Any) -> None:
    directory = Path(root)
    files = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.name != "manifest_sha256.txt"
    )
    lines = [
        "{}  {}".format(_file_sha256(path), path.relative_to(directory).as_posix())
        for path in files
    ]
    (directory / "manifest_sha256.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _source_identity() -> Dict[str, Any]:
    paths = (
        Path(__file__).resolve(),
        REPO_ROOT / "scripts" / "routea_causal_diagnostics" / "action_feedback.py",
    )
    files = {
        path.relative_to(REPO_ROOT).as_posix(): _file_sha256(path) for path in paths
    }
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema": "routea_action_chunk_feedback_source_v1",
        "files": files,
        "content_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _atomic_output(destination: Path):
    class _Context:
        def __init__(self, target: Path):
            self.target = target.resolve()
            self.staging = None

        def __enter__(self):
            if self.target.exists():
                raise AuditFailure("OUTPUT_DIR_EXISTS", str(self.target))
            self.target.parent.mkdir(parents=True, exist_ok=True)
            self.staging = Path(
                tempfile.mkdtemp(
                    prefix=".{}.staging-".format(self.target.name),
                    dir=str(self.target.parent),
                )
            )
            return self.staging

        def __exit__(self, exc_type, exc, traceback):
            if exc_type is None:
                os.replace(str(self.staging), str(self.target))
            elif self.staging is not None and self.staging.exists():
                shutil.rmtree(str(self.staging))
            return False

    return _Context(destination)


def _validate_parent_identity(capture: Path, final: Path) -> str:
    selection = _load_json(capture / "selection.json")
    identity = _load_json(final / "identity.json")
    lineage = selection.get("run_lineage_id")
    if not isinstance(lineage, str) or not lineage:
        raise AuditFailure("PARENT_IDENTITY_MISMATCH", "capture lineage is missing")
    if identity.get("run_lineage_id") != lineage:
        raise AuditFailure("PARENT_IDENTITY_MISMATCH", "D0.5 capture/final lineage differs")
    if identity.get("capture_manifest_sha256") != _file_sha256(
        capture / "manifest_sha256.txt"
    ):
        raise AuditFailure(
            "PARENT_IDENTITY_MISMATCH", "D0.5 final does not bind capture manifest"
        )
    if identity.get("selection_sha256") != _file_sha256(capture / "selection.json"):
        raise AuditFailure(
            "PARENT_IDENTITY_MISMATCH", "D0.5 final does not bind selection"
        )
    if not (final / "selection.json").is_file():
        raise AuditFailure("PARENT_IDENTITY_MISMATCH", "final selection.json is missing")
    if _file_sha256(capture / "selection.json") != _file_sha256(final / "selection.json"):
        raise AuditFailure("PARENT_IDENTITY_MISMATCH", "D0.5 selection content differs")
    return lineage


def _run_audit(args: argparse.Namespace) -> int:
    capture = Path(args.d05_capture_dir).resolve()
    final = Path(args.d05_final_dir).resolve()
    b1 = Path(args.b1_dir).resolve()
    output = Path(args.output_dir).resolve()
    capture_verified = _verify_manifest(capture)
    final_verified = _verify_manifest(final)
    b1_verified = _verify_manifest(b1)
    _require_manifest_members(
        capture_verified,
        ("selection.json", "paired_trace_transitions.jsonl", "bc_repeat_null.json"),
        "D0.5 capture",
    )
    _require_manifest_members(
        final_verified,
        ("identity.json", "selection.json", "d05_verdict.json"),
        "D0.5 final",
    )
    _require_manifest_members(
        b1_verified, ("bc_retention_drift.csv",), "B1"
    )
    lineage = _validate_parent_identity(capture, final)
    selection = _load_json(capture / "selection.json")
    targets = selection.get("targets")
    if not isinstance(targets, list) or not targets:
        raise AuditFailure("INVALID_D05_INPUT", "selection targets are missing")
    transitions = _read_jsonl(capture / "paired_trace_transitions.jsonl")
    nulls = _load_json(capture / "bc_repeat_null.json")
    b1_rows = _read_csv(b1 / "bc_retention_drift.csv")
    try:
        audit = audit_action_feedback(targets, transitions, nulls)
        b1_summary = summarize_b1_progression(b1_rows)
    except (KeyError, TypeError, ValueError) as exc:
        raise AuditFailure("INVALID_AUDIT_INPUT", str(exc)) from exc
    summary = dict(audit["summary"])
    summary.update(
        {
            "d05_run_lineage_id": lineage,
            "b1_seed_update_lane_count": 12,
            "b1_progression_row_count": len(b1_summary),
            "evidence_scope": (
                "existing_D05B_seed3_paired_actions_plus_existing_B1_three_seed_"
                "enriched_bank"
            ),
            "boundary": (
                "This audit localizes action-chunk drift and feedback growth. It does "
                "not identify a reward cause or authorize training."
            ),
        }
    )
    input_identity = {
        "schema": "routea_action_chunk_feedback_input_identity_v1",
        "d05_run_lineage_id": lineage,
        "d05_capture_manifest_sha256": _file_sha256(capture / "manifest_sha256.txt"),
        "d05_final_manifest_sha256": _file_sha256(final / "manifest_sha256.txt"),
        "b1_manifest_sha256": _file_sha256(b1 / "manifest_sha256.txt"),
        "d05_selection_sha256": _file_sha256(capture / "selection.json"),
        "d05_transition_sha256": _file_sha256(
            capture / "paired_trace_transitions.jsonl"
        ),
        "d05_null_sha256": _file_sha256(capture / "bc_repeat_null.json"),
        "b1_drift_sha256": _file_sha256(b1 / "bc_retention_drift.csv"),
        "actor_rerun": False,
        "capture_performed": False,
        "model_forward": False,
        "training_performed": False,
    }
    tripwire = {
        "schema": "routea_action_chunk_feedback_tripwire_v1",
        "status": "READ_ONLY_VERIFIED",
        "model_loaded": False,
        "model_forward": False,
        "backward_called": False,
        "optimizer_step_called": False,
        "training_performed": False,
        "reward_modified": False,
        "checkpoint_modified": False,
        "world_modified": False,
    }
    with _atomic_output(output) as staging:
        _write_csv(staging / "action_chunk_decision.csv", audit["decision_rows"])
        _write_csv(staging / "action_chunk_waypoint.csv", audit["waypoint_rows"])
        _write_csv(staging / "action_chunk_repeat.csv", audit["repeat_rows"])
        _write_csv(staging / "action_chunk_task.csv", audit["task_rows"])
        _write_csv(staging / "feedback_progression.csv", audit["progression_rows"])
        _write_csv(staging / "b1_cross_seed_progression.csv", b1_summary)
        _write_json(staging / "audit_summary.json", summary)
        _write_json(staging / "input_identity.json", input_identity)
        _write_json(staging / "tripwire_verification.json", tripwire)
        _write_json(staging / "diagnostic_source_manifest.json", _source_identity())
        _write_manifest(staging)
    _single_line(
        {
            "status": "ACTION_CHUNK_FEEDBACK_AUDIT_COMPLETE",
            "route_candidate": summary["route_candidate"],
            "repeats": summary["repeat_count"],
            "tasks": summary["task_count"],
            "training_authorized": False,
            "output_dir": str(output),
        }
    )
    return 0


def _validate_output(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    verified = _verify_manifest(output)
    required = {
        "action_chunk_decision.csv",
        "action_chunk_waypoint.csv",
        "action_chunk_repeat.csv",
        "action_chunk_task.csv",
        "feedback_progression.csv",
        "b1_cross_seed_progression.csv",
        "audit_summary.json",
        "input_identity.json",
        "tripwire_verification.json",
        "diagnostic_source_manifest.json",
    }
    if set(verified) != required:
        raise AuditFailure(
            "OUTPUT_CONTRACT_MISMATCH",
            "manifest files differ missing={} extra={}".format(
                sorted(required - set(verified)), sorted(set(verified) - required)
            ),
        )
    summary = _load_json(output / "audit_summary.json")
    identity = _load_json(output / "input_identity.json")
    tripwire = _load_json(output / "tripwire_verification.json")
    source = _load_json(output / "diagnostic_source_manifest.json")
    if source != _source_identity():
        raise AuditFailure("OUTPUT_CONTRACT_MISMATCH", "diagnostic source identity differs")
    if summary.get("status") != "ACTION_CHUNK_FEEDBACK_AUDIT_COMPLETE":
        raise AuditFailure("OUTPUT_CONTRACT_MISMATCH", "audit status is not complete")
    allowed_routes = {
        "EXECUTED_H4_DRIFT_AND_FEEDBACK_AMPLIFICATION_SUPPORTED",
        "EXECUTED_H4_DRIFT_SUPPORTED_FEEDBACK_NOT_REPEATED",
        "TAIL_DOMINANT_DRIFT_D2X_CANDIDATE",
        "ACTION_CHUNK_ROUTE_NOT_DETERMINATE",
    }
    if summary.get("route_candidate") not in allowed_routes:
        raise AuditFailure("OUTPUT_CONTRACT_MISMATCH", "unknown route candidate")
    if bool(summary.get("training_authorized")) or bool(
        summary.get("reward_change_authorized")
    ):
        raise AuditFailure("OUTPUT_CONTRACT_MISMATCH", "audit authorized a forbidden change")
    if tripwire.get("status") != "READ_ONLY_VERIFIED" or any(
        bool(tripwire.get(name))
        for name in (
            "model_loaded",
            "model_forward",
            "backward_called",
            "optimizer_step_called",
            "training_performed",
            "reward_modified",
            "checkpoint_modified",
            "world_modified",
        )
    ):
        raise AuditFailure("OUTPUT_CONTRACT_MISMATCH", "read-only tripwire failed")
    if identity.get("d05_run_lineage_id") != summary.get("d05_run_lineage_id"):
        raise AuditFailure("OUTPUT_CONTRACT_MISMATCH", "output lineage differs")
    decision_count = len(_read_csv(output / "action_chunk_decision.csv"))
    waypoint_count = len(_read_csv(output / "action_chunk_waypoint.csv"))
    repeat_count = len(_read_csv(output / "action_chunk_repeat.csv"))
    task_count = len(_read_csv(output / "action_chunk_task.csv"))
    progression_count = len(_read_csv(output / "feedback_progression.csv"))
    b1_count = len(_read_csv(output / "b1_cross_seed_progression.csv"))
    if decision_count < 1 or waypoint_count != decision_count * 8:
        raise AuditFailure("OUTPUT_CONTRACT_MISMATCH", "decision/waypoint row count differs")
    if repeat_count != int(summary.get("repeat_count", -1)):
        raise AuditFailure("OUTPUT_CONTRACT_MISMATCH", "repeat row count differs")
    if task_count != int(summary.get("task_count", -1)):
        raise AuditFailure("OUTPUT_CONTRACT_MISMATCH", "task row count differs")
    if progression_count < 1 or b1_count != int(summary.get("b1_progression_row_count", -1)):
        raise AuditFailure("OUTPUT_CONTRACT_MISMATCH", "progression row count differs")
    _single_line(
        {
            "status": "ACTION_CHUNK_FEEDBACK_OUTPUT_VALID",
            "route_candidate": summary["route_candidate"],
            "decisions": decision_count,
            "waypoints": waypoint_count,
            "repeats": repeat_count,
            "tasks": task_count,
            "b1_rows": b1_count,
            "training_authorized": False,
        }
    )
    return 0


def _self_check() -> int:
    base = np.zeros((8, 11), dtype=np.float32)
    changed = base.copy()
    changed[:4, 0] = 0.25
    changed[4:, 0] = 0.5
    decomposition = decompose_action_pair(
        base,
        changed,
        position_before=[0.0, 0.0, 0.0],
        goal=[10.0, 0.0, 0.0],
    )
    b1_rows = []
    for seed in (1, 2, 3):
        for update in (8, 16, 24, 32):
            b1_rows.append(
                {
                    "checkpoint": "seed{}-u{}".format(seed, update),
                    "update": update,
                    "stratum": "SELF_CHECK",
                    "task_id": "task-{}".format(seed),
                    "snapshot_id": "self-check-snapshot",
                    "action_h4_l2": 0.1,
                    "goal_parallel_m": 0.1,
                    "goal_perpendicular_m": 0.0,
                    "vertical_m": 0.0,
                    "yaw_total_variation_rad": 0.0,
                    "path_shape_norm_m": 0.0,
                }
            )
    progression = summarize_b1_progression(b1_rows)
    passed = bool(
        abs(decomposition["executed_endpoint_delta_m"] - 1.0) <= 1e-12
        and abs(decomposition["tail_dpos_l2"] - 1.0) <= 1e-12
        and len([row for row in progression if row["stratum"] == "ALL"]) == 12
    )
    payload = {
        "status": "SELF_CHECK_PASS" if passed else "SELF_CHECK_FAIL",
        "executed_tail_separation": passed,
        "b1_three_seed_update_lanes": 12,
        "read_only": True,
        "training_authorized": False,
        "diagnostic_source_sha256": _source_identity()["content_sha256"],
    }
    _single_line(payload)
    return 0 if passed else EXIT_USAGE


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    subparsers = parser.add_subparsers(dest="command")
    run = subparsers.add_parser("run-audit")
    run.add_argument("--d05-capture-dir", type=Path, required=True)
    run.add_argument("--d05-final-dir", type=Path, required=True)
    run.add_argument("--b1-dir", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    validate = subparsers.add_parser("validate-output")
    validate.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.self_check:
            return _self_check()
        if args.command == "run-audit":
            return _run_audit(args)
        if args.command == "validate-output":
            return _validate_output(args)
        raise AuditFailure("USAGE_ERROR", "choose --self-check or run-audit")
    except AuditFailure as exc:
        _single_line(
            {"status": exc.status, "detail": exc.detail, "exit_code": exc.exit_code}
        )
        return exc.exit_code


if __name__ == "__main__":
    sys.exit(main())
