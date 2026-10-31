"""Atomic stage publication, strict tables, and content-addressed manifests."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .contracts import DiagnosticFailure, load_strict_json, write_strict_json


MINIMUM_FINAL_OUTPUTS = frozenset(
    {
        "identity.json",
        "checkpoint_comparability.json",
        "stratum_manifest.json",
        "observation_bank_manifest.json",
        "observation_bank.npz",
        "bc_retention_drift.csv",
        "b2_sentinel_pairs.csv",
        "goal_sensitivity_summary.json",
        "stratum_summary.json",
        "rng_pairing_verification.json",
        "tripwire_verification.json",
        "verdict.json",
        "diagnostic_source_manifest.json",
        "manifest_sha256.txt",
        "console.log",
    }
)

D1F_MINIMUM_FINAL_OUTPUTS = frozenset(
    {
        "identity.json",
        "endpoint_identity_audit.json",
        "bc_repeat_null.json",
        "shared_goal_response_reference.json",
        "failure_state_selection.json",
        "failure_state_bank_manifest.json",
        "failure_state_bank.npz",
        "goal_variants.jsonl",
        "failure_state_sensitivity.csv",
        "failure_state_jacobians.csv",
        "failure_state_task_summary.csv",
        "failure_state_stage_coverage.csv",
        "failure_state_distance_role_coverage.csv",
        "failure_state_progression.json",
        "failure_state_summary.json",
        "rng_pairing_verification.json",
        "tripwire_verification.json",
        "verdict.json",
        "diagnostic_source_manifest.json",
        "manifest_sha256.txt",
        "console.log",
    }
)

D1F_SELECTION_BOUND_FILES = (
    "failure_state_selection.json",
    "endpoint_identity_audit.json",
    "bc_repeat_null.json",
    "shared_goal_response_reference.json",
    "trigger_checkpoint_comparability.json",
)


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def d1f_selection_identity(directory: Path | str) -> dict[str, Any]:
    """Return the complete content and run lineage bound into a D1-F bank."""

    root = Path(directory)
    selection = load_strict_json(root / "failure_state_selection.json")
    lineage = selection.get("run_lineage_id")
    if not isinstance(lineage, str) or not lineage:
        raise DiagnosticFailure(
            "STAGE_INPUT_HASH_MISMATCH",
            "D1-F selection is missing run_lineage_id",
            exit_code=2,
        )
    source = load_strict_json(root / "diagnostic_source_manifest.json")
    source_sha = source.get("content_sha256")
    if not isinstance(source_sha, str) or len(source_sha) != 64:
        raise DiagnosticFailure(
            "STAGE_INPUT_HASH_MISMATCH",
            "D1-F selection has invalid diagnostic source identity",
            exit_code=2,
        )
    return {
        "run_lineage_id": lineage,
        "stage_manifest_sha256": file_sha256(root / "manifest_sha256.txt"),
        "diagnostic_source_sha256": source_sha,
        "files": {
            filename: file_sha256(root / filename)
            for filename in D1F_SELECTION_BOUND_FILES
        },
    }


class AtomicOutputDirectory:
    """Publish a new output directory only after every validation passes."""

    def __init__(self, destination: Path | str):
        self.destination = Path(destination).resolve()
        self.staging: Path | None = None

    def __enter__(self) -> Path:
        if self.destination.exists():
            raise DiagnosticFailure(
                "OUTPUT_DIR_EXISTS", str(self.destination), exit_code=2
            )
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.staging = Path(
            tempfile.mkdtemp(
                prefix=f".{self.destination.name}.staging-",
                dir=str(self.destination.parent),
            )
        )
        return self.staging

    def __exit__(self, exc_type, exc, traceback) -> bool:
        assert self.staging is not None
        if exc_type is None:
            os.replace(self.staging, self.destination)
        elif self.staging.exists():
            shutil.rmtree(self.staging)
        return False


def _csv_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        raise DiagnosticFailure("INVALID_TABULAR_ARTIFACT", "non-finite CSV value")
    if isinstance(value, (list, tuple, np.ndarray, Mapping)):
        return json.dumps(
            np.asarray(value).tolist() if isinstance(value, np.ndarray) else value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    if isinstance(value, (str, bool, int, float)):
        return value
    raise DiagnosticFailure(
        "INVALID_TABULAR_ARTIFACT", f"unsupported CSV value {type(value).__name__}"
    )


def write_csv_rows(
    path: Path | str, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with target.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row.get(name)) for name in fieldnames})


def write_jsonl(path: Path | str, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            try:
                handle.write(
                    json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False)
                    + "\n"
                )
            except (TypeError, ValueError) as exc:
                raise DiagnosticFailure("INVALID_JSON_ARTIFACT", str(exc)) from exc


def write_sha256_manifest(
    root: Path | str, *, excluded: Sequence[str] = ("manifest_sha256.txt", "console.log")
) -> Path:
    directory = Path(root)
    excluded_set = set(excluded)
    files = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.relative_to(directory).as_posix() not in excluded_set
    )
    lines = [
        f"{file_sha256(path)}  {path.relative_to(directory).as_posix()}"
        for path in files
    ]
    target = directory / "manifest_sha256.txt"
    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return target


def verify_sha256_manifest(root: Path | str) -> dict[str, str]:
    directory = Path(root).resolve()
    manifest = directory / "manifest_sha256.txt"
    if not manifest.is_file():
        raise DiagnosticFailure(
            "STAGE_INPUT_HASH_MISMATCH", f"missing {manifest}", exit_code=2
        )
    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise DiagnosticFailure(
                "STAGE_INPUT_HASH_MISMATCH", f"invalid manifest line: {line!r}"
            )
        digest, relative = parts
        path = (directory / relative).resolve()
        if directory not in path.parents or not path.is_file():
            raise DiagnosticFailure(
                "STAGE_INPUT_HASH_MISMATCH", f"invalid/missing path: {relative}"
            )
        actual = file_sha256(path)
        if actual != digest.lower():
            raise DiagnosticFailure(
                "STAGE_INPUT_HASH_MISMATCH",
                f"{relative}: expected={digest} actual={actual}",
                exit_code=2,
            )
        expected[relative] = digest.lower()
    return expected


def publish_stage_manifest(root: Path | str, source_manifest: Mapping[str, Any]) -> None:
    directory = Path(root)
    write_strict_json(directory / "diagnostic_source_manifest.json", source_manifest)
    (directory / "console.log").touch(exist_ok=False)
    write_sha256_manifest(directory)


def _copy_verified(source_dir: Path, filename: str, destination: Path) -> None:
    source = source_dir / filename
    if not source.is_file():
        raise DiagnosticFailure(
            "MISSING_STAGE_OUTPUT", f"{source_dir.name}/{filename}", exit_code=2
        )
    shutil.copy2(source, destination / filename)


def build_stratum_summary(
    stratum_manifest: Mapping[str, Any],
    bank_manifest: Mapping[str, Any],
    sentinel_summary: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    snapshots = bank_manifest.get("snapshots", [])
    for name, details in stratum_manifest["strata"].items():
        stratum_snapshots = [row for row in snapshots if row.get("stratum") == name]
        sentinel = (sentinel_summary.get("strata") or {}).get(name, {})
        result[name] = {
            "parent_candidate_tasks": details["candidate_task_count"],
            "selected_tasks": details["selected_count"],
            "snapshot_count": len(stratum_snapshots),
            "snapshot_task_count": len({row.get("task_id") for row in stratum_snapshots}),
            "sentinel_task_count": sentinel.get("task_count", 0),
            "status": details["status"],
            "underpowered_reason": details["underpowered_reason"],
            "median": sentinel.get("median"),
            "range": sentinel.get("range"),
            "ci_available": bool(sentinel.get("ci_available", False)),
        }
    return {
        "schema": "routea_stratum_summary_v1",
        "strata": result,
        "parent_weighted_exploratory": None,
        "parent_weighted_reason": (
            "B0 is a deterministic case-control enriched bank; complete parent prevalence "
            "and validated inclusion probabilities are unavailable"
        ),
    }


def build_verdict(
    checkpoint_report: Mapping[str, Any],
    b1_summary: Mapping[str, Any],
    b2_summary: Mapping[str, Any],
) -> dict[str, Any]:
    if checkpoint_report.get("status") != "COMPARABLE":
        status = "CHECKPOINT_CONTRACT_MISMATCH"
        next_action = "repair checkpoint/runtime identity before any scientific forward"
    else:
        b2_status = b2_summary.get("status")
        if b2_status == "SENTINEL_ABNORMAL_EXPAND_FULL_B2":
            status = b2_status
            next_action = "implement and run the pre-specified full B2 only"
        elif b2_status == "MEASUREMENT_INVALID":
            status = "MEASUREMENT_INVALID"
            next_action = "repair measurement pairing; do not infer a mechanism"
        elif b1_summary.get("route") == "SYSTEMATIC_DRIFT":
            status = "OBJECTIVE_CREDIT_DRIFT_ROUTE"
            next_action = "proceed to objective/credit diagnosis under the approved design"
        else:
            status = "TRIGGER_D1F"
            next_action = "run the pre-specified D1-F diagnostic before any training change"
    return {
        "schema": "routea_b0_b1_b2_verdict_v1",
        "status": status,
        "evidence_scope": "lightweight_kinematic_network_forward",
        "proven_facts": [
            "checkpoint comparability was evaluated before scientific aggregation",
            "B2-sentinel is mandatory after every valid B1 result",
            "all reported action geometry uses executed H4 only",
        ],
        "bank_supported_judgments": [
            "judgments are stratum-conditional and do not estimate natural prevalence"
        ],
        "uncovered_variables": [
            "full B2 target surface",
            "D1-F objective/credit attribution",
            "Gazebo/PX4 closed-loop capability",
            "post-change retraining outcome",
        ],
        "next_allowed_action": next_action,
        "forbidden_actions": [
            "start retraining from this diagnostic alone",
            "claim reward repair is proven",
            "claim Gazebo/PX4 closure",
            "pool enriched strata without valid parent weights",
        ],
    }


def finalize_outputs(
    b0_dir: Path | str,
    b1_dir: Path | str,
    b2_dir: Path | str,
    output_dir: Path | str,
    *,
    source_manifest: Mapping[str, Any],
) -> Path:
    b0, b1, b2 = (Path(path).resolve() for path in (b0_dir, b1_dir, b2_dir))
    stage_hashes = {
        "b0": verify_sha256_manifest(b0),
        "b1": verify_sha256_manifest(b1),
        "b2": verify_sha256_manifest(b2),
    }
    with AtomicOutputDirectory(output_dir) as staging:
        for filename in (
            "stratum_manifest.json",
            "observation_bank_manifest.json",
            "observation_bank.npz",
        ):
            _copy_verified(b0, filename, staging)
        for filename in ("checkpoint_comparability.json", "bc_retention_drift.csv"):
            _copy_verified(b1, filename, staging)
        for filename in ("b2_sentinel_pairs.csv", "goal_sensitivity_summary.json"):
            _copy_verified(b2, filename, staging)

        checkpoint = load_strict_json(staging / "checkpoint_comparability.json")
        stratum = load_strict_json(staging / "stratum_manifest.json")
        bank = load_strict_json(staging / "observation_bank_manifest.json")
        b1_summary_path = b1 / "b1_summary.json"
        b1_summary = load_strict_json(b1_summary_path) if b1_summary_path.is_file() else {}
        b2_summary = load_strict_json(staging / "goal_sensitivity_summary.json")
        write_strict_json(
            staging / "identity.json",
            {
                "schema": "routea_b0_b1_b2_identity_v1",
                "stage_manifest_sha256": {
                    name: file_sha256(path / "manifest_sha256.txt")
                    for name, path in (("b0", b0), ("b1", b1), ("b2", b2))
                },
                "stage_files": stage_hashes,
                "diagnostic_source_sha256": source_manifest["content_sha256"],
            },
        )
        write_strict_json(
            staging / "stratum_summary.json",
            build_stratum_summary(stratum, bank, b2_summary),
        )
        write_strict_json(staging / "verdict.json", build_verdict(checkpoint, b1_summary, b2_summary))
        write_strict_json(
            staging / "rng_pairing_verification.json",
            {
                "schema": "routea_combined_rng_pairing_v1",
                "b1": load_strict_json(b1 / "rng_pairing_verification.json"),
                "b2": load_strict_json(b2 / "rng_pairing_verification.json"),
            },
        )
        write_strict_json(
            staging / "tripwire_verification.json",
            {
                "schema": "routea_combined_tripwire_v1",
                "b0": load_strict_json(b0 / "tripwire_verification.json"),
                "b1": load_strict_json(b1 / "tripwire_verification.json"),
                "b2": load_strict_json(b2 / "tripwire_verification.json"),
            },
        )
        publish_stage_manifest(staging, source_manifest)
        actual = {path.name for path in staging.iterdir()}
        missing = MINIMUM_FINAL_OUTPUTS - actual
        if missing:
            raise DiagnosticFailure("MISSING_FINAL_OUTPUT", str(sorted(missing)))
    return Path(output_dir).resolve()


def finalize_d1f_outputs(
    selection_dir: Path | str,
    bank_dir: Path | str,
    forward_dir: Path | str,
    output_dir: Path | str,
    *,
    source_manifest: Mapping[str, Any],
) -> Path:
    selection, bank, forward = (
        Path(path).resolve() for path in (selection_dir, bank_dir, forward_dir)
    )
    stage_hashes = {
        "selection": verify_sha256_manifest(selection),
        "bank": verify_sha256_manifest(bank),
        "forward": verify_sha256_manifest(forward),
    }
    current_source_sha = source_manifest["content_sha256"]
    stage_source_hashes = {
        name: load_strict_json(path / "diagnostic_source_manifest.json").get(
            "content_sha256"
        )
        for name, path in (
            ("selection", selection),
            ("bank", bank),
            ("forward", forward),
        )
    }
    if set(stage_source_hashes.values()) != {current_source_sha}:
        raise DiagnosticFailure(
            "STAGE_INPUT_HASH_MISMATCH",
            f"D1-F stage diagnostic sources differ: {stage_source_hashes}",
            exit_code=2,
        )
    bank_manifest = load_strict_json(bank / "failure_state_bank_manifest.json")
    supplied_selection_identity = d1f_selection_identity(selection)
    if bank_manifest.get("selection_identity") != supplied_selection_identity:
        raise DiagnosticFailure(
            "STAGE_INPUT_HASH_MISMATCH",
            "bank selection identity does not match supplied selection stage",
            exit_code=2,
        )
    copied_selection_identity = {
        filename: file_sha256(bank / filename)
        for filename in D1F_SELECTION_BOUND_FILES
    }
    if (
        supplied_selection_identity["files"] != copied_selection_identity
        or bank_manifest.get("run_lineage_id")
        != supplied_selection_identity["run_lineage_id"]
    ):
        raise DiagnosticFailure(
            "STAGE_INPUT_HASH_MISMATCH",
            "bank copied selection files or run lineage differ",
            exit_code=2,
        )
    selection_sha = file_sha256(selection / "failure_state_selection.json")
    if bank_manifest.get("selection_manifest_sha256") != selection_sha:
        raise DiagnosticFailure(
            "STAGE_INPUT_HASH_MISMATCH",
            "bank selection identity does not match supplied selection stage",
            exit_code=2,
        )
    forward_summary = load_strict_json(forward / "failure_state_summary.json")
    forward_identity = forward_summary.get("input_identity") or {}
    expected_forward_identity = {
        "run_lineage_id": bank_manifest["run_lineage_id"],
        "diagnostic_source_sha256": current_source_sha,
        "selection_identity": supplied_selection_identity,
        "bank_stage_manifest_sha256": file_sha256(bank / "manifest_sha256.txt"),
        "bank_manifest_sha256": file_sha256(bank / "failure_state_bank_manifest.json"),
        "bank_npz_sha256": file_sha256(bank / "failure_state_bank.npz"),
        "selection_sha256": file_sha256(bank / "failure_state_selection.json"),
        "shared_reference_sha256": file_sha256(
            bank / "shared_goal_response_reference.json"
        ),
    }
    if forward_identity != expected_forward_identity:
        raise DiagnosticFailure(
            "STAGE_INPUT_HASH_MISMATCH",
            "forward stage is not bound to the supplied bank stage",
            exit_code=2,
        )
    with AtomicOutputDirectory(output_dir) as staging:
        for filename in D1F_SELECTION_BOUND_FILES:
            _copy_verified(bank, filename, staging)
        for filename in ("failure_state_bank_manifest.json", "failure_state_bank.npz"):
            _copy_verified(bank, filename, staging)
        for filename in (
            "checkpoint_comparability.json",
            "goal_variants.jsonl",
            "failure_state_sensitivity.csv",
            "failure_state_jacobians.csv",
            "failure_state_task_summary.csv",
            "failure_state_stage_coverage.csv",
            "failure_state_distance_role_coverage.csv",
            "failure_state_progression.json",
            "failure_state_summary.json",
            "rng_pairing_verification.json",
            "verdict.json",
        ):
            _copy_verified(forward, filename, staging)
        write_strict_json(
            staging / "identity.json",
            {
                "schema": "routea_d1f_identity_v1",
                "stage_manifest_sha256": {
                    name: file_sha256(path / "manifest_sha256.txt")
                    for name, path in (
                        ("selection", selection),
                        ("bank", bank),
                        ("forward", forward),
                    )
                },
                "stage_files": stage_hashes,
                "run_lineage_id": supplied_selection_identity["run_lineage_id"],
                "diagnostic_source_sha256": current_source_sha,
            },
        )
        write_strict_json(
            staging / "tripwire_verification.json",
            {
                "schema": "routea_d1f_combined_tripwire_v1",
                "bank": load_strict_json(bank / "tripwire_verification.json"),
                "forward": load_strict_json(forward / "tripwire_verification.json"),
            },
        )
        publish_stage_manifest(staging, source_manifest)
        missing = D1F_MINIMUM_FINAL_OUTPUTS - {path.name for path in staging.iterdir()}
        if missing:
            raise DiagnosticFailure("MISSING_FINAL_OUTPUT", str(sorted(missing)))
    return Path(output_dir).resolve()
