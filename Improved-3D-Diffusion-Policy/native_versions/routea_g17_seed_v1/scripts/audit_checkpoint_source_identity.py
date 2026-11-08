#!/usr/bin/env python3
"""Read-only inventory of the source set bound to a DPPO checkpoint.

This script intentionally uses only the Python standard library and never
imports the training package.  It is therefore safe to run before loading a
checkpoint whose executable source identity may have drifted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Sequence, Tuple


ENVIRONMENT_SOURCE_PATHS: Tuple[str, ...] = (
    "diffusion_policy_3d/env/collision_geom.py",
    "diffusion_policy_3d/env/kinematic_uav_env.py",
    "diffusion_policy_3d/env/traj_utils.py",
    "diffusion_policy_3d/model/common/synth_pointcloud.py",
)
REWARD_SOURCE_PATHS: Tuple[str, ...] = (
    "diffusion_policy_3d/env/reward_v17.py",
    "scripts/prepare_reward_v17.py",
    "scripts/generate_reward_v17_fixtures.py",
)
BEHAVIOR_ARTIFACT_PATHS: Tuple[str, ...] = (
    "data/reward_v17/fixture_manifest_v2.json",
    "data/reward_v17/frozen_density_v1.json",
    "data/reward_v17/calibration_report_v2.json",
)
PRODUCTION_SOURCE_PATHS: Tuple[str, ...] = (
    "diffusion_policy_3d/rl/dppo_uav/integration.py",
    "diffusion_policy_3d/rl/routea_rl_env_adapter_v1.py",
)
CLI_SOURCE_PATHS: Tuple[str, ...] = (
    "scripts/train_dppo_uav_rebuild.py",
    "scripts/probe_dppo_credit.py",
    "scripts/run_routea_v17_gates.py",
    "scripts/calibrate_routea_v17_pointcloud.py",
    "scripts/build_routea_v17_formal_plan.py",
    "scripts/evaluate_dppo_uav_formal_endpoint.py",
    "scripts/formal_endpoint_eval_core.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def enumerate_source_set(repo_root: Path) -> List[Path]:
    """Mirror entrypoints.identity.source_set_hash path selection exactly."""

    package = repo_root / "diffusion_policy_3d" / "rl" / "dppo_uav_rebuild"
    vendor_overlay = repo_root / "third_party" / "dppo_official_cc7234a" / "uav_idp3"
    credit_support = repo_root / "scripts" / "dppo_credit_support"
    reward_support = repo_root / "diffusion_policy_3d" / "env" / "reward_v17_support"
    paths = sorted(
        list(package.rglob("*.py"))
        + list(package.rglob("*.yaml"))
        + list(vendor_overlay.rglob("*.py"))
        + list(vendor_overlay.rglob("*.patch"))
        + [
            repo_root / relative
            for relative in (
                ENVIRONMENT_SOURCE_PATHS
                + REWARD_SOURCE_PATHS
                + BEHAVIOR_ARTIFACT_PATHS
                + PRODUCTION_SOURCE_PATHS
                + CLI_SOURCE_PATHS
            )
            if (repo_root / relative).is_file()
        ]
        + list(reward_support.glob("*.py"))
        + list(credit_support.glob("*.py"))
    )
    unique = {path.resolve(): path.resolve() for path in paths if "__pycache__" not in path.parts}
    return sorted(unique.values())


def build_inventory(repo_root: Path) -> Tuple[List[Dict[str, Any]], str]:
    paths = enumerate_source_set(repo_root)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing source-set paths: " + ", ".join(missing))

    records: List[Dict[str, Any]] = []
    entries: List[Tuple[str, str]] = []
    for path in paths:
        relative = path.relative_to(repo_root).as_posix()
        digest = _sha256(path)
        records.append(
            {"path": relative, "sha256": digest, "bytes": path.stat().st_size}
        )
        entries.append((relative, digest))
    return records, _canonical_hash(entries)


def read_expected_source_set_hash(checkpoint_header: Path) -> str:
    data = json.loads(checkpoint_header.read_text(encoding="utf-8"))
    candidates: Iterable[Any] = (
        data.get("source_set_hash"),
        (data.get("runtime_identity") or {}).get("source_set_hash"),
        (data.get("identity") or {}).get("source_set_hash"),
    )
    for candidate in candidates:
        if isinstance(candidate, str) and len(candidate) == 64:
            try:
                int(candidate, 16)
            except ValueError:
                continue
            return candidate.lower()
    raise ValueError("checkpoint header has no valid source_set_hash")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--checkpoint-header", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    checkpoint_header = args.checkpoint_header.resolve()
    try:
        expected = read_expected_source_set_hash(checkpoint_header)
        records, actual = build_inventory(repo_root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "schema": "dppo_checkpoint_source_identity_inventory_v1",
                    "status": "ERROR",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            )
        )
        return 1

    matches = actual == expected
    print(
        json.dumps(
            {
                "schema": "dppo_checkpoint_source_identity_inventory_v1",
                "status": "MATCH" if matches else "MISMATCH",
                "repo_root": str(repo_root),
                "checkpoint_header": str(checkpoint_header),
                "expected_source_set_hash": expected,
                "actual_source_set_hash": actual,
                "source_file_count": len(records),
                "source_files": records,
            },
            sort_keys=True,
        )
    )
    return 0 if matches else 2


if __name__ == "__main__":
    sys.exit(main())
