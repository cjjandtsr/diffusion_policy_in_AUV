#!/usr/bin/env python3
"""Fail-closed Reward v1.7 admission geometry/reset audit.

Exit codes: 0 complete/pass, 2 complete/blocked, 1 audit error.  This tool
never changes the production environment, reward, policy, or admission files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from diffusion_policy_3d.rl.dppo_uav_rebuild.audit.admission_geometry import (
    audit_astar_pair,
    build_shape_reference_grid,
    compare_occupancy_grids,
    grid_spec_from_environment,
)
from diffusion_policy_3d.rl.dppo_uav_rebuild.audit.reset_bank import (
    GeometryAuditPointCloud,
    ResetAuditTask,
    audit_reset_bank,
    canonical_json_bytes,
    normalize_tasks_and_seeds,
)


MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_WORLDS = 1_000
PACK_DATA_FILES = (
    "provenance.json",
    "world_geometry_records.jsonl",
    "reset_records.jsonl",
    "summary.json",
)


def _reject_duplicate_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: {!r}".format(key))
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("non-finite JSON constant: {}".format(value))


def load_json_strict(path: Path, *, max_bytes: int = MAX_JSON_BYTES) -> Any:
    source = Path(path)
    size = source.stat().st_size
    if size > max_bytes:
        raise ValueError("JSON exceeds {} bytes: {}".format(max_bytes, source))
    with source.open("r", encoding="utf-8") as stream:
        return json.load(
            stream,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_file_hashes_unchanged(snapshot: Mapping[str, str]) -> None:
    for raw_path, expected_hash in sorted(snapshot.items()):
        path = Path(raw_path)
        try:
            actual_hash = sha256_file(path)
        except OSError as exc:
            raise RuntimeError(
                "input hash changed during audit: {} is unavailable".format(path)
            ) from exc
        if actual_hash != expected_hash:
            raise RuntimeError("input hash changed during audit: {}".format(path))


def build_runtime_provenance(
    *,
    config: Path,
    manifest: Path,
    worlds: Path,
    seed_start: int,
    seeds_per_world: int,
    source_files: Mapping[str, Path],
    created_utc: Optional[str] = None,
) -> Dict[str, Any]:
    if len(source_files) < 6:
        raise ValueError("source provenance must cover at least six behavior files")
    normalized_sources = {}
    for label, source in sorted(source_files.items()):
        source_path = Path(source).resolve()
        if not source_path.is_file():
            raise FileNotFoundError("provenance source missing: {}".format(source_path))
        normalized_sources[str(label).replace("\\", "/")] = sha256_file(source_path)
    return {
        "schema": "reward_v17_admission_audit_provenance_v1",
        "created_utc": created_utc or datetime.now(timezone.utc).isoformat(),
        "mode": "audit_only_no_behavior_change",
        "config": {
            "path": str(Path(config).resolve()),
            "sha256": sha256_file(Path(config)),
        },
        "admission_manifest": {
            "path": str(Path(manifest).resolve()),
            "sha256": sha256_file(Path(manifest)),
        },
        "worlds_train_dir": str(Path(worlds).resolve()),
        "source_files": normalized_sources,
        "seed_start": int(seed_start),
        "seeds_per_world": int(seeds_per_world),
        "production_grid_contract": {
            "resolution": 0.5,
            "xy_margin": 0.15,
            "vertical_inflation": 0.0,
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "environment_contract": "KinematicUAVEnv_C2_H4_reset_only",
        "reward_runtime_note": (
            "v13 reset shell; reward is never computed and sampling/A* are shared with v17"
        ),
    }


def validate_obstacle_path(candidate: Path, worlds_train_dir: Path) -> Path:
    root = Path(worlds_train_dir).resolve()
    resolved = Path(candidate).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("obstacle path escapes worlds-train-dir: {}".format(candidate)) from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("obstacle path uses a symlink/reparse path: {}".format(current))
    if not resolved.is_file():
        raise FileNotFoundError("admitted obstacle file missing: {}".format(resolved))
    if not resolved.name.endswith(".world.obstacles.json"):
        raise ValueError("unexpected obstacle filename: {}".format(resolved.name))
    return resolved


def load_production_tasks(
    manifest_path: Path,
    worlds_train_dir: Path,
) -> Sequence[ResetAuditTask]:
    """Load exactly the trainer's admitted entries; never emulate its parser."""
    try:
        from diffusion_policy_3d.rl.dppo_uav.integration import manifest_train_entries
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("production admission loader unavailable") from exc
    manifest = Path(manifest_path).resolve()
    worlds_root = Path(worlds_train_dir).resolve()
    if not manifest.is_file():
        raise FileNotFoundError("admission manifest missing: {}".format(manifest))
    entries = manifest_train_entries(str(manifest))
    if len(entries) > MAX_WORLDS:
        raise ValueError("production loader returned more than 1,000 worlds")
    tasks = []
    for entry in entries:
        uid = str(entry.get("world_uid", ""))
        role = str(entry.get("role") or entry.get("role_active") or "")
        split = str(entry.get("split", "rl_train"))
        admission_id = str(
            entry.get("admission_id") or "{}__{}__{}".format(uid, role, split)
        )
        obstacle = validate_obstacle_path(
            worlds_root / (uid + ".world.obstacles.json"), worlds_root
        )
        tasks.append(ResetAuditTask(
            task_id=admission_id,
            world_uid=uid,
            role=role,
            admission_id=admission_id,
            obstacle_json=str(obstacle),
            split=split,
            training_eligible=bool(entry.get("training_eligible", True)),
            historical_eval_only=bool(entry.get("historical_eval_only", False)),
        ))
    ordered, _ = normalize_tasks_and_seeds(tasks, (0,))
    return ordered


def _write_and_verify(path: Path, payload: bytes) -> str:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    expected = hashlib.sha256(payload).hexdigest()
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError("written file hash mismatch: {}".format(path.name))
    return actual


def publish_pack(output_dir: Path, files: Mapping[str, bytes]) -> Dict[str, str]:
    """Publish the five-file pack without overwriting an existing result."""
    output = Path(output_dir)
    if set(files) != set(PACK_DATA_FILES):
        raise ValueError("pack must contain exactly the four canonical data files")
    if output.exists():
        raise FileExistsError("output directory already exists: {}".format(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.parent / ("." + output.name + ".tmp." + uuid.uuid4().hex)
    temp.mkdir()
    try:
        hashes = {}
        for name in PACK_DATA_FILES:
            payload = files[name]
            if not isinstance(payload, bytes):
                raise TypeError("pack payloads must be bytes")
            hashes[name] = _write_and_verify(temp / name, payload)
        _write_and_verify(temp / "SHA256.json", canonical_json_bytes(hashes))
        try:
            os.rename(str(temp), str(output))
        except OSError as exc:
            if output.exists():
                raise FileExistsError(
                    "output directory won publication race: {}".format(output)
                ) from exc
            raise
        for name, expected in hashes.items():
            if sha256_file(output / name) != expected:
                raise RuntimeError("published file hash mismatch: {}".format(name))
        return hashes
    finally:
        if temp.exists():
            shutil.rmtree(temp)


def determine_exit_code(world_records: Sequence[Mapping[str, Any]], invalid_fallback_count: int) -> int:
    if any(record.get("status") != "pass" for record in world_records):
        return 2
    if int(invalid_fallback_count) > 0:
        return 2
    return 0


def _jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(record) for record in records)


def _wilson_95(count: int, denominator: int) -> Dict[str, float]:
    if denominator <= 0:
        raise ValueError("Wilson interval denominator must be positive")
    z = 1.959963984540054
    proportion = float(count) / float(denominator)
    scale = 1.0 + z * z / denominator
    centre = (proportion + z * z / (2.0 * denominator)) / scale
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / denominator
        + z * z / (4.0 * denominator * denominator)
    ) / scale
    return {
        "lower": max(0.0, centre - radius),
        "upper": min(1.0, centre + radius),
    }


def _rate_record(count: int, denominator: int) -> Dict[str, Any]:
    return {
        "count": int(count),
        "denominator": int(denominator),
        "rate": float(count) / float(denominator),
        "wilson_95": _wilson_95(int(count), int(denominator)),
    }


def _obstacle_counts(obstacles: Sequence[Any]) -> Dict[str, int]:
    counts = Counter(str(obstacle.shape) for obstacle in obstacles)
    result = {name: int(count) for name, count in sorted(counts.items())}
    result["total"] = int(sum(counts.values()))
    return result


def _with_path_stretch(
    astar: Mapping[str, Any],
    start: Sequence[float],
    goal: Sequence[float],
    role: str,
) -> Dict[str, Any]:
    result = dict(astar)
    direct = float(np.linalg.norm(
        np.asarray(goal, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    ))
    for branch_name in ("production", "reference"):
        branch = dict(result[branch_name])
        branch["path_stretch"] = (
            float(branch["path_length_m"]) / direct
            if branch["reachable"] and direct > 0.0 else None
        )
        result[branch_name] = branch
    result["path_length_changed"] = (
        result["production"]["path_length_m"]
        != result["reference"]["path_length_m"]
    )
    result["path_stretch_changed"] = (
        result["production"]["path_stretch"]
        != result["reference"]["path_stretch"]
    )
    applicable = role == "dense_recovery"
    production_pass = (
        applicable
        and result["production"]["path_stretch"] is not None
        and 1.1 <= result["production"]["path_stretch"] <= 1.8
    )
    reference_pass = (
        applicable
        and result["reference"]["path_stretch"] is not None
        and 1.1 <= result["reference"]["path_stretch"] <= 1.8
    )
    result["path_stretch_classification"] = {
        "applicable": applicable,
        "production_pass": production_pass if applicable else None,
        "reference_pass": reference_pass if applicable else None,
    }
    result["path_stretch_classification_changed"] = (
        applicable and production_pass != reference_pass
    )
    return result


def _summarize_fallbacks(
    reset_records: Sequence[Mapping[str, Any]], roles: Sequence[str]
) -> Dict[str, Any]:
    overall_count = sum(record["fallback_type"] is not None for record in reset_records)
    by_role = {}
    for role in sorted(set(roles)):
        role_records = [record for record in reset_records if record["role"] == role]
        role_count = sum(record["fallback_type"] is not None for record in role_records)
        by_role[role] = _rate_record(role_count, len(role_records))
    return {
        "overall": _rate_record(overall_count, len(reset_records)),
        "by_role": by_role,
    }


def _summarize_rejections(
    reset_records: Sequence[Mapping[str, Any]],
) -> Dict[str, int]:
    totals = Counter()
    for record in reset_records:
        totals.update(record["rejection_histogram"])
    return {name: int(count) for name, count in sorted(totals.items())}


def _identity_sha256(identities: Sequence[Sequence[str]]) -> str:
    return hashlib.sha256(canonical_json_bytes({"identities": identities})).hexdigest()


def build_audit_files(
    tasks: Sequence[ResetAuditTask],
    seeds: Sequence[int],
    make_environment,
    *,
    provenance: Mapping[str, Any],
):
    """Build canonical pack bytes from already validated production tasks."""
    ordered_tasks, ordered_seeds = normalize_tasks_and_seeds(tasks, seeds)
    obstacle_hashes = {
        str(Path(task.obstacle_json).resolve()): sha256_file(Path(task.obstacle_json))
        for task in ordered_tasks
    }
    reset_result = audit_reset_bank(ordered_tasks, ordered_seeds, make_environment)
    reset_records = reset_result["records"]
    reset_by_task = defaultdict(list)
    for reset_record in reset_records:
        reset_by_task[reset_record["task_id"]].append(reset_record)

    world_records = []
    geometry_seed = ordered_seeds[0]
    for task in ordered_tasks:
        environment = make_environment(task, geometry_seed)
        environment.reset(seed=geometry_seed)
        production = np.asarray(environment._get_occupancy_grid(), dtype=bool).copy()
        spec = grid_spec_from_environment(environment)
        reference = build_shape_reference_grid(environment.obstacles, spec)
        comparison = compare_occupancy_grids(
            production, reference, obstacles=environment.obstacles, spec=spec,
        )
        astar_shadow = []
        valid_resets = [
            record for record in reset_by_task[task.task_id]
            if record["status"] == "pass"
        ][:8]
        for reset_record in valid_resets:
            shadow_environment = make_environment(task, reset_record["seed"])
            if shadow_environment is None:
                raise ValueError("environment factory returned None")
            shadow_environment.reset(seed=reset_record["seed"])
            start = np.asarray(shadow_environment._drone_pos, dtype=np.float64)
            goal = np.asarray(shadow_environment.goal_w, dtype=np.float64)
            if not np.array_equal(start, np.asarray(reset_record["start_world"])):
                raise RuntimeError("non-deterministic reset start for A* shadow")
            if not np.array_equal(goal, np.asarray(reset_record["goal_world"])):
                raise RuntimeError("non-deterministic reset goal for A* shadow")
            astar = _with_path_stretch(
                audit_astar_pair(
                    shadow_environment, production, reference, start, goal,
                ),
                start,
                goal,
                task.role,
            )
            astar.update({
                "seed": int(reset_record["seed"]),
                "start_world": list(reset_record["start_world"]),
                "goal_world": list(reset_record["goal_world"]),
            })
            astar_shadow.append(astar)
        record = dict(comparison)
        record.update({
            "schema": "reward_v17_world_geometry_audit_v1",
            "task_id": task.task_id,
            "world_uid": task.world_uid,
            "role": task.role,
            "admission_id": task.admission_id,
            "obstacle_json_sha256": obstacle_hashes[
                str(Path(task.obstacle_json).resolve())
            ],
            "obstacle_counts": _obstacle_counts(environment.obstacles),
            "grid": {
                "shape": list(spec.shape),
                "resolution": spec.resolution,
                "xy_margin": spec.xy_margin,
            },
            "geometry_seed": geometry_seed,
            "astar_shadow": astar_shadow,
            "contract_parameter_drift": {
                "production_resolution": spec.resolution,
                "design_resolution_xy": 0.2,
                "production_xy_margin": spec.xy_margin,
                "design_inflation_radius": 0.5,
                "drone_radius_z_applied": False,
            },
        })
        canonical_json_bytes(record)
        world_records.append(record)

    fallback = _summarize_fallbacks(
        reset_records, [task.role for task in ordered_tasks],
    )
    exit_code = determine_exit_code(
        world_records, reset_result["summary"]["invalid_fallback_count"]
    )
    mismatch_worlds = [
        record["admission_id"] for record in world_records
        if record["status"] == "mismatch"
    ]
    shadow_records = [
        shadow for record in world_records for shadow in record["astar_shadow"]
    ]
    shadow_branches = [
        shadow[branch_name]
        for shadow in shadow_records
        for branch_name in ("production", "reference")
    ]
    shape_counts = Counter()
    for record in world_records:
        shape_counts.update(record["obstacle_counts"])
    shape_counts["total"] = sum(
        record["obstacle_counts"]["total"] for record in world_records
    )
    final_reasons = [
        "geometry_mismatch:{}:missed={}:extra={}".format(
            record["admission_id"],
            record["missed_blocked_cells"],
            record["extra_blocked_cells"],
        )
        for record in world_records if record["status"] == "mismatch"
    ]
    invalid_count = int(reset_result["summary"]["invalid_fallback_count"])
    if invalid_count:
        final_reasons.append("invalid_fallback_count:{}".format(invalid_count))
    if fallback["overall"]["rate"] >= 0.05:
        final_reasons.append(
            "fallback_overall_rate:{:.12g}".format(fallback["overall"]["rate"])
        )
        exit_code = 2
    for role, role_rate in fallback["by_role"].items():
        if role_rate["rate"] >= 0.10:
            final_reasons.append(
                "fallback_role_rate:{}:{:.12g}".format(role, role_rate["rate"])
            )
            exit_code = 2
    summary = {
        "schema": "reward_v17_admission_audit_summary_v1",
        "decision": "pass" if exit_code == 0 else "blocked",
        "final_decision": "pass" if exit_code == 0 else "blocked",
        "final_reasons": final_reasons,
        "expected_exit_code": exit_code,
        "world_record_count": len(world_records),
        "task_count": len(ordered_tasks),
        "world_mismatch_count": sum(
            record["status"] == "mismatch" for record in world_records
        ),
        "mismatch_worlds": mismatch_worlds,
        "shape_counts": dict(sorted(shape_counts.items())),
        "missed_blocked_cells_total": sum(
            record["missed_blocked_cells"] for record in world_records
        ),
        "extra_blocked_cells_total": sum(
            record["extra_blocked_cells"] for record in world_records
        ),
        "astar_shadow_count": len(shadow_records),
        "astar_reported_path_z_bounds_mismatch_count": sum(
            branch["reported_path_z_bounds_match_derived"] is False
            for branch in shadow_branches
        ),
        "astar_bottleneck_unverified_reachable_branch_count": sum(
            branch["reachable"]
            and branch["bottleneck_clearance_semantics"]
            == "production_reported_unverified"
            for branch in shadow_branches
        ),
        "astar_reachability_classification_change_count": sum(
            shadow["classification_changed"] for shadow in shadow_records
        ),
        "astar_path_length_change_count": sum(
            shadow["path_length_changed"] for shadow in shadow_records
        ),
        "astar_path_stretch_change_count": sum(
            shadow["path_stretch_changed"] for shadow in shadow_records
        ),
        "astar_path_stretch_classification_change_count": sum(
            shadow["path_stretch_classification_changed"]
            for shadow in shadow_records
        ),
        "astar_reported_path_z_bounds_mismatch_count": sum(
            shadow[branch]["reported_path_z_bounds_match_derived"] is False
            for shadow in shadow_records
            for branch in ("production", "reference")
        ),
        "astar_bottleneck_unverified_reachable_branch_count": sum(
            shadow[branch]["reachable"]
            and shadow[branch]["bottleneck_clearance_semantics"]
            == "production_reported_unverified"
            for shadow in shadow_records
            for branch in ("production", "reference")
        ),
        "fallback": fallback,
        "rejection_histogram": _summarize_rejections(reset_records),
        "audit_errors": [],
        **reset_result["summary"],
    }
    provenance_record = dict(provenance)
    provenance_record.update({
        "ordered_task_identity_sha256": _identity_sha256(
            reset_result["ordered_task_identities"]
        ),
        "pointcloud_semantics": GeometryAuditPointCloud.semantics,
    })
    assert_file_hashes_unchanged(obstacle_hashes)
    files = {
        "provenance.json": canonical_json_bytes(provenance_record),
        "world_geometry_records.jsonl": _jsonl_bytes(world_records),
        "reset_records.jsonl": _jsonl_bytes(reset_result["records"]),
        "summary.json": canonical_json_bytes(summary),
    }
    return files, exit_code


def _make_kinematic_environment(task: ResetAuditTask, seed: int):
    try:
        from diffusion_policy_3d.env.kinematic_uav_env import KinematicUAVEnv
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("KinematicUAVEnv runtime unavailable") from exc
    return KinematicUAVEnv(
        world_pool=[(task.world_uid, task.obstacle_json)],
        synth_pc_factory=lambda _: GeometryAuditPointCloud(),
        rl_action_steps=8,
        n_exec_steps=4,
        max_episode_steps=60,
        disturb_enable=False,
        curriculum_stage="C2",
        episode_role=task.role,
        reward_version="v13",
        seed=seed,
    )


def run_audit(
    *,
    config_path: Path,
    manifest_path: Path,
    worlds_train_dir: Path,
    seed_start: int,
    seeds_per_world: int,
    output_dir: Path,
) -> int:
    config = Path(config_path).resolve()
    manifest = Path(manifest_path).resolve()
    worlds = Path(worlds_train_dir).resolve()
    output = Path(output_dir).resolve()
    if not config.is_file():
        raise FileNotFoundError("formal config missing: {}".format(config))
    if config.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("formal config exceeds 64 MiB")
    if not worlds.is_dir():
        raise FileNotFoundError("worlds-train-dir missing: {}".format(worlds))
    load_json_strict(manifest)
    tasks = load_production_tasks(manifest, worlds)
    for task in tasks:
        data = load_json_strict(Path(task.obstacle_json))
        obstacles = data.get("obstacles", []) if isinstance(data, dict) else None
        if not isinstance(obstacles, list):
            raise ValueError("obstacles must be a list: {}".format(task.obstacle_json))
        if len(obstacles) > 10_000:
            raise ValueError("world exceeds 10,000 obstacles: {}".format(task.world_uid))
    if isinstance(seed_start, bool) or not 0 <= int(seed_start) <= 0xFFFFFFFF:
        raise ValueError("seed-start must be a non-negative 32-bit integer")
    if isinstance(seeds_per_world, bool) or not 1 <= int(seeds_per_world) <= 100_000:
        raise ValueError("seeds-per-world must be in [1,100000]")
    last_seed = int(seed_start) + int(seeds_per_world) - 1
    if last_seed > 0xFFFFFFFF:
        raise ValueError("seed bank exceeds 32-bit range")
    seeds = tuple(range(int(seed_start), last_seed + 1))
    repo_root = Path(__file__).resolve().parents[1]
    loader_module = sys.modules.get("diffusion_policy_3d.rl.dppo_uav.integration")
    loader_file = getattr(loader_module, "__file__", None)
    if not loader_file:
        raise RuntimeError("production admission loader has no source file identity")
    source_files = {
        "scripts/audit_reward_v17_admission.py": Path(__file__).resolve(),
        "diffusion_policy_3d/env/kinematic_uav_env.py":
            repo_root / "diffusion_policy_3d/env/kinematic_uav_env.py",
        "diffusion_policy_3d/env/collision_geom.py":
            repo_root / "diffusion_policy_3d/env/collision_geom.py",
        "diffusion_policy_3d/rl/dppo_uav_rebuild/environments/task_spec.py":
            repo_root / "diffusion_policy_3d/rl/dppo_uav_rebuild/environments/task_spec.py",
        "diffusion_policy_3d/rl/dppo_uav_rebuild/audit/admission_geometry.py":
            repo_root / "diffusion_policy_3d/rl/dppo_uav_rebuild/audit/admission_geometry.py",
        "diffusion_policy_3d/rl/dppo_uav_rebuild/audit/reset_bank.py":
            repo_root / "diffusion_policy_3d/rl/dppo_uav_rebuild/audit/reset_bank.py",
        "diffusion_policy_3d/rl/dppo_uav/integration.py": Path(loader_file),
    }
    provenance = build_runtime_provenance(
        config=config,
        manifest=manifest,
        worlds=worlds,
        seed_start=seed_start,
        seeds_per_world=seeds_per_world,
        source_files=source_files,
    )
    input_hash_snapshot = {
        str(config): provenance["config"]["sha256"],
        str(manifest): provenance["admission_manifest"]["sha256"],
    }
    input_hash_snapshot.update({
        str(Path(path).resolve()): provenance["source_files"][label]
        for label, path in source_files.items()
    })
    input_hash_snapshot.update({
        str(Path(task.obstacle_json).resolve()): sha256_file(Path(task.obstacle_json))
        for task in tasks
    })
    files, exit_code = build_audit_files(
        tasks, seeds, _make_kinematic_environment, provenance=provenance,
    )
    assert_file_hashes_unchanged(input_hash_snapshot)
    publish_pack(output, files)
    return exit_code


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--admission-manifest", required=True, type=Path)
    parser.add_argument("--worlds-train-dir", required=True, type=Path)
    parser.add_argument("--seed-start", type=int, default=2026090800)
    parser.add_argument("--seeds-per-world", type=int, default=64)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return run_audit(
            config_path=args.config,
            manifest_path=args.admission_manifest,
            worlds_train_dir=args.worlds_train_dir,
            seed_start=args.seed_start,
            seeds_per_world=args.seeds_per_world,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print("ADMISSION_AUDIT_ERROR: {}: {}".format(type(exc).__name__, exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
