"""Fixed32 parent audit, causal strata, and frozen sentinel selection."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Sequence

import numpy as np

from scripts.analyze_dppo_u32_trajectory_repeats import (
    obstacle_clearance,
    parse_trace_csv,
    trajectory_metrics,
)

from .contracts import DiagnosticFailure, canonical_tree_sha256, load_strict_json


STRATA = (
    "S1_SEED1_RESCUE",
    "S2_SEED3_BREAK",
    "S3_STABLE_FAILURE",
    "S4_CLEAR_STABLE_SUCCESS",
)
STRATUM_DEFINITIONS = {
    "S1_SEED1_RESCUE": "BC does not reach 0.5 m sphere and seed1-u32 does",
    "S2_SEED3_BREAK": "BC reaches 0.5 m sphere and seed3-u32 does not",
    "S3_STABLE_FAILURE": "BC and all three u32 policies do not reach 0.5 m sphere",
    "S4_CLEAR_STABLE_SUCCESS": (
        "BC and all three u32 policies reach 0.5 m sphere and the static "
        "start-goal segment has surface clearance >0.1 m"
    ),
}
ANCHOR_KINDS = ("early", "closest", "clearance")
ANCHOR_PREFERENCE = {
    "S1_SEED1_RESCUE": "clearance",
    "S2_SEED3_BREAK": "clearance",
    "S3_STABLE_FAILURE": "closest",
    "S4_CLEAR_STABLE_SUCCESS": "early",
}


@dataclass(frozen=True)
class ParentRepeat:
    task_id: str
    repeat: int
    env_reset_seed: int
    world: str
    split: str
    role: str
    start: tuple[float, float, float]
    goal: tuple[float, float, float]
    bc_reached: bool
    seed1_reached: bool
    seed2_reached: bool
    seed3_reached: bool
    clear_line: bool
    bc_trace_sha256: str
    rl_trace_sha256: tuple[str, str, str]
    trace_paths: tuple[str, str, str, str]


@dataclass(frozen=True)
class SelectedTask:
    stratum: str
    task_id: str
    repeat: int
    env_reset_seed: int
    world: str
    split: str
    role: str
    selection_key: str
    parent: ParentRepeat


@dataclass(frozen=True)
class SentinelEntry:
    stratum: str
    task_id: str
    repeat: int
    env_reset_seed: int
    snapshot_id: str
    anchor_kind: str
    selection_key: str
    substitution_reason: str | None


@dataclass(frozen=True)
class StratumSelection(Mapping):
    selected: Mapping[str, tuple[SelectedTask, ...]]
    status: Mapping[str, str]
    candidate_task_count: Mapping[str, int]
    max_tasks: int
    digest: str

    def __getitem__(self, key: str) -> tuple[SelectedTask, ...]:
        return self.selected[key]

    def __iter__(self) -> Iterator[str]:
        return iter(STRATA)

    def __len__(self) -> int:
        return len(STRATA)


@dataclass(frozen=True)
class _TraceRecord:
    task_id: str
    repeat: int
    env_reset_seed: int
    world: str
    split: str
    role: str
    policy: str
    done_reason: str
    start: tuple[float, float, float]
    goal: tuple[float, float, float]
    reached: bool
    sha256: str
    path: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _xyz(value: Any, *, label: str) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise DiagnosticFailure("PAIRED_IDENTITY_MISMATCH", f"invalid {label}: {value!r}")
    return tuple(float(item) for item in array)


def _same_identity(left: _TraceRecord, right: _TraceRecord) -> bool:
    return (
        left.task_id,
        left.repeat,
        left.env_reset_seed,
        left.world,
        left.split,
        left.start,
        left.goal,
    ) == (
        right.task_id,
        right.repeat,
        right.env_reset_seed,
        right.world,
        right.split,
        right.start,
        right.goal,
    )


def minimum_segment_clearance(
    start: Sequence[float],
    goal: Sequence[float],
    obstacles: Sequence[Mapping[str, Any]],
    *,
    spacing_m: float = 0.05,
) -> float:
    start_array = np.asarray(start, dtype=np.float64)
    goal_array = np.asarray(goal, dtype=np.float64)
    length = float(np.linalg.norm(goal_array - start_array))
    steps = max(1, int(math.ceil(length / float(spacing_m))))
    fractions = np.linspace(0.0, 1.0, steps + 1)
    points = start_array[None, :] + fractions[:, None] * (goal_array - start_array)[None, :]
    if not obstacles:
        return float("inf")
    return min(
        obstacle_clearance(point, dict(obstacle))
        for obstacle in obstacles
        for point in points
    )


def assign_stratum(parent_repeat: ParentRepeat) -> str | None:
    row = parent_repeat
    matches = []
    if not row.bc_reached and row.seed1_reached:
        matches.append("S1_SEED1_RESCUE")
    if row.bc_reached and not row.seed3_reached:
        matches.append("S2_SEED3_BREAK")
    if not any(
        (row.bc_reached, row.seed1_reached, row.seed2_reached, row.seed3_reached)
    ):
        matches.append("S3_STABLE_FAILURE")
    if (
        all((row.bc_reached, row.seed1_reached, row.seed2_reached, row.seed3_reached))
        and row.clear_line
    ):
        matches.append("S4_CLEAR_STABLE_SUCCESS")
    if len(matches) > 1:
        raise DiagnosticFailure(
            "STRATUM_DEFINITION_OVERLAP",
            f"task={row.task_id} repeat={row.repeat} matches={matches}",
        )
    return matches[0] if matches else None


def _selection_key(stratum: str, task_id: str, reset_seed: int) -> str:
    return hashlib.sha256(
        f"{stratum}||{task_id}||{int(reset_seed)}".encode("utf-8")
    ).hexdigest()


def select_stratified_tasks(
    rows: Iterable[ParentRepeat], max_tasks: int = 8
) -> StratumSelection:
    if int(max_tasks) < 1:
        raise DiagnosticFailure("INVALID_SELECTION_LIMIT", f"max_tasks={max_tasks}")
    seen: Dict[tuple[str, int], ParentRepeat] = {}
    candidates: Dict[str, list[ParentRepeat]] = {name: [] for name in STRATA}
    for row in rows:
        identity = (row.task_id, int(row.repeat))
        if identity in seen:
            raise DiagnosticFailure(
                "PAIRED_IDENTITY_MISMATCH", f"duplicate parent identity: {identity}"
            )
        seen[identity] = row
        stratum = assign_stratum(row)
        if stratum is not None:
            candidates[stratum].append(row)

    selected: Dict[str, tuple[SelectedTask, ...]] = {}
    status: Dict[str, str] = {}
    task_counts: Dict[str, int] = {}
    for stratum in STRATA:
        ordered = sorted(
            candidates[stratum],
            key=lambda row: (
                _selection_key(stratum, row.task_id, row.env_reset_seed),
                row.task_id,
                row.repeat,
            ),
        )
        task_counts[stratum] = len({row.task_id for row in ordered})
        chosen: list[SelectedTask] = []
        used_tasks: set[str] = set()
        for row in ordered:
            if row.task_id in used_tasks:
                continue
            used_tasks.add(row.task_id)
            key = _selection_key(stratum, row.task_id, row.env_reset_seed)
            chosen.append(
                SelectedTask(
                    stratum=stratum,
                    task_id=row.task_id,
                    repeat=row.repeat,
                    env_reset_seed=row.env_reset_seed,
                    world=row.world,
                    split=row.split,
                    role=row.role,
                    selection_key=key,
                    parent=row,
                )
            )
            if len(chosen) == int(max_tasks):
                break
        selected[stratum] = tuple(chosen)
        status[stratum] = (
            "READY" if len(chosen) == int(max_tasks) else "UNDERPOWERED_STRATUM"
        )
    digest = canonical_tree_sha256(
        {
            stratum: [item.selection_key for item in selected[stratum]]
            for stratum in STRATA
        }
    )
    return StratumSelection(selected, status, task_counts, int(max_tasks), digest)


def build_stratum_manifest(
    parent_rows: Sequence[ParentRepeat],
    selection: StratumSelection,
    *,
    parent_sources: Sequence[Mapping[str, Any]] = (),
    geometry_sources: Sequence[Mapping[str, Any]] = (),
) -> Dict[str, Any]:
    parent_task_count = len({row.task_id for row in parent_rows})
    strata: Dict[str, Any] = {}
    for name in STRATA:
        chosen = selection[name]
        status = selection.status[name]
        strata[name] = {
            "definition": STRATUM_DEFINITIONS[name],
            "candidate_task_count": selection.candidate_task_count[name],
            "selected_count": len(chosen),
            "status": status,
            "underpowered_reason": (
                None
                if status == "READY"
                else f"selected {len(chosen)} of required {selection.max_tasks} unique tasks"
            ),
            "inclusion_rule": (
                "one repeat per unique task; ascending sha256("
                "'<stratum>||<task_id>||<env_reset_seed>'); no cross-stratum backfill"
            ),
            "inclusion_probability": None,
            "selected": [
                {
                    "task_id": item.task_id,
                    "repeat": item.repeat,
                    "env_reset_seed": item.env_reset_seed,
                    "world": item.world,
                    "split": item.split,
                    "role": item.role,
                    "selection_key": item.selection_key,
                }
                for item in chosen
            ],
        }
    return {
        "schema": "routea_b0_stratum_manifest_v1",
        "parent_task_count": parent_task_count,
        "parent_repeat_count": len(parent_rows),
        "selection_digest": selection.digest,
        "allows_parent_weighted_pooling": False,
        "pooling_reason": (
            "case-control enrichment has no parent prevalence or inclusion probabilities"
        ),
        "strata": strata,
        "parent_sources": list(parent_sources),
        "geometry_sources": list(geometry_sources),
    }


def _snapshot_field(row: Any, name: str) -> Any:
    if isinstance(row, Mapping):
        if name not in row:
            raise DiagnosticFailure(
                "INVALID_SNAPSHOT_MANIFEST", f"missing snapshot field: {name}"
            )
        return row[name]
    if not hasattr(row, name):
        raise DiagnosticFailure(
            "INVALID_SNAPSHOT_MANIFEST", f"missing snapshot field: {name}"
        )
    return getattr(row, name)


def freeze_sentinel_subset(snapshot_rows: Iterable[Any]) -> tuple[SentinelEntry, ...]:
    grouped: Dict[str, Dict[str, list[Any]]] = {
        stratum: defaultdict(list) for stratum in STRATA
    }
    snapshot_identities: Dict[str, tuple[str, str, int, int]] = {}
    seen_snapshot_anchors: set[tuple[str, str]] = set()
    for row in snapshot_rows:
        snapshot_id = str(_snapshot_field(row, "snapshot_id"))
        stratum = str(_snapshot_field(row, "stratum"))
        task_id = str(_snapshot_field(row, "task_id"))
        anchor = str(_snapshot_field(row, "anchor_kind"))
        identity = (
            stratum,
            task_id,
            int(_snapshot_field(row, "repeat")),
            int(_snapshot_field(row, "env_reset_seed")),
        )
        previous_identity = snapshot_identities.get(snapshot_id)
        if previous_identity is not None and previous_identity != identity:
            raise DiagnosticFailure(
                "INVALID_SNAPSHOT_MANIFEST",
                f"snapshot_id aliases different physical identities: {snapshot_id}",
            )
        snapshot_identities[snapshot_id] = identity
        snapshot_anchor = (snapshot_id, anchor)
        if snapshot_anchor in seen_snapshot_anchors:
            raise DiagnosticFailure(
                "INVALID_SNAPSHOT_MANIFEST",
                f"duplicate snapshot anchor: snapshot_id={snapshot_id} anchor={anchor}",
            )
        seen_snapshot_anchors.add(snapshot_anchor)
        if stratum not in STRATA or anchor not in ANCHOR_KINDS:
            raise DiagnosticFailure(
                "INVALID_SNAPSHOT_MANIFEST",
                f"snapshot={snapshot_id} stratum={stratum} anchor={anchor}",
            )
        grouped[stratum][task_id].append(row)

    frozen: list[SentinelEntry] = []
    for stratum in STRATA:
        task_rows = grouped[stratum]
        task_order = sorted(
            task_rows,
            key=lambda task_id: (
                _selection_key(
                    stratum,
                    task_id,
                    int(_snapshot_field(task_rows[task_id][0], "env_reset_seed")),
                ),
                task_id,
            ),
        )[:2]
        for task_id in task_order:
            rows = task_rows[task_id]
            preferred = ANCHOR_PREFERENCE[stratum]
            anchor_order = [preferred] + [
                anchor for anchor in ANCHOR_KINDS if anchor != preferred
            ]
            chosen = None
            for anchor in anchor_order:
                matches = sorted(
                    (
                        row
                        for row in rows
                        if str(_snapshot_field(row, "anchor_kind")) == anchor
                    ),
                    key=lambda row: str(_snapshot_field(row, "snapshot_id")),
                )
                if matches:
                    chosen = matches[0]
                    break
            if chosen is None:
                continue
            reset_seed = int(_snapshot_field(chosen, "env_reset_seed"))
            frozen.append(
                SentinelEntry(
                    stratum=stratum,
                    task_id=task_id,
                    repeat=int(_snapshot_field(chosen, "repeat")),
                    env_reset_seed=reset_seed,
                    snapshot_id=str(_snapshot_field(chosen, "snapshot_id")),
                    anchor_kind=str(_snapshot_field(chosen, "anchor_kind")),
                    selection_key=_selection_key(stratum, task_id, reset_seed),
                    substitution_reason=(
                        None
                        if str(_snapshot_field(chosen, "anchor_kind")) == preferred
                        else f"preferred {preferred} unavailable after snapshot deduplication; used frozen fallback order early->closest->clearance"
                    ),
                )
            )
    return tuple(frozen)


def _geometry_root_map(
    geometry_roots: Mapping[str, Path | str] | Sequence[Path | str],
) -> Dict[str, Path]:
    if isinstance(geometry_roots, Mapping):
        return {str(key): Path(value) for key, value in geometry_roots.items()}
    roots = list(geometry_roots)
    if len(roots) != 2:
        raise DiagnosticFailure(
            "GEOMETRY_IDENTITY_MISMATCH", "geometry_roots must contain layerA and rl_test"
        )
    return {"layerA": Path(roots[0]), "rl_test": Path(roots[1])}


def _load_geometry_map(
    request: Mapping[str, Any], roots: Mapping[str, Path]
) -> Dict[str, list[dict[str, Any]]]:
    try:
        sources = request["resolved_inputs"]["obstacle_sources"]
    except (KeyError, TypeError) as exc:
        raise DiagnosticFailure(
            "GEOMETRY_IDENTITY_MISMATCH", "request has no obstacle_sources"
        ) from exc
    geometries: Dict[str, list[dict[str, Any]]] = {}
    for world, source in sorted(sources.items()):
        split = str(source["split"])
        if split not in roots:
            raise DiagnosticFailure(
                "GEOMETRY_IDENTITY_MISMATCH", f"no geometry root for split={split}"
            )
        path = roots[split] / Path(str(source["path"])).name
        actual_sha = _sha256(path) if path.is_file() else None
        if actual_sha != str(source["sha256"]).lower():
            raise DiagnosticFailure(
                "GEOMETRY_IDENTITY_MISMATCH",
                f"world={world} path={path} expected={source['sha256']} actual={actual_sha}",
            )
        data = load_strict_json(path)
        if not isinstance(data, Mapping) or not isinstance(data.get("obstacles"), list):
            raise DiagnosticFailure(
                "GEOMETRY_IDENTITY_MISMATCH", f"invalid obstacle JSON: {path}"
            )
        geometries[str(world)] = [dict(item) for item in data["obstacles"]]
    return geometries


def _load_seed_directory(seed_dir: Path, seed_index: int) -> tuple[Dict[Any, _TraceRecord], dict]:
    request_path = seed_dir / "replay_request.json"
    verification_path = seed_dir / "replay_verification.json"
    if not request_path.is_file() or not verification_path.is_file():
        raise DiagnosticFailure(
            "PAIRED_IDENTITY_MISMATCH", f"missing request/verification in {seed_dir}"
        )
    request = load_strict_json(request_path)
    verification = load_strict_json(verification_path)
    csv_paths = sorted(seed_dir.glob("*.csv"))
    checks = verification.get("checks") if isinstance(verification, Mapping) else None
    if len(csv_paths) != 1280 or not isinstance(checks, list) or len(checks) != 1280:
        raise DiagnosticFailure(
            "PAIRED_IDENTITY_MISMATCH",
            f"seed{seed_index}: csv={len(csv_paths)} checks={None if checks is None else len(checks)}; expected 1280/1280",
        )
    checkpoint = str((request.get("resolved_inputs") or {}).get("checkpoint_dir", ""))
    if f"seed{seed_index}" not in checkpoint:
        raise DiagnosticFailure(
            "PAIRED_IDENTITY_MISMATCH",
            f"directory order says seed{seed_index}, checkpoint={checkpoint}",
        )
    check_by_file: Dict[str, Mapping[str, Any]] = {}
    for check in checks:
        filename = str(check.get("trajectory_file", ""))
        if not filename or filename in check_by_file:
            raise DiagnosticFailure(
                "PAIRED_IDENTITY_MISMATCH", f"duplicate/missing trajectory_file: {filename!r}"
            )
        check_by_file[filename] = check
    if set(check_by_file) != {path.name for path in csv_paths}:
        raise DiagnosticFailure(
            "PAIRED_IDENTITY_MISMATCH", f"CSV/check file set differs in {seed_dir}"
        )

    records: Dict[Any, _TraceRecord] = {}
    for path in csv_paths:
        check = check_by_file[path.name]
        actual_sha = _sha256(path)
        if actual_sha != str(check.get("trajectory_sha256", "")).lower():
            raise DiagnosticFailure(
                "PAIRED_IDENTITY_MISMATCH", f"trajectory hash mismatch: {path}"
            )
        try:
            trace = parse_trace_csv(path)
            header = trace.header
            actual = check["actual"]
            task_id = str(header["task_id"])
            policy = str(header["policy"])
            repeat = int(header["repeat"])
            reset_seed = int(header["env_reset_seed"])
            world = str(header["world"])
            split = str(header["split"])
            start = _xyz(header["start"], label="start")
            goal = _xyz(header["goal"], label="goal")
            executed_xyz = np.asarray(trace.executed[:, 3:6], dtype=np.float64)
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise DiagnosticFailure(
                "PAIRED_IDENTITY_MISMATCH", f"invalid trace/check {path}: {exc}"
            ) from exc
        comparisons = {
            "task_id": (task_id, str(check.get("task_id"))),
            "policy": (policy, str(check.get("policy"))),
            "repeat": (repeat, int(check.get("repeat", -1)) + 1),
            "actual.task_id": (task_id, str(actual.get("task_id"))),
            "actual.policy": (policy, str(actual.get("policy"))),
            "actual.world_id": (world, str(actual.get("world_id"))),
            "actual.split": (split, str(actual.get("split"))),
            "actual.seed": (reset_seed, int(actual.get("seed", -1))),
        }
        mismatch = {key: pair for key, pair in comparisons.items() if pair[0] != pair[1]}
        if mismatch:
            raise DiagnosticFailure(
                "PAIRED_IDENTITY_MISMATCH", f"{path.name}: {mismatch}"
            )
        key = (task_id, repeat, policy)
        if key in records or policy not in {"bc", "rl"}:
            raise DiagnosticFailure(
                "PAIRED_IDENTITY_MISMATCH", f"duplicate/invalid policy identity: {key}"
            )
        records[key] = _TraceRecord(
            task_id=task_id,
            repeat=repeat,
            env_reset_seed=reset_seed,
            world=world,
            split=split,
            role=str(actual.get("role", "")),
            policy=policy,
            done_reason=str(actual.get("done_reason", "")),
            start=start,
            goal=goal,
            reached=bool(trajectory_metrics(executed_xyz, np.asarray(goal))["entered_goal_0p5"]),
            sha256=actual_sha,
            path=str(path.resolve()),
        )
    return records, request


def load_fixed32_parent(
    seed_dirs: Sequence[Path | str],
    geometry_roots: Mapping[str, Path | str] | Sequence[Path | str],
) -> tuple[ParentRepeat, ...]:
    if len(seed_dirs) != 3:
        raise DiagnosticFailure(
            "PAIRED_IDENTITY_MISMATCH", f"expected 3 seed directories, got {len(seed_dirs)}"
        )
    loaded = [
        _load_seed_directory(Path(directory), index)
        for index, directory in enumerate(seed_dirs, start=1)
    ]
    record_sets = [set(records) for records, _ in loaded]
    if not all(keys == record_sets[0] for keys in record_sets[1:]):
        raise DiagnosticFailure(
            "PAIRED_IDENTITY_MISMATCH", "seed directories have different task/repeat/policy sets"
        )
    request_geometry_hashes = [
        canonical_tree_sha256(request["resolved_inputs"]["obstacle_sources"])
        for _, request in loaded
    ]
    if len(set(request_geometry_hashes)) != 1:
        raise DiagnosticFailure(
            "GEOMETRY_IDENTITY_MISMATCH", "seed requests bind different obstacle source maps"
        )
    geometries = _load_geometry_map(loaded[0][1], _geometry_root_map(geometry_roots))

    identities = sorted({(task, repeat) for task, repeat, _ in record_sets[0]})
    parent: list[ParentRepeat] = []
    for task_id, repeat in identities:
        bc_records = [records[(task_id, repeat, "bc")] for records, _ in loaded]
        rl_records = [records[(task_id, repeat, "rl")] for records, _ in loaded]
        reference = bc_records[0]
        if len({record.sha256 for record in bc_records}) != 1:
            raise DiagnosticFailure(
                "PAIRED_IDENTITY_MISMATCH",
                f"BC trace differs across seeds: task={task_id} repeat={repeat}",
            )
        for record in bc_records[1:] + rl_records:
            if not _same_identity(reference, record):
                raise DiagnosticFailure(
                    "PAIRED_IDENTITY_MISMATCH",
                    f"paired identity differs: task={task_id} repeat={repeat}",
                )
        if reference.world not in geometries:
            raise DiagnosticFailure(
                "GEOMETRY_IDENTITY_MISMATCH", f"no obstacles for world={reference.world}"
            )
        clear_line = (
            minimum_segment_clearance(
                reference.start, reference.goal, geometries[reference.world]
            )
            > 0.1
        )
        parent.append(
            ParentRepeat(
                task_id=task_id,
                repeat=repeat,
                env_reset_seed=reference.env_reset_seed,
                world=reference.world,
                split=reference.split,
                role=reference.role,
                start=reference.start,
                goal=reference.goal,
                bc_reached=reference.reached,
                seed1_reached=rl_records[0].reached,
                seed2_reached=rl_records[1].reached,
                seed3_reached=rl_records[2].reached,
                clear_line=clear_line,
                bc_trace_sha256=reference.sha256,
                rl_trace_sha256=tuple(record.sha256 for record in rl_records),
                trace_paths=(reference.path,) + tuple(record.path for record in rl_records),
            )
        )
    return tuple(parent)
