#!/usr/bin/env python3
"""Analyze Route-A u32 BC/RL traces from raw paths and obstacle geometry.

The analysis deliberately recomputes behavior from the CSV polylines.  Endpoint
JSON fields are used only for identity/outcome labels and are audited against the
recomputed values.  Repeats are paired within task and collapsed before any
task/world summary; they are never counted as extra task-level samples.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PLANNED_COLUMNS = (
    "decision_step", "k", "time_s", "x", "y", "z", "vx", "vy", "vz",
    "ax", "ay", "az", "yaw", "yaw_rate",
)
EXECUTED_COLUMNS = ("decision_step", "k", "time_s", "x", "y", "z")
SUCCESS_RADIUS_M = 1.0
COLLISION_CLEARANCE_M = 0.1


@dataclass(frozen=True)
class TraceData:
    path: Path
    header: dict[str, Any]
    planned: np.ndarray
    executed: np.ndarray


def _parse_header_value(value: str) -> Any:
    value = value.strip()
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value


def resolve_cli_path(path: Path | str) -> Path:
    """Resolve before appending long trace names (avoids Win32 relative MAX_PATH)."""
    return Path(path).expanduser().resolve()


def parse_trace_csv(path: Path | str) -> TraceData:
    """Parse the two-table v2 trace format without trusting file-name fields."""
    path = Path(path)
    header: dict[str, Any] = {}
    planned_rows: list[list[float]] = []
    executed_rows: list[list[float]] = []
    table: str | None = None
    columns: tuple[str, ...] | None = None
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line == "# table: planned":
                table, columns = "planned", None
                continue
            if line == "# table: executed":
                table, columns = "executed", None
                continue
            if line.startswith("#"):
                content = line[1:].strip()
                if "=" in content:
                    key, value = content.split("=", 1)
                    header[key.strip()] = _parse_header_value(value)
                continue
            row = next(csv.reader([line]))
            if columns is None:
                columns = tuple(row)
                expected = PLANNED_COLUMNS if table == "planned" else EXECUTED_COLUMNS
                if table is None or columns != expected:
                    raise ValueError(f"{path}: unexpected {table!r} columns {columns}")
                continue
            values = [float(value) for value in row]
            if len(values) != len(columns):
                raise ValueError(f"{path}: row width {len(values)} != {len(columns)}")
            (planned_rows if table == "planned" else executed_rows).append(values)
    required = {"task_id", "world", "split", "policy", "repeat", "start", "goal"}
    missing = required.difference(header)
    if missing or not planned_rows or not executed_rows:
        raise ValueError(f"{path}: incomplete trace; missing={sorted(missing)}")
    return TraceData(path, header, np.asarray(planned_rows), np.asarray(executed_rows))


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if abs(denominator) > 1e-12 else float("nan")


def trajectory_metrics(points: np.ndarray, goal: np.ndarray) -> dict[str, Any]:
    """Geometry-only metrics for one executed world-frame polyline."""
    points = np.asarray(points, dtype=float)
    goal = np.asarray(goal, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 1:
        raise ValueError("points must have shape (N,3), N>=1")
    if goal.shape != (3,) or not np.isfinite(points).all() or not np.isfinite(goal).all():
        raise ValueError("points and goal must be finite")
    start = points[0]
    delta = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(delta, axis=1)
    distances = np.linalg.norm(points - goal, axis=1)
    direct = float(np.linalg.norm(goal - start))
    path_length = float(segment_lengths.sum())
    distance_changes = np.diff(distances)
    if direct > 1e-12:
        axis = (goal - start) / direct
        along = (points - start) @ axis
        cross = np.linalg.norm((points - start) - np.outer(along, axis), axis=1)
    else:
        along = np.zeros(len(points))
        cross = np.linalg.norm(points - start, axis=1)
    closest_index = int(np.argmin(distances))
    backtrack = float(np.maximum(distance_changes, 0.0).sum())
    return {
        "start_distance_m": float(distances[0]),
        "final_distance_m": float(distances[-1]),
        "min_distance_m": float(distances[closest_index]),
        "closest_index": closest_index,
        "path_length_m": path_length,
        "net_progress_m": float(distances[0] - distances[-1]),
        "progress_per_path": _safe_ratio(float(distances[0] - distances[-1]), path_length),
        "directness": _safe_ratio(direct, path_length),
        "backtrack_length_m": backtrack,
        "backtrack_fraction": _safe_ratio(backtrack, path_length),
        "nonprogress_step_fraction": float(np.mean(distance_changes >= -1e-6)) if len(distance_changes) else 0.0,
        "max_cross_track_m": float(cross.max()),
        "mean_cross_track_m": float(cross.mean()),
        "z_min_m": float(points[:, 2].min()),
        "z_max_m": float(points[:, 2].max()),
        "z_span_m": float(np.ptp(points[:, 2])),
        "final_abs_z_error_m": float(abs(points[-1, 2] - goal[2])),
        "post_closest_retreat_m": float(distances[-1] - distances[closest_index]),
        "overshoot_0p5": bool(float(along.max()) > direct + 0.5),
        "entered_goal_0p5": bool(np.any(distances <= 0.5)),
        "entered_goal_1p0": bool(np.any(distances <= SUCCESS_RADIUS_M)),
        "endpoint_count": int(len(points)),
    }


def planning_metrics(planned: np.ndarray, goal: np.ndarray) -> dict[str, Any]:
    """Measure H4/H8 goal promises and next-decision plan revision."""
    planned = np.asarray(planned, dtype=float)
    goal = np.asarray(goal, dtype=float)
    decisions = sorted({int(value) for value in planned[:, 0]})
    h4_min: list[float] = []
    h8_min: list[float] = []
    tail_only = 0
    revisions: list[float] = []
    decision_rows: dict[int, np.ndarray] = {}
    for decision in decisions:
        rows = planned[planned[:, 0] == decision]
        rows = rows[np.argsort(rows[:, 1])]
        if len(rows) != 8 or tuple(rows[:, 1].astype(int)) != tuple(range(8)):
            raise ValueError(f"decision {decision} does not contain exactly k=0..7")
        decision_rows[decision] = rows
        distances = np.linalg.norm(rows[:, 3:6] - goal, axis=1)
        h4_min.append(float(distances[:4].min()))
        h8_min.append(float(distances.min()))
        tail_only += int(distances[:4].min() > SUCCESS_RADIUS_M and distances[4:].min() <= SUCCESS_RADIUS_M)
    for left, right in zip(decisions, decisions[1:]):
        if right != left + 1:
            continue
        previous_tail = decision_rows[left][4:8, 3:6]
        current_head = decision_rows[right][0:4, 3:6]
        revisions.extend(np.linalg.norm(previous_tail - current_head, axis=1).tolist())
    executed_horizon_yaw = np.asarray([decision_rows[d][3, 12] for d in decisions])
    yaw_delta = np.diff(np.unwrap(executed_horizon_yaw))
    return {
        "decisions": len(decisions),
        "mean_h4_min_goal_m": float(np.mean(h4_min)),
        "mean_h8_min_goal_m": float(np.mean(h8_min)),
        "tail_only_goal_decisions": int(tail_only),
        "tail_only_goal_rate": float(tail_only / len(decisions)),
        "mean_tail_revision_m": float(np.mean(revisions)) if revisions else 0.0,
        "p90_tail_revision_m": float(np.quantile(revisions, 0.9)) if revisions else 0.0,
        "yaw_total_variation_rad": float(np.abs(yaw_delta).sum()),
        "yaw_reversal_count": int(np.sum(np.sign(yaw_delta[1:]) * np.sign(yaw_delta[:-1]) < 0)) if len(yaw_delta) > 1 else 0,
    }


def obstacle_clearances(points: np.ndarray, obstacle: dict[str, Any]) -> np.ndarray:
    """Vectorized signed Euclidean point-to-solid distance in world frame."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N,3)")
    center_z = float(obstacle.get("z", float(obstacle.get("height", obstacle.get("size_z", 0.0))) / 2.0))
    relative = points - np.array([float(obstacle["x"]), float(obstacle["y"]), center_z])
    dx, dy, dz = relative[:, 0], relative[:, 1], relative[:, 2]
    shape = obstacle.get("shape", "cylinder")
    if shape == "cylinder":
        q = np.column_stack([
            np.hypot(dx, dy) - float(obstacle["radius"]),
            np.abs(dz) - float(obstacle["height"]) / 2.0,
        ])
    elif shape == "box":
        yaw = float(obstacle.get("yaw", 0.0))
        c, s = math.cos(yaw), math.sin(yaw)
        local_x, local_y = c * dx + s * dy, -s * dx + c * dy
        q = np.abs(np.column_stack([local_x, local_y, dz])) - np.array([
            float(obstacle["size_x"]) / 2.0,
            float(obstacle["size_y"]) / 2.0,
            float(obstacle["size_z"]) / 2.0,
        ])
    elif shape == "sphere":
        return np.linalg.norm(relative, axis=1) - float(obstacle["radius"])
    else:
        raise ValueError(f"unsupported obstacle shape {shape!r}")
    return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(np.max(q, axis=1), 0.0)


def obstacle_clearance(point: np.ndarray, obstacle: dict[str, Any]) -> float:
    """Scalar wrapper for one world-frame point."""
    return float(obstacle_clearances(np.asarray(point, dtype=float)[None, :], obstacle)[0])


def _densify(points: np.ndarray, spacing_m: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    dense = [points[0]]
    source_fraction = [0.0]
    total = max(len(points) - 1, 1)
    for index, (start, end) in enumerate(zip(points, points[1:])):
        length = float(np.linalg.norm(end - start))
        steps = max(1, int(math.ceil(length / spacing_m)))
        for step in range(1, steps + 1):
            dense.append(start + (end - start) * (step / steps))
            source_fraction.append((index + step / steps) / total)
    return np.asarray(dense), np.asarray(source_fraction)


def obstacle_metrics(points: np.ndarray, goal: np.ndarray,
                     obstacles: list[dict[str, Any]]) -> dict[str, Any]:
    if not obstacles:
        return {"min_surface_clearance_m": float("inf"), "p10_surface_clearance_m": float("inf"),
                "collision_proxy_margin_m": float("inf"), "first_below_1m_path_fraction": float("nan"),
                "direct_path_min_clearance_m": float("inf"), "lateral_onset_path_fraction": float("nan"),
                "clearance_at_lateral_onset_m": float("nan")}
    dense, fractions = _densify(points)
    clearances = np.min(np.vstack([obstacle_clearances(dense, obs) for obs in obstacles]), axis=0)
    below = np.flatnonzero(clearances < 1.0)
    start = points[0]
    goal = np.asarray(goal, dtype=float)
    direct_line, _ = _densify(np.vstack([start, goal]))
    direct_clearance = float(np.min(np.vstack([
        obstacle_clearances(direct_line, obs) for obs in obstacles
    ])))
    displacement = points - points[0]
    direct_vector = goal - points[0]
    direct_norm = float(np.linalg.norm(direct_vector))
    onset_fraction = float("nan")
    onset_clearance = float("nan")
    if direct_norm > 1e-9:
        axis = direct_vector / direct_norm
        cross = np.linalg.norm(displacement - np.outer(displacement @ axis, axis), axis=1)
        onset = np.flatnonzero(cross > 0.25)
        if len(onset):
            onset_index = int(onset[0])
            onset_fraction = onset_index / max(len(points) - 1, 1)
            onset_clearance = min(obstacle_clearance(points[onset_index], obs) for obs in obstacles)
    return {
        "min_surface_clearance_m": float(clearances.min()),
        "p10_surface_clearance_m": float(np.quantile(clearances, 0.1)),
        "collision_proxy_margin_m": float(clearances.min() - COLLISION_CLEARANCE_M),
        "first_below_1m_path_fraction": float(fractions[below[0]]) if len(below) else float("nan"),
        "direct_path_min_clearance_m": float(direct_clearance),
        "lateral_onset_path_fraction": onset_fraction,
        "clearance_at_lateral_onset_m": float(onset_clearance),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_geometries(request: dict[str, Any], layera: Path, rl_test: Path) -> dict[str, list[dict[str, Any]]]:
    geometries = {}
    for world, source in request["resolved_inputs"]["obstacle_sources"].items():
        root = layera if source["split"] == "layerA" else rl_test
        path = root / Path(source["path"]).name
        if not path.exists() or _sha256(path) != source["sha256"]:
            raise ValueError(f"geometry identity mismatch for {world}: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        geometries[world] = data["obstacles"]
    return geometries


def _classification(done: str, metrics: dict[str, Any]) -> str:
    if done == "reached":
        return "safe_arrival"
    if done == "collision":
        return "collision"
    if metrics["entered_goal_1p0"]:
        return "seal_failure"
    if metrics["final_distance_m"] <= 0.5 * metrics["start_distance_m"]:
        return "timeout_with_progress"
    return "route_failure"


def _numeric_keys(rows: list[dict[str, Any]], prefix: str = "delta_") -> list[str]:
    return sorted(key for key, value in rows[0].items()
                  if key.startswith(prefix) and isinstance(value, (int, float)) and not isinstance(value, bool))


def aggregate_repeats_by_task(rows: list[dict[str, Any]], numeric_keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["seed"]), str(row["task_id"]))].append(row)
    result = []
    for (seed, task_id), members in sorted(groups.items()):
        first = members[0]
        out: dict[str, Any] = {
            "seed": seed, "task_id": task_id, "world": first["world"], "role": first["role"],
            "repeat_count": len(members),
            "lost_rate": float(np.mean([m["bc_done"] == "reached" and m["rl_done"] != "reached" for m in members])),
            "rescue_rate": float(np.mean([m["bc_done"] != "reached" and m["rl_done"] == "reached" for m in members])),
            "bc_success_rate": float(np.mean([m["bc_done"] == "reached" for m in members])),
            "rl_success_rate": float(np.mean([m["rl_done"] == "reached" for m in members])),
        }
        out["delta_success_rate"] = out["rl_success_rate"] - out["bc_success_rate"]
        for key in numeric_keys:
            values = np.asarray([float(m[key]) for m in members], dtype=float)
            finite = values[np.isfinite(values)]
            out[key] = float(finite.mean()) if len(finite) else float("nan")
        result.append(out)
    return result


def world_cluster_interval(rows: list[dict[str, Any]], key: str, *, draws: int = 10000,
                           seed: int = 20260920) -> dict[str, Any]:
    """Equal-world cluster bootstrap after repeats have already been collapsed."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row[key])
        if np.isfinite(value):
            grouped[str(row["world"])].append(value)
    world_means = np.asarray([np.mean(values) for values in grouped.values()], dtype=float)
    if not len(world_means):
        return {"worlds": 0, "tasks": len(rows), "estimate": None, "lower": None, "upper": None}
    result = {"worlds": int(len(world_means)), "tasks": len(rows),
              "estimate": float(world_means.mean()), "lower": None, "upper": None}
    if len(world_means) >= 2:
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, len(world_means), size=(draws, len(world_means)))
        boot = world_means[indices].mean(axis=1)
        result["lower"], result["upper"] = (float(value) for value in np.quantile(boot, [0.025, 0.975]))
    return result


def aggregate_tasks_by_world(rows: list[dict[str, Any]], numeric_keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["seed"]), str(row["world"]))].append(row)
    result = []
    for (seed, world), members in sorted(groups.items()):
        out: dict[str, Any] = {"seed": seed, "world": world, "role": members[0]["role"],
                               "task_count": len(members)}
        for key in numeric_keys:
            values = np.asarray([float(member[key]) for member in members], dtype=float)
            finite = values[np.isfinite(values)]
            out[key] = float(finite.mean()) if len(finite) else float("nan")
        result.append(out)
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _finite_summary(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"n": 0, "mean": None, "median": None, "q25": None, "q75": None}
    return {"n": int(len(array)), "mean": float(array.mean()), "median": float(np.median(array)),
            "q25": float(np.quantile(array, 0.25)), "q75": float(np.quantile(array, 0.75))}


def _aggregate_report(task_rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]],
                      audit: dict[str, Any], bc_identity: dict[str, Any]) -> dict[str, Any]:
    metrics = _numeric_keys(pair_rows) + ["delta_success_rate"]
    report: dict[str, Any] = {"unit_of_analysis": "task after within-task repeat aggregation",
                              "endpoint_audit": audit, "bc_identity": bc_identity, "groups": {}}
    for seed in sorted({row["seed"] for row in task_rows}):
        seed_rows = [row for row in task_rows if row["seed"] == seed]
        group = {
            "tasks": len(seed_rows),
            "mean_bc_success_rate": float(np.mean([row["bc_success_rate"] for row in seed_rows])),
            "mean_rl_success_rate": float(np.mean([row["rl_success_rate"] for row in seed_rows])),
            "mean_lost_rate": float(np.mean([row["lost_rate"] for row in seed_rows])),
            "mean_rescue_rate": float(np.mean([row["rescue_rate"] for row in seed_rows])),
            "metric_deltas": {key: _finite_summary(row[key] for row in seed_rows) for key in metrics},
            "world_clustered": {key: world_cluster_interval(seed_rows, key) for key in metrics},
            "roles": {},
        }
        for role in sorted({row["role"] for row in seed_rows}):
            role_rows = [row for row in seed_rows if row["role"] == role]
            group["roles"][role] = {
                "tasks": len(role_rows),
                "bc_success_rate": float(np.mean([row["bc_success_rate"] for row in role_rows])),
                "rl_success_rate": float(np.mean([row["rl_success_rate"] for row in role_rows])),
                "lost_rate": float(np.mean([row["lost_rate"] for row in role_rows])),
                "rescue_rate": float(np.mean([row["rescue_rate"] for row in role_rows])),
                "metric_deltas": {key: _finite_summary(row[key] for row in role_rows) for key in metrics},
                "world_clustered": {key: world_cluster_interval(role_rows, key) for key in metrics},
            }
        report["groups"][seed] = group
    return report


def analyze(traces_root: Path, layera: Path, rl_test: Path, output: Path) -> dict[str, Any]:
    run_dirs = sorted(path for path in traces_root.iterdir()
                      if path.is_dir() and re.search(r"seed[123]_v2$", path.name))
    if len(run_dirs) != 3:
        raise ValueError(f"expected three seed*_v2 directories, got {len(run_dirs)}")
    episode_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    bc_hashes: dict[tuple[str, int], list[str]] = defaultdict(list)
    final_errors: list[float] = []
    terminal_errors: list[float] = []
    length_errors: list[float] = []
    source_geometry_hashes: dict[str, str] = {}
    for run_dir in run_dirs:
        seed = re.search(r"(seed[123])_v2$", run_dir.name).group(1)  # type: ignore[union-attr]
        request = json.loads((run_dir / "replay_request.json").read_text(encoding="utf-8"))
        verification = json.loads((run_dir / "replay_verification.json").read_text(encoding="utf-8"))
        geometries = _load_geometries(request, layera, rl_test)
        for world, source in request["resolved_inputs"]["obstacle_sources"].items():
            previous = source_geometry_hashes.setdefault(world, source["sha256"])
            if previous != source["sha256"]:
                raise ValueError(f"seed geometry mismatch for {world}")
        checks = {entry["trajectory_file"]: entry for entry in verification["checks"]}
        csv_paths = sorted(run_dir.glob("*.csv"))
        if len(csv_paths) != 1280 or len(checks) != 1280:
            raise ValueError(f"{run_dir}: expected 1280 CSV/checks, got {len(csv_paths)}/{len(checks)}")
        for path in csv_paths:
            trace = parse_trace_csv(path)
            check = checks[path.name]
            actual = check["actual"]
            points = trace.executed[:, 3:6]
            goal = np.asarray(trace.header["goal"], dtype=float)
            raw = trajectory_metrics(points, goal)
            plan = planning_metrics(trace.planned, goal)
            obstacle = obstacle_metrics(points, goal, geometries[str(trace.header["world"])])
            final_errors.append(abs(raw["final_distance_m"] - float(actual["final_distance_3d"])))
            terminal_errors.append(float(np.linalg.norm(points[-1] - np.asarray(actual["terminal_xyz"], dtype=float))))
            length_errors.append(abs(raw["path_length_m"] - float(actual["v17_metrics"]["executed_length_m"])))
            row: dict[str, Any] = {
                "seed": seed, "task_id": trace.header["task_id"], "world": trace.header["world"],
                "split": trace.header["split"], "role": actual["role"], "policy": trace.header["policy"],
                "repeat": int(trace.header["repeat"]), "env_reset_seed": int(trace.header["env_reset_seed"]),
                "done": actual["done_reason"], "steps": int(actual["steps"]),
                "trajectory_file": path.name, "classification": _classification(actual["done_reason"], raw),
            }
            row.update(raw)
            row.update(plan)
            row.update(obstacle)
            episode_rows.append(row)
            if row["policy"] == "bc":
                bc_hashes[(row["task_id"], row["repeat"])].append(_sha256(path))
    by_pair: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in episode_rows:
        by_pair[(row["seed"], row["task_id"], row["repeat"])][row["policy"]] = row
    excluded = {"seed", "task_id", "world", "split", "role", "policy", "repeat", "env_reset_seed",
                "done", "trajectory_file", "classification", "closest_index", "endpoint_count", "decisions",
                "entered_goal_0p5", "entered_goal_1p0", "overshoot_0p5"}
    metric_keys = sorted(key for key, value in episode_rows[0].items()
                         if key not in excluded and isinstance(value, (int, float)) and not isinstance(value, bool))
    for (seed, task_id, repeat), policies in sorted(by_pair.items()):
        if set(policies) != {"bc", "rl"}:
            raise ValueError(f"unpaired trace: {(seed, task_id, repeat)}")
        bc, rl = policies["bc"], policies["rl"]
        row = {"seed": seed, "task_id": task_id, "world": bc["world"], "role": bc["role"],
               "repeat": repeat, "bc_done": bc["done"], "rl_done": rl["done"],
               "bc_classification": bc["classification"], "rl_classification": rl["classification"]}
        for key in metric_keys:
            row[f"delta_{key}"] = float(rl[key]) - float(bc[key])
        for key in ("entered_goal_0p5", "entered_goal_1p0", "overshoot_0p5"):
            row[f"bc_{key}"] = bool(bc[key])
            row[f"rl_{key}"] = bool(rl[key])
        pair_rows.append(row)
    numeric_delta_keys = _numeric_keys(pair_rows)
    task_rows = aggregate_repeats_by_task(pair_rows, numeric_delta_keys)
    world_rows = aggregate_tasks_by_world(task_rows, numeric_delta_keys + ["delta_success_rate"])
    identity_mismatches = sum(len(set(hashes)) != 1 for hashes in bc_hashes.values())
    bc_identity = {"task_repeat_keys": len(bc_hashes), "expected_copies_per_key": 3,
                   "copy_count_errors": sum(len(v) != 3 for v in bc_hashes.values()),
                   "hash_mismatches": identity_mismatches}
    audit = {"episodes": len(episode_rows), "pairs": len(pair_rows), "task_seed_units": len(task_rows),
             "max_final_distance_abs_error": max(final_errors),
             "max_terminal_xyz_l2_error": max(terminal_errors),
             "max_path_length_abs_error": max(length_errors),
             "geometry_worlds": len(source_geometry_hashes)}
    report = _aggregate_report(task_rows, pair_rows, audit, bc_identity)
    report["outcome_transitions"] = dict(Counter(
        f"{row['bc_done']}->{row['rl_done']}" for row in pair_rows))
    report["classification_transitions"] = dict(Counter(
        f"{row['bc_classification']}->{row['rl_classification']}" for row in pair_rows))
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "episode_metrics.csv", episode_rows)
    _write_csv(output / "paired_repeat_deltas.csv", pair_rows)
    _write_csv(output / "task_seed_summary.csv", task_rows)
    _write_csv(output / "world_seed_summary.csv", world_rows)
    (output / "aggregate.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces-root", type=Path, required=True)
    parser.add_argument("--layera-geometry", type=Path, required=True)
    parser.add_argument("--rl-test-geometry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        resolve_cli_path(args.traces_root),
        resolve_cli_path(args.layera_geometry),
        resolve_cli_path(args.rl_test_geometry),
        resolve_cli_path(args.output_dir),
    )
    print(json.dumps({"endpoint_audit": report["endpoint_audit"], "bc_identity": report["bc_identity"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
