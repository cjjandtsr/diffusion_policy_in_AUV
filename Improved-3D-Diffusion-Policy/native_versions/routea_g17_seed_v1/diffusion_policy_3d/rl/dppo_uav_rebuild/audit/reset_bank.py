"""Deterministic, reset-only admission evidence collection."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

import numpy as np


EVAL_ONLY_WORLD_IDS = frozenset(("w4", "w8", "w11", "w12", "w18"))


@dataclass(frozen=True)
class ResetAuditTask:
    task_id: str
    world_uid: str
    role: str
    admission_id: str
    obstacle_json: str
    split: str = "rl_train"
    training_eligible: bool = True
    historical_eval_only: bool = False

    @property
    def identity(self) -> Tuple[str, str, str, str]:
        return self.role, self.world_uid, self.admission_id, self.task_id


class GeometryAuditPointCloud:
    """Zero-cost reset dependency; it makes no sensor-quality assertion."""

    semantics = "stub_not_audited"

    def synthesize(self, *args, num_points=4096, **kwargs):
        if isinstance(num_points, bool) or int(num_points) != num_points or int(num_points) < 1:
            raise ValueError("num_points must be a positive integer")
        return np.zeros((int(num_points), 3), dtype=np.float32)


def canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    text = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return (text + "\n").encode("utf-8")


def _clean_text(value: Any, field: str) -> str:
    text = str(value)
    if not text or "\x00" in text:
        raise ValueError("{} must be non-empty text".format(field))
    return text


def normalize_tasks_and_seeds(
    tasks: Sequence[ResetAuditTask],
    seeds: Sequence[int],
) -> Tuple[Tuple[ResetAuditTask, ...], Tuple[int, ...]]:
    normalized = []
    identities = set()
    for task in tasks:
        if not isinstance(task, ResetAuditTask):
            raise TypeError("tasks must contain ResetAuditTask")
        cleaned = ResetAuditTask(
            task_id=_clean_text(task.task_id, "task_id"),
            world_uid=_clean_text(task.world_uid, "world_uid"),
            role=_clean_text(task.role, "role"),
            admission_id=_clean_text(task.admission_id, "admission_id"),
            obstacle_json=_clean_text(task.obstacle_json, "obstacle_json"),
            split=_clean_text(task.split, "split"),
            training_eligible=bool(task.training_eligible),
            historical_eval_only=bool(task.historical_eval_only),
        )
        uid = cleaned.world_uid.lower()
        if (uid in EVAL_ONLY_WORLD_IDS or cleaned.historical_eval_only
                or cleaned.split != "rl_train" or not cleaned.training_eligible):
            raise ValueError("eval-only or non-training task rejected: {}".format(cleaned.world_uid))
        if cleaned.identity in identities:
            raise ValueError("duplicate task identity: {!r}".format(cleaned.identity))
        identities.add(cleaned.identity)
        normalized.append(cleaned)
    if not normalized:
        raise ValueError("at least one audit task is required")

    normalized_seeds = []
    seen_seeds = set()
    for seed in seeds:
        if isinstance(seed, bool) or int(seed) != seed or not 0 <= int(seed) <= 0xFFFFFFFF:
            raise ValueError("seed must be a non-negative 32-bit integer")
        seed_value = int(seed)
        if seed_value in seen_seeds:
            raise ValueError("duplicate seed: {}".format(seed_value))
        seen_seeds.add(seed_value)
        normalized_seeds.append(seed_value)
    if not normalized_seeds:
        raise ValueError("at least one seed is required")
    return tuple(sorted(normalized, key=lambda item: item.identity)), tuple(sorted(normalized_seeds))


def _finite_vector(value: Any, field: str):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError("non-finite reset state in {}".format(field))
    return [float(item) for item in array]


def _finite_scalar(value: Any, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite reset state in {}".format(field))
    return result


def _record_from_environment(
    env: Any,
    task: ResetAuditTask,
    task_index: int,
    seed: int,
) -> Dict[str, Any]:
    fallback_type = getattr(env, "_episode_fallback_type", None)
    if fallback_type not in (None, "fallback_validated", "fallback_invalid"):
        raise ValueError("unknown fallback type: {!r}".format(fallback_type))
    fallback_reasons = sorted(str(item) for item in getattr(env, "_episode_fallback_reasons", []))
    rejection = {
        str(key): int(value)
        for key, value in sorted(getattr(env, "_last_rejection_histogram", {}).items())
    }
    if any(value < 0 for value in rejection.values()):
        raise ValueError("rejection histogram counts must be non-negative")
    status = "pass"
    sampling_outcome = "sampled"
    if fallback_type == "fallback_validated":
        sampling_outcome = "fallback_validated"
    elif fallback_type == "fallback_invalid":
        status = "invalid_fallback"
        sampling_outcome = "fallback_invalid"
    record = {
        "schema": "reward_v17_reset_audit_v1",
        "task_index": int(task_index),
        "task_id": task.task_id,
        "world_uid": task.world_uid,
        "role": task.role,
        "admission_id": task.admission_id,
        "seed": int(seed),
        "status": status,
        "sampling_outcome": sampling_outcome,
        "start_world": _finite_vector(env._drone_pos, "start_world"),
        "goal_world": _finite_vector(env.goal_w, "goal_world"),
        "start_yaw": _finite_scalar(env._drone_yaw, "start_yaw"),
        "fallback_type": fallback_type,
        "fallback_reasons": fallback_reasons,
        "rejection_histogram": rejection,
        "episode_step": int(getattr(env, "episode_step", 0)),
        "pointcloud_semantics": GeometryAuditPointCloud.semantics,
        "finite": True,
    }
    canonical_json_bytes(record)
    return record


def audit_reset_bank(
    tasks: Sequence[ResetAuditTask],
    seeds: Sequence[int],
    make_environment: Callable[[ResetAuditTask, int], Any],
) -> Dict[str, Any]:
    ordered_tasks, ordered_seeds = normalize_tasks_and_seeds(tasks, seeds)
    records = []
    for task_index, task in enumerate(ordered_tasks):
        for seed in ordered_seeds:
            environment = make_environment(task, seed)
            if environment is None:
                raise ValueError("environment factory returned None")
            environment.reset(seed=seed)
            records.append(
                _record_from_environment(environment, task, task_index, seed)
            )
    invalid = sum(record["status"] == "invalid_fallback" for record in records)
    validated = sum(
        record["sampling_outcome"] == "fallback_validated" for record in records
    )
    result = {
        "ordered_task_identities": [list(task.identity) for task in ordered_tasks],
        "ordered_seeds": list(ordered_seeds),
        "records": records,
        "summary": {
            "task_count": len(ordered_tasks),
            "seed_count": len(ordered_seeds),
            "expected_reset_count": len(ordered_tasks) * len(ordered_seeds),
            "reset_count": len(records),
            "invalid_fallback_count": int(invalid),
            "validated_fallback_count": int(validated),
            "pointcloud_semantics": GeometryAuditPointCloud.semantics,
        },
    }
    canonical_json_bytes(result)
    return result


__all__ = (
    "EVAL_ONLY_WORLD_IDS",
    "GeometryAuditPointCloud",
    "ResetAuditTask",
    "audit_reset_bank",
    "canonical_json_bytes",
    "normalize_tasks_and_seeds",
)
