"""Canonical hash-bound core for formal BC/RL endpoint evaluation.

The executed-source closure includes this module. Formal evaluation therefore
cannot silently replace endpoint semantics without changing source identity.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Optional, Tuple

import numpy as np


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SPLITS = frozenset(("rl_val", "rl_test", "layerA", "historical_eval_only"))
_ROLES = frozenset(
    ("anticipatory", "dense_recovery", "support_expansion", "yaw_lateral_recovery")
)
_HISTORICAL_WORLDS = frozenset(("w4", "w8", "w11", "w12", "w18"))
_DONE_REASONS = frozenset(
    (
        "reached",
        "collision",
        "crash_ground",
        "out_of_bounds",
        "altitude_exceed",
        "invalid_state",
        "time_limit",
    )
)


@dataclass(frozen=True)
class PairedEpisodeRecord:
    task_id: str
    config_id: str
    config_hash: str
    obstacle_hash: str
    source_manifest_hash: str
    policy_identity_hash: str
    world_id: str
    split: str
    role: str
    seed: int
    policy: str
    done_reason: str
    raw_done_reason: str
    episode_return: float
    onset_triggered: bool
    onset_first_step: Optional[int]
    future_clearance_reward_sum: float
    d_soft_h_min: float
    d_soft_h_steps: int
    support_cells: int
    terminal_xyz: Tuple[float, float, float]
    min_xyz: Tuple[float, float, float]
    max_xyz: Tuple[float, float, float]
    action_dz_mean: float
    action_dz_min: float
    action_dz_max: float
    trap_recovery_count: int
    loop_count: int
    side_flip_count_max: int
    steps: int
    evaluator: str = "lightweight_kinematic"
    invalid_steps: int = 0
    reward_schema_version: str = "legacy_unknown"
    trajectory_statistics_scope: str = "complete_decision_endpoints"
    v17_metrics: Optional[dict] = None
    goal_xyz: Optional[Tuple[float, float, float]] = None
    final_distance_3d: Optional[float] = None
    final_error_xy: Optional[float] = None
    final_error_z: Optional[float] = None
    terminal_safety_flags: Tuple[str, ...] = ()
    observation_validity_summary: Optional[dict] = None

    def __post_init__(self):
        if not self.task_id or not self.config_id or not self.world_id:
            raise ValueError("task_id, config_id, and world_id are required")
        for name in (
            "config_hash",
            "obstacle_hash",
            "source_manifest_hash",
            "policy_identity_hash",
        ):
            if _SHA256_RE.fullmatch(getattr(self, name)) is None:
                raise ValueError("{} must be a lowercase SHA256".format(name))
        if self.split not in _SPLITS:
            raise ValueError("unknown evaluation split")
        if self.role not in _ROLES:
            raise ValueError("unknown Route A evaluation role")
        if self.policy not in ("bc", "rl"):
            raise ValueError("policy must be bc or rl")
        if self.seed < 0 or self.steps <= 0:
            raise ValueError("invalid seed/steps")

        if self.done_reason not in _DONE_REASONS:
            raise ValueError("unknown canonical done_reason")
        if self.done_reason == "reached" and str(self.raw_done_reason).startswith(("collide(", "penetrate_flyover(")):
            raise ValueError("arrival and collision on the same transition is failure")
        if not self.raw_done_reason:
            raise ValueError("raw_done_reason is required")
        if self.done_reason in ("crash_ground", "out_of_bounds", "altitude_exceed", "invalid_state"):
            if self.raw_done_reason != self.done_reason:
                raise ValueError("boundary/ground failure requires its authoritative raw reason")
        if type(self.invalid_steps) is not int or self.invalid_steps not in (0, 1) or (self.done_reason == "invalid_state") != bool(self.invalid_steps):
            raise ValueError("invalid terminal and invalid-step count must agree")
        if self.onset_triggered != (self.onset_first_step is not None):
            raise ValueError("onset_triggered must match onset_first_step nullness")
        if self.onset_first_step is not None and not 1 <= self.onset_first_step <= self.steps:
            raise ValueError("onset_first_step must be within the episode")

        positions = (self.terminal_xyz, self.min_xyz, self.max_xyz)
        if any(len(value) != 3 for value in positions):
            raise ValueError("position summaries must be xyz triples")
        scalars = (
            self.episode_return,
            self.future_clearance_reward_sum,
            self.d_soft_h_min,
            self.action_dz_mean,
            self.action_dz_min,
            self.action_dz_max,
        )
        vector_scalars = tuple(value for vector in positions for value in vector)
        dz_values = (self.action_dz_mean, self.action_dz_min, self.action_dz_max)
        if any(value is None for value in (self.episode_return, self.future_clearance_reward_sum, self.d_soft_h_min) + vector_scalars):
            raise ValueError("required non-action statistics cannot be missing")
        if any(v is None for v in dz_values) and not (self.invalid_steps > 0 and all(v is None for v in dz_values)):
            raise ValueError("missing action statistics require an invalid episode")
        if not all(math.isfinite(float(value)) for value in scalars + vector_scalars if value is not None):
            raise ValueError("episode metrics must be finite")
        if any(low > high for low, high in zip(self.min_xyz, self.max_xyz)):
            raise ValueError("min_xyz cannot exceed max_xyz")
        if self.action_dz_mean is not None and not self.action_dz_min <= self.action_dz_mean <= self.action_dz_max:
            raise ValueError("action dz summary is inconsistent")
        if self.support_cells < 0 or self.d_soft_h_steps < 0:
            raise ValueError("support_cells and d_soft_h_steps must be non-negative")
        for name in ("trap_recovery_count", "loop_count", "side_flip_count_max"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError("{} must be non-negative".format(name))
        if self.goal_xyz is not None:
            if len(self.goal_xyz) != 3:
                raise ValueError("goal_xyz must be a three-dimensional point")
            delta = tuple(float(a) - float(b) for a, b in zip(self.terminal_xyz, self.goal_xyz))
            expected = (
                math.sqrt(sum(value * value for value in delta)),
                math.hypot(delta[0], delta[1]),
                abs(delta[2]),
            )
            observed = (self.final_distance_3d, self.final_error_xy, self.final_error_z)
            if any(value is None or not math.isclose(float(value), target, abs_tol=1e-6) for value, target in zip(observed, expected)):
                raise ValueError("reported endpoint error does not match goal/final xyz")
            safe_arrival = expected[0] <= 0.5 and not self.terminal_safety_flags
            if (self.done_reason == "reached") != safe_arrival:
                raise ValueError("reached must mean safe three-dimensional arrival within 0.5 m")

    @property
    def success(self):
        if self.goal_xyz is not None:
            return self.final_distance_3d <= 0.5 and not self.terminal_safety_flags
        return self.done_reason == "reached"

    @property
    def collision(self):
        return self.done_reason == "collision"

    @property
    def crash_ground(self):
        return self.done_reason == "crash_ground"

    @property
    def out_of_bounds(self):
        return self.done_reason == "out_of_bounds"

    @property
    def altitude_exceed(self):
        return self.done_reason == "altitude_exceed"

    @property
    def early_death(self):
        return self.done_reason in (
            "crash_ground",
            "out_of_bounds",
            "altitude_exceed",
            "invalid_state",
        )

    @property
    def time_limit(self):
        return self.done_reason == "time_limit"

    @property
    def environment_identity(self):
        return (
            self.task_id,
            self.config_id,
            self.config_hash,
            self.obstacle_hash,
            self.source_manifest_hash,
            self.world_id,
            self.split,
            self.role,
            self.seed,
            self.evaluator,
            self.reward_schema_version,
        )


class PairedEvaluationBank:
    def __init__(self, records):
        self.records = tuple(records)
        groups = {}
        for record in self.records:
            key = (record.config_id, record.seed)
            policies = groups.setdefault(key, {})
            if record.policy in policies:
                raise ValueError("duplicate policy evaluation record: {}".format(key))
            policies[record.policy] = record
        incomplete = [key for key, value in groups.items() if set(value) != {"bc", "rl"}]
        if incomplete:
            raise ValueError("unpaired evaluation records: {}".format(incomplete[:5]))
        pairs = []
        for _, value in sorted(groups.items()):
            bc, rl = value["bc"], value["rl"]
            if bc.environment_identity != rl.environment_identity:
                raise ValueError("paired environment identity mismatch")
            pairs.append((bc, rl))
        self.pairs = tuple(pairs)

    @property
    def content_hash(self):
        payload = [asdict(record) for record in self.records]
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def assert_not_training_source(self):
        return all(
            record.split in _SPLITS
            and (
                record.world_id not in _HISTORICAL_WORLDS
                or record.split == "historical_eval_only"
            )
            for record in self.records
        )


@dataclass(frozen=True)
class DeltaCI:
    mean: float
    lower: float
    upper: float
    pairs: int


@dataclass(frozen=True)
class EndpointJudgment:
    status: str
    start_additional_seed: bool
    start_gazebo_px4: bool
    reason: str


def bootstrap_delta(values, *, seed=20260828, samples=10000, alpha=0.05):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("paired bootstrap requires at least two finite scalar deltas")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return DeltaCI(
        mean=float(values.mean()),
        lower=float(np.quantile(means, alpha / 2)),
        upper=float(np.quantile(means, 1 - alpha / 2)),
        pairs=len(values),
    )


def paired_metrics(bank):
    pairs = bank.pairs
    deltas = {
        "success": [float(rl.success) - float(bc.success) for bc, rl in pairs],
        "collision": [float(rl.collision) - float(bc.collision) for bc, rl in pairs],
        "crash_ground": [
            float(rl.crash_ground) - float(bc.crash_ground) for bc, rl in pairs
        ],
        "out_of_bounds": [
            float(rl.out_of_bounds) - float(bc.out_of_bounds) for bc, rl in pairs
        ],
        "altitude_exceed": [
            float(rl.altitude_exceed) - float(bc.altitude_exceed) for bc, rl in pairs
        ],
        "early_death": [
            float(rl.early_death) - float(bc.early_death) for bc, rl in pairs
        ],
        "time_limit": [
            float(rl.time_limit) - float(bc.time_limit) for bc, rl in pairs
        ],
        "combined_failure": [
            float(rl.collision or rl.early_death)
            - float(bc.collision or bc.early_death)
            for bc, rl in pairs
        ],
        "return": [rl.episode_return - bc.episode_return for bc, rl in pairs],
        "onset_triggered": [
            float(rl.onset_triggered) - float(bc.onset_triggered) for bc, rl in pairs
        ],
        "future_clearance_reward": [
            rl.future_clearance_reward_sum - bc.future_clearance_reward_sum
            for bc, rl in pairs
        ],
        "support_cells": [rl.support_cells - bc.support_cells for bc, rl in pairs],
        "x_span": [
            (rl.max_xyz[0] - rl.min_xyz[0]) - (bc.max_xyz[0] - bc.min_xyz[0])
            for bc, rl in pairs
        ],
        "y_span": [
            (rl.max_xyz[1] - rl.min_xyz[1]) - (bc.max_xyz[1] - bc.min_xyz[1])
            for bc, rl in pairs
        ],
        "z_span": [
            (rl.max_xyz[2] - rl.min_xyz[2]) - (bc.max_xyz[2] - bc.min_xyz[2])
            for bc, rl in pairs
        ],
        "action_dz_mean": [rl.action_dz_mean - bc.action_dz_mean for bc, rl in pairs
                           if not (bc.invalid_steps or rl.invalid_steps)
                           and bc.action_dz_mean is not None and rl.action_dz_mean is not None],
        "trap_recovery_count": [
            rl.trap_recovery_count - bc.trap_recovery_count for bc, rl in pairs
        ],
        "loop_count": [rl.loop_count - bc.loop_count for bc, rl in pairs],
        "side_flip_count_max": [
            rl.side_flip_count_max - bc.side_flip_count_max for bc, rl in pairs
        ],
    }
    # Never drop invalid episodes from success/failure denominators. Only
    # incomplete trajectory summaries use a separate, explicit valid subset.
    valid_pairs = [(bc, rl) for bc, rl in pairs if not (bc.invalid_steps or rl.invalid_steps)]
    for dim, name in enumerate(("x_span", "y_span", "z_span")):
        deltas[name] = [(rl.max_xyz[dim] - rl.min_xyz[dim]) - (bc.max_xyz[dim] - bc.min_xyz[dim])
                        for bc, rl in valid_pairs]
    deltas["invalid_state"] = [float(rl.done_reason == "invalid_state") - float(bc.done_reason == "invalid_state") for bc, rl in pairs]
    endpoint_pairs = [(bc, rl) for bc, rl in pairs if bc.final_distance_3d is not None and rl.final_distance_3d is not None]
    for name in ("final_distance_3d", "final_error_xy", "final_error_z"):
        deltas[name] = [float(getattr(rl, name)) - float(getattr(bc, name)) for bc, rl in endpoint_pairs]
    result = {name: bootstrap_delta(value) for name, value in deltas.items() if len(value) >= 2}
    d_soft_pairs = [
        (bc, rl)
        for bc, rl in pairs
        if bc.d_soft_h_steps > 0 and rl.d_soft_h_steps > 0
    ]
    if len(d_soft_pairs) >= 2:
        result["d_soft_h_min"] = bootstrap_delta(
            [rl.d_soft_h_min - bc.d_soft_h_min for bc, rl in d_soft_pairs]
        )
    onset_pairs = [
        (bc, rl)
        for bc, rl in pairs
        if bc.onset_first_step is not None and rl.onset_first_step is not None
    ]
    if len(onset_pairs) >= 2:
        # Positive means RL triggered earlier on the same episode config.
        result["onset_earliness"] = bootstrap_delta(
            [bc.onset_first_step - rl.onset_first_step for bc, rl in onset_pairs]
        )
    for role in ("anticipatory", "dense_recovery", "support_expansion", "yaw_lateral_recovery"):
        # layerA's generic config role is not a validated Route A capability label.
        selected = [(bc, rl) for bc, rl in pairs if bc.role == role and bc.split != "layerA"]
        if len(selected) >= 2:
            result[role + "_success"] = bootstrap_delta(
                [float(rl.success) - float(bc.success) for bc, rl in selected]
            )
    return result


def judge_endpoint_effect(metrics):
    if (
        metrics["collision"].mean < 0.0
        and metrics["early_death"].mean > 0.0
        and metrics["success"].lower <= 0.0
    ):
        return EndpointJudgment(
            status="failure_mode_transfer",
            start_additional_seed=False,
            start_gazebo_px4=False,
            reason=(
                "collision decreased while early_death increased without a "
                "paired success improvement"
            ),
        )
    positive = (
        metrics["success"].lower > 0.0
        and metrics["combined_failure"].upper <= 0.0
        and metrics["early_death"].upper <= 0.0
    )
    if positive:
        return EndpointJudgment(
            status="positive_lightweight_effect",
            start_additional_seed=True,
            start_gazebo_px4=False,
            reason=(
                "paired success improved without increased combined failure or "
                "early death; independent formal seeds are next"
            ),
        )
    return EndpointJudgment(
        status="mixed_or_no_benefit",
        start_additional_seed=False,
        start_gazebo_px4=False,
        reason="endpoint evidence is mixed or does not establish net capability benefit",
    )


def _policy_aggregate(records):
    result = {}
    for policy in ("bc", "rl"):
        selected = [record for record in records if record.policy == policy]
        count = len(selected)
        reasons = Counter(record.done_reason for record in selected)
        raw_reasons = Counter(record.raw_done_reason for record in selected)
        rates = {
            name: reasons.get(name, 0) / count
            for name in (
                "reached",
                "collision",
                "crash_ground",
                "out_of_bounds",
                "altitude_exceed",
                "invalid_state",
                "time_limit",
            )
        }
        rates["early_death"] = sum(record.early_death for record in selected) / count
        result[policy] = {
            "episodes": count,
            "done_reason_counts": dict(sorted(reasons.items())),
            "raw_done_reason_counts": dict(sorted(raw_reasons.items())),
            "rates": rates,
            "return_mean": sum(record.episode_return for record in selected) / count,
            "steps_mean": sum(record.steps for record in selected) / count,
        }
    return result


def _role_aggregate(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[(record.role, record.policy)].append(record)
    result = {}
    for (role, policy), selected in sorted(grouped.items()):
        count = len(selected)
        result.setdefault(role, {})[policy] = {
            "episodes": count,
            "success_rate": sum(record.success for record in selected) / count,
            "invalid_state_rate": sum(record.done_reason == "invalid_state" for record in selected) / count,
            "collision_rate": sum(record.collision for record in selected) / count,
            "crash_ground_rate": sum(record.crash_ground for record in selected) / count,
            "out_of_bounds_rate": sum(record.out_of_bounds for record in selected) / count,
            "altitude_exceed_rate": sum(
                record.altitude_exceed for record in selected
            )
            / count,
            "early_death_rate": sum(record.early_death for record in selected) / count,
            "time_limit_rate": sum(record.time_limit for record in selected) / count,
            "return_mean": sum(record.episode_return for record in selected) / count,
        }
    return result


def _split_aggregate(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[(record.split, record.policy)].append(record)
    result = {}
    for (split, policy), selected in sorted(grouped.items()):
        count = len(selected)
        result.setdefault(split, {})[policy] = {
            "episodes": count,
            "success_rate": sum(record.success for record in selected) / count,
            "invalid_state_rate": sum(record.done_reason == "invalid_state" for record in selected) / count,
            "collision_rate": sum(record.collision for record in selected) / count,
            "crash_ground_rate": sum(record.crash_ground for record in selected) / count,
            "out_of_bounds_rate": sum(record.out_of_bounds for record in selected) / count,
            "altitude_exceed_rate": sum(record.altitude_exceed for record in selected)
            / count,
            "early_death_rate": sum(record.early_death for record in selected) / count,
            "time_limit_rate": sum(record.time_limit for record in selected) / count,
            "return_mean": sum(record.episode_return for record in selected) / count,
        }
    return result


def v17_behavior_summary(bank):
    """Descriptive high-altitude behavior; no training/termination threshold."""
    if not all(r.reward_schema_version == "v17" for r in bank.records):
        return None

    def summarize(records):
        above = [r for r in records if (r.v17_metrics or {}).get("above_demo_height", False)]
        dwell, unknown, stall = (sum((r.v17_metrics or {}).get(key, 0.) for r in records)
                                  for key in ("above_demo_dwell_seconds", "above_demo_unknown_seconds", "above_demo_stall_seconds"))
        known = dwell - unknown
        return dict(episodes=len(records), episodes_above_demo=len(above),
                    episodes_above_demo_rate=len(above) / len(records) if records else None,
                    outcomes_given_above_demo=dict(Counter(r.done_reason for r in above)),
                    reached_given_above_demo=sum(r.success for r in above) / len(above) if above else None,
                    above_demo_dwell_seconds=dwell, above_demo_unknown_seconds=unknown,
                    above_demo_stall_seconds=stall, above_demo_stall_rate=stall / known if known > 0 else None,
                    above_demo_coverage=known / dwell if dwell > 0 else None,
                    invalid_incomplete_episodes=sum(bool(r.invalid_steps) for r in records),
                    onset_history_steps=sum((r.v17_metrics or {}).get("onset_history_steps", 0) for r in records),
                    onset_contrast_steps=sum((r.v17_metrics or {}).get("onset_contrast_steps", 0) for r in records))

    grouped, goal_grouped = {}, {}
    for p in ("bc", "rl"):
        chosen = [r for r in bank.records if r.policy == p]
        grouped[p] = summarize(chosen)
        goal_grouped[p] = {label: summarize([r for r in chosen if (r.v17_metrics or {}).get("goal_above_demo_height") is value])
                           for label, value in (("above", True), ("not_above", False), ("unknown", None))}
    return dict(by_policy=grouped, by_goal_height=goal_grouped,
                onset_evidence="reward_activation_not_independent_event_timing",
                invalid_coverage="known prefix only; invalid suffix is missing, not zero flight",
                efficiency_attribution="descriptive only; no causal support attribution")


def build_endpoint_report(bank):
    if not bank.pairs:
        raise ValueError("endpoint report requires at least one BC/RL pair")
    if not bank.assert_not_training_source():
        raise ValueError("endpoint bank contains an invalid training/evaluation source")
    evaluators = {record.evaluator for record in bank.records}
    if evaluators != {"lightweight_kinematic"}:
        raise ValueError("formal endpoint report requires one lightweight evaluator scope")
    metrics = paired_metrics(bank)
    judgment = judge_endpoint_effect(metrics)
    splits = sorted({record.split for record in bank.records})
    routea_role_records = [record for record in bank.records if record.split != "layerA"]
    world_pairs = Counter((bc.split, bc.world_id) for bc, _ in bank.pairs)
    return {
        "audit_schema_version": 2 if any(r.reward_schema_version == "v17" for r in bank.records) else 1,
        "reward_schema_versions": sorted({r.reward_schema_version for r in bank.records}),
        "v17_behavior": v17_behavior_summary(bank),
        "altitude_termination_enabled": False if all(r.reward_schema_version == "v17" for r in bank.records) else None,
        "invalid_statistics": {p: {"count": sum(r.done_reason == "invalid_state" for r in bank.records if r.policy == p),
                                    "denominator": sum(r.policy == p for r in bank.records),
                                    "trajectory_complete_episodes": sum(not r.invalid_steps for r in bank.records if r.policy == p)}
                               for p in ("bc", "rl")},
        "bank_content_hash": bank.content_hash,
        "evidence_scope": "lightweight_kinematic",
        "gazebo_px4_evidence": False,
        "record_count": len(bank.records),
        "pair_count": len(bank.pairs),
        "world_count": len(world_pairs),
        "episode_pairs_per_world": {
            split: {
                world: count for (item_split, world), count in sorted(world_pairs.items())
                if item_split == split
            }
            for split in splits
        },
        "splits": splits,
        "source_manifest_hashes": sorted(
            {record.source_manifest_hash for record in bank.records}
        ),
        "policy_identity_hashes": {
            policy: sorted(
                {
                    record.policy_identity_hash
                    for record in bank.records
                    if record.policy == policy
                }
            )
            for policy in ("bc", "rl")
        },
        "metrics": {name: asdict(value) for name, value in sorted(metrics.items())},
        "judgment": asdict(judgment),
        "judgment_is_execution_gate": False,
        "interpretation_notes": {
            "primary_outcome": "safe goal arrival (reached); report obstacle/ground/bounds/invalid failures separately; altitude termination is deprecated in v17",
            "legacy_judgment": "retained for compatibility, descriptive only; it does not authorize or block training, new seeds, or deployment",
            "uncertainty": "existing paired bootstrap resamples episode pairs, not independent worlds or training seeds; 4 configurations in one world are not 4 independent worlds",
            "layerA_roles": "config role labels are retained but excluded from pooled Route A capability role metrics",
            "onset": "reward onset is a proxy; conditional earliness only includes pairs with both triggers, not a general avoidance capability verdict",
            "support": "v17 XYZ span includes the complete executed polyline; legacy records use decision endpoints. Span/support cells are descriptive, not proof of safe low-support navigation",
            "early_death": "evaluation aggregate includes ground, bounds, legacy altitude and invalid; training canonical early_death excludes ground collisions. Invalid always remains in the safe-arrival denominator",
        },
        "policy_aggregate": _policy_aggregate(bank.records),
        "role_aggregate": _role_aggregate(routea_role_records),
        "role_aggregate_scope": "non_layerA_config_roles",
        "role_aggregate_by_split": {
            split: _role_aggregate([record for record in bank.records if record.split == split])
            for split in splits
        },
        "split_aggregate": _split_aggregate(bank.records),
    }


def write_endpoint_artifacts(output_dir, bank):
    """Write one immutable evidence pack; never merge or overwrite runs."""

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "episodes": output / "paired_episodes.jsonl",
        "report": output / "endpoint_report.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite endpoint artifacts: {}".format(
                ", ".join(path.name for path in existing)
            )
        )

    episode_text = "".join(
        json.dumps(asdict(record), sort_keys=True, allow_nan=False) + "\n"
        for record in bank.records
    )
    report_text = json.dumps(
        build_endpoint_report(bank), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    temporary = {
        name: path.with_suffix(path.suffix + ".tmp") for name, path in paths.items()
    }
    if any(path.exists() for path in temporary.values()):
        raise FileExistsError("refusing to reuse stale endpoint temporary artifacts")
    temporary["episodes"].write_text(episode_text, encoding="utf-8")
    temporary["report"].write_text(report_text, encoding="utf-8")
    temporary["episodes"].replace(paths["episodes"])
    temporary["report"].replace(paths["report"])
    return paths
