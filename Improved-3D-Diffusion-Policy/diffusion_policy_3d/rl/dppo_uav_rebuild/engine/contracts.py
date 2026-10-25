"""Immutable cross-module contracts for the canonical DDPM RL pipeline.

This module deliberately contains no PyTorch dependency. The contracts are
shared by CPU admission checks, the CUDA actor, and deployment tooling.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Any, Tuple


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _shape(value: Any) -> Tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value.shape)
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError("action value must expose an integer shape") from exc


def _values_equal(left: Any, right: Any) -> bool:
    """Compare NumPy or torch-like values without importing either package."""
    if _shape(left) != _shape(right):
        return False
    comparison = left == right
    if hasattr(comparison, "all"):
        comparison = comparison.all()
    if hasattr(comparison, "item"):
        comparison = comparison.item()
    return bool(comparison)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ObservationSpec:
    """Bridge contract between Route A observations and the native E5 actor."""

    environment_history: int
    policy_history: int
    state_dim: int
    point_count: int
    checkpoint_point_dim: int
    policy_point_dim: int
    use_pc_color: bool
    history_order: str
    feature_dim_per_frame: int

    def __post_init__(self) -> None:
        positive_fields = (
            "environment_history",
            "policy_history",
            "state_dim",
            "point_count",
            "checkpoint_point_dim",
            "policy_point_dim",
            "feature_dim_per_frame",
        )
        for name in positive_fields:
            if getattr(self, name) <= 0:
                raise ValueError("{} must be positive".format(name))
        if self.environment_history < self.policy_history:
            raise ValueError("environment_history must cover policy_history")
        if self.checkpoint_point_dim < self.policy_point_dim:
            raise ValueError("checkpoint_point_dim must cover policy_point_dim")
        if self.history_order != "chronological_oldest_to_latest":
            raise ValueError("history_order must be chronological_oldest_to_latest")
        if self.use_pc_color:
            raise ValueError("E5 identity requires use_pc_color=False")

    @classmethod
    def e5_route_a(cls) -> "ObservationSpec":
        return cls(
            environment_history=8,
            policy_history=3,
            state_dim=9,
            point_count=4096,
            checkpoint_point_dim=6,
            policy_point_dim=3,
            use_pc_color=False,
            history_order="chronological_oldest_to_latest",
            feature_dim_per_frame=192,
        )

    @property
    def policy_condition_dim(self) -> int:
        return self.policy_history * self.feature_dim_per_frame

    @property
    def legacy_environment_condition_dim(self) -> int:
        """Audit-only value; it is never an accepted E5 actor input."""
        return self.environment_history * self.feature_dim_per_frame

    @property
    def content_hash(self) -> str:
        return _canonical_sha256(asdict(self))


@dataclass(frozen=True)
class ActionSpec:
    """Names every action horizon so deployment and rollout cannot be mixed."""

    full_horizon: int
    action_start: int
    policy_horizon: int
    deployment_exec_horizon: int
    rollout_exec_horizon: int
    action_dim: int
    traj_dt: float

    def __post_init__(self) -> None:
        if self.full_horizon <= 0 or self.policy_horizon <= 0:
            raise ValueError("action horizons must be positive")
        if self.action_start < 0:
            raise ValueError("action_start must be non-negative")
        if self.action_start + self.policy_horizon > self.full_horizon:
            raise ValueError("policy slice exceeds full horizon")
        for name in ("deployment_exec_horizon", "rollout_exec_horizon"):
            value = getattr(self, name)
            if value <= 0 or value > self.policy_horizon:
                raise ValueError("{} must be within policy horizon".format(name))
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if self.traj_dt <= 0:
            raise ValueError("traj_dt must be positive")

    @classmethod
    def e5(cls) -> "ActionSpec":
        return cls(
            full_horizon=16,
            action_start=2,
            policy_horizon=8,
            deployment_exec_horizon=2,
            rollout_exec_horizon=4,
            action_dim=11,
            traj_dt=0.3,
        )

    def validate_full(self, full: Any) -> None:
        shape = _shape(full)
        expected_tail = (self.full_horizon, self.action_dim)
        if len(shape) < 2 or shape[-2:] != expected_tail:
            raise ValueError(
                "full action must end with shape {}; got {}".format(
                    expected_tail, shape
                )
            )

    def split(self, full: Any) -> Tuple[Any, Any, Any]:
        self.validate_full(full)
        stop = self.action_start + self.policy_horizon
        policy = full[..., self.action_start:stop, :]
        deployment = policy[..., : self.deployment_exec_horizon, :]
        rollout = policy[..., : self.rollout_exec_horizon, :]
        return policy, deployment, rollout

    @property
    def content_hash(self) -> str:
        return _canonical_sha256(asdict(self))


@dataclass(frozen=True)
class ActionBundle:
    """One authoritative tensor with validated, named protocol views."""

    spec: ActionSpec
    full: Any
    policy: Any
    deployment_executed: Any
    rollout_executed: Any

    def __post_init__(self) -> None:
        expected_policy, expected_deployment, expected_rollout = self.spec.split(
            self.full
        )
        checks = (
            (
                "policy action does not match authoritative full slice",
                self.policy,
                expected_policy,
            ),
            (
                "deployment action does not match authoritative policy prefix",
                self.deployment_executed,
                expected_deployment,
            ),
            (
                "rollout action does not match authoritative policy prefix",
                self.rollout_executed,
                expected_rollout,
            ),
        )
        for message, actual, expected in checks:
            if not _values_equal(actual, expected):
                raise ValueError(message)

    @classmethod
    def from_full(cls, full: Any, spec: ActionSpec = None) -> "ActionBundle":
        resolved = spec if spec is not None else ActionSpec.e5()
        policy, deployment, rollout = resolved.split(full)
        return cls(
            spec=resolved,
            full=full,
            policy=policy,
            deployment_executed=deployment,
            rollout_executed=rollout,
        )


@dataclass(frozen=True)
class RuntimeIdentity:
    """Hash-linked runtime identity stored in artifacts and checkpoints."""

    bc_identity_hash: str
    vendor_commit: str
    vendor_manifest_hash: str
    config_hash: str
    source_set_hash: str
    task_manifest_hash: str
    reward_identity_hash: str
    observation_contract_hash: str
    action_contract_hash: str
    caveats: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "bc_identity_hash",
            "vendor_manifest_hash",
            "config_hash",
            "source_set_hash",
            "task_manifest_hash",
            "reward_identity_hash",
            "observation_contract_hash",
            "action_contract_hash",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                raise ValueError("{} must be a lowercase SHA256".format(name))
        if not isinstance(self.vendor_commit, str) or _GIT_COMMIT_RE.fullmatch(
            self.vendor_commit
        ) is None:
            raise ValueError("vendor_commit must be a lowercase 40-character git hash")
        if not isinstance(self.caveats, tuple):
            raise TypeError("caveats must be an immutable tuple")
        if any(not isinstance(item, str) or not item.strip() for item in self.caveats):
            raise ValueError("caveats must contain non-empty strings")

    @property
    def content_hash(self) -> str:
        return _canonical_sha256(asdict(self))
