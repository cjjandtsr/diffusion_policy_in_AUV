"""Immutable identity for the native four-step Phase1aPolicyFM checkpoint."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Tuple


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class FMIdentityError(ValueError):
    """Raised when a checkpoint would violate the admitted FM contract."""


def _require_sha256(value: str, field: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise FMIdentityError(f"{field} must be a lowercase SHA256")


def _canonical_sha256(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class FMCheckpointIdentity:
    """Checkpoint-derived policy, numerical, observation, and action identity."""

    checkpoint_sha256: str
    policy_target: str
    policy_source_sha256: str
    normalizer_sha256: str
    weight_source: str
    num_fm_steps: int
    sigma_min: float
    fm_t_scale: float
    cfg_lambda: float
    anchor_enabled: bool
    horizon_steps: int
    action_dim: int
    n_obs_steps: int
    n_action_steps: int
    execution_slice: Tuple[int, int]

    def __post_init__(self) -> None:
        _require_sha256(self.checkpoint_sha256, "checkpoint_sha256")
        _require_sha256(self.policy_source_sha256, "policy_source_sha256")
        _require_sha256(self.normalizer_sha256, "normalizer_sha256")
        if not isinstance(self.policy_target, str) or not self.policy_target.strip():
            raise FMIdentityError("policy_target must be non-empty")
        if self.weight_source != "model":
            raise FMIdentityError("weight_source must be 'model'")

        required = {
            "num_fm_steps": (self.num_fm_steps, 4),
            "sigma_min": (self.sigma_min, 0.005),
            "fm_t_scale": (self.fm_t_scale, 50.0),
            "cfg_lambda": (self.cfg_lambda, 1.5),
            "anchor_enabled": (self.anchor_enabled, False),
            "horizon_steps": (self.horizon_steps, 16),
            "action_dim": (self.action_dim, 11),
        }
        for field, (actual, expected) in required.items():
            if actual != expected:
                raise FMIdentityError(
                    f"{field} mismatch: expected {expected!r}, got {actual!r}"
                )
        if not isinstance(self.n_obs_steps, int) or self.n_obs_steps <= 0:
            raise FMIdentityError("n_obs_steps must be a positive runtime integer")
        if not isinstance(self.n_action_steps, int) or self.n_action_steps <= 0:
            raise FMIdentityError("n_action_steps must be a positive runtime integer")
        expected_slice = (
            self.n_obs_steps - 1,
            self.n_obs_steps - 1 + self.n_action_steps,
        )
        if tuple(self.execution_slice) != expected_slice:
            raise FMIdentityError(
                f"execution_slice mismatch: expected {expected_slice}, "
                f"got {self.execution_slice}"
            )
        if expected_slice[1] > self.horizon_steps:
            raise FMIdentityError(
                "execution_slice exceeds the full trajectory horizon"
            )

    @property
    def content_hash(self) -> str:
        return _canonical_sha256(asdict(self))


__all__ = ["FMCheckpointIdentity", "FMIdentityError"]
