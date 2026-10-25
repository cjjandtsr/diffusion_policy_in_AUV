"""Authoritative Route A arrival, terminal precedence, and reset failure facts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Optional

import numpy as np


def _finite_xyz(value, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != (3,):
        raise ValueError("{} must have shape (3,)".format(name))
    if raw.dtype.kind in ("O", "S", "U", "V"):
        raise TypeError("{} must be numeric".format(name))
    result = raw.astype(np.float64, copy=False)
    if not np.isfinite(result).all():
        raise ValueError("{} must be finite".format(name))
    return result


def reached_goal(position_xyz, goal_xyz, radius_m: float = 0.5) -> bool:
    """Return finite 3D Euclidean arrival using a closed-radius boundary."""

    if isinstance(radius_m, bool):
        raise TypeError("radius_m must be a finite positive number")
    try:
        radius = float(radius_m)
    except (TypeError, ValueError) as exc:
        raise TypeError("radius_m must be a finite positive number") from exc
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius_m must be finite and positive")
    position = _finite_xyz(position_xyz, "position_xyz")
    goal = _finite_xyz(goal_xyz, "goal_xyz")
    return bool(np.linalg.norm(position - goal) <= radius)


@dataclass(frozen=True)
class TransitionFacts:
    collided: bool
    ground_crash: bool
    out_of_bounds: bool
    altitude_exceed: bool
    invalid_state: bool
    early_death: bool
    reached: bool
    time_limit: bool
    done_reason: str
    terminated: bool
    truncated: bool
    raw_done_reason: Optional[str] = None

    @property
    def physical_failure(self) -> bool:
        return bool(
            self.collided
            or self.ground_crash
            or self.out_of_bounds
            or self.altitude_exceed
            or self.invalid_state
            or self.early_death
        )

    @classmethod
    def resolve(
        cls,
        *,
        collided: bool = False,
        collision: Optional[bool] = None,
        ground_crash: bool = False,
        out_of_bounds: bool = False,
        altitude_exceed: bool = False,
        invalid_state: bool = False,
        early_death: bool = False,
        reached: bool = False,
        time_limit: bool = False,
        raw_done_reason: Optional[str] = None,
    ) -> "TransitionFacts":
        if collision is not None:
            if bool(collided) != bool(collision) and bool(collided):
                raise ValueError("collision and collided disagree")
            collided = bool(collision)
        raw = str(raw_done_reason) if raw_done_reason is not None else None
        flags = (
            ("invalid_state", bool(invalid_state)),
            ("collision", bool(collided)),
            ("ground_crash", bool(ground_crash)),
            ("out_of_bounds", bool(out_of_bounds)),
            ("altitude_exceed", bool(altitude_exceed)),
            ("early_death", bool(early_death)),
            ("reached", bool(reached)),
            ("time_limit", bool(time_limit)),
        )
        reason = next((name for name, active in flags if active), "none")
        failure = reason in {
            "invalid_state",
            "collision",
            "ground_crash",
            "out_of_bounds",
            "altitude_exceed",
            "early_death",
        }
        effective_reached = bool(reached) and not failure
        return cls(
            collided=bool(collided),
            ground_crash=bool(ground_crash),
            out_of_bounds=bool(out_of_bounds),
            altitude_exceed=bool(altitude_exceed),
            invalid_state=bool(invalid_state),
            early_death=bool(early_death),
            reached=effective_reached,
            time_limit=bool(time_limit),
            done_reason=reason,
            terminated=reason not in ("none", "time_limit"),
            truncated=reason == "time_limit",
            raw_done_reason=raw,
        )


def resolve_terminal(
    *,
    reached: bool,
    collision: bool,
    ground_crash: bool,
    out_of_bounds: bool,
    altitude_exceed: bool,
    invalid_state: bool,
    time_limit: bool,
    raw_done_reason: Optional[str] = None,
) -> TransitionFacts:
    """Resolve all terminal facts once using the reviewed precedence."""

    return TransitionFacts.resolve(
        collision=collision,
        ground_crash=ground_crash,
        out_of_bounds=out_of_bounds,
        altitude_exceed=altitude_exceed,
        invalid_state=invalid_state,
        reached=reached,
        time_limit=time_limit,
        raw_done_reason=raw_done_reason,
    )


class ResetExhausted(RuntimeError):
    """Raised when no physically valid reset can be committed."""

    def __init__(self, task_identity, role, attempts, rejection_histogram):
        self.task_identity = str(task_identity)
        self.role = str(role)
        self.attempts = int(attempts)
        self.rejection_histogram = {
            str(key): int(value)
            for key, value in sorted(dict(rejection_histogram).items())
        }
        super().__init__(
            "reset exhausted for task={} role={} after {} attempts; rejections={}".format(
                self.task_identity,
                self.role,
                self.attempts,
                self.rejection_histogram,
            )
        )


__all__ = (
    "ResetExhausted",
    "TransitionFacts",
    "reached_goal",
    "resolve_terminal",
)
