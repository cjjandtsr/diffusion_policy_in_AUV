"""EpisodeOutcome — the SINGLE truth source for episode endings (R5.8-C).

The env computes one outcome per step (carried in ``info["outcome"]``);
reward, buffer, metrics, the episode trace and the paired evaluator all
consume it — no layer re-infers success/collision from distances,
velocities or reward values (N5/N9/F-2/F-3 closure).

Invariants (ruling §6.4, adapted):
  - success -> terminated and not collision and no early-death;
  - collision -> not success;
  - time_limit iff truncated;
  - done_reason is a FIXED enum — never a free-form diagnostic string.
"""

from dataclasses import asdict, dataclass

# Fixed done-reason enum (free-form subtype strings are forbidden —
# collision subtypes collapse into "collision").
DONE_REASON_ENUM = frozenset({
    "none", "reached", "collision", "crash_ground",
    "out_of_bounds", "altitude_exceed", "timeout",
})

# R6-E §11.6.4 (F-10): the MUTUALLY EXCLUSIVE terminal category — every
# episode ends in exactly one; a state that cannot be mapped uniquely
# hard-fails (no priority-silent reclassification).
TERMINAL_CATEGORY_ENUM = frozenset({
    "none", "reached", "collision", "reached_collision",
    "out_of_bounds", "altitude_exceed", "timeout",
    "other_terminated", "observation_fault", "rollout_cut",
})

# Non-env termination categories: never produced by the env itself
# (rollout_cut = loop-level budget cut, observation_fault = invalid
# frame void, other_terminated = external). They require a CLEAN
# non-terminated env state — co-occurrence with any env termination is
# a contradiction.
NON_ENV_CATEGORIES = frozenset({
    "observation_fault", "rollout_cut", "other_terminated",
})


def terminal_category(
    *,
    terminated: bool,
    truncated: bool,
    success: bool,
    collision: bool,
    done_reason: str,
    out_of_bounds: bool = False,
    altitude_exceed: bool = False,
    other: str = "none",
) -> str:
    """Map the raw termination signals to the ONE mutually exclusive
    terminal category (§11.6.4).

    Contradiction rules (ValueError — an unmappable state never gets a
    priority-silent category):
      - collision=True requires done_reason in {"collision",
        "crash_ground"};
      - success=True without collision requires done_reason == "reached"
        (a success+collision record carries done_reason "collision" —
        the env folds collision-wins, so reached_collision can only
        exist at this mapper layer);
      - terminated/truncated never co-occur with done_reason "none";
      - done_reason == "timeout" requires truncated (the env guarantees
        timeout ⇒ truncated);
      - other must be "none" or a NON_ENV_CATEGORIES member, and a
        non-"none" other requires a clean non-terminated env state.

    Mapping (unique by construction once the contradictions pass):
      other≠none → itself; success∧collision → reached_collision;
      collision → collision; success → reached; done_reason ∈
      {out_of_bounds, altitude_exceed, timeout} → itself; fully clean
      state → "none"; anything else → ValueError.
    """
    if collision and done_reason not in ("collision", "crash_ground"):
        raise ValueError(
            f"collision=True requires done_reason in ('collision', "
            f"'crash_ground'), got {done_reason!r}"
        )
    if success and not collision and done_reason != "reached":
        raise ValueError(
            f"success=True without collision requires done_reason "
            f"'reached', got {done_reason!r}"
        )
    if terminated and done_reason == "none":
        raise ValueError(
            "terminated=True with done_reason 'none' — an env "
            "termination must carry its reason"
        )
    if truncated and done_reason == "none":
        raise ValueError(
            "truncated=True with done_reason 'none' — a truncated "
            "episode must carry its reason"
        )
    if done_reason == "timeout" and not truncated:
        raise ValueError(
            "done_reason 'timeout' without truncated=True — the env "
            "guarantees timeout ⇒ truncated"
        )
    if other not in ("none",) and other not in NON_ENV_CATEGORIES:
        raise ValueError(
            f"other {other!r} not in NON_ENV_CATEGORIES "
            f"{sorted(NON_ENV_CATEGORIES)}"
        )
    if other != "none" and (terminated or truncated or done_reason != "none"):
        raise ValueError(
            f"other={other!r} requires a clean non-terminated env state "
            "(terminated=False, truncated=False, done_reason='none')"
        )

    if other != "none":
        return other
    if success and collision:
        return "reached_collision"
    if collision:
        return "collision"
    if success:
        return "reached"
    if done_reason in ("out_of_bounds", "altitude_exceed", "timeout"):
        return done_reason
    if not terminated and not truncated:
        return "none"
    raise ValueError(
        f"cannot uniquely map terminal state (terminated={terminated}, "
        f"truncated={truncated}, success={success}, collision={collision}, "
        f"done_reason={done_reason!r}) to one terminal_category — "
        "§11.6.4 hard-fail, no priority-silent reclassification"
    )


@dataclass(frozen=True)
class EpisodeOutcome:
    """One episode-end truth. ``time_limit`` is the env time limit
    (truncated); ``terminated``/``truncated`` mirror the Gymnasium
    flags of the SAME step (they may co-fire on the final step)."""

    success: bool
    collision: bool
    time_limit: bool
    out_of_bounds: bool
    altitude_exceed: bool
    terminated: bool
    truncated: bool
    done_reason: str

    def __post_init__(self) -> None:
        if self.done_reason not in DONE_REASON_ENUM:
            raise ValueError(
                f"done_reason {self.done_reason!r} not in the fixed "
                f"enum {sorted(DONE_REASON_ENUM)}"
            )
        if self.success and (self.collision or self.out_of_bounds
                             or self.altitude_exceed):
            raise ValueError(
                "success must never co-occur with collision/early-death "
                f"(done_reason={self.done_reason!r})"
            )
        if self.success and not self.terminated:
            raise ValueError("success implies terminated")
        if self.time_limit != self.truncated:
            raise ValueError("time_limit iff truncated")

    @staticmethod
    def ongoing() -> "EpisodeOutcome":
        """The per-step outcome while the episode is still running."""
        return EpisodeOutcome(
            success=False, collision=False, time_limit=False,
            out_of_bounds=False, altitude_exceed=False,
            terminated=False, truncated=False, done_reason="none",
        )

    @property
    def terminal_category(self) -> str:
        """The ONE mutually exclusive terminal category (§11.6.4).

        Delegates to the module-level mapper; the dataclass invariants
        forbid success∧collision so "reached_collision" can never be
        produced here — it exists only at the mapper layer (raw env
        signals) and in the evaluator re-derivation.
        """
        return terminal_category(
            terminated=self.terminated,
            truncated=self.truncated,
            success=self.success,
            collision=self.collision,
            done_reason=self.done_reason,
            out_of_bounds=self.out_of_bounds,
            altitude_exceed=self.altitude_exceed,
        )

    def to_dict(self) -> dict:
        """JSON-safe dict for info['outcome'] and downstream consumers."""
        return asdict(self)
