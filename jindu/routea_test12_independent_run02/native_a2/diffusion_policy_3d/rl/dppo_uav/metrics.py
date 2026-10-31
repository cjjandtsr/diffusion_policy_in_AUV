"""Learning-signal recorders (design §4.15) as pure functions.

§4.15 requires every update to record:
  actor_grad_norm, critic_grad_norm, parameter_update_norm,
  per-denoise approx_kl, per-denoise clipfrac,
  ratio mean/std/quantiles, action variance by dimension,
  policy-action diversity, success-conditioned advantage,
  collision-conditioned advantage, role-conditioned advantage,
  executed-vs-unexecuted gradient norm.

The per-update dict assembly lives in trainer.py; this module owns the
pure numeric helpers so they are individually testable (RTM EVAL-AUDIT
companion: explained variance per §4.13, not a capability substitute).

R6-C (§11.4.1): statistics whose denominator may legitimately be empty
(e.g. no success terminal this batch) migrate to the OptionalStatistic
two-state contract — NEVER a NaN float that would need reader-side
sentinel handling:

  sample_count > 0  -> value finite float, valid=True,  reason=None
  sample_count == 0 -> value None,       valid=False, reason=frozen enum

Forbidden third states: count>0 with value None/NaN, or count==0 with
a numeric value. Per-denoise KL/clipfrac are NOT optional statistics —
every fine-tune denoise index must carry samples every update.
"""

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
    ROLE_NAMES,
    RoleSampler,
)

# ───────────────────────────────────────────────────────────────────
# R6-C §11.4.1 统一可选统计结构
# ───────────────────────────────────────────────────────────────────

# Frozen invalid-reason enum (§11.4.1) — free-form reasons are rejected.
OPTIONAL_INVALID_REASONS = frozenset({
    "no_success_terminal",
    "no_collision_terminal",
    "constant_value_target",
    "zero_executed_gradient_denominator",
    "capability_opportunity_absent",
})


def optional_statistic(
    values: Sequence[float], invalid_reason: str,
) -> Dict[str, Any]:
    """The §11.4.1 two-state contract.

    count>0  -> finite mean, valid=True,  reason=None;
    count==0 -> value None,  valid=False, reason=``invalid_reason``
    (must be a frozen-enum member). A non-finite mean is a programming
    error and raises — the OptionalStatistic must never carry NaN out
    of this module.
    """
    if invalid_reason not in OPTIONAL_INVALID_REASONS:
        raise ValueError(
            f"invalid_reason {invalid_reason!r} not in the frozen enum "
            f"{sorted(OPTIONAL_INVALID_REASONS)}"
        )
    n = len(values)
    if n == 0:
        return {
            "sample_count": 0, "value": None, "valid": False,
            "invalid_reason": invalid_reason,
        }
    mean = float(np.mean(np.asarray(values, dtype=np.float64)))
    if not math.isfinite(mean):
        raise ValueError(
            f"optional_statistic with {n} samples produced non-finite "
            f"mean {mean} — the two-state contract forbids it"
        )
    return {
        "sample_count": n, "value": mean, "valid": True,
        "invalid_reason": None,
    }


def role_stats_report(
    attempted: Dict[str, int],
    completed: Dict[str, int],
    voided: Dict[str, int],
    transitions: Dict[str, int],
) -> Dict[str, Dict[str, object]]:
    """The truthful per-role batch statistics (7 fields).

    - attempted: episodes that passed reset (completed + voided);
    - completed: episodes that reached a terminal state without fault;
    - voided: episodes ended by an observation fault — NEVER completed;
    - transitions: transitions the role contributed to the buffer.

    Proportions come from ``RoleSampler.proportion_report`` (the single
    implementation); an all-zero input yields the all-zero map instead of
    raising. ``attempted_episode_proportions`` is also the role-proportion
    metric report.
    """
    zeros = {r: 0 for r in ROLE_NAMES}
    return {
        "attempted_episode_counts": dict(attempted),
        "completed_episode_counts": dict(completed),
        "voided_episode_counts": dict(voided),
        "transition_counts": dict(transitions),
        "attempted_episode_proportions": (
            RoleSampler.proportion_report(attempted)
            if sum(attempted.values()) > 0 else dict(zeros)
        ),
        "completed_episode_proportions": (
            RoleSampler.proportion_report(completed)
            if sum(completed.values()) > 0 else dict(zeros)
        ),
        "transition_proportions": (
            RoleSampler.proportion_report(transitions)
            if sum(transitions.values()) > 0 else dict(zeros)
        ),
    }


def tensor_mean_std(t: torch.Tensor) -> Dict[str, float]:
    """mean/std of a 1-D tensor; empty input raises ValueError."""
    if t.numel() == 0:
        raise ValueError("tensor_mean_std on empty tensor")
    return {"mean": float(t.mean().item()), "std": float(t.std().item())}


def ratio_statistics(ratio: torch.Tensor) -> Dict[str, float]:
    """§4.15 ratio mean/std/quantiles (q05/q50/q95)."""
    assert ratio.dim() == 1, f"ratio must be 1-D, got {ratio.shape}"
    r = ratio.detach().cpu()
    return {
        "mean": float(r.mean().item()),
        # correction=0 matches numpy np.std (ddof=0) — the declared
        # "matches numpy" contract of this module
        "std": float(r.std(correction=0).item()),
        "q05": float(torch.quantile(r, 0.05).item()),
        "q50": float(torch.quantile(r, 0.50).item()),
        "q95": float(torch.quantile(r, 0.95).item()),
    }


def success_collision_role_conditioned_advantages(
    advantages: torch.Tensor,  # (N,) per-transition advantages
    terminated: np.ndarray,  # (N,) bool
    success: np.ndarray,  # (N,) bool — env EpisodeOutcome.success
    collision: np.ndarray,  # (N,) bool — env EpisodeOutcome.collision
    roles: List[str],  # (N,) episode role labels
) -> Dict[str, object]:
    """§4.15 success/collision/role-conditioned advantage means.

    success/collision come from the env's EpisodeOutcome (R5.8-C N9) —
    NEVER re-inferred from reward thresholds (a reward-number guess
    broke under reach+collision co-fire and reward-scale changes).
    Only the TERMINAL transition of an episode carries True.

    R6-C (§11.4.1): empty categories yield the OptionalStatistic null
    (sample_count=0/value=None/valid=False + frozen reason) — never a
    NaN float.
    """
    adv_np = advantages.detach().cpu().numpy()
    term = np.asarray(terminated, dtype=bool)
    success_mask = term & np.asarray(success, dtype=bool)
    collision_mask = term & np.asarray(collision, dtype=bool)
    succ = optional_statistic(
        adv_np[success_mask], "no_success_terminal")
    coll = optional_statistic(
        adv_np[collision_mask], "no_collision_terminal")
    roles_np = np.asarray(roles)
    role_adv: Dict[str, float] = {}
    for role in np.unique(roles_np):
        mask = roles_np == role
        role_adv[str(role)] = float(adv_np[mask].mean())
    return {
        "success_conditioned_advantage": succ,
        "collision_conditioned_advantage": coll,
        "role_conditioned_advantage": role_adv,
    }


def explained_variance(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """§4.13/§4.15 explained variance, official semantics
    (train_ppo_diffusion_agent.py:386-390): undefined when
    var(y_true)==0.

    R6-C (§11.4.1): returns an OptionalStatistic-bearing dict —
    ``{"target_variance": float, **two-state statistic}``. A constant
    value target is the legitimate no-data case: the statistic is null
    with ``constant_value_target`` and the raw variance is preserved
    (never a NaN float, never a fake 0). ``sample_count`` counts the
    samples for which the statistic is computable (0 when the target
    is constant).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    var_y = float(np.var(y_true))
    if var_y == 0:
        return {
            "target_variance": 0.0,
            **optional_statistic([], "constant_value_target"),
        }
    # EV is an aggregate statistic (no per-sample decomposition), so the
    # two-state contract is assembled directly — sample_count is the
    # number of y_true samples the statistic is computed over.
    return {
        "target_variance": var_y,
        "sample_count": int(len(y_true)),
        "value": 1.0 - float(np.var(y_true - y_pred)) / var_y,
        "valid": True,
        "invalid_reason": None,
    }


def executed_unexecuted_ratio_statistic(
    executed_mean: float, unexecuted_mean: float,
) -> Dict[str, Any]:
    """§11.4.1 executed/unexecuted gradient ratio as an
    OptionalStatistic.

    With the F10 ghost-action mask intact the unexecuted gradient is
    IDENTICALLY 0 (R5.8-F H-2) — the old implementation manufactured a
    ~1e8 pseudo-signal; the NaN-as-no-data sentinel convention is
    replaced by the two-state contract:
    ``zero_executed_gradient_denominator`` when denominator <= 1e-8.
    """
    if unexecuted_mean <= 1e-8:
        return optional_statistic(
            [], "zero_executed_gradient_denominator")
    return optional_statistic(
        [executed_mean / unexecuted_mean], "zero_executed_gradient_denominator")


def per_denoise_mean_dict(
    per_denoise: Dict[int, List[float]], name: str,
) -> Dict[int, float]:
    """§11.4.1 per-denoise means — NOT optional statistics.

    Formal training fixes ``ft`` fine-tune denoise indices; every index
    must carry samples every update. An index with no samples (or a
    non-finite mean) is an ENGINE hard-fail, never a silent NaN.
    """
    out: Dict[int, float] = {}
    for k, v in per_denoise.items():
        if not v:
            raise RuntimeError(
                f"{name}: denoise index {k} has no samples — every "
                "fine-tune index must carry samples per update "
                "(R6-C §11.4.1, ENGINE_HARD_GATE)"
            )
        mean = float(np.mean(np.asarray(v, dtype=np.float64)))
        if not math.isfinite(mean):
            raise RuntimeError(
                f"{name}: denoise index {k} mean {mean} is not finite "
                "(R6-C §11.4.1, ENGINE_HARD_GATE)"
            )
        out[k] = mean
    return out


# ───────────────────────────────────────────────────────────────────
# R6-C §11.4.2 outcome 计数与一致性
# ───────────────────────────────────────────────────────────────────


def compute_outcome_counts(
    *,
    terminated: np.ndarray,
    truncated: np.ndarray,
    done_reasons: Sequence[str],
    successes: np.ndarray,
    collisions: np.ndarray,
    roles: Sequence[str],
    rollout_cut: np.ndarray,
) -> Dict[str, Any]:
    """§11.4.2 per-batch recomputable outcome counts.

    Terminal = env ``terminated`` OR (``truncated`` AND NOT
    ``rollout_cut``): a rollout cut flips truncated=True purely as the
    GAE bootstrap (train_loop R5.3) and never counts as a completed
    episode; the loop-level truncation at max_ep_len (truncated=True,
    done_reason="none", not a cut) DOES complete an episode. Voided
    (fault) episodes never enter the buffer, so they are excluded by
    construction — they are counted separately by train_loop's
    ``voided_episodes``.
    """
    term = np.asarray(terminated, dtype=bool)
    trunc = np.asarray(truncated, dtype=bool)
    cut = np.asarray(rollout_cut, dtype=bool)
    success = np.asarray(successes, dtype=bool)
    collision = np.asarray(collisions, dtype=bool)
    reasons = np.asarray(list(done_reasons))
    roles_arr = np.asarray(list(roles))
    n = len(term)
    terminal_mask = term | (trunc & ~cut)
    # Per-role maps are built from the roles ACTUALLY present (same
    # convention as the role-conditioned advantages) — the invariant
    # check re-derives the sum, never the preset universe.
    per_role_trans: Dict[str, int] = {}
    per_role_completed: Dict[str, int] = {}
    for r in roles_arr:
        per_role_trans[str(r)] = per_role_trans.get(str(r), 0) + 1
    for r in roles_arr[terminal_mask]:
        per_role_completed[str(r)] = per_role_completed.get(str(r), 0) + 1
    return {
        "transition_count_total": n,
        "terminal_transition_count": int(terminal_mask.sum()),
        "success_event_count": int(success.sum()),
        "collision_event_count": int(collision.sum()),
        "timeout_terminal_count": int(np.count_nonzero(
            reasons == "timeout")),
        "oob_terminal_count": int(np.count_nonzero(
            reasons == "out_of_bounds")),
        "altitude_terminal_count": int(np.count_nonzero(
            reasons == "altitude_exceed")),
        "rollout_cut_count": int(cut.sum()),
        "per_role_transition_count": per_role_trans,
        "per_role_completed_episode_count": per_role_completed,
    }


def verify_outcome_invariants(
    counts: Dict[str, Any],
    *,
    episode_ids: Sequence[str],
    terminated: np.ndarray,
    truncated: np.ndarray,
    rollout_cut: np.ndarray,
) -> None:
    """§11.4.2 recomputable buffer invariants — ENGINE hard-fail.

    - each episode id has at most ONE terminal transition (a second
      terminal in the same episode is a double-counted completion);
    - a rollout_cut transition must be truncated=True and terminated=
      False (cut = GAE bootstrap, episode stays ACTIVE);
    - role transition counts sum to the buffer total and completed
      episode counts sum to the terminal count (defensive re-check of
      compute_outcome_counts).

    Raises RuntimeError naming the offending episode ids.
    """
    term = np.asarray(terminated, dtype=bool)
    trunc = np.asarray(truncated, dtype=bool)
    cut = np.asarray(rollout_cut, dtype=bool)
    terminal_mask = term | (trunc & ~cut)
    ids = np.asarray(list(episode_ids))
    if len(ids) != len(term):
        raise RuntimeError(
            "outcome invariant: episode_ids length "
            f"{len(ids)} != transitions {len(term)}"
        )
    seen_terminal: Dict[str, int] = {}
    for ep, is_term in zip(ids, terminal_mask):
        if is_term:
            seen_terminal[str(ep)] = seen_terminal.get(str(ep), 0) + 1
    dup = [ep for ep, c in seen_terminal.items() if c > 1]
    if dup:
        raise RuntimeError(
            "outcome invariant violated: episodes with more than one "
            f"terminal transition: {sorted(dup)} (double-counted "
            "completion — §11.4.2)"
        )
    bad_cut = np.flatnonzero(cut & (~trunc | term))
    if bad_cut.size:
        raise RuntimeError(
            "outcome invariant violated: rollout_cut transitions must "
            "be truncated=True and terminated=False, but indices "
            f"{bad_cut.tolist()} are not (§11.4.2)"
        )
    if sum(counts["per_role_transition_count"].values()) != counts[
            "transition_count_total"]:
        raise RuntimeError(
            "outcome invariant violated: per-role transition counts "
            f"{counts['per_role_transition_count']} do not sum to "
            f"total {counts['transition_count_total']}"
        )
    if sum(counts["per_role_completed_episode_count"].values()) != counts[
            "terminal_transition_count"]:
        raise RuntimeError(
            "outcome invariant violated: per-role completed episodes "
            "do not sum to terminal_transition_count "
            f"{counts['terminal_transition_count']}"
        )


# Thresholds for the §11.4.2 terminal-reward cross check. They encode
# the PHYSICS of the active reward v1.6, not an outcome re-inference:
#   r = w_progress * Δdist + r_smooth + terminal_bonus
# with r_reach=+100 and r_collision=-20 (reward_v16.py:37-39). A single
# step's shaping is bounded by w_progress * max_speed * dt ≪ 50, so a
# success terminal's reward must exceed reach_reward/2 and a collision
# terminal's reward must be negative. A violation means the reward the
# env handed back carries no terminal bonus/penalty at all — the
# learning signal is decoupled from the outcome truth.
_REACH_BONUS_FRACTION = 0.5
_COLLISION_PENALTY_CEIL = 0.0


def cross_check_terminal_rewards(
    *,
    rewards: np.ndarray,
    terminated: np.ndarray,
    successes: np.ndarray,
    collisions: np.ndarray,
    episode_ids: Sequence[str],
    reach_reward: float,
    collision_penalty: float = -20.0,
) -> None:
    """§11.4.2 invariant 4 — reward side is only a CROSS CHECK, never
    the success/collision definition (EpisodeOutcome is the truth).

    Forward direction: success terminal must carry the reach bonus
    (>= reach_reward/2); collision terminal must carry the collision
    penalty (<= 0).

    Reverse direction (Gate B R6-METRIC-1 REWORK, 2026-08-19 — F-03
    fix requirement 3): a terminal reward carrying the reach bonus
    (>= reach_reward/2) or the collision penalty
    (<= collision_penalty/2) MUST have the matching outcome flag —
    a terminal +100 with success=False/collision=False is the R5.9-C1
    info-forwarding break signature and hard-fails with episode ids.
    The collision threshold binds to the penalty magnitude (default
    -20.0, the v16 r_collision calibrated value), so timeout-terminal
    micro-negative rewards (r_smooth floor ~-0.2) never misfire.

    Any mismatch hard-fails and reports the offending episode ids.
    """
    term = np.asarray(terminated, dtype=bool)
    r = np.asarray(rewards, dtype=np.float64)
    succ = np.asarray(successes, dtype=bool) & term
    coll = np.asarray(collisions, dtype=bool) & term
    ids = np.asarray(list(episode_ids))
    problems: List[str] = []
    for idx in np.flatnonzero(succ):
        if r[idx] < _REACH_BONUS_FRACTION * float(reach_reward):
            problems.append(
                f"episode {ids[idx]}: success terminal reward "
                f"{r[idx]:.3f} < reach_reward * "
                f"{_REACH_BONUS_FRACTION} ({float(reach_reward) * _REACH_BONUS_FRACTION})"
                " — the reach bonus is missing from the reward the env "
                "returned (R6-C §11.4.2)"
            )
    for idx in np.flatnonzero(coll):
        if r[idx] > _COLLISION_PENALTY_CEIL:
            problems.append(
                f"episode {ids[idx]}: collision terminal reward "
                f"{r[idx]:.3f} > {_COLLISION_PENALTY_CEIL} — the collision "
                "penalty is missing from the reward the env returned "
                "(R6-C §11.4.2)"
            )
    for idx in np.flatnonzero(term):
        if r[idx] >= _REACH_BONUS_FRACTION * float(reach_reward) \
                and not (succ[idx] or coll[idx]):
            problems.append(
                f"episode {ids[idx]}: terminal reward {r[idx]:.3f} "
                "carries the reach bonus but no success/collision "
                "outcome is recorded — the outcome flags disagree "
                "with the reward (F-03 reverse cross-check, R5.9-C1 "
                "info-forwarding signature)"
            )
        if r[idx] <= collision_penalty / 2.0 and not coll[idx]:
            problems.append(
                f"episode {ids[idx]}: terminal reward {r[idx]:.3f} "
                "carries the collision penalty but no collision "
                "outcome is recorded (F-03 reverse cross-check)"
            )
    if problems:
        raise RuntimeError("terminal-reward cross check FAILED:\n  "
                           + "\n  ".join(problems))
