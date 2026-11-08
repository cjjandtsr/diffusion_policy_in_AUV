"""Pure contracts for Route A D0.5-B paired first-divergence diagnosis.

This module deliberately contains no model, environment, optimizer, or file
system operations.  It validates the registered reward schema, reconstructs
the exact S2 1,280-pair advantage normalization exposure, and aggregates only
at the task level.  The GPU-facing CLI is responsible for collecting facts
through the production environment and critic paths.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import math
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from diffusion_policy_3d.env.reward_v17 import TERM_NAMES


REWARD_OBJECTIVE_SUSPECTED = "REWARD_OBJECTIVE_SUSPECTED"
CREDIT_PROPAGATION_SUSPECTED = "CREDIT_PROPAGATION_SUSPECTED"
CONTEXT_GATING_SUSPECTED = "CONTEXT_GATING_SUSPECTED"
SIGNAL_OPPOSES_BAD_ACTION = "SIGNAL_OPPOSES_BAD_ACTION"
INCONCLUSIVE = "INCONCLUSIVE"

ALLOWED_D05_VERDICTS = frozenset(
    {
        REWARD_OBJECTIVE_SUSPECTED,
        CREDIT_PROPAGATION_SUSPECTED,
        CONTEXT_GATING_SUSPECTED,
        SIGNAL_OPPOSES_BAD_ACTION,
        INCONCLUSIVE,
    }
)

# These groups are the additive Reward-v1.7 groups frozen by design v1.4
# section 7.2.  ``core`` is R6; the other groups are its five deletions.
REWARD_GROUPS = {
    "core": (
        "terminal",
        "progress",
        "clearance",
        "ground",
        "overflight",
        "climb",
        "smoothness",
    ),
    "proactive": ("onset", "future_clearance"),
    "recovery": ("trap", "flip"),
    "support": ("support_escrow",),
    "free_pbrs": ("pbrs_free",),
    "guard": ("guard",),
}


def classify_endpoint_pair(bc_done_reason: str, rl_done_reason: str) -> str:
    """Classify an endpoint pair without substituting continuous heuristics."""

    bc_reached = str(bc_done_reason) == "reached"
    rl_reached = str(rl_done_reason) == "reached"
    if bc_reached and not rl_reached:
        return "break"
    if not bc_reached and rl_reached:
        return "rescue"
    return "stable_success" if bc_reached else "stable_failure"


def trace_window_indices(
    k_div: Optional[int], *, decision_count: int
) -> Tuple[int, ...]:
    """Return the registered inclusive [k_div-3, k_div+5] trace window."""

    count = int(decision_count)
    if count < 1:
        raise ValueError("decision_count must be positive")
    if k_div is None:
        return ()
    index = int(k_div)
    if index < 0 or index >= count:
        raise ValueError("k_div is outside the decision sequence")
    return tuple(range(max(0, index - 3), min(count - 1, index + 5) + 1))


def validate_reward_trace_row(
    row: Mapping[str, Any], *, tolerance: float = 1e-10
) -> Dict[str, Any]:
    """Validate exact Reward-v1.7 keys and its FP64 additive scalar contract."""

    terms = row.get("reward_terms")
    if not isinstance(terms, Mapping) or set(terms) != set(TERM_NAMES):
        raise ValueError("reward term keys must exactly match Reward v1.7 TERM_NAMES")
    numeric = {}
    for name in TERM_NAMES:
        value = float(terms[name])
        if not math.isfinite(value):
            raise ValueError("reward term contains a non-finite value: {}".format(name))
        numeric[name] = value
    reward = float(row["reward"])
    if not math.isfinite(reward):
        raise ValueError("reward must be finite")
    summed = float(np.sum(np.asarray([numeric[name] for name in TERM_NAMES], dtype=np.float64)))
    error = abs(summed - reward)
    if error > float(tolerance):
        raise ValueError(
            "reward sum error {} exceeds tolerance {}".format(error, tolerance)
        )
    return {
        "reward": reward,
        "reward_terms": numeric,
        "reward_sum_fp64": summed,
        "reward_sum_error_abs": error,
    }


def reward_group_totals(terms: Mapping[str, Any]) -> Dict[str, float]:
    """Collapse all 14 additive terms into the six frozen D2 reward groups."""

    validated = validate_reward_trace_row(
        {"reward": float(sum(float(value) for value in terms.values())), "reward_terms": terms}
    )["reward_terms"]
    groups = {
        group: float(np.sum(np.asarray([validated[name] for name in names], dtype=np.float64)))
        for group, names in REWARD_GROUPS.items()
    }
    if set(name for names in REWARD_GROUPS.values() for name in names) != set(TERM_NAMES):
        raise RuntimeError("registered reward groups do not partition TERM_NAMES")
    return groups


def _permutation_sha256(values: torch.Tensor) -> str:
    array = np.ascontiguousarray(values.detach().cpu().numpy())
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def s2_normalize_advantages(
    advantages: Sequence[float],
    transition_ids: Sequence[str],
    schedule_seed: int,
) -> Dict[str, Any]:
    """Rebuild formal S2 normalization for one immutable 1,024-transition buffer.

    S2 expands every transition over all ten denoising indices, permutes the
    10,240 unique pairs once, then computes ``torch.std`` statistics separately
    in eight 1,280-pair logical batches.  A transition therefore has ten
    schedule-specific normalized exposures rather than a fabricated single
    128-sample probe value.
    """

    if len(advantages) != 1024 or len(transition_ids) != 1024:
        raise ValueError("S2 diagnostic buffer requires exactly 1024 transitions")
    identities = tuple(str(value) for value in transition_ids)
    if len(set(identities)) != len(identities):
        raise ValueError("transition identities must be unique")
    values = torch.as_tensor(advantages, dtype=torch.float32, device="cpu")
    if values.ndim != 1 or not bool(torch.isfinite(values).all()):
        raise ValueError("advantages must be a finite rank-1 vector")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(schedule_seed))
    flat = torch.randperm(10240, generator=generator)
    by_transition = {identity: [] for identity in identities}
    per_k = Counter()
    seen_pairs = set()
    batch_audits = []
    for logical_batch_index, start in enumerate(range(0, 10240, 1280)):
        pair_ids = flat[start : start + 1280]
        transition_indices = torch.div(pair_ids, 10, rounding_mode="floor")
        denoising_indices = pair_ids.remainder(10)
        expanded = values[transition_indices]
        mean = expanded.mean()
        std = expanded.std()
        normalized = (expanded - mean) / (std + 1e-8)
        batch_audits.append(
            {
                "logical_batch_index": logical_batch_index,
                "pair_count": 1280,
                "advantage_mean": float(mean),
                "advantage_std": float(std),
                "pair_permutation_sha256": _permutation_sha256(pair_ids),
            }
        )
        for offset in range(1280):
            transition_index = int(transition_indices[offset])
            denoising_index = int(denoising_indices[offset])
            pair = (transition_index, denoising_index)
            seen_pairs.add(pair)
            per_k[denoising_index] += 1
            by_transition[identities[transition_index]].append(
                {
                    "denoising_index": denoising_index,
                    "logical_batch_index": logical_batch_index,
                    "normalized_advantage": float(normalized[offset]),
                }
            )
    for rows in by_transition.values():
        rows.sort(key=lambda row: row["denoising_index"])
    return {
        "by_transition": by_transition,
        "logical_batches": batch_audits,
        "audit": {
            "mode": "full_joint_accum_batch1280",
            "schedule_seed": int(schedule_seed),
            "transition_count": 1024,
            "denoising_count": 10,
            "pair_count": 10240,
            "logical_batch_pairs": 1280,
            "logical_batch_count": 8,
            "unique_pair_count": len(seen_pairs),
            "duplicate_pair_count": 10240 - len(seen_pairs),
            "missing_pair_count": 10240 - len(seen_pairs),
            "per_k_count": {str(index): per_k[index] for index in range(10)},
            "full_permutation_sha256": _permutation_sha256(flat),
        },
    }


def classify_transition_signal(
    reward: float, raw_advantage: float, normalized_advantage: float
) -> str:
    """Classify the sign-chain at one registered first-divergence decision."""

    values = tuple(float(value) for value in (reward, raw_advantage, normalized_advantage))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("signal values must be finite")
    immediate, raw, normalized = values
    if immediate > 0.0:
        return REWARD_OBJECTIVE_SUSPECTED
    if immediate < 0.0 and (raw > 0.0 or normalized > 0.0):
        return CREDIT_PROPAGATION_SUSPECTED
    if immediate < 0.0 and raw <= 0.0 and normalized <= 0.0:
        return SIGNAL_OPPOSES_BAD_ACTION
    return INCONCLUSIVE


def _collapse_task_mechanisms(rows: Iterable[Mapping[str, Any]]) -> Dict[str, str]:
    grouped = defaultdict(set)
    for row in rows:
        task_id = str(row.get("task_id") or "")
        mechanism = str(row.get("mechanism") or "")
        if not task_id:
            raise ValueError("task-level verdict input requires task_id")
        if mechanism not in ALLOWED_D05_VERDICTS:
            raise ValueError("unknown D0.5 mechanism: {}".format(mechanism))
        grouped[task_id].add(mechanism)
    collapsed = {}
    for task_id, values in grouped.items():
        determinate = values - {INCONCLUSIVE}
        collapsed[task_id] = (
            next(iter(determinate)) if len(determinate) == 1 else INCONCLUSIVE
        )
    return collapsed


def build_d05_verdict(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate repeat evidence once per independent task.

    Two tasks establish repeated support for a mechanism.  If two distinct
    mechanisms each cross that gate, D0.5-B remains inconclusive rather than
    promoting whichever has more repeated trajectories.
    """

    by_task = _collapse_task_mechanisms(rows)
    counts = Counter(value for value in by_task.values() if value != INCONCLUSIVE)
    determinate = sum(counts.values())
    repeated = sorted(name for name, count in counts.items() if count >= 2)
    if len(repeated) == 1:
        status = repeated[0]
        reason = "one_mechanism_has_repeated_task_support"
    elif len(repeated) > 1:
        status = INCONCLUSIVE
        reason = "multiple_mechanisms_have_repeated_task_support"
    else:
        status = INCONCLUSIVE
        reason = "no_mechanism_reaches_two_independent_tasks"
    denominator = determinate
    return {
        "schema": "routea_d05b_verdict_v1",
        "status": status,
        "reason": reason,
        "evidence_scope": "paired_first_divergence_correlation",
        "causal_claim_allowed": False,
        "task_count": len(by_task),
        "determinate_task_count": determinate,
        "indeterminate_task_count": len(by_task) - determinate,
        "mechanism_task_counts": dict(sorted(counts.items())),
        "coverage": {
            name: (float(count) / denominator if denominator else None)
            for name, count in sorted(counts.items())
        },
        "task_mechanisms": dict(sorted(by_task.items())),
        "next_allowed_action": (
            "design the mechanism-specific D2 local actor-credit test; do not train"
            if status in (REWARD_OBJECTIVE_SUSPECTED, CREDIT_PROPAGATION_SUSPECTED)
            else "inspect reward activation context and closed-loop state gating before training"
            if status == CONTEXT_GATING_SUSPECTED
            else "inspect action chunk, history/remaining-time, and closed-loop feedback before training"
            if status == SIGNAL_OPPOSES_BAD_ACTION
            else "do not escalate to D2 or training without additional determinate evidence"
        ),
    }


__all__ = (
    "ALLOWED_D05_VERDICTS",
    "CONTEXT_GATING_SUSPECTED",
    "CREDIT_PROPAGATION_SUSPECTED",
    "INCONCLUSIVE",
    "REWARD_GROUPS",
    "REWARD_OBJECTIVE_SUSPECTED",
    "SIGNAL_OPPOSES_BAD_ACTION",
    "build_d05_verdict",
    "classify_endpoint_pair",
    "classify_transition_signal",
    "reward_group_totals",
    "s2_normalize_advantages",
    "trace_window_indices",
    "validate_reward_trace_row",
)
