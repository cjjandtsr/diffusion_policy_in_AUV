#!/usr/bin/env python3
"""
routea_rl_calibration_stats.py — Route A RL v1 Calibration Statistics
======================================================================
Wilson 95% CI, percentile bootstrap, CI-aware admission gates,
and paired episode data structures.

All statistics are deterministic given seed.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Wilson 95% CI (for binary metrics: SR, CR)
# ─────────────────────────────────────────────────────────────────────────────


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson 95% confidence interval for binomial proportion.

    Args:
        k: number of successes
        n: number of trials
        z: z-score (1.96 for 95% CI)

    Returns:
        (ci_low, ci_high) — bounded to [0, 1].
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    half = z / denom * math.sqrt(
        (phat * (1.0 - phat) / n) + z * z / (4.0 * n * n)
    )
    return (max(0.0, center - half), min(1.0, center + half))


def wilson_ci_halfwidth(k: int, n: int, z: float = 1.96) -> float:
    """Half-width of the Wilson 95% CI."""
    lo, hi = wilson_ci(k, n, z)
    if math.isnan(lo):
        return float("nan")
    return (hi - lo) / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Percentile Bootstrap (for continuous metrics)
# ─────────────────────────────────────────────────────────────────────────────


def percentile_bootstrap_ci(
    values: List[float],
    n_boot: int = 2000,
    seed: int = 20260703,
) -> Tuple[float, float]:
    """Percentile bootstrap 95% CI for the mean of a list of values.

    Args:
        values: list of continuous metric values
        n_boot: number of bootstrap resamples
        seed: deterministic seed

    Returns:
        (ci_2_5, ci_97_5) — 95% CI bounds.
    """
    vals = [float(v) for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return (float("nan"), float("nan"))

    rng = random.Random(seed)
    n = len(vals)
    means = []
    for _ in range(n_boot):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = int(0.025 * (len(means) - 1))
    hi_idx = int(0.975 * (len(means) - 1))
    return (means[lo_idx], means[hi_idx])


def bootstrap_mean_ci(
    values: List[float],
    n_boot: int = 2000,
    seed: int = 20260703,
) -> Dict[str, float]:
    """Full bootstrap statistics: mean, CI, std."""
    vals = [float(v) for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return {
            "mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
            "std": float("nan"), "n": 0,
        }
    ci_lo, ci_hi = percentile_bootstrap_ci(vals, n_boot, seed)
    mean_val = sum(vals) / len(vals)
    variance = sum((v - mean_val) ** 2 for v in vals) / len(vals)
    return {
        "mean": round(mean_val, 6),
        "ci_low": round(ci_lo, 6),
        "ci_high": round(ci_hi, 6),
        "std": round(math.sqrt(variance), 6),
        "n": len(vals),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Paired episode data structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PairedEpisodeResult:
    """Result of one paired episode rollout."""
    world_uid: str
    role_validated: Optional[str]
    role_active: Optional[str]
    split: str
    episode_seed: int
    baseline_name: str          # "frozen_bc_fm" | "ego_or_expert" | "random_or_noop" | "feasibility_checker"
    success: bool
    collision: bool
    timeout: bool
    path_length: float
    onset_distance: Optional[float]
    future_clearance_softmin: Optional[float]
    trap_active: bool
    trap_recovered: bool
    deadend_dwell_steps: int
    successful_low_density_coverage: Optional[float]
    yaw_error_integral: Optional[float]
    failure_reason: str
    # Extended fields
    reward_total: float = 0.0
    n_steps: int = 0
    final_dist: float = 0.0
    final_clr: float = 0.0
    start_pos: Optional[List[float]] = None
    goal_pos: Optional[List[float]] = None
    # Per-step traces (optional)
    per_step_clearance: Optional[List[float]] = None
    per_step_positions: Optional[List[List[float]]] = None


@dataclass
class WorldCalibrationResult:
    """Per-world calibration summary after pass-1 and pass-2."""
    world_uid: str
    split: str
    role_generated: str
    role_validated: Optional[str]
    role_active: Optional[str]
    admission_status: str
    # Episode counts
    n_episodes_pass1: int
    n_episodes_final: int
    # BC/FM metrics
    bc_fm_sr: float
    bc_fm_sr_ci_low: float
    bc_fm_sr_ci_high: float
    bc_fm_cr: float
    bc_fm_cr_ci_low: float
    bc_fm_cr_ci_high: float
    bc_fm_path_length_mean: float
    bc_fm_future_clr_softmin_mean: float
    # EGO/expert metrics
    ego_sr: Optional[float]
    ego_sr_ci_low: Optional[float]
    ego_sr_ci_high: Optional[float]
    ego_available: bool
    ego_unavailable_reason: str
    # Random/noop baseline
    random_sr: Optional[float]
    random_cr: Optional[float]
    # Feasibility
    feasibility_reachable: bool
    feasibility_path_stretch: Optional[float]
    # Pass-2
    needs_pass2: bool
    pass2_trigger_reason: str
    # Admission
    beyond_boundary: bool
    stop_reason: str
    # Raw episodes
    episodes: List[Dict[str, Any]] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# CI-aware admission gates
# ─────────────────────────────────────────────────────────────────────────────


def compute_binary_metric_ci(
    episodes: List[PairedEpisodeResult],
    metric_fn,
    baseline_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute Wilson CI for a binary metric across episodes."""
    filtered = episodes
    if baseline_name:
        filtered = [e for e in episodes if e.baseline_name == baseline_name]

    k = sum(1 for e in filtered if metric_fn(e))
    n = len(filtered)
    ci_lo, ci_hi = wilson_ci(k, n)
    return {
        "rate": round(k / max(n, 1), 4) if n > 0 else float("nan"),
        "ci_low": round(ci_lo, 4),
        "ci_high": round(ci_hi, 4),
        "ci_halfwidth": round(wilson_ci_halfwidth(k, n), 4),
        "k": k,
        "n": n,
    }


def ci_aware_gate(
    sr_ci: Dict[str, Any],
    min_sr: float = 0.0,
    max_sr: float = 1.0,
    max_ci_halfwidth: float = 0.15,
) -> Tuple[str, str]:
    """Apply CI-aware gate to SR metric.

    Returns:
        (gate_status, reason)
        gate_status: "pass" | "informational" | "needs_more_eval" | "fail"
    """
    ci_hw = sr_ci.get("ci_halfwidth", float("nan"))
    sr = sr_ci.get("rate", float("nan"))
    ci_low = sr_ci.get("ci_low", float("nan"))
    ci_high = sr_ci.get("ci_high", float("nan"))

    if math.isnan(sr):
        return ("fail", "no_data")

    # CI too wide → needs more eval, don't hard pass
    if ci_hw > max_ci_halfwidth:
        return ("needs_more_eval", f"CI half-width {ci_hw:.4f} > {max_ci_halfwidth}")

    # Below lower bound
    if ci_high < min_sr:
        return ("fail", f"SR ci_high={ci_high:.4f} < min={min_sr}")

    # Above upper bound (too easy)
    if ci_low > max_sr:
        return ("fail", f"SR ci_low={ci_low:.4f} > max={max_sr} (too easy)")

    # In range but CI wide
    if ci_hw > max_ci_halfwidth * 0.67:
        return ("informational", f"CI half-width {ci_hw:.4f} in acceptable range but wide")

    return ("pass", f"SR {sr:.4f} CI [{ci_low:.4f}, {ci_high:.4f}] in [{min_sr}, {max_sr}]")


# ─────────────────────────────────────────────────────────────────────────────
# Admission rule logic
# ─────────────────────────────────────────────────────────────────────────────


ADMISSION_GATES = {
    "anticipatory": {
        "min_sr": 0.35,
        "max_sr": 0.85,
        "require_onset_insufficient": True,
        "require_ego_above_bc": True,
    },
    "dense_recovery": {
        "min_sr": 0.15,
        "max_sr": 0.70,
        "require_trap_dense_semantic": True,
    },
    "support_expansion": {
        "min_sr": 0.20,
        "max_sr": 0.70,
        "min_low_density_coverage": 0.05,
        "max_low_density_coverage": 0.40,
    },
    "yaw_lateral_recovery": {
        "min_sr": 0.20,
        "max_sr": 0.70,
        "require_nominal_high_disturb_drop": True,
    },
}


def determine_role_active(
    role_validated: str,
    bc_fm_metrics: Dict[str, Any],
    ego_metrics: Optional[Dict[str, Any]],
    semantic_metrics: Dict[str, Any],
) -> Tuple[Optional[str], str, bool]:
    """Determine role_active based on calibration metrics.

    Returns:
        (role_active, admission_status, beyond_boundary)
    """
    gate = ADMISSION_GATES.get(role_validated)
    if gate is None:
        return (None, "reject_unknown_role", False)

    sr_rate = bc_fm_metrics.get("sr_rate", 0.0)
    sr_ci_low = bc_fm_metrics.get("sr_ci_low", 0.0)
    sr_ci_high = bc_fm_metrics.get("sr_ci_high", 0.0)
    cr_rate = bc_fm_metrics.get("cr_rate", 0.0)
    ci_hw = bc_fm_metrics.get("sr_ci_halfwidth", 0.0)

    min_sr = gate["min_sr"]
    max_sr = gate["max_sr"]

    # Check CI width
    if ci_hw > 0.15:
        return (None, "needs_more_eval", False)

    # Beyond boundary: both BC and EGO too low
    if sr_ci_high < min_sr:
        ego_sr = ego_metrics.get("sr_rate", -1.0) if ego_metrics else -1.0
        if ego_sr < 0.20:
            return ("stress_only", "beyond_BC_EGO_boundary", True)

    # Too easy
    if sr_ci_low > max_sr:
        return (None, "reject_too_easy", False)

    # EGO check for anticipatory
    if gate.get("require_ego_above_bc") and ego_metrics:
        ego_sr = ego_metrics.get("sr_rate", -1.0)
        if ego_sr <= sr_rate:
            return (None, "ego_not_above_bc_fm", False)

    # In range → active
    admission = f"admitted_{role_validated}"
    return (role_validated, admission, False)


# ─────────────────────────────────────────────────────────────────────────────
# Mock / synthetic data generation (for smoke testing without Gazebo)
# ─────────────────────────────────────────────────────────────────────────────


def generate_synthetic_episodes(
    world_uid: str,
    role_validated: str,
    split: str,
    baseline_name: str,
    n_episodes: int,
    seed: int = 20260703,
    sr_target: float = 0.5,
    cr_target: float = 0.3,
) -> List[PairedEpisodeResult]:
    """Generate synthetic paired episodes for smoke testing."""
    rng = random.Random(seed + hash(world_uid) % 10000)
    episodes = []

    # Per-role parameter adjustments
    if role_validated == "anticipatory":
        sr_target = rng.uniform(0.4, 0.75)
        cr_target = rng.uniform(0.1, 0.3)
    elif role_validated == "dense_recovery":
        sr_target = rng.uniform(0.2, 0.55)
        cr_target = rng.uniform(0.2, 0.5)
    elif role_validated == "support_expansion":
        sr_target = rng.uniform(0.25, 0.6)
        cr_target = rng.uniform(0.15, 0.4)
    elif role_validated == "yaw_lateral_recovery":
        sr_target = rng.uniform(0.3, 0.65)
        cr_target = rng.uniform(0.15, 0.35)

    # EGO always better than BC
    if baseline_name == "ego_or_expert":
        sr_target = min(1.0, sr_target + rng.uniform(0.1, 0.3))
        cr_target = max(0.0, cr_target - rng.uniform(0.05, 0.15))

    # Random
    if baseline_name == "random_or_noop":
        sr_target = rng.uniform(0.0, 0.1)
        cr_target = rng.uniform(0.5, 0.9)

    for ep_i in range(n_episodes):
        ep_seed = seed + ep_i * 100 + hash(world_uid) % 10000
        rng_ep = random.Random(ep_seed)

        success = rng_ep.random() < sr_target
        if baseline_name == "random_or_noop":
            collision = rng_ep.random() < cr_target
        else:
            collision = rng_ep.random() < cr_target and not success
        timeout = not success and not collision

        failure_reason = (
            "none" if success
            else "collision" if collision
            else "timeout"
        )

        episodes.append(PairedEpisodeResult(
            world_uid=world_uid,
            role_validated=role_validated,
            role_active=None,
            split=split,
            episode_seed=ep_seed,
            baseline_name=baseline_name,
            success=success,
            collision=collision,
            timeout=timeout,
            path_length=rng_ep.uniform(16.0, 30.0),
            onset_distance=rng_ep.uniform(0.5, 3.0) if not success else None,
            future_clearance_softmin=rng_ep.uniform(0.3, 2.0),
            trap_active=rng_ep.random() < 0.3,
            trap_recovered=rng_ep.random() < 0.15,
            deadend_dwell_steps=int(rng_ep.uniform(0, 5)),
            successful_low_density_coverage=rng_ep.uniform(0.0, 0.5) if success else None,
            yaw_error_integral=rng_ep.uniform(0.0, 0.5),
            failure_reason=failure_reason,
            reward_total=rng_ep.uniform(-50, 100) if success else rng_ep.uniform(-150, -20),
            n_steps=int(rng_ep.uniform(20, 60)),
            final_dist=0.2 if success else rng_ep.uniform(2.0, 10.0),
            final_clr=1.5 if success else rng_ep.uniform(-0.5, 0.3),
        ))

    return episodes


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────


def self_test() -> bool:
    """Minimal self-test of statistical functions."""
    all_ok = True

    # Wilson CI
    lo, hi = wilson_ci(5, 10)  # 50% with 10 trials
    assert 0.2 < lo < 0.8, f"Wilson CI lo={lo} unexpected"
    assert 0.2 < hi < 0.8, f"Wilson CI hi={hi} unexpected"
    assert lo < 0.5 < hi, f"Wilson CI {lo}-{hi} does not contain 0.5"

    # Wilson CI edge cases
    lo0, hi0 = wilson_ci(0, 0)
    assert math.isnan(lo0), f"Wilson CI(0,0) should be NaN, got {lo0}"

    lo10, hi10 = wilson_ci(10, 10)  # 100%
    assert lo10 > 0.65, f"Wilson CI 100% lo={lo10} too low"
    assert hi10 == 1.0, f"Wilson CI 100% hi={hi10} != 1.0"

    # Bootstrap
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    ci_lo, ci_hi = percentile_bootstrap_ci(values, n_boot=500, seed=42)
    assert ci_lo < 3.0 < ci_hi, f"Bootstrap CI {ci_lo}-{ci_hi} does not contain mean 3.0"

    # Bootstrap edge case: empty
    lo_nan, hi_nan = percentile_bootstrap_ci([], n_boot=500)
    assert math.isnan(lo_nan), f"Empty bootstrap should be NaN"

    # Synthetic episodes
    eps = generate_synthetic_episodes(
        world_uid="test__anticipatory__rl_train__front_offset__000000",
        role_validated="anticipatory",
        split="rl_train",
        baseline_name="frozen_bc_fm",
        n_episodes=20,
    )
    assert len(eps) == 20, f"Expected 20 episodes, got {len(eps)}"

    # Binary CI on synthetic
    bc_sr_ci = compute_binary_metric_ci(eps, lambda e: e.success)
    assert 0.0 <= bc_sr_ci["rate"] <= 1.0, f"SR rate {bc_sr_ci['rate']} out of [0,1]"

    # CI gate
    status, reason = ci_aware_gate(bc_sr_ci, min_sr=0.35, max_sr=0.85)
    assert status in ("pass", "fail", "needs_more_eval", "informational"), f"Bad gate status: {status}"

    print("self_test: all checks passed ✓")
    return True


if __name__ == "__main__":
    self_test()
