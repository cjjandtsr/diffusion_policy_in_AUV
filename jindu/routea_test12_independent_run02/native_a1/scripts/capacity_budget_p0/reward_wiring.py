#!/usr/bin/env python3
"""reward_wiring.py — P0 reward-wiring audit + role-distribution audit.

Two independent audits:

1. Reward wiring audit (run_reward_wiring_audit):
   Drive the real reward_v13 on fixed legal samples that should trigger each
   formal reward term (onset, future_clearance, trap, support low-density).
   Verify: each term's eligibility fires (eligible>0, active>0), no NaN/Inf
   values appear, and unsafe support reward == 0.

2. Role distribution audit (run_role_distribution_audit):
   Count per-role transition proportions from the B0 replay.pt and compute
   expected online-phase proportions given the C2 (formal) collection logic
   (role_quota_pilot=True, round-robin after 2-per-role minima).
"""

import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from scripts.capacity_budget_p0 import SCHEMA_VERSION, GATE_PASS_KEY

# ── Terminology ────────────────────────────────────────────────────────────────

FORMAL_ROLES = ("anticipatory", "dense_recovery", "support_expansion")

# Default reward config (copied from REWARD_V13_DEFAULTS for reference only;
# the real defaults live in reward_v13.py and are used by compute_reward_v13)
H_FUTURE = 8
ACTION_DIM = 11

# B0 replay path
B0_REPLAY_PATH = (
    "/home/cjj/AUV/program/shiyan"
    "/results_routeA_RL_b0b1_rerun_v13_20260710"
    "/b0_collect_rolefix/replay.pt"
)


# ═══════════════════════════════════════════════════════════════════════════════
# §A  Pure gate functions
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_reward_wiring_gate(
    term_stats: Dict[str, Dict[str, int]],
    unsafe_support_reward_zero: bool,
) -> Dict[str, Any]:
    """Pure gate evaluation for reward wiring.

    Args:
        term_stats: Dict mapping term name to {"eligible_steps": int,
                     "active_steps": int, "nonfinite_steps": int}.
                     Must contain all 4 formal terms: onset, future_clearance,
                     trap, support.
        unsafe_support_reward_zero: True if unsafe support reward == 0.

    Returns:
        dict with schema_version, run_id (None), gate_pass (bool),
        fail_reasons (list), term_details (dict).
    """
    fail_reasons: List[str] = []
    formal_terms = ["onset", "future_clearance", "trap", "support"]
    term_details: Dict[str, Dict[str, Any]] = {}
    all_ok = True

    for term in formal_terms:
        if term not in term_stats:
            fail_reasons.append(f"Missing term '{term}' in term_stats")
            all_ok = False
            term_details[term] = {"missing": True}
            continue

        stats = term_stats[term]
        eligible = stats.get("eligible_steps", 0)
        active = stats.get("active_steps", 0)
        nonfinite = stats.get("nonfinite_steps", 0)

        term_detail: Dict[str, Any] = {
            "eligible_steps": eligible,
            "active_steps": active,
            "nonfinite_steps": nonfinite,
        }

        if eligible == 0:
            fail_reasons.append(
                f"{term}: eligible_steps=0 (no legal inputs triggered eligibility)"
            )
            all_ok = False
            term_detail["eligible_ok"] = False
        else:
            term_detail["eligible_ok"] = True

        if active == 0:
            fail_reasons.append(
                f"{term}: active_steps=0 (eligibility met but reward never activated)"
            )
            all_ok = False
            term_detail["active_ok"] = False
        else:
            term_detail["active_ok"] = True

        if nonfinite > 0:
            fail_reasons.append(
                f"{term}: nonfinite_steps={nonfinite}"
            )
            all_ok = False
            term_detail["nonfinite_ok"] = False
        else:
            term_detail["nonfinite_ok"] = True

        term_details[term] = term_detail

    if not unsafe_support_reward_zero:
        fail_reasons.append(
            "unsafe_support_reward_zero=False: unsafe sample produced non-zero "
            "support reward"
        )
        all_ok = False

    # Count total nonfinite across all formal terms
    total_nonfinite = sum(
        term_stats.get(t, {}).get("nonfinite_steps", 0) for t in formal_terms
        if t in term_stats
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": None,  # filled by run_reward_wiring_audit
        GATE_PASS_KEY: all_ok,
        "fail_reasons": fail_reasons,
        "term_details": term_details,
        "unsafe_support_reward_zero": unsafe_support_reward_zero,
        "total_nonfinite_steps": total_nonfinite,
    }


def evaluate_role_distribution_gate(
    online_props: Dict[str, float],
    tol: float = 0.05,
) -> Dict[str, Any]:
    """Pure gate evaluation for role distribution.

    The pre-registered target is equal weight (1/3 each) with tolerance
    ``tol``.  Online phase uses role_quota_pilot=True with per-role minima
    of 2 each, then round-robin among 3 formal roles, giving effectively
    equal proportions in the limit.

    Args:
        online_props: Dict of {role_name: proportion} for the online phase.
        tol: Absolute tolerance around 1/3.

    Returns:
        dict with schema_version, run_id (None), gate_pass (bool),
        within_tolerance (bool), fail_reasons (list).
    """
    fail_reasons: List[str] = []
    all_ok = True

    target = 1.0 / 3.0
    lower = target - tol
    upper = target + tol

    # Check all formal roles present
    for r in FORMAL_ROLES:
        if r not in online_props:
            fail_reasons.append(f"Online props missing role '{r}'")
            all_ok = False

    if not all_ok:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": None,
            GATE_PASS_KEY: False,
            "within_tolerance": False,
            "fail_reasons": fail_reasons,
            "online_phase": {"mode": "quota_pilot_with_rr",
                             "per_role_props": dict(online_props)},
            "preregistered_target": {
                "type": "equal_weight",
                "target": round(target, 4),
                "tolerance": tol,
                "tolerance_range": [round(lower, 4), round(upper, 4)],
                "rationale": (
                    "C2 formal training uses role_quota_pilot=True: per-role "
                    "minimum quotas (2 each), then round-robin among 3 formal "
                    "roles.  In the limit, each role converges to 1/3."
                ),
            },
        }

    within_tolerance = True
    for r in FORMAL_ROLES:
        prop = online_props[r]
        if prop < lower or prop > upper:
            within_tolerance = False
            fail_reasons.append(
                f"{r}: {prop:.4f} outside [{lower:.4f}, {upper:.4f}]"
            )

    gate_pass = within_tolerance and len(fail_reasons) == 0
    if not gate_pass:
        all_ok = False

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": None,
        GATE_PASS_KEY: gate_pass,
        "within_tolerance": within_tolerance,
        "fail_reasons": fail_reasons,
        "online_phase": {
            "mode": "quota_pilot_with_rr",
            "per_role_props": dict(online_props),
            "source": "computed from collection logic (round-robin with underfilled quotas)",
        },
        "preregistered_target": {
            "type": "equal_weight",
            "target": round(target, 4),
            "tolerance": tol,
            "tolerance_range": [round(lower, 4), round(upper, 4)],
            "rationale": (
                "C2 formal training uses role_quota_pilot=True: per-role "
                "minimum quotas (2 each), then round-robin among 3 formal "
                "roles.  In the limit, each role converges to 1/3."
            ),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# §B  Helper: construct legal fixed samples that trigger each reward term
# ═══════════════════════════════════════════════════════════════════════════════

def _make_action_chunk(pos_from, pos_to, h=H_FUTURE):
    """Create action_chunk (H, 11) going linearly from pos_from to pos_to."""
    chunk = np.zeros((h, ACTION_DIM), dtype=np.float64)
    dpos = (np.array(pos_to) - np.array(pos_from)) / h
    for i in range(h):
        chunk[i, 0:3] = dpos
        chunk[i, 3:6] = dpos  # velocity approximation
    return chunk


def _make_obstacle_at(x, y, z=1.5, radius=0.5, height=0.5):
    """Create a single cylindrical ObstacleSpec."""
    from diffusion_policy_3d.env.collision_geom import ObstacleSpec
    return [ObstacleSpec(
        name="pillar", shape="cylinder",
        x=x, y=y, z=z, radius=radius, height=height,
    )]


def _clearance_fn_factory(obstacles):
    """Build a clearance_fn that wraps compute_obstacle_clearance."""
    from diffusion_policy_3d.env.collision_geom import compute_obstacle_clearance
    def fn(px, py, pz):
        d, name, flyover = compute_obstacle_clearance(px, py, pz, obstacles, check_3d=True)
        return (d if d is not None else 100.0), name, flyover
    return fn


def _count_nonfinite(arr: np.ndarray) -> int:
    """Count NaN and Inf values in an array."""
    return int(np.sum(~np.isfinite(arr)))


# ── Onset sample ──────────────────────────────────────────────────────────────

def build_onset_samples() -> Tuple[List[dict], int, int, int]:
    """Build samples that SHOULD trigger onset eligibility.

    Onset requires a low-risk BASELINE in history to detect lateral
    displacement increase.  We pre-seed history with low-risk steps
    (far from obstacles, low lateral displacement) before the high-risk
    evasive step.

    Returns (samples_list, n_eligible, n_active, nonfinite_count).
    Each sample is a dict with keys usable as **kwargs to compute_reward_v13.
    """
    from diffusion_policy_3d.env.reward_v13 import (
        compute_reward_v13, create_reward_history, REWARD_V13_DEFAULTS,
    )
    from diffusion_policy_3d.env.collision_geom import (
        ObstacleSpec, compute_obstacle_clearance,
    )

    obs = _make_obstacle_at(0.0, 0.0, 1.5, 0.5, 0.5)
    clr_fn = _clearance_fn_factory(obs)
    goal = np.array([6.0, 0.0, 2.0], dtype=np.float64)

    # ── Pre-seed history with low-risk steps (baseline for onset) ──
    # Use a position far from obstacles so risk_gate is low.
    far_pos = np.array([5.0, 5.0, 2.0], dtype=np.float64)
    far_goal = np.array([8.0, 5.0, 2.0], dtype=np.float64)
    far_vel = np.array([0.5, 0.0, 0.0], dtype=np.float64)

    history = create_reward_history()
    for step_i in range(5):
        s_far_t = {"pos_world": far_pos + np.array([0.1*step_i, 0.0, 0.0]),
                   "yaw": 0.0, "vel_world": far_vel,
                   "z": float(far_pos[2])}
        s_far_tp1 = {"pos_world": far_pos + np.array([0.1*(step_i+1), 0.0, 0.0]),
                     "yaw": 0.0, "collided": False}
        fc = _make_action_chunk(
            far_pos + np.array([0.1*step_i, 0.0, 0.0]),
            far_pos + np.array([0.1*(step_i+1) + 1.0, 0.0, 0.0]))
        _, _, m_far, history = compute_reward_v13(
            s_far_t, fc, s_far_tp1,
            obstacle_json=[], goal_world=far_goal,
            history=history,  # no obstacle → low risk
        )
        # Verify low risk
        assert m_far.get("risk_gate", 1.0) < 0.2, (
            f"Baseline risk_gate={m_far['risk_gate']:.4f} not low"
        )

    samples: List[dict] = []
    n_eligible = 0
    n_active = 0
    nonfinite_total = 0

    # Sample 1: near obstacle, moving right (evasive action)
    # Now history has low-risk baseline → delta_lat should be positive
    pos_t = np.array([0.0, 0.6, 1.5], dtype=np.float64)
    vel_t = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    pos_tp1 = np.array([0.0, 1.2, 1.8], dtype=np.float64)
    action_chunk = _make_action_chunk([0.0, 0.6, 1.5], [0.0, 2.0, 2.0])

    s_t = {"pos_world": pos_t, "yaw": 0.0, "vel_world": vel_t, "z": float(pos_t[2])}
    s_tp1 = {"pos_world": pos_tp1, "yaw": 0.0, "collided": False}

    reward, terms, metrics, new_h = compute_reward_v13(
        s_t, action_chunk, s_tp1,
        obstacle_json=obs, goal_world=goal,
        history=history, clearance_fn=clr_fn,
    )
    nf = _count_nonfinite(np.array(list(terms.values()) + [reward]))
    nonfinite_total += nf
    if metrics.get("onset_active", False):
        n_eligible += 1
        # Active = reward component non-zero
        if terms.get("onset", 0.0) > 0:
            n_active += 1
    samples.append({
        "label": "onset_sample1",
        "terms": terms,
        "metrics": metrics,
        "reward": reward,
        "nonfinite": nf,
    })

    # Sample 2: alternative configuration
    pos_t2 = np.array([0.0, 0.6, 1.5], dtype=np.float64)
    vel_t2 = np.array([0.5, 0.3, 0.1], dtype=np.float64)
    pos_tp1_2 = np.array([0.2, 1.0, 1.7], dtype=np.float64)
    action_chunk2 = _make_action_chunk([0.0, 0.6, 1.5], [1.0, 1.8, 2.0])

    s_t2 = {"pos_world": pos_t2, "yaw": 0.0, "vel_world": vel_t2, "z": float(pos_t2[2])}
    s_tp1_2 = {"pos_world": pos_tp1_2, "yaw": 0.0, "collided": False}

    history2 = create_reward_history()
    reward2, terms2, metrics2, new_h2 = compute_reward_v13(
        s_t2, action_chunk2, s_tp1_2,
        obstacle_json=obs, goal_world=goal,
        history=history2, clearance_fn=clr_fn,
    )
    nf2 = _count_nonfinite(np.array(list(terms2.values()) + [reward2]))
    nonfinite_total += nf2
    if metrics2.get("onset_active", False):
        n_eligible += 1
        if terms2.get("onset", 0.0) > 0:
            n_active += 1
    samples.append({
        "label": "onset_sample2",
        "terms": terms2,
        "metrics": metrics2,
        "reward": reward2,
        "nonfinite": nf2,
    })

    return samples, n_eligible, n_active, nonfinite_total


# ── Future clearance sample ───────────────────────────────────────────────────

def build_future_clearance_samples() -> Tuple[List[dict], int, int, int]:
    """Build samples triggering future_clearance eligibility.

    Active definition: progress_fc > -epsilon_back AND reward != 0.
    """
    from diffusion_policy_3d.env.reward_v13 import (
        compute_reward_v13, create_reward_history, REWARD_V13_DEFAULTS,
    )
    from diffusion_policy_3d.env.collision_geom import ObstacleSpec

    obs = _make_obstacle_at(0.0, 0.0, 1.5, 0.5, 0.5)
    goal = np.array([6.0, 0.0, 2.0], dtype=np.float64)

    epsilon_back = 0.05
    samples: List[dict] = []
    n_eligible = 0
    n_active = 0
    nonfinite_total = 0

    # Position far from obstacles → high d_soft_h
    pos_t = np.array([0.0, 3.0, 2.0], dtype=np.float64)
    vel_t = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    # Move closer to goal → positive progress
    pos_tp1 = np.array([0.5, 3.0, 2.0], dtype=np.float64)
    action_chunk = _make_action_chunk([0.0, 3.0, 2.0], [3.0, 3.0, 2.5])

    s_t = {"pos_world": pos_t, "yaw": 0.0, "vel_world": vel_t, "z": float(pos_t[2])}
    s_tp1 = {"pos_world": pos_tp1, "yaw": 0.0, "collided": False}

    history = create_reward_history()
    reward, terms, metrics, new_h = compute_reward_v13(
        s_t, action_chunk, s_tp1,
        obstacle_json=obs, goal_world=goal,
        history=history,
    )
    nf = _count_nonfinite(np.array(list(terms.values()) + [reward]))
    nonfinite_total += nf

    # Eligibility: progress towards goal > -epsilon_back
    dist_t = float(np.linalg.norm(pos_t - goal))
    dist_tp1 = float(np.linalg.norm(pos_tp1 - goal))
    progress_fc = dist_t - dist_tp1
    eligible = progress_fc > -epsilon_back
    active = eligible and abs(terms.get("future_clearance", 0.0)) > 0

    if eligible:
        n_eligible += 1
    if active:
        n_active += 1

    samples.append({
        "label": "fc_sample1",
        "terms": terms,
        "metrics": metrics,
        "reward": reward,
        "nonfinite": nf,
        "progress_fc": progress_fc,
        "eligible": eligible,
        "active": active,
    })

    return samples, n_eligible, n_active, nonfinite_total


# ── Trap sample ───────────────────────────────────────────────────────────────

def build_trap_samples() -> Tuple[List[dict], int, int, int]:
    """Build samples triggering trap eligibility.

    Requires pre-seeded history with low progress, low clearance, and m_t flips.
    """
    from diffusion_policy_3d.env.reward_v13 import (
        compute_reward_v13, create_reward_history, REWARD_V13_DEFAULTS,
    )
    from diffusion_policy_3d.env.collision_geom import ObstacleSpec

    obs = _make_obstacle_at(0.0, 0.0, 1.5, 0.5, 0.5)
    clr_fn = _clearance_fn_factory(obs)
    goal = np.array([6.0, 0.0, 2.0], dtype=np.float64)

    samples: List[dict] = []
    n_eligible = 0
    n_active = 0
    nonfinite_total = 0

    pos_t = np.array([0.0, 0.55, 1.5], dtype=np.float64)
    vel_t = np.array([0.05, 0.0, 0.0], dtype=np.float64)
    pos_tp1 = np.array([0.01, 0.55, 1.5], dtype=np.float64)
    action_chunk = _make_action_chunk(pos_t, [0.1, 0.55, 1.5])

    s_t = {"pos_world": pos_t, "yaw": 0.0, "vel_world": vel_t, "z": float(pos_t[2])}
    s_tp1 = {"pos_world": pos_tp1, "yaw": 0.0, "collided": False}

    # Pre-seed history with 6 steps of stalled/low-clearance/flip conditions
    history = create_reward_history()
    for i in range(6):
        history["progress"].append(0.01)   # total ~0.06 < 0.25
        history["d_soft_h"].append(0.3)    # low clearance
        history["d_min"].append(0.3)       # min_d_over_window < 0.65
        history["m_t"].append(1 if i % 2 == 0 else -1)  # flip each step > n_flip_max=2
        history["risk_gate"].append(0.8)
        history["u_lat_abs"].append(0.5 + 0.1 * i)
        history["yaw_rate_abs"].append(0.1)

    reward, terms, metrics, new_h = compute_reward_v13(
        s_t, action_chunk, s_tp1,
        obstacle_json=obs, goal_world=goal,
        history=history, clearance_fn=clr_fn,
    )
    nf = _count_nonfinite(np.array(list(terms.values()) + [reward]))
    nonfinite_total += nf

    trap_active = metrics.get("trap_active", False)
    trap_val = terms.get("trap", 0.0)
    trap_recovered = metrics.get("trap_recovered", False)

    if trap_active or trap_recovered:
        n_eligible += 1
    # Active: reward component != 0
    if trap_val != 0.0:
        n_active += 1

    samples.append({
        "label": "trap_sample1",
        "terms": terms,
        "metrics": metrics,
        "reward": reward,
        "nonfinite": nf,
        "trap_active": trap_active,
        "trap_recovered": trap_recovered,
    })

    # Sample 2: with higher vel to ensure risk_gate activates
    pos_t2 = np.array([0.0, 0.55, 1.5], dtype=np.float64)
    vel_t2 = np.array([0.5, 0.2, 0.0], dtype=np.float64)
    pos_tp1_2 = np.array([0.02, 0.55, 1.5], dtype=np.float64)
    action_chunk2 = _make_action_chunk([0.0, 0.55, 1.5], [0.1, 0.60, 1.5])

    s_t2 = {"pos_world": pos_t2, "yaw": 0.0, "vel_world": vel_t2, "z": float(pos_t2[2])}
    s_tp1_2 = {"pos_world": pos_tp1_2, "yaw": 0.0, "collided": False}

    history2 = create_reward_history()
    for i in range(6):
        history2["progress"].append(0.02)
        history2["d_soft_h"].append(0.3)
        history2["d_min"].append(0.3)
        history2["m_t"].append(1 if i % 2 == 0 else -1)
        history2["risk_gate"].append(0.8)
        history2["u_lat_abs"].append(0.6)
        history2["yaw_rate_abs"].append(0.1)

    reward2, terms2, metrics2, new_h2 = compute_reward_v13(
        s_t2, action_chunk2, s_tp1_2,
        obstacle_json=obs, goal_world=goal,
        history=history2, clearance_fn=clr_fn,
    )
    nf2 = _count_nonfinite(np.array(list(terms2.values()) + [reward2]))
    nonfinite_total += nf2

    trap_active2 = metrics2.get("trap_active", False)
    trap_val2 = terms2.get("trap", 0.0)
    if trap_active2 or metrics2.get("trap_recovered", False):
        n_eligible += 1
    if trap_val2 != 0.0:
        n_active += 1

    samples.append({
        "label": "trap_sample2",
        "terms": terms2,
        "metrics": metrics2,
        "reward": reward2,
        "nonfinite": nf2,
        "trap_active": trap_active2,
        "trap_recovered": metrics2.get("trap_recovered", False),
    })

    return samples, n_eligible, n_active, nonfinite_total


# ── Support (low-density) sample ──────────────────────────────────────────────

def build_support_samples() -> Tuple[List[dict], int, int, int]:
    """Build samples triggering support low-density eligibility."""
    from diffusion_policy_3d.env.reward_v13 import (
        compute_reward_v13, create_reward_history, compute_support_cell,
        REWARD_V13_DEFAULTS,
    )
    from diffusion_policy_3d.env.collision_geom import ObstacleSpec

    # Far obstacle so d_soft_h > support_safe (0.75)
    obs = _make_obstacle_at(5.0, 5.0, 1.5, 0.5, 0.5)
    goal = np.array([6.0, 0.0, 2.0], dtype=np.float64)

    # Offline density: most cells well-visited, but our target cell is empty
    offline_density = {
        "cell_counts": {},
        "total": 1000,
        "p20_threshold": 1.0,   # tau_d = 1, so count_c=0 < 1 → low density
        "p50_threshold": 5.0,
        "bin_size": (1.0, 0.5, 0.5),
    }

    # Pre-populate some other cells so the density is not completely empty
    for i in range(10):
        cell = (i, 0, 4)
        offline_density["cell_counts"][cell] = 5

    samples: List[dict] = []
    n_eligible = 0
    n_active = 0
    nonfinite_total = 0

    pos_t = np.array([3.0, 0.0, 2.0], dtype=np.float64)
    vel_t = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    pos_tp1 = np.array([4.0, 0.0, 2.0], dtype=np.float64)
    action_chunk = _make_action_chunk(pos_t, [6.0, 0.0, 2.0])

    s_t = {"pos_world": pos_t, "yaw": 0.0, "vel_world": vel_t, "z": float(pos_t[2])}
    s_tp1 = {"pos_world": pos_tp1, "yaw": 0.0, "collided": False}

    history = create_reward_history()
    for i in range(6):
        history["progress"].append(0.1)    # total 0.6 > 0.2
        history["d_soft_h"].append(2.0)    # high clearance
        history["d_min"].append(2.0)
        history["risk_gate"].append(0.01)  # low risk
        history["u_lat_abs"].append(0.0)
        history["yaw_rate_abs"].append(0.0)
        history["m_t"].append(0)

    reward, terms, metrics, new_h = compute_reward_v13(
        s_t, action_chunk, s_tp1,
        obstacle_json=obs, goal_world=goal,
        history=history,
        offline_density=offline_density,
    )
    nf = _count_nonfinite(np.array(list(terms.values()) + [reward]))
    nonfinite_total += nf

    low_density = metrics.get("low_density_cell", False)
    support_val = terms.get("support", 0.0)

    if low_density:
        n_eligible += 1
    # Active: reward component > 0
    if support_val > 0:
        n_active += 1

    samples.append({
        "label": "support_sample1",
        "terms": terms,
        "metrics": metrics,
        "reward": reward,
        "nonfinite": nf,
        "low_density_cell": low_density,
    })

    return samples, n_eligible, n_active, nonfinite_total


# ── Unsafe support sample ─────────────────────────────────────────────────────

def build_unsafe_support_sample() -> Tuple[dict, float]:
    """Build a sample that is low-density BUT unsafe (d_soft_h <= support_safe
    or collided). Returns (sample, reward_value) for checking support == 0.

    The sample uses a nearby obstacle so d_soft_h < support_safe (0.75).
    """
    from diffusion_policy_3d.env.reward_v13 import (
        compute_reward_v13, create_reward_history, compute_support_cell,
        REWARD_V13_DEFAULTS,
    )
    from diffusion_policy_3d.env.collision_geom import ObstacleSpec

    # Obstacle near position → d_soft_h < 0.75
    obs = _make_obstacle_at(3.0, 0.0, 1.5, 0.5, 0.5)
    clr_fn = _clearance_fn_factory(obs)
    goal = np.array([6.0, 0.0, 2.0], dtype=np.float64)

    offline_density = {
        "cell_counts": {},
        "total": 1000,
        "p20_threshold": 1.0,
        "p50_threshold": 5.0,
        "bin_size": (1.0, 0.5, 0.5),
    }

    pos_t = np.array([3.0, 0.0, 1.5], dtype=np.float64)
    vel_t = np.array([0.1, 0.0, 0.0], dtype=np.float64)
    pos_tp1 = np.array([3.0, 0.0, 1.5], dtype=np.float64)  # same spot (collocated with obstacle)
    action_chunk = _make_action_chunk(pos_t, [3.5, 0.0, 1.5])

    s_t = {"pos_world": pos_t, "yaw": 0.0, "vel_world": vel_t, "z": float(pos_t[2])}
    s_tp1 = {"pos_world": pos_tp1, "yaw": 0.0, "collided": False}

    history = create_reward_history()
    for i in range(6):
        history["progress"].append(0.1)
        history["d_soft_h"].append(0.3)
        history["d_min"].append(0.3)
        history["risk_gate"].append(0.8)
        history["u_lat_abs"].append(0.5)
        history["yaw_rate_abs"].append(0.1)
        history["m_t"].append(1 if i % 2 == 0 else -1)

    reward, terms, metrics, new_h = compute_reward_v13(
        s_t, action_chunk, s_tp1,
        obstacle_json=obs, goal_world=goal,
        history=history,
        offline_density=offline_density,
        clearance_fn=clr_fn,
    )

    return {
        "label": "unsafe_support",
        "terms": terms,
        "metrics": metrics,
        "reward": reward,
        "d_soft_h": metrics.get("d_soft_h", -1),
        "low_density_cell": metrics.get("low_density_cell", False),
    }, terms.get("support", -999.0)


# ═══════════════════════════════════════════════════════════════════════════════
# §C  Role distribution: count B0 replay
# ═══════════════════════════════════════════════════════════════════════════════

def count_b0_replay_roles(replay_path: str = B0_REPLAY_PATH) -> dict:
    """Count transitions per role in B0 replay.

    Returns dict with per_role_counts, per_role_props, total_transitions.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from diffusion_policy_3d.rl.routea_replay_buffer import RouteAReplayBuffer

    replay = RouteAReplayBuffer.load(replay_path)
    trans = replay.get_all_transitions()

    counts: Dict[str, int] = defaultdict(int)
    eps_by_role: Dict[str, set] = defaultdict(set)

    for t in trans:
        role = getattr(t, "role_active", "") or getattr(t, "role_validated", "") or "unknown"
        counts[role] += 1
        wuid = getattr(t, "world_uid", "")
        seed = getattr(t, "episode_seed", -1)
        eps_by_role[role].add((wuid, seed))

    total = sum(counts.values())
    props = {}
    for r in FORMAL_ROLES:
        props[r] = counts.get(r, 0) / max(total, 1)

    return {
        "per_role_counts": dict(counts),
        "per_role_props": props,
        "per_role_episodes": {r: len(eps_by_role.get(r, set())) for r in FORMAL_ROLES},
        "total_transitions": total,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# §D  Online role distribution from C2 collection logic
# ═══════════════════════════════════════════════════════════════════════════════

def compute_online_role_proportions(
    n_episodes: int = 150,
    ft_min_episodes: Dict[str, int] = None,
    formal_roles: Tuple[str, ...] = FORMAL_ROLES,
) -> Dict[str, float]:
    """Compute expected role proportions for C2 online phased collection.

    The C2 collection uses role_quota_pilot=True with:
    - Per-role minimum quotas (default 2 each)
    - After quotas met: round-robin over formal_roles with support share cap of 0.50
    - With many episodes, this converges to 1/3 each

    Args:
        n_episodes: Number of episodes to simulate.
        ft_min_episodes: Per-role minimum quotas.
        formal_roles: Tuple of role names.

    Returns:
        dict of role -> proportion.
    """
    if ft_min_episodes is None:
        ft_min_episodes = {r: 2 for r in formal_roles}

    counts = {r: 0 for r in formal_roles}
    cursor = 0

    for ep_i in range(n_episodes):
        # Quota logic (replicates _select_next_role_for_quota)
        underfilled = [r for r in formal_roles if counts[r] < ft_min_episodes[r]]
        if underfilled:
            # Most underfilled
            role = min(underfilled, key=lambda r: (
                counts[r] / max(ft_min_episodes[r], 1), formal_roles.index(r)
            ))
        else:
            # Round-robin with support share cap
            role = formal_roles[cursor % len(formal_roles)]
            cursor += 1
            _total = sum(counts.values())
            if role == "support_expansion" and _total > 0:
                _support_share = (counts.get("support_expansion", 0) + 1) / (_total + 1)
                if _support_share > 0.50:
                    role = formal_roles[cursor % len(formal_roles)]
                    cursor += 1
        counts[role] += 1

    total = sum(counts.values())
    return {r: counts[r] / max(total, 1) for r in formal_roles}


# ═══════════════════════════════════════════════════════════════════════════════
# §E  Main audit runners
# ═══════════════════════════════════════════════════════════════════════════════

def run_reward_wiring_audit(config: dict) -> dict:
    """Drive real reward_v13 on fixed legal samples and produce audit report.

    Config keys:
        output_dir (str): Write audit JSON here.
        run_id (str, optional): Run identifier.
    """
    output_dir = Path(config.get("output_dir", "."))
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = config.get("run_id", output_dir.name)

    print("\n  [RewardWiring] Building fixed legal samples for each reward term...")

    # ── Onset ────────────────────────────────────────────────────────────────
    print("    onset: building samples near obstacle with evasive action...")
    onset_samples, onset_eligible, onset_active, onset_nf = build_onset_samples()
    print(f"      eligible={onset_eligible}, active={onset_active}, "
          f"nonfinite={onset_nf}")
    if onset_samples:
        for s in onset_samples:
            print(f"      {s['label']}: onset_term={s['terms'].get('onset', 0):.6f}, "
                  f"risk_gate={s['metrics'].get('risk_gate', 0):.4f}, "
                  f"onset_active={s['metrics'].get('onset_active', False)}")

    # ── Future clearance ─────────────────────────────────────────────────────
    print("    future_clearance: building samples with positive progress...")
    fc_samples, fc_eligible, fc_active, fc_nf = build_future_clearance_samples()
    print(f"      eligible={fc_eligible}, active={fc_active}, "
          f"nonfinite={fc_nf}")
    for s in fc_samples:
        print(f"      {s['label']}: fc_term={s['terms'].get('future_clearance', 0):.6f}, "
              f"eligible={s.get('eligible', False)}, active={s.get('active', False)}")

    # ── Trap ─────────────────────────────────────────────────────────────────
    print("    trap: building samples with stalled/low-clearance/flip history...")
    trap_samples, trap_eligible, trap_active, trap_nf = build_trap_samples()
    print(f"      eligible={trap_eligible}, active={trap_active}, "
          f"nonfinite={trap_nf}")
    for s in trap_samples:
        print(f"      {s['label']}: trap_term={s['terms'].get('trap', 0):.6f}, "
              f"trap_active={s['metrics'].get('trap_active', False)}, "
              f"trap_recovered={s['metrics'].get('trap_recovered', False)}")

    # ── Support (low-density) ────────────────────────────────────────────────
    print("    support: building samples in low-density cell with sufficient progress...")
    sup_samples, sup_eligible, sup_active, sup_nf = build_support_samples()
    print(f"      eligible={sup_eligible}, active={sup_active}, "
          f"nonfinite={sup_nf}")
    for s in sup_samples:
        print(f"      {s['label']}: support_term={s['terms'].get('support', 0):.6f}, "
              f"low_density_cell={s['metrics'].get('low_density_cell', False)}, "
              f"d_soft_h={s['metrics'].get('d_soft_h', -1):.4f}")

    # ── Unsafe support ───────────────────────────────────────────────────────
    print("    unsafe_support: low-density BUT near obstacle (d_soft_h <= safe)...")
    unsafe_sample, unsafe_support_val = build_unsafe_support_sample()
    unsafe_support_zero = abs(unsafe_support_val) < 1e-10
    print(f"      support_term={unsafe_support_val:.10f}, zero={unsafe_support_zero}, "
          f"d_soft_h={unsafe_sample.get('d_soft_h', -1):.4f}, "
          f"low_density_cell={unsafe_sample.get('low_density_cell', False)}")

    # ── Compile term stats ───────────────────────────────────────────────────
    term_stats = {
        "onset": {
            "eligible_steps": onset_eligible,
            "active_steps": onset_active,
            "nonfinite_steps": onset_nf,
        },
        "future_clearance": {
            "eligible_steps": fc_eligible,
            "active_steps": fc_active,
            "nonfinite_steps": fc_nf,
        },
        "trap": {
            "eligible_steps": trap_eligible,
            "active_steps": trap_active,
            "nonfinite_steps": trap_nf,
        },
        "support": {
            "eligible_steps": sup_eligible,
            "active_steps": sup_active,
            "nonfinite_steps": sup_nf,
        },
    }

    # ── Evaluate gate ────────────────────────────────────────────────────────
    gate_result = evaluate_reward_wiring_gate(term_stats, unsafe_support_zero)
    gate_result["run_id"] = run_id

    # ── Build per-term fired info ─────────────────────────────────────────────
    fired_info = {}
    for term in ["onset", "future_clearance", "trap", "support"]:
        stats = term_stats.get(term, {})
        fired_info[term] = {
            "eligible_steps": stats.get("eligible_steps", 0),
            "active_steps": stats.get("active_steps", 0),
            "nonfinite_steps": stats.get("nonfinite_steps", 0),
            "fired": stats.get("active_steps", 0) > 0,
        }

    # ── Build audit report ───────────────────────────────────────────────────
    audit = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        GATE_PASS_KEY: gate_result[GATE_PASS_KEY],
        "terms": fired_info,
        "unsafe_support_reward_zero": unsafe_support_zero,
        "active_definition_future_clearance": (
            "future_clearance has no dedicated 'active' boolean in reward_v13 "
            "metrics.  'active_steps' is defined as: eligibility met "
            "(progress_fc > -epsilon_back) AND reward component non-zero."
        ),
        "sample_construction_notes": {
            "onset": (
                "Position at (0.0, 0.6, 1.5), near pillar obstacle at origin "
                "(r=0.5).  Clearance ~0.1m → high risk.  Evasive action "
                "moving right (+y) improves future clearance."
            ),
            "future_clearance": (
                "Position at (0.0, 3.0, 2.0), far from obstacles.  Positive "
                "goal progress ensures progress_fc > -epsilon_back gate passes.  "
                "High d_soft_h → positive tanh reward."
            ),
            "trap": (
                "Position at (0.0, 0.55, 1.5) near pillar.  Pre-seeded 6-step "
                "history with total_progress < 0.25, min_d_over_window < 0.65, "
                "and n_flip > 2 (alternating m_t = +1/-1)."
            ),
            "support": (
                "Position at (3.0, 0.0, 2.0), goal at (6.0, 0.0, 2.0).  "
                "Offline density has p20_threshold=1.0, cell at (4,0,2) has "
                "count=0 < 1 → low density.  Pre-seeded history with "
                "total_progress > 0.2.  Obstacle far away → d_soft_h > 0.75."
            ),
            "unsafe_support": (
                "Same position/density setup but obstacle directly at (3,0,1.5) "
                "so d_soft_h < support_safe (0.75), blocking the safety gate."
            ),
        },
        "nonfinite_total": gate_result.get("total_nonfinite_steps", 0),
        "notes": [],
    }

    if not unsafe_support_zero:
        audit["notes"].append(
            "WARNING: unsafe support reward != 0 — the safety gate "
            "(d_soft_h <= support_safe → 0) may be broken."
        )

    if audit[GATE_PASS_KEY]:
        print(f"\n  [RewardWiring] GATE PASS: all 4 formal terms fire correctly, "
              f"no NaN/Inf, unsafe support=0")
    else:
        print(f"\n  [RewardWiring] GATE FAIL: {gate_result['fail_reasons']}")

    # ── Persist ──────────────────────────────────────────────────────────────
    out_path = output_dir / "reward_wiring_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)
    print(f"  [RewardWiring] Wrote {out_path}")

    return audit


def run_role_distribution_audit(config: dict) -> dict:
    """Count B0 replay transitions and compute online role distribution.

    Config keys:
        output_dir (str): Write audit JSON here.
        run_id (str, optional): Run identifier.
    """
    output_dir = Path(config.get("output_dir", "."))
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = config.get("run_id", output_dir.name)

    print("\n  [RoleDistribution] Counting B0 replay transitions per role...")
    b0_stats = count_b0_replay_roles()
    print(f"    Total transitions: {b0_stats['total_transitions']}")
    for r in FORMAL_ROLES:
        cnt = b0_stats["per_role_counts"].get(r, 0)
        prop = b0_stats["per_role_props"].get(r, 0)
        eps = b0_stats["per_role_episodes"].get(r, 0)
        print(f"    {r}: {cnt} trans ({prop:.4f}), {eps} eps")

    # Online phase: C2 uses role_quota_pilot=True, round-robin after minima
    print("\n  [RoleDistribution] Computing online phase role proportions "
          "(simulated)...")
    online_props = compute_online_role_proportions(n_episodes=150)
    for r in FORMAL_ROLES:
        print(f"    {r}: {online_props[r]:.4f}")
    print("    source: simulated quota pilot (min 2 each) + round-robin")

    # ── Evaluate gate ────────────────────────────────────────────────────────
    gate_result = evaluate_role_distribution_gate(online_props, tol=0.05)
    gate_result["run_id"] = run_id

    # ── B0 replay section for audit ──────────────────────────────────────────
    b0_replay_section = {
        "per_role_counts": b0_stats["per_role_counts"],
        "per_role_props": b0_stats["per_role_props"],
        "per_role_episodes": b0_stats["per_role_episodes"],
        "total_transitions": b0_stats["total_transitions"],
        "source": str(B0_REPLAY_PATH),
    }

    # ── Build audit report ───────────────────────────────────────────────────
    audit = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        GATE_PASS_KEY: gate_result[GATE_PASS_KEY],
        "b0_replay": b0_replay_section,
        "online_phase": {
            "mode": "quota_pilot_with_rr",
            "per_role_props": online_props,
            "source": (
                "Simulated C2 collection with role_quota_pilot=True: "
                "per-role minimum quotas (2 each), then round-robin over "
                "3 formal roles with support share cap 0.50.  150 episodes "
                "simulated."
            ),
        },
        "preregistered_target": {
            "type": "equal_weight",
            "target": round(1.0 / 3.0, 4),
            "tolerance": 0.05,
            "tolerance_range": [round(1.0 / 3.0 - 0.05, 4),
                                round(1.0 / 3.0 + 0.05, 4)],
            "rationale": (
                "C2 formal training uses role_quota_pilot=True.  Per-role "
                "minimum quotas of 2 each are met in the first 6 episodes; "
                "thereafter round-robin over the 3 formal roles gives equal "
                "weight in the limit.  The support share cap of 0.50 is "
                "unlikely to be triggered with equal round-robin (support "
                "would only exceed 50% if it was deliberately over-sampled "
                "in the first 6 episodes, which is not the case).  "
                "Therefore the effective target is 1/3 per role."
            ),
        },
        "within_tolerance": gate_result["within_tolerance"],
        "notes": [],
    }

    # B0 replay is a bootstrap set — skew is expected and documented
    b0_informal = [
        r for r in b0_stats["per_role_counts"]
        if r not in FORMAL_ROLES
    ]
    if b0_informal:
        audit["notes"].append(
            f"B0 replay contains non-formal role transitions: {b0_informal}.  "
            "These are excluded from role-proportion calculations."
        )

    max_prop = max(b0_stats["per_role_props"].values())
    min_prop = min(b0_stats["per_role_props"].values())
    if max_prop - min_prop > 0.05:
        audit["notes"].append(
            f"B0 replay role proportions deviate from equal: range "
            f"[{min_prop:.4f}, {max_prop:.4f}].  "
            "This is expected: B0 is a bootstrap set collected with "
            "a different schedule (not role_quota_pilot).  The B0 "
            "skew does not affect the online training distribution, "
            "which uses role_quota_pilot= True with round-robin."
        )

    if audit[GATE_PASS_KEY]:
        print(f"\n  [RoleDistribution] GATE PASS: online role props within "
              f"1/3 ± 0.05")
    else:
        print(f"\n  [RoleDistribution] GATE FAIL: {gate_result['fail_reasons']}")

    # ── Persist ──────────────────────────────────────────────────────────────
    out_path = output_dir / "role_distribution_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)
    print(f"  [RoleDistribution] Wrote {out_path}")

    return audit
