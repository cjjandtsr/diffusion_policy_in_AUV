"""CAPABILITY-AUDIT: hand-built BC/RL traces -> capability metrics.

Metric definitions come from Reward_v1_3 §3.2/§3.4/§3.7/§3.8/§3.9/§5.2 and
RouteA_RL_Environment_Design_v1_3 §5.4; each test hand-constructs the
per-role traces (manual onset, manual low-density cells, manual trap
escape) and asserts the delta direction, the denominators, failed-episode
handling, and the empty-role / missing-density hard-fails.

The default frame is start=(-8,0,2), goal=(8,0,2) — the goal direction is
+x, so x_parallel = pos_x+8, x_perp = pos_y, z = pos_z, and the
goal-aligned cell is (floor(x_parallel/1.0), floor(x_perp/0.5),
floor(z/0.5)).
"""

import dataclasses
import math

import numpy as np
import pytest

from diffusion_policy_3d.rl.dppo_uav.capability_metrics import (
    OfflineDensity,
    compute_capability_metrics,
)
from diffusion_policy_3d.rl.dppo_uav.evaluator import (
    CAPABILITY_METRICS,
    EpisodePairedRecord,
)

START = [-8.0, 0.0, 2.0]
GOAL = [8.0, 0.0, 2.0]


def _step(i: int, pos: list, *, clearance: float, yaw: float = 0.0,
          terminated: bool = False, done_reason: str = "none") -> dict:
    """One trace step in the default frame (goal distance derived from
    the frozen start/goal geometry)."""
    return {
        "step": i, "elapsed": 0.1 * i,
        "pos": [float(v) for v in pos], "z": float(pos[2]),
        "yaw": float(yaw),
        "goal_dist": float(math.dist(pos, GOAL)),
        "clearance": float(clearance), "hit_count": 0,
        "action_norm": 1.0, "action_max_abs": 0.5,
        "done_reason": done_reason,
        "terminated": bool(terminated), "truncated": False,
    }


def _straight(n: int, *, y: float = 0.0, z: float = 2.0,
              clearances: list = None) -> list:
    """A straight +x flight, 1.0m per step from START."""
    cl = (clearances or [4.0] * n)
    assert len(cl) == n
    out = []
    for i in range(n):
        out.append(_step(i, [-8.0 + i, y, z], clearance=cl[i]))
    return out


def _record(role: str, bc_trace: list, rl_trace: list, *,
            bc_success: bool, rl_success: bool,
            bc_collision: bool = False, rl_collision: bool = False,
            seed: int = 42) -> EpisodePairedRecord:
    return EpisodePairedRecord(
        episode_config={"world": "w1", "seed": seed},
        seed=seed, world="w1", role=role,
        start=list(START), goal=list(GOAL),
        disturbance={"events": []},
        bc={"success": bc_success, "collision": bc_collision,
            "timeout": False, "return": 10.0, "steps": len(bc_trace)},
        rl={"success": rl_success, "collision": rl_collision,
            "timeout": False, "return": 10.0, "steps": len(rl_trace)},
        trace={"bc": bc_trace, "rl": rl_trace},
    )


def _density() -> OfflineDensity:
    """Corridor density in the default frame (z=2.0 -> z_bin 4).

    17 corridor cells (x_bin 0..16, y_bin 0, z_bin 4) visited 100x each —
    the full BC trace covers x_bin 0..16, so the corridor must span it or
    the final BC cell would count as novelty. 4 periphery cells (x_bin
    8/9, y_bin +-2, z_bin 4) visited once. p20 of the visit-count
    distribution is 100.0 (numpy linear interpolation: sorted
    [1,1,1,1,100x17], position 4.0), so corridor cells (100 >= 100) are
    NOT low-density and every other cell (0 or 1 < 100) IS low-density.
    """
    counts = {(x, 0, 4): 100 for x in range(17)}
    counts[(8, 2, 4)] = 1
    counts[(8, -2, 4)] = 1
    counts[(9, 2, 4)] = 1
    counts[(9, -2, 4)] = 1
    return OfflineDensity(counts=counts)


_CAPABILITY_ROLES = ("anticipatory", "support_expansion", "dense_recovery")


def _capability(records):
    """compute_capability_metrics padded to the full three-role set.

    Unit tests exercise one group at a time; the three-role hard-fail
    (D1) must NOT fire, so the missing capability roles get trivial
    1-step successful records that never touch the asserted group.
    """
    padded = list(records)
    present = {record.role for record in records}
    for role in _CAPABILITY_ROLES:
        if role not in present:
            padded.append(_record(
                role, [_step(0, START, clearance=4.0)],
                [_step(0, START, clearance=4.0)],
                bc_success=True, rl_success=True,
            ))
    return compute_capability_metrics(padded, _density())


# ═══════════════════════════════════════════════════════════════════
# anticipatory: early lateral onset (Reward v1.3 §3.4 / §5.2)
# ═══════════════════════════════════════════════════════════════════

def test_anticipatory_early_onset_delta():
    """An RL that deviates BEFORE the near-obstacle shell scores a longer
    onset_time and a higher onset_distance than a BC that swerves only
    after the shell is entered. Clearance cliff at t=8 (shell), so a
    deviation at t=4 is 4 steps early; a deviation at t=8 is inside the
    shell and does NOT count as an onset (no event, not 0-time)."""
    clearances = [4.0] * 8 + [0.5, 0.3, 0.3, 0.3]
    bc_trace = _straight(12, clearances=clearances)
    # BC swerves only at/after the shell entry.
    for i in range(8, 12):
        bc_trace[i]["pos"][1] = 0.5
    rl_trace = _straight(12, clearances=clearances)
    # RL swerves 4 steps before the shell, at clearance 4.0.
    for i in range(4, 12):
        rl_trace[i]["pos"][1] = 0.5
    metrics = _capability([_record(
        "anticipatory", bc_trace, rl_trace,
        bc_success=True, rl_success=True,
    )])["anticipatory_avoidance"]

    assert metrics["risk_episodes_bc"] == 1
    assert metrics["risk_episodes_rl"] == 1
    # The RL onset is the first pre-shell deviation: 4 steps before the
    # shell, at clearance 4.0.
    assert metrics["onset_n_rl"] == 1
    assert metrics["onset_n_bc"] == 0  # swerving inside the shell = no onset
    assert metrics["onset_distance_rl"] == 4.0
    assert metrics["onset_time_rl"] == 4  # shell step 8 - onset step 4
    assert metrics["onset_distance_bc"] == 0.0
    assert metrics["onset_time_bc"] == 0.0
    assert metrics["onset_distance_rl"] > metrics["onset_distance_bc"]
    assert metrics["onset_time_rl"] > metrics["onset_time_bc"]
    assert metrics["sr_bc"] == 1.0 and metrics["sr_rl"] == 1.0
    assert metrics["cr_bc"] == 0.0 and metrics["cr_rl"] == 0.0


def test_anticipatory_born_offset_is_not_an_onset():
    """An agent born off-axis (|x_perp| > eps) with a CONSTANT deviation
    is absorbed by the baseline and never counts as an onset — the onset
    must be a NEW deviation relative to the pre-shell baseline."""
    clearances = [4.0] * 8 + [0.5, 0.3, 0.3, 0.3]
    bc_trace = _straight(12, clearances=clearances)
    rl_trace = _straight(12, y=0.5, clearances=clearances)  # born off-axis
    metrics = _capability([_record(
        "anticipatory", bc_trace, rl_trace,
        bc_success=True, rl_success=True,
    )])["anticipatory_avoidance"]
    assert metrics["onset_n_rl"] == 0
    assert metrics["onset_n_bc"] == 0


def test_anticipatory_constant_yaw_rate_is_not_an_onset():
    """A constant yaw rate is a steady command, not a new avoidance — the
    §3.4 Delta is relative to the low-risk baseline (|yaw_rate| minus its
    pre-shell mean), so a straight flight in a constant turn must NOT
    register an onset (a cumulative |yaw - yaw[0]| would grow linearly
    and falsely trigger)."""
    clearances = [4.0] * 8 + [0.5, 0.3, 0.3, 0.3]
    bc_trace = _straight(12, clearances=clearances)
    rl_trace = _straight(12, clearances=clearances)
    for i in range(12):
        rl_trace[i]["yaw"] = 0.15 * i  # constant 0.15 rad/step, no lateral
    metrics = _capability([_record(
        "anticipatory", bc_trace, rl_trace,
        bc_success=True, rl_success=True,
    )])["anticipatory_avoidance"]
    assert metrics["onset_n_rl"] == 0
    assert metrics["onset_n_bc"] == 0


def test_onset_nan_clearance_step_hard_fails():
    """R3.5: a NaN clearance must NOT be skipped into a valid-looking
    metric — the metric layer hard-fails on a non-finite required trace
    field (P1-4, supersedes the old skip/attribute-to-next-step behavior)."""
    clearances = [4.0] * 8 + [0.5, 0.3, 0.3, 0.3]
    rl_trace = _straight(12, clearances=clearances)
    for i in range(4, 12):
        rl_trace[i]["pos"][1] = 0.5
    rl_trace[4]["clearance"] = float("nan")  # first onset candidate
    with pytest.raises(ValueError, match="finite"):
        _capability([_record(
            "anticipatory", _straight(12, clearances=clearances), rl_trace,
            bc_success=True, rl_success=True,
        )])


def test_future_clearance_softmin_delta():
    """A gentler RL clearance profile raises the pooled future-clearance
    soft-min (mean and p10) over the same episode geometry — PAIRED
    inside one record, so the bc and rl pools each contain one trace."""
    record = _record("anticipatory",
                     _straight(12, clearances=[4.0] * 10 + [0.7, 0.4]),
                     _straight(12, clearances=[4.0] * 10 + [2.5, 2.0]),
                     bc_success=True, rl_success=True)
    metrics = _capability([record])["anticipatory_avoidance"]
    assert metrics["future_clearance_softmin_rl"] > \
        metrics["future_clearance_softmin_bc"]
    assert metrics["future_clearance_softmin_p10_rl"] > \
        metrics["future_clearance_softmin_p10_bc"]


def test_future_clearance_softmin_constant_exact():
    """The temporal soft-min of a CONSTANT clearance series is the
    constant itself (soft-min identity) — locks the H/beta/omega
    formula and the truncated-window renormalization."""
    trace = _straight(8, clearances=[4.0] * 8)
    metrics = _capability([_record(
        "anticipatory", trace, trace,
        bc_success=True, rl_success=True,
    )])["anticipatory_avoidance"]
    assert metrics["future_clearance_softmin_bc"] == pytest.approx(4.0)
    assert metrics["future_clearance_softmin_p10_bc"] == pytest.approx(4.0)


# ═══════════════════════════════════════════════════════════════════
# support expansion: goal-aligned low-density coverage (Reward v1.3 §3.9)
# ═══════════════════════════════════════════════════════════════════

def _mixed_rl_trace() -> list:
    """17-step flight that leaves the corridor on three axes: y=+1 for
    steps 0..7 (cells (0..7, 2, 4)), z=2.5 for steps 8..15 (cells
    (8..15, 0, 5)), then y=-1 at step 16 (cell (16, -2, 5)). Every cell
    visited has offline count 0 < tau_D -> low density."""
    trace = []
    for i in range(8):
        trace.append(_step(i, [-8.0 + i, 1.0, 2.0], clearance=4.0))
    for i in range(8, 16):
        trace.append(_step(i, [-8.0 + i, 0.0, 2.5], clearance=4.0))
    trace.append(_step(16, [8.0, -1.0, 2.5], clearance=4.0))
    return trace


def test_support_low_density_coverage_and_axes_delta():
    """BC stays in the offline corridor (0 low-density cells); RL visits
    17 distinct low-density cells across two axes (y and z)."""
    bc_trace = _straight(17)
    rl_trace = _mixed_rl_trace()
    metrics = _capability([_record(
        "support_expansion", bc_trace, rl_trace,
        bc_success=True, rl_success=True,
    )])["fullspace_support_expansion"]

    assert metrics["successful_episodes_bc"] == 1
    assert metrics["successful_episodes_rl"] == 1
    assert metrics["successful_low_density_coverage_bc"] == 0
    assert metrics["successful_low_density_coverage_rl"] == 17
    assert metrics["successful_low_density_coverage_rl"] > \
        metrics["successful_low_density_coverage_bc"]
    # y-cells: 9 steps contribute (0,1,0); z-cells: 8 steps contribute
    # (0,0,1) -> y share 9/17, z share 8/17, x never below tau_D.
    assert metrics["altitude_only_ratio_rl"] == pytest.approx(8.0 / 17.0)
    assert metrics["altitude_only_ratio_bc"] == 0.0
    assert metrics["low_density_axes_count_rl"] == 2
    assert metrics["low_density_axes_count_bc"] == 0
    assert metrics["out_of_support_rate_rl"] == 1.0
    assert metrics["out_of_support_rate_bc"] == 0.0
    # Successful-trajectory spans on the three goal-aligned axes.
    assert metrics["successful_x_parallel_span_rl"] == pytest.approx(16.0)
    assert metrics["successful_x_perp_span_rl"] == pytest.approx(2.0)
    assert metrics["successful_z_span_rl"] == pytest.approx(0.5)
    assert metrics["successful_x_perp_span_bc"] == 0.0
    assert metrics["successful_z_span_bc"] == 0.0
    # Anti-detour: straight path ratio 1.0; the mixed flight detours.
    assert metrics["path_length_ratio_bc"] == pytest.approx(1.0)
    assert metrics["path_length_ratio_rl"] > 1.0


def test_support_altitude_only_ratio():
    """An RL that expands only in z collects altitude-only novelty:
    z explains 100% of the low-density contribution (altitude-only risk,
    design threshold 70%) and only one axis counts."""
    rl_trace = _straight(17, z=2.5)  # cells (0..16, 0, 5), all low-density
    metrics = _capability([_record(
        "support_expansion", _straight(17), rl_trace,
        bc_success=True, rl_success=True,
    )])["fullspace_support_expansion"]
    assert metrics["altitude_only_ratio_rl"] == pytest.approx(1.0)
    assert metrics["low_density_axes_count_rl"] == 1
    assert metrics["successful_low_density_coverage_rl"] == 17


def test_support_failed_episode_excluded():
    """Coverage counts SUCCESSFUL trajectories only; a failed (collision)
    episode still contributes the SR/CR denominators but none of its
    low-density cells."""
    good = _record("support_expansion", _straight(17), _straight(17, z=2.5),
                   bc_success=True, rl_success=True, seed=1)
    bad = _record("support_expansion", _straight(17), _straight(17, z=2.5),
                  bc_success=True, rl_success=False, rl_collision=True,
                  seed=2)
    metrics = _capability([good, bad])["fullspace_support_expansion"]
    assert metrics["successful_episodes_rl"] == 1
    assert metrics["successful_low_density_coverage_rl"] == 17
    assert metrics["sr_rl"] == 0.5 and metrics["cr_rl"] == 0.5
    assert metrics["sr_bc"] == 1.0 and metrics["cr_bc"] == 0.0


# ═══════════════════════════════════════════════════════════════════
# dense obstacle: trap recovery / dwell / side flips (Reward v1.3 §3.7/3.8)
# ═══════════════════════════════════════════════════════════════════

def _dense_bc_trace() -> list:
    """Approach x=0 (t<=8), stall at x=0 (t=9..16), alternate the lateral
    side 7 times (8 alternating positions), collide at t=16. Trap (W=6
    window: zero progress + min clearance 0.3 + 5 flips) fires at
    t=13..16 -> one open segment [13..16] that never recovers."""
    trace = []
    for i in range(9):
        trace.append(_step(i, [-8.0 + i, 0.0, 2.0], clearance=4.0))
    y_side = [0.4, -0.4, 0.4, -0.4, 0.4, -0.4, 0.4, -0.4]
    cl = [0.7, 0.5, 0.5, 0.5, 0.4, 0.4, 0.3, 0.3]
    for k, (yy, cc) in enumerate(zip(y_side, cl)):
        i = 9 + k
        trace.append(_step(
            i, [0.0, yy, 2.0], clearance=cc,
            terminated=(i == 16), done_reason=("collision" if i == 16
                                               else "none"),
        ))
    return trace


def _dense_rl_trace() -> list:
    """Approach x=0 (t<=8), stall at x=0 (t=9..15) while alternating the
    side 6 times, then COMMIT to +y, resume 2m/step at t=16 and reach
    the goal at t=19. Trap fires at t=13..15 (one valid 3-step segment);
    the segment exits with progress and rising future clearance ->
    recovered."""
    trace = []
    for i in range(9):
        trace.append(_step(i, [-8.0 + i, 0.0, 2.0], clearance=4.0))
    y_seq = [0.4, -0.4, 0.4, -0.4, 0.4, -0.4, 0.4]
    cl = [0.7] + [0.5] * 7
    for k in range(7):
        trace.append(_step(9 + k, [0.0, y_seq[k], 2.0], clearance=cl[k]))
    for k, xx in enumerate([2.0, 4.0, 6.0, 8.0]):
        i = 16 + k
        trace.append(_step(
            i, [xx, 0.4, 2.0], clearance=[1.0, 1.8, 2.6, 3.4][k],
            terminated=(i == 19), done_reason=("reach" if i == 19
                                               else "none"),
        ))
    return trace


def test_dense_trap_recovery_and_side_flip_delta():
    bc_trace = _dense_bc_trace()
    rl_trace = _dense_rl_trace()
    metrics = _capability([_record(
        "dense_recovery", bc_trace, rl_trace,
        bc_success=False, bc_collision=True, rl_success=True,
    )])["dense_obstacle_ceiling"]

    # BC: one open trap segment [13..16] that never recovers; RL: one
    # 3-step segment [13..15] that recovers within the N-step window.
    assert metrics["trap_segments_bc"] == 1
    assert metrics["trap_segments_rl"] == 1
    assert metrics["trap_recovery_rate_bc"] == 0.0
    assert metrics["trap_recovery_rate_rl"] == 1.0
    assert metrics["trap_recovery_rate_rl"] > metrics["trap_recovery_rate_bc"]
    assert metrics["deadend_dwell_steps_mean_bc"] == 4.0
    assert metrics["deadend_dwell_steps_mean_rl"] == 3.0
    assert metrics["deadend_dwell_steps_mean_rl"] < \
        metrics["deadend_dwell_steps_mean_bc"]
    # Hesitation: BC flips side 7 times inside the obstacle interaction
    # (8 alternating positions); RL flips 6 times then commits.
    assert metrics["obstacle_interactions_bc"] == 1
    assert metrics["obstacle_interactions_rl"] == 1
    assert metrics["side_flip_count_mean_bc"] == 7.0
    assert metrics["side_flip_count_mean_rl"] == 6.0
    assert metrics["side_flip_count_mean_rl"] < \
        metrics["side_flip_count_mean_bc"]
    assert metrics["sr_bc"] == 0.0 and metrics["cr_bc"] == 1.0
    assert metrics["sr_rl"] == 1.0 and metrics["cr_rl"] == 0.0


# ═══════════════════════════════════════════════════════════════════
# denominators, failed episodes, hard-fails
# ═══════════════════════════════════════════════════════════════════

def test_sr_cr_denominators_exact():
    """SR/CR are exact fractions over ALL role episodes, failed episodes
    included."""
    traces = [_straight(8) for _ in range(3)]
    records = [
        _record("anticipatory", traces[0], traces[0],
                bc_success=True, rl_success=True, seed=1),
        _record("anticipatory", traces[1], traces[1],
                bc_success=True, rl_success=False, rl_collision=True,
                seed=2),
        _record("anticipatory", traces[2], traces[2],
                bc_success=False, bc_collision=True, rl_success=True,
                seed=3),
    ]
    metrics = _capability(records)["anticipatory_avoidance"]
    assert metrics["n_episodes"] == 3
    assert metrics["sr_bc"] == pytest.approx(2.0 / 3.0)
    assert metrics["cr_bc"] == pytest.approx(1.0 / 3.0)
    assert metrics["sr_rl"] == pytest.approx(2.0 / 3.0)
    assert metrics["cr_rl"] == pytest.approx(1.0 / 3.0)


def test_empty_role_hard_fail():
    """A capability group without a single record of its role is a hard
    failure — never an empty dict or a fabricated metric."""
    records = [
        _record("anticipatory", _straight(4), _straight(4),
                bc_success=True, rl_success=True),
        _record("dense_recovery", _straight(4), _straight(4),
                bc_success=True, rl_success=True),
    ]
    with pytest.raises(ValueError, match="support_expansion"):
        compute_capability_metrics(records, _density())
    with pytest.raises(ValueError, match="episode"):
        compute_capability_metrics([], _density())


def test_missing_offline_density_hard_fail():
    """offline_density=None hard-fails — the support metric must never
    return 0 pretending no improvement (Reward v1.3 §3.9)."""
    records = [
        _record(role, _straight(4), _straight(4),
                bc_success=True, rl_success=True)
        for role in ("anticipatory", "support_expansion",
                     "dense_recovery")
    ]
    with pytest.raises(ValueError, match="offline density"):
        compute_capability_metrics(records, None)


def test_compute_output_schema_all_finite():
    """The full three-role output: exact group set, required keys, every
    value a finite real number."""
    records = [
        _record("anticipatory", _straight(12,
                clearances=[4.0] * 8 + [1.2, 0.5, 0.3, 0.3]),
                _straight(12, clearances=[4.0] * 10 + [2.5, 2.0]),
                bc_success=True, rl_success=True),
        _record("support_expansion", _straight(17), _mixed_rl_trace(),
                bc_success=True, rl_success=True),
        _record("dense_recovery", _dense_bc_trace(), _dense_rl_trace(),
                bc_success=False, bc_collision=True, rl_success=True),
    ]
    metrics = compute_capability_metrics(records, _density())
    assert set(metrics) == set(CAPABILITY_METRICS)
    for group, group_metrics in metrics.items():
        assert group_metrics, f"{group} must not be empty"
        for key, value in group_metrics.items():
            assert isinstance(value, (int, float)), (group, key, value)
            assert np.isfinite(value), (group, key, value)
    for group in CAPABILITY_METRICS:
        assert metrics[group]["n_episodes"] == 1


def test_nan_pos_step_hard_fails():
    """R3.5: a NaN position cannot be attributed to a goal-aligned cell —
    the metric layer hard-fails instead of excluding the step (P1-4, the
    old skip behavior produced silently different coverage statistics)."""
    rl_trace = _mixed_rl_trace()
    rl_trace[5]["pos"][1] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        _capability([_record(
            "support_expansion", _straight(17), rl_trace,
            bc_success=True, rl_success=True,
        )])


def test_degenerate_start_equals_goal_no_crash():
    """A record whose start coincides with the goal cannot define a
    goal-aligned frame — the fallback frame is used and every metric
    stays finite (zero-norm g_hat would otherwise be NaN and crash the
    cell mapping) (L6)."""
    record = dataclasses.replace(
        _record("support_expansion", _straight(17), _mixed_rl_trace(),
                bc_success=True, rl_success=True),
        start=[0.0, 0.0, 0.0], goal=[0.0, 0.0, 0.0],
    )
    metrics = _capability([record])["fullspace_support_expansion"]
    for value in metrics.values():
        assert np.isfinite(value), value


def test_all_nan_clearance_window_hard_fails():
    """R3.5: a window of all-NaN clearances must NOT enter the trap
    comparison — the metric layer hard-fails on the first non-finite
    field (P1-4, supersedes the old nanmin-skip behavior)."""
    clearances = [4.0] * 6 + [float("nan")] * 6 + [0.4] * 4
    trace = _straight(16, clearances=clearances)
    with pytest.raises(ValueError, match="finite"):
        _capability([_record(
            "dense_recovery", _dense_bc_trace(), trace,
            bc_success=False, bc_collision=True,
            rl_success=False, rl_collision=True,
        )])


# ═══════════════════════════════════════════════════════════════════
# OfflineDensity: tau_D resolution (Reward v1.3 §3.9)
# ═══════════════════════════════════════════════════════════════════

def test_offline_density_tau_D_is_p20():
    d = OfflineDensity(counts={
        (0, 0, 0): 2, (1, 0, 0): 4, (2, 0, 0): 8, (3, 0, 0): 16,
        (4, 0, 0): 32,
    })
    # np.percentile linear interpolation: index 0.8 of [2,4,8,16,32].
    assert d.tau_D == pytest.approx(3.6)
    assert d.is_low_density((0, 0, 0))  # 2 < 3.6
    assert d.is_low_density((9, 9, 9))  # absent cell = 0 visits
    assert not d.is_low_density((4, 0, 0))  # 32 >= 3.6


def test_offline_density_tau_D_falls_back_to_p50():
    """p20 == 0 falls back to p50 (Reward v1.3 §3.9)."""
    d = OfflineDensity(counts={
        (0, 0, 0): 0, (1, 0, 0): 0, (2, 0, 0): 2, (3, 0, 0): 2,
    })
    assert d.tau_D == pytest.approx(1.0)
    assert d.is_low_density((0, 0, 0))  # 0 < 1
    assert not d.is_low_density((2, 0, 0))  # 2 >= 1


def test_offline_density_construction_failures():
    with pytest.raises(ValueError, match="counts"):
        OfflineDensity(counts={})
    with pytest.raises(ValueError, match="positive"):
        OfflineDensity(counts={(0, 0, 0): 0})


def test_offline_density_round_trip():
    d = _density()
    d2 = OfflineDensity.from_dict(d.to_dict())
    assert d2.counts == d.counts
    assert d2.bin_size == d.bin_size
    assert d2.tau_D == d.tau_D


def test_offline_density_tau_D_zero_hard_fails():
    """p20 == 0 AND p50 == 0 leaves no usable low-density threshold — a
    silent tau_D=0 would label every cell low-density (M2)."""
    with pytest.raises(ValueError, match="tau_D"):
        OfflineDensity(counts={
            (0, 0, 0): 0, (1, 0, 0): 0, (2, 0, 0): 5,
        })


def test_offline_density_negative_count_hard_fails():
    """Direct construction with a negative visit count is a malformed
    density, not a candidate for renormalization (L5)."""
    with pytest.raises(ValueError, match="non-negative"):
        OfflineDensity(counts={(0, 0, 0): -1, (1, 0, 0): 5})


def test_offline_density_from_dict_rejects_bool_count():
    """bool is an int subclass — True must not pass the count check as
    visit count 1 (L4)."""
    with pytest.raises(ValueError, match="int"):
        OfflineDensity.from_dict({
            "schema_version": "1",
            "bin_size": [1.0, 0.5, 0.5],
            "counts": {"0,0,0": True},
        })


def test_offline_density_constructor_rejects_bool_count():
    """Direct construction must reject bool counts the same way the JSON
    loader does — the constructor is public API, not a bypass (L2)."""
    with pytest.raises(ValueError, match="int"):
        OfflineDensity(counts={(0, 0, 0): True, (1, 0, 0): 5})


def test_offline_density_constructor_rejects_bad_bin_size():
    """A zero bin component would divide by zero in the cell mapping —
    the constructor validates bin_size like from_dict does (L2)."""
    with pytest.raises(ValueError, match="bin_size"):
        OfflineDensity(counts={(0, 0, 0): 5}, bin_size=(1.0, 0.0, 0.5))


def test_nan_clearance_hard_fails():
    """R3.5: a NaN clearance must NOT be masked out of the d_soft window
    into a valid-looking metric — the metric layer hard-fails (P1-4,
    supersedes the old mask-and-continue behavior)."""
    bc = _straight(12, clearances=[4.0] * 8 + [1.2, 0.5, 0.3, 0.3])
    bc[5]["clearance"] = float("nan")  # approach-region NaN
    with pytest.raises(ValueError, match="finite"):
        compute_capability_metrics(
            [_record("anticipatory", bc,
                     _straight(12, clearances=[4.0] * 10 + [2.5, 2.0]),
                     bc_success=True, rl_success=True)],
            _density(),
        )
