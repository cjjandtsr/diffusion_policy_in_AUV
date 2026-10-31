"""Route A capability metrics (Reward v1.3 §5.2, RTM EVAL-AUDIT).

Three BC/RL capability groups computed from the paired episode traces:
  - anticipatory_avoidance: near-obstacle shell entry, early lateral
    onset (time/distance before the shell), temporal future-clearance
    soft-min d_soft^H (mean / p10);
  - fullspace_support_expansion: goal-aligned low-density cell coverage
    of SUCCESSFUL episodes, axis attribution (lateral vs altitude) with
    the altitude-only risk ratio, out-of-support time share and the
    path-length anti-detour guard;
  - dense_obstacle_ceiling: W-step trap detection, trap segment dwell,
    escape recovery within N steps, side-flip count per obstacle
    interaction.

Hard-fail rules: no episodes, offline_density=None, or a
capability role without a single record -> ValueError (never an empty
dict or a fabricated metric). Truthful zeros: 0.0 appears only for a
real zero measurement — offline_density=None never "returns 0 pretending
no improvement".

Two metric-side adjudications (R3.5 D7, recorded here as the audit
anchor): the lateral side sign m_t uses the INSTANTANEOUS goal-aligned
x_perp sign with the eps_m dead-zone, not the §3.8 H-step mean — a mean
averages out an alternating hesitation within half the window and
destroys the flip signal this metric exists to measure; and the support
axis attribution is the 2-axis max deviation (lateral vs altitude), the
x_parallel axis is not counted as its own attribution axis. A third:
the §5.4.1 "normalized novelty contribution" is operationalized as a
per-step argmax over the two deviations with count shares — a
low-density step with zero deviation on both axes is counted in neither
share, so the two axis shares under-sum (never over-claim).

Every constant carries the Reward v1.3 §3.2-§3.9 parameter table; metric
definitions trace to the §5.2 metric table. The record object is
duck-typed (role / trace / start / goal / bc / rl) to keep this module
import-cycle-free from evaluator.py.
"""

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from diffusion_policy_3d.env.reward_v13 import compute_support_cell

# --- Reward v1.3 parameter table (§3.2-§3.9) ---------------------------
H = 8                       # §3.2 future-clearance window (RL steps)
BETA = 8.0                  # §3.2 soft-min temperature
OMEGA_DECAY = 0.15          # §3.2 exp(-0.15*(i-1)) near-term weights
D_RISK = 0.9                # §3.3 d_risk — near-obstacle shell threshold
KAPPA = 0.3                 # §3.4 yaw weight in the lateral onset signal
ETA_O = 0.03                # §3.4 onset delta threshold (m)
EPSILON_M = 0.10            # §3.8 lateral sign dead-zone (m)
W = 6                       # §3.7 trap window (RL steps)
EPSILON_P = 0.25            # §3.7 trap progress threshold (m over W)
D_TRAP = 0.65               # §3.7 trap clearance threshold (m)
N_MAX_FLIPS = 2             # §3.7 max side flips inside a non-trap window
MIN_TRAP_LEN = 2            # §3.7 valid trap segment length (steps)
EPSILON_REC = 0.5           # §3.7 escape progress threshold (m over W)
EPSILON_D = 0.15            # §3.7 escape d_soft rise threshold (m)
RECOVERY_WINDOW = 6         # §3.7 escape check window after segment exit
AXIS_CONTRIBUTION_THRESHOLD = 0.20   # RouteA env design §5.4.1 (20%)
DEFAULT_BIN_SIZE = (1.0, 0.5, 0.5)   # §3.9 goal-aligned cell (m)

# Each capability group is computed from exactly one role's records;
# yaw_lateral_recovery belongs to no capability group.
ROLE_OF_CAPABILITY = {
    "anticipatory": "anticipatory_avoidance",
    "support_expansion": "fullspace_support_expansion",
    "dense_recovery": "dense_obstacle_ceiling",
}
GROUP_OF_ROLE = {group: role for role, group in ROLE_OF_CAPABILITY.items()}

# Required keys per group — the validator schema: an artifact must carry
# the audited metric set, never a placeholder.
CAPABILITY_METRIC_KEYS = {
    "anticipatory_avoidance": (
        "n_episodes", "sr_bc", "sr_rl", "cr_bc", "cr_rl",
        "risk_episodes_bc", "risk_episodes_rl",
        "onset_n_bc", "onset_n_rl",
        "onset_time_bc", "onset_time_rl",
        "onset_distance_bc", "onset_distance_rl",
        "future_clearance_softmin_bc", "future_clearance_softmin_rl",
        "future_clearance_softmin_p10_bc", "future_clearance_softmin_p10_rl",
    ),
    "fullspace_support_expansion": (
        "n_episodes", "sr_bc", "sr_rl", "cr_bc", "cr_rl",
        "successful_episodes_bc", "successful_episodes_rl",
        "successful_low_density_coverage_bc",
        "successful_low_density_coverage_rl",
        "successful_x_parallel_span_bc", "successful_x_parallel_span_rl",
        "successful_x_perp_span_bc", "successful_x_perp_span_rl",
        "successful_z_span_bc", "successful_z_span_rl",
        "altitude_only_ratio_bc", "altitude_only_ratio_rl",
        "low_density_axes_count_bc", "low_density_axes_count_rl",
        "out_of_support_rate_bc", "out_of_support_rate_rl",
        "path_length_ratio_bc", "path_length_ratio_rl",
    ),
    "dense_obstacle_ceiling": (
        "n_episodes", "sr_bc", "sr_rl", "cr_bc", "cr_rl",
        "trap_recovery_rate_bc", "trap_recovery_rate_rl",
        "trap_segments_bc", "trap_segments_rl",
        "deadend_dwell_steps_mean_bc", "deadend_dwell_steps_mean_rl",
        "deadend_dwell_steps_max_bc", "deadend_dwell_steps_max_rl",
        "side_flip_count_mean_bc", "side_flip_count_mean_rl",
        "obstacle_interactions_bc", "obstacle_interactions_rl",
    ),
}


@dataclass(frozen=True)
class OfflineDensity:
    """Frozen offline cell visit counts (§3.9) with tau_D resolved.

    tau_D = p20 of the offline visit-count distribution; when p20 == 0
    the design falls back to p50. A cell is low-density when its offline
    count is strictly below tau_D (an absent cell counts as 0 visits).
    The bin is goal-aligned (x_parallel, x_perp, z), size
    (1.0m, 0.5m, 0.5m).
    """

    counts: Dict[Tuple[int, int, int], int]
    bin_size: Tuple[float, float, float] = DEFAULT_BIN_SIZE
    tau_D: float = field(init=False)

    def __post_init__(self) -> None:
        if not self.counts:
            raise ValueError("offline density counts must be non-empty")
        if any(
            isinstance(count, bool) or not isinstance(count, int)
            or count < 0
            for count in self.counts.values()
        ):
            raise ValueError(
                "offline density counts must be non-negative ints"
            )
        if (
            len(self.bin_size) != 3
            or any(not isinstance(v, (int, float)) or v <= 0
                   for v in self.bin_size)
        ):
            raise ValueError(
                "offline density bin_size must be 3 positive numbers"
            )
        if all(count <= 0 for count in self.counts.values()):
            raise ValueError(
                "offline density counts must contain positive visit counts"
            )
        values = sorted(self.counts.values())
        tau_d = float(np.percentile(values, 20))
        if tau_d <= 0:
            tau_d = float(np.percentile(values, 50))
        if tau_d <= 0:
            raise ValueError(
                "offline density tau_D is 0 — no usable low-density "
                "threshold; counts must contain positive visits"
            )
        object.__setattr__(self, "tau_D", tau_d)

    def is_low_density(self, cell: Tuple[int, int, int]) -> bool:
        """True when the offline visit count is below tau_D."""
        return self.counts.get(cell, 0) < self.tau_D

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "1",
            "bin_size": list(self.bin_size),
            "counts": {
                f"{x},{y},{z}": count
                for (x, y, z), count in self.counts.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OfflineDensity":
        """Schema validation of an offline density JSON (CLI level,
        before preflight). Raises ValueError with a precise message."""
        if not isinstance(data, dict) or data.get("schema_version") != "1":
            raise ValueError(
                "offline density must be a JSON object with "
                "schema_version 1"
            )
        bin_size = data.get("bin_size")
        if (
            not isinstance(bin_size, (list, tuple)) or len(bin_size) != 3
            or any(not isinstance(v, (int, float)) or v <= 0
                   for v in bin_size)
        ):
            raise ValueError(
                "offline density bin_size must be 3 positive numbers"
            )
        counts_raw = data.get("counts")
        if not isinstance(counts_raw, dict) or not counts_raw:
            raise ValueError(
                "offline density counts must be a non-empty object"
            )
        counts: Dict[Tuple[int, int, int], int] = {}
        for key, count in counts_raw.items():
            try:
                x, y, z = (int(part) for part in key.split(","))
            except (ValueError, AttributeError) as e:
                raise ValueError(
                    f"offline density cell key {key!r} must be 'x,y,z'"
                ) from e
            if (
                isinstance(count, bool) or not isinstance(count, int)
                or count < 0
            ):
                raise ValueError(
                    f"offline density count for {key!r} must be an int >= 0"
                )
            counts[(x, y, z)] = count
        return cls(
            counts=counts, bin_size=tuple(float(v) for v in bin_size)
        )

    @classmethod
    def read_offline_density_file(cls, path: str) -> "OfflineDensity":
        """Read + schema-validate an offline density file — the shared
        loader for both entrypoints."""
        density_path = Path(path)
        if not density_path.is_file():
            raise ValueError(f"offline density file not found: {density_path}")
        try:
            with open(density_path, "r") as f:
                data = json.load(f)
        except ValueError as e:
            raise ValueError(
                f"offline density {density_path} is not valid JSON: {e}"
            ) from e
        return cls.from_dict(data)


def _goal_aligned_frame(
    start: Sequence[float], goal: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Goal direction and lateral unit vectors of the goal-aligned frame
    (§3.9): g_hat along the goal line, side_hat perpendicular to it (and
    horizontal for any non-vertical goal line). A degenerate or
    non-finite start-goal vector (start == goal) falls back to the +x
    frame — a zero-norm g_hat would otherwise be NaN and poison every
    cell (L6)."""
    g = np.asarray(goal, dtype=float) - np.asarray(start, dtype=float)
    norm_g = float(np.linalg.norm(g))
    if not np.isfinite(norm_g) or norm_g < 1e-6:
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    g_hat = g / norm_g
    side = np.cross(np.array([0.0, 0.0, 1.0]), g_hat)
    if np.linalg.norm(side) < 1e-6:
        side = np.array([1.0, 0.0, 0.0])
    return g_hat, side / np.linalg.norm(side)


def _d_soft(clearances: np.ndarray, t: int) -> float:
    """d_soft^H(t) — §3.2 temporal soft-min over the next H clearances.

    The window is truncated to the remaining steps, NaN clearances are
    masked out (they must not poison the weighted sum), and the omega
    weights renormalized; a constant clearance c gives exactly c
    (soft-min identity, locked by the tests).
    """
    window = clearances[t:t + H]
    if window.size == 0:
        return float("nan")
    finite = np.isfinite(window)
    window = window[finite]
    if window.size == 0:
        return float("nan")
    i = np.arange(1, window.size + 1)
    weights = np.exp(-OMEGA_DECAY * (i - 1))
    weights = weights / weights.sum()
    return float(-np.log(np.sum(weights * np.exp(-BETA * window))) / BETA)


def _shell_entry(clearances: np.ndarray) -> Optional[int]:
    """First step inside the near-obstacle shell: RAW clearance < d_risk
    (§3.3 d_risk threshold; D2 — not d_soft, so the shell marks the
    actual distance-to-obstacle entry and onset_time keeps its
    semantics)."""
    for t, clearance in enumerate(clearances):
        if np.isfinite(clearance) and clearance < D_RISK:
            return t
    return None


def _side_onset(
    trace: Sequence[Dict[str, Any]],
    start: Sequence[float], goal: Sequence[float],
    g_hat: np.ndarray, side_hat: np.ndarray,
) -> Tuple[bool, int, float]:
    """(has_onset, onset_time, onset_distance) for one trace (§3.4/§5.2).

    The first step BEFORE the shell where the lateral signal
    s(t) = |x_perp| + kappa*|yaw_rate| exceeds the pre-shell baseline
    mean by eta_o — a constant born offset and a constant yaw rate are
    absorbed by the baseline and never count as an onset (§3.4 Delta is
    relative to the low-risk baseline: |yaw_rate| is the per-step
    |yaw[t] - yaw[t-1]|, not the cumulative |yaw - yaw[0]|, whose delta
    would grow linearly under a steady turn). onset_time = shell_step -
    onset_step; onset_distance = the clearance at the onset step — the
    "onset 到障碍 distance" of §5.2 (how far from the obstacle the
    policy began the avoidance), NOT the geometric path length to the
    shell position. A candidate step whose clearance is NaN cannot
    report an onset_distance and is skipped (W1).
    """
    if not trace:
        return False, 0, 0.0
    pos = np.asarray([step["pos"] for step in trace], dtype=float)
    yaw = np.asarray([step["yaw"] for step in trace], dtype=float)
    clearances = np.asarray([step["clearance"] for step in trace], dtype=float)
    shell = _shell_entry(clearances)
    if shell is None or shell == 0:
        return False, 0, 0.0
    x_perp = (pos - np.asarray(start, dtype=float)) @ side_hat
    yaw_rate = np.abs(np.diff(yaw, prepend=yaw[0]))
    s = np.abs(x_perp) + KAPPA * yaw_rate
    baseline = float(np.mean(s[:shell]))
    for t in range(shell):
        if not np.isfinite(clearances[t]):
            continue
        lateral = (
            np.abs(x_perp[t]) > EPSILON_M or yaw_rate[t] > ETA_O / KAPPA
        )
        if lateral and s[t] - baseline > ETA_O:
            return True, shell - t, float(clearances[t])
    return False, 0, 0.0


def _m_sign(x_perp: float) -> int:
    """§3.8 lateral side sign with the eps_m dead-zone."""
    if x_perp > EPSILON_M:
        return 1
    if x_perp < -EPSILON_M:
        return -1
    return 0


def _flip_count(m: Sequence[int]) -> int:
    """Side flips (§3.8): sign changes of the NONZERO m series — brief
    m=0 states are transparent, so +1 -> 0 -> -1 counts as one flip."""
    prev = 0
    flips = 0
    for sign in m:
        if sign != 0:
            if prev != 0 and sign != prev:
                flips += 1
            prev = sign
    return flips


def _trap_recovery(
    trace: Sequence[Dict[str, Any]], m: Sequence[int],
) -> Tuple[List[Tuple[int, int]], List[bool]]:
    """Trap segments and their recovery (§3.7) for one trace.

    m is the per-step lateral side sign (goal-aligned frame from the
    record's start/goal). Returns (segments, recovered): a segment is a
    run of trap-active steps of length >= MIN_TRAP_LEN; a segment is
    recovered when, within RECOVERY_WINDOW steps after its exit, some
    step has W-window progress > EPSILON_REC and d_soft risen by
    > EPSILON_D above the segment-start reference.
    """
    clearances = np.asarray(
        [step["clearance"] for step in trace], dtype=float
    )
    goal_dists = np.asarray([step["goal_dist"] for step in trace], dtype=float)
    d_soft = np.asarray([_d_soft(clearances, t) for t in range(len(trace))])
    n = len(trace)
    trap = [False] * n
    for t in range(n):
        lo = max(0, t - W + 1)
        progress = goal_dists[lo] - goal_dists[t]
        window = clearances[lo:t + 1][np.isfinite(clearances[lo:t + 1])]
        min_clearance = (
            float(np.min(window)) if window.size else float("inf")
        )
        flips = _flip_count(m[lo:t + 1])
        trap[t] = (
            progress < EPSILON_P
            and min_clearance < D_TRAP
            and flips > N_MAX_FLIPS
        )
    segments: List[Tuple[int, int]] = []
    recovered: List[bool] = []
    run_start = None
    for t, active in enumerate(trap + [False]):
        if active and run_start is None:
            run_start = t
        elif not active and run_start is not None:
            if t - run_start >= MIN_TRAP_LEN:
                segments.append((run_start, t - 1))
                recovered.append(_segment_recovered(
                    trace, goal_dists, d_soft, run_start, t - 1))
            run_start = None
    return segments, recovered


def _segment_recovered(
    trace: Sequence[Dict[str, Any]],
    goal_dists: np.ndarray,
    d_soft: np.ndarray,
    seg_start: int, seg_end: int,
) -> bool:
    """Escape check (§3.7): within RECOVERY_WINDOW steps after the
    segment exit, W-window progress > EPSILON_REC and d_soft risen by
    > EPSILON_D above the segment-start reference."""
    n = len(trace)
    reference = d_soft[seg_start]
    for t in range(seg_end + 1, min(seg_end + 1 + RECOVERY_WINDOW, n)):
        lo = max(0, t - W + 1)
        progress = goal_dists[lo] - goal_dists[t]
        if progress > EPSILON_REC and d_soft[t] - reference > EPSILON_D:
            return True
    return False


def _interactions(trace: Sequence[Dict[str, Any]]) -> List[Tuple[int, int]]:
    """Obstacle interactions: contiguous runs of steps with raw clearance
    < d_risk (D7 — the side-flip denominator)."""
    runs: List[Tuple[int, int]] = []
    run_start = None
    for t, step in enumerate(trace):
        clearance = step["clearance"]
        inside = np.isfinite(clearance) and clearance < D_RISK
        if inside and run_start is None:
            run_start = t
        elif not inside and run_start is not None:
            runs.append((run_start, t - 1))
            run_start = None
    if run_start is not None:
        runs.append((run_start, len(trace) - 1))
    return runs


def _anticipatory_side_stats(
    records: Sequence[Any], side: str,
) -> Dict[str, Any]:
    risk_episodes = 0
    onset_episodes = 0
    onset_times: List[float] = []
    onset_distances: List[float] = []
    softmins: List[float] = []
    for record in records:
        trace = record.trace.get(side) or []
        if not trace:
            continue
        clearances = np.asarray(
            [step["clearance"] for step in trace], dtype=float
        )
        if _shell_entry(clearances) is not None:
            risk_episodes += 1
        g_hat, side_hat = _goal_aligned_frame(record.start, record.goal)
        has_onset, onset_time, onset_distance = _side_onset(
            trace, record.start, record.goal, g_hat, side_hat
        )
        if has_onset:
            onset_episodes += 1
            onset_times.append(float(onset_time))
            onset_distances.append(onset_distance)
        for t in range(len(trace)):
            value = _d_soft(clearances, t)
            if np.isfinite(value):
                softmins.append(value)
    return {
        "risk_episodes": risk_episodes,
        "onset_n": onset_episodes,
        "onset_time": float(np.mean(onset_times)) if onset_times else 0.0,
        "onset_distance": (
            float(np.mean(onset_distances)) if onset_distances else 0.0
        ),
        "future_clearance_softmin": (
            float(np.mean(softmins)) if softmins else 0.0
        ),
        "future_clearance_softmin_p10": (
            float(np.percentile(softmins, 10)) if softmins else 0.0
        ),
    }


def _support_side_stats(
    records: Sequence[Any], side: str, density: OfflineDensity,
) -> Dict[str, Any]:
    """Pooled per-side support metrics (§3.9/§5.2).

    Coverage, spans, axis shares and path-length use SUCCESSFUL episodes
    only; the out-of-support time share pools every episode of the role.
    Steps whose position is not finite cannot be attributed to a cell
    and are excluded from every support statistic — a NaN position must
    not crash the cell mapping or poison the spans (L6).
    """
    total_steps = 0
    low_density_steps = 0
    lateral_count = 0
    z_count = 0
    successful_episodes = 0
    coverage: set = set()
    x_parallel_pts: List[float] = []
    x_perp_pts: List[float] = []
    z_pts: List[float] = []
    path_ratios: List[float] = []
    for record in records:
        trace = record.trace.get(side) or []
        if not trace:
            continue
        g_hat, side_hat = _goal_aligned_frame(record.start, record.goal)
        outcome = record.bc if side == "bc" else record.rl
        successful = bool(outcome["success"])
        if successful:
            successful_episodes += 1
        pos = np.asarray([step["pos"] for step in trace], dtype=float)
        finite = np.all(np.isfinite(pos), axis=1)
        pos = pos[finite]
        if pos.size == 0:
            continue
        rel = pos - np.asarray(record.start, dtype=float)
        x_parallel = rel @ g_hat
        x_perp = rel @ side_hat
        z_axis = pos[:, 2]
        z_start = float(record.start[2])
        cells = [
            (
                int(np.floor(x_parallel[t] / density.bin_size[0])),
                int(np.floor(x_perp[t] / density.bin_size[1])),
                int(np.floor(z_axis[t] / density.bin_size[2])),
            )
            for t in range(pos.shape[0])
        ]
        total_steps += pos.shape[0]
        for t, cell in enumerate(cells):
            if not density.is_low_density(cell):
                continue
            low_density_steps += 1
            if successful:
                coverage.add(cell)
            dev_lateral = abs(float(x_perp[t]))
            dev_altitude = abs(float(z_axis[t]) - z_start)
            if dev_lateral >= dev_altitude and dev_lateral > 0:
                lateral_count += 1
            elif dev_altitude > 0:
                z_count += 1
        if successful:
            x_parallel_pts.extend(x_parallel.tolist())
            x_perp_pts.extend(x_perp.tolist())
            z_pts.extend(z_axis.tolist())
            path = float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
            straight = float(np.linalg.norm(
                np.asarray(record.goal, dtype=float)
                - np.asarray(record.start, dtype=float)
            ))
            path_ratios.append(path / straight if straight > 0 else 0.0)
    axes = 0
    if low_density_steps > 0:
        if lateral_count / low_density_steps > AXIS_CONTRIBUTION_THRESHOLD:
            axes += 1
        if z_count / low_density_steps > AXIS_CONTRIBUTION_THRESHOLD:
            axes += 1
    return {
        "successful_episodes": successful_episodes,
        "successful_low_density_coverage": len(coverage),
        "successful_x_parallel_span": _span(x_parallel_pts),
        "successful_x_perp_span": _span(x_perp_pts),
        "successful_z_span": _span(z_pts),
        "altitude_only_ratio": (
            z_count / low_density_steps if low_density_steps else 0.0
        ),
        "low_density_axes_count": axes,
        "out_of_support_rate": (
            low_density_steps / total_steps if total_steps else 0.0
        ),
        "path_length_ratio": (
            float(np.mean(path_ratios)) if path_ratios else 0.0
        ),
    }


def _span(values: Sequence[float]) -> float:
    return float(np.max(values) - np.min(values)) if values else 0.0


def _dense_side_stats(
    records: Sequence[Any], side: str,
) -> Dict[str, Any]:
    segments_total = 0
    recovered_total = 0
    dwells: List[int] = []
    interaction_flips: List[int] = []
    interactions_total = 0
    for record in records:
        trace = record.trace.get(side) or []
        if not trace:
            continue
        pos = np.asarray([step["pos"] for step in trace], dtype=float)
        start = record.start
        side_hat = _goal_aligned_frame(start, record.goal)[1]
        x_perp = (pos - np.asarray(start, dtype=float)) @ side_hat
        m = [_m_sign(float(v)) for v in x_perp]
        segments, recovered = _trap_recovery(trace, m)
        segments_total += len(segments)
        recovered_total += int(sum(recovered))
        dwells.extend(end - start_idx + 1
                      for start_idx, end in segments)
        runs = _interactions(trace)
        interactions_total += len(runs)
        interaction_flips.extend(
            _flip_count(m[lo:hi + 1]) for lo, hi in runs
        )
    return {
        "trap_segments": segments_total,
        "trap_recovery_rate": (
            recovered_total / segments_total if segments_total else 0.0
        ),
        "deadend_dwell_steps_mean": (
            float(np.mean(dwells)) if dwells else 0.0
        ),
        "deadend_dwell_steps_max": (
            float(np.max(dwells)) if dwells else 0.0
        ),
        "side_flip_count_mean": (
            float(np.mean(interaction_flips)) if interaction_flips else 0.0
        ),
        "obstacle_interactions": interactions_total,
    }


def _sr_cr(records: Sequence[Any], side: str) -> Tuple[float, float]:
    n = len(records)
    successes = sum(
        1 for r in records
        if bool((r.bc if side == "bc" else r.rl)["success"])
    )
    collisions = sum(
        1 for r in records
        if bool((r.bc if side == "bc" else r.rl)["collision"])
    )
    return successes / n, collisions / n


def _side_into(out: Dict[str, Any], prefix: str, stats: Dict[str, Any],
               sr: float, cr: float) -> None:
    out[f"sr_{prefix}"] = sr
    out[f"cr_{prefix}"] = cr
    for key, value in stats.items():
        out[f"{key}_{prefix}"] = value


def _anticipatory_metrics(records: Sequence[Any]) -> Dict[str, Any]:
    role_records = [r for r in records if r.role == "anticipatory"]
    out: Dict[str, Any] = {"n_episodes": len(role_records)}
    for side in ("bc", "rl"):
        sr, cr = _sr_cr(role_records, side)
        _side_into(out, side, _anticipatory_side_stats(role_records, side),
                   sr, cr)
    return out


def _support_metrics(
    records: Sequence[Any], density: OfflineDensity,
) -> Dict[str, Any]:
    role_records = [r for r in records if r.role == "support_expansion"]
    out: Dict[str, Any] = {"n_episodes": len(role_records)}
    for side in ("bc", "rl"):
        sr, cr = _sr_cr(role_records, side)
        _side_into(out, side, _support_side_stats(role_records, side,
                                                  density), sr, cr)
    return out


def _dense_metrics(records: Sequence[Any]) -> Dict[str, Any]:
    role_records = [r for r in records if r.role == "dense_recovery"]
    out: Dict[str, Any] = {"n_episodes": len(role_records)}
    for side in ("bc", "rl"):
        sr, cr = _sr_cr(role_records, side)
        _side_into(out, side, _dense_side_stats(role_records, side), sr, cr)
    return out


# The numeric trace axes the capability gate checks for finiteness
# (P1-4 / guide R3.5: NaN is never a valid metric — the old np.isfinite
# masks turned NaN traces into valid-looking numbers). "pos" is the
# 3-value position list, checked component-wise.
_TRACE_FINITE_KEYS = (
    "step", "elapsed", "pos", "z", "yaw", "goal_dist", "clearance",
    "hit_count", "action_norm", "action_max_abs",
)


def _validate_trace_finite(records: Sequence[Any]) -> None:
    """Input gate: any non-finite value in a required trace field is a
    hard fail — the metric layer refuses to mask NaN/Inf into a
    valid-looking metric. Raises ValueError with "finite" in the message.
    """
    for record in records:
        for side in ("bc", "rl"):
            steps = record.trace.get(side) or []
            for k, step in enumerate(steps):
                if not isinstance(step, dict):
                    continue  # structural checks live in the artifact layer
                for key in _TRACE_FINITE_KEYS:
                    if key not in step:
                        continue
                    value = step[key]
                    values = value if key == "pos" else [value]
                    for v in values:
                        try:
                            finite = np.isfinite(float(v))
                        except (TypeError, ValueError):
                            continue  # non-numeric values: artifact layer
                        if not finite:
                            raise ValueError(
                                f"trace[{side}][{k}].{key} is non-finite "
                                f"({v!r}) — a required trace field must be "
                                "finite; NaN/Inf is never a valid metric"
                            )


def compute_capability_metrics(
    records: Sequence[Any],
    offline_density: Optional[OfflineDensity],
) -> Dict[str, Dict[str, float]]:
    """The three capability groups from the paired episode traces (§5.2).

    Hard-fails on: no episodes, offline_density=None, or any of
    the three capability roles (anticipatory / support_expansion /
    dense_recovery) without a single record. P1-4: any non-finite
    required trace field hard-fails here (input gate), never masks.
    """
    _validate_trace_finite(records)
    if not records:
        raise ValueError("no episodes to compute capability metrics from")
    if offline_density is None:
        raise ValueError(
            "offline density is required for capability metrics — the "
            "support group must never return 0 pretending no improvement"
        )
    for role, group in ROLE_OF_CAPABILITY.items():
        if not any(record.role == role for record in records):
            raise ValueError(
                f"no records with role {role!r} for capability group "
                f"{group}"
            )
    return {
        "anticipatory_avoidance": _anticipatory_metrics(records),
        "fullspace_support_expansion": _support_metrics(records,
                                                        offline_density),
        "dense_obstacle_ceiling": _dense_metrics(records),
    }


# ═══════════════════════════════════════════════════════════════════
# R6-EVAL-2 (F-04): capability opportunity schema + five-piece outcome
# contract (Guide §11.5.3/§11.5.5)
# ═══════════════════════════════════════════════════════════════════

# The five pieces every capability group reports — opportunity is a
# frozen bank fact, exposure/event are measured from the evaluated
# episodes, value is null (never 0) when no paired event exists, and
# validity_reason says WHY a null appears.
FIVE_PIECE_KEYS = ("opportunity_count", "exposure_count", "event_count",
                   "value", "validity_reason")

# Provenance allowlists: an opportunity must be a policy-INDEPENDENT
# fact — nominal-path geometry / world topology / offline density.
# Anything derived from the evaluated policy's own trajectory is the
# F-04 counterexample and is rejected, never silently accepted.
ANTICIPATORY_OPPORTUNITY_SOURCES = ("nominal_path_geometry",)
DENSE_OPPORTUNITY_SOURCES = ("world_topology",)
SUPPORT_OPPORTUNITY_SOURCES = ("offline_density",)
POLICY_DEPENDENT_SOURCES = ("evaluated_trace",)

# Enum allowlists for the world-geometry classes.
TOPOLOGY_CLASSES = ("trap", "open", "cluttered")
SUPPORT_STRATA = ("inside", "outside", "boundary")

# validity_reason / gate-reason templates (null-value honesty: an
# opportunity with no evidence is NOT_INFORMATIVE, never a 0 score).
OPPORTUNITY_INVALID_REASONS = {
    "no_opportunity": (
        "no opportunity — absent or zero (opportunity_count=0)"
    ),
    # Gate B R6-EVAL-2: a block present but declared invalid (valid=False)
    # is a DIFFERENT no-evidence state than an absent block — the reason
    # must say "present but invalid", never masquerade as absent.
    "no_valid_opportunity": (
        "no valid opportunity — blocks present but all declared "
        "invalid (valid=False)"
    ),
    "no_exposure": (
        "no exposure — opportunity present but never encountered by "
        "either side (exposure_count=0)"
    ),
    "no_paired_events": (
        "no paired events — exposure without the paired capability "
        "signal on both sides (event_count=0)"
    ),
}


@dataclass(frozen=True)
class CapabilityOpportunity:
    """One FROZEN capability opportunity block (§11.5.3).

    Every field is a policy-independent fact about the bank entry:
    nominal-path geometry (anticipatory group), world topology (dense
    group), offline density (support group). Each group that declares a
    field must name its provenance from the allowlist — an empty or
    unknown provenance with declared fields is rejected; a provenance
    derived from the evaluated trajectory ("evaluated_trace") is the
    F-04 counterexample and is REJECTED, never silently accepted.
    """

    valid: bool
    # anticipatory group: does the nominal path cross the d_risk shell?
    nominal_path_risk: float = 0.0
    risk_shell_crossing: bool = False
    anticipatory_provenance: str = ""
    # dense group: does the world topology carry trap opportunities?
    topology_class: str = ""
    trap_opportunity_count: int = 0
    dense_provenance: str = ""
    # support group: where does the nominal path sit vs offline support?
    support_stratum: str = ""
    reference_density: float = 0.0
    support_provenance: str = ""

    def __post_init__(self) -> None:
        validate_opportunity(self)


def _group_declares(op: CapabilityOpportunity, group: str) -> bool:
    """A group DECLARES an opportunity when any of its fields is set —
    declared groups must name their provenance (never silent)."""
    if group == "anticipatory":
        return bool(op.risk_shell_crossing) or op.nominal_path_risk != 0.0
    if group == "dense":
        return bool(op.topology_class) or op.trap_opportunity_count != 0
    if group == "support":
        return bool(op.support_stratum) or op.reference_density != 0.0
    return False


def _validate_provenance(value: str, allowlist: Sequence[str],
                         group: str) -> None:
    """Provenance gate for one group. Empty provenance is fine when the
    group declares nothing (its fields are at the empty defaults); a
    declared group without provenance, an unknown provenance, or a
    policy-dependent provenance all hard-fail — never a silent default.
    """
    if not value:
        return  # group not declared (or a bare provenance-only block)
    if value in allowlist:
        return
    if value in POLICY_DEPENDENT_SOURCES:
        raise ValueError(
            f"opportunity {group} provenance {value!r} is "
            "policy-dependent — the opportunity must come from frozen "
            "geometry / nominal path / offline density, never from the "
            "evaluated policy's trajectory"
        )
    raise ValueError(
        f"opportunity {group} provenance {value!r} not in the allowlist "
        f"{allowlist} — an opportunity without auditable provenance "
        "cannot gate a capability"
    )


def validate_opportunity(op: CapabilityOpportunity) -> None:
    """Strict opportunity validation (§11.5.3) — called on construction
    and again by opportunity_from_dict. Raises ValueError with a precise
    message; every violation is a hard fail, never a silent default.
    """
    if not isinstance(op.valid, bool):
        raise ValueError(
            f"opportunity valid must be a bool, got {op.valid!r}"
        )
    for name, value in (("nominal_path_risk", op.nominal_path_risk),
                        ("reference_density", op.reference_density)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"opportunity {name} must be a finite number, got "
                f"{value!r}"
            )
        if not np.isfinite(float(value)):
            raise ValueError(
                f"opportunity {name} must be finite, got {value!r}"
            )
    count = op.trap_opportunity_count
    if isinstance(count, bool) or not isinstance(count, int):
        raise ValueError(
            "opportunity trap_opportunity_count must be an int, not a "
            f"bool, got {count!r}"
        )
    if count < 0:
        raise ValueError(
            f"opportunity trap_opportunity_count must be non-negative, "
            f"got {count!r}"
        )
    if op.topology_class and op.topology_class not in TOPOLOGY_CLASSES:
        raise ValueError(
            f"opportunity topology_class {op.topology_class!r} not in "
            f"{TOPOLOGY_CLASSES}"
        )
    if op.support_stratum and op.support_stratum not in SUPPORT_STRATA:
        raise ValueError(
            f"opportunity support_stratum {op.support_stratum!r} not in "
            f"{SUPPORT_STRATA}"
        )
    for group, declared in (
        ("anticipatory", _group_declares(op, "anticipatory")),
        ("dense", _group_declares(op, "dense")),
        ("support", _group_declares(op, "support")),
    ):
        if declared and not (
            (group == "anticipatory" and op.anticipatory_provenance)
            or (group == "dense" and op.dense_provenance)
            or (group == "support" and op.support_provenance)
        ):
            raise ValueError(
                f"opportunity {group} provenance is required when the "
                "group declares opportunity fields — a declared "
                "opportunity without provenance is not auditable"
            )
    _validate_provenance(op.anticipatory_provenance,
                         ANTICIPATORY_OPPORTUNITY_SOURCES, "anticipatory")
    _validate_provenance(op.dense_provenance, DENSE_OPPORTUNITY_SOURCES,
                         "dense")
    _validate_provenance(op.support_provenance, SUPPORT_OPPORTUNITY_SOURCES,
                         "support")


def opportunity_to_dict(op: CapabilityOpportunity) -> Dict[str, Any]:
    """Lossless opportunity serialization (bank entry block)."""
    return {
        "valid": op.valid,
        "nominal_path_risk": float(op.nominal_path_risk),
        "risk_shell_crossing": op.risk_shell_crossing,
        "anticipatory_provenance": op.anticipatory_provenance,
        "topology_class": op.topology_class,
        "trap_opportunity_count": int(op.trap_opportunity_count),
        "dense_provenance": op.dense_provenance,
        "support_stratum": op.support_stratum,
        "reference_density": float(op.reference_density),
        "support_provenance": op.support_provenance,
    }


def opportunity_from_dict(data: Any) -> CapabilityOpportunity:
    """Schema validation of an opportunity block (bank load). Missing
    fields fall back to the empty defaults (a partial block for a group
    the entry does not declare is valid); present fields are validated
    strictly — an invalid block hard-fails the bank load, never a
    silent default.
    """
    if not isinstance(data, dict):
        raise ValueError(
            "capability opportunity must be a JSON object, got "
            f"{data!r}"
        )
    return CapabilityOpportunity(
        valid=data.get("valid", True),
        nominal_path_risk=data.get("nominal_path_risk", 0.0),
        risk_shell_crossing=data.get("risk_shell_crossing", False),
        anticipatory_provenance=data.get("anticipatory_provenance", ""),
        topology_class=data.get("topology_class", ""),
        trap_opportunity_count=data.get("trap_opportunity_count", 0),
        dense_provenance=data.get("dense_provenance", ""),
        support_stratum=data.get("support_stratum", ""),
        reference_density=data.get("reference_density", 0.0),
        support_provenance=data.get("support_provenance", ""),
    )


# ═══════════════════════════════════════════════════════════════════
# R7-8: policy-INDEPENDENT geometry opportunity annotation (guide
# Task R7-8 Step 1/2). The frozen bank's opportunity labels come ONLY
# from nominal-path geometry / world topology / offline density —
# never from any evaluated RL trajectory (F-04, provenance allowlists
# above). Every rule is deterministic, auditable and fail-closed on
# malformed input: a bad obstacle record raises, it never silently
# becomes "no opportunity".
# ═══════════════════════════════════════════════════════════════════

NOMINAL_PATH_SAMPLES = 32
TRAP_TRIANGLE_LIMIT = 5
CLUTTER_THRESHOLD = 5


def _obstacle_surface_clearance_2d(obstacle: Dict[str, Any],
                                   point: Sequence[float]) -> float:
    """2D surface distance from a point to one obstacle.

    Cylinders: center distance minus radius (a point inside the disc
    gives a negative value -> shell crossing). Boxes: the point is
    rotated into the box frame (yaw) and the distance to the axis-
    aligned rectangle is computed (0 inside). 3D is intentionally
    ignored — the UAV flies inside the obstacle height band.

    Raises:
        ValueError: record without a known shape / finite coordinates —
            a malformed world must never silently become "no
            opportunity".
    """
    if not isinstance(obstacle, dict):
        raise ValueError(
            f"obstacle record must be a dict, got {obstacle!r}"
        )
    shape = obstacle.get("shape")
    try:
        x, y = float(obstacle["x"]), float(obstacle["y"])
        px, py = float(point[0]), float(point[1])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(
            f"obstacle record missing finite x/y: {obstacle!r}"
        ) from e
    dx, dy = px - x, py - y
    if shape == "cylinder":
        radius = obstacle.get("radius")
        if isinstance(radius, bool) or not isinstance(radius, (int, float)):
            raise ValueError(
                f"cylinder obstacle missing finite radius: {obstacle!r}"
            )
        return float(np.hypot(dx, dy) - float(radius))
    if shape == "box":
        try:
            size_x, size_y = float(obstacle["size_x"]), float(
                obstacle["size_y"])
            yaw = float(obstacle.get("yaw", 0.0))
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(
                f"box obstacle missing finite size_x/size_y: {obstacle!r}"
            ) from e
        cos_t, sin_t = np.cos(-yaw), np.sin(-yaw)
        qx, qy = dx * cos_t - dy * sin_t, dx * sin_t + dy * cos_t
        hx, hy = size_x / 2.0, size_y / 2.0
        ox, oy = max(abs(qx) - hx, 0.0), max(abs(qy) - hy, 0.0)
        return float(np.hypot(ox, oy))
    raise ValueError(
        f"obstacle record shape {shape!r} not in ('cylinder', 'box'): "
        f"{obstacle!r}"
    )


def _nominal_path_samples(start: Sequence[float],
                          goal: Sequence[float]) -> List[np.ndarray]:
    """Uniformly sampled points on the start->goal line (endpoints
    included) — the policy-independent nominal path. A degenerate
    (zero-length) path still yields valid samples (all identical)."""
    start_a = np.asarray(start, dtype=float)
    goal_a = np.asarray(goal, dtype=float)
    return [
        start_a + (goal_a - start_a) * (i / (NOMINAL_PATH_SAMPLES - 1))
        for i in range(NOMINAL_PATH_SAMPLES)
    ]


def _point_in_triangle_2d(p: Sequence[float], a: Sequence[float],
                          b: Sequence[float],
                          c: Sequence[float]) -> bool:
    """Barycentric point-in-triangle test (2D, boundary inclusive)."""
    px, py = float(p[0]), float(p[1])
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    cx, cy = float(c[0]), float(c[1])
    area2 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    if abs(area2) < 1e-12:
        return False  # degenerate triangle never encloses anything
    u = ((bx - ax) * (py - ay) - (by - ay) * (px - ax)) / area2
    v = ((cx - bx) * (py - by) - (cy - by) * (px - bx)) / area2
    w = 1.0 - u - v
    eps = 1e-9
    return u >= -eps and v >= -eps and w >= -eps


def _dense_topology(obstacles: Sequence[Dict[str, Any]],
                    start: Sequence[float],
                    goal: Sequence[float]) -> Tuple[str, int]:
    """World-topology trap count and class.

    A trap opportunity is an obstacle-center triangle enclosing start or
    goal (the waypoint is surrounded by obstacles — the recovery
    setting the dense group measures). Counts up to TRAP_TRIANGLE_LIMIT
    (deterministic cap; the count is a label, not a budget). Class:
    trap when any triangle encloses, cluttered when the world is dense
    (>= CLUTTER_THRESHOLD obstacles), open otherwise.
    """
    centers = [
        (float(obs["x"]), float(obs["y"])) for obs in obstacles
        if isinstance(obs, dict) and "x" in obs and "y" in obs
    ]
    if len(centers) < 3:
        return "cluttered" if len(obstacles) >= CLUTTER_THRESHOLD else "open", 0
    count = 0
    for a, b, c in itertools.combinations(centers, 3):
        if count >= TRAP_TRIANGLE_LIMIT:
            break
        if (_point_in_triangle_2d(start, a, b, c)
                or _point_in_triangle_2d(goal, a, b, c)):
            count += 1
    if count > 0:
        return "trap", count
    return "cluttered" if len(centers) >= CLUTTER_THRESHOLD else "open", 0


def _support_stratum(density: OfflineDensity,
                     samples: Sequence[np.ndarray]) -> Tuple[str, float]:
    """Offline-density support label for the nominal path.

    Each sample maps to its cell with the SAME mapping the offline
    density was built with (reward_v13.compute_support_cell — goal-unit
    frame, round indexing), so the annotation's cells and the density
    counts' cells always agree. reference_density is the fraction of
    samples inside low-density cells (counts below tau_D). Stratum:
    inside when any sample is in a low-density cell; boundary when only
    the 8-neighbourhood of the path touches low-density cells; outside
    otherwise.
    """
    low_flags: List[bool] = []
    neighbour_low: List[bool] = []
    goal = np.asarray(samples[-1], dtype=float)
    for sample in samples:
        cell = compute_support_cell(sample, goal, density.bin_size)
        low_flags.append(density.is_low_density(cell))
        neighbour_low.append(any(
            density.is_low_density(
                (cell[0] + i, cell[1] + j, cell[2]))
            for i in (-1, 0, 1) for j in (-1, 0, 1)
            if (i, j) != (0, 0)
        ))
    if any(low_flags):
        return "inside", float(np.mean(low_flags))
    if any(neighbour_low):
        return "boundary", 0.0
    return "outside", 0.0


def annotate_geometry_opportunity(
    obstacles: Sequence[Dict[str, Any]],
    start: Sequence[float],
    goal: Sequence[float],
    offline_density: Optional[OfflineDensity] = None,
) -> CapabilityOpportunity:
    """Frozen, policy-INDEPENDENT geometry opportunity for a bank entry.

    Anticipatory: the nominal path's minimum surface clearance vs the
    d_risk shell (risk_shell_crossing, nominal_path_risk = 1 - c/d_risk,
    provenance nominal_path_geometry). Dense: trap triangles around
    start/goal (provenance world_topology). Support: nominal-path
    samples vs the offline low-density region — only when the caller
    provides the real density (provenance offline_density); without it
    the support group stays undecleared, never faked.

    Returns:
        A validated CapabilityOpportunity (never policy-dependent).
    """
    samples = _nominal_path_samples(start, goal)
    clearance = min(
        (_obstacle_surface_clearance_2d(obs, s) for obs in obstacles
         for s in samples),
        default=float("inf"),
    )
    crossing = bool(np.isfinite(clearance) and clearance < D_RISK)
    topology_class, trap_count = _dense_topology(obstacles, start, goal)
    if offline_density is not None:
        stratum, reference_density = _support_stratum(
            offline_density, samples)
        support_provenance = "offline_density"
    else:
        stratum, reference_density = "", 0.0
        support_provenance = ""
    return CapabilityOpportunity(
        valid=True,
        nominal_path_risk=(
            min(1.0, max(0.0, 1.0 - clearance / D_RISK))
            if crossing else 0.0),
        risk_shell_crossing=crossing,
        anticipatory_provenance=(
            "nominal_path_geometry" if crossing else ""),
        topology_class=topology_class,
        trap_opportunity_count=trap_count,
        dense_provenance="world_topology",
        support_stratum=stratum,
        reference_density=reference_density,
        support_provenance=support_provenance,
    )


def _five_piece(
    opportunity_count: int, exposure_count: int, event_count: int,
    value: Optional[float], reason: Optional[str],
) -> Dict[str, Any]:
    return {
        "opportunity_count": int(opportunity_count),
        "exposure_count": int(exposure_count),
        "event_count": int(event_count),
        "value": value,
        "validity_reason": reason,
    }


def _no_opportunity_reason(
    role_records: Sequence[Any], opportunity_by_entry: Dict[str, Any],
) -> str:
    """Why is opportunity_count 0 — absent blocks or blocks all invalid?
    Gate B R6-EVAL-2: a block present but declared invalid (valid=False)
    is a DISTINCT no-evidence state and must say so, never masquerade as
    absent (which would misread a frozen-bank invalidation as a bank
    without any opportunity block)."""
    if any(
        opportunity_by_entry.get(
            r.episode_config.get("episode_config_hash", "")) is not None
        for r in role_records
    ):
        return OPPORTUNITY_INVALID_REASONS["no_valid_opportunity"]
    return OPPORTUNITY_INVALID_REASONS["no_opportunity"]


def _side_clearances(trace: Sequence[Dict[str, Any]]) -> np.ndarray:
    return np.asarray([step["clearance"] for step in trace], dtype=float)


def _anticipatory_outcomes(
    records: Sequence[Any], opportunity_by_entry: Dict[str, Any],
) -> Dict[str, Any]:
    """Five pieces for anticipatory_avoidance.

    opportunity: entry whose frozen block declares risk_shell_crossing
    AND is valid (Gate B R6-EVAL-2: valid=False 的 block 不得计入);
    exposure: paired episode entering the d_risk shell on EITHER side;
    event: BOTH sides showing a lateral onset; value: per-event paired
    onset-distance delta (rl_onset_distance - bc_onset_distance — RL
    avoiding earlier is a positive delta), mean over events.
    """
    role_records = [r for r in records if r.role == "anticipatory"]
    opp_records = []
    for r in role_records:
        op = opportunity_by_entry.get(
            r.episode_config.get("episode_config_hash", "")
        )
        if op is not None and op.valid and op.risk_shell_crossing:
            opp_records.append(r)
    opportunity_count = len(opp_records)
    if opportunity_count == 0:
        return _five_piece(
            0, 0, 0, None,
            _no_opportunity_reason(role_records, opportunity_by_entry))
    exposed = []
    for r in opp_records:
        bc_shell = _shell_entry(_side_clearances(r.trace.get("bc") or []))
        rl_shell = _shell_entry(_side_clearances(r.trace.get("rl") or []))
        if bc_shell is not None or rl_shell is not None:
            exposed.append(r)
    exposure_count = len(exposed)
    deltas: List[float] = []
    for r in exposed:
        g_hat, side_hat = _goal_aligned_frame(r.start, r.goal)
        bc_has, _, bc_dist = _side_onset(
            r.trace.get("bc") or [], r.start, r.goal, g_hat, side_hat)
        rl_has, _, rl_dist = _side_onset(
            r.trace.get("rl") or [], r.start, r.goal, g_hat, side_hat)
        if bc_has and rl_has:
            deltas.append(float(rl_dist) - float(bc_dist))
    event_count = len(deltas)
    if event_count == 0:
        reason = (OPPORTUNITY_INVALID_REASONS["no_exposure"]
                  if exposure_count == 0
                  else OPPORTUNITY_INVALID_REASONS["no_paired_events"])
        return _five_piece(opportunity_count, exposure_count, 0, None,
                           reason)
    return _five_piece(
        opportunity_count, exposure_count, event_count,
        float(np.mean(deltas)), None)


def _record_dense_stats(
    trace: Sequence[Dict[str, Any]], record: Any,
) -> Tuple[int, int, int]:
    """(segments, recovered, interactions) for ONE trace, reusing the
    §3.7/§5.2 helpers (_trap_recovery/_interactions with the goal-aligned
    lateral sign)."""
    if not trace:
        return 0, 0, 0
    pos = np.asarray([step["pos"] for step in trace], dtype=float)
    side_hat = _goal_aligned_frame(record.start, record.goal)[1]
    x_perp = (pos - np.asarray(record.start, dtype=float)) @ side_hat
    m = [_m_sign(float(v)) for v in x_perp]
    segments, recovered = _trap_recovery(trace, m)
    return len(segments), int(sum(recovered)), len(_interactions(trace))


def _dense_outcomes(
    records: Sequence[Any], opportunity_by_entry: Dict[str, Any],
) -> Dict[str, Any]:
    """Five pieces for dense_obstacle_ceiling.

    opportunity: entry with a dense topology opportunity AND valid —
    (Gate B R6-EVAL-2: valid=False 的 block 不得计入); dense topology
    is trap OR cluttered (user ruling 2026-08-24 — the dense worlds'
    corridor/clutter layouts rarely form trap triangles, so a
    trap-only counter never reaches the R7-8 Step-2 >= 20 quota);
    exposure: paired episode with obstacle interactions on EITHER
    side; event: BOTH sides with at least one trap segment; value:
    per-event paired trap recovery-rate delta (rl_rate - bc_rate),
    mean over events. A dense opportunity with zero observed trap
    segments is null + NOT_INFORMATIVE (§11.5.5), never a 0 score.
    """
    role_records = [r for r in records if r.role == "dense_recovery"]
    opp_records = []
    for r in role_records:
        op = opportunity_by_entry.get(
            r.episode_config.get("episode_config_hash", "")
        )
        if (
            op is not None and op.valid
            and (
                op.trap_opportunity_count > 0
                or op.topology_class == "cluttered"
            )
        ):
            opp_records.append(r)
    opportunity_count = len(opp_records)
    if opportunity_count == 0:
        return _five_piece(
            0, 0, 0, None,
            _no_opportunity_reason(role_records, opportunity_by_entry))
    exposed = []
    for r in opp_records:
        bc_stats = _record_dense_stats(r.trace.get("bc") or [], r)
        rl_stats = _record_dense_stats(r.trace.get("rl") or [], r)
        if bc_stats[2] > 0 or rl_stats[2] > 0:
            exposed.append(r)
    exposure_count = len(exposed)
    deltas: List[float] = []
    for r in exposed:
        bc_seg, bc_rec, _ = _record_dense_stats(r.trace.get("bc") or [], r)
        rl_seg, rl_rec, _ = _record_dense_stats(r.trace.get("rl") or [], r)
        if bc_seg > 0 and rl_seg > 0:
            bc_rate = bc_rec / bc_seg if bc_seg else 0.0
            rl_rate = rl_rec / rl_seg if rl_seg else 0.0
            deltas.append(float(rl_rate) - float(bc_rate))
    event_count = len(deltas)
    if event_count == 0:
        reason = (OPPORTUNITY_INVALID_REASONS["no_exposure"]
                  if exposure_count == 0
                  else OPPORTUNITY_INVALID_REASONS["no_paired_events"])
        return _five_piece(opportunity_count, exposure_count, 0, None,
                           reason)
    return _five_piece(
        opportunity_count, exposure_count, event_count,
        float(np.mean(deltas)), None)


def _out_of_support_share(
    trace: Sequence[Dict[str, Any]], record: Any, density: OfflineDensity,
) -> float:
    """Per-side out-of-support step share — the SAME cell mapping as
    _support_side_stats (§3.9): goal-aligned x_parallel/x_perp from
    (pos - start), z in world frame, cell size density.bin_size; a cell
    is out-of-support when offline visits < tau_D. Non-finite positions
    are excluded from both the denominator and the numerator (L6)."""
    if not trace:
        return 0.0
    pos = np.asarray([step["pos"] for step in trace], dtype=float)
    finite = np.all(np.isfinite(pos), axis=1)
    pos = pos[finite]
    if pos.size == 0:
        return 0.0
    g_hat, side_hat = _goal_aligned_frame(record.start, record.goal)
    rel = pos - np.asarray(record.start, dtype=float)
    x_parallel = rel @ g_hat
    x_perp = rel @ side_hat
    z_axis = pos[:, 2]
    low_density_steps = 0
    for t in range(pos.shape[0]):
        cell = (
            int(np.floor(x_parallel[t] / density.bin_size[0])),
            int(np.floor(x_perp[t] / density.bin_size[1])),
            int(np.floor(z_axis[t] / density.bin_size[2])),
        )
        if density.is_low_density(cell):
            low_density_steps += 1
    return low_density_steps / pos.shape[0]


def _support_outcomes(
    records: Sequence[Any], opportunity_by_entry: Dict[str, Any],
    density: OfflineDensity,
) -> Dict[str, Any]:
    """Five pieces for fullspace_support_expansion.

    opportunity: entry whose frozen block puts the nominal path outside
    the offline support (support_stratum == "outside") AND is valid (Gate
    B R6-EVAL-2: valid=False 的 block 不得计入); exposure/event:
    paired episode with out-of-support steps on EITHER side (the
    expansion signal); value: per-event paired out-of-support share
    delta (bc_share - rl_share — RL expanding into out-of-support space
    is a negative delta), mean over events.
    """
    role_records = [r for r in records if r.role == "support_expansion"]
    opp_records = []
    for r in role_records:
        op = opportunity_by_entry.get(
            r.episode_config.get("episode_config_hash", "")
        )
        if op is not None and op.valid and op.support_stratum == "outside":
            opp_records.append(r)
    opportunity_count = len(opp_records)
    if opportunity_count == 0:
        return _five_piece(
            0, 0, 0, None,
            _no_opportunity_reason(role_records, opportunity_by_entry))
    shares = []
    for r in opp_records:
        bc_share = _out_of_support_share(r.trace.get("bc") or [], r, density)
        rl_share = _out_of_support_share(r.trace.get("rl") or [], r, density)
        if bc_share > 0 or rl_share > 0:
            shares.append((bc_share, rl_share))
    exposure_count = len(shares)
    if exposure_count == 0:
        return _five_piece(
            opportunity_count, 0, 0, None,
            OPPORTUNITY_INVALID_REASONS["no_exposure"])
    deltas = [bc - rl for bc, rl in shares]
    return _five_piece(
        opportunity_count, exposure_count, len(deltas),
        float(np.mean(deltas)), None)


def compute_capability_outcomes(
    records: Sequence[Any],
    opportunity_by_entry: Dict[str, Any],
    offline_density: Optional[OfflineDensity],
) -> Dict[str, Dict[str, Any]]:
    """The five-piece outcome contract (§11.5.3) for every capability
    group.

    opportunity_by_entry maps episode_config_hash -> CapabilityOpportunity
    (the frozen bank facts). The outcome layer reuses the existing
    §3.2-§3.9 signal helpers (_shell_entry/_side_onset/_interactions/
    _trap_recovery/is_low_density) — no new signal semantics are
    invented. Value is the per-event paired delta mean; a group without
    events reports value=None + validity_reason (null, never a 0 score).
    """
    _validate_trace_finite(records)
    if offline_density is None:
        raise ValueError(
            "offline density is required for capability outcomes — the "
            "support group must never return 0 pretending no improvement"
        )
    return {
        "anticipatory_avoidance": _anticipatory_outcomes(
            records, opportunity_by_entry),
        "fullspace_support_expansion": _support_outcomes(
            records, opportunity_by_entry, offline_density),
        "dense_obstacle_ceiling": _dense_outcomes(
            records, opportunity_by_entry),
    }


def capability_informativeness(
    out: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, str]]:
    """Gate verdict per capability group (§11.5.3).

    opportunity_count == 0 => NOT_INFORMATIVE — the bank carries no
    evidence for this capability, which is never a 0 score. Any
    positive opportunity count keeps the gate INFORMATIVE even when
    exposure is 0 (the zero is reported explicitly in the five pieces,
    not hidden).

    Gate B R6-EVAL-2: the NOT_INFORMATIVE reason carries the group's
    OWN validity_reason — "absent" vs "present but invalid (valid=False)"
    are distinct no-evidence states and the verdict must not flatten
    them into one template.
    """
    verdicts: Dict[str, Dict[str, str]] = {}
    for group, pieces in out.items():
        if pieces["opportunity_count"] == 0:
            verdicts[group] = {
                "status": "NOT_INFORMATIVE",
                "reason": (
                    pieces["validity_reason"]
                    or OPPORTUNITY_INVALID_REASONS["no_opportunity"]
                ),
            }
        else:
            verdicts[group] = {
                "status": "INFORMATIVE",
                "reason": (
                    f"opportunity present (opportunity_count="
                    f"{pieces['opportunity_count']}); exposure_count="
                    f"{pieces['exposure_count']}; event_count="
                    f"{pieces['event_count']}"
                ),
            }
    return verdicts
