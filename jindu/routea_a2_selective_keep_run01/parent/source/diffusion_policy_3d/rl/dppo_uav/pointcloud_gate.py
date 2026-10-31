"""Point-cloud validity gate (design §4.7).

Kills the degenerate no-ray branch: SynthPointCloud no-ray fallback returned
4096 far-plane placeholder points (synth_pointcloud.py:489-500) that were
indistinguishable from real observations and poisoned training (883,575
no-ray warnings in the E040 log). All frames now carry explicit validity
status; invalid frames are routed per the 5 handling rules.
"""

from collections import deque
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.validated_synth_adapter import (
    RayGeometryReport,
    frame_hit_count,
    frame_sha256,
)


class InvalidFramePolicy(Enum):
    """Routing decision for an invalid frame (§4.7 rules 1-4)."""

    REJECT_EPISODE = "reject_episode"  # rule 1: reset/admission phase
    EXCLUDE_TRANSITION = "exclude_transition"  # rule 2: rollout, non-first
    SAFE_END_EPISODE = "safe_end_episode"  # rule 3: first fault of episode
    HARD_FAILURE = "hard_failure"  # rule 4: rolling rate exceeded


@dataclass(frozen=True)
class PointCloudFrame:
    """Explicit frame validity result — the synthesis interface contract."""

    points: np.ndarray  # (4096, 3), float32
    valid: bool
    invalid_reason: Optional[str]
    hit_count: int
    unique_point_count: int
    finite_ratio: float
    range_min: float
    range_max: float


# A frame validity record IS the per-frame validity type (R7-5 window gate).
PointCloudValidity = PointCloudFrame


@dataclass(frozen=True)
class ObservationWindowValidity:
    """Validity of the full latest-``obs_steps`` policy window (R7-5).

    ``frame_results`` are oldest→newest (index 0 = history[-obs_steps],
    index obs_steps-1 = newest). Any invalid frame invalidates the whole
    window — the caller ends the episode as observation_fault (guide
    exception matrix). ``source_manifest_hash`` / ``runtime_episode_hash``
    echo the production mapping's provenance ("" on the plain-sequence
    test path).
    """

    valid: bool
    obs_steps: int
    frame_results: Tuple[PointCloudValidity, ...]
    invalid_frame_indices: Tuple[int, ...]
    source_manifest_hash: str
    runtime_episode_hash: str


# Four-stage gate configurations (§4.7 table).
# unit: expected-valid poses must yield 0 invalid; expected-invalid must error
# explicitly. calibration: >=10k frames, invalid=0 before training.
# pilot: rolling 10k window, invalid rate=0, any invalid -> pause diagnosis.
# formal: count and archive EVERY frame — the §4.7 table authorizes no rate
# gate here (review C2): rate=None means hard_failure() never trips.
PHASE_GATES: Dict[str, Dict[str, Optional[float]]] = {
    "unit": {"rolling_window": 10000, "rolling_max_invalid_rate": 0.0},
    "calibration": {"rolling_window": 10000, "rolling_max_invalid_rate": 0.0},
    "pilot": {"rolling_window": 10000, "rolling_max_invalid_rate": 0.0},
    "formal": {"rolling_window": 100000, "rolling_max_invalid_rate": None},
}


class PointCloudValidityGate:
    """Validates frames and routes invalid frames per §4.7 rules 1-5.

    Rule 5 (no silent stale substitution) is enforced by API shape: validate()
    accepts only the current frame; there is no prev-frame fallback parameter.
    """

    def __init__(
        self,
        contract: RLSystemContract,
        phase: str = "unit",
        n_unique_min: int = 2,  # uncalibrated floor: unique_point_count > 1
        rolling_window: Optional[int] = None,
        rolling_max_invalid_rate: Optional[float] = None,
        min_samples_for_rate: Optional[int] = None,
    ) -> None:
        assert phase in PHASE_GATES, f"unknown phase {phase!r}"
        assert n_unique_min >= 2, "N_unique_min floor is > 1 before calibration"
        self.contract = contract
        self.phase = phase
        self.n_unique_min = n_unique_min
        gate = PHASE_GATES[phase]
        self.rolling_window = (
            int(gate["rolling_window"])
            if rolling_window is None
            else rolling_window
        )
        self.rolling_max_invalid_rate = (
            gate["rolling_max_invalid_rate"]
            if rolling_max_invalid_rate is None
            else rolling_max_invalid_rate
        )
        # Rate gate engages only once the rolling window is (near-)full,
        # so a single early invalid frame routes to EXCLUDE_TRANSITION
        # instead of instantly hard-failing (§4.7 rule 2 vs rule 4).
        self.min_samples_for_rate = (
            self.rolling_window
            if min_samples_for_rate is None
            else min_samples_for_rate
        )
        self._total_frames = 0
        self._invalid_count = 0
        self._recent: Deque[bool] = deque()  # valid flags, rolling window

    def validate(
        self,
        points: np.ndarray,
        hit_count: int,
        cam_far: float,
        ray_report: Optional[RayGeometryReport] = None,
    ) -> PointCloudFrame:
        """Validate one frame against the hard conditions (§4.7).

        NO-RAY-SEMANTICS: hit_count=0 (every ray's nearest hit beyond
        the camera far clip) is valid ONLY when the caller supplies a
        passing analytic ray-geometry report — the physical far-endpoint
        encoding of ValidatedSynthAdapter. Without analytic evidence a
        far-only frame is indistinguishable from the legacy synth fallback
        (synth_pointcloud.py:489-500) and stays invalid. The single-point
        all-far collapse remains ALWAYS invalid via the unique-count floor
        and the all-equal-far check below — a report cannot rescue it.
        """
        assert points.ndim == 2 and points.shape[1] == 3, (
            f"points must be (N, 3), got {points.shape}"
        )
        finite = np.isfinite(points)
        finite_ratio = float(finite.all(axis=1).mean())
        unique_count = len(np.unique(points, axis=0))

        reasons: List[str] = []
        if points.shape[0] != self.contract.point_count:
            reasons.append(
                f"shape {points.shape[0]} != point_count"
                f" {self.contract.point_count}"
            )
        if finite_ratio != 1.0:
            reasons.append(f"finite_ratio {finite_ratio:.4f} != 1.0")
        if hit_count <= 0 and (ray_report is None or not ray_report.passes):
            reasons.append(
                "hit_count <= 0 (no analytic ray-geometry evidence)"
            )
        if unique_count < self.n_unique_min:
            reasons.append(f"unique_point_count {unique_count} < {self.n_unique_min}")
        if np.allclose(points, cam_far):
            reasons.append(f"all points equal camera far plane ({cam_far})")

        valid = not reasons
        invalid_reason = "; ".join(reasons) if reasons else None
        frame = PointCloudFrame(
            points=points.astype(np.float32),
            valid=valid,
            invalid_reason=invalid_reason,
            hit_count=int(hit_count),
            unique_point_count=unique_count,
            finite_ratio=finite_ratio,
            range_min=float(np.nanmin(points)),
            range_max=float(np.nanmax(points)),
        )
        self.record(frame)  # every frame counted (formal phase archives via summary)
        return frame

    def decide(
        self,
        frame: PointCloudFrame,
        phase: str,
        episode_id: str,
        episode_first_fault: bool,
    ) -> InvalidFramePolicy:
        """Route an INVALID frame per rules 1-4 (valid frames need no policy)."""
        assert not frame.valid, "decide() is for invalid frames only"
        assert phase in ("reset", "rollout"), f"unknown phase {phase!r}"
        if phase == "reset":
            return InvalidFramePolicy.REJECT_EPISODE  # rule 1
        if episode_first_fault:
            return InvalidFramePolicy.SAFE_END_EPISODE  # rule 3 (no collision/reach)
        if self.hard_failure():
            return InvalidFramePolicy.HARD_FAILURE  # rule 4
        return InvalidFramePolicy.EXCLUDE_TRANSITION  # rule 2

    def record(self, frame: PointCloudFrame) -> None:
        """Count a frame (formal phase archives via summary())."""
        self._total_frames += 1
        if not frame.valid:
            self._invalid_count += 1
        self._recent.append(frame.valid)
        if len(self._recent) > self.rolling_window:
            self._recent.popleft()

    def rolling_invalid_rate(self) -> float:
        if not self._recent:
            return 0.0
        return 1.0 - (sum(self._recent) / len(self._recent))

    def hard_failure(self) -> bool:
        """Rule 4: rolling invalid rate exceeds the gate (window near-full).

        Phases without a rate gate (formal: count+archive only, §4.7 table)
        never trip hard failure — the caller owns formal-phase diagnosis via
        the archived per-frame data.
        """
        if self.rolling_max_invalid_rate is None:
            return False
        if len(self._recent) < self.min_samples_for_rate:
            return False
        return self.rolling_invalid_rate() > self.rolling_max_invalid_rate

    def should_pause(self) -> bool:
        """pilot phase: any invalid triggers pause diagnosis."""
        return self.phase == "pilot" and self._invalid_count > 0

    def summary(self) -> Dict[str, object]:
        total = self._total_frames
        invalid = self._invalid_count
        if self.phase == "calibration":
            passes = total >= 10000 and invalid == 0
        else:
            passes = invalid == 0
        return {
            "phase": self.phase,
            "total_frames": total,
            "invalid_count": invalid,
            "rolling_invalid_rate": self.rolling_invalid_rate(),
            "passes": passes,
            "hard_failure": self.hard_failure(),
            "should_pause": self.should_pause(),
        }


# ── R7-5: the observation-window gate ──────────────────────────────


def _structural_invalid_frame(reason: str) -> PointCloudFrame:
    """Placeholder invalid frame for structurally broken windows.

    The window gate always returns ``obs_steps`` frame_results so
    callers can iterate and always have an invalid frame for
    ``gate.decide()``; a window that cannot be aligned (wrong row
    count / missing keys / malformed provenance) yields these instead
    of real frames.
    """
    return PointCloudFrame(
        points=np.zeros((0, 3), dtype=np.float32),
        valid=False,
        invalid_reason=reason,
        hit_count=0,
        unique_point_count=0,
        finite_ratio=0.0,
        range_min=0.0,
        range_max=0.0,
    )


def _structural_invalid_window(
    obs_steps: int,
    reason: str,
    obs: Optional[Mapping[str, Any]] = None,
) -> ObservationWindowValidity:
    """Invalid window for structural failure (fail-closed, R7-5).

    Echoes the mapping's hashes when present so hash-presence audits
    still see them even on a structurally broken window.
    """
    source_hash = "" if obs is None else str(obs.get("source_manifest_hash", ""))
    runtime_hash = "" if obs is None else str(obs.get("runtime_episode_hash", ""))
    return ObservationWindowValidity(
        valid=False,
        obs_steps=obs_steps,
        frame_results=tuple(
            _structural_invalid_frame(reason) for _ in range(obs_steps)
        ),
        invalid_frame_indices=tuple(range(obs_steps)),
        source_manifest_hash=source_hash,
        runtime_episode_hash=runtime_hash,
    )


def validate_policy_observation_window(
    obs: Union[Mapping[str, Any], Sequence[PointCloudFrame]],
    obs_steps: int,
    gate: Optional[PointCloudValidityGate] = None,
) -> ObservationWindowValidity:
    """R7-5: validate ALL ``obs_steps`` frames of the policy window.

    The policy consumes the latest ``obs_steps`` point-cloud frames
    (history[-3:] — guide R7-5); any invalid frame invalidates the
    window and the caller ends the episode as observation_fault.

    Two call forms:

    * Sequence[PointCloudFrame] — unit tests: the frames are already
      validated; their ``valid`` flags are read directly, hashes are "".
    * Mapping — production provenance (guide R7-5 interfaces): keys
      ``points`` ((obs_steps, point_count, 3) float32), ``cam_far``,
      optional per-frame provenance lists ``point_cloud_valid`` /
      ``point_cloud_frame_hash`` / ``hit_count_history`` /
      ``ray_report_history`` (each exactly ``obs_steps`` long, absent =
      recompute-only), and the two independent hashes
      ``source_manifest_hash`` / ``runtime_episode_hash`` (echoed back).

    Mapping path re-validates EVERY frame through ``gate`` (required)
    and cross-checks the recorded provenance: a recorded valid flag or
    frame hash that disagrees with the recompute invalidates the frame,
    as does a failing (or non-certificate) ray report. Provenance keys
    present with the wrong length make the window structurally invalid,
    as does a points array that cannot be aligned — both fail-closed.

    Note: the gate's rolling statistics track frame-CONTENT validity
    (recorded inside ``PointCloudValidityGate.validate``); a provenance
    cross-check failure flips the frame to invalid in this result only —
    the caller's verdict comes from ``ObservationWindowValidity``.

    Raises:
        ValueError: mapping path without a gate — no un-validated
            production window may pass (fail-closed).
    """
    if isinstance(obs, Sequence):
        frames = list(obs)
        if len(frames) != obs_steps:
            return _structural_invalid_window(
                obs_steps,
                f"window length {len(frames)} != obs_steps {obs_steps}",
            )
        invalid = tuple(i for i, f in enumerate(frames) if not f.valid)
        return ObservationWindowValidity(
            valid=not invalid,
            obs_steps=obs_steps,
            frame_results=tuple(frames),
            invalid_frame_indices=invalid,
            source_manifest_hash="",
            runtime_episode_hash="",
        )
    if gate is None:
        raise ValueError(
            "mapping window validation requires a PointCloudValidityGate"
            " (fail-closed: production windows are re-validated)"
        )
    points = obs.get("points")
    shape_ok = (
        isinstance(points, np.ndarray)
        and points.ndim == 3
        and points.shape[0] == obs_steps
        and points.shape[1:] == (gate.contract.point_count, 3)
    )
    if not shape_ok:
        shape = None if not isinstance(points, np.ndarray) else points.shape
        return _structural_invalid_window(
            obs_steps,
            f"points shape {shape} != ({obs_steps},"
            f" {gate.contract.point_count}, 3)",
            obs,
        )
    cam_far = obs.get("cam_far")
    if not isinstance(cam_far, (int, float, np.integer, np.floating)):
        return _structural_invalid_window(
            obs_steps, f"cam_far missing or non-numeric: {cam_far!r}", obs
        )
    cam_far = float(cam_far)
    recorded_valid = obs.get("point_cloud_valid")
    recorded_hashes = obs.get("point_cloud_frame_hash")
    hit_counts = obs.get("hit_count_history")
    ray_reports = obs.get("ray_report_history")
    prov_bad: List[str] = []
    for key, value in (
        ("point_cloud_valid", recorded_valid),
        ("point_cloud_frame_hash", recorded_hashes),
        ("hit_count_history", hit_counts),
        ("ray_report_history", ray_reports),
    ):
        if value is None:
            continue
        if not isinstance(value, (list, tuple)) or len(value) != obs_steps:
            prov_bad.append(
                f"{key} must be list/tuple of length obs_steps={obs_steps}"
            )
    if hit_counts is not None and not prov_bad:
        for i, hc in enumerate(hit_counts):
            if not isinstance(hc, (int, float, np.integer, np.floating)):
                prov_bad.append(f"hit_count_history[{i}] non-numeric: {hc!r}")
    if prov_bad:
        return _structural_invalid_window(obs_steps, "; ".join(prov_bad), obs)

    frame_results: List[PointCloudFrame] = []
    invalid_indices: List[int] = []
    for i in range(obs_steps):
        frame_points = points[i]
        hit_count = (
            int(hit_counts[i])
            if hit_counts is not None
            else frame_hit_count(frame_points, cam_far)
        )
        ray_report = ray_reports[i] if ray_reports is not None else None
        frame = gate.validate(
            frame_points, hit_count, cam_far, ray_report=ray_report
        )
        if frame.valid:
            reason: Optional[str] = None
            if (
                recorded_valid is not None
                and bool(recorded_valid[i]) is not True
            ):
                reason = "recorded point_cloud_valid disagrees with recompute"
            elif (
                recorded_hashes is not None
                and str(recorded_hashes[i]) != frame_sha256(frame_points)
            ):
                reason = "recorded frame hash disagrees with recompute"
            elif ray_report is not None and not getattr(
                ray_report, "passes", False
            ):
                reason = "ray report fails certification"
            if reason is not None:
                frame = replace(frame, valid=False, invalid_reason=reason)
        if not frame.valid:
            invalid_indices.append(i)
        frame_results.append(frame)
    return ObservationWindowValidity(
        valid=not invalid_indices,
        obs_steps=obs_steps,
        frame_results=tuple(frame_results),
        invalid_frame_indices=tuple(invalid_indices),
        source_manifest_hash=str(obs.get("source_manifest_hash", "")),
        runtime_episode_hash=str(obs.get("runtime_episode_hash", "")),
    )


def build_observation_window_mapping(
    obs_batched: Mapping[str, np.ndarray],
    obs_steps: int,
    cam_far: float,
    slot: int = 0,
) -> Dict[str, Any]:
    """Fallback window provenance for callers WITHOUT production
    provenance (mock envs / unit tests).

    Minimal mapping: the window's ``points`` (oldest→newest last
    ``obs_steps`` frames of slot ``slot``) and the camera far clip. No
    recorded-provenance keys, no hashes — the gate re-validates every
    frame from content alone (recompute-only path).
    """
    return {
        "points": np.asarray(obs_batched["point_cloud"])[slot, -obs_steps:],
        "cam_far": float(cam_far),
    }
