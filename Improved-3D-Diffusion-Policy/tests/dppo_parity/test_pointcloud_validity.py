"""PC-VALID: point-cloud validity gate (design §4.7).

Hard validity conditions + 5 invalid-frame handling rules + 4-phase gates.
Kills the P0-PC degenerate branch: SynthPointCloud no-ray fallback returned
4096 far-plane points (synth_pointcloud.py:489-500) that poisoned training.
"""

import numpy as np
import pytest

from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.pointcloud_gate import (
    InvalidFramePolicy,
    PointCloudFrame,
    PointCloudValidityGate,
)
from diffusion_policy_3d.rl.dppo_uav.validated_synth_adapter import (
    RayGeometryReport,
)

CAM_FAR = 10.0  # DepthCameraIntrinsics.far
N_PTS = 4096


def _valid_points(rng: np.random.RandomState) -> np.ndarray:
    """Scattered points: many unique, finite, within [0.3, 9.5]."""
    pts = rng.uniform(0.3, 9.5, size=(N_PTS, 3)).astype(np.float32)
    return pts


def _far_points() -> np.ndarray:
    return np.full((N_PTS, 3), CAM_FAR, dtype=np.float32)


# ---------- hard validity conditions ----------

def test_valid_frame_passes():
    rng = np.random.RandomState(0)
    gate = PointCloudValidityGate(RLSystemContract())
    frame = gate.validate(_valid_points(rng), hit_count=2048, cam_far=CAM_FAR)
    assert frame.valid is True
    assert frame.invalid_reason is None
    assert frame.hit_count == 2048
    assert frame.unique_point_count > 1
    assert frame.finite_ratio == 1.0
    assert frame.range_max <= CAM_FAR


def test_shape_mismatch_invalid():
    gate = PointCloudValidityGate(RLSystemContract())
    pts = _far_points()[:-1]  # (4095, 3)
    frame = gate.validate(pts, hit_count=4095, cam_far=CAM_FAR)
    assert frame.valid is False
    assert "shape" in frame.invalid_reason


def test_all_far_plane_points_invalid():
    """Regression: no-ray fallback far-plane points must be flagged."""
    gate = PointCloudValidityGate(RLSystemContract())
    frame = gate.validate(_far_points(), hit_count=0, cam_far=CAM_FAR)
    assert frame.valid is False
    assert "far" in frame.invalid_reason


def test_zero_hit_count_invalid():
    gate = PointCloudValidityGate(RLSystemContract())
    frame = gate.validate(_valid_points(np.random.RandomState(1)),
                          hit_count=0, cam_far=CAM_FAR)
    assert frame.valid is False
    assert "hit" in frame.invalid_reason


# ---------- NO-RAY-SEMANTICS: hit_count=0 needs analytic evidence ----------
# The frozen synth's no-ray fallback collapsed every ray onto one far-plane
# point (unique=1); the fix (ValidatedSynthAdapter) encodes no-return rays
# at their geometric far endpoints and certifies the frame analytically.
# Gate contract (guide Step 3): hit_count=0 is valid ONLY with a passing
# ray-geometry report; single-point all-far stays ALWAYS invalid.

def _far_dome_points() -> np.ndarray:
    """4096 points at distance CAM_FAR from the origin in distinct
    directions — a physical far-plane encoding, NOT the single-point
    collapse."""
    rng = np.random.RandomState(11)
    d = rng.normal(size=(N_PTS, 3)).astype(np.float32)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return (CAM_FAR * d).astype(np.float32)


def _ray_report(passes: bool, unique_count: int = N_PTS) -> RayGeometryReport:
    return RayGeometryReport(
        n_rays=320 * 240, far=CAM_FAR, all_finite=True,
        unique_count=unique_count, unique_passes=unique_count >= 1024,
        all_at_far=True, direction_consistent=passes,
        ray_indices=np.arange(N_PTS), passes=passes,
    )


def test_zero_hit_count_with_passing_ray_report_is_valid():
    """R2.5: hit_count=0 is valid only when the analytic ray-geometry
    checks all pass (physical far-endpoint encoding)."""
    gate = PointCloudValidityGate(RLSystemContract())
    frame = gate.validate(_far_dome_points(), 0, CAM_FAR,
                          ray_report=_ray_report(True))
    assert frame.valid is True
    assert frame.hit_count == 0


def test_zero_hit_count_with_failing_ray_report_invalid():
    """A report whose analytic checks fail must not relax the hit rule."""
    gate = PointCloudValidityGate(RLSystemContract())
    frame = gate.validate(_far_dome_points(), 0, CAM_FAR,
                          ray_report=_ray_report(False))
    assert frame.valid is False
    assert "hit_count" in frame.invalid_reason


def test_single_point_all_far_invalid_even_with_passing_report():
    """Guide Step 3: single-point all-far is ALWAYS invalid — the unique
    floor and the all-equal-far check are independently enforced even if a
    report claims passing analytic checks."""
    gate = PointCloudValidityGate(RLSystemContract())
    frame = gate.validate(_far_points(), 0, CAM_FAR,
                          ray_report=_ray_report(True, unique_count=1))
    assert frame.valid is False
    assert "unique" in frame.invalid_reason
    assert "far" in frame.invalid_reason


def test_single_unique_point_invalid():
    """unique_point_count > 1 (N_unique_min uncalibrated floor, §4.7)."""
    gate = PointCloudValidityGate(RLSystemContract())
    pts = np.tile(np.array([1.0, 2.0, 3.0], dtype=np.float32), (N_PTS, 1))
    frame = gate.validate(pts, hit_count=N_PTS, cam_far=CAM_FAR)
    assert frame.unique_point_count == 1
    assert frame.valid is False
    assert "unique" in frame.invalid_reason


def test_nan_points_invalid():
    gate = PointCloudValidityGate(RLSystemContract())
    pts = _valid_points(np.random.RandomState(2))
    pts[0, 0] = np.nan
    frame = gate.validate(pts, hit_count=N_PTS, cam_far=CAM_FAR)
    assert frame.finite_ratio < 1.0
    assert frame.valid is False


def test_invalid_frame_reports_explicit_reason():
    """Unit gate: expected-invalid poses must error explicitly, never silently."""
    gate = PointCloudValidityGate(RLSystemContract(), phase="unit")
    frame = gate.validate(_far_points(), hit_count=0, cam_far=CAM_FAR)
    assert frame.invalid_reason is not None
    assert gate.summary()["invalid_count"] == 1


# ---------- 5 invalid-frame handling rules ----------

def test_rule1_reset_invalid_rejects_episode():
    gate = PointCloudValidityGate(RLSystemContract())
    frame = gate.validate(_far_points(), hit_count=0, cam_far=CAM_FAR)
    decision = gate.decide(frame, phase="reset", episode_id="ep1",
                           episode_first_fault=True)
    assert decision == InvalidFramePolicy.REJECT_EPISODE


def test_rule2_rollout_invalid_excludes_transition():
    gate = PointCloudValidityGate(RLSystemContract())
    frame = gate.validate(_far_points(), hit_count=0, cam_far=CAM_FAR)
    decision = gate.decide(frame, phase="rollout", episode_id="ep1",
                           episode_first_fault=False)
    assert decision == InvalidFramePolicy.EXCLUDE_TRANSITION


def test_rule3_first_fault_safe_ends_without_collision_flag():
    """First invalid frame of an episode -> safe end; collision/reach unmarked.
    The gate marks SAFE_END_EPISODE; the collision/reach flags are mutually
    exclusive by policy (documented contract, asserted at API level)."""
    gate = PointCloudValidityGate(RLSystemContract())
    frame = gate.validate(_far_points(), hit_count=0, cam_far=CAM_FAR)
    decision = gate.decide(frame, phase="rollout", episode_id="ep1",
                           episode_first_fault=True)
    assert decision == InvalidFramePolicy.SAFE_END_EPISODE
    # collision/reach must NOT be set — policy enum is disjoint from them
    assert InvalidFramePolicy.SAFE_END_EPISODE not in (
        InvalidFramePolicy.REJECT_EPISODE,
        InvalidFramePolicy.EXCLUDE_TRANSITION,
    )


def test_rule4_rolling_invalid_rate_hard_failure():
    gate = PointCloudValidityGate(RLSystemContract(), rolling_window=100,
                                  rolling_max_invalid_rate=0.0)
    rng = np.random.RandomState(5)
    bad = _far_points()
    for _ in range(99):
        gate.validate(_valid_points(rng), hit_count=512, cam_far=CAM_FAR)
    assert gate.hard_failure() is False  # 0/99 invalid
    gate.validate(bad, hit_count=0, cam_far=CAM_FAR)  # 1/100 invalid
    assert gate.hard_failure() is True


def test_rule5_no_stale_replacement_api():
    """validate() accepts only the current frame — no prev-frame fallback path."""
    import inspect

    gate = PointCloudValidityGate(RLSystemContract())
    sig = inspect.signature(gate.validate)
    params = set(sig.parameters)
    assert "prev_points" not in params
    assert "prev_frame" not in params
    assert "fallback" not in params


# ---------- 4-phase gates ----------

def test_calibration_phase_requires_zero_invalid():
    """calibration: >=10k frames, invalid=0 required before training."""
    gate = PointCloudValidityGate(RLSystemContract(), phase="calibration")
    rng = np.random.RandomState(3)
    for _ in range(10000):
        gate.validate(_valid_points(rng), hit_count=1024, cam_far=CAM_FAR)
    s = gate.summary()
    assert s["total_frames"] == 10000
    assert s["invalid_count"] == 0
    assert s["passes"] is True
    # one invalid frame -> fails calibration
    gate.validate(_far_points(), hit_count=0, cam_far=CAM_FAR)
    assert gate.summary()["passes"] is False


def test_pilot_phase_pauses_on_any_invalid():
    gate = PointCloudValidityGate(RLSystemContract(), phase="pilot",
                                  rolling_window=10000)
    gate.validate(_far_points(), hit_count=0, cam_far=CAM_FAR)
    assert gate.should_pause() is True


def test_formal_phase_counts_every_frame():
    gate = PointCloudValidityGate(RLSystemContract(), phase="formal")
    rng = np.random.RandomState(4)
    for _ in range(5000):
        gate.validate(_valid_points(rng), hit_count=512, cam_far=CAM_FAR)
    gate.validate(_far_points(), hit_count=0, cam_far=CAM_FAR)
    s = gate.summary()
    assert s["total_frames"] == 5001
    assert s["invalid_count"] == 1


def test_formal_phase_never_hard_fails():
    """§4.7 formal row = count+archive only, NO rate gate (review C2):
    even a majority-invalid window must not trip hard_failure."""
    gate = PointCloudValidityGate(RLSystemContract(), phase="formal",
                                  rolling_window=50)
    rng = np.random.RandomState(6)
    for _ in range(25):
        gate.validate(_valid_points(rng), hit_count=512, cam_far=CAM_FAR)
    for _ in range(25):
        gate.validate(_far_points(), hit_count=0, cam_far=CAM_FAR)
    s = gate.summary()
    assert s["total_frames"] == 50
    assert s["invalid_count"] == 25
    assert s["hard_failure"] is False


def test_decide_rejects_unknown_phase():
    """decide() phases are the two live contexts only (reset/rollout)."""
    gate = PointCloudValidityGate(RLSystemContract())
    frame = gate.validate(_far_points(), hit_count=0, cam_far=CAM_FAR)
    with pytest.raises(AssertionError):
        gate.decide(frame, phase="bogus", episode_id="ep1",
                    episode_first_fault=True)
