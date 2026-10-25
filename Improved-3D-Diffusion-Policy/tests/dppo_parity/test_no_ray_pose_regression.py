"""R2.5 NO-RAY-SEMANTICS: fixed-pose replay of the G5 degenerate family.

G5 traced a pose family (x≈-10.1..-10.7, y≈-4.3..-4.6, z≈5.2..5.4, world
edge, outward yaw) whose synthesized frames were ALL camera-far:
SynthPointCloud.synthesize hit the no-ray fallback
(synth_pointcloud.py:489-500) and returned 4096 rows of (far, far, far)
— unique=1, hit_count=0, all-far.

Analytic root cause: pitch 8° nose-down + vertical half-FOV ≈ 23.41° means
the lowest ray depresses 31.41° (vertical component sin ≈ 0.5211); the
ground plane — the nearest surface when nothing else is within 10 m — exits
the far clip (far=10.0) at altitude z > 10·0.5211 ≈ 5.21 m. The traced band
z∈[5.2, 5.4] matches exactly. The fallback then collapses EVERY no-return
ray onto the single point (10, 10, 10), inventing a wall 10 m away in all
directions.

Resolution (guide branch 1 DEGENERATE_ENCODING): ValidatedSynthAdapter
re-encodes the frame as per-ray physical far endpoints — what a clipped
depth camera returns — and certifies it with an analytic RayGeometryReport
(4096 finite, unique > FAR_DOME_UNIQUE_FLOOR, |p − cam_body| == far,
direction-consistent with the sampled rays). The gate accepts hit_count=0
ONLY with passing analytic evidence; the legacy single-point all-far
collapse stays ALWAYS invalid, so it can never enter the training buffer.

TDD: new-feature tests import ValidatedSynthAdapter / RayGeometryReport
inside the test body — before R2.5 the module does not exist (RED), while
the legacy-behavior pins below run against the frozen synth today.
"""

import json
import warnings

import numpy as np
import pytest

from diffusion_policy_3d.model.common.synth_pointcloud import (
    DEFAULT_CAM_PITCH_DEG,
    DepthCameraIntrinsics,
    SynthPointCloud,
)
from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.pointcloud_gate import (
    PointCloudValidityGate,
)

CAM_FAR = DepthCameraIntrinsics().far
N_PTS = 4096
RNG_SEED = 20260808

# G5 trace pose family (world edge, outward yaw). Exact trace pose
# (jiaojie A-OfficialCore_G5门禁_交接_20260807.md): (-10.182, -4.548, 5.415)
# at step 436, yaw pointing outward.
TRACE_POSE = np.array([-10.182, -4.548, 5.415], dtype=np.float32)
TRACE_YAW = float(np.pi)  # outward: facing -x from the west edge
FAMILY_POSES = [
    TRACE_POSE,
    np.array([-10.7, -4.6, 5.2], dtype=np.float32),
    np.array([-10.1, -4.3, 5.4], dtype=np.float32),
    np.array([-10.4, -4.45, 5.3], dtype=np.float32),
]


# ═══════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════

def _empty_world(tmp_path) -> str:
    """Ground-only world: no obstacle within 10 m anywhere — the controlled
    equivalent of the G5 edge pose (nearest surface = ground > far)."""
    p = tmp_path / "env_r25.obstacles.json"
    p.write_text(json.dumps({"world_id": "env_r25", "obstacles": []}))
    return str(p)


def _legacy_frame(synth: SynthPointCloud, pose: np.ndarray,
                  yaw: float) -> np.ndarray:
    """Replay through the FROZEN synth exactly as the env calls it
    (kinematic_uav_env_v2.py:487-489). The frozen fallback warns on the
    degenerate path — expected, pinned by test_* below."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return synth.synthesize(pose, yaw, num_points=N_PTS, add_noise=True,
                                rng_seed=RNG_SEED)


def _adapter_hit_count(points: np.ndarray) -> int:
    """Mirror of environment_adapter._hit_count (R2.5): rows whose distance
    from the camera mount is strictly below the far clip (0.1 m band) —
    far-dome rows (dist ≈ 10.0) and the legacy all-far fallback both read 0."""
    cam_body = np.array([0.12, 0.0, 0.015], dtype=np.float32)
    dist = np.linalg.norm(np.asarray(points, dtype=np.float32) - cam_body,
                          axis=1)
    return int(np.count_nonzero(dist < CAM_FAR - 0.1))


def _expected_far_dome(synth: SynthPointCloud, pose: np.ndarray, yaw: float,
                       num_points: int, ray_indices: np.ndarray) -> np.ndarray:
    """Independent numpy re-derivation of the physical far-endpoint frame:
    ray grid from intrinsics, camera→world pose, far endpoints, sampled
    rays, body-frame rotation — WITHOUT calling any synth method. This is
    the analytic oracle for the direction-consistency contract."""
    cam = synth.cam
    u = np.arange(cam.width, dtype=np.float32)
    v = np.arange(cam.height, dtype=np.float32)
    uu, vv = np.meshgrid(u, v, indexing="xy")  # same ordering as _build_rays
    uu = uu.T.reshape(-1)
    vv = vv.T.reshape(-1)
    dx = (uu - cam.cx) / cam.fx
    dy = (vv - cam.cy) / cam.fy
    dz = np.ones_like(dx)
    dirs_cam = np.stack([dx, dy, dz], axis=-1)
    dirs_cam = dirs_cam / np.linalg.norm(dirs_cam, axis=-1, keepdims=True)

    pitch = np.radians(DEFAULT_CAM_PITCH_DEG)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    R_body2world = np.array(
        [[cos_y, -sin_y, 0], [sin_y, cos_y, 0], [0, 0, 1]]
    )
    R_cam2body = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    R_pitch = np.array([[cos_p, 0, sin_p], [0, 1, 0], [-sin_p, 0, cos_p]])
    R_cam2world = R_body2world @ R_pitch @ R_cam2body
    cam_body = np.array([0.12, 0.0, 0.015], dtype=np.float32)
    cam_world = pose + R_body2world @ cam_body

    sel = np.asarray(ray_indices, dtype=np.int64)
    dirs_world = (R_cam2world @ dirs_cam[sel].T).T
    endpoints_world = cam_world + CAM_FAR * dirs_world
    R_world2body = R_body2world.T
    return ((R_world2body @ (endpoints_world - pose).T).T).astype(np.float32)


def _far_dome_points(synth: SynthPointCloud) -> np.ndarray:
    """Far-dome along the first N_PTS grid rays — a gate-semantics fixture
    that depends only on the frozen synth's intrinsics (no adapter)."""
    return _expected_far_dome(synth, TRACE_POSE, TRACE_YAW, N_PTS,
                              np.arange(N_PTS))


# ═══════════════════════════════════════════════════════════════════
# Replay pins on the LEGACY degenerate output — these run against the
# frozen synth today and pin the G5 evidence + the gate invariant
# (hit_count=0 / unique=1 / all-far ⇒ explicitly invalid).
# ═══════════════════════════════════════════════════════════════════

def test_legacy_no_ray_fallback_collapses_to_single_far_point(tmp_path):
    """Pins the G5 evidence: at the trace pose the frozen synth returns the
    all-far collapse (unique=1, hit_count=0) and the gate rejects it."""
    synth = SynthPointCloud(_empty_world(tmp_path), device="cpu")
    pc = _legacy_frame(synth, TRACE_POSE, TRACE_YAW)
    assert pc.shape == (N_PTS, 3)
    assert np.allclose(pc, CAM_FAR)            # every row at the far plane
    assert len(np.unique(pc, axis=0)) == 1     # single collapsed point
    assert _adapter_hit_count(pc) == 0         # adapter reading
    frame = PointCloudValidityGate(RLSystemContract()).validate(
        pc, 0, CAM_FAR
    )
    assert frame.valid is False                # explicitly invalid
    assert "hit_count" in frame.invalid_reason
    assert "far" in frame.invalid_reason


def test_family_replay_is_all_far_for_frozen_synth(tmp_path):
    """Every family corner replays to the same collapse (deterministic)."""
    synth = SynthPointCloud(_empty_world(tmp_path), device="cpu")
    for pose in FAMILY_POSES:
        pc = _legacy_frame(synth, pose, TRACE_YAW)
        assert np.allclose(pc, CAM_FAR)
        assert len(np.unique(pc, axis=0)) == 1


def test_gate_rejects_far_dome_without_analytic_evidence(tmp_path):
    """Guide Step 3 contract: hit_count=0 WITHOUT a passing analytic report
    stays invalid — no silent acceptance of far-only frames. The points
    here are physically meaningful (distinct far endpoints), yet the gate
    must reject them on evidence grounds."""
    synth = SynthPointCloud(_empty_world(tmp_path), device="cpu")
    pc = _far_dome_points(synth)
    assert len(np.unique(pc, axis=0)) > 1      # NOT the single-point collapse
    gate = PointCloudValidityGate(RLSystemContract())
    frame = gate.validate(pc, 0, CAM_FAR)      # no ray_report
    assert frame.valid is False
    assert "hit_count" in frame.invalid_reason


# ═══════════════════════════════════════════════════════════════════
# RED — the DEGENERATE_ENCODING fix. All of these fail today: the module
# does not exist (ImportError) / the gate has no ray_report parameter
# (TypeError).
# ═══════════════════════════════════════════════════════════════════

def test_validated_adapter_encodes_physical_far_endpoints(tmp_path):
    """DEGENERATE_ENCODING: no-return rays are encoded at their geometric
    far endpoints and certified by an analytic ray-geometry report."""
    from diffusion_policy_3d.rl.dppo_uav.validated_synth_adapter import (
        FAR_DOME_UNIQUE_FLOOR,
        ValidatedSynthAdapter,
    )

    synth = SynthPointCloud(_empty_world(tmp_path), device="cpu")
    adapter = ValidatedSynthAdapter(synth)
    pc = adapter.synthesize(TRACE_POSE, TRACE_YAW, num_points=N_PTS,
                            add_noise=True, rng_seed=RNG_SEED)
    report = adapter.last_report()
    assert pc.shape == (N_PTS, 3)
    assert np.isfinite(pc).all()
    assert len(np.unique(pc, axis=0)) > FAR_DOME_UNIQUE_FLOOR
    # distance from the camera mount == far (body frame)
    cam_body = np.array([0.12, 0.0, 0.015], dtype=np.float32)
    dist = np.linalg.norm(pc - cam_body, axis=1)
    assert np.allclose(dist, CAM_FAR, atol=5e-3)
    assert report is not None
    assert report.passes is True
    assert report.n_rays == 320 * 240
    assert report.all_finite is True
    assert report.all_at_far is True
    assert report.direction_consistent is True


def test_far_dome_matches_independent_ray_geometry_oracle(tmp_path):
    """The analytic oracle: recompute the expected far-dome from intrinsics
    + pose + the sampled ray indices (independent numpy implementation) and
    require the adapter frame to match."""
    from diffusion_policy_3d.rl.dppo_uav.validated_synth_adapter import (
        ValidatedSynthAdapter,
    )

    synth = SynthPointCloud(_empty_world(tmp_path), device="cpu")
    adapter = ValidatedSynthAdapter(synth)
    pc = adapter.synthesize(TRACE_POSE, TRACE_YAW, num_points=N_PTS,
                            add_noise=True, rng_seed=RNG_SEED)
    report = adapter.last_report()
    assert report is not None
    expected = _expected_far_dome(synth, TRACE_POSE, TRACE_YAW, N_PTS,
                                  report.ray_indices)
    assert np.allclose(pc, expected, atol=1e-3)


def test_gate_accepts_far_dome_with_analytic_report(tmp_path):
    """hit_count=0 is valid ONLY when the analytic ray-geometry checks all
    pass (guide Step 3)."""
    from diffusion_policy_3d.rl.dppo_uav.validated_synth_adapter import (
        ValidatedSynthAdapter,
    )

    synth = SynthPointCloud(_empty_world(tmp_path), device="cpu")
    adapter = ValidatedSynthAdapter(synth)
    pc = adapter.synthesize(TRACE_POSE, TRACE_YAW, num_points=N_PTS,
                            add_noise=True, rng_seed=RNG_SEED)
    report = adapter.last_report()
    gate = PointCloudValidityGate(RLSystemContract())
    frame = gate.validate(pc, 0, CAM_FAR, ray_report=report)
    assert frame.valid is True
    assert frame.hit_count == 0


def test_gate_always_rejects_single_point_all_far(tmp_path):
    """Guide Step 3: single-point all-far is ALWAYS invalid — even a report
    that claims passing analytic checks cannot rescue the collapse (the
    unique floor is independently enforced by the gate)."""
    from diffusion_policy_3d.rl.dppo_uav.validated_synth_adapter import (
        RayGeometryReport,
    )

    synth = SynthPointCloud(_empty_world(tmp_path), device="cpu")
    pc = _legacy_frame(synth, TRACE_POSE, TRACE_YAW)  # single-point all-far
    fake_report = RayGeometryReport(
        n_rays=320 * 240, far=CAM_FAR, all_finite=True, unique_count=1,
        unique_passes=False, all_at_far=True, direction_consistent=True,
        ray_indices=np.arange(N_PTS), passes=True,
    )
    gate = PointCloudValidityGate(RLSystemContract())
    frame = gate.validate(pc, 0, CAM_FAR, ray_report=fake_report)
    assert frame.valid is False
    assert "unique" in frame.invalid_reason


def test_family_replay_all_recovered_by_validated_adapter(tmp_path):
    """The whole G5 family encodes to valid far-dome frames (no episode is
    wasted at the edge any more)."""
    from diffusion_policy_3d.rl.dppo_uav.validated_synth_adapter import (
        ValidatedSynthAdapter,
    )

    synth = SynthPointCloud(_empty_world(tmp_path), device="cpu")
    adapter = ValidatedSynthAdapter(synth)
    for pose in FAMILY_POSES:
        pc = adapter.synthesize(pose, TRACE_YAW, num_points=N_PTS,
                                add_noise=True, rng_seed=RNG_SEED)
        report = adapter.last_report()
        assert np.isfinite(pc).all()
        assert len(np.unique(pc, axis=0)) > 1
        assert report is not None and report.passes is True


# ═══════════════════════════════════════════════════════════════════
# End-to-end: the PRODUCTION path (frozen env → wrapper → adapter info →
# gate). A rollout frame at the degenerate pose was invalid → SAFE_END →
# the episode voided; now the same frame is accepted via the analytic
# report and the episode continues.
# ═══════════════════════════════════════════════════════════════════

def test_real_adapter_flows_ray_report_end_to_end(tmp_path):
    """A rollout frame at the G5 degenerate pose flows through the real
    adapter: hit_count=0 rides the info with a passing analytic report and
    the gate accepts the frame."""
    from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
        EpisodeTaskSpec,
        UAVVectorEnvAdapter,
        manifest_record_key,
    )

    world_json = _empty_world(tmp_path)
    contract = RLSystemContract()
    adapter = UAVVectorEnvAdapter(
        contract,
        [("env_r25", world_json)],
        device="cpu",
        max_episode_steps=30,
        manifest_records={
            manifest_record_key("env_r25", "anticipatory", "a"): "mhash-a",
        },
    )
    spec = EpisodeTaskSpec(
        world_uid="env_r25", obstacle_json="unused", role="anticipatory",
        split="rl_train", admission_id="a",
        generator_version="routeA_rl_v1.3", manifest_entry_hash="mhash-a",
    )
    obs, info = adapter.reset([spec], seeds=[1])
    gate = PointCloudValidityGate(contract, phase="unit")
    reset_frame = gate.validate(
        obs["point_cloud"][0, -1], info[0].get("hit_count", 0), CAM_FAR,
        ray_report=info[0].get("ray_report"),
    )
    assert reset_frame.valid is True   # admission pose is inside the domain

    # teleport slot 0 into the G5 degenerate family (world edge, z > 5.21)
    env0 = adapter._envs[0]
    env0._drone_pos = TRACE_POSE.copy()
    env0._drone_yaw = TRACE_YAW
    zero = np.zeros((1, contract.exec_steps, contract.action_dim),
                    dtype=np.float32)
    obs, reward, terminated, truncated, info = adapter.step(zero)
    assert info[0]["hit_count"] == 0
    rpt = info[0].get("ray_report")
    assert rpt is not None and rpt.passes is True
    frame = gate.validate(
        obs["point_cloud"][0, -1], info[0]["hit_count"], CAM_FAR,
        ray_report=rpt,
    )
    assert frame.valid is True
    assert not bool(terminated[0])
