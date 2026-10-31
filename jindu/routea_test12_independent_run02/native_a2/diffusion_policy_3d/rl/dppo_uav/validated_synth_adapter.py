"""ValidatedSynthAdapter — NO-RAY-SEMANTICS fix.

Root cause (analytic): with pitch 8° nose-down and vertical half-FOV
≈ 23.41°, the lowest ray depresses 31.41° (vertical component
sin ≈ 0.5211). The ground plane — the nearest surface when nothing else
is within 10 m — exits the camera far clip (far = 10.0) at altitude
z > 10·0.5211 ≈ 5.21 m. Above that every one of the 76,800 rays misses the
far range and the frozen SynthPointCloud falls into its no-ray fallback
(synth_pointcloud.py:489-500), collapsing ALL rays onto the single point
(far, far, far) — a frame that reads as a wall 10 m away in every
direction, including below.

This adapter wraps the frozen synth (ZERO modification) and replaces that
degenerate encoding with a physical far-endpoint encoding: every no-return
ray is sampled at its geometric far endpoint (camera origin + far · dir),
exactly what a clipped depth camera returns. The frame is certified by an
analytic RayGeometryReport (4096 finite points, unique count above the
calibration floor, every point at distance == far from the camera mount,
direction-consistent with the sampled ray grid). The gate accepts
hit_count=0 ONLY with a passing report; the legacy single-point all-far
collapse stays ALWAYS invalid.

Wiring: the frozen env consumes a synth through its ``synth_pc_factory``
(kind_uav_env_v2.py:315-316); the environment adapter now injects this
wrapper instead of the raw synth. The env's call signature
(synthesize(pos, yaw, num_points=..., add_noise=..., rng_seed=...)) is
unchanged. Because the adapter processes slots sequentially, each
``last_report()`` read corresponds to the newest observation frame the
gate validates.
"""

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

# R7-5 shared provenance helpers live in the env package (both the env
# and the gate import them; see pointcloud_provenance docstring for the
# cycle rationale). Re-exported here so rl-side importers keep the
# single pointcloud_gate import surface.
from diffusion_policy_3d.env.pointcloud_provenance import (
    frame_hit_count,
    frame_sha256,
)
from diffusion_policy_3d.model.common.synth_pointcloud import (
    CAM_MOUNT_X,
    CAM_MOUNT_Y,
    CAM_MOUNT_Z,
    DEFAULT_CAM_PITCH_DEG,
    SynthPointCloud,
)

# Calibration floor for the far-dome unique count: 25% of the 4096-frame
# size. The dome subsamples distinct rays (0.19° grid at 10 m → 0.03-0.2 m
# separation, far above float32 precision) so the true count is ~4096;
# the floor rejects any residual collapse while leaving a large margin.
FAR_DOME_UNIQUE_FLOOR = 1024
# |p − cam_body| == far within 5e-3 m (float32 arithmetic on ~10 m vectors).
FAR_DOME_DIST_TOL = 5e-3
# Unit-direction mismatch <= 1e-3 (float32 unit-vector precision ~1e-7).
FAR_DOME_DIR_TOL = 1e-3


@dataclass(frozen=True)
class RayGeometryReport:
    """Analytic certification of a physical far-endpoint frame.

    Contract (guide Step 3): hit_count=0 is valid ONLY when ``passes`` is
    true. ``ray_indices`` records which grid rays were sampled so any
    caller can re-derive the expected frame from intrinsics + pose.
    """

    n_rays: int
    far: float
    all_finite: bool
    unique_count: int
    unique_passes: bool
    all_at_far: bool
    direction_consistent: bool
    ray_indices: np.ndarray
    passes: bool


class ValidatedSynthAdapter:
    """Frozen-synth wrapper with the no-ray fallback replaced.

    Non-degenerate frames pass through untouched (no report). A frame is
    degenerate iff every row sits on the far plane — the frozen fallback's
    signature, impossible for real hits within the far clip (a point at
    (far, far, far) is ~17.3 m from the camera origin).
    """

    def __init__(self, synth: SynthPointCloud) -> None:
        self._synth = synth
        self._cam = synth.cam
        self._far = float(self._cam.far)
        self._last_report: Optional[RayGeometryReport] = None
        self._degenerate_frames = 0
        self._ray_dirs_cam: Optional[np.ndarray] = None

    # ── state ──────────────────────────────────────────────────────

    def last_report(self) -> Optional[RayGeometryReport]:
        """Report of the most recently synthesized frame (None if the frame
        was non-degenerate). The adapter reads it right after each env
        reset/step, so it matches the newest observation frame."""
        return self._last_report

    @property
    def degenerate_frames(self) -> int:
        """Count of no-ray frames re-encoded as far-dome (evidence for the
        NO_RAY_ROOT_CAUSE_REPORT stress run)."""
        return self._degenerate_frames

    # ── env-callable surface (mirror of the frozen synth) ───────────

    def synthesize(
        self,
        drone_pos_world: np.ndarray,
        drone_yaw: float,
        num_points: int = 4096,
        add_noise: bool = True,
        ground_z: float = 0.0,
        drone_pitch: Optional[float] = None,
        rng_seed: Optional[int] = None,
    ) -> np.ndarray:
        """Delegate to the frozen synth; re-encode degenerate no-ray frames.

        The frozen no-ray RuntimeWarning is suppressed here — this adapter
        owns that path and keeps an explicit counter instead (evidence in
        ``degenerate_frames``).
        """
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="synth: no ray hit", category=RuntimeWarning
            )
            pc = self._synth.synthesize(
                drone_pos_world, drone_yaw,
                num_points=num_points, add_noise=add_noise,
                ground_z=ground_z, drone_pitch=drone_pitch,
                rng_seed=rng_seed,
            )
        if not self._is_degenerate(pc):
            self._last_report = None
            return pc
        frame, report = self._encode_far_dome(
            drone_pos_world, drone_yaw, num_points, drone_pitch, rng_seed
        )
        self._degenerate_frames += 1
        self._last_report = report
        return frame

    # ── internals ──────────────────────────────────────────────────

    def _is_degenerate(self, pc: np.ndarray) -> bool:
        return bool(np.allclose(pc, self._far))

    def _grid_ray_dirs(self) -> np.ndarray:
        """(H*W, 3) normalized camera-frame ray grid — the same closed-form
        grid the frozen synth builds (_build_rays), derived independently
        here so the far-dome and its certification rest on the intrinsics,
        not on the frozen fallback."""
        if self._ray_dirs_cam is not None:
            return self._ray_dirs_cam
        cam = self._cam
        u = np.arange(cam.width, dtype=np.float32)
        v = np.arange(cam.height, dtype=np.float32)
        uu, vv = np.meshgrid(u, v, indexing="xy")
        uu = uu.T.reshape(-1)
        vv = vv.T.reshape(-1)
        dx = (uu - cam.cx) / cam.fx
        dy = (vv - cam.cy) / cam.fy
        dz = np.ones_like(dx)
        dirs = np.stack([dx, dy, dz], axis=-1)
        dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
        self._ray_dirs_cam = dirs
        return dirs

    def _cam_to_world(self, pose: np.ndarray, yaw: float,
                      pitch: float) -> np.ndarray:
        """Camera world position (same transform chain as the frozen synth
        _cam_to_world_pose, re-derived here: R_body2world @ R_pitch @
        R_cam2body)."""
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        cos_p, sin_p = np.cos(pitch), np.sin(pitch)
        R_body2world = np.array(
            [[cos_y, -sin_y, 0], [sin_y, cos_y, 0], [0, 0, 1]]
        )
        R_cam2body = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        R_pitch = np.array([[cos_p, 0, sin_p], [0, 1, 0], [-sin_p, 0, cos_p]])
        R_cam2world = R_body2world @ R_pitch @ R_cam2body
        cam_body = np.array(
            [CAM_MOUNT_X, CAM_MOUNT_Y, CAM_MOUNT_Z], dtype=np.float32
        )
        cam_world = pose + R_body2world @ cam_body
        return cam_world, R_body2world, R_cam2world

    def _encode_far_dome(
        self,
        drone_pos_world: np.ndarray,
        drone_yaw: float,
        num_points: int,
        drone_pitch: Optional[float],
        rng_seed: Optional[int],
    ) -> tuple:
        """Build the physical far-endpoint frame + analytic certification.

        Sampling mirrors the frozen valid path's convention: uniform random
        without replacement via torch.randperm seeded from the episode RNG
        (synth_pointcloud.py:519-522 + 431-432). Far-endpoint points carry
        NO sensor noise — a clipped depth camera reads exactly far.
        """
        pose = np.asarray(drone_pos_world, dtype=np.float32)
        pitch = (
            np.radians(DEFAULT_CAM_PITCH_DEG) if drone_pitch is None
            else float(drone_pitch)
        )
        cam_world, R_body2world, R_cam2world = self._cam_to_world(
            pose, float(drone_yaw), pitch
        )
        dirs_cam = self._grid_ray_dirs()
        n_rays = dirs_cam.shape[0]

        if rng_seed is None:
            perm = torch.randperm(n_rays, device=self._synth.device)
        else:
            gen = torch.Generator(device=self._synth.device).manual_seed(
                int(rng_seed) & 0xFFFFFFFF
            )
            perm = torch.randperm(n_rays, generator=gen,
                                  device=self._synth.device)
        ray_indices = perm[:num_points].cpu().numpy().astype(np.int64)

        dirs_world = (R_cam2world @ dirs_cam[ray_indices].T).T
        endpoints_world = cam_world + self._far * dirs_world
        R_world2body = R_body2world.T
        frame = (
            R_world2body @ (endpoints_world - pose).T
        ).T.astype(np.float32)

        cam_body = np.array(
            [CAM_MOUNT_X, CAM_MOUNT_Y, CAM_MOUNT_Z], dtype=np.float32
        )
        all_finite = bool(np.isfinite(frame).all())
        unique_count = int(len(np.unique(frame, axis=0)))
        unique_passes = unique_count >= FAR_DOME_UNIQUE_FLOOR
        dist = np.linalg.norm(frame - cam_body, axis=1)
        all_at_far = bool(
            np.all(np.abs(dist - self._far) <= FAR_DOME_DIST_TOL)
        )
        dir_body = (frame - cam_body) / self._far
        expected_dir = (R_world2body @ dirs_world.T).T
        direction_consistent = bool(
            np.all(np.linalg.norm(dir_body - expected_dir, axis=1)
                   <= FAR_DOME_DIR_TOL)
        )
        passes = bool(
            all_finite and unique_passes and all_at_far
            and direction_consistent
        )
        report = RayGeometryReport(
            n_rays=n_rays,
            far=self._far,
            all_finite=all_finite,
            unique_count=unique_count,
            unique_passes=unique_passes,
            all_at_far=all_at_far,
            direction_consistent=direction_consistent,
            ray_indices=ray_indices,
            passes=passes,
        )
        return frame, report
