"""Per-frame point-cloud provenance helpers (R7-5).

Shared bottom layer: the frozen env (kinematic_uav_env_v2) records
per-frame provenance (validity flag, sha256 hash, distance-based hit
count) when a frame is pushed, and the RL observation-window gate
(pointcloud_gate) recomputes the same values at gating time and
cross-checks them — any silent drift fails closed.

Housed in the env package (whose __init__ is empty) because BOTH sides
import it: the env must stay independently importable (scripts/tests
import kinematic_uav_env_v2 directly), and the rl side already imports
the env — placing these in rl/dppo_uav would create an env → rl package
import cycle through ``rl/dppo_uav/__init__.py`` → environment_adapter.
"""

import hashlib

import numpy as np

from diffusion_policy_3d.model.common.synth_pointcloud import (
    CAM_MOUNT_X,
    CAM_MOUNT_Y,
    CAM_MOUNT_Z,
)


def frame_sha256(points: np.ndarray) -> str:
    """First 16 hex chars of sha256 over the frame's float32 bytes.

    Deterministic per-frame fingerprint: the env records it when a
    frame is pushed, the window gate recomputes it at gating time, and
    any silent mutation of the buffered window shows up as a mismatch.
    float64/layout differences normalize away via the astype+contiguous
    conversion, so recorded and recomputed values agree on honest
    frames.
    """
    arr = np.ascontiguousarray(np.asarray(points, dtype=np.float32))
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def frame_hit_count(points: np.ndarray, cam_far: float) -> int:
    """Rows with a REAL return: distance from the camera mount strictly
    below the far clip (0.1 m band).

    Shared formula: the environment's per-frame hit counts and the
    observation-window gate's recompute path use exactly the same
    distance rule, so recorded and recomputed values agree on honest
    frames. The old heuristic ("rows not equal to (far, far, far)")
    broke under the physical far-endpoint encoding — far-dome rows sit
    at distance == far in DIFFERENT directions and would read as hits;
    distance-based counting reads 0 for both the far-dome (dist ≈ far)
    and the legacy all-far fallback (dist ≈ 17.3 at far=10).
    """
    cam_body = np.array(
        [CAM_MOUNT_X, CAM_MOUNT_Y, CAM_MOUNT_Z], dtype=np.float32
    )
    dist = np.linalg.norm(
        np.asarray(points, dtype=np.float32) - cam_body, axis=1
    )
    return int(np.count_nonzero(dist < float(cam_far) - 0.1))
