"""RED #4/#5 (R7-PC-WINDOW): latest-3 full-frame point-cloud validity.

Current state (measured 2026-08-22): train_loop.py:2092-2095 and
2161-2165 validate only ``point_cloud[..., -1]`` — the NEWEST frame of the
policy's latest-3 observation window. A window whose history[-3] or
history[-2] frame is invalid (stale/far/zero-hit) passes today, letting a
policy act on corrupted context.

R7-PC-WINDOW contract: an observation-window gate
``validate_policy_observation_window`` checks ALL ``obs_steps`` frames
(guide R7-5); any invalid frame invalidates the window and the caller
ends the episode as observation_fault (guide exception matrix:
"latest-3 任一 invalid → 结束 episode 为 observation_fault").

These tests MUST FAIL now (API absent) and go green in R7-5.
"""

import numpy as np
import pytest

from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.pointcloud_gate import (
    PointCloudValidityGate,
)


def _valid_points(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(1.0, 4.0, size=(4096, 3)).astype(np.float32)


def _invalid_far_points() -> np.ndarray:
    return np.full((4096, 3), 10.0, dtype=np.float32)  # all-far collapse


def _make_window_gate() -> PointCloudValidityGate:
    return PointCloudValidityGate(
        contract=RLSystemContract(),
        phase="unit",
        n_unique_min=2,
    )


def test_window_rejects_invalid_oldest_frame() -> None:
    """RED #4: history[-3] invalid, [-2]/-1 valid → window must FAIL.

    Current code has no window gate, so this scenario is accepted.
    """
    import importlib

    try:
        mod = importlib.import_module(
            "diffusion_policy_3d.rl.dppo_uav.pointcloud_gate"
        )
    except ModuleNotFoundError as exc:
        pytest.fail(f"pointcloud_gate import failed: {exc}")
    assert hasattr(mod, "validate_policy_observation_window"), (
        "validate_policy_observation_window missing — no observation-window"
        " gate exists"
    )

    gate = _make_window_gate()
    frames = [
        gate.validate(_invalid_far_points(), hit_count=0, cam_far=10.0),  # -3
        gate.validate(_valid_points(1), hit_count=4096, cam_far=10.0),    # -2
        gate.validate(_valid_points(2), hit_count=4096, cam_far=10.0),    # -1
    ]
    assert not frames[0].valid, "precondition: oldest frame must be invalid"
    result = mod.validate_policy_observation_window(frames, obs_steps=3)
    assert result.valid is False, (
        "window with invalid history[-3] frame accepted — the policy would"
        " act on corrupted context"
    )


def test_window_rejects_invalid_second_oldest_frame() -> None:
    """RED #5: history[-2] invalid, [-3]/-1 valid → window must FAIL."""
    import importlib

    mod = importlib.import_module(
        "diffusion_policy_3d.rl.dppo_uav.pointcloud_gate"
    )
    gate = _make_window_gate()
    frames = [
        gate.validate(_valid_points(3), hit_count=4096, cam_far=10.0),    # -3
        gate.validate(_invalid_far_points(), hit_count=0, cam_far=10.0),  # -2
        gate.validate(_valid_points(4), hit_count=4096, cam_far=10.0),    # -1
    ]
    assert not frames[1].valid, "precondition: middle frame must be invalid"
    result = mod.validate_policy_observation_window(frames, obs_steps=3)
    assert result.valid is False, (
        "window with invalid history[-2] frame accepted"
    )
