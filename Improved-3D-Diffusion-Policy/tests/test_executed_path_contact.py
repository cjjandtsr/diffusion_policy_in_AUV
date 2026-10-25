"""WSL-only focused regressions; no policy, world assets or real point cloud.

Includes the unchanged 12-case diagnostic, analytic boundary controls, and
real KinematicAdapter -> env.step -> contact -> terminal reward integration.
Only world-file loading and synthetic observations are replaced in step tests.
"""

import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diffusion_policy_3d.env import collision_geom as geom
from diffusion_policy_3d.env import kinematic_uav_env as kenv
from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.kinematic_adapter import KinematicAdapter
from diffusion_policy_3d.rl.dppo_uav_rebuild.rewards.v13.facade import RewardV13Facade


def test_original_twelve_case_diagnostic_unchanged():
    completed = subprocess.run(
        [sys.executable, "-B", str(ROOT / "scripts/diagnose_executed_path_contact.py")],
        cwd=str(ROOT), capture_output=True, text=True, timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(completed.stdout)
    assert report["summary"] == {
        "total": 12, "matched": 12, "mismatched": 0, "failed_cases": [],
    }


def obstacle(shape="box", yaw=0.0, z=1.0):
    return geom.ObstacleSpec(
        name="fixture_" + shape, shape=shape, x=0.0, y=0.0, z=z,
        yaw=yaw, radius=0.05, height=2.0, size_x=0.1, size_y=0.4, size_z=2.0,
    )


@pytest.mark.parametrize("shape", ["box", "cylinder"])
@pytest.mark.parametrize("a,b,want", [
    ((-0.3, 0, 1), (0.3, 0, 1), True),       # Solid crossing.
    ((0.3, 0, 1), (-0.3, 0, 1), True),       # Direction must not matter.
    ((-0.3, 0, 2.12), (0.3, 0, 2.12), False),  # Positive 2cm top gap.
    ((-0.3, 0, 2.06), (0.3, 0, 2.06), True),   # 4cm top overlap.
    ((0, 0, 2.5), (0, 0, -0.5), True),      # Safe endpoints, vertical crossing.
    ((0, 0, 2.5), (0, 0, 2.5), False),      # Zero-length separated segment.
    ((0, 0, 1), (0, 0, 1), True),           # Zero-length interior segment.
    ((-0.3, 1, 1), (0.3, 1, 1), False),     # Clear lateral bypass.
    ((-0.4, 0, 1), (0, 0, 3), False),       # XY/Z overlap at DIFFERENT times.
])
def test_segment_contact_boundary(shape, a, b, want):
    hit = geom.segment_obstacle_collision(a, b, [obstacle(shape)], collision_thresh=0.1)
    assert (hit is not None) == want


@pytest.mark.parametrize("shape", ["box", "cylinder"])
def test_suspended_underpass_is_not_ground_flight(shape):
    # Suspended body occupies z=[3,5], UAV top=2.9; true air gap.
    hit = geom.segment_obstacle_collision(
        (-0.3, 0, 2.8), (0.3, 0, 2.8), [obstacle(shape, z=4.0)], 0.1,
    )
    assert hit is None


def test_rotated_box_uses_oriented_shape_not_bounding_circle():
    obs = obstacle(yaw=math.pi / 2)
    # Rotated XY extents: x=+-0.2, y=+-0.05; x=.28 has .08 clearance.
    assert geom.segment_obstacle_collision((0.28, -0.3, 1), (0.28, 0.3, 1), [obs], 0.1)
    assert geom.segment_obstacle_collision((0.32, -0.3, 1), (0.32, 0.3, 1), [obs], 0.1) is None


def test_box_corner_keeps_euclidean_clearance_not_square_inflation():
    # Offsets from corner (.05,.2) are (.08,.08), distance .113 > .1.
    assert geom.segment_obstacle_collision((0.13, 0.28, 1), (0.2, 0.35, 1), [obstacle()], 0.1) is None


def test_thin_wall_crossing_is_not_tied_to_a_sampling_grid():
    obs = obstacle()
    obs.x = 0.012345
    obs.size_x = 1e-6
    assert geom.segment_obstacle_collision((-1, 0, 1), (1, 0, 1), [obs], 0.1)


class FixedObservation:
    """Avoid point-cloud work; not evidence about sensor correctness."""
    def synthesize(self, pos, yaw, num_points=4096, add_noise=True, rng_seed=None):
        return np.zeros((num_points, 3), dtype=np.float32)


class FixedImpulse:
    """Deterministic execution-noise fixture for the existing env noise branch."""
    def __init__(self, xyz):
        self.xyz = np.array(xyz, dtype=np.float32)

    def normal(self, loc, scale, size):
        assert (loc, scale, size) == (0, 1.0, 3)
        return self.xyz.copy()


def make_adapter(monkeypatch, start, goal, obstacles, impulse):
    # Patch only two world-file I/O boundaries, not constructor/step/reward.
    monkeypatch.setattr(kenv, "load_obstacles", lambda _: list(obstacles))
    monkeypatch.setattr(kenv.KinematicUAVEnv, "_load_spawn_offset",
                        staticmethod(lambda _: np.zeros(3, dtype=np.float32)))
    env = kenv.KinematicUAVEnv(
        synth_pc=FixedObservation(), goal_world=np.array(goal, dtype=np.float32),
        offline_density={}, reward_version="v13", disturb_enable=False,
        exec_noise_scale=1.0 if impulse is not None else 0.0, seed=1,
    )
    env._drone_pos = np.array(start, dtype=np.float32)
    if impulse is not None:
        env._rng = FixedImpulse(impulse)
    facade = RewardV13Facade()
    state = facade.initial_state()
    env._reward_history = state.history
    return KinematicAdapter(env, facade, state)


@pytest.mark.parametrize("label,start,positions,goal,impulse,want", [
    ("first_segment", (-0.3, 0, 1),
     [(0.3, 0, 1), (0.4, 0, 1), (0.5, 0, 1), (0.6, 0, 1)] * 2,
     (0.6, 0, 1), None, "collision"),
    ("ground_then_goal", (-0.3, 0, 1),
     [(-0.3, 0, -0.5), (-0.3, 0, 1), (-0.3, 0, 1), (-0.3, 0, 1)] * 2,
     (-0.3, 0, 1), None, "crash_ground"),
    ("unexecuted_tail", (-0.5, 0, 1),
     [(-0.4, 0, 1)] * 4 + [(0, 0, 1)] * 4,
     (-0.4, 0, 1), None, "reached"),
    ("perturbed_segment", (-0.5, 0, 1), [(-0.4, 0, 1)] * 8,
     (0.4, 0, 1), (0.8, 0, 0), "collision"),
    ("unused_nominal_endpoint", (-0.5, 0, 1),
     [(-0.4, 0, 1)] * 3 + [(0, 0, 1)] * 5,
     (-0.4, 0, 1), (-0.4, 0, 0), "reached"),
])
def test_real_step_and_dppo_terminal_wiring(monkeypatch, label, start, positions, goal, impulse, want):
    adapter = make_adapter(monkeypatch, start, goal, [obstacle()], impulse)
    points = np.array(positions, dtype=np.float32)
    action = np.zeros((8, 11), dtype=np.float32)
    action[:, :3] = np.diff(np.vstack((np.array(start, dtype=np.float32), points)), axis=0)
    original = action.copy()
    obs, reward, terminated, truncated, info = adapter.step(SimpleNamespace(policy=action))
    assert terminated and not truncated, label
    facts = info["transition_facts"]
    assert facts.done_reason == ("collision" if want == "crash_ground" else want), label
    if want == "crash_ground":
        assert facts.raw_done_reason == "crash_ground"
    assert (reward > 0) == (want == "reached"), (label, reward, info)
    if want != "reached":
        assert info["reward_terms"]["support"] == 0.0
    np.testing.assert_array_equal(action, original)  # Never rewrite policy actions.
    np.testing.assert_allclose(adapter.environment._drone_pos, goal, atol=1e-6)
    # Actual endpoint is still published; contact flags, not trajectory clipping.
    np.testing.assert_allclose(obs["agent_pos"][-1, :3], goal, atol=1e-6)
