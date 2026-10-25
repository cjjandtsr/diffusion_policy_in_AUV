"""Reset final-validation and diagnostic-only A* contracts."""

from __future__ import annotations

import importlib
import numpy as np
import pytest

from diffusion_policy_3d.env.collision_geom import ObstacleSpec
from diffusion_policy_3d.env.kinematic_uav_env import KinematicUAVEnv


def _terminal():
    return importlib.import_module(
        "diffusion_policy_3d.rl.dppo_uav_rebuild.environments.terminal_contract"
    )


class SequenceRng:
    def __init__(self, uniform_values, random_value=1.0):
        self.uniform_values = iter(uniform_values)
        self.random_value = random_value

    def uniform(self, _low, _high):
        return next(self.uniform_values)

    def random(self):
        return self.random_value


def _bare_env():
    env = object.__new__(KinematicUAVEnv)
    env._episode_role = "anticipatory"
    env.curriculum_stage = "C2"
    env.z_min = 1.0
    env.z_max = 5.0
    env.z_target = 3.0
    env.max_reset_attempts = 3
    env.mid_state_init_ratio = 0.0
    env.reward_version = "v17"
    env.obstacles = []
    env.world_id = "task-world"
    env._get_occupancy_grid = lambda: np.ones((2, 2, 2), dtype=bool)
    return env


@pytest.mark.parametrize(
    "astar_result,status",
    [
        ((False, float("inf"), 0.0, 0.0, 0.0, []), "unreachable"),
        ((True, 100.0, 0.0, 0.0, 0.0, [(0, 0, 0)]), "reachable"),
    ],
)
def test_astar_and_path_stretch_never_reject_a_physically_legal_candidate(
    astar_result, status
):
    """Catches A* reachability or path difficulty filtering training episodes."""

    env = _bare_env()
    env._rng = SequenceRng([-5.0, 0.0, 2.0, 5.0, 0.0, 2.0, 0.0])
    env._astar_3d = lambda *_args: astar_result
    start, _yaw, goal = env._sample_start_goal("dense_recovery")
    np.testing.assert_allclose(start, [-5.0, 0.0, 2.0])
    np.testing.assert_allclose(goal, [5.0, 0.0, 2.0])
    assert env._last_astar_diagnostic["status"] == status
    assert env._last_astar_diagnostic["diagnostic_only"] is True


def test_final_validation_runs_after_mid_state_perturbation():
    """Catches committing a start that a later perturbation moved into an obstacle."""

    env = _bare_env()
    env.mid_state_init_ratio = 1.0
    env._rng = SequenceRng([], random_value=0.0)
    env.obstacles = [
        ObstacleSpec("blocked", "sphere", 0.0, 0.0, 2.0, radius=1.0)
    ]
    env._draw_start_goal_candidate = lambda _role: (
        np.array([-4.0, 0.0, 2.0], dtype=np.float32),
        0.0,
        np.array([4.0, 0.0, 2.0], dtype=np.float32),
    )
    env._perturb_to_mid_state = lambda _position, yaw: (
        np.array([0.0, 0.0, 2.0], dtype=np.float32),
        yaw,
    )
    with pytest.raises(_terminal().ResetExhausted) as caught:
        env._prepare_reset_candidate("anticipatory", apply_mid_state=True)
    assert caught.value.attempts == 3
    assert caught.value.rejection_histogram == {"clearance_start": 3}


@pytest.mark.parametrize("mid_state", [False, True])
def test_r1_reset_rejects_goal_sphere_start_before_reward_initialization(mid_state):
    """A sampled or perturbed start inside the goal ball must be redrawn."""

    env = _bare_env()
    env.reward_version = "v2"
    env.mid_state_init_ratio = 1.0 if mid_state else 0.0
    env._rng = SequenceRng([], random_value=0.0)
    env._draw_start_goal_candidate = lambda _role: (
        np.array([-2.0 if mid_state else 0.0, 0.0, 2.0], dtype=np.float32),
        0.0,
        np.array([0.1, 0.0, 2.0], dtype=np.float32),
    )
    if mid_state:
        env._perturb_to_mid_state = lambda _position, yaw: (
            np.array([0.0, 0.0, 2.0], dtype=np.float32), yaw,
        )
    env._compute_astar_diagnostic = lambda *_args: (_ for _ in ()).throw(
        AssertionError("goal-sphere candidate reached admission")
    )

    with pytest.raises(_terminal().ResetExhausted) as caught:
        env._prepare_reset_candidate("anticipatory", apply_mid_state=True)

    assert caught.value.attempts == 3
    assert caught.value.rejection_histogram == {"goal_sphere_start": 3}
    assert env._candidate_rejection_reasons(
        np.array([0.0, 0.0, 2.0], dtype=np.float32),
        np.array([0.1, 0.0, 2.0], dtype=np.float32),
        "anticipatory",
    ) == ["goal_sphere_start"]


def test_r1_reset_boundary_matches_reward_v2_half_meter_sphere():
    env = _bare_env()
    env.reward_version = "v2"
    start = np.array([0.0, 0.0, 2.0], dtype=np.float32)
    assert env._candidate_rejection_reasons(
        start, np.array([0.5, 0.0, 2.0], dtype=np.float32), "anticipatory"
    ) == ["goal_sphere_start"]
    assert env._candidate_rejection_reasons(
        start, np.array([0.6, 0.0, 2.0], dtype=np.float32), "anticipatory"
    ) == []


def test_reset_exhaustion_never_returns_nominal_fallback():
    """Catches the legacy fallback_invalid pair entering the rollout stream."""

    env = _bare_env()
    env._rng = SequenceRng([np.nan] * 21)
    env._astar_3d = lambda *_args: (_ for _ in ()).throw(
        AssertionError("A* must not run for invalid physical candidates")
    )
    with pytest.raises(_terminal().ResetExhausted) as caught:
        env._sample_start_goal("anticipatory")
    assert caught.value.task_identity == "task-world"
    assert caught.value.role == "anticipatory"
    assert caught.value.attempts == 3
    assert caught.value.rejection_histogram == {"nonfinite": 3}
