"""Authoritative three-dimensional arrival and terminal precedence tests."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
import numpy as np
import pytest

from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.kinematic_adapter import (
    KinematicAdapter,
)


def _terminal():
    return importlib.import_module(
        "diffusion_policy_3d.rl.dppo_uav_rebuild.environments.terminal_contract"
    )


@pytest.mark.parametrize(
    "distance,expected",
    [
        (np.nextafter(0.5, 0.0), True),
        (0.5, True),
        (np.nextafter(0.5, np.inf), False),
    ],
)
def test_reached_goal_uses_closed_three_dimensional_boundary(distance, expected):
    """Catches strict-inequality or XY-plus-extra-Z arrival definitions."""

    assert _terminal().reached_goal(np.array([distance, 0.0, 0.0]), np.zeros(3)) is expected


def test_reached_goal_allows_z_error_inside_the_three_dimensional_ball():
    """Catches retaining the legacy abs(dz)<0.3 add-on."""

    assert _terminal().reached_goal(np.array([0.0, 0.0, 0.4]), np.zeros(3)) is True


@pytest.mark.parametrize(
    "position,goal,radius",
    [
        ([0.0, 0.0], [0.0, 0.0, 0.0], 0.5),
        ([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.5),
        ([np.nan, 0.0, 0.0], [0.0, 0.0, 0.0], 0.5),
        ([0.0, 0.0, 0.0], [np.inf, 0.0, 0.0], 0.5),
        ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0),
        ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], np.inf),
    ],
)
def test_reached_goal_rejects_malformed_or_nonfinite_inputs(position, goal, radius):
    """Catches malformed state being converted into a successful arrival."""

    with pytest.raises((TypeError, ValueError)):
        _terminal().reached_goal(position, goal, radius)


def test_collision_wins_over_reached_and_clears_success():
    """Catches awarding success on the same transition as obstacle contact."""

    facts = _terminal().resolve_terminal(
        reached=True,
        collision=True,
        ground_crash=False,
        out_of_bounds=False,
        altitude_exceed=False,
        invalid_state=False,
        time_limit=False,
    )
    assert facts.done_reason == "collision"
    assert facts.reached is False
    assert facts.terminated is True
    assert facts.truncated is False


@pytest.mark.parametrize(
    "active,reason",
    [
        ({"invalid_state": True, "collision": True}, "invalid_state"),
        ({"collision": True, "ground_crash": True}, "collision"),
        ({"ground_crash": True, "out_of_bounds": True}, "ground_crash"),
        ({"out_of_bounds": True, "altitude_exceed": True}, "out_of_bounds"),
        ({"altitude_exceed": True, "reached": True}, "altitude_exceed"),
        ({"reached": True, "time_limit": True}, "reached"),
        ({"time_limit": True}, "time_limit"),
        ({}, "none"),
    ],
)
def test_terminal_precedence_is_total_and_deterministic(active, reason):
    """Catches a reward branch choosing a lower-priority terminal fact."""

    values = dict(
        reached=False,
        collision=False,
        ground_crash=False,
        out_of_bounds=False,
        altitude_exceed=False,
        invalid_state=False,
        time_limit=False,
    )
    values.update(active)
    assert _terminal().resolve_terminal(**values).done_reason == reason


def test_adapter_consumes_environment_terminal_record_without_recomputing():
    """Catches the adapter retaining its legacy XY-plus-Z success branch."""

    facts = _terminal().resolve_terminal(
        reached=True,
        collision=False,
        ground_crash=False,
        out_of_bounds=False,
        altitude_exceed=False,
        invalid_state=False,
        time_limit=False,
    )

    class Environment:
        _drone_pos = np.array([0.0, 0.0, 0.4])
        goal_w = np.zeros(3)
        goal_tol_xy = 0.5
        goal_tol_z = 0.3
        _reward_history = {}

        def step(self, _action):
            return None, 0.0, True, False, {
                "transition_facts": facts,
                "reward_terms": {},
                "reward_metrics": {},
            }

        def _check_collision(self, **_kwargs):
            return False

    class Reward:
        version = "v13"

        def terminal(self, reason, state, _config):
            return SimpleNamespace(total=7.0, state=state, terms={}, metrics={})

    adapter = KinematicAdapter(Environment(), Reward(), SimpleNamespace(history={}))
    _, reward, terminated, truncated, info = adapter.step(
        SimpleNamespace(policy=np.zeros((8, 11)))
    )
    assert reward == 7.0
    assert terminated is True and truncated is False
    assert info["transition_facts"] is facts
    assert info["done_reason"] == "reached"
