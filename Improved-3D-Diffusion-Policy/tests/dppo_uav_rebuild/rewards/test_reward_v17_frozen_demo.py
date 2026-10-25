"""Frozen-demo export preserves data and rejects unexplained coordinate changes."""

import math

import numpy as np
import pytest

from diffusion_policy_3d.env.reward_v17_support.density import build_density
from diffusion_policy_3d.env.reward_v17_support.frozen_demo import (
    REQUIRED_SOURCE_BASENAMES, array_sha256, export_frozen_demo,
)


def encoded_state(position, goal, yaw):
    position, goal = np.asarray(position, dtype=float), np.asarray(goal, dtype=float)
    c, s = math.cos(yaw), math.sin(yaw)
    delta = goal - position
    body = [c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1], delta[2]]
    return [*position, *body, np.linalg.norm(delta), s, c]


def export(state, episode_ends, source_files=None):
    state = np.asarray(state, dtype=np.float32)
    episode_ends = np.asarray(episode_ends, dtype=np.int64)
    return export_frozen_demo(
        state, episode_ends, ground_z=0., state_array_sha256=array_sha256(state),
        episode_ends_array_sha256=array_sha256(episode_ends),
        source_files=source_files if source_files is not None else
        [dict(path=name, sha256="c" * 64) for name in sorted(REQUIRED_SOURCE_BASENAMES)])


def test_export_recovers_goal_and_only_removes_proven_stale_first_frame():
    fixed = np.array([encoded_state([i, 0, 2], [10, 0, 2], 0) for i in range(3)])
    stale = np.array([
        encoded_state([0, 1, 2], [-10, 0, 2], .2),
        encoded_state([1, 1, 2], [10, 2, 2.5], .2),
        encoded_state([2, 1, 2], [10, 2, 2.5], .2),
    ])
    state = np.concatenate([fixed, stale]).astype(np.float32)

    result = export(state, [3, 6])

    assert result["summary"] == {
        "episodes": 2, "source_frames": 6, "exported_frames": 5,
        "stale_goal_prefix_episodes": 1, "dropped_stale_goal_frames": 1,
        "negative_z_frames_retained": 0,
    }
    assert result["episodes"][0]["goal"] == pytest.approx([10, 0, 2])
    assert result["episodes"][1]["goal"] == pytest.approx([10, 2, 2.5])
    assert result["episodes"][1]["start"] == pytest.approx([1, 1, 2])
    assert result["episodes"][1]["source_frame_range"] == [4, 6]
    assert result["episodes"][1]["dropped_prefix_frames"] == 1
    assert result["episodes"][1]["initial_yaw"] == pytest.approx(.2)
    density = build_density(result["episodes"], dataset_sha256="d" * 64)
    assert density["episode_count"] == 2 and density["sample_count"] == 5


def test_export_keeps_negative_z_as_source_fact():
    state = np.array([
        encoded_state([0, 0, 1], [2, 0, 1], 0),
        encoded_state([1, 0, -.2], [2, 0, 1], 0),
    ])
    result = export(state, [2])
    assert result["summary"]["negative_z_frames_retained"] == 1
    assert result["episodes"][0]["positions"][1][2] == pytest.approx(-.2)


def test_export_rejects_any_unexplained_later_goal_change():
    state = np.array([
        encoded_state([0, 0, 1], [3, 0, 1], 0),
        encoded_state([1, 0, 1], [3, 0, 1], 0),
        encoded_state([2, 0, 1], [30, 0, 1], 0),
    ])
    with pytest.raises(ValueError, match="unexplained goal change"):
        export(state, [3])


def test_export_rejects_broken_state_contract_and_missing_evidence():
    state = np.array([encoded_state([0, 0, 1], [3, 0, 1], 0)])
    state[0, 8] = .5
    with pytest.raises(ValueError, match="yaw unit circle"):
        export(state, [1])
    with pytest.raises(ValueError, match="source evidence"):
        export([encoded_state([0, 0, 1], [3, 0, 1], 0)], [1], source_files=[])
    with pytest.raises(ValueError, match="exactly the eight"):
        export([encoded_state([0, 0, 1], [3, 0, 1], 0)], [1], source_files=[
            dict(path="ego_gazebo.sh", sha256="c" * 64)])
