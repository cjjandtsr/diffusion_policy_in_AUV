"""OBS-RECENCY: select_latest_observations returns the LATEST obs_steps frames.

Design §2.2 / §4.4: input chronological (B, T_history, ...); output
(B, obs_steps, ...) with history_indices = [T_h-T_o, ..., T_h-1] = [5,6,7].
This kills the P0-OBS-1 bug (x[:, :n_obs_steps] took the OLDEST frames).
"""

import torch
import pytest

from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.observation_adapter import (
    select_latest_observations,
)

B, T_H, N_PTS, S_DIM = 2, 8, 4096, 9


def _sentinel_obs():
    """Each frame t is filled with value t so slicing direction is verifiable."""
    t = torch.arange(T_H).reshape(1, T_H, 1, 1).float()  # (1, T, 1, 1)
    point_cloud = t.expand(B, T_H, N_PTS, 3)  # (B, T, 4096, 3)
    t9 = torch.arange(T_H).reshape(1, T_H, 1).float()  # (1, T, 1)
    agent_pos = t9.expand(B, T_H, S_DIM)  # (B, T, 9)
    return {"point_cloud": point_cloud, "agent_pos": agent_pos}


def test_returns_latest_three_frames_not_first():
    """Regression for P0-OBS-1: must be frames [5,6,7], never [0,1,2]."""
    obs = _sentinel_obs()
    out, meta = select_latest_observations(obs, obs_steps=3)
    assert meta.history_indices == [5, 6, 7]
    for k in ("point_cloud", "agent_pos"):
        assert torch.all(out[k][:, 0] == 5.0)
        assert torch.all(out[k][:, 1] == 6.0)
        assert torch.all(out[k][:, 2] == 7.0)
        assert not torch.all(out[k][:, 0] == 0.0)  # old bug: first frames


def test_output_shapes_match_design():
    """point_cloud (B,3,4096,3), agent_pos (B,3,9) — design §4.4."""
    out, _ = select_latest_observations(_sentinel_obs(), obs_steps=3)
    assert out["point_cloud"].shape == (B, 3, N_PTS, 3)
    assert out["agent_pos"].shape == (B, 3, S_DIM)


def test_obs_steps_one_returns_last_frame():
    out, meta = select_latest_observations(_sentinel_obs(), obs_steps=1)
    assert meta.history_indices == [7]
    assert torch.all(out["agent_pos"][:, 0] == 7.0)


def test_rejects_zero_obs_steps():
    with pytest.raises(AssertionError):
        select_latest_observations(_sentinel_obs(), obs_steps=0)


def test_rejects_obs_steps_exceeding_history():
    with pytest.raises(AssertionError):
        select_latest_observations(_sentinel_obs(), obs_steps=T_H + 1)


def test_metadata_written_for_debug_trace():
    """Design §4.4: selection metadata is part of the return (debug trace)."""
    _, meta = select_latest_observations(_sentinel_obs(), obs_steps=3)
    assert meta.history_steps == T_H
    assert meta.obs_steps == 3


def test_matches_default_contract():
    c = RLSystemContract()
    out, _ = select_latest_observations(_sentinel_obs(), obs_steps=c.obs_steps)
    assert out["point_cloud"].shape[1] == c.obs_steps
    assert out["point_cloud"].shape[2] == c.point_count
    assert out["point_cloud"].shape[3] == c.point_dim
    assert out["agent_pos"].shape[2] == c.state_dim
