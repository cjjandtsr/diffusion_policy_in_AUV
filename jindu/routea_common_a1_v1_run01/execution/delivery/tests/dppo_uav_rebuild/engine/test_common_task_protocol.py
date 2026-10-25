"""Finite task boundaries, critic-only context and paired update organization."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from diffusion_policy_3d.rl.dppo_uav_rebuild.engine import critic_state
from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.gae import compute_gae
from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.value import TemporalValueNetwork
from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.initialization_parity import construct_r1_critic_with_parity
from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.rollout_buffer import OnPolicyRolloutBuffer
from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.pair_schedule import iter_pair_passes


def test_task_deadline_and_collection_cut_have_different_bootstrap():
    rewards = torch.tensor([1., 1., 1., 1.])
    values = torch.zeros(4)
    next_values = torch.full((4,), 10.)
    terminated = torch.tensor([False, True, False, False])
    truncated = torch.tensor([False, False, True, True])
    advantages, returns = compute_gae(
        rewards, values, next_values, terminated, truncated, .9, .5,
        bootstrap_allowed=torch.tensor([True, False, False, True]))
    assert torch.allclose(advantages, torch.tensor([10.45, 1., 1., 10.]))
    assert torch.equal(advantages, returns)
    legacy, _ = compute_gae(rewards, values, next_values, terminated, truncated, .9, .5)
    assert legacy[2] == 10.  # Legacy time-limit semantics remain available.


def test_context_tracks_current_next_and_body_rotation_without_actor_mutation():
    assert hasattr(critic_state, "with_episode_context")
    actor = {"state": torch.zeros(1, 576), "global_cond": torch.zeros(1, 576)}
    env = SimpleNamespace(_reward_history={"initial_distance": 20., "best_distance": 10.},
                          _episode_start_world=np.array([0., 0., 2.]),
                          _drone_pos=np.array([0., 10., 4.]), _drone_yaw=math.pi / 2,
                          episode_step=15, max_episode_steps=60)
    before = critic_state.with_episode_context(actor, env)["episode_context"]
    assert torch.allclose(before, torch.tensor([[.5, 2., -1., 0., -.2, .75]]), atol=1e-7)
    env._reward_history["best_distance"] = 8.
    env.episode_step = 60
    after = critic_state.with_episode_context(actor, env)["episode_context"]
    assert after[0, 0] == .4 and after[0, -1] == 0.
    assert before[0, 0] == .5 and set(actor) == {"state", "global_cond"}


def test_context_critic_zero_initialization_preserves_rng_and_has_gradient():
    torch.manual_seed(71)
    legacy = TemporalValueNetwork(hidden_dim=32)
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(71)
    critic, evidence = construct_r1_critic_with_parity(32, "cpu", (), episode_context=True)
    assert torch.equal(expected_rng, torch.get_rng_state())
    assert evidence["value_max_abs_delta"] == 0.
    condition = {"global_cond": torch.randn(2, 576), "episode_context": torch.ones(2, 6)}
    assert torch.equal(legacy(condition), critic(condition))
    critic(condition).sum().backward()
    assert critic.context_weight.grad.abs().sum() > 0
    assert critic.context_weight.shape == (32, 6)
    assert "history_weight" not in critic.state_dict()
    restored = TemporalValueNetwork(hidden_dim=32, episode_context=True)
    restored.load_state_dict(critic.state_dict(), strict=True)
    assert torch.equal(restored(condition), critic(condition))


def test_buffer_preserves_context_alignment_and_explicit_bootstrap():
    buffer = OnPolicyRolloutBuffer(2, episode_context=True)
    for index in range(2):
        buffer.add(condition={"state": torch.zeros(576), "global_cond": torch.zeros(576),
                              "episode_context": torch.full((6,), float(index))},
                   chain=torch.zeros(11, 16, 11), old_logprobs=torch.zeros(10, 16, 11),
                   reward=1., value=0., next_value=10., terminated=False, truncated=True,
                   bootstrap_allowed=bool(index))
    buffer.finalize()
    assert torch.allclose(buffer._returns, torch.tensor([1., 10.9]))
    batch = buffer.materialize_pairs([1, 0, 1], [0, 2, 9], device="cpu", pass_id=0,
                                     mode="full_joint_accum", microbatch_pairs=128,
                                     schedule_audit={})
    assert torch.equal(batch.condition["episode_context"][:, 0], torch.tensor([1., 0., 1.]))


@pytest.mark.parametrize("logical,commits", [(1280, 40), (2560, 20)])
def test_new_schedule_covers_same_51200_pairs(logical, commits):
    buffer = OnPolicyRolloutBuffer(1024)
    for index in range(1024):
        buffer.add(condition={"state": torch.zeros(576), "global_cond": torch.zeros(576)},
                   chain=torch.zeros(11, 16, 11), old_logprobs=torch.zeros(10, 16, 11),
                   reward=0., value=0., next_value=0., terminated=False, truncated=index == 1023)
    buffer.finalize()
    covered = {}
    count = 0
    for batch in iter_pair_passes(buffer, mode="full_joint_accum", pair_passes=5,
                                  logical_batch_pairs=logical, microbatch_pairs=128,
                                  generator=torch.Generator().manual_seed(9), device="cpu"):
        count += 1
        assert batch.pair_count == logical and batch.microbatch_count == logical // 128
        covered.setdefault(batch.pass_id, []).extend(zip(batch.transition_indices.tolist(),
                                                       batch.denoising_inds.tolist()))
    assert count == commits
    assert len(covered) == 5
    assert all(len(pairs) == len(set(pairs)) == 10240 for pairs in covered.values())
