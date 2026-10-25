"""R7-4 (guide Step 6 matrix): per-transition next values — chunked
critic forward over each transition's OWN next_obs.

History: this file originally pinned the R1-era chunked update-start
next-value critic forward, then the R5.8-B successor-value shortcut.
R7-4 deletes the shortcut (it leaked the successor transition's
value_old across reset/interleave boundaries) — the file now pins the
R7-NEXT-VALUE contract:

  - fill_buffer_next_values(buffer, critic, chunk_size) evaluates EVERY
    transition (terminated finals included — GAE's bootstrap_mask masks
    their bootstrap contribution, never the fill) over ITS OWN next_obs,
    chunked in fixed contiguous transition order;
  - evaluate_transition_next_values(next_obs_seq, critic, chunk_size)
    returns the (N,) FP32 value tensor; chunking never changes values
    (bitwise-identical across legal chunk sizes);
  - ppo_update READS buffer.next_values and hard-fails when unfilled;
  - compute_gae recurses per EPISODE — an interleaved vector-env buffer
    (A0,B0,A1,B1,..) matches a per-slot oracle, never one-step TD.

The guide's Step 1 RED counterexample (A0.next_value == 3.0, never
A1.value_old == 10.0) lives in test_interleaved_next_value_provenance.py.

Run: pytest -q tests/dppo_parity/test_next_value_chunking.py
"""

from typing import List, Optional

import numpy as np
import pytest
import torch

from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.gae import compute_gae
from diffusion_policy_3d.rl.dppo_uav.rollout_buffer import (
    RolloutBuffer,
    RolloutTransition,
    fill_buffer_next_values,
)
from diffusion_policy_3d.rl.dppo_uav.trainer import (
    TrainerConfig,
    ppo_update,
)

from facade_testutil import make_test_context


def _single_episode_buffer(n: int, *, terminated_final: bool,
                           truncated_final: bool):
    contract = RLSystemContract()
    rng = np.random.RandomState(3)
    buf = RolloutBuffer(contract)
    obs_list = []
    for i in range(n):
        agent_pos = rng.uniform(-1, 1, (3, 9)).astype(np.float32)
        agent_pos[0, 0] = float(i)
        obs_list.append({
            "point_cloud": rng.uniform(0.3, 9.5, (3, 4096, 3)).astype(np.float32),
            "agent_pos": agent_pos,
        })
    final_obs = {
        "point_cloud": rng.uniform(0.3, 9.5, (3, 4096, 3)).astype(np.float32),
        "agent_pos": rng.uniform(-1, 1, (3, 9)).astype(np.float32),
    }
    for i in range(n):
        buf.add(RolloutTransition(
            obs=obs_list[i],
            next_obs=obs_list[i + 1] if i + 1 < n else final_obs,
            chains_policy=rng.uniform(-1, 1, (
                contract.ft_denoising_steps + 1, contract.action_horizon,
                contract.action_dim,
            )).astype(np.float32),
            old_logprobs=rng.uniform(-3, 0, (
                contract.ft_denoising_steps, contract.action_horizon,
                contract.action_dim,
            )).astype(np.float32),
            reward=float(rng.uniform(-1, 1)),
            terminated=terminated_final and i == n - 1,
            truncated=truncated_final and i == n - 1,
            value_old=float(rng.uniform(-2, 2)),
            episode_id="ep0",
            episode_step=i,
            world_uid="w1",
            role="anticipatory",
            manifest_entry_hash="mh-mock",
            episode_config_hash="ch-mock",
            observation_valid=True,
            env_id=0,
        ))
    return buf, contract


# ═══════════════════════════════════════════════════════════════════
# R7-4 Step 6 matrix — marked-buffer fixture (guide Step 1 style:
# direct parallel-list writes, invariant-3-exempt so each transition's
# next_obs can carry its own marker; the FILL is the production path).
# ═══════════════════════════════════════════════════════════════════

def _agent_pos_marker(v: float) -> np.ndarray:
    """(3, 9) agent_pos with the value marker at [0, 0]."""
    s = np.zeros((3, 9), dtype=np.float32)
    s[0, 0] = v
    return s


def _make_marked_buffer(
    n: int,
    *,
    next_markers: Optional[List[float]] = None,
    episode_ids: Optional[List[str]] = None,
    terminated: Optional[List[bool]] = None,
    truncated: Optional[List[bool]] = None,
    episode_steps: Optional[List[int]] = None,
) -> RolloutBuffer:
    """Flat buffer with full obs key set (point_cloud + agent_pos).

    obs[i] marker = i; next_obs[i] marker = next_markers[i]
    (default 100+i — never equal to any obs marker). value_old = 0.0
    throughout, so any successor-shortcut leak is immediately visible
    as a next_values entry != its own next_obs marker.
    """
    buf = RolloutBuffer(contract=object())
    rng = np.random.RandomState(3)
    buf.episode_ids[:] = (
        episode_ids if episode_ids is not None else ["ep0"] * n
    )
    buf.values[:] = [0.0] * n
    buf.terminated[:] = (
        terminated if terminated is not None else [False] * n
    )
    buf.truncated[:] = (
        truncated if truncated is not None else [False] * n
    )
    buf.episode_steps[:] = (
        episode_steps if episode_steps is not None else list(range(n))
    )
    buf.obs[:] = [
        {
            "point_cloud": rng.uniform(
                0.3, 9.5, (3, 4096, 3)
            ).astype(np.float32),
            "agent_pos": _agent_pos_marker(float(i)),
        }
        for i in range(n)
    ]
    markers = (
        next_markers
        if next_markers is not None
        else [100.0 + i for i in range(n)]
    )
    buf.next_obs[:] = [
        {
            "point_cloud": rng.uniform(
                0.3, 9.5, (3, 4096, 3)
            ).astype(np.float32),
            "agent_pos": _agent_pos_marker(markers[i]),
        }
        for i in range(n)
    ]
    return buf


def _marker_critic(obs_b):
    """Critic maps the agent_pos marker at [0,0] to the value."""
    return obs_b["agent_pos"][:, 0, 0].float()


# ───────────────────────────────────────────────────────────────────
# Kept legacy contracts (unchanged semantics)
# ───────────────────────────────────────────────────────────────────

def test_ppo_update_hard_fails_unfilled_next_values():
    """B-1: an unfilled buffer (None placeholders) must hard-fail the
    update BEFORE the facade is touched — a silent fallback to
    update-start critic values would resurrect the two-value-function
    mixing."""
    buf, contract = _single_episode_buffer(
        4, terminated_final=False, truncated_final=True)
    config = TrainerConfig(
        update_epochs=1, batch_size=4, max_grad_norm=None, target_kl=None,
        ent_coef=0.0, vf_coef=1.0, device="cpu",
    )
    policy, facade, encode_cond, critic = make_test_context(contract)
    with pytest.raises(FloatingPointError):
        ppo_update(
            buffer=buf, contract=contract, facade=facade,
            encode_cond=encode_cond,
            actor_optimizer=torch.optim.Adam(
                policy.model.parameters(), lr=1e-3),
            critic_optimizer=torch.optim.Adam(
                critic.parameters(), lr=1e-3),
            config=config,
        )


def test_interleaved_gae_matches_per_slot_oracle():
    """N3 core: an interleaved two-slot buffer (A0,B0,A1,B1,A2) must
    match a per-episode reference recursion, not one-step TD — GAE's
    dual-mask semantics are untouched by R7-4."""
    rng = np.random.RandomState(11)
    # per-slot data, in slot order
    slot_a = {"ep": "epA", "r": [0.1, 0.2, -0.5], "v": [1.0, 0.8, 0.6],
              "nv": [0.8, 0.6, 0.0], "term": [False, False, True],
              "trunc": [False, False, False]}
    slot_b = {"ep": "epB", "r": [0.4, -0.3], "v": [0.5, 0.4],
              "nv": [0.4, 0.35], "term": [False, False],
              "trunc": [False, True]}
    gamma, lam = 0.99, 0.95

    def reference(slot):
        n = len(slot["r"])
        adv = np.zeros(n)
        last = 0.0
        for pos in range(n - 1, -1, -1):
            bootstrap = 1.0 - float(slot["term"][pos])
            delta = (slot["r"][pos] + gamma * bootstrap * slot["nv"][pos]
                     - slot["v"][pos])
            cont = float(pos + 1 < n and not slot["term"][pos]
                         and not slot["trunc"][pos])
            adv[pos] = last = delta + gamma * lam * cont * last
        return adv

    adv_a, adv_b = reference(slot_a), reference(slot_b)
    # flat interleave order A0,B0,A1,B1,A2
    oracle = np.array([adv_a[0], adv_b[0], adv_a[1], adv_b[1], adv_a[2]])
    # interleave flat arrays: A0,B0,A1,B1,A2
    flat = {
        "r": [slot_a["r"][0], slot_b["r"][0], slot_a["r"][1],
              slot_b["r"][1], slot_a["r"][2]],
        "v": [slot_a["v"][0], slot_b["v"][0], slot_a["v"][1],
              slot_b["v"][1], slot_a["v"][2]],
        "nv": [slot_a["nv"][0], slot_b["nv"][0], slot_a["nv"][1],
               slot_b["nv"][1], slot_a["nv"][2]],
        "term": [slot_a["term"][0], slot_b["term"][0], slot_a["term"][1],
                 slot_b["term"][1], slot_a["term"][2]],
        "trunc": [slot_a["trunc"][0], slot_b["trunc"][0], slot_a["trunc"][1],
                  slot_b["trunc"][1], slot_a["trunc"][2]],
        "ids": ["epA", "epB", "epA", "epB", "epA"],
    }
    adv, ret = compute_gae(
        np.asarray(flat["r"]), np.asarray(flat["v"]), np.asarray(flat["nv"]),
        np.asarray(flat["term"]), np.asarray(flat["trunc"]),
        flat["ids"], gamma, lam,
    )
    assert np.allclose(adv, oracle, atol=1e-12), (adv, oracle)


def test_next_value_batch_size_validation():
    """R7-4: TrainerConfig validates next_value_batch_size is None or
    > 0 — the field is the production chunk-size channel for the
    per-transition next-value fill."""
    TrainerConfig(next_value_batch_size=None)  # single forward: valid
    TrainerConfig(next_value_batch_size=64)  # production default: valid
    with pytest.raises(ValueError):
        TrainerConfig(next_value_batch_size=0)
    with pytest.raises(ValueError):
        TrainerConfig(next_value_batch_size=-1)


def test_batch_size_must_exceed_one():
    """Rework #2 (audit 2026-08-08): the optimizer minibatch batch_size
    must be > 1 — a B=1 minibatch would make the ratio_std diagnostic
    degenerate and is never a valid production configuration."""
    TrainerConfig(batch_size=128)  # production default: valid
    with pytest.raises(ValueError):
        TrainerConfig(batch_size=1)
    with pytest.raises(ValueError):
        TrainerConfig(batch_size=0)


# ───────────────────────────────────────────────────────────────────
# R7-4 Step 6 matrix
# ───────────────────────────────────────────────────────────────────

def test_matrix_single_episode_all_values_from_own_next_obs():
    """single episode: every next value is critic(own next_obs) — the
    100+i markers, never the adjacent value_old (0.0)."""
    buf = _make_marked_buffer(5)
    fill_buffer_next_values(buf, _marker_critic, chunk_size=2)
    assert buf.next_values == [100.0, 101.0, 102.0, 103.0, 104.0]
    p = buf.next_value_provenance
    assert p is not None
    assert p["next_value_source"] == "critic(each_transition.next_obs)"
    assert p["next_obs_count"] == 5
    assert p["next_value_count"] == 5
    assert p["chunk_size"] == 2
    assert len(p["next_obs_hash"]) == 16
    assert len(p["next_value_hash"]) == 16
    assert p["interleaved_episode_count"] == 1


def test_matrix_terminated_final_still_evaluated_from_own_next_obs():
    """terminated final: evaluated like every other transition — GAE's
    bootstrap_mask (1 - terminated) masks its contribution; the fill
    never zeroes it."""
    buf = _make_marked_buffer(4, terminated=[False, False, False, True])
    fill_buffer_next_values(buf, _marker_critic, chunk_size=3)
    assert buf.next_values[3] == 103.0  # own next_obs marker, not 0.0


def test_matrix_truncated_final_evaluated_from_own_next_obs():
    """truncated final: bootstraps from its own next_obs value."""
    buf = _make_marked_buffer(4, truncated=[False, False, False, True])
    fill_buffer_next_values(buf, _marker_critic, chunk_size=3)
    assert buf.next_values[3] == 103.0


def test_matrix_two_interleaved_episodes_no_successor_leak():
    """two interleaved episodes A0,B0,A1,B1: each value comes from its
    own next_obs; A1's marker must never leak into A0 (successor
    shortcut)."""
    buf = _make_marked_buffer(
        4,
        episode_ids=["A", "B", "A", "B"],
        next_markers=[3.0, 4.0, 5.0, 6.0],
    )
    fill_buffer_next_values(buf, _marker_critic, chunk_size=2)
    assert buf.next_values == [3.0, 4.0, 5.0, 6.0]
    assert buf.next_value_provenance["interleaved_episode_count"] == 2


def test_matrix_episode_step_gap_own_next_obs_still_used():
    """episode step gap: the fill never assumes flat adjacency or
    consecutive steps — every transition still gets critic(own
    next_obs)."""
    buf = _make_marked_buffer(
        3,
        episode_steps=[0, 1, 5],
        next_markers=[7.0, 8.0, 9.0],
    )
    fill_buffer_next_values(buf, _marker_critic, chunk_size=1)
    assert buf.next_values == [7.0, 8.0, 9.0]


def test_matrix_next_obs_missing_hard_fails():
    """next_obs missing: a None next_obs must hard-fail the fill with a
    clear error before any forward."""
    buf = _make_marked_buffer(4)
    buf.next_obs[2] = None
    with pytest.raises(ValueError, match="missing"):
        fill_buffer_next_values(buf, _marker_critic, chunk_size=2)


def test_matrix_next_obs_key_mismatch_hard_fails():
    """next_obs missing a key: the unified collate requires the full
    key set — a transition dropping point_cloud hard-fails."""
    buf = _make_marked_buffer(4)
    buf.next_obs[2] = {"agent_pos": _agent_pos_marker(1.0)}
    with pytest.raises(ValueError, match="key mismatch"):
        fill_buffer_next_values(buf, _marker_critic, chunk_size=2)


def test_matrix_nan_critic_hard_fails():
    """NaN critic: non-finite next values must never reach GAE —
    FloatingPointError."""
    buf = _make_marked_buffer(4)

    def nan_critic(obs_b):
        return torch.full(
            (obs_b["agent_pos"].shape[0],), float("nan")
        )

    with pytest.raises(FloatingPointError):
        fill_buffer_next_values(buf, nan_critic, chunk_size=2)


def test_matrix_chunk_sizes_bitwise_consistent():
    """chunk sizes 1/7/128/N+1 (+ None = single forward): all legal
    chunk sizes produce bitwise-identical fills; the provenance records
    the RESOLVED chunk size (min(chunk_size, n))."""
    baseline = None
    for cs in [1, 7, 128, None, 9]:
        b = _make_marked_buffer(8)
        fill_buffer_next_values(b, _marker_critic, chunk_size=cs)
        if baseline is None:
            baseline = list(b.next_values)
            continue
        assert list(b.next_values) == baseline, f"chunk_size={cs} differs"
    assert baseline is not None
    b1 = _make_marked_buffer(8)
    fill_buffer_next_values(b1, _marker_critic, chunk_size=7)
    assert b1.next_value_provenance["chunk_size"] == 7
    b2 = _make_marked_buffer(8)
    fill_buffer_next_values(b2, _marker_critic, chunk_size=128)
    assert b2.next_value_provenance["chunk_size"] == 8  # resolved min
    b3 = _make_marked_buffer(8)
    fill_buffer_next_values(b3, _marker_critic, chunk_size=None)
    assert b3.next_value_provenance["chunk_size"] == 8  # single forward
