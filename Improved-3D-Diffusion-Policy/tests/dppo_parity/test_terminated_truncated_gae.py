"""GAE-MASK (§3.5, R1.3): per-transition next-value + episode boundaries.

Semantics (R1 guide Task R1.3, fixed formula):

    bootstrap_mask_t = 1.0 - terminated_t          # truncated never masks
    delta_t = r_t + gamma * bootstrap_mask_t * V(next_obs_t) - V(s_t)
    continuation_t = (t+1 < T and episode_ids[t+1] == episode_ids[t]
                      and not terminated_t and not truncated_t)
    A_t = delta_t + gamma * lambda * continuation_t * A_{t+1}

[PROVEN] treating time-limit truncation as terminated zeroes V(s_{t+1})
and introduces a negative terminal bias — truncated never masks bootstrap.
The episode_id check is the last-resort boundary: even if terminated or
truncated were wrongly False, a changed episode id still stops the
advantage chain from crossing episodes.

All expected numbers below are hand-computed with gamma=0.5, lambda=0.5.
"""

import numpy as np

from diffusion_policy_3d.rl.dppo_uav.gae import compute_gae

GAMMA = 0.5
LAM = 0.5


def _close(a, b, atol=1e-6):
    assert abs(a - b) <= atol, f"expected {b}, got {a}"


# --- migrated tests (same hand-computed constants, new API) -----


def test_truncation_bootstraps_next_value():
    """truncated=True at the last step still bootstraps V(next_obs): A_1
    uses next_values[1]=5 (not 0) — the §3.5 anti-bias case."""
    r = np.array([1.0, 1.0])
    v = np.array([2.0, 3.0])
    nv = np.array([3.0, 5.0])  # V(s_1), V(s_T)
    term = np.array([False, False])
    trunc = np.array([False, True])
    adv, ret = compute_gae(
        r, v, nv, term, trunc, ["A", "A"], GAMMA, LAM,
    )
    # t=1: delta = 1 + 0.5*5*1 - 3 = 0.5 ; A_1 = 0.5 (continuation=0: last)
    _close(adv[1], 0.5)
    # t=0: delta = 1 + 0.5*3 - 2 = 0.5 ; A_0 = 0.5 + 0.5*0.5*0.5 = 0.625
    _close(adv[0], 0.625)
    _close(ret[0], 2.625)
    _close(ret[1], 3.5)


def test_terminated_masks_bootstrap():
    """terminated=True at the last step: bootstrap is zeroed — A_1 = delta
    alone even though next_values[1] is nonzero."""
    r = np.array([1.0, 1.0])
    v = np.array([2.0, 3.0])
    nv = np.array([3.0, 5.0])
    term = np.array([False, True])
    trunc = np.array([False, False])
    adv, ret = compute_gae(
        r, v, nv, term, trunc, ["A", "A"], GAMMA, LAM,
    )
    # t=1: delta = 1 + 0.5*5*0 - 3 = -2 ; A_1 = -2
    _close(adv[1], -2.0)
    # t=0: delta = 1 + 0.5*3 - 2 = 0.5 ; A_0 = 0.5 + 0.5*0.5*(-2) = 0.0
    _close(adv[0], 0.0)
    _close(ret[0], 2.0)
    _close(ret[1], 1.0)


def test_terminated_mid_buffer_blocks_cross_episode_propagation():
    """Two episodes in one flat buffer: A_0/A_1 chain stops at the
    terminated boundary; A_2/A_3 belong to the second episode."""
    r = np.zeros(4)
    v = np.array([1.0, 2.0, 3.0, 4.0])
    nv = np.array([2.0, 3.0, 4.0, 10.0])
    term = np.array([False, True, False, False])
    trunc = np.zeros(4, dtype=bool)
    adv, ret = compute_gae(
        r, v, nv, term, trunc, ["A", "A", "B", "B"], GAMMA, LAM,
    )
    # episode 2: t=3 delta = 0 + 0.5*10 - 4 = 1 ; A_3 = 1
    _close(adv[3], 1.0)
    # t=2 delta = 0 + 0.5*4 - 3 = -1 ; A_2 = -1 + 0.5*0.5*1 = -0.75
    _close(adv[2], -0.75)
    # episode 1: t=1 terminated -> A_1 = delta = 0 + 0.5*3*0 - 2 = -2
    _close(adv[1], -2.0)
    # t=0 delta = 0 + 0.5*2*1 - 1 = 0 ; A_0 = 0 + 0.5*0.5*(-2) = -0.5
    #   (A_2 must NOT leak in — only A_1 propagates)
    _close(adv[0], -0.5)
    _close(ret[0], 0.5)
    _close(ret[1], 0.0)  # A_1 + V_1 = -2 + 2


def test_returns_are_advantages_plus_values():
    r = np.array([0.5, -1.0, 2.0])
    v = np.array([1.0, 1.5, 0.5])
    nv = np.array([1.5, 0.5, 2.0])
    term = np.array([False, False, False])
    trunc = np.array([False, False, False])
    adv, ret = compute_gae(
        r, v, nv, term, trunc, ["A", "A", "A"], GAMMA, LAM,
    )
    assert np.allclose(ret, adv + v)


def test_official_formula_structure():
    """One-step chain against the R1.3 fixed formula: bootstrap_mask =
    1 - terminated, delta = r + gamma * mask * V(next_obs) - V(s)."""
    r = np.array([0.3])
    v = np.array([1.1])
    nv = np.array([4.0])
    term = np.array([False])
    trunc = np.array([False])
    adv, _ = compute_gae(r, v, nv, term, trunc, ["A"], GAMMA, LAM)
    delta = r[0] + GAMMA * (1.0 - float(term[0])) * nv[0] - v[0]
    assert abs(adv[0] - delta) <= 1e-6  # A = delta for the single step


# --- Four new analytic RED tests -----------------------------


def test_truncated_mid_buffer_does_not_leak_new_episode_advantage():
    """Truncated transition followed by a NEW episode: A_t must NOT
    receive the new episode's advantage (continuation blocked by the
    episode_id change even though truncated alone would stop it too)."""
    r = np.zeros(3)
    v = np.array([1.0, 2.0, 3.0])
    nv = np.array([2.0, 3.0, 4.0])
    term = np.array([False, False, False])
    trunc = np.array([False, True, False])
    adv, _ = compute_gae(
        r, v, nv, term, trunc, ["A", "A", "B"], GAMMA, LAM,
    )
    # t=2 (epB): delta = 0 + 0.5*4 - 3 = -1 ; A_2 = -1
    _close(adv[2], -1.0)
    # t=1 (epA, truncated): delta = 0 + 0.5*3 - 2 = -0.5 ; continuation=0
    #   (next is a different episode) -> A_1 = -0.5 (NOT -0.5+0.25*(-1))
    _close(adv[1], -0.5)
    # t=0 (epA): delta = 0 + 0.5*2 - 1 = 0 ; A_0 = 0 + 0.25*(-0.5) = -0.125
    _close(adv[0], -0.125)


def test_truncated_uses_own_nonzero_next_value():
    """A truncated transition still bootstraps from its own nonzero
    next_values[t] — truncation never zeroes the bootstrap (anti-bias),
    while the episode_id change still stops the chain."""
    r = np.array([1.0, 0.0])
    v = np.array([2.0, 1.0])
    nv = np.array([1.0, 3.0])
    term = np.array([False, False])
    trunc = np.array([True, False])
    adv, _ = compute_gae(
        r, v, nv, term, trunc, ["A", "B"], GAMMA, LAM,
    )
    # t=1 (epB): delta = 0 + 0.5*3 - 1 = 0.5 ; A_1 = 0.5
    _close(adv[1], 0.5)
    # t=0 (epA, truncated): delta = 1 + 0.5*1 - 2 = -0.5 ; continuation=0
    #   -> A_0 = -0.5. If truncation zeroed the bootstrap: 1 - 2 = -1.
    _close(adv[0], -0.5)


def test_episode_id_change_blocks_chain_even_without_flags():
    """Two episodes with terminated/truncated both wrongly False: the
    episode_id change alone must block the advantage recurrence."""
    r = np.zeros(2)
    v = np.array([1.0, 2.0])
    nv = np.array([2.0, 3.0])
    term = np.array([False, False])
    trunc = np.array([False, False])
    adv, _ = compute_gae(
        r, v, nv, term, trunc, ["A", "B"], GAMMA, LAM,
    )
    # t=1: delta = 0 + 0.5*3 - 2 = -0.5 ; A_1 = -0.5
    _close(adv[1], -0.5)
    # t=0: delta = 0 + 0.5*2 - 1 = 0 ; continuation=0 (id changed)
    #   -> A_0 = 0.0. If the id check were missing: 0.25*(-0.5) = -0.125.
    _close(adv[0], 0.0)


def test_last_sampling_cut_bootstraps_own_next_value():
    """Buffer's last transition is a plain sampling cut (not terminal, not
    time-limit): it bootstraps from its own next_values[t] but its
    continuation is 0 (no t+1)."""
    r = np.array([0.0, 0.5])
    v = np.array([1.0, 1.0])
    nv = np.array([1.0, 4.0])
    term = np.array([False, False])
    trunc = np.array([False, False])
    adv, _ = compute_gae(
        r, v, nv, term, trunc, ["A", "A"], GAMMA, LAM,
    )
    # t=1 (cut): delta = 0.5 + 0.5*4 - 1 = 1.5 ; continuation=0 -> A_1 = 1.5
    #   (if the bootstrap were zeroed: 0.5 - 1 = -0.5)
    _close(adv[1], 1.5)
    # t=0: delta = 0 + 0.5*1 - 1 = -0.5 ; A_0 = -0.5 + 0.25*1.5 = -0.125
    _close(adv[0], -0.125)


# --- Input validation --------------------------------------


def _check_rejected(**kwargs):
    import pytest

    with pytest.raises(ValueError):
        compute_gae(**kwargs)


def test_rewards_values_length_mismatch_rejected():
    _check_rejected(
        rewards=np.array([1.0, 2.0]),
        values=np.array([1.0]),
        next_values=np.array([1.0, 2.0]),
        terminated=np.array([False, False]),
        truncated=np.array([False, False]),
        episode_ids=["A", "A"],
        gamma=GAMMA, gae_lambda=LAM,
    )


def test_empty_buffer_rejected():
    _check_rejected(
        rewards=np.array([]),
        values=np.array([]),
        next_values=np.array([]),
        terminated=np.array([], dtype=bool),
        truncated=np.array([], dtype=bool),
        episode_ids=[],
        gamma=GAMMA, gae_lambda=LAM,
    )


def test_non_finite_input_rejected():
    _check_rejected(
        rewards=np.array([1.0, np.nan]),
        values=np.array([1.0, 1.0]),
        next_values=np.array([1.0, 1.0]),
        terminated=np.array([False, False]),
        truncated=np.array([False, False]),
        episode_ids=["A", "A"],
        gamma=GAMMA, gae_lambda=LAM,
    )


def test_empty_episode_id_rejected():
    _check_rejected(
        rewards=np.array([1.0]),
        values=np.array([1.0]),
        next_values=np.array([1.0]),
        terminated=np.array([False]),
        truncated=np.array([False]),
        episode_ids=[""],
        gamma=GAMMA, gae_lambda=LAM,
    )
