"""GAE-MASK §4.11: rollout buffer invariants + update identity.

RolloutTransition per design §4.11; the 6 invariants are enforced at
add() time. [JUDGMENT] next_obs is written in the same add() call —
there is no cross-iteration prev-frame auto-fill parameter.
"""

import numpy as np
import pytest

from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.rollout_buffer import (
    RolloutBuffer,
    RolloutTransition,
)

D_A = 11


def _obs(seed: int, n_frames: int = 3) -> dict:
    rng = np.random.RandomState(seed)
    return {
        "point_cloud": rng.uniform(0.3, 9.5, (n_frames, 4096, 3)).astype(np.float32),
        "state": rng.uniform(-1, 1, (n_frames, 9)).astype(np.float32),
    }


def _chains(contract: RLSystemContract, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.uniform(-1, 1, (
        contract.ft_denoising_steps + 1,
        contract.action_horizon,
        D_A,
    )).astype(np.float32)


def _logprobs(contract: RLSystemContract, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.uniform(-3, 0, (
        contract.ft_denoising_steps,
        contract.action_horizon,
        D_A,
    )).astype(np.float32)


def _trans(contract: RLSystemContract, episode_id: str, step: int,
           seed: int, obs=None, next_obs=None,
           manifest_hash: str = "mh", episode_hash: str = "eh") -> RolloutTransition:
    return RolloutTransition(
        obs=obs if obs is not None else _obs(seed),
        next_obs=next_obs if next_obs is not None else _obs(seed + 1000),
        chains_policy=_chains(contract, seed),
        old_logprobs=_logprobs(contract, seed),
        reward=float(seed),
        terminated=False,
        truncated=False,
        value_old=float(seed) / 10.0,
        episode_id=episode_id,
        episode_step=step,
        world_uid="w1",
        role="anticipatory",
        observation_valid=True,
        manifest_entry_hash=manifest_hash,
        episode_config_hash=episode_hash,
    )


# ---------- invariant 1: equal lengths ----------

def test_invariant1_equal_lengths():
    contract = RLSystemContract()
    buf = RolloutBuffer(contract)
    t0 = _trans(contract, "ep1", 0, seed=1)
    buf.add(t0)
    # same episode: obs must chain onto t0.next_obs (invariant 3 precondition)
    buf.add(_trans(contract, "ep1", 1, seed=2, obs=t0.next_obs))
    assert len(buf.obs) == len(buf.next_obs) == len(buf.rewards) == len(
        buf.terminated
    ) == len(buf.truncated) == len(buf.episode_ids) == 2


# ---------- invariant 2: episode_step strictly increases within episode ----------

def test_invariant2_step_strictly_increases():
    contract = RLSystemContract()
    buf = RolloutBuffer(contract)
    t0 = _trans(contract, "ep1", 0, seed=1)
    buf.add(t0)
    t1 = _trans(contract, "ep1", 1, seed=2, obs=t0.next_obs)
    buf.add(t1)
    with pytest.raises(AssertionError):
        buf.add(_trans(contract, "ep1", 1, seed=3, obs=t1.next_obs))  # dup step


# ---------- invariant 3: next_obs[t] == obs[t+1] when no reset ----------

def test_invariant3_next_obs_identity_within_episode():
    contract = RLSystemContract()
    buf = RolloutBuffer(contract)
    obs0 = _obs(10)
    obs1 = _obs(20)
    buf.add(_trans(contract, "ep1", 0, seed=1, obs=obs0, next_obs=obs1))
    buf.add(_trans(contract, "ep1", 1, seed=2, obs=obs1, next_obs=_obs(30)))
    # invariant holds silently; buffer length reached
    assert len(buf.obs) == 2


def test_invariant3_next_obs_mismatch_rejected():
    """next_obs[t] != obs[t+1] within a continuing episode -> add() fails."""
    contract = RLSystemContract()
    buf = RolloutBuffer(contract)
    obs0 = _obs(10)
    buf.add(_trans(contract, "ep1", 0, seed=1, obs=obs0, next_obs=_obs(20)))
    with pytest.raises(AssertionError):
        buf.add(_trans(contract, "ep1", 1, seed=2, obs=_obs(99),
                       next_obs=_obs(30)))


def test_invariant3_new_episode_can_diverge():
    """A reset starts a new episode: obs[new] need not equal next_obs[old]."""
    contract = RLSystemContract()
    buf = RolloutBuffer(contract)
    buf.add(_trans(contract, "ep1", 0, seed=1, obs=_obs(10), next_obs=_obs(20)))
    buf.add(_trans(contract, "ep2", 0, seed=2, obs=_obs(99), next_obs=_obs(30)))
    assert len(buf.obs) == 2


# ---------- invariants 4+5: chain and logprob shapes ----------

def test_invariant4_5_chain_and_logprob_shapes():
    contract = RLSystemContract()
    buf = RolloutBuffer(contract)
    t = _trans(contract, "ep1", 0, seed=1)
    assert t.chains_policy.shape == (
        contract.ft_denoising_steps + 1, contract.action_horizon, D_A,
    )
    assert t.old_logprobs.shape == (
        contract.ft_denoising_steps, contract.action_horizon, D_A,
    )
    buf.add(t)  # accepted


def test_invariant4_5_wrong_shapes_rejected():
    contract = RLSystemContract()
    buf = RolloutBuffer(contract)
    t = _trans(contract, "ep1", 0, seed=1)
    bad_chains = t.chains_policy[:-1]  # (K_f, H_a, D_A)
    bad_t = RolloutTransition(
        obs=t.obs, next_obs=t.next_obs, chains_policy=bad_chains,
        old_logprobs=t.old_logprobs, reward=t.reward,
        terminated=t.terminated, truncated=t.truncated, value_old=t.value_old,
        episode_id=t.episode_id, episode_step=t.episode_step,
        world_uid=t.world_uid, role=t.role,
        observation_valid=t.observation_valid,
        manifest_entry_hash=t.manifest_entry_hash,
        episode_config_hash=t.episode_config_hash,
    )
    with pytest.raises(AssertionError):
        buf.add(bad_t)


# ---------- invariant 6: trainable transitions all observation_valid ----------

def test_invariant6_invalid_transition_rejected():
    contract = RLSystemContract()
    buf = RolloutBuffer(contract)
    t = _trans(contract, "ep1", 0, seed=1)
    bad_t = RolloutTransition(
        obs=t.obs, next_obs=t.next_obs, chains_policy=t.chains_policy,
        old_logprobs=t.old_logprobs, reward=t.reward,
        terminated=t.terminated, truncated=t.truncated, value_old=t.value_old,
        episode_id=t.episode_id, episode_step=t.episode_step,
        world_uid=t.world_uid, role=t.role,
        observation_valid=False,
        manifest_entry_hash=t.manifest_entry_hash,
        episode_config_hash=t.episode_config_hash,
    )
    with pytest.raises(AssertionError):
        buf.add(bad_t)


# ---------- update identity: read-back is exact, no silent mutation ----------

def test_update_identity_readback_exact():
    """What goes in is exactly what comes out — byte-identical arrays."""
    contract = RLSystemContract()
    buf = RolloutBuffer(contract)
    t0 = _trans(contract, "ep1", 0, seed=1)
    t1 = _trans(contract, "ep1", 1, seed=2, obs=t0.next_obs, next_obs=_obs(30))
    buf.add(t0)
    buf.add(t1)
    assert np.array_equal(buf.obs[0]["point_cloud"], t0.obs["point_cloud"])
    assert np.array_equal(buf.next_obs[1]["point_cloud"], t1.next_obs["point_cloud"])
    assert np.array_equal(buf.chains[1], t1.chains_policy)
    assert np.array_equal(buf.logprobs[0], t0.old_logprobs)
    assert buf.rewards == [t0.reward, t1.reward]
    # arrays are stored by reference; caller mutation is caller's contract,
    # but the buffer must never copy-and-alter
    assert len(buf) == 2


def test_update_identity_clear_and_readd():
    contract = RLSystemContract()
    buf = RolloutBuffer(contract)
    t = _trans(contract, "ep1", 0, seed=7)
    buf.add(t)
    buf.clear()
    assert len(buf) == 0
    buf.add(t)
    assert np.array_equal(buf.obs[0]["state"], t.obs["state"])
    assert buf.episode_ids == ["ep1"]


# ---------- Dual-hash trace on every transition ----------

def test_transition_carries_dual_hashes():
    """R2.3: every transition carries manifest_entry_hash (manifest source)
    + episode_config_hash (sampled episode); the buffer traces both."""
    contract = RLSystemContract()
    buf = RolloutBuffer(contract)
    t0 = _trans(contract, "ep1", 0, seed=1,
                manifest_hash="mh-a", episode_hash="ch-1")
    t1 = _trans(contract, "ep1", 1, seed=2, obs=t0.next_obs,
                manifest_hash="mh-a", episode_hash="ch-2")
    buf.add(t0)
    buf.add(t1)
    assert buf.manifest_entry_hashes == ["mh-a", "mh-a"]
    assert buf.episode_config_hashes == ["ch-1", "ch-2"]
    assert t0.manifest_entry_hash == "mh-a"
    assert t1.episode_config_hash == "ch-2"
