"""R1 rework P0 (audit 2026-08-08): official per-epoch randperm semantics.

Official DPPO (third_party/dppo_official_cc7234a/agent/finetune/
train_ppo_diffusion_agent.py:306-311) draws ONE randperm per update epoch
and walks it in consecutive minibatch slices:

    inds_k = torch.randperm(total_steps)         # once per epoch
    for batch in range(num_batch):
        start, end = batch*batch_size, start+batch_size
        inds_b = inds_k[start:end]               # consecutive slice

so each flat (n, ft) element is trained at most once per epoch and
coverage per epoch is floor(total_steps/batch_size)*batch_size, with
total_steps % batch_size elements missed at the tail.

The R1.2 rework implementation instead drew a fresh randperm per
MINIBATCH — replacement sampling across minibatches. Measured impact for
N=4515, K_f=5, B=128, num_batch=35: only ~2865 of 4515 elements trained
per epoch (~1650 skipped, many repeated) — rare early-avoid / dense-
obstacle-success / full-space samples are the ones most likely missed.
The existing tests only checked index ranges, not coverage, uniqueness,
or the randperm call count, so the defect went unnoticed.

R7-3 (guide Step 3): the B=1 likelihood microbatch loop is DELETED — the
vendor loss consumes one B=32 minibatch per facade.loss call, so the old
per-element get_new_logprobs spy cannot see the flat indices. The spy is
now a dual wrapper on the R7-3 production entries:
  - encode_cond  -> records the per-transition marker (agent_pos[:,0,0])
    of every minibatch element the trainer feeds the loss
  - facade.get_logprobs_subsample -> records the denoising index of the
    same minibatch (the vendor loss calls it once per minibatch; the
    observer reads the cache, it never calls it)
  zipped per minibatch, flat = marker*ft + denoising_k.

This file pins the official semantics:
  - randperm_call_count == update_epochs
  - no flat index repeats within an epoch
  - epoch coverage == floor(total_steps/batch_size)*batch_size
  - missed count == total_steps % batch_size
  - the visited set equals the official slice of the SAME permutation
    (verified against a deterministic monkeypatched permutation)

Run: pytest -q tests/dppo_parity/test_randperm_epoch_coverage.py
"""

import numpy as np
import pytest
import torch

from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.rollout_buffer import (
    RolloutBuffer,
    RolloutTransition,
)
from diffusion_policy_3d.rl.dppo_uav.trainer import (
    TrainerConfig,
    ppo_update,
)

from facade_testutil import make_test_context

# layout: n=16 transitions, ft=5 -> total_steps=80; batch_size=32 ->
# num_batch=2, coverage=64, missed=16.
N_TRANS, FT = 16, 5
BATCH = 32
TOTAL = N_TRANS * FT
NUM_BATCH = TOTAL // BATCH  # 2
COVERAGE = NUM_BATCH * BATCH  # 64
MISSED = TOTAL % BATCH  # 16


def _marker_buffer(contract):
    """Buffer whose obs agent_pos carries a UNIQUE per-transition marker at
    agent_pos[0, 0] — the spy recovers the flat index from the minibatch
    obs (flat = marker*ft + denoising_k), so the visited set is
    observable.

    Buffer invariant 3: next_obs[i] == obs[i+1] within an episode — obs
    arrays are built first, next_obs shares the following transition's obs.
    """
    rng = np.random.RandomState(7)
    obs_list = []
    for i in range(N_TRANS):
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
    buf = RolloutBuffer(contract)
    for i in range(N_TRANS):
        t = RolloutTransition(
            obs=obs_list[i],
            next_obs=obs_list[i + 1] if i + 1 < N_TRANS else final_obs,
            chains_policy=rng.uniform(-1, 1, (
                contract.ft_denoising_steps + 1, contract.action_horizon,
                contract.action_dim,
            )).astype(np.float32),
            old_logprobs=rng.uniform(-3, 0, (
                contract.ft_denoising_steps, contract.action_horizon,
                contract.action_dim,
            )).astype(np.float32),
            reward=0.5 if i % 4 == 0 else 0.0,
            terminated=False,
            truncated=False,
            value_old=0.0,
            episode_id="ep0",
            episode_step=i,
            world_uid="w1",
            role="anticipatory",
            manifest_entry_hash="mh-mock",
            episode_config_hash="ch-mock",
            observation_valid=True,
        )
        buf.add(t)
    # R7-4: GAE now requires rollout-filled next values — the shared
    # fixture fills them via the production chunked critic API (single
    # episode; the final transition bootstraps 0.0 for fixture
    # simplicity). Random obs, no numeric GAE assertions.
    from diffusion_policy_3d.rl.dppo_uav.rollout_buffer import (
        fill_buffer_next_values,
    )

    fill_buffer_next_values(
        buf,
        lambda obs_b: obs_b["agent_pos"][:, 0, 0].float(),
    )
    return buf


def _run_ppo_update(monkeypatch, visited, randperm_calls, update_epochs=1):
    """Run ppo_update with a deterministic, call-counting torch.randperm.

    The fixed permutation makes the expected visited set computable; the
    dual spy (encode_cond markers + facade.get_logprobs_subsample
    denoising inds) records the flat index of every element the trainer
    actually feeds the vendor loss.
    """
    contract = RLSystemContract()
    buf = _marker_buffer(contract)
    policy, facade, encode_cond, critic = make_test_context(contract)
    fixed_perm = torch.randperm(
        TOTAL, generator=torch.Generator().manual_seed(20260808)
    )

    def fake_randperm(n, device=None, **kwargs):
        randperm_calls.append(n)
        return fixed_perm.to(device) if device is not None else fixed_perm.clone()

    monkeypatch.setattr(torch, "randperm", fake_randperm)

    # R7-3 dual spy: encode_cond sees the RAW minibatch obs (the marker
    # survives at agent_pos[:, 0, 0]); the vendor loss calls the facade's
    # get_logprobs_subsample once per minibatch (the observer reads the
    # cache, it never calls it again), so the two streams zip 1:1 in
    # minibatch order — flat = marker*ft + denoising_k.
    orig_encode = encode_cond
    markers_b: list = []
    inds_b: list = []

    def spy_encode(obs_b):
        markers_b.append(obs_b["agent_pos"][:, 0, 0].clone())
        return orig_encode(obs_b)

    orig_get = facade.get_logprobs_subsample

    def spy_get(obs, chains_prev, chains_next, denoising_inds, get_ent=False):
        inds_b.append(denoising_inds.clone())
        return orig_get(
            obs, chains_prev, chains_next, denoising_inds, get_ent=get_ent
        )

    monkeypatch.setattr(facade, "get_logprobs_subsample", spy_get)
    ppo_update(
        buffer=buf, contract=contract, facade=facade, encode_cond=spy_encode,
        actor_optimizer=torch.optim.Adam(policy.model.parameters(), lr=1e-3),
        critic_optimizer=torch.optim.Adam(critic.parameters(), lr=1e-3),
        config=TrainerConfig(
            update_epochs=update_epochs, batch_size=BATCH,
            max_grad_norm=None, target_kl=None, ent_coef=0.0, vf_coef=1.0,
        ),
    )
    expected_calls = NUM_BATCH * update_epochs
    assert len(markers_b) == len(inds_b) == expected_calls, (
        f"spy streams {len(markers_b)}/{len(inds_b)} calls, expected"
        f" {expected_calls} (one encode_cond + one vendor loss call per"
        " minibatch)"
    )
    for mk, kk in zip(markers_b, inds_b):
        m = mk.to(torch.long)
        for i in range(m.shape[0]):
            visited.append(int(m[i]) * FT + int(kk[i]))


def test_one_randperm_call_per_epoch(monkeypatch):
    """P0 gate: torch.randperm must be called once per update epoch —
    never once per minibatch (audit 2026-08-08)."""
    visited: list = []
    randperm_calls: list = []
    _run_ppo_update(monkeypatch, visited, randperm_calls, update_epochs=3)
    assert len(randperm_calls) == 3, (
        f"randperm called {len(randperm_calls)} times for 3 epochs"
        " (expected 3: once per epoch — per-minibatch randperm is"
        " replacement sampling and breaks epoch coverage)"
    )


def test_epoch_coverage_without_replacement(monkeypatch):
    """P0 gate: within one epoch every sampled element is distinct, the
    visited set equals the official consecutive slices of the SAME
    permutation, coverage == floor(total/batch)*batch, missed ==
    total % batch."""
    visited: list = []
    randperm_calls: list = []
    _run_ppo_update(monkeypatch, visited, randperm_calls, update_epochs=1)
    assert len(randperm_calls) == 1, (
        f"randperm called {len(randperm_calls)} times for 1 epoch"
    )
    assert len(visited) == COVERAGE, (
        f"trainer processed {len(visited)} elements, expected {COVERAGE}"
        f" (= {NUM_BATCH} minibatches x {BATCH})"
    )
    uniq = set(visited)
    assert len(uniq) == COVERAGE, (
        f"{len(uniq)} distinct elements in one epoch (expected"
        f" {COVERAGE}) — duplicates mean within-epoch replacement sampling"
    )
    # the visited set must equal the first COVERAGE elements of the ONE
    # permutation drawn that epoch (official consecutive-slice semantics).
    # The monkeypatched randperm is a fixed random permutation, so the
    # expected visited set is that permutation's first COVERAGE elements.
    visited_expected = set(
        torch.randperm(
            TOTAL, generator=torch.Generator().manual_seed(20260808)
        )[:COVERAGE].tolist()
    )
    assert uniq == visited_expected, (
        "visited set does not match the official slices of the epoch"
        " permutation (missing training signal for some elements)"
    )
    missed = set(range(TOTAL)) - uniq
    assert len(missed) == MISSED, (
        f"missed {len(missed)} elements per epoch, expected {MISSED}"
        f" (= {TOTAL} % {BATCH})"
    )
