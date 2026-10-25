"""R7-3 historical reference (guide Task R7-3 Step 9).

The R1 rework P1 (audit 2026-08-08) gate proved that TRUE microbatch
gradient accumulation (per-element forward -> per-element loss ->
backward, gradients summed across the minibatch) equals the official
full-batch PPO loss. R7-3 DELETES the B=1 likelihood microbatch loop —
the production path is now a SINGLE official ``facade.loss`` call per
minibatch at the actual minibatch batch (B=128 production; B=8 here).

Per the guide, this file remains as a HISTORICAL/REFERENCE gate — it
proves the deleted path's math reference still holds — and is NOT a
production gate. It pins:

  - the production actual-batch gradient equals the
    official_loss_adapter full-batch reference on the same minibatch
    (the vendor loss and the project-side adapter formula agree through
    gradient flow — the mathematical reference the deleted path was
    built on; the log probs come from the SAME facade kernel the vendor
    loss runs, so any difference surfaces in the loss assembly);
  - the structural INVERSION of the old per-element gate: exactly
    num_batch facade.loss calls, each at the actual minibatch batch — a
    B=1 per-element call pattern must never return;
  - the K5 cleanest-step gradient gate (still a LIVE production
    semantic: a minibatch drawn entirely from the cleanest window step
    must carry real executed-region gradient).

Run: pytest -q tests/dppo_parity/test_microbatch_equivalence.py
"""

import math

import numpy as np
import pytest
import torch

from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.gae import compute_gae
from diffusion_policy_3d.rl.dppo_uav.official_loss_adapter import (
    dppo_policy_loss,
    dppo_value_loss,
)
from diffusion_policy_3d.rl.dppo_uav.rollout_buffer import (
    RolloutBuffer,
    RolloutTransition,
)
from diffusion_policy_3d.rl.dppo_uav.trainer import (
    TrainerConfig,
    ppo_update,
)

from facade_testutil import make_test_context
from test_randperm_epoch_coverage import _marker_buffer  # shared fixture

# The shared _marker_buffer (test_randperm_epoch_coverage) builds N_TRANS=16
# transitions -> total_steps = 80; BATCH=8 -> 10 minibatches, full coverage.
N_TRANS, FT = 16, 5
TOTAL = N_TRANS * FT
BATCH = 8
NUM_BATCH = TOTAL // BATCH


class RecordingSGD(torch.optim.SGD):
    """Captures ppo_update's post-accumulation gradients at step time.

    Only the FIRST step's gradients are kept: ppo_update runs several
    minibatches (total 80 / batch 8 = 10), and later steps are taken on
    already-updated parameters — the reference gradient is computed on
    untouched parameters over the first minibatch.
    """

    def __init__(self, params, lr):
        super().__init__(params, lr=lr)
        self.recorded: list = []

    def step(self, *args, **kwargs):
        if not self.recorded:
            self.recorded = [
                p.grad.detach().clone() for p in self.param_groups[0]["params"]
            ]
        return super().step(*args, **kwargs)


def _fixed_perm(monkeypatch, total, seed=20260808):
    perm = torch.randperm(total, generator=torch.Generator().manual_seed(seed))

    def fake_randperm(n, device=None, **kwargs):
        return perm.to(device) if device is not None else perm.clone()

    monkeypatch.setattr(torch, "randperm", fake_randperm)
    return perm


def _reference_full_batch_grads(contract, buf, policy, facade, encode_cond,
                                critic, flat_slice):
    """Gradient of the official_loss_adapter full-batch PPO loss over the
    same minibatch — the historical math reference the deleted B=1
    microbatch path was built on.

    Mirrors ppo_update's composition (norm_adv -> quantile clamp ->
    denoising discount -> clamped surrogate, vendor order
    diffusion_ppo.py:129-144); the log probs come from the facade's own
    kernel (the same VPGDiffusion.get_logprobs_subsample the vendor loss
    runs), so any difference between the production gradient and this
    reference surfaces in the loss assembly, not in a parallel logp.
    """
    ft = contract.ft_denoising_steps
    n = len(buf)
    obs_t = {
        k: torch.from_numpy(np.stack([t[k] for t in buf.obs])).float()
        for k in buf.obs[0]
    }
    chains_t = torch.from_numpy(np.stack(buf.chains)).float()
    logprobs_t = torch.from_numpy(np.stack(buf.logprobs)).float()
    rewards = np.asarray(buf.rewards, dtype=np.float64)
    terminated = np.asarray(buf.terminated, dtype=bool)
    truncated = np.asarray(buf.truncated, dtype=bool)
    values = np.asarray(buf.values, dtype=np.float64)
    # R5.8-B (B-1): GAE inputs are contract-level data — the collection-
    # time filled next values (a single value function). The reference
    # consumes the SAME array ppo_update reads.
    next_values = np.asarray(buf.next_values, dtype=np.float64)
    advantages, returns = compute_gae(
        rewards, values, next_values, terminated, truncated,
        list(buf.episode_ids), contract.gamma_env, contract.gae_lambda,
    )
    adv_t = torch.from_numpy(advantages).float()
    ret_t = torch.from_numpy(returns).float()
    val_t = torch.from_numpy(values).float()

    batch_inds, denoise_inds = torch.unravel_index(flat_slice, (n, ft))
    obs_b = {k: obs_t[k][batch_inds] for k in obs_t}
    cond_b = encode_cond(obs_b)
    chains_prev = chains_t[batch_inds, denoise_inds]
    chains_next = chains_t[batch_inds, denoise_inds + 1]
    new_logprobs, _eta = facade.get_logprobs_subsample(
        cond_b, chains_prev, chains_next, denoise_inds, get_ent=True
    )
    out = dppo_policy_loss(
        new_logprobs, logprobs_t[batch_inds, denoise_inds],
        adv_t[batch_inds], denoise_inds, exec_steps=contract.exec_steps,
        ft_steps=ft, gamma_denoising=contract.gamma_denoising,
    )  # norm_adv=True (default): internal full-batch normalization
    newvalues = critic(cond_b).reshape(-1)
    v_out = dppo_value_loss(newvalues, val_t[batch_inds], ret_t[batch_inds], None)
    loss = out.pg_loss + v_out.v_loss  # vf_coef=1.0, ent_coef=0.0
    grads = torch.autograd.grad(
        loss, list(policy.model.parameters()) + list(critic.parameters())
    )
    return [g.detach() for g in grads]


def test_actual_batch_grads_match_official_math_reference(monkeypatch):
    """Historical gate: ppo_update's production single-call actual-batch
    gradient (one facade.loss forward/backward per minibatch at B=8)
    equals the official_loss_adapter full-batch reference on the same
    minibatch — the mathematical reference the deleted B=1 path was
    built on. (Reference-only; the production gate is the R7-3 B128
    identity test + G6 GPU one-update.)"""
    contract = RLSystemContract()
    buf = _marker_buffer(contract)
    policy, facade, encode_cond, critic = make_test_context(contract)
    flat_slice = _fixed_perm(monkeypatch, TOTAL)[: BATCH]  # first minibatch

    # reference on an UNTOUCHED twin (ppo_update below steps its own)
    ref_policy, ref_facade, ref_encode, ref_critic = make_test_context(
        contract
    )
    with torch.no_grad():
        for pa, pr in zip(policy.model.parameters(),
                          ref_policy.model.parameters()):
            pr.copy_(pa)
        # the encode path (encode_cond -> policy.obs_encoder) feeds the
        # global_cond every loss reads — a fresh encoder would produce a
        # different cond and different gradients for model + critic.
        for ea, er in zip(policy.obs_encoder.parameters(),
                          ref_policy.obs_encoder.parameters()):
            er.copy_(ea)
        for ca, cr in zip(critic.parameters(), ref_critic.parameters()):
            cr.copy_(ca)
    ref_grads = _reference_full_batch_grads(
        contract, buf, ref_policy, ref_facade, ref_encode, ref_critic,
        flat_slice,
    )

    rec_actor = RecordingSGD(policy.model.parameters(), lr=0.01)
    rec_critic = RecordingSGD(critic.parameters(), lr=0.01)
    ppo_update(
        buffer=buf, contract=contract, facade=facade, encode_cond=encode_cond,
        actor_optimizer=rec_actor, critic_optimizer=rec_critic,
        config=TrainerConfig(
            update_epochs=1, batch_size=BATCH, max_grad_norm=None,
            target_kl=None, ent_coef=0.0, vf_coef=1.0,
        ),
    )
    assert rec_actor.recorded, "actor gradients not recorded"
    got = rec_actor.recorded + rec_critic.recorded
    assert len(got) == len(ref_grads)
    names = ["model.scale", "model.cond_proj.weight", "model.cond_proj.bias",
             "critic.mlp.weight", "critic.mlp.bias"]
    for g, r, name in zip(got, ref_grads, names):
        torch.testing.assert_close(g, r, rtol=1e-5, atol=1e-7,
                                   msg=f"actual-batch grad != adapter"
                                       f" reference for {name}")


def test_single_facade_loss_call_at_actual_batch(monkeypatch):
    """Structural INVERSION of the old per-element gate (guide R7-3
    Step 3/9): the likelihood is evaluated ONCE per minibatch at the
    ACTUAL minibatch batch — exactly NUM_BATCH facade.loss calls, each
    carrying BATCH elements, never one per element. A B=1 per-element
    call pattern would re-introduce the deleted microbatch loop."""
    contract = RLSystemContract()
    buf = _marker_buffer(contract)
    policy, facade, encode_cond, critic = make_test_context(contract)
    calls = []

    orig_loss = facade.loss

    def counting_loss(*args, **kwargs):
        out = orig_loss(*args, **kwargs)
        # the facade caches the vendor loss's get_logprobs_subsample I/O
        # (guide Step 7) — the cache now holds THIS call's minibatch
        calls.append(int(facade.observer_last["denoising_inds"].shape[0]))
        return out

    monkeypatch.setattr(facade, "loss", counting_loss)
    ppo_update(
        buffer=buf, contract=contract, facade=facade, encode_cond=encode_cond,
        actor_optimizer=torch.optim.Adam(policy.model.parameters(), lr=1e-3),
        critic_optimizer=torch.optim.Adam(critic.parameters(), lr=1e-3),
        config=TrainerConfig(
            update_epochs=1, batch_size=BATCH, max_grad_norm=None,
            target_kl=None, ent_coef=0.0, vf_coef=1.0,
        ),
    )
    assert len(calls) == NUM_BATCH, (
        f"expected {NUM_BATCH} facade.loss calls across the update"
        f" (= {TOTAL} cells / {BATCH}), got {len(calls)}"
    )
    assert all(c == BATCH for c in calls), (
        f"every facade.loss call must run at the actual minibatch batch"
        f" {BATCH}, got batch sizes {set(calls)}"
    )


def _cleanest_first_perm(monkeypatch, total, ft, batch):
    """Full-length fixed permutation whose FIRST minibatch is drawn only
    from the cleanest window steps (k = ft-1) — under the official
    kernel (R5.1) the cleanest step is a REAL stochastic transition
    (floored std), so minibatch 1 carries genuine per-element gradient;
    the rest of the pool follows in order."""
    clean = torch.tensor(
        [i * ft + (ft - 1) for i in range(batch)], dtype=torch.long
    )
    rest = torch.tensor(
        [i for i in range(total) if i not in set(clean.tolist())],
        dtype=torch.long,
    )
    inds = torch.cat([clean, rest])

    def fake_randperm(n, device=None, **kwargs):
        return inds.to(device) if device is not None else inds.clone()

    monkeypatch.setattr(torch, "randperm", fake_randperm)
    return inds


def _adapter_marker_buffer(contract):
    """Marker buffer with agent_pos obs keys + a compact point cloud
    (N=16) — the encode path (PolicyObsEncoder -> FakeObsEncoder) reads
    obs_b['agent_pos'] and pc.mean. Marker at agent_pos[0, 0];
    next_obs[i] == obs[i+1] (buffer invariant 3)."""
    rng = np.random.RandomState(7)
    obs_list = []
    for i in range(N_TRANS):
        ap = rng.uniform(-1, 1, (3, 9)).astype(np.float32)
        ap[0, 0] = float(i)
        obs_list.append({
            "point_cloud": rng.uniform(0.3, 9.5, (3, 16, 3)).astype(np.float32),
            "agent_pos": ap,
        })
    final_obs = {
        "point_cloud": rng.uniform(0.3, 9.5, (3, 16, 3)).astype(np.float32),
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
    # R7-4: GAE consumes collection-time next values — fill the shared
    # fixture via the production chunked critic API (single episode;
    # final bootstraps 0.0). Random obs, no numeric GAE assertions.
    from diffusion_policy_3d.rl.dppo_uav.rollout_buffer import (
        fill_buffer_next_values,
    )

    fill_buffer_next_values(
        buf,
        lambda obs_b: obs_b["agent_pos"][:, 0, 0].float(),
    )
    return buf


def test_cleanest_step_minibatch_has_real_gradient(monkeypatch):
    """K5 (review R5.1 — still a LIVE production semantic): a minibatch
    drawn ENTIRELY from the cleanest window step must carry genuine
    per-element actor gradient — the official kernel makes the
    transition into valid timestep 0 stochastic (floored std), so the
    old "grad-less cleanest step is legitimate" bypass is dead code and
    must never return."""

    contract = RLSystemContract()
    buf = _adapter_marker_buffer(contract)
    policy, facade, encode_cond, critic = make_test_context(contract)
    _cleanest_first_perm(monkeypatch, TOTAL, FT, BATCH)  # mb 1 = all k=FT-1
    metrics = ppo_update(
        buffer=buf, contract=contract, facade=facade, encode_cond=encode_cond,
        actor_optimizer=torch.optim.Adam(policy.model.parameters(), lr=1e-3),
        critic_optimizer=torch.optim.Adam(critic.parameters(), lr=1e-3),
        config=TrainerConfig(
            update_epochs=1, batch_size=BATCH, max_grad_norm=None,
            target_kl=None, ent_coef=0.0, vf_coef=1.0,
        ),
    )
    # the update ran to completion; the cleanest-step minibatch must
    # contribute REAL (non-zero) executed-region gradient — a zero norm
    # here is the F1 symptom (dead cleanest transition)
    assert metrics["update_batches"] == TOTAL // BATCH
    assert math.isfinite(metrics["actor_grad_norm"])
    assert math.isfinite(metrics["critic_grad_norm"])
    assert metrics["executed_vs_unexecuted_grad_norm"]["executed"] > 0.0, (
        "cleanest-step elements must carry real gradient under the "
        "official kernel (K5)"
    )
