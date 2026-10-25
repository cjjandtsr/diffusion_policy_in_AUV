"""T4 trainer core (design §8.2 + §4.15): PPO update loop running the
FROZEN official runtime (guide R7-3 Step 3).

Structure under test:
  - one official ``facade.loss`` call per minibatch at the ACTUAL
    minibatch batch (B=128 production; B=64 in these fixtures) — the
    vendor PPODiffusion.loss math (clamp [-5,2], denoising discount,
    interpolated clip range, internal norm_adv). The B=1 likelihood
    microbatch loop is DELETED; the loss is never re-assembled on the
    project side.
  - batch sampling via ONE randperm per epoch + consecutive minibatch
    slices + unravel_index((N, ft_denoising_steps)) — official semantics
  - loss = pg + ent*ent_coef + v*vf_coef (bc_loss off by default)
  - actor grad clipped at max_grad_norm, critic unclipped (official order)
  - target_kl early stop
  - §4.15 signals recorded; NaN hard-fail
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


def _trans(contract, seed, episode_id="ep0", step=0, reward=0.0,
           terminated=False, truncated=False):
    rng = np.random.RandomState(seed)
    return RolloutTransition(
        obs={
            "point_cloud": rng.uniform(0.3, 9.5, (3, 4096, 3)).astype(np.float32),
            "agent_pos": rng.uniform(-1, 1, (3, 9)).astype(np.float32),
        },
        next_obs={
            "point_cloud": rng.uniform(0.3, 9.5, (3, 4096, 3)).astype(np.float32),
            "agent_pos": rng.uniform(-1, 1, (3, 9)).astype(np.float32),
        },
        chains_policy=rng.uniform(-1, 1, (
            contract.ft_denoising_steps + 1, contract.action_horizon,
            contract.action_dim,
        )).astype(np.float32),
        old_logprobs=rng.uniform(-3, 0, (
            contract.ft_denoising_steps, contract.action_horizon,
            contract.action_dim,
        )).astype(np.float32),
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        value_old=0.0,
        episode_id=episode_id,
        episode_step=step,
        world_uid="w1",
        role="anticipatory",
        manifest_entry_hash="mh-mock",
        episode_config_hash="ch-mock",
        observation_valid=True,
    )


def _make_buffer(contract, n=16, seed_base=100):
    buf = RolloutBuffer(contract)
    prev_next = None
    for i in range(n):
        t = _trans(contract, seed_base + i, episode_id="ep0", step=i,
                   reward=0.5 if i % 4 == 0 else 0.0)
        if prev_next is not None:
            t = RolloutTransition(
                obs=prev_next, next_obs=t.next_obs,
                chains_policy=t.chains_policy, old_logprobs=t.old_logprobs,
                reward=t.reward, terminated=t.terminated, truncated=t.truncated,
                value_old=t.value_old, episode_id=t.episode_id,
                episode_step=t.episode_step, world_uid=t.world_uid,
                role=t.role, manifest_entry_hash=t.manifest_entry_hash,
                episode_config_hash=t.episode_config_hash,
                observation_valid=t.observation_valid,
            )
        buf.add(t)
        prev_next = t.next_obs
    # R7-4: GAE consumes collection-time next values — the shared test
    # fixture fills them via the production chunked critic API (single
    # episode; the final transition bootstraps 0.0 for fixture
    # simplicity). The critic maps the obs marker at agent_pos[0,0] to
    # the value, identical to the old lambda per transition.
    from diffusion_policy_3d.rl.dppo_uav.rollout_buffer import (
        fill_buffer_next_values,
    )

    fill_buffer_next_values(
        buf,
        lambda obs_b: obs_b["agent_pos"][:, 0, 0].float(),
    )
    return buf


def test_update_runs_and_moves_actor():
    contract = RLSystemContract()
    buf = _make_buffer(contract)
    policy, facade, encode_cond, critic = make_test_context(contract)
    actor_opt = torch.optim.Adam(policy.model.parameters(), lr=1e-3)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
    cfg = TrainerConfig(update_epochs=2, batch_size=8, max_grad_norm=10.0,
                        target_kl=None, ent_coef=0.0, vf_coef=1.0)
    before = [p.clone() for p in policy.model.parameters()]
    ppo_update(
        buffer=buf, contract=contract, facade=facade, encode_cond=encode_cond,
        actor_optimizer=actor_opt, critic_optimizer=critic_opt,
        config=cfg,
    )
    moved = any(
        not torch.equal(b, p.detach())
        for b, p in zip(before, policy.model.parameters())
    )
    assert moved, "actor parameters must move after an update"


def test_metrics_cover_section_4_15_signals():
    contract = RLSystemContract()
    buf = _make_buffer(contract)
    policy, facade, encode_cond, critic = make_test_context(contract)
    cfg = TrainerConfig(update_epochs=1, batch_size=8, max_grad_norm=None,
                        target_kl=None, ent_coef=0.0, vf_coef=1.0)
    metrics = ppo_update(
        buffer=buf, contract=contract, facade=facade, encode_cond=encode_cond,
        actor_optimizer=torch.optim.Adam(policy.model.parameters(), lr=1e-3),
        critic_optimizer=torch.optim.Adam(critic.parameters(), lr=1e-3),
        config=cfg,
    )
    for key in (
        "actor_grad_norm", "critic_grad_norm", "parameter_update_norm",
        "per_denoise_approx_kl", "per_denoise_clipfrac",
        "ratio_mean", "ratio_std", "ratio_q05", "ratio_q50", "ratio_q95",
        "action_variance_by_dim", "policy_action_diversity",
        "success_conditioned_advantage", "collision_conditioned_advantage",
        "role_conditioned_advantage", "executed_vs_unexecuted_grad_norm",
        "explained_variance", "approx_kl", "clipfrac",
    ):
        assert key in metrics, f"missing §4.15 signal {key}"


def test_batch_sampling_matches_official_structure():
    """Official: one randperm(N*ft) per epoch, walked in consecutive
    minibatch slices; the loss batch's denoising index is uniform over
    [0, ft). The facade caches the vendor loss's
    get_logprobs_subsample I/O (guide R7-3 Step 7) — the trainer
    hard-fails if that cache was not populated by facade.loss, so
    reading observer_last is production-path evidence of the batch the
    vendor loss actually consumed."""
    contract = RLSystemContract()
    buf = _make_buffer(contract, n=16)
    policy, facade, encode_cond, critic = make_test_context(contract)
    cfg = TrainerConfig(update_epochs=1, batch_size=64, max_grad_norm=None,
                        target_kl=None, ent_coef=0.0, vf_coef=1.0)
    ppo_update(
        buffer=buf, contract=contract, facade=facade, encode_cond=encode_cond,
        actor_optimizer=torch.optim.Adam(policy.model.parameters(), lr=1e-3),
        critic_optimizer=torch.optim.Adam(critic.parameters(), lr=1e-3),
        config=cfg,
    )
    inds = facade.observer_last["denoising_inds"]
    assert inds.dtype == torch.long
    assert inds.shape == (64,), (
        "the vendor loss batch must be full-sized (official tail-skip;"
        f" got {tuple(inds.shape)})"
    )
    assert inds.min() >= 0 and inds.max() < contract.ft_denoising_steps


def test_target_kl_early_stop():
    """approx_kl > target_kl stops gradient updates (official line 378-383).

    R6-C (§11.4.1): every fine-tune denoise index must carry samples in
    every update (ENGINE hard-fail otherwise). batch_size=n*ft makes
    the first (and only) batch cover every (sample, index) cell of the
    randperm — every index present by construction, no RNG dependence.
    """
    contract = RLSystemContract()
    buf = _make_buffer(contract, n=16)
    policy, facade, encode_cond, critic = make_test_context(contract)
    cfg = TrainerConfig(update_epochs=4,
                        batch_size=16 * contract.ft_denoising_steps,
                        max_grad_norm=None, target_kl=0.0, ent_coef=0.0,
                        vf_coef=1.0)
    metrics = ppo_update(
        buffer=buf, contract=contract, facade=facade, encode_cond=encode_cond,
        actor_optimizer=torch.optim.Adam(policy.model.parameters(), lr=1e-3),
        critic_optimizer=torch.optim.Adam(critic.parameters(), lr=1e-3),
        config=cfg,
    )
    assert metrics["early_stopped"] is True
    assert metrics["update_batches"] < 4 * 2  # stopped before all epochs


def test_nan_advantage_hard_fail():
    """§4.15 hard failure: NaN/Inf in the update must raise, not silently
    continue. The NaN enters through the critic's value output; the
    combined loss is checked for finiteness BEFORE backward."""
    contract = RLSystemContract()
    buf = _make_buffer(contract, n=16)

    class NanCritic(torch.nn.Module):
        def forward(self, obs_cond):
            b = obs_cond["global_cond"].shape[0]
            return torch.full((b,), float("nan"))

    policy, facade, encode_cond, _ = make_test_context(
        contract, critic=NanCritic()
    )
    with pytest.raises((ValueError, RuntimeError, FloatingPointError)):
        ppo_update(
            buffer=buf, contract=contract, facade=facade,
            encode_cond=encode_cond,
            actor_optimizer=torch.optim.Adam(policy.model.parameters(), lr=1e-3),
            critic_optimizer=torch.optim.Adam(policy.model.parameters(), lr=1e-3),
            config=TrainerConfig(update_epochs=1, batch_size=8,
                                 max_grad_norm=None, target_kl=None,
                                 ent_coef=0.0, vf_coef=1.0),
        )


def _agent_pos_with_value(v: float) -> np.ndarray:
    """(3, 9) agent_pos with v embedded at [0, 0] — the fixture value
    the fill function reads back (V = agent_pos[0, 0])."""
    s = np.zeros((3, 9), dtype=np.float32)
    s[0, 0] = v
    return s


def _episode_boundary_buffer(contract):
    """Two episodes; epB's critic values are ~1000x epA's (leak probe).

    epA: i=0..2, ends truncated (reset leads into epB's first obs).
    epB: i=3..5, plain cut at i=5. next_obs[i] == obs[i+1] within an
    episode (buffer invariant), so next_values[i] = V(s_{i+1}).
    """
    rng = np.random.RandomState(11)
    values = [1.0, 2.0, 3.0, 1000.0, 1001.0, 1002.0]
    next_final = 7.0  # V after the plain-cut last transition
    rewards = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0]
    ids = ["A", "A", "A", "B", "B", "B"]
    trunc = [False, False, True, False, False, False]
    buf = RolloutBuffer(contract)
    ft, ha, da = (contract.ft_denoising_steps, contract.action_horizon,
                  contract.action_dim)
    # next_obs[i] MUST equal obs[i+1] (buffer invariant 3) — share arrays
    obs_arr = [
        {
            "point_cloud": rng.uniform(0.3, 9.5, (3, 4096, 3))
            .astype(np.float32),
            "agent_pos": _agent_pos_with_value(values[i]),
        }
        for i in range(6)
    ]
    next_obs_arr = obs_arr[1:] + [
        {
            "point_cloud": rng.uniform(0.3, 9.5, (3, 4096, 3))
            .astype(np.float32),
            "agent_pos": _agent_pos_with_value(next_final),
        }
    ]
    for i in range(6):
        t = RolloutTransition(
            obs=obs_arr[i],
            next_obs=next_obs_arr[i],
            chains_policy=rng.uniform(-1, 1, (ft + 1, ha, da))
            .astype(np.float32),
            old_logprobs=rng.uniform(-3, 0, (ft, ha, da)).astype(np.float32),
            reward=rewards[i],
            terminated=False,
            truncated=trunc[i],
            value_old=values[i],
            episode_id=ids[i],
            episode_step=i,
            world_uid="w1",
            role="anticipatory",
            manifest_entry_hash="mh-mock",
            episode_config_hash="ch-mock",
            observation_valid=True,
        )
        buf.add(t)
    return buf


def test_episode_boundary_stops_advantage_leak(monkeypatch):
    """R1.4 RED/GREEN: GAE must bootstrap EVERY transition from its own
    next observation and stop the chain at episode boundaries (R1.3).

    epA's final transition (i=2, truncated) must use A_2 = delta_2 ONLY
    (no gamma*lambda*A_3 leak from epB, whose values are ~1000x larger).
    The fixture embeds V(s) = agent_pos[0, 0], so next_values[i] are
    known exactly; expected advantages are written out by hand with the
    contract's gamma/gae_lambda (NOT produced by compute_gae). R1
    rework P1: the spy sits on compute_gae — the R7-3 path normalizes
    advantages over the minibatch INSIDE the vendor loss, so spying the
    loss would only see normalized values; GAE output (raw) is what the
    leak probe verifies.

    RED: the old scalar-final-next-value trainer TypeError on the new
    compute_gae signature; a numeric leak would additionally fail
    A_2 = 987.3.
    """
    contract = RLSystemContract()
    buf = _episode_boundary_buffer(contract)
    # R7-4: next values now come from the collection-time fill (single
    # value function). Fill with the fixture's own semantics (V =
    # agent_pos-embedded value) so the hand-computed expectations
    # (a0..a5) stay byte-identical to the pre-fill design.
    from diffusion_policy_3d.rl.dppo_uav.rollout_buffer import (
        fill_buffer_next_values,
    )

    fill_buffer_next_values(
        buf, lambda obs_b: obs_b["agent_pos"][:, 0, 0].float())
    policy, facade, encode_cond, _ = make_test_context(contract)
    captured: list = []

    from diffusion_policy_3d.rl.dppo_uav import trainer as trainer_mod

    orig_gae = trainer_mod.compute_gae

    def spy_gae(rewards, values, next_values, terminated, truncated,
                episode_ids, gamma, lam):
        advantages, returns = orig_gae(
            rewards, values, next_values, terminated, truncated,
            episode_ids, gamma, lam,
        )
        captured.append(advantages)
        return advantages, returns

    monkeypatch.setattr(trainer_mod, "compute_gae", spy_gae)
    ppo_update(
        buffer=buf, contract=contract, facade=facade, encode_cond=encode_cond,
        actor_optimizer=torch.optim.Adam(policy.model.parameters(), lr=1e-3),
        critic_optimizer=torch.optim.Adam(policy.model.parameters(), lr=1e-3),
        config=TrainerConfig(update_epochs=1, batch_size=8,
                             max_grad_norm=None, target_kl=None,
                             ent_coef=0.0, vf_coef=1.0),
    )
    assert captured, "no GAE ran"
    got = captured[0].tolist()  # (6,) raw GAE advantages over all transitions

    # hand-computed expected advantages (gamma=0.99, lambda=0.95)
    g, l = contract.gamma_env, contract.gae_lambda
    a5 = g * 7.0 - 1002.0                 # plain cut: bootstrap, cont=0
    a4 = g * 1002.0 - 1001.0 + g * l * a5
    a3 = g * 1001.0 - 1000.0 + g * l * a4
    a2 = 0.3 + g * 1000.0 - 3.0           # truncated: OWN bootstrap, no leak
    a1 = 0.2 + g * 3.0 - 2.0 + g * l * a2
    a0 = 0.1 + g * 2.0 - 1.0 + g * l * a1
    # a5 (plain-cut last transition, continuation=0) is also in the pool
    # and can be drawn by the randperm — the hand-computed set is all six.
    hand = {"a0": a0, "a1": a1, "a2": a2, "a3": a3, "a4": a4, "a5": a5}
    # the probe must discriminate: with a leak, A_2 includes epB's A_3
    assert abs((a2 + g * l * a3) - a2) > 1.0, "leak probe not discriminating"
    counts = {k: 0 for k in hand}
    for v in got:
        matches = [k for k, e in hand.items() if abs(v - e) < 1e-4]
        assert len(matches) == 1, (
            f"advantage {v} matches {matches} of the hand-computed set"
            " (episode leak or wrong transition type?)"
        )
        counts[matches[0]] += 1
    # GAE output covers all 6 transitions exactly once (raw advantages)
    assert all(counts[k] == 1 for k in hand), counts
    assert sum(counts.values()) == len(got) == 6, (counts, len(got))


def test_ppo_update_empty_buffer_hard_fails():
    """ppo_update on an empty rollout buffer must raise ValueError before
    any batch work (fail-closed — no silent no-op update)."""
    contract = RLSystemContract()
    buf = RolloutBuffer(contract)
    policy, facade, encode_cond, critic = make_test_context(contract)
    with pytest.raises(ValueError):
        ppo_update(
            buffer=buf, contract=contract, facade=facade,
            encode_cond=encode_cond,
            actor_optimizer=torch.optim.Adam(policy.model.parameters(), lr=1e-3),
            critic_optimizer=torch.optim.Adam(critic.parameters(), lr=1e-3),
            config=TrainerConfig(update_epochs=1, batch_size=8,
                                 max_grad_norm=None, target_kl=None,
                                 ent_coef=0.0, vf_coef=1.0),
        )


def test_red30_ppo_update_logs_explained_var_contract(caplog):
    """REWORK 2026-08-19 (P2): explained_variance 现返回 dict
    （metrics.py:186，target_variance + OptionalStatistic 两态）。
    旧 explained_var=%.4f 在 logging handler 格式化时 TypeError
    （被 logging handleError 吞掉 → 日志行丢失 + stderr 污染）。
    修复 = 显式记录 value/valid/sample_count/invalid_reason/
    target_variance。本测试用 caplog 捕获生产日志路径。"""
    import logging

    contract = RLSystemContract()
    buf = _make_buffer(contract)
    policy, facade, encode_cond, critic = make_test_context(contract)
    cfg = TrainerConfig(update_epochs=1, batch_size=8, max_grad_norm=10.0,
                        target_kl=None, ent_coef=0.0, vf_coef=1.0)
    with caplog.at_level(logging.INFO):
        ppo_update(
            buffer=buf, contract=contract, facade=facade,
            encode_cond=encode_cond,
            actor_optimizer=torch.optim.Adam(policy.model.parameters(), lr=1e-3),
            critic_optimizer=torch.optim.Adam(critic.parameters(), lr=1e-3),
            config=cfg,
        )
    # getMessage() 触发 % 格式化——旧代码在此抛 TypeError（RED 崩溃）；
    # 新代码必须产出含显式字段的日志行
    texts = [r.getMessage() for r in caplog.records
             if "ppo_update done" in r.getMessage()]
    assert texts, "ppo_update done log line missing"
    t = texts[0]
    assert "explained_variance=" in t, t
    assert "sample_count=" in t and "invalid_reason=" in t, t
    assert "target_variance=" in t, t
