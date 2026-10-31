"""PPO update loop mirroring the official trainer (design §8.2 + §4.15).

Official reference: third_party/dppo_official_cc7234a/agent/finetune/
train_ppo_diffusion_agent.py:281-390 (update section), kept in order:

  - values via critic at update start: V(s_t) = buffer value_old, GAE
    bootstraps EVERY transition from critic(next_obs_t) — per-transition
    next values, chain stops at episode boundaries via continuation
    masks (terminated masks bootstrap; truncated does not; episode-id
    change blocks the chain)
  - batch sampling: ONE randperm(N*K_f) per update epoch + consecutive
    minibatch slices (official train_ppo_diffusion_agent.py:306-311 — a
    per-MINIBATCH randperm would be replacement sampling across
    minibatches: only ~2865/4515 elements would train per epoch at
    N=4515, K_f=5, B=128)
  - likelihood recompute at the ACTUAL minibatch (guide R7-3 Step 3):
    ONE facade.loss call per B=128 minibatch — the vendor
    PPODiffusion.loss computes the full-batch PPO loss (advantage
    normalization, [-5,2] clamp, denoising discount, interpolated clip
    range) and one backward accumulates the exact official gradient;
    the B=1 microbatch loop is DELETED (test_microbatch_equivalence
    survives only as a historical reference)
  - loss = pg + ent*ent_coef + v*vf_coef; bc off (official use_bc_loss
    default False)
  - backward; actor gradient clipped at max_grad_norm; critic unclipped
    (official order, line 365-373)
  - actor.step() then critic.step()
  - target_kl early stop (official line 378-383)
  - explained variance (official line 386-390; nan when var_y == 0)

§4.15 hard fail: NaN/Inf in GAE or in any loss component raises
FloatingPointError instead of silently corrupting the update (official
does not check; the design does).

Notes:
  - entropy term comes from the vendor (entropy_loss = -eta.mean());
    the composition line is official.
  - bc loss: off by default, same as official use_bc_loss=False.
  - executed-vs-unexecuted gradient norm is measured on the policy loss
    w.r.t. the vendor-loss cached log probs (guide R7-3 Step 7): the
    vendor truncates the likelihood to [:, :exec_steps, :], so the
    unexecuted region is identically 0 by construction — a nonzero
    value flags a contract regression, not an intended signal.
  - observer diagnostics (per-denoise KL/clipfrac, per-element ratios)
    derive from the facade's cached get_logprobs_subsample I/O under
    no_grad; they never participate in loss/backward and never form a
    second policy loss.
"""

import hashlib
import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.gae import compute_gae
from diffusion_policy_3d.rl.dppo_uav.metrics import (
    compute_outcome_counts,
    cross_check_terminal_rewards,
    executed_unexecuted_ratio_statistic,
    explained_variance,
    per_denoise_mean_dict,
    ratio_statistics,
    success_collision_role_conditioned_advantages,
    verify_outcome_invariants,
)
from diffusion_policy_3d.rl.dppo_uav.optimizer_contract import (
    require_single_param_group,
)
# D6 (R7-3): official_loss_adapter is deleted from the training path —
# ppo_update runs the FROZEN vendor PPODiffusion.loss through the
# OfficialPhase1aFacade; no project-side loss formula remains (the
# adapter module itself is retired; see tests/dppo_parity/
# test_production_vendor_direct_reuse.py for the direct-reuse gate).
from diffusion_policy_3d.rl.dppo_uav.rollout_buffer import RolloutBuffer
from diffusion_policy_3d.rl.dppo_uav.vendor_loader import (  # noqa: F401
    load_official_dppo_symbols as _load_official_dppo_symbols,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainerConfig:
    """PPO update hyperparameters (design §8.2)."""

    update_epochs: int = 1
    # 128 is the production optimizer minibatch. R7-3 (guide Task R7-3
    # Step 3): the B=1 likelihood microbatch loop is DELETED — each
    # minibatch is a SINGLE official facade.loss call at the actual
    # minibatch batch (B=128 production), one forward/backward, then the
    # official-order actor/critic steps. The identity gate is pinned at
    # B=128 recompute vs stored B=1 old logprobs; the fallback default
    # must not silently introduce 512.
    batch_size: int = 128
    max_grad_norm: Optional[float] = None  # actor only, official order
    target_kl: Optional[float] = None
    ent_coef: float = 0.0
    vf_coef: float = 1.0
    clip_vloss_coef: Optional[float] = None
    # R5.8-G (C-2/C-4): explicit critic-side controls — a warmup window
    # where only the critic steps (the official n_critic_warmup_itr
    # semantics, exposed instead of hardcoded-absent), and an optional
    # critic-grad alarm (this project's history: twin divergence,
    # DQPG toxicity — a diverging critic silently poisons GAE).
    n_critic_warmup_updates: int = 0
    critic_grad_norm_limit: Optional[float] = None
    reach_reward: float = 100.0  # active reward version's reach bonus
    device: str = "cpu"
    # R7-4: the chunk size for the per-transition next-value critic
    # forward (fill_buffer_next_values) — bounds peak memory on real
    # rollouts (~900 transitions, three-frame point-cloud encoding).
    # None = single forward (small fixtures unchanged); 64 = production
    # default; the resolved chunk is recorded in the fill's provenance.
    next_value_batch_size: Optional[int] = 64

    def __post_init__(self) -> None:
        """The resolved chunk size is persisted in the checkpoint — reject
        configurations that could never be a valid production run:
        batch_size <= 1 (a B=1 minibatch makes the ratio_std diagnostic
        degenerate) and next_value_batch_size <= 0 (None = single forward
        is the only non-positive form allowed)."""
        if self.batch_size <= 1:
            raise ValueError(
                "batch_size must be > 1 (a B=1 optimizer minibatch is"
                f" never a valid production configuration); got"
                f" {self.batch_size}"
            )
        if (
            self.next_value_batch_size is not None
            and self.next_value_batch_size <= 0
        ):
            raise ValueError(
                "next_value_batch_size must be None or > 0 (the resolved"
                f" chunk is persisted in the checkpoint); got"
                f" {self.next_value_batch_size}"
            )
        if self.n_critic_warmup_updates < 0:
            raise ValueError(
                "n_critic_warmup_updates must be >= 0 "
                f"(got {self.n_critic_warmup_updates})"
            )


def _dist_stats(arr: np.ndarray) -> Dict[str, float]:
    """mean/std/min/max/q05/q50/q95 of a (N,) array — the §4.15
    distribution record for reward/return/advantage."""
    a = np.asarray(arr, dtype=np.float64)
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "q05": float(np.quantile(a, 0.05)),
        "q50": float(np.quantile(a, 0.50)),
        "q95": float(np.quantile(a, 0.95)),
    }


def _param_grad_norm(params: Any) -> float:
    """L2 norm of the current .grad over an iterable of parameters."""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum().item())
    return math.sqrt(total)


def _ndarray_hash(arr: np.ndarray) -> str:
    """First 16 hex chars of sha256 over the contiguous float64 bytes."""
    return hashlib.sha256(
        np.ascontiguousarray(arr).tobytes()
    ).hexdigest()[:16]


def _tensor_sha256(t: torch.Tensor) -> str:
    """Cross-process stable EXACT hash of a tensor (R7-6 layer 6/7).

    Keys on shape/dtype and the raw bytes — under the formal
    deterministic numerical mode equal tensors hash equal across
    processes (guide R7-6 Step 7 exact gate; no tolerances anywhere).
    """
    h = hashlib.sha256()
    h.update(repr(tuple(t.shape)).encode("ascii"))
    h.update(str(t.dtype).encode("ascii"))
    h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _grads_sha256(params: Sequence[torch.nn.Parameter]) -> str:
    """Exact hash over every non-None grad tensor (R7-6 layer 7)."""
    h = hashlib.sha256()
    for p in params:
        g = p.grad
        if g is None:
            h.update(b"<no-grad>")
            continue
        h.update(_tensor_sha256(g).encode("ascii"))
    return h.hexdigest()


def _hash_join(a: str, b: str) -> str:
    """Append one hex hash onto another (order-sensitive)."""
    return hashlib.sha256((a + b).encode("ascii")).hexdigest()


def _stack_obs(obs_list: List[Dict[str, np.ndarray]],
               device: str) -> Dict[str, torch.Tensor]:
    """Stack per-transition obs dicts into per-key tensors (N, ...).

    Buffer transitions store obs WITHOUT a batch axis ((T_o, ...) per key);
    np.stack restores the (N, ...) batch.
    """
    return {
        key: torch.from_numpy(np.stack([t[key] for t in obs_list]))
        .float()
        .to(device)
        for key in obs_list[0]
    }


def ppo_update(
    buffer: RolloutBuffer,
    contract: RLSystemContract,
    facade: "OfficialPhase1aFacade",
    encode_cond: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]],
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    config: TrainerConfig,
    action_normalizer: Any = None,
    update_index: int = 0,
    reward_scaler: Any = None,
) -> Dict[str, Any]:
    """One PPO update over the rollout buffer; returns §4.15 metrics.

    The update runs the FROZEN official runtime (guide R7-3 Step 3):
    per minibatch the vendor ``PPODiffusion.loss`` is called ONCE on the
    actual minibatch (production B=128) — ``facade.loss`` returns
    (pg, ent, v, clipfrac, approx_kl, ratio, bc, eta) and the combined
    loss is ``pg + ent_coef*ent + vf_coef*v`` (no B=1 microbatch loop,
    no project-side loss formula). ``encode_cond(obs_b)`` maps a
    minibatch of raw obs dicts to the official cond dict the facade
    consumes (flat global_cond (+ uncond branch under CFG)); the
    value head is the facade's ``critic`` (OfficialValueHead over the
    rollout critic's shared mlp).

    Observer diagnostics (guide R7-3 Step 7) come from the facade's
    cached ``get_logprobs_subsample`` I/O under ``torch.no_grad()`` —
    per-denoise KL/clipfrac and the per-element ratio distribution;
    the aggregate approx_kl/clipfrac are the vendor's returned values.
    The observer never participates in loss/backward and never
    provides a second policy loss.

    ``reward_scaler`` (R6-F §11.7.3): the RunningRewardScaler instance
    that transforms ONLY the training reward consumed by GAE/value
    targets — buffer rewards, artifact/episode returns and the raw
    reward metric stay raw (env is the reward source; a scaler is
    training-side only, never eval-side). ``metrics["actor_stepped"]``
    reports whether this update executed an actor optimizer step (the
    critic-only warmup window is the only skip path) — the train-loop
    scheduler pair (training_schedules.step_scheduler_pair) advances the
    actor schedule from it.

    Raises:
        ValueError: empty buffer.
        FloatingPointError: NaN/Inf GAE output or loss component (§4.15).
    """
    if len(buffer) == 0:
        raise ValueError("ppo_update on an empty rollout buffer")
    # R6-E §11.6.2 (F-08 → R6-OPT-1): every consumer below takes its
    # parameter list from require_single_param_group — a second group
    # hard-fails HERE, before any minibatch, instead of being silently
    # ignored by param_groups[0] indexing.
    actor_params = require_single_param_group("actor", actor_optimizer)
    critic_params = require_single_param_group("critic", critic_optimizer)
    device = config.device
    ft = contract.ft_denoising_steps
    n = len(buffer)

    # --- stack the buffer into tensors -----------------------------------
    obs_t = _stack_obs(buffer.obs, device)
    chains_t = (
        torch.from_numpy(np.stack(buffer.chains)).float().to(device)
    )  # (N, K_f+1, H_a, D_a)
    logprobs_t = (
        torch.from_numpy(np.stack(buffer.logprobs)).float().to(device)
    )  # (N, K_f, H_a, D_a)
    rewards_np = np.asarray(buffer.rewards, dtype=np.float64)
    terminated_np = np.asarray(buffer.terminated, dtype=bool)
    truncated_np = np.asarray(buffer.truncated, dtype=bool)
    values_np = np.asarray(buffer.values, dtype=np.float64)
    roles = list(buffer.roles)

    # --- GAE at update start (§3.5 / R5.8-B B-1) ---------------------------
    # The next values come from the ROLLOUT critic (filled at collection
    # end by fill_buffer_next_values) — a SINGLE value function, matching
    # the official vendor's values_trajs[t+1] semantics. The update-start
    # critic must never be mixed into the rollout-time values.
    next_values_np = np.asarray(buffer.next_values, dtype=np.float64)
    if (next_values_np.shape[0] != n
            or not np.isfinite(next_values_np).all()):
        raise FloatingPointError(
            "buffer next_values not filled/finite before GAE "
            "(R5.8-B B-1 — fill_buffer_next_values must run at "
            "collection end)"
        )
    # R6-F §11.7.3: the scaler touches ONLY the training reward GAE/
    # value targets consume. rewards_np is a fresh array (buffer rewards
    # stay raw); the scaler advances its per-slot state transactionally —
    # a raise here fails the update before publish, so no checkpoint ever
    # contains a half-advanced scaler.
    training_rewards_np = rewards_np
    if reward_scaler is not None:
        training_rewards_np = reward_scaler.scale_flat(
            rewards_np,
            np.asarray(list(buffer.episode_ids)),
            np.asarray(list(buffer.env_ids)),
        )
    advantages_np, returns_np = compute_gae(
        training_rewards_np,
        values_np,
        next_values_np,
        terminated_np,
        truncated_np,
        list(buffer.episode_ids),
        contract.gamma_env,
        contract.gae_lambda,
    )
    advantages_t = torch.from_numpy(advantages_np).float().to(device)
    returns_t = torch.from_numpy(returns_np).float().to(device)
    values_t = torch.from_numpy(values_np).float().to(device)
    if not (torch.isfinite(advantages_t).all()
            and torch.isfinite(returns_t).all()):
        raise FloatingPointError(
            "GAE produced NaN/Inf advantages/returns (§4.15 hard fail)"
        )

    # --- official batch sampling (randperm + unravel_index) --------------
    total_steps = n * ft
    num_batch = max(1, total_steps // config.batch_size)  # skip last ones

    # accumulators for §4.15 signals
    actor_grad_norms: List[float] = []
    critic_grad_norms: List[float] = []
    ratios: List[torch.Tensor] = []
    pg_loss_means: List[float] = []
    v_loss_means: List[float] = []
    approx_kls: List[float] = []
    clipfracs: List[float] = []
    executed_grad_norms: List[float] = []
    unexecuted_grad_norms: List[float] = []
    per_denoise_kl: Dict[int, List[float]] = {k: [] for k in range(ft)}
    per_denoise_clip: Dict[int, List[float]] = {k: [] for k in range(ft)}
    update_batches = 0
    early_stopped = False
    # R7-6 Step 6: layer-6/7 audit hashes for the same-seed comparator —
    # exact hashes of every epoch's randperm and of the full actor+critic
    # gradient tensors before each step (deterministic under the formal
    # numerical mode); consumed only by scripts/r7_compare_resume.py,
    # never by any production branch.
    randperm_hash: Optional[str] = None
    grads_hash: Optional[str] = None

    before_params = [p.detach().clone() for p in actor_params]
    # R6-F §11.7.2: whether THIS update executes an actor optimizer
    # step. The critic-only warmup window is the only skip path; a
    # successful update always runs >= 1 minibatch (K5 gradient check),
    # so the window predicate IS the stepped predicate — the train-loop
    # scheduler pair (training_schedules.step_scheduler_pair) advances
    # the actor schedule from this flag.
    actor_stepped = not (
        config.n_critic_warmup_updates > 0
        and update_index <= config.n_critic_warmup_updates
    )

    for _epoch in range(config.update_epochs):
        # OFFICIAL sampling semantics — ONE randperm per epoch, walked in
        # consecutive minibatch slices (official
        # train_ppo_diffusion_agent.py line 306-311:
        # inds_k = torch.randperm(total_steps); inds_b = inds_k[start:end]).
        # A fresh randperm per minibatch would be replacement sampling:
        # within one epoch a large fraction of elements would never train
        # while others update repeatedly (measured for N=4515, K_f=5,
        # B=128, num_batch=35: only ~2865/4515 elements covered). Per-epoch
        # coverage = floor(total_steps/batch_size)*batch_size; the
        # total_steps % batch_size tail is skipped (official behavior).
        inds_k = torch.randperm(total_steps, device=device)
        # R7-6 layer 6: every epoch's randperm, exact hash (the
        # minibatch-index gate of the same-seed comparator).
        inds_hash = _tensor_sha256(inds_k)
        randperm_hash = (
            inds_hash if randperm_hash is None
            else _hash_join(randperm_hash, inds_hash)
        )
        for step in range(num_batch):
            start = step * config.batch_size
            inds_b = inds_k[start : start + config.batch_size]
            batch_inds_b, denoising_inds_b = torch.unravel_index(
                inds_b, (n, ft)
            )
            obs_b = {k: obs_t[k][batch_inds_b] for k in obs_t}
            chains_prev_b = chains_t[batch_inds_b, denoising_inds_b]
            chains_next_b = chains_t[batch_inds_b, denoising_inds_b + 1]
            returns_b = returns_t[batch_inds_b]
            values_b = values_t[batch_inds_b]
            advantages_b = advantages_t[batch_inds_b]
            old_logprobs_b = logprobs_t[batch_inds_b, denoising_inds_b]

            # --- loss (official composition, guide R7-3 Step 3) ----------
            # ONE actual-minibatch facade.loss call per minibatch — the
            # vendor PPODiffusion.loss computes the full-batch PPO loss
            # internally (advantage normalization over the minibatch,
            # [-5,2] clamp, denoising discount, interpolated clip range,
            # meaned pg/entropy/value). The B=1 microbatch loop is
            # DELETED; test_microbatch_equivalence remains only as a
            # historical reference. Sequence per minibatch: zero grads ->
            # single backward -> clip actor grads -> actor step
            # (non-warmup) -> critic step (official order). The shape
            # contract below hard-fails on any drift from the
            # production B=128 layout (the official tail-skip keeps
            # every minibatch full-sized).
            b = inds_b.shape[0]
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            cond_b = encode_cond(obs_b)
            assert cond_b["global_cond"].shape[0] == b, (
                "cond batch mismatch — encode_cond must map the minibatch"
            )
            assert chains_prev_b.shape == (
                b, contract.action_horizon, contract.action_dim,
            ), f"chains_prev shape {tuple(chains_prev_b.shape)}"
            assert chains_next_b.shape == chains_prev_b.shape, (
                f"chains_next shape {tuple(chains_next_b.shape)}"
            )
            assert denoising_inds_b.shape == (b,), (
                f"denoising_inds shape {tuple(denoising_inds_b.shape)}"
            )
            assert returns_b.shape == (b,), f"returns {tuple(returns_b.shape)}"
            assert values_b.shape == (b,), f"values {tuple(values_b.shape)}"
            assert advantages_b.shape == (b,), (
                f"advantages {tuple(advantages_b.shape)}"
            )
            assert old_logprobs_b.shape == chains_prev_b.shape, (
                f"old_logprobs {tuple(old_logprobs_b.shape)}"
            )
            pg, ent, v, clipfrac, approx_kl, ratio, _bc, _eta = facade.loss(
                cond_b,
                chains_prev_b,
                chains_next_b,
                denoising_inds_b,
                returns_b,
                values_b,
                advantages_b,
                old_logprobs_b,
                use_bc_loss=False,
                reward_horizon=contract.exec_steps,
            )
            loss = pg + config.ent_coef * ent + config.vf_coef * v
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss {loss.item()} (§4.15 hard fail)"
                )
            # observer cache (guide R7-3 Step 7): the vendor loss called
            # get_logprobs_subsample(..., get_ent=True) on this exact
            # minibatch — measure executed/unexecuted gradient flow and
            # derive the diagnostics from that SAME call, never a second
            # forward and never a second policy loss.
            observer = facade.observer_last
            if observer is None or observer["get_ent"] is not True:
                raise RuntimeError(
                    "facade observer cache not populated by the vendor "
                    "loss — get_logprobs_subsample(get_ent=True) must "
                    "run inside facade.loss (R7-3 Step 7)"
                )
            new_logprobs_b = observer["log_prob"]
            assert new_logprobs_b.shape == chains_prev_b.shape, (
                f"observer log_prob {tuple(new_logprobs_b.shape)}"
            )
            # §4.15 executed vs unexecuted pg-norm (strict autograd.grad
            # on the cached log probs, retain_graph for the backward
            # below; reads the graph, never touches parameter grads, and
            # does not participate in the backward). The vendor loss
            # truncates the likelihood to [:, :exec_steps, :], so the
            # unexecuted region has exactly zero gradient path by
            # construction.
            grads_new = torch.autograd.grad(
                pg, new_logprobs_b, retain_graph=True,
            )[0]
            executed_grad_norms.append(
                float(grads_new[:, : contract.exec_steps, :].norm().item())
            )
            unexecuted_grad_norms.append(
                float(grads_new[:, contract.exec_steps:, :].norm().item())
            )
            loss.backward()  # single backward — graph freed here
            pg_loss_means.append(float(pg.item()))
            v_loss_means.append(float(v.item()))

            # --- optimizer steps (official order) -------------------------
            if config.max_grad_norm is not None:
                actor_norm = torch.nn.utils.clip_grad_norm_(
                    actor_params,
                    config.max_grad_norm,
                )
            else:
                actor_norm = _param_grad_norm(actor_params)
            actor_grad_norms.append(float(actor_norm))
            critic_grad_norms.append(float(_param_grad_norm(critic_params)))
            # R5.8-F (N8): transactional boundary — every trainable
            # gradient must be finite BEFORE the step, and every trainable
            # parameter + optimizer state must be finite AFTER. An Inf
            # flowing in here would otherwise be persisted by the next
            # checkpoint before anything noticed.
            for p in (
                list(actor_params)
                + list(critic_params)
            ):
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    raise FloatingPointError(
                        "non-finite trainable gradient before "
                        "optimizer.step (§4.15 transactional hard fail)"
                    )
            # R7-6 layer 7: full actor+critic gradients before this
            # minibatch's step (captured after the finiteness gate — a
            # raise above never reaches a checkpoint).
            mb_grads = _grads_sha256(
                list(actor_params) + list(critic_params))
            grads_hash = (
                mb_grads if grads_hash is None
                else _hash_join(grads_hash, mb_grads)
            )
            if (config.n_critic_warmup_updates > 0
                    and update_index <= config.n_critic_warmup_updates):
                # R5.8-G (C-4): critic-only warmup — the actor step is
                # SKIPPED while the global update index is inside the
                # warmup window (official n_critic_warmup_itr semantics).
                critic_optimizer.step()
            else:
                actor_optimizer.step()
                critic_optimizer.step()
            for opt in (actor_optimizer, critic_optimizer):
                for g in opt.param_groups:
                    for p in g["params"]:
                        if not torch.isfinite(p).all():
                            raise FloatingPointError(
                                "non-finite trainable parameter after "
                                "optimizer.step (§4.15 transactional "
                                "hard fail — checkpoint not published)"
                            )
                        state = opt.state.get(p, {})
                        for name in ("exp_avg", "exp_avg_sq"):
                            v = state.get(name)
                            if v is not None and not torch.isfinite(v).all():
                                raise FloatingPointError(
                                    f"non-finite optimizer state "
                                    f"{name} after optimizer.step "
                                    "(§4.15 transactional hard fail)"
                                )
            update_batches += 1

            # --- per-batch diagnostics (observer, no_grad) ---------------
            # Per-element logratio replicates the vendor's clamped /
            # truncated / meaned computation on the CACHED log probs
            # under no_grad; the aggregate approx_kl/clipfrac use the
            # vendor's returned values (single source of truth); only
            # the per-denoise and per-element distributions are derived
            # here (guide R7-3 Step 7: observer metrics, no second
            # policy loss, no loss/backward participation).
            with torch.no_grad():
                logratio_b = (
                    new_logprobs_b.clamp(-5, 2)
                    - old_logprobs_b.clamp(-5, 2)
                )[:, : contract.exec_steps, :].mean(dim=(-1, -2))
                ratio_b = logratio_b.exp()
                t = denoising_inds_b.float() / (ft - 1)
                if ft > 1:
                    # vendor clip_ploss_coef interpolation
                    # (diffusion_ppo loss; the facade carries the
                    # constants — never duplicated elsewhere)
                    eps_b = facade.clip_ploss_coef_base + (
                        facade.clip_ploss_coef
                        - facade.clip_ploss_coef_base
                    ) * (torch.exp(facade.clip_ploss_coef_rate * t) - 1) / (
                        math.exp(facade.clip_ploss_coef_rate) - 1
                    )
                else:
                    eps_b = t
                kl_i = (ratio_b - 1) - logratio_b  # k3 estimator
                clip_i = (ratio_b - 1).abs() > eps_b
            ratios.append(ratio_b.detach().cpu())
            for k in range(ft):
                mask = denoising_inds_b == k
                if mask.any():
                    per_denoise_kl[k].append(
                        float(kl_i[mask].mean().item())
                    )
                    per_denoise_clip[k].append(
                        float(clip_i[mask].float().mean().item())
                    )
            approx_kls.append(float(approx_kl))  # vendor aggregate
            clipfracs.append(float(clipfrac))    # vendor aggregate
            logger.info(
                "ppo batch %s/%s epoch %s approx_kl %.4f clipfrac %.3f",
                step, num_batch, _epoch, approx_kls[-1], clipfracs[-1],
            )

            # target_kl early stop (official line 378-383)
            if config.target_kl is not None and approx_kls[-1] > config.target_kl:
                early_stopped = True
                logger.info(
                    "early stop: approx_kl %.4f > target_kl %s",
                    approx_kls[-1], config.target_kl,
                )
                break
        if early_stopped:
            break

    # R5.8-F (C-1): the K5 gate upgraded to a REAL LoRA-gradient check —
    # every actor parameter must have accumulated a gradient over the
    # epoch (a missing one means an update path detach or a dropped
    # LoRA hook — the old logprob-flow check could not see it).
    for p in actor_params:
        if p.grad is None:
            raise FloatingPointError(
                "actor parameter without gradient after the update epoch "
                "(K5: a denoising step or a LoRA hook lost its path — "
                "hard fail instead of silent ineffectiveness)"
            )
    # R5.8-G (C-2): the critic-divergence alarm — opt-in; when set, a
    # critic gradient norm beyond the documented limit hard-fails the
    # update instead of silently poisoning every downstream GAE.
    if (config.critic_grad_norm_limit is not None
            and critic_grad_norms
            and max(critic_grad_norms) > config.critic_grad_norm_limit):
        raise FloatingPointError(
            f"critic grad norm {max(critic_grad_norms):.2f} exceeds the "
            f"limit {config.critic_grad_norm_limit} — a diverging critic "
            "silently poisons GAE (§4.15 hard fail)"
        )

    # --- §4.15 aggregate signals ------------------------------------------
    param_update_norm = 0.0
    for before, p in zip(before_params, actor_params):
        diff = p.detach() - before
        param_update_norm += float(diff.pow(2).sum().item())
    param_update_norm = math.sqrt(param_update_norm)

    # policy-action diversity: executed region of the final (cleanest)
    # denoising sample per transition, variance per action dimension.
    # R5.8-E (N7): when the action normalizer is supplied (the production
    # path), the variance is computed in the PHYSICAL action space — the
    # same space measure_bc_variance measures — so the G7 variance gate
    # compares like with like. Fixtures without a normalizer keep the
    # normalized-space diagnostic.
    chains_all = np.stack(buffer.chains)  # (N, K_f+1, H_a, D_a)
    exec_actions = chains_all[:, -1, : contract.exec_steps, :]
    variance_space = "normalized"
    if action_normalizer is not None:
        # The normalizer's OWN unnormalize (works for both the
        # production SingleFieldLinearNormalizer and the fixture
        # transforms — their internals differ, the callable contract
        # does not).
        un = action_normalizer.unnormalize(torch.from_numpy(exec_actions))
        if torch.is_tensor(un):
            un = un.detach().cpu().numpy()
        exec_actions = np.asarray(un, dtype=np.float64)
        variance_space = "physical"
    action_var = exec_actions.var(axis=(0, 1))  # (D_a,)
    action_variance_by_dim = [float(x) for x in action_var]
    policy_action_diversity = float(np.mean(action_var))

    cond_adv = success_collision_role_conditioned_advantages(
        advantages=advantages_t.detach(),
        terminated=terminated_np,
        success=np.asarray(buffer.successes, dtype=bool),
        collision=np.asarray(buffer.collisions, dtype=bool),
        roles=roles,
    )
    # R6-C (§11.4.2): per-batch recomputable outcome counts + buffer
    # invariants + the terminal-reward cross check (reward side is a
    # CROSS CHECK only — EpisodeOutcome remains the success/collision
    # truth; a reward that carries no terminal bonus/penalty hard-fails
    # with the offending episode ids).
    successes_np = np.asarray(buffer.successes, dtype=bool)
    collisions_np = np.asarray(buffer.collisions, dtype=bool)
    rollout_cut_np = np.asarray(buffer.rollout_cut, dtype=bool)
    outcome_counts = compute_outcome_counts(
        terminated=terminated_np, truncated=truncated_np,
        done_reasons=buffer.done_reasons, successes=successes_np,
        collisions=collisions_np, roles=roles,
        rollout_cut=rollout_cut_np,
    )
    verify_outcome_invariants(
        outcome_counts,
        episode_ids=buffer.episode_ids, terminated=terminated_np,
        truncated=truncated_np, rollout_cut=rollout_cut_np,
    )
    cross_check_terminal_rewards(
        rewards=rewards_np, terminated=terminated_np,
        successes=successes_np, collisions=collisions_np,
        episode_ids=buffer.episode_ids,
        reach_reward=float(config.reach_reward),
    )
    ratio_all = ratio_statistics(torch.cat(ratios))
    ev = explained_variance(values_np, returns_np)
    # Corrected-GAE hashes — G6 resume parity compares them bitwise.
    # numpy float64 arithmetic on exactly-represented inputs is
    # deterministic; the CUDA critic forward feeding it is bit-stable
    # under the formal numerical mode (G5 identity measured 0.0).
    advantages_hash = _ndarray_hash(advantages_np)
    returns_hash = _ndarray_hash(returns_np)

    metrics: Dict[str, Any] = {
        "actor_grad_norm": max(actor_grad_norms) if actor_grad_norms else 0.0,
        "critic_grad_norm": max(critic_grad_norms) if critic_grad_norms else 0.0,
        "parameter_update_norm": param_update_norm,
        # R6-C (§11.4.1): per-denoise means are NOT optional statistics
        # — a denoise index with no samples hard-fails (never a NaN).
        "per_denoise_approx_kl": per_denoise_mean_dict(
            per_denoise_kl, "per_denoise_approx_kl"),
        "per_denoise_clipfrac": per_denoise_mean_dict(
            per_denoise_clip, "per_denoise_clipfrac"),
        "ratio_mean": ratio_all["mean"],
        "ratio_std": ratio_all["std"],
        "ratio_q05": ratio_all["q05"],
        "ratio_q50": ratio_all["q50"],
        "ratio_q95": ratio_all["q95"],
        "action_variance_by_dim": action_variance_by_dim,
        "action_variance_space": variance_space,
        "policy_action_diversity": policy_action_diversity,
        "success_conditioned_advantage": cond_adv[
            "success_conditioned_advantage"
        ],
        "collision_conditioned_advantage": cond_adv[
            "collision_conditioned_advantage"
        ],
        "role_conditioned_advantage": cond_adv["role_conditioned_advantage"],
        "executed_vs_unexecuted_grad_norm": {
            "executed": float(np.mean(executed_grad_norms)),
            "unexecuted": float(np.mean(unexecuted_grad_norms)),
            # R6-C (§11.4.1): the ratio follows the OptionalStatistic
            # two-state contract — zero_executed_gradient_denominator
            # when the F10 mask keeps the unexecuted gradient at 0
            # (R5.8-F H-2), never a ~1e8 pseudo-signal nor a NaN.
            "ratio": executed_unexecuted_ratio_statistic(
                float(np.mean(executed_grad_norms)),
                float(np.mean(unexecuted_grad_norms)),
            ),
        },
        "explained_variance": ev,
        # R6-C (§11.4.2): per-batch recomputable outcome counts (the
        # terminal/role invariants were verified above; the gate reader
        # re-checks the conditioned-statistic linkage).
        "outcome_counts": outcome_counts,
        "policy_loss_mean": (
            float(np.mean(pg_loss_means)) if pg_loss_means else math.nan
        ),
        "value_loss_mean": (
            float(np.mean(v_loss_means)) if v_loss_means else math.nan
        ),
        "reward_dist": _dist_stats(rewards_np),
        "return_dist": _dist_stats(returns_np),
        "advantage_dist": _dist_stats(advantages_np),
        "approx_kl": float(np.mean(approx_kls)),
        "clipfrac": float(np.mean(clipfracs)),
        # R7-4: GAE next values come from the ROLLOUT critic over each
        # transition's OWN next_obs (fill_buffer_next_values chunked by
        # TrainerConfig.next_value_batch_size) — the successor-value
        # shortcut is deleted. The legacy key carries the RESOLVED chunk
        # the fill actually used (the checkpoint ERC round-trip contract,
        # train_loop.py:1583 / test_train_loop:700-707 — "the chunk
        # ppo_update actually used"); the full provenance record is
        # persisted alongside.
        "next_value_batch_size": (
            (buffer.next_value_provenance or {}).get("chunk_size")
        ),
        "next_value_provenance": getattr(
            buffer, "next_value_provenance", None
        ),
        "early_stopped": early_stopped,
        "update_batches": update_batches,
        # R6-F §11.7.2: did this update step the actor (critic-only
        # warmup is the only skip path)? The train-loop scheduler pair
        # consumes it — never multi-step/zero-step per committed update.
        "actor_stepped": actor_stepped,
        "advantages_hash": advantages_hash,
        "returns_hash": returns_hash,
        # R7-6 Step 6: layer-6/7 audit hashes (the same-seed comparator
        # consumes these; no production branch reads them).
        "randperm_hash": randperm_hash,
        "grads_hash": grads_hash,
    }
    # R6-C rework (P2): explained_variance returns a dict
    # (target_variance + two-state OptionalStatistic, metrics.py:186) —
    # log the fields explicitly instead of %.4f on a dict (the old
    # format raised TypeError inside the logging handler, which
    # swallowed the line and polluted stderr with "Logging error").
    logger.info(
        "ppo_update done: batches=%s early_stopped=%s approx_kl=%.4f "
        "explained_variance={value=%s valid=%s sample_count=%s "
        "invalid_reason=%s} target_variance=%s actor_grad_norm=%.4f",
        update_batches, early_stopped, metrics["approx_kl"],
        ev.get("value"), ev.get("valid"), ev.get("sample_count"),
        ev.get("invalid_reason"), ev.get("target_variance"),
        metrics["actor_grad_norm"],
    )
    return metrics
