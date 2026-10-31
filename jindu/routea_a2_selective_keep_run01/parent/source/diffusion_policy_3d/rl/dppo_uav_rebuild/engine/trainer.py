"""Transactional trainer that delegates policy/value math to official DPPO."""

from dataclasses import dataclass
import math

import torch

from .official_loss_components import AdvantageStats, compute_official_components
from .update_transaction import UpdateRejected, UpdateTransaction


@dataclass(frozen=True)
class TrainerConfig:
    value_loss_coefficient: float = 0.5
    entropy_loss_coefficient: float = 0.0
    reward_start: int = 2
    reward_horizon: int = 4
    max_approx_kl: float = 0.2
    ratio_mean_min: float = 0.2
    ratio_mean_max: float = 5.0
    max_grad_norm: float = 0.0

    def __post_init__(self):
        if (self.reward_start, self.reward_horizon) not in ((2, 4), (2, 8)):
            raise ValueError("E5 rollout credit must use A0 [2:6] or R1 [2:10]")
        if self.max_grad_norm < 0:
            raise ValueError("max_grad_norm must be zero (disabled) or positive")


@dataclass
class TrainerCounters:
    attempted_updates: int = 0
    committed_updates: int = 0
    rejected_updates: int = 0
    env_steps: int = 0
    completed_episodes: int = 0
    rollout_segments: int = 0
    consecutive_all_rejected_outer_updates: int = 0
    consecutive_zero_actor_delta_outer_updates: int = 0


@dataclass(frozen=True)
class UpdateMetrics:
    policy_loss: float
    value_loss: float
    entropy_loss: float
    total_loss: float
    approx_kl: float
    ratio_mean: float
    clip_fraction: float
    actor_grad_norm: float
    critic_grad_norm: float
    actor_updated: bool = True
    actor_learning_rate: float = 0.0
    critic_learning_rate: float = 0.0
    actor_grad_clipped: bool = False
    critic_grad_clipped: bool = False
    slice_diagnostics: dict = None


def _grad_norm(parameters):
    squared = torch.zeros((), device=parameters[0].device)
    for parameter in parameters:
        if parameter.grad is not None:
            squared = squared + parameter.grad.detach().float().pow(2).sum()
    return float(squared.sqrt().item())


def _all_finite(values):
    checks = [torch.isfinite(value).all() for value in values]
    return bool(torch.stack(checks).all().item())


class DDPMTrainer:
    def __init__(
        self,
        *,
        runtime,
        lora_bundle,
        actor_optimizer,
        critic_optimizer,
        config: TrainerConfig = None,
    ):
        self.runtime = runtime
        self.lora_bundle = lora_bundle
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.config = config or TrainerConfig()
        self.counters = TrainerCounters()
        self.actor_parameters = tuple(
            parameter for _, parameter in lora_bundle.named_adapter_parameters()
        )
        self.critic_parameters = tuple(runtime.critic.parameters())
        if not self.actor_parameters or not self.critic_parameters:
            raise ValueError("actor and critic trainable parameter sets must be non-empty")
        optimizer_contracts = (
            (actor_optimizer, {id(value) for value in self.actor_parameters}, "actor"),
            (critic_optimizer, {id(value) for value in self.critic_parameters}, "critic"),
        )
        for optimizer, allowed, name in optimizer_contracts:
            for group in optimizer.param_groups:
                if any(id(value) not in allowed for value in group["params"]):
                    raise ValueError(
                        "{} optimizer contains a frozen or unowned parameter".format(name)
                    )

    def update(self, batch, *, update_actor=True):
        if batch.chains_prev.shape[0] != 128:
            raise UpdateRejected(
                "official DPPO update requires true batch size 128",
                reason="invalid_batch_size",
            )
        self.counters.attempted_updates += 1
        mutable = self.actor_parameters + self.critic_parameters
        transaction = UpdateTransaction(
            trainable_parameters=mutable,
            critic=self.runtime.critic,
            optimizers=(self.actor_optimizer, self.critic_optimizer),
        )
        try:
            with transaction:
                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                outputs = self.runtime.loss(
                    batch.condition,
                    batch.chains_prev,
                    batch.chains_next,
                    batch.denoising_inds,
                    batch.returns,
                    batch.oldvalues,
                    batch.advantages.clone(),
                    batch.oldlogprobs,
                    use_bc_loss=False,
                    reward_horizon=self.config.reward_horizon,
                    reward_start=self.config.reward_start,
                )
                (
                    policy_loss,
                    entropy_loss,
                    value_loss,
                    clip_fraction,
                    approx_kl,
                    ratio_mean,
                    _,
                    _,
                ) = outputs
                if update_actor:
                    total_loss = (
                        policy_loss
                        + self.config.value_loss_coefficient * value_loss
                        + self.config.entropy_loss_coefficient * entropy_loss
                    )
                else:
                    total_loss = self.config.value_loss_coefficient * value_loss
                if not torch.isfinite(total_loss):
                    raise UpdateRejected(
                        "non-finite official DPPO loss",
                        reason="nonfinite_loss",
                    )
                if approx_kl > self.config.max_approx_kl:
                    raise UpdateRejected(
                        "official approximate KL exceeded per-minibatch gate",
                        reason="approx_kl_gate",
                        recoverable=True,
                    )
                if not self.config.ratio_mean_min <= ratio_mean <= self.config.ratio_mean_max:
                    raise UpdateRejected(
                        "official ratio mean exceeded per-minibatch gate",
                        reason="ratio_mean_gate",
                        recoverable=True,
                    )
                total_loss.backward()
                actor_norm = (
                    _grad_norm(list(self.actor_parameters)) if update_actor else 0.0
                )
                critic_norm = _grad_norm(list(self.critic_parameters))
                if update_actor and not math.isfinite(actor_norm):
                    raise UpdateRejected(
                        "actor gradient is non-finite",
                        reason="nonfinite_actor_gradient",
                    )
                if update_actor and actor_norm <= 0:
                    raise UpdateRejected(
                        "actor gradient is zero",
                        reason="zero_actor_gradient",
                        recoverable=True,
                    )
                if not math.isfinite(critic_norm):
                    raise UpdateRejected(
                        "critic gradient is non-finite",
                        reason="nonfinite_critic_gradient",
                    )
                if critic_norm <= 0:
                    raise UpdateRejected(
                        "critic gradient is zero",
                        reason="zero_critic_gradient",
                        recoverable=True,
                    )
                actor_grad_clipped = False
                critic_grad_clipped = False
                if self.config.max_grad_norm > 0:
                    if update_actor:
                        torch.nn.utils.clip_grad_norm_(
                            self.actor_parameters, self.config.max_grad_norm
                        )
                        actor_grad_clipped = actor_norm > self.config.max_grad_norm
                    torch.nn.utils.clip_grad_norm_(
                        self.critic_parameters, self.config.max_grad_norm
                    )
                    critic_grad_clipped = critic_norm > self.config.max_grad_norm
                if update_actor:
                    self.actor_optimizer.step()
                self.critic_optimizer.step()
                if not _all_finite(mutable):
                    raise UpdateRejected(
                        "optimizer produced non-finite parameters",
                        reason="nonfinite_parameters",
                    )
                transaction.commit()
            self.counters.committed_updates += 1
            return UpdateMetrics(
                policy_loss=float(policy_loss.detach()),
                value_loss=float(value_loss.detach()),
                entropy_loss=float(entropy_loss.detach()),
                total_loss=float(total_loss.detach()),
                approx_kl=float(approx_kl),
                ratio_mean=float(ratio_mean),
                clip_fraction=float(clip_fraction),
                actor_grad_norm=actor_norm,
                critic_grad_norm=critic_norm,
                actor_updated=bool(update_actor),
                actor_learning_rate=float(self.actor_optimizer.param_groups[0]["lr"]),
                critic_learning_rate=float(self.critic_optimizer.param_groups[0]["lr"]),
                actor_grad_clipped=actor_grad_clipped,
                critic_grad_clipped=critic_grad_clipped,
            )
        except Exception:
            self.counters.rejected_updates += 1
            raise

    def update_logical_batch(self, logical_batch, *, microbatch_pairs, update_actor=True):
        """Accumulate globally reduced official loss and commit exactly once."""

        logical_n = int(logical_batch.chains_prev.shape[0])
        if logical_n < 1 or int(microbatch_pairs) < 1:
            raise UpdateRejected("invalid logical batch size", reason="invalid_batch_size")
        self.counters.attempted_updates += 1
        mutable = self.actor_parameters + self.critic_parameters
        transaction = UpdateTransaction(
            trainable_parameters=mutable,
            critic=self.runtime.critic,
            optimizers=(self.actor_optimizer, self.critic_optimizer),
            counters=self.counters,
        )
        lower = float(getattr(self.runtime, "clip_advantage_lower_quantile", 0.0))
        upper = float(getattr(self.runtime, "clip_advantage_upper_quantile", 1.0))
        stats = AdvantageStats.from_tensor(logical_batch.advantages, lower, upper)
        detached = []
        try:
            with transaction:
                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                for start in range(0, logical_n, int(microbatch_pairs)):
                    end = min(start + int(microbatch_pairs), logical_n)
                    micro = _slice_batch(logical_batch, start, end)
                    parts = compute_official_components(
                        self.runtime,
                        micro,
                        stats,
                        reward_start=self.config.reward_start,
                        reward_horizon=self.config.reward_horizon,
                        use_bc_loss=False,
                    )
                    weight = (end - start) / logical_n
                    policy_loss = parts.policy.mean()
                    value_loss = 0.5 * parts.value.mean()
                    entropy_loss = parts.entropy.mean()
                    total = (
                        policy_loss
                        + self.config.value_loss_coefficient * value_loss
                        + self.config.entropy_loss_coefficient * entropy_loss
                        if update_actor
                        else self.config.value_loss_coefficient * value_loss
                    )
                    if not torch.isfinite(total):
                        raise UpdateRejected("non-finite logical loss", reason="nonfinite_loss")
                    (total * weight).backward()
                    detached.append(parts)
                policy_values = torch.cat([part.policy.detach() for part in detached])
                value_values = torch.cat([part.value.detach() for part in detached])
                entropy_values = torch.cat([part.entropy.detach() for part in detached])
                kl_values = torch.cat([part.approx_kl.detach() for part in detached])
                ratio_values = torch.cat([part.ratio.detach() for part in detached])
                clip_values = torch.cat([part.clip_mask.detach() for part in detached]).float()
                policy_loss_value = policy_values.mean()
                value_loss_value = 0.5 * value_values.mean()
                entropy_loss_value = entropy_values.mean()
                total_loss_value = (
                    policy_loss_value
                    + self.config.value_loss_coefficient * value_loss_value
                    + self.config.entropy_loss_coefficient * entropy_loss_value
                    if update_actor
                    else self.config.value_loss_coefficient * value_loss_value
                )
                approx_kl = float(kl_values.mean())
                ratio_mean = float(ratio_values.mean())
                clip_fraction = float(clip_values.mean())
                slice_diagnostics = None
                if self.config.reward_horizon == 8:
                    slice_diagnostics = {}
                    for label in ("prefix", "tail", "full"):
                        entries = [part.slice_diagnostics[label] for part in detached]
                        slice_diagnostics[label] = {
                            "approx_kl": float(torch.cat([e["approx_kl"] for e in entries]).mean()),
                            "clip_fraction": float(torch.cat([e["clip_mask"] for e in entries]).float().mean()),
                            "logratio": float(torch.cat([e["logratio"] for e in entries]).mean()),
                            "logprob_drift": float(torch.cat([e["logprob_drift"] for e in entries]).mean()),
                            "action_variance": float(torch.stack([e["action_variance"] for e in entries]).mean()),
                        }
                if approx_kl > self.config.max_approx_kl:
                    raise UpdateRejected(
                        "official approximate KL exceeded logical gate",
                        reason="approx_kl_gate", recoverable=True,
                    )
                if not self.config.ratio_mean_min <= ratio_mean <= self.config.ratio_mean_max:
                    raise UpdateRejected(
                        "official ratio mean exceeded logical gate",
                        reason="ratio_mean_gate", recoverable=True,
                    )
                actor_norm = _grad_norm(list(self.actor_parameters)) if update_actor else 0.0
                critic_norm = _grad_norm(list(self.critic_parameters))
                if update_actor and (not math.isfinite(actor_norm) or actor_norm <= 0):
                    raise UpdateRejected(
                        "actor gradient is non-finite or zero",
                        reason="nonfinite_actor_gradient" if not math.isfinite(actor_norm) else "zero_actor_gradient",
                        recoverable=math.isfinite(actor_norm),
                    )
                if not math.isfinite(critic_norm) or critic_norm <= 0:
                    raise UpdateRejected(
                        "critic gradient is non-finite or zero",
                        reason="nonfinite_critic_gradient" if not math.isfinite(critic_norm) else "zero_critic_gradient",
                        recoverable=math.isfinite(critic_norm),
                    )
                actor_grad_clipped = critic_grad_clipped = False
                if self.config.max_grad_norm > 0:
                    if update_actor:
                        torch.nn.utils.clip_grad_norm_(self.actor_parameters, self.config.max_grad_norm)
                        actor_grad_clipped = actor_norm > self.config.max_grad_norm
                    torch.nn.utils.clip_grad_norm_(self.critic_parameters, self.config.max_grad_norm)
                    critic_grad_clipped = critic_norm > self.config.max_grad_norm
                if update_actor:
                    self.actor_optimizer.step()
                self.critic_optimizer.step()
                if not _all_finite(mutable):
                    raise UpdateRejected("optimizer produced non-finite parameters", reason="nonfinite_parameters")
                transaction.commit()
            self.counters.committed_updates += 1
            return UpdateMetrics(
                policy_loss=float(policy_loss_value), value_loss=float(value_loss_value),
                entropy_loss=float(entropy_loss_value), total_loss=float(total_loss_value),
                approx_kl=approx_kl, ratio_mean=ratio_mean, clip_fraction=clip_fraction,
                actor_grad_norm=actor_norm, critic_grad_norm=critic_norm,
                actor_updated=bool(update_actor),
                actor_learning_rate=float(self.actor_optimizer.param_groups[0]["lr"]),
                critic_learning_rate=float(self.critic_optimizer.param_groups[0]["lr"]),
                actor_grad_clipped=actor_grad_clipped,
                critic_grad_clipped=critic_grad_clipped,
                slice_diagnostics=slice_diagnostics,
            )
        except Exception:
            self.counters.rejected_updates += 1
            raise


def _slice_batch(batch, start, end):
    """Tensor-only microbatch view; logical statistics stay outside this view."""

    from types import SimpleNamespace
    return SimpleNamespace(
        condition={key: value[start:end] for key, value in batch.condition.items()},
        chains_prev=batch.chains_prev[start:end],
        chains_next=batch.chains_next[start:end],
        denoising_inds=batch.denoising_inds[start:end],
        returns=batch.returns[start:end],
        oldvalues=batch.oldvalues[start:end],
        advantages=batch.advantages[start:end],
        oldlogprobs=batch.oldlogprobs[start:end],
    )
