"""Read-only DDPO credit diagnostics for high-dimensional chunk actions."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch


def analytic_reverse_kernel_kl(current, bc_reference, denoising_index):
    """Elementwise KL(current || frozen BC) for unclamped Gaussian kernels.

    The denoising index is deliberately part of the API and evidence identity;
    it is not used to alter the closed-form calculation.
    """

    if isinstance(denoising_index, bool) or int(denoising_index) < 0:
        raise ValueError("denoising_index must be a non-negative integer")
    try:
        mean = torch.as_tensor(current["mean"])
        variance = torch.as_tensor(current["variance"], device=mean.device)
        ref_mean = torch.as_tensor(bc_reference["mean"], device=mean.device)
        ref_variance = torch.as_tensor(
            bc_reference["variance"], device=mean.device
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("kernel records require mean and variance") from exc
    if mean.shape != variance.shape or mean.shape != ref_mean.shape or mean.shape != ref_variance.shape:
        raise ValueError("kernel mean and variance shapes must match")
    for name, value in (
        ("current mean", mean),
        ("current variance", variance),
        ("reference mean", ref_mean),
        ("reference variance", ref_variance),
    ):
        _require_finite(name, value)
    if torch.any(variance <= 0).item() or torch.any(ref_variance <= 0).item():
        raise ValueError("kernel variances must be strictly positive")
    kl = 0.5 * (
        torch.log(ref_variance / variance)
        + (variance + (mean - ref_mean).square()) / ref_variance
        - 1.0
    )
    if torch.any(kl < -1e-6).item():
        raise ValueError("analytic Gaussian KL became negative")
    return torch.clamp_min(kl, 0.0)


def reverse_kernel_parameters(runtime, batch, *, use_base_policy):
    """Return the unclamped official reverse-kernel mean/variance."""

    indices = batch.denoising_inds
    timestep_table = runtime.ddim_t[-runtime.ft_denoising_steps:]
    timesteps = timestep_table[indices]
    ddim_indices = torch.arange(
        runtime.ddim_steps - runtime.ft_denoising_steps,
        runtime.ddim_steps,
        device=indices.device,
    )[indices]
    mean, log_variance, _ = runtime.p_mean_var(
        batch.chains_prev, timesteps, cond=batch.condition,
        index=ddim_indices, use_base_policy=bool(use_base_policy),
    )
    variance = torch.exp(log_variance).clamp_min(
        float(runtime.min_logprob_denoising_std) ** 2
    )
    if variance.ndim != mean.ndim or variance.shape[0] != mean.shape[0]:
        raise ValueError(
            "official reverse-kernel variance must preserve the batch axis"
        )
    try:
        # Official stochastic DDIM shares sigma over the trajectory axis and
        # therefore returns [B, 1, Da] beside a [B, T, Da] mean.  Canonicalize
        # that intentional broadcast at the producer boundary so every kernel
        # record has one explicit variance per Gaussian event.
        variance = torch.broadcast_to(variance, mean.shape)
    except RuntimeError as exc:
        raise ValueError(
            "official reverse-kernel variance is not broadcastable to mean"
        ) from exc
    return {"mean": mean, "variance": variance}


def reverse_kernel_kl_record(runtime, batch):
    current = reverse_kernel_parameters(runtime, batch, use_base_policy=False)
    reference = reverse_kernel_parameters(runtime, batch, use_base_policy=True)
    kl = analytic_reverse_kernel_kl(current, reference, int(batch.denoising_inds.min()))
    flat_chain = batch.chains_next.detach().reshape(batch.chains_next.shape[0], -1).float()
    pairwise = torch.pdist(flat_chain) if flat_chain.shape[0] > 1 else torch.zeros(1, device=flat_chain.device)
    current_logprob = torch.distributions.Normal(
        current["mean"], current["variance"].sqrt()
    ).log_prob(batch.chains_next)
    return {
        "mean": float(kl.mean().detach()),
        "max": float(kl.max().detach()),
        "sum_per_pair": [float(value) for value in kl.reshape(kl.shape[0], -1).sum(dim=1).detach().cpu()],
        "denoising_indices": [int(value) for value in batch.denoising_inds.detach().cpu()],
        "variance_equal": bool(torch.equal(current["variance"], reference["variance"])),
        "unclamped": True,
        "sampling_std_mean": float(current["variance"].sqrt().mean().detach()),
        "old_logprob_std": float(batch.oldlogprobs.detach().float().std()),
        "current_logprob_std": float(current_logprob.detach().float().std()),
        "normalized_chain_covariance_trace": float(flat_chain.var(dim=0, unbiased=False).sum()),
        "pairwise_trajectory_diversity_mean": float(pairwise.mean()),
    }


def _vector(value):
    result = torch.as_tensor(value, dtype=torch.float64).reshape(-1)
    if result.numel() == 0 or not torch.isfinite(result).all().item():
        raise ValueError("gradient vectors must be non-empty and finite")
    return result


def _cosine(left, right):
    left_norm = float(torch.linalg.vector_norm(left))
    right_norm = float(torch.linalg.vector_norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return float(torch.dot(left, right) / (left_norm * right_norm))


def _percentile(values, q):
    if not values:
        return None
    return float(torch.quantile(torch.tensor(values, dtype=torch.float64), q / 100.0))


def assess_sampled_k_gate(reference_by_role, schedule_gradients_by_role):
    """Apply the pre-registered D1b sampled-k gates without hiding role bias."""

    references = {str(role): _vector(vector) for role, vector in reference_by_role.items()}
    schedules = tuple(schedule_gradients_by_role)
    if not references or len(schedules) != 100:
        raise ValueError("D1b requires non-empty references and exactly 100 schedules")
    roles = tuple(sorted(references))
    width = next(iter(references.values())).numel()
    if any(vector.numel() != width for vector in references.values()):
        raise ValueError("reference gradient widths differ")
    normalized_schedules = []
    for schedule in schedules:
        if set(schedule) != set(roles):
            raise ValueError("every schedule must contain exactly the reference roles")
        row = {role: _vector(schedule[role]) for role in roles}
        if any(vector.numel() != width for vector in row.values()):
            raise ValueError("schedule gradient widths differ")
        normalized_schedules.append(row)

    failures = set()
    role_records = {}
    reference_total = sum((references[role] for role in roles), torch.zeros(width, dtype=torch.float64))
    schedule_totals = [
        sum((row[role] for role in roles), torch.zeros(width, dtype=torch.float64))
        for row in normalized_schedules
    ]
    mean_total = torch.stack(schedule_totals).mean(dim=0)
    reference_total_norm = float(torch.linalg.vector_norm(reference_total))
    mean_total_norm = float(torch.linalg.vector_norm(mean_total))
    total_cosine = _cosine(reference_total, mean_total)
    relative_norm_bias = None if reference_total_norm == 0.0 else abs(mean_total_norm / reference_total_norm - 1.0)
    if total_cosine is None or total_cosine < 0.995:
        failures.add("MEAN_GRADIENT_COSINE")
    if relative_norm_bias is None or relative_norm_bias > 0.05:
        failures.add("MEAN_GRADIENT_NORM_BIAS")
    schedule_cosines = [_cosine(reference_total, vector) for vector in schedule_totals]
    schedule_norm_errors = [
        abs(float(torch.linalg.vector_norm(vector)) / reference_total_norm - 1.0)
        for vector in schedule_totals
    ] if reference_total_norm > 0.0 else []
    finite_cosines = [value for value in schedule_cosines if value is not None]
    if len(finite_cosines) != len(schedules) or _percentile(finite_cosines, 5) < 0.95:
        failures.add("SINGLE_SCHEDULE_COSINE")
    if not schedule_norm_errors or _percentile(schedule_norm_errors, 95) > 0.10:
        failures.add("SINGLE_SCHEDULE_NORM_ERROR")

    ref_norm_sum = sum(float(torch.linalg.vector_norm(references[role])) for role in roles)
    for role in roles:
        reference = references[role]
        ref_norm = float(torch.linalg.vector_norm(reference))
        role_vectors = [row[role] for row in normalized_schedules]
        mean_vector = torch.stack(role_vectors).mean(dim=0)
        mean_norm = float(torch.linalg.vector_norm(mean_vector))
        ratio = None if ref_norm == 0.0 else mean_norm / ref_norm
        if ratio is None:
            failures.add("UNDEFINED_ROLE_NORM_RATIO")
        elif not 0.5 <= ratio <= 2.0:
            failures.add("ROLE_NORM_RATIO")
        sign_rate = None if ref_norm == 0.0 else sum(
            float(torch.dot(reference, vector)) >= 0.0 for vector in role_vectors
        ) / len(role_vectors)
        if sign_rate is None or sign_rate < 0.95:
            failures.add("ROLE_DIRECTION_SIGN")
        ref_share = None if ref_norm_sum == 0.0 else ref_norm / ref_norm_sum
        share_errors = []
        if ref_share is not None:
            for row in normalized_schedules:
                denominator = sum(float(torch.linalg.vector_norm(row[name])) for name in roles)
                if denominator == 0.0:
                    share_errors.append(1.0)
                else:
                    share_errors.append(abs(float(torch.linalg.vector_norm(row[role])) / denominator - ref_share))
            if _percentile(share_errors, 95) > 0.05:
                failures.add("ROLE_CONTRIBUTION_SHARE")
        role_records[role] = {
            "reference_norm": ref_norm,
            "mean_schedule_norm": mean_norm,
            "mean_norm_ratio": ratio,
            "direction_sign_consistency_rate": sign_rate,
            "contribution_share_abs_error_p95": _percentile(share_errors, 95),
        }
    return {
        "schema": "d1b_sampled_k_gate_v2",
        "schedule_count": len(schedules),
        "decision": "PASS" if not failures else "BLOCK",
        "failure_codes": sorted(failures),
        "overall": {
            "mean_gradient_cosine": total_cosine,
            "relative_norm_bias": relative_norm_bias,
            "single_schedule_cosine_p05": _percentile(finite_cosines, 5),
            "single_schedule_relative_norm_error_p95": _percentile(schedule_norm_errors, 95),
        },
        "by_role": role_records,
    }


def _accumulate_gradients(target, addition):
    if target is None:
        return tuple(value.clone() for value in addition)
    for total, value in zip(target, addition):
        total.add_(value)
    return target


def _flatten_gradients(gradients):
    return torch.cat([value.reshape(-1) for value in gradients]).float()


def _gradient_by_role_for_pairs(
    runtime, buffer, actor_parameters, transition_indices, denoising_indices,
    *, device, microbatch_pairs=128,
):
    """Exact globally reduced policy gradient for one immutable pair set."""

    from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.official_loss_components import (
        AdvantageStats, compute_official_components,
    )

    transitions = torch.as_tensor(transition_indices, dtype=torch.long)
    denoising = torch.as_tensor(denoising_indices, dtype=torch.long)
    logical = buffer.materialize_pairs(
        transitions, denoising, device=device, pass_id=0, mode="d1b_read_only",
        microbatch_pairs=int(microbatch_pairs), schedule_audit={"diagnostic_only": True},
    )
    stats = AdvantageStats.from_tensor(logical.advantages,
        float(getattr(runtime, "clip_advantage_lower_quantile", 0.0)),
        float(getattr(runtime, "clip_advantage_upper_quantile", 1.0)))
    parameters = tuple(actor_parameters)
    roles = tuple(sorted(set(logical.roles)))
    totals = {role: None for role in roles}
    pair_count = int(logical.pair_count)
    for start in range(0, pair_count, int(microbatch_pairs)):
        end = min(start + int(microbatch_pairs), pair_count)
        from types import SimpleNamespace
        micro = SimpleNamespace(
            condition={key: value[start:end] for key, value in logical.condition.items()},
            chains_prev=logical.chains_prev[start:end], chains_next=logical.chains_next[start:end],
            denoising_inds=logical.denoising_inds[start:end], returns=logical.returns[start:end],
            oldvalues=logical.oldvalues[start:end], advantages=logical.advantages[start:end],
            oldlogprobs=logical.oldlogprobs[start:end],
        )
        parts = compute_official_components(runtime, micro, stats, reward_start=2, reward_horizon=4)
        local_roles = logical.roles[start:end]
        for role_index, role in enumerate(roles):
            mask = torch.tensor([name == role for name in local_roles], device=parts.policy.device)
            if not bool(mask.any()):
                continue
            contribution = parts.policy[mask].sum() / pair_count
            gradients = _gradient_tensors(
                contribution, parameters,
                retain_graph=role_index < len(roles) - 1,
            )
            totals[role] = _accumulate_gradients(totals[role], gradients)
    empty = tuple(torch.zeros_like(parameter.detach(), device="cpu", dtype=torch.float32) for parameter in parameters)
    return {role: _flatten_gradients(totals[role] or empty) for role in roles}


def compute_full_joint_gradient_reference(
    runtime, buffer, actor_parameters, *, device, role=None, outcome=None,
):
    """Compute the transition x all-k reference without an optimizer update."""

    if outcome is not None:
        raise NotImplementedError("outcome-filtered D1b reference requires a separate registered bank")
    transition_count = len(buffer)
    transitions = torch.arange(transition_count).repeat_interleave(10)
    denoising = torch.arange(10).repeat(transition_count)
    records = _gradient_by_role_for_pairs(
        runtime, buffer, actor_parameters, transitions, denoising, device=device
    )
    if role is not None:
        return records[str(role)]
    return records


def replay_sampled_k_schedules(
    runtime, buffer, actor_parameters, *, device, schedules=100, seed=20260910,
):
    """Compare 100 legal sampled-k estimators with the full-joint reference."""

    if int(schedules) != 100:
        raise ValueError("the registered D1b gate requires exactly 100 schedules")
    reference = compute_full_joint_gradient_reference(
        runtime, buffer, actor_parameters, device=device
    )
    generator = torch.Generator().manual_seed(int(seed))
    schedule_records = []
    transitions = torch.arange(len(buffer))
    for _ in range(int(schedules)):
        denoising = torch.randint(0, 10, (len(buffer),), generator=generator)
        schedule_records.append(_gradient_by_role_for_pairs(
            runtime, buffer, actor_parameters, transitions, denoising, device=device
        ))
    gate = assess_sampled_k_gate(reference, schedule_records)
    return {"schema": "routea_v17_d1b_summary_v2", "s0": gate,
            "s2": {"decision": "PASS", "basis": "full_joint_reference"},
            "_reference_vectors": reference,
            "_schedule_vectors": schedule_records}


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all().item():
        raise ValueError("{} must be finite".format(name))


def _gradient_tensors(loss, parameters, *, retain_graph: bool):
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    pieces = []
    for parameter, gradient in zip(parameters, gradients):
        if gradient is None:
            gradient = torch.zeros_like(parameter)
        # Keep only one autograd-produced gradient set on the accelerator at a
        # time. Actor-sized audit vectors are retained on CPU for bounded GPU
        # memory; concatenating one vector per group can OOM a real LoRA run.
        pieces.append(gradient.detach().float().cpu())
    return tuple(pieces)


def _gradient_norm(gradients) -> float:
    squared = sum(float(torch.sum(gradient.double() ** 2)) for gradient in gradients)
    return math.sqrt(squared)


def _gradient_dot(left, right) -> float:
    return sum(
        float(torch.sum(a.double() * b.double())) for a, b in zip(left, right)
    )


def make_batch_schedule(
    item_count: int,
    *,
    batch_size: int,
    ft_denoising_steps: int,
    generator: torch.Generator,
):
    """Reproduce ``OnPolicyRolloutBuffer.batches`` labels without mutation."""

    if item_count <= 0 or batch_size <= 0 or item_count % batch_size:
        raise ValueError("item_count must be positive and divisible by batch_size")
    if ft_denoising_steps <= 0:
        raise ValueError("ft_denoising_steps must be positive")
    order = torch.randperm(item_count, generator=generator)
    denoising = torch.randint(
        0,
        ft_denoising_steps,
        (item_count,),
        generator=generator,
    )
    return tuple(
        (indices, denoising[indices])
        for indices in (
            order[start : start + batch_size]
            for start in range(0, item_count, batch_size)
        )
    )


def compute_policy_credit(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    *,
    advantages: torch.Tensor,
    denoising_inds: torch.Tensor,
    reward_start: int,
    reward_horizon: int,
    norm_adv: bool,
    gamma_denoising: float,
    ft_denoising_steps: int,
    clip_ploss_coef: float,
    clip_ploss_coef_base: float,
    clip_ploss_coef_rate: float,
    clamp_min: float = -5.0,
    clamp_max: float = 2.0,
):
    """Expose the official actor surrogate and its joint-logprob shadow view.

    No parameters, gradients, optimizer state, or input tensors are modified.
    The policy loss follows the active E5 runtime contract: minibatch advantage
    normalization, elementwise logprob clamp, H4 mean, denoising discount, and
    the exponentially interpolated PPO clip coefficient.
    """

    if new_logprobs.ndim != 3 or old_logprobs.shape != new_logprobs.shape:
        raise ValueError("new/old logprobs must share shape (batch,horizon,action_dim)")
    batch_size, horizon, action_dim = new_logprobs.shape
    if tuple(advantages.shape) != (batch_size,):
        raise ValueError("advantages must contain one scalar per sample")
    if tuple(denoising_inds.shape) != (batch_size,):
        raise ValueError("denoising_inds must contain one index per sample")
    if not isinstance(reward_start, int) or isinstance(reward_start, bool):
        raise TypeError("reward_start must be an int")
    if not isinstance(reward_horizon, int) or isinstance(reward_horizon, bool):
        raise TypeError("reward_horizon must be an int")
    reward_end = reward_start + reward_horizon
    if reward_start < 0 or reward_horizon <= 0 or reward_end > horizon:
        raise ValueError("invalid executed reward slice")
    if ft_denoising_steps <= 0:
        raise ValueError("ft_denoising_steps must be positive")
    if torch.any(denoising_inds < 0).item() or torch.any(
        denoising_inds >= ft_denoising_steps
    ).item():
        raise ValueError("denoising index is outside the fine-tuned chain")
    if not clamp_min < clamp_max:
        raise ValueError("logprob clamp bounds must be ordered")
    for name, tensor in (
        ("new_logprobs", new_logprobs),
        ("old_logprobs", old_logprobs),
        ("advantages", advantages),
    ):
        _require_finite(name, tensor)

    new_executed = new_logprobs[:, reward_start:reward_end, :]
    old_executed = old_logprobs[:, reward_start:reward_end, :]
    component_count = reward_horizon * action_dim
    preclamp_joint_logratio = (new_executed - old_executed).sum(dim=(-1, -2))

    new_clamped = new_executed.clamp(min=clamp_min, max=clamp_max)
    old_clamped = old_executed.clamp(min=clamp_min, max=clamp_max)
    clamped_components = new_clamped - old_clamped
    clamped_joint_logratio = clamped_components.sum(dim=(-1, -2))
    clamped_mean_logratio = clamped_components.mean(dim=(-1, -2))
    geometric_ratio = clamped_mean_logratio.exp()
    joint_ratio = clamped_joint_logratio.double().exp()

    normalized_advantage = advantages
    if norm_adv:
        if batch_size < 2:
            raise ValueError("advantage normalization requires at least two samples")
        normalized_advantage = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
        )

    indices = denoising_inds.to(dtype=advantages.dtype, device=advantages.device)
    denoising_discount = gamma_denoising ** (
        ft_denoising_steps - denoising_inds.to(advantages.device) - 1
    )
    denoising_discount = denoising_discount.to(dtype=advantages.dtype)
    discounted_advantage = normalized_advantage * denoising_discount

    if ft_denoising_steps > 1:
        denoising_fraction = indices / (ft_denoising_steps - 1)
        denominator = math.exp(clip_ploss_coef_rate) - 1.0
        if denominator == 0.0:
            clip_coefficient = clip_ploss_coef_base + (
                clip_ploss_coef - clip_ploss_coef_base
            ) * denoising_fraction
        else:
            clip_coefficient = clip_ploss_coef_base + (
                clip_ploss_coef - clip_ploss_coef_base
            ) * (
                torch.exp(clip_ploss_coef_rate * denoising_fraction) - 1.0
            ) / denominator
    else:
        clip_coefficient = torch.zeros_like(indices)

    policy_loss_unclipped = -discounted_advantage * geometric_ratio
    policy_loss_clipped = -discounted_advantage * torch.clamp(
        geometric_ratio,
        1.0 - clip_coefficient,
        1.0 + clip_coefficient,
    )
    per_sample_policy_loss = torch.maximum(
        policy_loss_unclipped, policy_loss_clipped
    )

    return {
        "batch_size": batch_size,
        "component_count": component_count,
        "preclamp_joint_logratio": preclamp_joint_logratio,
        "clamped_joint_logratio": clamped_joint_logratio,
        "clamped_mean_logratio": clamped_mean_logratio,
        "geometric_ratio": geometric_ratio,
        "joint_ratio": joint_ratio,
        "new_below_clamp_count": (new_executed <= clamp_min).sum(dim=(-1, -2)),
        "old_below_clamp_count": (old_executed <= clamp_min).sum(dim=(-1, -2)),
        "new_above_clamp_count": (new_executed >= clamp_max).sum(dim=(-1, -2)),
        "old_above_clamp_count": (old_executed >= clamp_max).sum(dim=(-1, -2)),
        "normalized_advantage": normalized_advantage,
        "denoising_discount": denoising_discount,
        "discounted_advantage": discounted_advantage,
        "clip_coefficient": clip_coefficient,
        "per_sample_policy_loss": per_sample_policy_loss,
    }


def probe_minibatch(
    runtime,
    batch,
    *,
    actor_parameters: Iterable[torch.Tensor],
    reward_start: int,
    reward_horizon: int,
    outcome_labels: Sequence[str],
    role_labels: Sequence[str],
):
    """Recompute one live minibatch and attribute its actor gradient."""

    lower = float(runtime.clip_advantage_lower_quantile)
    upper = float(runtime.clip_advantage_upper_quantile)
    if (lower, upper) != (0.0, 1.0):
        raise ValueError(
            "credit probe currently requires the active E5 0/100 advantage contract"
        )
    actor_parameters = tuple(actor_parameters)
    new_logprobs = runtime.get_logprobs_subsample(
        batch.condition,
        batch.chains_prev,
        batch.chains_next,
        batch.denoising_inds,
    )
    credit = compute_policy_credit(
        new_logprobs,
        batch.oldlogprobs,
        advantages=batch.advantages,
        denoising_inds=batch.denoising_inds,
        reward_start=reward_start,
        reward_horizon=reward_horizon,
        norm_adv=bool(runtime.norm_adv),
        gamma_denoising=float(runtime.gamma_denoising),
        ft_denoising_steps=int(runtime.ft_denoising_steps),
        clip_ploss_coef=float(runtime.clip_ploss_coef),
        clip_ploss_coef_base=float(runtime.clip_ploss_coef_base),
        clip_ploss_coef_rate=float(runtime.clip_ploss_coef_rate),
    )
    losses = credit["per_sample_policy_loss"]
    return {
        "credit": credit,
        "gradient_by_outcome": summarize_group_gradient_contributions(
            losses, actor_parameters, outcome_labels
        ),
        "gradient_by_role": summarize_group_gradient_contributions(
            losses, actor_parameters, role_labels
        ),
    }


def summarize_group_gradient_contributions(
    per_sample_losses: torch.Tensor,
    parameters: Iterable[torch.Tensor],
    group_labels: Sequence[str],
):
    """Measure additive group contributions without populating ``.grad``."""

    if per_sample_losses.ndim != 1 or per_sample_losses.numel() == 0:
        raise ValueError("per_sample_losses must be a non-empty rank-1 tensor")
    _require_finite("per_sample_losses", per_sample_losses)
    parameters = tuple(parameters)
    if not parameters or any(not parameter.requires_grad for parameter in parameters):
        raise ValueError("parameters must be a non-empty trainable collection")
    labels = tuple(str(label) for label in group_labels)
    if len(labels) != per_sample_losses.numel() or any(not label for label in labels):
        raise ValueError("group_labels must provide one non-empty label per sample")

    overall_loss = per_sample_losses.mean()
    overall_gradient = _gradient_tensors(
        overall_loss, parameters, retain_graph=True
    )
    overall_norm = _gradient_norm(overall_gradient)
    reconstruction_residual = tuple(-gradient.clone() for gradient in overall_gradient)
    groups = {}
    sample_count = per_sample_losses.numel()
    for label in sorted(set(labels)):
        mask = torch.tensor(
            [value == label for value in labels],
            dtype=torch.bool,
            device=per_sample_losses.device,
        )
        contribution_loss = per_sample_losses[mask].sum() / sample_count
        gradient = _gradient_tensors(
            contribution_loss, parameters, retain_graph=True
        )
        gradient_norm = _gradient_norm(gradient)
        cosine = None
        if overall_norm > 0.0 and gradient_norm > 0.0:
            cosine = _gradient_dot(overall_gradient, gradient) / (
                overall_norm * gradient_norm
            )
        count = int(mask.sum().item())
        groups[label] = {
            "count": count,
            "frequency": count / sample_count,
            "sample_loss_mean": float(per_sample_losses[mask].mean().detach().item()),
            "loss_contribution": float(contribution_loss.detach().item()),
            "gradient_norm": gradient_norm,
            "cosine_with_overall": cosine,
        }
        for residual, contribution in zip(reconstruction_residual, gradient):
            residual.add_(contribution)

    reconstruction_error = max(
        float(residual.abs().max().item()) for residual in reconstruction_residual
    )
    return {
        "overall": {
            "sample_count": sample_count,
            "loss_mean": float(overall_loss.detach().item()),
            "gradient_norm": overall_norm,
        },
        "groups": groups,
        "gradient_reconstruction_max_abs": reconstruction_error,
    }
