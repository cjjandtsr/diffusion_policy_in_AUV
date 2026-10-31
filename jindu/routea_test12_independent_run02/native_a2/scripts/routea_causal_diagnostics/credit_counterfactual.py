"""Pure D2-C-LAC credit counterfactual contracts.

The functions in this module do not load models, mutate rollout buffers, or
touch the filesystem.  They distinguish natural time limits from diagnostic
collection boundaries and construct the five preregistered credit variants.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional, Sequence

import torch

from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.gae import compute_gae


VARIANT_NAMES = (
    "C0-current",
    "C1-finite",
    "C2-finite-c10",
    "C3-finite-c20",
    "C4-role-c20",
)

REFERENCE_NAMES = (
    "R-progress",
    "R-ratchet",
    "R-safe-arrival",
    "R-rescue",
    "R-reached",
)


def _done_reason(row: Mapping[str, Any]) -> str:
    terminal = row.get("terminal_facts")
    if not isinstance(terminal, Mapping) or "done_reason" not in terminal:
        raise ValueError("transition requires terminal_facts.done_reason")
    return str(terminal["done_reason"])


def _validate_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("D2-C requires at least one transition")
    identities = []
    roles = []
    timeout_count = 0
    boundary_count = 0
    for row in rows:
        identity = str(row.get("transition_id") or "")
        if not identity:
            raise ValueError("transition_id must be non-empty")
        identities.append(identity)
        role = str(row.get("role") or "")
        if not role:
            raise ValueError("role must be non-empty")
        roles.append(role)
        values = (row.get("reward"), row.get("value"), row.get("next_value"))
        if not all(value is not None and math.isfinite(float(value)) for value in values):
            raise ValueError("reward/value/next_value must be finite")
        terminated = bool(row.get("terminated", False))
        truncated = bool(row.get("truncated", False))
        boundary = bool(row.get("collection_boundary", False))
        reason = _done_reason(row)
        if truncated and (terminated or boundary or reason != "time_limit"):
            raise ValueError(
                "truncated transition must be a real time_limit distinct from "
                "terminated and collection_boundary"
            )
        if boundary and (terminated or truncated):
            raise ValueError("collection_boundary must be a separate diagnostic fact")
        if reason == "time_limit" and not truncated:
            raise ValueError("time_limit done_reason requires truncated=true")
        timeout_count += int(truncated)
        boundary_count += int(boundary)
    if len(set(identities)) != len(identities):
        raise ValueError("transition_id values must be unique")
    return {
        "transition_count": len(rows),
        "timeout_count": timeout_count,
        "collection_boundary_count": boundary_count,
        "terminated_count": sum(bool(row.get("terminated", False)) for row in rows),
        "role_pair_counts": dict(sorted(Counter(roles).items())),
    }


def _normalized(values: torch.Tensor) -> torch.Tensor:
    if values.numel() < 2:
        return torch.zeros_like(values)
    std = values.std()
    if float(std) == 0.0:
        return torch.zeros_like(values)
    return (values - values.mean()) / (std + 1e-8)


def normalize_advantages(
    advantages: torch.Tensor,
    roles: Sequence[str],
    *,
    mode: str,
    pair_repeats: int = 1,
) -> torch.Tensor:
    """Apply the registered global or per-role FP32 calibration."""

    values = torch.as_tensor(advantages, dtype=torch.float32, device="cpu")
    if values.ndim != 1 or len(values) != len(roles):
        raise ValueError("advantages and roles must have the same rank-1 length")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("advantages must be finite")
    if int(pair_repeats) <= 0:
        raise ValueError("pair_repeats must be positive")
    expanded = values.repeat_interleave(int(pair_repeats))
    if mode == "global":
        return _normalized(expanded)[:: int(pair_repeats)]
    if mode != "per_role":
        raise ValueError("normalization mode must be global or per_role")
    result = torch.empty_like(values)
    by_role = defaultdict(list)
    for index, role in enumerate(roles):
        by_role[str(role)].append(index)
    for indices in by_role.values():
        selected = torch.tensor(indices, dtype=torch.long)
        role_values = values[selected]
        result[selected] = _normalized(
            role_values.repeat_interleave(int(pair_repeats))
        )[:: int(pair_repeats)]
    return result


def _counterfactual_gae(
    rows: Sequence[Mapping[str, Any]],
    *,
    gamma: float,
    gae_lambda: float,
    timeout_cost: float,
) -> torch.Tensor:
    rewards = torch.tensor([float(row["reward"]) for row in rows], dtype=torch.float32)
    values = torch.tensor([float(row["value"]) for row in rows], dtype=torch.float32)
    next_values = torch.tensor(
        [float(row["next_value"]) for row in rows], dtype=torch.float32
    )
    timeouts = torch.tensor([bool(row["truncated"]) for row in rows])
    terminated = torch.tensor([bool(row["terminated"]) for row in rows])
    boundaries = torch.tensor([bool(row["collection_boundary"]) for row in rows])
    rewards = rewards - timeouts.to(rewards.dtype) * float(timeout_cost)
    continuation_stop = torch.logical_or(torch.logical_or(terminated, timeouts), boundaries)
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros((), dtype=torch.float32)
    for index in range(len(rows) - 1, -1, -1):
        bootstrap = 1.0 - torch.logical_or(terminated[index], timeouts[index]).float()
        continuation = 1.0 - continuation_stop[index].float()
        delta = (
            rewards[index]
            + float(gamma) * next_values[index] * bootstrap
            - values[index]
        )
        carry = delta + float(gamma) * float(gae_lambda) * continuation * carry
        advantages[index] = carry
    return advantages


def build_credit_variants(
    rows: Sequence[Mapping[str, Any]],
    *,
    gamma: float,
    gae_lambda: float,
    official_advantages: torch.Tensor,
    c0_tolerance: float = 1e-6,
) -> Dict[str, Dict[str, Any]]:
    """Construct C0-C4 while preserving the official C0 tensor identity."""

    audit = _validate_rows(rows)
    roles = tuple(str(row["role"]) for row in rows)
    rewards = torch.tensor([float(row["reward"]) for row in rows], dtype=torch.float32)
    values = torch.tensor([float(row["value"]) for row in rows], dtype=torch.float32)
    next_values = torch.tensor(
        [float(row["next_value"]) for row in rows], dtype=torch.float32
    )
    terminated = torch.tensor([bool(row["terminated"]) for row in rows])
    effective_truncated = torch.tensor(
        [bool(row["truncated"] or row["collection_boundary"]) for row in rows]
    )
    official = torch.as_tensor(official_advantages)
    if official.dtype != torch.float32 or official.device.type != "cpu":
        raise ValueError("official_advantages must be a CPU FP32 tensor")
    if official.ndim != 1 or len(official) != len(rows):
        raise ValueError("official_advantages length mismatch")
    rebuilt_c0, _ = compute_gae(
        rewards,
        values,
        next_values,
        terminated,
        effective_truncated,
        float(gamma),
        float(gae_lambda),
    )
    parity = float(torch.max(torch.abs(rebuilt_c0 - official)))
    if parity > float(c0_tolerance) or not torch.equal(rebuilt_c0, official):
        raise ValueError(
            "C0 official-path parity failed: max_abs={} exact={}".format(
                parity, torch.equal(rebuilt_c0, official)
            )
        )

    raw = {
        "C0-current": official,
        "C1-finite": _counterfactual_gae(
            rows, gamma=gamma, gae_lambda=gae_lambda, timeout_cost=0.0
        ),
        "C2-finite-c10": _counterfactual_gae(
            rows, gamma=gamma, gae_lambda=gae_lambda, timeout_cost=10.0
        ),
        "C3-finite-c20": _counterfactual_gae(
            rows, gamma=gamma, gae_lambda=gae_lambda, timeout_cost=20.0
        ),
    }
    raw["C4-role-c20"] = raw["C3-finite-c20"]
    result = {}
    for name in VARIANT_NAMES:
        mode = "per_role" if name == "C4-role-c20" else "global"
        variant_audit = dict(audit)
        variant_audit.update(
            {
                "normalization": (
                    "per_role_pair_weighted" if mode == "per_role" else "global"
                ),
                "timeout_cost": 20.0 if name in ("C3-finite-c20", "C4-role-c20")
                else 10.0 if name == "C2-finite-c10" else 0.0,
                "timeout_bootstrap": name == "C0-current",
            }
        )
        result[name] = {
            "raw_advantages": raw[name],
            "returns": raw[name] + values,
            "normalized_advantages": normalize_advantages(
                raw[name], roles, mode=mode, pair_repeats=10
            ),
            "audit": variant_audit,
        }
    result["C0-current"]["c0_parity_max_abs"] = parity
    result["C0-current"]["c0_exact"] = True
    return result


def _scaled_positive(values: Sequence[float]) -> Dict[str, Any]:
    tensor = torch.tensor(values, dtype=torch.float32)
    if bool(torch.any(tensor < 0)) or not bool(torch.isfinite(tensor).all()):
        raise ValueError("reference weights must be finite and non-negative")
    selected = tensor[tensor > 0]
    if selected.numel() == 0:
        return {"weights": tensor, "nonzero_count": 0, "reason": "no_positive_weight"}
    return {
        "weights": tensor / selected.mean(),
        "nonzero_count": int(selected.numel()),
        "reason": None,
    }


def build_reference_weights(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Build the five preregistered explanatory reference weight vectors."""

    _validate_rows(rows)
    progress = []
    ratchet = [0.0] * len(rows)
    by_episode = defaultdict(list)
    for index, row in enumerate(rows):
        before = float(row["goal_distance_before_3d"])
        after = float(row["goal_distance_after_3d"])
        if not math.isfinite(before) or not math.isfinite(after):
            raise ValueError("goal distances must be finite")
        progress.append(max(before - after, 0.0))
        by_episode[str(row["episode_identity_hash"])].append((index, row))
    for episode_rows in by_episode.values():
        episode_rows.sort(key=lambda pair: int(pair[1]["decision_index"]))
        best = float(episode_rows[0][1]["goal_distance_before_3d"])
        for index, row in episode_rows:
            after = float(row["goal_distance_after_3d"])
            new_best = min(best, after)
            ratchet[index] = max(best - new_best, 0.0)
            best = new_best
    raw = {
        "R-progress": progress,
        "R-ratchet": ratchet,
        "R-safe-arrival": [float(bool(row["entered_goal_sphere_safe"])) for row in rows],
        "R-rescue": [float(str(row["pair_class"]) == "rescue") for row in rows],
        "R-reached": [float(str(row["outcome"]) == "reached") for row in rows],
    }
    return {name: _scaled_positive(raw[name]) for name in REFERENCE_NAMES}


def _gradient_tensors(loss, parameters, *, retain_graph):
    gradients = torch.autograd.grad(
        loss, tuple(parameters), retain_graph=retain_graph, allow_unused=True
    )
    return tuple(
        (
            torch.zeros_like(parameter, device="cpu", dtype=torch.float32)
            if gradient is None
            else gradient.detach().float().cpu()
        )
        for parameter, gradient in zip(parameters, gradients)
    )


def _accumulate(target, addition):
    if target is None:
        return tuple(value.clone() for value in addition)
    for total, value in zip(target, addition):
        total.add_(value)
    return target


def _flatten(gradients):
    return torch.cat([gradient.reshape(-1) for gradient in gradients]).float()


def compute_credit_variant_gradient(
    runtime,
    buffer,
    actor_parameters,
    *,
    raw_advantages,
    roles,
    group_labels=None,
    normalization,
    device,
    microbatch_pairs=128,
):
    """Compute a full-joint D2-C LoRA descent through official loss code."""

    from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.official_loss_components import (
        AdvantageStats,
        compute_official_components,
    )

    if normalization not in ("global", "per_role", "none"):
        raise ValueError("normalization must be global, per_role, or none")
    transition_count = len(buffer)
    raw = torch.as_tensor(raw_advantages, dtype=torch.float32, device="cpu")
    role_names = tuple(str(role) for role in roles)
    groups = (
        tuple(str(label) for label in group_labels)
        if group_labels is not None
        else tuple("all" for _ in range(transition_count))
    )
    if raw.ndim != 1 or len(raw) != transition_count:
        raise ValueError("raw_advantages must contain one value per transition")
    if len(role_names) != transition_count:
        raise ValueError("roles must contain one value per transition")
    if len(groups) != transition_count or any(not label for label in groups):
        raise ValueError("group_labels must contain one non-empty value per transition")
    if not bool(torch.isfinite(raw).all()):
        raise ValueError("raw_advantages must be finite")
    if int(microbatch_pairs) <= 0:
        raise ValueError("microbatch_pairs must be positive")

    transition_indices = torch.arange(transition_count).repeat_interleave(10)
    denoising_indices = torch.arange(10).repeat(transition_count)
    logical = buffer.materialize_pairs(
        transition_indices,
        denoising_indices,
        device=device,
        pass_id=0,
        mode="d2c_read_only",
        microbatch_pairs=int(microbatch_pairs),
        schedule_audit={"diagnostic_only": True, "full_joint": True},
    )
    expanded_advantages = raw[transition_indices].to(device)
    expanded_roles = tuple(role_names[int(index)] for index in transition_indices)
    expanded_groups = tuple(groups[int(index)] for index in transition_indices)
    unique_roles = tuple(sorted(set(expanded_roles)))
    lower = float(getattr(runtime, "clip_advantage_lower_quantile", 0.0))
    upper = float(getattr(runtime, "clip_advantage_upper_quantile", 1.0))
    if normalization == "none":
        identity_stats = AdvantageStats(
            mean=torch.zeros((), dtype=torch.float32, device=device),
            std=torch.ones((), dtype=torch.float32, device=device),
            lower_quantile=expanded_advantages.min(),
            upper_quantile=expanded_advantages.max(),
        )
        role_stats = {role: identity_stats for role in unique_roles}
    elif normalization == "global":
        stats = AdvantageStats.from_tensor(expanded_advantages, lower, upper)
        role_stats = {role: stats for role in unique_roles}
    else:
        role_stats = {}
        for role in unique_roles:
            mask = torch.tensor(
                [name == role for name in expanded_roles],
                dtype=torch.bool,
                device=device,
            )
            role_stats[role] = AdvantageStats.from_tensor(
                expanded_advantages[mask], lower, upper
            )

    parameters = tuple(actor_parameters)
    if not parameters or any(not parameter.requires_grad for parameter in parameters):
        raise ValueError("actor_parameters must be non-empty and trainable")
    role_totals = {role: None for role in unique_roles}
    group_totals = {label: None for label in sorted(set(expanded_groups))}
    pair_count = int(len(transition_indices))
    for start in range(0, pair_count, int(microbatch_pairs)):
        end = min(start + int(microbatch_pairs), pair_count)
        local_roles = expanded_roles[start:end]
        local_groups = expanded_groups[start:end]
        if normalization in ("global", "none"):
            micro = SimpleNamespace(
                condition={key: value[start:end] for key, value in logical.condition.items()},
                chains_prev=logical.chains_prev[start:end],
                chains_next=logical.chains_next[start:end],
                denoising_inds=logical.denoising_inds[start:end],
                returns=logical.returns[start:end],
                oldvalues=logical.oldvalues[start:end],
                advantages=expanded_advantages[start:end],
                oldlogprobs=logical.oldlogprobs[start:end],
            )
            parts = compute_official_components(
                runtime, micro, role_stats[unique_roles[0]],
                reward_start=2, reward_horizon=4,
            )
            present_roles = [role for role in unique_roles if role in local_roles]
            present_groups = sorted(set(local_groups))
            contributions = [
                ("role", role, local_roles) for role in present_roles
            ] + [
                ("group", label, local_groups) for label in present_groups
            ]
            for offset, (kind, label, labels) in enumerate(contributions):
                mask = torch.tensor(
                    [name == label for name in labels],
                    dtype=torch.bool,
                    device=device,
                )
                gradients = _gradient_tensors(
                    parts.policy[mask].sum() / pair_count,
                    parameters,
                    retain_graph=offset < len(contributions) - 1,
                )
                totals = role_totals if kind == "role" else group_totals
                totals[label] = _accumulate(totals[label], gradients)
            continue

        present_roles = [role for role in unique_roles if role in local_roles]
        for role in present_roles:
            local_mask = torch.tensor(
                [name == role for name in local_roles],
                dtype=torch.bool,
                device=device,
            )
            absolute = torch.arange(start, end, device=device)[local_mask]
            micro = SimpleNamespace(
                condition={key: value[absolute] for key, value in logical.condition.items()},
                chains_prev=logical.chains_prev[absolute],
                chains_next=logical.chains_next[absolute],
                denoising_inds=logical.denoising_inds[absolute],
                returns=logical.returns[absolute],
                oldvalues=logical.oldvalues[absolute],
                advantages=expanded_advantages[absolute],
                oldlogprobs=logical.oldlogprobs[absolute],
            )
            parts = compute_official_components(
                runtime, micro, role_stats[role], reward_start=2, reward_horizon=4
            )
            selected_groups = tuple(expanded_groups[int(index)] for index in absolute.cpu())
            present_groups = sorted(set(selected_groups))
            for group_offset, label in enumerate(present_groups):
                group_mask = torch.tensor(
                    [name == label for name in selected_groups],
                    dtype=torch.bool,
                    device=device,
                )
                gradients = _gradient_tensors(
                    parts.policy[group_mask].sum() / pair_count,
                    parameters,
                    retain_graph=group_offset < len(present_groups) - 1,
                )
                role_totals[role] = _accumulate(role_totals[role], gradients)
                group_totals[label] = _accumulate(group_totals[label], gradients)

    empty = tuple(
        torch.zeros_like(parameter.detach(), device="cpu", dtype=torch.float32)
        for parameter in parameters
    )
    role_gradients = {
        role: _flatten(role_totals[role] or empty) for role in unique_roles
    }
    group_gradients = {
        label: _flatten(group_totals[label] or empty) for label in sorted(group_totals)
    }
    total_descent = -torch.stack(tuple(role_gradients.values())).sum(dim=0)
    role_descent = {role: -gradient for role, gradient in role_gradients.items()}
    group_descent = {label: -gradient for label, gradient in group_gradients.items()}
    reconstructed = torch.stack(tuple(role_descent.values())).sum(dim=0)
    return {
        "total_descent": total_descent,
        "role_descent": role_descent,
        "group_descent": group_descent,
        "normalization": normalization,
        "pair_count": pair_count,
        "role_pair_counts": dict(sorted(Counter(expanded_roles).items())),
        "reconstruction_max_abs": float(
            torch.max(torch.abs(reconstructed - total_descent))
        ),
    }


__all__ = (
    "REFERENCE_NAMES",
    "VARIANT_NAMES",
    "build_credit_variants",
    "build_reference_weights",
    "compute_credit_variant_gradient",
    "normalize_advantages",
)
