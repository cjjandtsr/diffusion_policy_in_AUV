"""Independent, JSON-safe audit evidence for one on-policy update."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import math
import re
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import torch


REWARD_TERM_NAMES = (
    "base",
    "pbrs",
    "onset",
    "future_clearance",
    "trap",
    "flip",
    "support",
    "guard",
)
LEARNING_TARGET_NAMES = (
    "rewards",
    "values",
    "next_values",
    "advantages",
    "returns",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _finite_float(value, name):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("{} must be finite".format(name))
    return result


def _scalar_stats(values: Sequence[float]):
    values = tuple(float(value) for value in values)
    if not values:
        return {
            "count": 0,
            "sum": 0.0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "p05": None,
            "p50": None,
            "p95": None,
            "nonzero_count": 0,
            "nonzero_rate": 0.0,
        }
    for value in values:
        if not math.isfinite(value):
            raise ValueError("audit statistics contain a non-finite value")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    quantiles = np.percentile(np.asarray(values, dtype=np.float64), [5, 50, 95])
    nonzero_count = sum(value != 0.0 for value in values)
    return {
        "count": len(values),
        "sum": float(sum(values)),
        "mean": float(mean),
        "std": float(math.sqrt(variance)),
        "min": float(min(values)),
        "max": float(max(values)),
        "p05": float(quantiles[0]),
        "p50": float(quantiles[1]),
        "p95": float(quantiles[2]),
        "nonzero_count": int(nonzero_count),
        "nonzero_rate": float(nonzero_count / len(values)),
    }


def _physical_done_reason(info, canonical_reason):
    """Bounded diagnostic labels; never change authoritative terminal precedence."""

    facts = info.get("transition_facts")
    raw = getattr(facts, "raw_done_reason", None)
    if canonical_reason == "collision":
        if raw == "crash_ground":
            return "crash_ground"
        if isinstance(raw, str) and raw.startswith(("collide(", "penetrate_flyover(")):
            return "obstacle_collision"
        return "collision_unspecified"
    if canonical_reason == "early_death":
        if raw in ("out_of_bounds", "altitude_exceed", "invalid_state"):
            return raw
        return "early_death_unspecified"
    return canonical_reason


class TrainingAudit:
    """Accumulate rollout facts without changing reward or trainer behavior."""

    def __init__(self, term_names=None, *, terminate_at_space_limits=None):
        self.term_names = tuple(term_names or REWARD_TERM_NAMES)
        self.reward_version = ("v2" if "newbest" in self.term_names else
                               "v17" if "overflight" in self.term_names else "v13")
        self.terminate_at_space_limits = ("overflight" not in self.term_names
                                         if terminate_at_space_limits is None
                                         else bool(terminate_at_space_limits))
        self._rewards = []
        self._values = []
        self._next_values = []
        self._task_roles = []
        self._role_transition_counts = Counter()
        self._reward_terms = {name: [] for name in self.term_names}
        self._reward_terms_by_role = defaultdict(
            lambda: {name: [] for name in self.term_names}
        )
        self._term_sum_errors = []
        self._metric_booleans = Counter()
        self._metric_boolean_observations = Counter()
        self._metric_numeric = defaultdict(list)
        self._metric_booleans_by_role = defaultdict(Counter)
        self._metric_boolean_observations_by_role = defaultdict(Counter)
        self._metric_numeric_by_role = defaultdict(lambda: defaultdict(list))
        self._ignored_metric_keys = set()
        self._ignored_metric_keys_by_role = defaultdict(set)
        self._done_reasons = Counter()
        self._physical_done_reasons = Counter()
        self._physical_done_reasons_by_role = defaultdict(Counter)
        self._role_segments = Counter()
        self._role_completed_returns = defaultdict(list)
        self._role_done_reasons = defaultdict(Counter)
        self._segment_returns = []
        self._completed_returns = []
        self._completed_lengths = []
        self._completed_lengths_by_role = defaultdict(list)
        self._action_values_by_dimension = [[] for _ in range(11)]
        self._action_values_by_role = defaultdict(
            lambda: [[] for _ in range(11)]
        )
        self._action_chunk_l2 = []
        self._action_chunk_l2_by_role = defaultdict(list)
        self._partial_segments = 0
        self._active_segment = None
        self._transition_records = []

    def record_transition(
        self,
        *,
        task_id: str,
        task_role: str,
        reward,
        value,
        next_value,
        terminated: bool,
        truncated: bool,
        collection_boundary: bool,
        info: Mapping,
        action_chunk=None,
        audit_context=None,
        bootstrap_allowed=None,
        critic_context=None,
        next_critic_context=None,
    ):
        if not task_id or not task_role:
            raise ValueError("task_id and task_role are required for audit")
        reward = _finite_float(reward, "reward")
        value = _finite_float(value, "value")
        next_value = _finite_float(next_value, "next_value")
        terms = info.get("reward_terms")
        if not isinstance(terms, Mapping) or set(terms) != set(self.term_names):
            raise ValueError(
                "reward term keys must exactly match {}".format(self.term_names)
            )
        numeric_terms = {
            name: _finite_float(terms[name], "reward term {}".format(name))
            for name in self.term_names
        }
        context = dict(audit_context or {})
        required_context = {
            "run_seed", "update_index", "policy_state_update_index",
            "transition_id", "episode_id", "decision_index",
            "task_identity_hash", "world_id", "task_seed", "reset_seed",
            "admission_manifest_hash", "runtime_identity_hash", "source_set_hash",
            "config_hash", "checkpoint_identity_hash",
            "goal_xyz", "position_before_xyz",
            "position_after_xyz", "goal_distance_before_3d",
            "goal_distance_after_3d", "goal_error_xy_before",
            "goal_error_z_before", "goal_error_xy_after", "goal_error_z_after",
            "observation_provenance", "observation_provenance_after",
        }
        if "overflight" in self.term_names and not context:
            raise ValueError("Reward v1.7 audit requires audit_context")
        if context and set(context) != required_context:
            raise ValueError("audit_context keys differ from schema v5")
        if context:
            for name in (
                "episode_id", "task_identity_hash", "admission_manifest_hash",
                "runtime_identity_hash", "source_set_hash", "config_hash",
                "checkpoint_identity_hash",
            ):
                if not _SHA256.fullmatch(str(context[name])):
                    raise ValueError("{} must be a lowercase sha256".format(name))
            for name in (
                "goal_xyz", "position_before_xyz", "position_after_xyz",
            ):
                array = np.asarray(context[name], dtype=np.float64)
                if array.shape != (3,) or not np.isfinite(array).all():
                    raise ValueError("{} must be finite xyz".format(name))
            for name in (
                "goal_distance_before_3d", "goal_distance_after_3d",
                "goal_error_xy_before", "goal_error_z_before",
                "goal_error_xy_after", "goal_error_z_after",
            ):
                context[name] = _finite_float(context[name], name)
            for name in (
                "run_seed", "update_index", "policy_state_update_index",
                "decision_index", "task_seed", "reset_seed",
            ):
                if isinstance(context[name], bool) or int(context[name]) != context[name]:
                    raise ValueError("{} must be an integer".format(name))
                context[name] = int(context[name])
            facts = info.get("transition_facts")
            self._transition_records.append({
                "schema": "routea_training_transition_v5",
                **context,
                "task_id": task_id,
                "role": task_role,
                "reward": reward,
                "value": value,
                "next_value": next_value,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "collection_boundary": bool(collection_boundary),
                "bootstrap_allowed": bool(not terminated if bootstrap_allowed is None else bootstrap_allowed),
                "gae_continuation": not (terminated or truncated or collection_boundary),
                "critic_episode_context": critic_context,
                "next_critic_episode_context": next_critic_context,
                "terminal_facts": asdict(facts) if hasattr(facts, "__dataclass_fields__") else None,
                "reward_terms": numeric_terms,
                "reward_metrics": {
                    str(key): metric for key, metric in info.get("reward_metrics", {}).items()
                    if metric is None or isinstance(metric, (bool, int, float, str))
                },
                "action_chunk": (
                    torch.as_tensor(action_chunk).detach().cpu().float().tolist()
                    if action_chunk is not None else None
                ),
                "net_goal_approach_3d": float(
                    context["goal_distance_before_3d"] - context["goal_distance_after_3d"]
                ),
            })

        self._rewards.append(reward)
        self._values.append(value)
        self._next_values.append(next_value)
        self._task_roles.append(task_role)
        self._role_transition_counts[task_role] += 1
        for name, term_value in numeric_terms.items():
            self._reward_terms[name].append(term_value)
            self._reward_terms_by_role[task_role][name].append(term_value)
        self._term_sum_errors.append(reward - sum(numeric_terms.values()))

        if action_chunk is not None:
            action_tensor = torch.as_tensor(action_chunk).detach().cpu().float()
            if tuple(action_tensor.shape) != (4, 11):
                raise ValueError("rollout action_chunk must have shape (4,11)")
            if not torch.isfinite(action_tensor).all().item():
                raise ValueError("rollout action_chunk contains non-finite values")
            for dimension in range(11):
                values = action_tensor[:, dimension].tolist()
                self._action_values_by_dimension[dimension].extend(values)
                self._action_values_by_role[task_role][dimension].extend(values)
            chunk_l2 = float(action_tensor.pow(2).sum().sqrt().item())
            self._action_chunk_l2.append(chunk_l2)
            self._action_chunk_l2_by_role[task_role].append(chunk_l2)

        metrics = info.get("reward_metrics", {})
        if not isinstance(metrics, Mapping):
            raise ValueError("reward_metrics must be a mapping")
        for name, metric_value in metrics.items():
            if isinstance(metric_value, (bool, np.bool_)):
                self._metric_booleans[name] += int(bool(metric_value))
                self._metric_boolean_observations[name] += 1
                self._metric_booleans_by_role[task_role][name] += int(
                    bool(metric_value)
                )
                self._metric_boolean_observations_by_role[task_role][name] += 1
            elif isinstance(metric_value, (int, float, np.number)):
                numeric_metric = _finite_float(
                    metric_value, "reward metric {}".format(name)
                )
                self._metric_numeric[name].append(numeric_metric)
                self._metric_numeric_by_role[task_role][name].append(numeric_metric)
            elif metric_value is not None:
                self._ignored_metric_keys.add(str(name))
                self._ignored_metric_keys_by_role[task_role].add(str(name))

        if self._active_segment is None:
            self._active_segment = {
                "task_id": task_id,
                "task_role": task_role,
                "return": 0.0,
                "length": 0,
            }
        elif (
            self._active_segment["task_id"] != task_id
            or self._active_segment["task_role"] != task_role
        ):
            raise ValueError("task changed before the active segment ended")
        self._active_segment["return"] += reward
        self._active_segment["length"] += 1

        real_end = bool(terminated or truncated)
        segment_end = bool(real_end or collection_boundary)
        if segment_end:
            reason = str(info.get("done_reason", "none"))
            if not real_end:
                reason = "collection_boundary"
                self._partial_segments += 1
            elif reason == "none":
                raise ValueError("completed episode requires a done_reason")
            segment_return = float(self._active_segment["return"])
            self._segment_returns.append(segment_return)
            if real_end:
                physical_reason = _physical_done_reason(info, reason)
                self._physical_done_reasons[physical_reason] += 1
                self._physical_done_reasons_by_role[task_role][physical_reason] += 1
                self._completed_returns.append(segment_return)
                self._completed_lengths.append(self._active_segment["length"])
                self._role_completed_returns[task_role].append(segment_return)
                self._completed_lengths_by_role[task_role].append(
                    self._active_segment["length"]
                )
                self._role_done_reasons[task_role][reason] += 1
            self._done_reasons[reason] += 1
            self._role_segments[task_role] += 1
            self._active_segment = None

    def finalize(self, learning_targets: Mapping[str, torch.Tensor]):
        if self._active_segment is not None:
            raise RuntimeError("audit cannot finalize with an open rollout segment")
        if set(learning_targets) != set(LEARNING_TARGET_NAMES):
            raise ValueError(
                "learning target keys must exactly match {}".format(
                    LEARNING_TARGET_NAMES
                )
            )
        target_stats = {}
        target_tensors = {}
        for name in LEARNING_TARGET_NAMES:
            tensor = torch.as_tensor(learning_targets[name]).detach().cpu().float()
            if tensor.ndim != 1 or tensor.numel() != len(self._rewards):
                raise ValueError(
                    "{} must contain one scalar per transition".format(name)
                )
            if not torch.isfinite(tensor).all().item():
                raise ValueError("{} contains non-finite values".format(name))
            target_tensors[name] = tensor
            target_stats[name] = _scalar_stats(tensor.tolist())
        target_stats_by_role = {}
        for role in sorted(self._role_transition_counts):
            indices = [
                index
                for index, transition_role in enumerate(self._task_roles)
                if transition_role == role
            ]
            target_stats_by_role[role] = {
                name: _scalar_stats(target_tensors[name][indices].tolist())
                for name in LEARNING_TARGET_NAMES
            }
        if self._transition_records:
            if len(self._transition_records) != len(self._rewards):
                raise RuntimeError("transition audit record count mismatch")
            advantages = target_tensors["advantages"]
            advantage_mean = advantages.mean()
            advantage_std = advantages.std(unbiased=False)
            normalized = (advantages - advantage_mean) / (advantage_std + 1e-8)
            for index, record in enumerate(self._transition_records):
                record["advantage_raw"] = float(advantages[index])
                record["advantage_normalized_outer_update"] = float(normalized[index])
                record["return_target"] = float(target_tensors["returns"][index])
        returns = target_tensors["returns"]
        values = target_tensors["values"]
        return_variance = float(torch.var(returns, unbiased=False).item())
        explained_variance = None
        if return_variance > 0.0:
            residual_variance = float(
                torch.var(returns - values, unbiased=False).item()
            )
            explained_variance = float(1.0 - residual_variance / return_variance)
        return {
            "schema_version": 5 if "overflight" in self.term_names else 4,
            "reward_schema_version": self.reward_version,
            "altitude_termination_enabled": self.terminate_at_space_limits,
            "transitions": len(self._rewards),
            "transition_records": tuple(self._transition_records),
            "reward": {
                "total": _scalar_stats(self._rewards),
                "terms": {
                    name: _scalar_stats(self._reward_terms[name])
                    for name in self.term_names
                },
                "terms_by_role": {
                    role: {
                        name: _scalar_stats(role_terms[name])
                        for name in self.term_names
                    }
                    for role, role_terms in sorted(
                        self._reward_terms_by_role.items()
                    )
                },
                "term_sum_error_max_abs": float(
                    max((abs(value) for value in self._term_sum_errors), default=0.0)
                ),
            },
            "reward_metrics": {
                "boolean_true_counts": dict(sorted(self._metric_booleans.items())),
                "boolean_observation_counts": dict(
                    sorted(self._metric_boolean_observations.items())
                ),
                "boolean_true_rates": {
                    name: float(
                        self._metric_booleans[name]
                        / self._metric_boolean_observations[name]
                    )
                    for name in sorted(self._metric_boolean_observations)
                },
                "numeric": {
                    name: _scalar_stats(values)
                    for name, values in sorted(self._metric_numeric.items())
                },
                "ignored_non_scalar_keys": sorted(self._ignored_metric_keys),
                "by_role": {
                    role: {
                        "boolean_true_counts": dict(
                            sorted(self._metric_booleans_by_role[role].items())
                        ),
                        "boolean_observation_counts": dict(
                            sorted(
                                self._metric_boolean_observations_by_role[role].items()
                            )
                        ),
                        "boolean_true_rates": {
                            name: float(
                                self._metric_booleans_by_role[role][name]
                                / self._metric_boolean_observations_by_role[role][name]
                            )
                            for name in sorted(
                                self._metric_boolean_observations_by_role[role]
                            )
                        },
                        "numeric": {
                            name: _scalar_stats(values)
                            for name, values in sorted(
                                self._metric_numeric_by_role[role].items()
                            )
                        },
                        "ignored_non_scalar_keys": sorted(
                            self._ignored_metric_keys_by_role[role]
                        ),
                    }
                    for role in sorted(self._role_transition_counts)
                },
            },
            "learning_targets": target_stats,
            "learning_targets_by_role": target_stats_by_role,
            "critic": {"explained_variance": explained_variance},
            "action": {
                "rollout_view_shape": [4, 11],
                "per_dimension": {
                    str(index): _scalar_stats(values)
                    for index, values in enumerate(self._action_values_by_dimension)
                },
                "chunk_l2": _scalar_stats(self._action_chunk_l2),
                "by_role": {
                    role: {
                        "per_dimension": {
                            str(index): _scalar_stats(values)
                            for index, values in enumerate(role_dimensions)
                        },
                        "chunk_l2": _scalar_stats(
                            self._action_chunk_l2_by_role[role]
                        ),
                    }
                    for role, role_dimensions in sorted(
                        self._action_values_by_role.items()
                    )
                },
            },
            "episodes": {
                "completed": len(self._completed_returns),
                "segments": len(self._segment_returns),
                "partial_segments": self._partial_segments,
                "completed_return": _scalar_stats(self._completed_returns),
                "completed_length": _scalar_stats(self._completed_lengths),
                "segment_return": _scalar_stats(self._segment_returns),
                "done_reason_counts": dict(sorted(self._done_reasons.items())),
                # Completed episodes only; rollout collection cutoffs are excluded.
                "physical_done_reason_counts": dict(
                    sorted(self._physical_done_reasons.items())
                ),
                "physical_done_reason_rates": {
                    reason: count / len(self._completed_returns)
                    for reason, count in sorted(self._physical_done_reasons.items())
                },
                "physical_done_reason_counts_by_role": {
                    role: dict(sorted(reasons.items()))
                    for role, reasons in sorted(
                        self._physical_done_reasons_by_role.items()
                    )
                },
                "role_segment_counts": dict(sorted(self._role_segments.items())),
                "role_transition_counts": dict(
                    sorted(self._role_transition_counts.items())
                ),
                "completed_return_by_role": {
                    role: _scalar_stats(values)
                    for role, values in sorted(self._role_completed_returns.items())
                },
                "done_reason_counts_by_role": {
                    role: dict(sorted(reasons.items()))
                    for role, reasons in sorted(self._role_done_reasons.items())
                },
                "completed_length_by_role": {
                    role: _scalar_stats(values)
                    for role, values in sorted(
                        self._completed_lengths_by_role.items()
                    )
                },
            },
        }


def evaluate_transition_credit_audit(trainer, buffer, *, microbatch_pairs=128):
    """Bind one deterministic post-update denoising credit record to every transition.

    The optimizer may evaluate a transition repeatedly across passes or all ten
    denoising steps.  Audit identity must nevertheless remain one-to-one with
    rollout transitions, so the audited ``k`` is selected from the immutable
    transition id.  This is evidence only: it neither changes the training
    schedule nor calls backward/step.
    """

    from types import SimpleNamespace

    from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.official_loss_components import (
        AdvantageStats,
        compute_official_components,
    )
    from scripts.dppo_credit_support.credit_probe import reverse_kernel_kl_record

    count = len(buffer)
    if count < 1:
        raise ValueError("transition credit audit requires a non-empty buffer")
    if isinstance(microbatch_pairs, bool) or int(microbatch_pairs) < 1:
        raise ValueError("microbatch_pairs must be positive")
    transition_ids = tuple(str(item["transition_id"]) for item in buffer._items)
    if len(set(transition_ids)) != count:
        raise ValueError("transition credit audit requires unique transition ids")
    denoising_count = int(trainer.runtime.ft_denoising_steps)
    if denoising_count != 10:
        raise ValueError("Route A transition credit audit requires ten denoising steps")
    transition_indices = torch.arange(count, dtype=torch.long)
    denoising_indices = torch.tensor(
        [
            int.from_bytes(hashlib.sha256(identity.encode("utf-8")).digest()[:4], "big")
            % denoising_count
            for identity in transition_ids
        ],
        dtype=torch.long,
    )
    device = trainer.critic_parameters[0].device
    batch = buffer.materialize_pairs(
        transition_indices,
        denoising_indices,
        device=device,
        pass_id=-1,
        mode="post_update_transition_audit",
        microbatch_pairs=int(microbatch_pairs),
        schedule_audit={
            "mode": "post_update_transition_audit",
            "pair_count": count,
        },
    )
    stats = AdvantageStats.from_tensor(
        batch.advantages,
        float(getattr(trainer.runtime, "clip_advantage_lower_quantile", 0.0)),
        float(getattr(trainer.runtime, "clip_advantage_upper_quantile", 1.0)),
    )
    normalized = (batch.advantages - stats.mean) / (stats.std + 1e-8)
    normalized = normalized.clamp(stats.lower_quantile, stats.upper_quantile)
    records = []
    fork_devices = []
    if device.type == "cuda":
        fork_devices = [
            device.index if device.index is not None else torch.cuda.current_device()
        ]
    with torch.random.fork_rng(devices=fork_devices, enabled=True):
        with torch.no_grad():
            for start in range(0, count, int(microbatch_pairs)):
                end = min(start + int(microbatch_pairs), count)
                micro = SimpleNamespace(
                    condition={key: value[start:end] for key, value in batch.condition.items()},
                    chains_prev=batch.chains_prev[start:end],
                    chains_next=batch.chains_next[start:end],
                    denoising_inds=batch.denoising_inds[start:end],
                    returns=batch.returns[start:end],
                    oldvalues=batch.oldvalues[start:end],
                    advantages=batch.advantages[start:end],
                    oldlogprobs=batch.oldlogprobs[start:end],
                )
                parts = compute_official_components(
                    trainer.runtime,
                    micro,
                    stats,
                    reward_start=trainer.config.reward_start,
                    reward_horizon=trainer.config.reward_horizon,
                    use_bc_loss=False,
                )
                kl_record = reverse_kernel_kl_record(trainer.runtime, micro)
                old = micro.oldlogprobs.clamp(-5, 2)[
                    :,
                    trainer.config.reward_start : (
                        trainer.config.reward_start + trainer.config.reward_horizon
                    ),
                    :,
                ].mean(dim=(-1, -2))
                new = old + parts.logratio
                elements_per_pair = int(micro.chains_next[0].numel())
                for local_index in range(end - start):
                    global_index = start + local_index
                    record = {
                        "schema": "routea_transition_credit_v5",
                        "transition_id": batch.transition_ids[global_index],
                        "transition_index": int(batch.transition_indices[global_index]),
                        "role": batch.roles[global_index],
                        "outcome": batch.outcomes[global_index],
                        "observation_provenance_hash": batch.provenance_hashes[global_index],
                        "denoising_step_k": int(batch.denoising_inds[global_index]),
                        "old_log_prob": float(old[local_index]),
                        "new_log_prob": float(new[local_index]),
                        "log_ratio": float(parts.logratio[local_index]),
                        "ratio": float(parts.ratio[local_index]),
                        "approx_kl": float(parts.approx_kl[local_index]),
                        "clip_fraction": float(parts.clip_mask[local_index].float()),
                        "analytic_bc_reverse_kl_sum": float(
                            kl_record["sum_per_pair"][local_index]
                        ),
                        "analytic_bc_reverse_kl_mean": float(
                            kl_record["sum_per_pair"][local_index] / elements_per_pair
                        ),
                        "advantage_raw": float(batch.advantages[global_index]),
                        "advantage_normalized_for_update": float(normalized[global_index]),
                    }
                    numeric = (
                        value for key, value in record.items()
                        if key not in {
                            "schema", "transition_id", "role", "outcome",
                            "observation_provenance_hash",
                        }
                    )
                    if not all(math.isfinite(float(value)) for value in numeric):
                        raise ValueError("transition credit audit contains non-finite values")
                    if not _SHA256.fullmatch(record["observation_provenance_hash"]):
                        raise ValueError("transition credit provenance must be a lowercase sha256")
                    records.append(record)
    if len(records) != count:
        raise RuntimeError("transition credit audit coverage mismatch")
    return tuple(records)


@dataclass(frozen=True)
class ParameterSnapshot:
    _groups: Dict[str, Tuple[torch.Tensor, ...]]

    @classmethod
    def capture(cls, **groups: Iterable[torch.Tensor]):
        captured = {}
        for name, parameters in groups.items():
            values = tuple(parameter.detach().clone() for parameter in parameters)
            if not values:
                raise ValueError("parameter group {} must be non-empty".format(name))
            captured[name] = values
        return cls(captured)

    def measure(self, **groups: Iterable[torch.Tensor]):
        if set(groups) != set(self._groups):
            raise ValueError("parameter snapshot group mismatch")
        result = {}
        for name, parameters in groups.items():
            current = tuple(parameters)
            previous = self._groups[name]
            if len(current) != len(previous):
                raise ValueError("parameter count changed for {}".format(name))
            squared = 0.0
            max_abs = 0.0
            nonzero = 0
            elements = 0
            for before, after in zip(previous, current):
                if before.shape != after.shape:
                    raise ValueError("parameter shape changed for {}".format(name))
                delta = after.detach().float() - before.float()
                if not torch.isfinite(delta).all().item():
                    raise ValueError("parameter delta is non-finite for {}".format(name))
                squared += float(delta.pow(2).sum().item())
                max_abs = max(max_abs, float(delta.abs().max().item()))
                nonzero += int(torch.count_nonzero(delta).item())
                elements += delta.numel()
            result[name] = {
                "parameter_count": len(current),
                "element_count": elements,
                "l2_norm": float(math.sqrt(squared)),
                "max_abs": max_abs,
                "nonzero_elements": nonzero,
            }
        return result


def evaluate_post_update(trainer, batch, *, update_actor=True):
    """Re-evaluate official DPPO metrics without changing parameters or gradients."""

    device = trainer.critic_parameters[0].device
    fork_devices = []
    if device.type == "cuda":
        fork_devices = [
            device.index if device.index is not None else torch.cuda.current_device()
        ]
    with torch.random.fork_rng(devices=fork_devices, enabled=True):
        with torch.no_grad():
            outputs = trainer.runtime.loss(
                batch.condition,
                batch.chains_prev,
                batch.chains_next,
                batch.denoising_inds,
                batch.returns,
                batch.oldvalues,
                batch.advantages.clone(),
                batch.oldlogprobs,
                use_bc_loss=False,
                reward_horizon=trainer.config.reward_horizon,
                reward_start=trainer.config.reward_start,
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
            + trainer.config.value_loss_coefficient * value_loss
            + trainer.config.entropy_loss_coefficient * entropy_loss
        )
    else:
        total_loss = trainer.config.value_loss_coefficient * value_loss
    return {
        "policy_loss": _finite_float(policy_loss, "post-update policy_loss"),
        "value_loss": _finite_float(value_loss, "post-update value_loss"),
        "entropy_loss": _finite_float(entropy_loss, "post-update entropy_loss"),
        "total_loss": _finite_float(total_loss, "post-update total_loss"),
        "approx_kl": _finite_float(approx_kl, "post-update approx_kl"),
        "ratio_mean": _finite_float(ratio_mean, "post-update ratio_mean"),
        "clip_fraction": _finite_float(
            clip_fraction, "post-update clip_fraction"
        ),
    }


def evaluate_post_update_logical(trainer, batch, *, microbatch_pairs, update_actor=True):
    """Globally reduce a logical post-update batch without a 128-pair shortcut."""

    from types import SimpleNamespace
    from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.official_loss_components import (
        AdvantageStats, compute_official_components,
    )
    count = int(batch.chains_prev.shape[0])
    stats = AdvantageStats.from_tensor(
        batch.advantages,
        float(getattr(trainer.runtime, "clip_advantage_lower_quantile", 0.0)),
        float(getattr(trainer.runtime, "clip_advantage_upper_quantile", 1.0)),
    )
    collected = []
    with torch.no_grad():
        for start in range(0, count, int(microbatch_pairs)):
            end = min(start + int(microbatch_pairs), count)
            micro = SimpleNamespace(
                condition={key: value[start:end] for key, value in batch.condition.items()},
                chains_prev=batch.chains_prev[start:end], chains_next=batch.chains_next[start:end],
                denoising_inds=batch.denoising_inds[start:end], returns=batch.returns[start:end],
                oldvalues=batch.oldvalues[start:end], advantages=batch.advantages[start:end],
                oldlogprobs=batch.oldlogprobs[start:end],
            )
            collected.append(compute_official_components(
                trainer.runtime, micro, stats,
                reward_start=trainer.config.reward_start,
                reward_horizon=trainer.config.reward_horizon,
            ))
    policy = torch.cat([part.policy for part in collected]).mean()
    value = 0.5 * torch.cat([part.value for part in collected]).mean()
    entropy = torch.cat([part.entropy for part in collected]).mean()
    approx_kl = torch.cat([part.approx_kl for part in collected]).mean()
    ratio = torch.cat([part.ratio for part in collected]).mean()
    clip_fraction = torch.cat([part.clip_mask for part in collected]).float().mean()
    total = (
        policy + trainer.config.value_loss_coefficient * value
        + trainer.config.entropy_loss_coefficient * entropy
        if update_actor else trainer.config.value_loss_coefficient * value
    )
    return {
        "policy_loss": _finite_float(policy, "logical post-update policy_loss"),
        "value_loss": _finite_float(value, "logical post-update value_loss"),
        "entropy_loss": _finite_float(entropy, "logical post-update entropy_loss"),
        "total_loss": _finite_float(total, "logical post-update total_loss"),
        "approx_kl": _finite_float(approx_kl, "logical post-update approx_kl"),
        "ratio_mean": _finite_float(ratio, "logical post-update ratio_mean"),
        "clip_fraction": _finite_float(clip_fraction, "logical post-update clip_fraction"),
    }


def run_audited_update(
    trainer,
    batch,
    *,
    update_actor=True,
    evaluate_after=True,
    capture_parameter_delta=True,
    logical_microbatch_pairs=None,
):
    """Run one existing trainer step and collect independent before/after evidence."""

    snapshot = None
    if capture_parameter_delta:
        snapshot = ParameterSnapshot.capture(
            actor=trainer.actor_parameters,
            critic=trainer.critic_parameters,
        )
    if logical_microbatch_pairs is None:
        metrics = trainer.update(batch, update_actor=update_actor)
    else:
        metrics = trainer.update_logical_batch(
            batch,
            microbatch_pairs=logical_microbatch_pairs,
            update_actor=update_actor,
        )
    pre_update = asdict(metrics)
    parameter_delta = (
        snapshot.measure(
            actor=trainer.actor_parameters,
            critic=trainer.critic_parameters,
        )
        if snapshot is not None
        else None
    )
    post_update = (
        evaluate_post_update(trainer, batch, update_actor=update_actor)
        if evaluate_after
        else None
    )
    return {
        "pre_update": pre_update,
        "post_update": post_update,
        "parameter_delta": parameter_delta,
    }
