"""R6-E §11.6.2 (F-08 → R6-OPT-1): the optimizer single-param-group contract.

The trainer historically indexed ``param_groups[0]`` directly — a second
group would train group 0 only and be silently ignored. R6/G8 require
actor and critic to each hold EXACTLY one param group; every consumer
(creation, resume, ppo_update) receives the parameters from
``require_single_param_group``, which hard-fails BEFORE the first
rollout/update if the invariant is violated.

The frozen resolved config (type / lr / betas / eps / weight_decay /
param_group_count / ordered_parameter_name_hash) is captured at
optimizer creation and re-validated per field on resume — a
tensor-shape-only comparison is not enough (hyperparameter drift
would pass a shape check silently).

Python 3.8: no ``|`` union syntax, no dict ``|`` merge.
"""

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import torch

# Order of per-field resume comparison (field name appears in the
# ValueError so a drift is diagnosable without digging into state).
_RESOLVED_FIELDS = (
    "type", "lr", "betas", "eps", "weight_decay",
    "param_group_count", "ordered_parameter_name_hash",
)


@dataclass(frozen=True)
class ResolvedOptimizerConfig:
    """Frozen, JSON-safe resolved optimizer config (§11.6.2).

    ``betas`` is serialized as a list (JSON) and normalized back to a
    tuple by the consumers — equality is value-based, not type-based.
    ``betas``/``eps`` are None for optimizers whose param group does
    not resolve them (e.g. SGD has no betas/eps keys) — the frozen
    config records what the optimizer ACTUALLY resolved, and a None-vs-
    tuple mismatch on resume is a real drift, never tolerated.
    """

    type: str
    lr: float
    weight_decay: float
    param_group_count: int
    ordered_parameter_name_hash: str
    betas: Optional[Any] = None  # tuple (0.9, 0.999) or its JSON list
    eps: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict for the resolved-config layer of a checkpoint."""
        return {
            "type": self.type,
            "lr": self.lr,
            "betas": None if self.betas is None else list(self.betas),
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "param_group_count": self.param_group_count,
            "ordered_parameter_name_hash": self.ordered_parameter_name_hash,
        }


def _param_metadata(param: torch.nn.Parameter) -> str:
    """Cross-process stable metadata of ONE parameter.

    id() is not stable across process restarts (resume is a cold start
    in a NEW process), so the hash keys on shape/dtype/requires_grad —
    parameters with the identical triple are interchangeable under this
    view. The production optimizer mixes LoRA/value-head shapes, so a
    real order drift IS detectable.
    """
    return "{}{}{}".format(
        tuple(param.shape), param.dtype, int(param.requires_grad))


def ordered_parameter_name_hash(
    params: Sequence[torch.nn.Parameter],
) -> str:
    """sha256 over the ORDER-BOUND parameter metadata sequence.

    Order-sensitive (a permutation changes the hash), deterministic
    across process restarts (no id()). Same-shaped parameters are
    indistinguishable under a pure-metadata view — that is by design,
    see _param_metadata.
    """
    h = hashlib.sha256()
    for p in params:
        h.update(_param_metadata(p).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def require_single_param_group(
    name: str,
    optimizer: torch.optim.Optimizer,
) -> Sequence[torch.nn.Parameter]:
    """Return the parameters of the single param group, or hard-fail.

    §11.6.2: called at creation, after resume and inside ppo_update; a
    second group fails BEFORE any rollout/update — never silently
    ignored by ``param_groups[0]`` indexing.
    """
    groups = optimizer.param_groups
    if len(groups) != 1:
        raise ValueError(
            f"optimizer '{name}' must hold exactly one single param group "
            f"for R6/G8, got {len(groups)}"
        )
    params = list(groups[0]["params"])
    if not params:
        raise ValueError(
            f"optimizer '{name}' single param group has no parameters"
        )
    return params


def resolved_optimizer_config(
    name: str,
    optimizer: torch.optim.Optimizer,
) -> ResolvedOptimizerConfig:
    """Freeze the RESOLVED per-group config of ``optimizer`` (§11.6.2)."""
    params = require_single_param_group(name, optimizer)
    group = optimizer.param_groups[0]
    return ResolvedOptimizerConfig(
        type=optimizer.__class__.__name__,
        lr=float(group["lr"]),
        # SGD (and other non-adaptive optimizers) resolve NO betas/eps
        # keys — record None, never crash on group["betas"].
        betas=tuple(group["betas"]) if "betas" in group else None,
        eps=float(group["eps"]) if "eps" in group else None,
        weight_decay=float(group["weight_decay"]),
        param_group_count=len(optimizer.param_groups),
        ordered_parameter_name_hash=ordered_parameter_name_hash(params),
    )


def assert_optimizer_parameter_partition(
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    *,
    lora_params: Sequence[torch.nn.Parameter],
    base_model: torch.nn.Module,
) -> None:
    """Assert the actor/critic/base parameter partition (guide R7-3 Step 2).

    Required partition: the actor optimizer holds exactly the LoRA
    bundle parameters; the critic optimizer holds only value-head
    parameters; actor and critic parameter ID sets are disjoint; and no
    base (UNet/encoder, LoRA excluded) parameter enters either
    optimizer. Raises ValueError on any violation — fail closed, before
    the first rollout/update.
    """
    actor_params = require_single_param_group("actor", actor_optimizer)
    critic_params = require_single_param_group("value", critic_optimizer)
    actor_ids = {id(p) for p in actor_params}
    critic_ids = {id(p) for p in critic_params}
    lora_ids = {id(p) for p in lora_params}
    if not lora_ids:
        raise ValueError(
            "assert_optimizer_parameter_partition: empty LoRA parameter "
            "set — the actor must own the registered LoRA bundle"
        )
    overlap = actor_ids & critic_ids
    if overlap:
        raise ValueError(
            f"optimizer parameter overlap: {len(overlap)} parameter(s) in "
            "both the actor and critic optimizer — actor and critic "
            "parameter ID sets must be disjoint (R7-3 Step 2)"
        )
    missing = lora_ids - actor_ids
    if missing:
        raise ValueError(
            f"optimizer parameter partition: {len(missing)} LoRA "
            "parameter(s) not owned by the actor optimizer — the actor "
            "must own exactly the LoRA bundle trainable parameters "
            "(R7-3 Step 2)"
        )
    base_ids = {id(p) for p in base_model.parameters()} - lora_ids
    leaked = (base_ids & actor_ids) | (base_ids & critic_ids)
    if leaked:
        raise ValueError(
            f"optimizer parameter partition: {len(leaked)} base "
            "UNet/encoder parameter(s) entered an optimizer — base "
            "parameters must stay out of both optimizers (R7-3 Step 2)"
        )


def validate_optimizer_resume_config(
    name: str,
    optimizer: torch.optim.Optimizer,
    frozen: ResolvedOptimizerConfig,
) -> None:
    """Per-field comparison of the resumed optimizer vs the frozen
    config (§11.6.2). Shape-only checks are explicitly not enough.

    Raises:
        ValueError: on any drift — the message names the drifted field.
    """
    params = require_single_param_group(name, optimizer)
    group = optimizer.param_groups[0]
    actual = {
        "type": optimizer.__class__.__name__,
        "lr": float(group["lr"]),
        "betas": tuple(group["betas"]) if "betas" in group else None,
        "eps": float(group["eps"]) if "eps" in group else None,
        "weight_decay": float(group["weight_decay"]),
        "param_group_count": len(optimizer.param_groups),
        "ordered_parameter_name_hash": ordered_parameter_name_hash(params),
    }
    for field in _RESOLVED_FIELDS:
        expected = getattr(frozen, field)
        got = actual[field]
        if isinstance(expected, (list, tuple)):
            expected = tuple(expected)
        if expected != got:
            raise ValueError(
                f"resume optimizer '{name}' field {field!r} drifted: "
                f"expected {expected!r}, got {got!r}"
            )
