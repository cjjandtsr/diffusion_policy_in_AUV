"""On-policy storage for complete official four-step ReinFlow chains."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, Mapping, Sequence, Tuple

import torch


EVAL_ONLY_WORLDS = frozenset({"w4", "w8", "w11", "w12", "w18"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CONDITION_KEYS = (
    "state",
    "global_cond",
    "global_cond_uncond",
    "noise_cond",
)


class RolloutContractError(RuntimeError):
    """Raised when rollout evidence is unsafe for an on-policy update."""


@dataclass(frozen=True)
class FlowRolloutBatch:
    condition: Dict[str, torch.Tensor]
    chain: torch.Tensor
    old_logprob: torch.Tensor
    reward: torch.Tensor
    value: torch.Tensor
    next_value: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    world_ids: Tuple[str, ...]
    task_ids: Tuple[str, ...]
    reward_identity: str
    action_identity: str
    condition_identities: Tuple[str, ...]
    checkpoint_identity: str


def _require_sha256(value: str, field: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RolloutContractError(f"{field} must be a lowercase SHA256")


def _require_finite(name: str, value: torch.Tensor) -> None:
    if value.is_floating_point() and not torch.isfinite(value).all():
        raise RolloutContractError(f"{name} contains non-finite values")


class FlowRolloutBuffer:
    """Accumulate immutable chain batches while preserving provenance."""

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.clear()

    def clear(self) -> None:
        self._batches: list[FlowRolloutBatch] = []
        self._size = 0

    def add_batch(
        self,
        *,
        condition: Mapping[str, torch.Tensor],
        chain: torch.Tensor,
        old_logprob: torch.Tensor,
        reward: torch.Tensor,
        value: torch.Tensor,
        next_value: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        world_ids: Sequence[str],
        task_ids: Sequence[str],
        reward_identity: str,
        action_identity: str,
        condition_identities: Sequence[str],
        checkpoint_identity: str,
    ) -> None:
        if chain.ndim != 4 or tuple(chain.shape[1:]) != (5, 16, 11):
            raise RolloutContractError("chain must have shape (B,5,16,11)")
        batch_size = int(chain.shape[0])
        if batch_size <= 0 or self._size + batch_size > self.capacity:
            raise RolloutContractError("rollout buffer capacity exceeded")
        vectors = {
            "old_logprob": old_logprob,
            "reward": reward,
            "value": value,
            "next_value": next_value,
            "terminated": terminated,
            "truncated": truncated,
        }
        for name, tensor in vectors.items():
            if not torch.is_tensor(tensor) or tuple(tensor.shape) != (batch_size,):
                raise RolloutContractError(f"{name} must have shape (B,)")
            _require_finite(name, tensor)
        _require_finite("chain", chain)

        if set(condition) != set(_CONDITION_KEYS):
            raise RolloutContractError(
                f"condition keys must be exactly {_CONDITION_KEYS}"
            )
        for key in _CONDITION_KEYS:
            tensor = condition[key]
            if not torch.is_tensor(tensor) or tensor.ndim != 2 or tensor.shape[0] != batch_size:
                raise RolloutContractError(
                    f"condition {key} must have shape (B,F)"
                )
            _require_finite(f"condition.{key}", tensor)

        world_ids = tuple(str(value) for value in world_ids)
        task_ids = tuple(str(value) for value in task_ids)
        condition_identities = tuple(condition_identities)
        if len(world_ids) != batch_size or len(task_ids) != batch_size:
            raise RolloutContractError("world_ids/task_ids length must equal B")
        blocked = sorted(
            {world.lower() for world in world_ids if world.lower() in EVAL_ONLY_WORLDS}
        )
        if blocked:
            raise RolloutContractError(
                f"eval-only world(s) cannot enter training buffer: {blocked}"
            )
        if any(not task.strip() for task in task_ids):
            raise RolloutContractError("task_ids must be non-empty")
        _require_sha256(reward_identity, "reward_identity")
        _require_sha256(action_identity, "action_identity")
        _require_sha256(checkpoint_identity, "checkpoint_identity")
        if len(condition_identities) != batch_size:
            raise RolloutContractError("condition_identities length must equal B")
        for condition_identity in condition_identities:
            _require_sha256(condition_identity, "condition_identities")
        for previous in self._batches:
            if previous.reward_identity != reward_identity:
                raise RolloutContractError("reward_identity changed within buffer")
            if previous.action_identity != action_identity:
                raise RolloutContractError("action_identity changed within buffer")
            if previous.checkpoint_identity != checkpoint_identity:
                raise RolloutContractError("checkpoint_identity changed within buffer")

        def cpu_clone(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.detach().cpu().clone()

        self._batches.append(
            FlowRolloutBatch(
                condition={key: cpu_clone(condition[key]) for key in _CONDITION_KEYS},
                chain=cpu_clone(chain),
                old_logprob=cpu_clone(old_logprob),
                reward=cpu_clone(reward),
                value=cpu_clone(value),
                next_value=cpu_clone(next_value),
                terminated=cpu_clone(terminated),
                truncated=cpu_clone(truncated),
                world_ids=world_ids,
                task_ids=task_ids,
                reward_identity=reward_identity,
                action_identity=action_identity,
                condition_identities=condition_identities,
                checkpoint_identity=checkpoint_identity,
            )
        )
        self._size += batch_size

    def as_batch(self, *, device) -> FlowRolloutBatch:
        if self._size != self.capacity:
            raise RolloutContractError(
                f"formal update requires full buffer {self.capacity}, got {self._size}"
            )
        first = self._batches[0]
        concatenate = lambda field: torch.cat(
            [getattr(batch, field) for batch in self._batches], dim=0
        ).to(device)
        return FlowRolloutBatch(
            condition={
                key: torch.cat([batch.condition[key] for batch in self._batches], dim=0).to(device)
                for key in _CONDITION_KEYS
            },
            chain=concatenate("chain"),
            old_logprob=concatenate("old_logprob"),
            reward=concatenate("reward"),
            value=concatenate("value"),
            next_value=concatenate("next_value"),
            terminated=concatenate("terminated"),
            truncated=concatenate("truncated"),
            world_ids=tuple(value for batch in self._batches for value in batch.world_ids),
            task_ids=tuple(value for batch in self._batches for value in batch.task_ids),
            reward_identity=first.reward_identity,
            action_identity=first.action_identity,
            condition_identities=tuple(
                value for batch in self._batches for value in batch.condition_identities
            ),
            checkpoint_identity=first.checkpoint_identity,
        )

    def __len__(self) -> int:
        return self._size


__all__ = [
    "EVAL_ONLY_WORLDS",
    "FlowRolloutBatch",
    "FlowRolloutBuffer",
    "RolloutContractError",
]
