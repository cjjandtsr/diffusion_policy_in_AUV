"""On-policy storage for official DPPO denoising-chain updates."""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from .gae import compute_gae


@dataclass(frozen=True)
class RolloutBatch:
    condition: Dict[str, torch.Tensor]
    chains_prev: torch.Tensor
    chains_next: torch.Tensor
    denoising_inds: torch.Tensor
    returns: torch.Tensor
    oldvalues: torch.Tensor
    advantages: torch.Tensor
    oldlogprobs: torch.Tensor


@dataclass(frozen=True)
class LogicalPairBatch(RolloutBatch):
    transition_indices: torch.Tensor
    transition_ids: tuple
    roles: tuple
    outcomes: tuple
    provenance_hashes: tuple
    pass_id: int
    mode: str
    microbatch_pairs: int
    schedule_audit: dict

    @property
    def pair_count(self):
        return int(self.denoising_inds.numel())

    @property
    def microbatch_count(self):
        return (self.pair_count + self.microbatch_pairs - 1) // self.microbatch_pairs


class OnPolicyRolloutBuffer:
    def __init__(self, capacity: int, *, gamma: float = 0.99, gae_lambda: float = 0.95,
                 outer_update_id: Optional[int] = None, history_scalar: bool = False,
                 episode_context: bool = False):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.outer_update_id = outer_update_id
        self.history_scalar = bool(history_scalar)
        self.episode_context = bool(episode_context)
        if self.history_scalar and self.episode_context:
            raise ValueError("choose one critic context")
        self.clear()

    def clear(self):
        self._items: List[dict] = []
        self._advantages = None
        self._returns = None

    def add(
        self,
        *,
        condition,
        chain,
        old_logprobs,
        reward,
        value,
        next_value,
        terminated,
        truncated,
        transition_id=None,
        role="unknown",
        outcome="none",
        provenance_hash="",
        bootstrap_allowed=None,
    ):
        if len(self._items) >= self.capacity:
            raise RuntimeError("on-policy buffer capacity exceeded")
        if tuple(chain.shape) != (11, 16, 11):
            raise ValueError("chain must have shape (11,16,11)")
        if tuple(old_logprobs.shape) != (10, 16, 11):
            raise ValueError("old_logprobs must have shape (10,16,11)")
        expected_keys = {"state", "global_cond", "beta"} if self.history_scalar else {"state", "global_cond"}
        if self.episode_context:
            expected_keys.add("episode_context")
        if set(condition) != expected_keys:
            raise ValueError("condition keys do not match critic version")
        if any(tuple(condition[key].shape) != (576,) for key in ("state", "global_cond")):
            raise ValueError("actor condition must have shape (576,)")
        if self.history_scalar and (tuple(condition["beta"].shape) != (1,) or
                                    not bool(torch.isfinite(condition["beta"]).all())):
            raise ValueError("R1 beta must be finite shape (1,)")
        if self.episode_context and (tuple(condition["episode_context"].shape) != (6,) or
                                    not bool(torch.isfinite(condition["episode_context"]).all())):
            raise ValueError("episode_context must be finite shape (6,)")
        if bootstrap_allowed is None:
            bootstrap_allowed = not terminated
        if terminated and bootstrap_allowed:
            raise ValueError("terminated transitions cannot bootstrap")
        if transition_id is None:
            prefix = "u{}".format(self.outer_update_id) if self.outer_update_id is not None else "u_unknown"
            transition_id = "{}:t{}".format(prefix, len(self._items))
        if any(item["transition_id"] == str(transition_id) for item in self._items):
            raise ValueError("duplicate transition_id")
        self._items.append(
            {
                "condition": {key: item.detach().cpu() for key, item in condition.items()},
                "chain": chain.detach().cpu(),
                "old_logprobs": old_logprobs.detach().cpu(),
                "reward": float(reward),
                "value": float(value),
                "next_value": float(next_value),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "bootstrap_allowed": bool(bootstrap_allowed),
                "transition_id": str(transition_id),
                "role": str(role),
                "outcome": str(outcome),
                "provenance_hash": str(provenance_hash),
            }
        )
        self._advantages = self._returns = None

    def finalize(self):
        if len(self._items) != self.capacity:
            raise RuntimeError("formal update requires a full on-policy buffer")
        rewards = torch.tensor([item["reward"] for item in self._items])
        values = torch.tensor([item["value"] for item in self._items])
        next_values = torch.tensor([item["next_value"] for item in self._items])
        terminated = torch.tensor([item["terminated"] for item in self._items])
        truncated = torch.tensor([item["truncated"] for item in self._items])
        self._advantages, self._returns = compute_gae(
            rewards,
            values,
            next_values,
            terminated,
            truncated,
            self.gamma,
            self.gae_lambda,
            bootstrap_allowed=torch.tensor([
                item.get("bootstrap_allowed", not item["terminated"]) for item in self._items]),
        )

    def batches(self, batch_size: int, *, generator: torch.Generator, device):
        if self._advantages is None:
            raise RuntimeError("finalize buffer before sampling")
        if batch_size != 128:
            raise ValueError("formal DPPO minibatch size is fixed at 128")
        if len(self._items) % batch_size:
            raise ValueError("buffer size must be divisible by 128")
        order = torch.randperm(len(self._items), generator=generator)
        denoising = torch.randint(
            0, 10, (len(self._items),), generator=generator
        )
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            items = [self._items[int(index)] for index in indices]
            chains = torch.stack([item["chain"] for item in items])
            denoising_inds = denoising[indices]
            rows = torch.arange(batch_size)
            yield RolloutBatch(
                condition={
                    key: torch.stack([item["condition"][key] for item in items]).to(device)
                    for key in items[0]["condition"]
                },
                chains_prev=chains[rows, denoising_inds].to(device),
                chains_next=chains[rows, denoising_inds + 1].to(device),
                denoising_inds=denoising_inds.to(device),
                returns=self._returns[indices].to(device),
                oldvalues=torch.tensor([item["value"] for item in items]).to(device),
                advantages=self._advantages[indices].to(device),
                oldlogprobs=torch.stack(
                    [item["old_logprobs"] for item in items]
                )[rows, denoising_inds].to(device),
            )

    def materialize_pairs(
        self,
        transition_indices,
        denoising_indices,
        *,
        device,
        pass_id,
        mode,
        microbatch_pairs,
        schedule_audit,
    ):
        if self._advantages is None:
            raise RuntimeError("finalize buffer before materializing pairs")
        b = torch.as_tensor(transition_indices, dtype=torch.long, device="cpu")
        k = torch.as_tensor(denoising_indices, dtype=torch.long, device="cpu")
        if b.ndim != 1 or k.ndim != 1 or len(b) != len(k) or len(b) == 0:
            raise ValueError("pair indices must be equal non-empty vectors")
        if bool(torch.any(b < 0)) or bool(torch.any(b >= len(self._items))):
            raise ValueError("transition index out of range")
        if bool(torch.any(k < 0)) or bool(torch.any(k >= 10)):
            raise ValueError("denoising index out of range")
        items = [self._items[int(index)] for index in b]
        chains = torch.stack([item["chain"] for item in items])
        rows = torch.arange(len(b))
        return LogicalPairBatch(
            condition={
                key: torch.stack([item["condition"][key] for item in items]).to(device)
                for key in items[0]["condition"]
            },
            chains_prev=chains[rows, k].to(device),
            chains_next=chains[rows, k + 1].to(device),
            denoising_inds=k.to(device),
            returns=self._returns[b].to(device),
            oldvalues=torch.tensor([item["value"] for item in items]).to(device),
            advantages=self._advantages[b].to(device),
            oldlogprobs=torch.stack([item["old_logprobs"] for item in items])[rows, k].to(device),
            transition_indices=b.to(device),
            transition_ids=tuple(item["transition_id"] for item in items),
            roles=tuple(item["role"] for item in items),
            outcomes=tuple(item["outcome"] for item in items),
            provenance_hashes=tuple(item["provenance_hash"] for item in items),
            pass_id=int(pass_id),
            mode=str(mode),
            microbatch_pairs=int(microbatch_pairs),
            schedule_audit=dict(schedule_audit),
        )

    def __len__(self):
        return len(self._items)

    def audit_tensors(self):
        if self._advantages is None or self._returns is None:
            raise RuntimeError("finalize buffer before requesting audit tensors")
        return {
            "rewards": torch.tensor([item["reward"] for item in self._items]),
            "values": torch.tensor([item["value"] for item in self._items]),
            "next_values": torch.tensor(
                [item["next_value"] for item in self._items]
            ),
            "advantages": self._advantages.detach().cpu().clone(),
            "returns": self._returns.detach().cpu().clone(),
        }

    def state_dict(self):
        state = {
            "items": self._items,
            "advantages": self._advantages,
            "returns": self._returns,
            "capacity": self.capacity,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "outer_update_id": self.outer_update_id,
        }
        if self.history_scalar:
            state["history_scalar"] = True
        if self.episode_context:
            state["episode_context"] = True
        return state

    def load_state_dict(self, state):
        if (state["capacity"], state["gamma"], state["gae_lambda"], state.get("outer_update_id"),
            state.get("history_scalar", False), state.get("episode_context", False)) != (
            self.capacity,
            self.gamma,
            self.gae_lambda,
            self.outer_update_id,
            self.history_scalar,
            self.episode_context,
        ):
            raise ValueError("rollout buffer configuration mismatch")
        self._items = state["items"]
        self._advantages = state["advantages"]
        self._returns = state["returns"]
