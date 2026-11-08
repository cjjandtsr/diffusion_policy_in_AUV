"""Explicit transition x denoising-step schedules for official DPPO."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import hashlib
from typing import Iterator, Literal

import numpy as np
import torch

from .rollout_buffer import LogicalPairBatch, OnPolicyRolloutBuffer


PairScheduleMode = Literal[
    "sampled_k", "full_joint_flat", "full_joint_accum_batch1280", "full_joint_accum"
]


@dataclass(frozen=True)
class PairIndex:
    transition_index: int
    denoising_index: int


_MODES = {
    "sampled_k": (128, False),
    "full_joint_flat": (128, True),
    "full_joint_accum_batch1280": (1280, True),
    "full_joint_accum": (None, True),
}


def _permutation_hash(indices: torch.Tensor) -> str:
    array = np.ascontiguousarray(indices.cpu().numpy())
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _validate_buffer_identity(buffer: OnPolicyRolloutBuffer):
    identities = [item["transition_id"] for item in buffer._items]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate transition identity")
    if buffer.outer_update_id is not None:
        prefix = "u{}:".format(buffer.outer_update_id)
        if any(not identity.startswith(prefix) for identity in identities):
            raise ValueError("cross outer update transition identity")


def iter_pair_passes(
    buffer: OnPolicyRolloutBuffer,
    *,
    mode: PairScheduleMode,
    pair_passes: int,
    logical_batch_pairs: int,
    microbatch_pairs: int,
    generator: torch.Generator,
    device,
) -> Iterator[LogicalPairBatch]:
    """Yield immutable logical views after verifying exact per-pass coverage."""

    if mode not in _MODES:
        raise ValueError("unknown pair schedule mode: {}".format(mode))
    expected_logical, full_joint = _MODES[mode]
    if mode == "full_joint_accum":
        if logical_batch_pairs not in (1280, 2560):
            raise ValueError("full_joint_accum requires logical_batch_pairs=1280 or 2560")
        expected_logical = int(logical_batch_pairs)
    if int(logical_batch_pairs) != expected_logical:
        raise ValueError(
            "{} requires logical_batch_pairs={}".format(mode, expected_logical)
        )
    if isinstance(pair_passes, bool) or int(pair_passes) < 1:
        raise ValueError("pair_passes must be positive")
    if isinstance(microbatch_pairs, bool) or not 1 <= int(microbatch_pairs) <= expected_logical:
        raise ValueError("microbatch_pairs must fit a logical batch")
    if buffer._advantages is None:
        raise RuntimeError("finalize buffer before building a pair schedule")
    _validate_buffer_identity(buffer)
    transition_count = len(buffer)
    denoising_count = 10
    for pass_id in range(int(pair_passes)):
        if full_joint:
            flat = torch.randperm(
                transition_count * denoising_count, generator=generator
            )
            transition_indices = torch.div(flat, denoising_count, rounding_mode="floor")
            denoising_indices = flat.remainder(denoising_count)
            expected_pairs = transition_count * denoising_count
            unique_pairs = len(set(zip(
                transition_indices.tolist(), denoising_indices.tolist()
            )))
            if unique_pairs != expected_pairs:
                raise RuntimeError("full-joint schedule has duplicate or missing pairs")
            sampling_probability = 1.0
            permutation_source = flat
        else:
            transition_indices = torch.randperm(
                transition_count, generator=generator
            )
            sampled_k = torch.randint(
                0, denoising_count, (transition_count,), generator=generator
            )
            denoising_indices = sampled_k[transition_indices]
            expected_pairs = transition_count
            if sorted(transition_indices.tolist()) != list(range(transition_count)):
                raise RuntimeError("sampled-k schedule lost a transition")
            unique_pairs = len(set(zip(
                transition_indices.tolist(), denoising_indices.tolist()
            )))
            sampling_probability = 1.0 / denoising_count
            permutation_source = torch.stack(
                (transition_indices, denoising_indices), dim=1
            )
        per_k = Counter(int(value) for value in denoising_indices.tolist())
        audit = {
            "mode": mode,
            "pass_id": pass_id,
            "pair_count": expected_pairs,
            "unique_pair_count": unique_pairs,
            "missing_pair_count": (
                transition_count * denoising_count - unique_pairs
                if full_joint else 0
            ),
            "duplicate_pair_count": expected_pairs - unique_pairs,
            "per_k_count": {str(k): per_k.get(k, 0) for k in range(denoising_count)},
            "sampling_probability": sampling_probability,
            "permutation_sha256": _permutation_hash(permutation_source),
        }
        for start in range(0, expected_pairs, expected_logical):
            end = min(start + expected_logical, expected_pairs)
            yield buffer.materialize_pairs(
                transition_indices[start:end],
                denoising_indices[start:end],
                device=device,
                pass_id=pass_id,
                mode=mode,
                microbatch_pairs=int(microbatch_pairs),
                schedule_audit=audit,
            )


__all__ = (
    "LogicalPairBatch",
    "PairIndex",
    "PairScheduleMode",
    "iter_pair_passes",
)
