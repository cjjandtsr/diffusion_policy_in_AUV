"""Deterministic training-state and first-ratio audit helpers."""

from __future__ import annotations

import hashlib
import random
import struct
from typing import Mapping, Sequence

import numpy as np
import torch


class AuditError(RuntimeError):
    """Raised when an auditable training invariant is violated."""


def _update_digest(digest, value) -> None:
    if value is None:
        digest.update(b"N")
    elif torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"T")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"A")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    elif isinstance(value, Mapping):
        digest.update(b"D")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _update_digest(digest, key)
            _update_digest(digest, value[key])
    elif isinstance(value, (tuple, list)):
        digest.update(b"Q" if isinstance(value, tuple) else b"L")
        digest.update(str(len(value)).encode("ascii"))
        for item in value:
            _update_digest(digest, item)
    elif isinstance(value, bool):
        digest.update(b"B1" if value else b"B0")
    elif isinstance(value, int):
        digest.update(b"I" + str(value).encode("ascii"))
    elif isinstance(value, float):
        digest.update(b"F" + struct.pack("!d", value))
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(b"S" + str(len(encoded)).encode("ascii") + b":" + encoded)
    else:
        raise TypeError(f"unsupported audit value type: {type(value)!r}")


def training_state_sha256(
    *,
    actor,
    noise,
    critic,
    optimizers: Sequence,
    counters: Mapping,
    include_rng: bool,
) -> str:
    payload = {
        "actor": actor.state_dict(),
        "noise": noise.state_dict(),
        "critic": critic.state_dict(),
        "optimizers": tuple(optimizer.state_dict() for optimizer in optimizers),
        "counters": dict(counters),
    }
    if include_rng:
        payload["rng"] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": tuple(torch.cuda.get_rng_state_all())
            if torch.cuda.is_available()
            else (),
        }
    digest = hashlib.sha256()
    _update_digest(digest, payload)
    return digest.hexdigest()


def assert_first_ratio(
    old_logprob: torch.Tensor,
    new_logprob: torch.Tensor,
    *,
    tolerance: float = 1.0e-6,
) -> float:
    if old_logprob.shape != new_logprob.shape:
        raise AuditError("old/new logprob shapes differ")
    if not torch.isfinite(old_logprob).all() or not torch.isfinite(new_logprob).all():
        raise AuditError("old/new logprob contains NaN or Inf")
    maximum_error = float((torch.exp(new_logprob - old_logprob) - 1.0).abs().max())
    if maximum_error > tolerance:
        raise AuditError(
            f"first PPO ratio error {maximum_error:.9g} exceeds {tolerance:.9g}"
        )
    return maximum_error


__all__ = ["AuditError", "assert_first_ratio", "training_state_sha256"]
