"""Identity-first, integrity-checked checkpoint restore; RNG is restored last."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import numpy as np
import torch

from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.io_safety import (
    load_bounded_json,
    secure_open_under,
)
from .identity import verify_checkpoint_identity
from .schema import CheckpointHeader


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cpu_byte_rng_state(value, name):
    if not isinstance(value, torch.Tensor):
        raise TypeError("{} must be a torch.Tensor".format(name))
    if value.dtype != torch.uint8 or value.ndim != 1:
        raise ValueError("{} must be a one-dimensional uint8 tensor".format(name))
    return value.detach().cpu().contiguous()


def restore_checkpoint(
    path,
    *,
    runtime_identity,
    lora_identity,
    lora_bundle,
    critic,
    actor_optimizer,
    critic_optimizer,
    counters,
    schedules=None,
    rollout_buffer=None,
    task_sampler=None,
    map_location="cpu",
):
    raw_root = Path(path)
    if raw_root.is_symlink():
        raise ValueError("checkpoint root symlink is forbidden")
    root = raw_root.resolve()
    header_payload = load_bounded_json(root, "header.json", max_bytes=1024 * 1024)
    header = CheckpointHeader(**header_payload)
    verify_checkpoint_identity(header, runtime_identity, lora_identity)
    expected_files = {"header.json", "state.pt", "SHA256.json", "COMMITTED.json"}
    if {item.name for item in root.iterdir() if item.is_file()} != expected_files:
        raise ValueError("checkpoint is incomplete or contains unknown files")
    committed = load_bounded_json(root, "COMMITTED.json", max_bytes=1024 * 1024)
    if committed.get("status") != "committed" or committed.get("schema_version") != 2:
        raise ValueError("checkpoint committed manifest is invalid")
    if committed.get("header_schema_hash") != header.schema_hash:
        raise ValueError("checkpoint header schema hash mismatch")
    integrity = load_bounded_json(root, "SHA256.json", max_bytes=1024 * 1024)
    if committed.get("files") != integrity:
        raise ValueError("checkpoint committed file manifest mismatch")
    for name, expected in integrity.items():
        actual = _sha256(root / name)
        if actual != expected:
            raise ValueError("checkpoint integrity mismatch: {}".format(name))

    with secure_open_under(root, "state.pt", max_bytes=64 * 1024 * 1024) as stream:
        state = torch.load(stream, map_location=map_location, weights_only=True)
    lora_bundle.load_adapter_state_dict(state["lora"])
    critic.load_state_dict(state["critic"], strict=True)
    actor_optimizer.load_state_dict(state["actor_optimizer"])
    critic_optimizer.load_state_dict(state["critic_optimizer"])
    for name, value in state["counters"].items():
        setattr(counters, name, value)
    for name, schedule in (schedules or {}).items():
        schedule.load_state_dict(state["schedules"][name])
    if rollout_buffer is not None:
        rollout_buffer.load_state_dict(state["rollout_buffer"])
    if task_sampler is not None:
        task_sampler.load_state_dict(state["task_sampler"])

    # No operation that can consume randomness is allowed after these calls.
    torch.set_rng_state(_cpu_byte_rng_state(state["rng_cpu"], "rng_cpu"))
    if state["rng_cuda"] is not None:
        torch.cuda.set_rng_state_all(
            [
                _cpu_byte_rng_state(value, "rng_cuda[{}]".format(index))
                for index, value in enumerate(state["rng_cuda"])
            ]
        )
    numpy_state = state["rng_numpy_legacy"]
    np.random.set_state((
        numpy_state["bit_generator"],
        np.asarray(numpy_state["state"], dtype=np.uint32),
        int(numpy_state["position"]),
        int(numpy_state["has_gauss"]),
        float(numpy_state["cached_gaussian"]),
    ))
    random.setstate(state["rng_python"])
    return header
