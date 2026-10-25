"""Atomic checkpoint directory publication without overwrite."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import uuid

import numpy as np
import torch

from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.io_safety import (
    atomic_publish_directory,
)


COMMITTED_MANIFEST_NAME = "COMMITTED.json"


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(
    path,
    *,
    header,
    lora_bundle,
    critic,
    actor_optimizer,
    critic_optimizer,
    counters,
    schedules=None,
    rollout_buffer=None,
    task_sampler=None,
    training_config=None,
):
    destination = Path(path).resolve()
    if destination.exists():
        raise FileExistsError("checkpoint destination already exists: {}".format(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / ("." + destination.name + ".tmp-" + uuid.uuid4().hex)
    temporary.mkdir()
    try:
        header_path = temporary / "header.json"
        header_path.write_text(header.to_json(), encoding="utf-8")
        state = {
            "lora": {key: value.cpu() for key, value in lora_bundle.adapter_state_dict().items()},
            "critic": critic.state_dict(),
            "actor_optimizer": actor_optimizer.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
            "counters": vars(counters).copy(),
            "schedules": {
                key: value.state_dict() for key, value in (schedules or {}).items()
            },
            "rollout_buffer": rollout_buffer.state_dict() if rollout_buffer else None,
            "task_sampler": task_sampler.state_dict() if task_sampler else None,
            "rng_python": random.getstate(),
            "rng_numpy_legacy": {
                "bit_generator": np.random.get_state()[0],
                "state": np.random.get_state()[1].tolist(),
                "position": int(np.random.get_state()[2]),
                "has_gauss": int(np.random.get_state()[3]),
                "cached_gaussian": float(np.random.get_state()[4]),
            },
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        state_path = temporary / "state.pt"
        if training_config is not None:
            state["training_config"] = training_config
        torch.save(state, state_path)
        integrity = {
            "header.json": _sha256(header_path),
            "state.pt": _sha256(state_path),
        }
        (temporary / "SHA256.json").write_text(
            json.dumps(integrity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for file_path in temporary.iterdir():
            with file_path.open("r+b") as stream:
                stream.flush()
                os.fsync(stream.fileno())
        atomic_publish_directory(
            temporary,
            destination,
            {
                "schema_version": 2,
                "header_schema_hash": header.schema_hash,
                "files": integrity,
                "status": "committed",
            },
        )
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return destination
