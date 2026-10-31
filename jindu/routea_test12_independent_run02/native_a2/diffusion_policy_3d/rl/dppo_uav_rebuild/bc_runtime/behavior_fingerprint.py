"""Deterministic fixed-input fingerprints for native E5 behavior."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Dict, Mapping, Tuple

import torch

from diffusion_policy_3d.rl.dppo_uav_rebuild.bc_runtime.inference import (
    ObservationBridge,
    run_native_inference,
)
from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.contracts import (
    ActionSpec,
    ObservationSpec,
)


class BehaviorFingerprintError(RuntimeError):
    """Raised when repeated native inference is not bitwise deterministic."""


@dataclass(frozen=True)
class TensorRecord:
    name: str
    shape: Tuple[int, ...]
    dtype: str
    sha256: str
    minimum: float
    maximum: float
    mean: float


@dataclass(frozen=True)
class BehaviorFingerprint:
    bc_identity_hash: str
    seed: int
    timesteps: Tuple[int, ...]
    observation_records: Tuple[TensorRecord, ...]
    action_records: Tuple[TensorRecord, ...]

    @property
    def action_shapes(self) -> Dict[str, Tuple[int, ...]]:
        return {record.name: record.shape for record in self.action_records}

    @property
    def action_sha256(self) -> Dict[str, str]:
        return {record.name: record.sha256 for record in self.action_records}

    @property
    def observation_sha256(self) -> Dict[str, str]:
        return {record.name: record.sha256 for record in self.observation_records}

    @property
    def content_hash(self) -> str:
        encoded = json.dumps(
            asdict(self),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _tensor_record(name: str, tensor: torch.Tensor) -> TensorRecord:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(value.numpy().tobytes())
    numeric = value.float()
    return TensorRecord(
        name=name,
        shape=tuple(int(dim) for dim in value.shape),
        dtype=str(value.dtype),
        sha256=digest.hexdigest(),
        minimum=float(numeric.min().item()),
        maximum=float(numeric.max().item()),
        mean=float(numeric.mean().item()),
    )


def _action_tensors(result: Any) -> Dict[str, torch.Tensor]:
    return {
        "full": result.actions.full,
        "policy": result.actions.policy,
        "deployment_executed": result.actions.deployment_executed,
        "rollout_executed": result.actions.rollout_executed,
    }


def fingerprint_runtime(
    runtime: Any,
    environment_observation: Mapping[str, Any],
    observation_spec: ObservationSpec,
    action_spec: ActionSpec,
    seed: int,
) -> BehaviorFingerprint:
    """Run the same seeded native behavior twice and record exact tensor hashes."""
    first = run_native_inference(
        runtime=runtime,
        environment_observation=environment_observation,
        observation_spec=observation_spec,
        action_spec=action_spec,
        seed=seed,
    )
    second = run_native_inference(
        runtime=runtime,
        environment_observation=environment_observation,
        observation_spec=observation_spec,
        action_spec=action_spec,
        seed=seed,
    )
    if first.timesteps != second.timesteps:
        raise BehaviorFingerprintError("same-seed timesteps are not deterministic")

    first_actions = _action_tensors(first)
    second_actions = _action_tensors(second)
    for name in first_actions:
        left = first_actions[name].detach().cpu()
        right = second_actions[name].detach().cpu()
        if not torch.equal(left, right):
            max_abs_diff = float((left.float() - right.float()).abs().max().item())
            raise BehaviorFingerprintError(
                "same-seed {} is not deterministic; max_abs_diff={}".format(
                    name, max_abs_diff
                )
            )

    policy_observation = ObservationBridge(observation_spec).to_policy_observation(
        environment_observation, device=torch.device(runtime.device)
    )
    observation_records = tuple(
        _tensor_record(name, policy_observation[name])
        for name in sorted(policy_observation)
    )
    action_records = tuple(
        _tensor_record(name, first_actions[name])
        for name in (
            "full",
            "policy",
            "deployment_executed",
            "rollout_executed",
        )
    )
    bc_identity_hash = getattr(getattr(runtime, "identity", None), "content_hash", None)
    if not isinstance(bc_identity_hash, str) or len(bc_identity_hash) != 64:
        raise BehaviorFingerprintError("runtime has no valid BCIdentity hash")

    return BehaviorFingerprint(
        bc_identity_hash=bc_identity_hash,
        seed=seed,
        timesteps=first.timesteps,
        observation_records=observation_records,
        action_records=action_records,
    )
