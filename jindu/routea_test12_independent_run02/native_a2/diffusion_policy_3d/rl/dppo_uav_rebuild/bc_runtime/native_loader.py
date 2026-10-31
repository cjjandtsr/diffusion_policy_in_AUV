"""Native E5 construction and strict checkpoint loading.

The implementation preserves the probe-verified construction path:
``hydra.utils.instantiate(cfg.policy)`` followed by ``state_dicts['model']``.
It adds identity checks, zero-tolerance key validation, and device auditing.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Optional

import hydra
from omegaconf import OmegaConf
import torch

from diffusion_policy_3d.rl.dppo_uav_rebuild.bc_runtime.checkpoint_contract import (
    BCIdentity,
    EXPECTED_E5_CHECKPOINT_SHA256,
    EXPECTED_POLICY_TARGET,
    EXPECTED_SCHEDULER_TARGET,
    EXPECTED_WEIGHT_SOURCE,
)


DEFAULT_E5_CHECKPOINT_RELPATH = (
    "data/outputs/2026.08.16/08.53.26_e5_lambda0p5/checkpoints/latest.ckpt"
)
WEIGHT_SOURCE_CAVEAT = (
    "historical Gazebo deployment log is unavailable; two repository code "
    "paths use state_dicts['model']"
)


class NativeE5LoadError(RuntimeError):
    """Raised before a runtime with ambiguous E5 behavior can be returned."""


@dataclass(frozen=True)
class NativeBCRuntime:
    policy: Any
    identity: BCIdentity
    checkpoint_path: Path
    device: torch.device


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_normalizer_state_dict(state_dict: Mapping[str, Any]) -> str:
    """Reproduce the P3 normalizer hash over checkpoint tensor content."""
    prefix = "normalizer.params_dict."
    keys = sorted(key for key in state_dict if key.startswith(prefix))
    if not keys:
        raise NativeE5LoadError("checkpoint model contains no normalizer state")
    digest = hashlib.sha256()
    for key in keys:
        tensor = state_dict[key]
        if not torch.is_tensor(tensor):
            raise NativeE5LoadError("normalizer state is not a tensor: {}".format(key))
        tensor = tensor.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _plain_config(cfg: Any) -> Mapping[str, Any]:
    if OmegaConf.is_config(cfg):
        resolved = OmegaConf.to_container(cfg, resolve=True)
    else:
        resolved = cfg
    if not isinstance(resolved, Mapping):
        raise NativeE5LoadError("checkpoint cfg must resolve to a mapping")
    return resolved


def _qualified_name(value: Any) -> str:
    cls = type(value)
    return "{}.{}".format(cls.__module__, cls.__name__)


def _audit_policy_contract(policy: Any, device: torch.device) -> None:
    if _qualified_name(policy) != EXPECTED_POLICY_TARGET:
        raise NativeE5LoadError(
            "instantiated policy class mismatch: {}".format(_qualified_name(policy))
        )
    scheduler = getattr(policy, "noise_scheduler", None)
    if scheduler is None or _qualified_name(scheduler) != EXPECTED_SCHEDULER_TARGET:
        raise NativeE5LoadError(
            "instantiated scheduler class mismatch: {}".format(
                _qualified_name(scheduler) if scheduler is not None else None
            )
        )
    expected_attributes = {
        "num_inference_steps": 10,
        "n_obs_steps": 3,
        "n_action_steps": 8,
        "horizon": 16,
        "action_dim": 11,
        "obs_feature_dim": 192,
        "obs_as_global_cond": True,
    }
    for name, expected in expected_attributes.items():
        actual = getattr(policy, name, None)
        if actual != expected:
            raise NativeE5LoadError(
                "policy {} mismatch: expected {!r}, got {!r}".format(
                    name, expected, actual
                )
            )
    scheduler_config = scheduler.config
    if scheduler_config.num_train_timesteps != 50:
        raise NativeE5LoadError("scheduler num_train_timesteps mismatch")
    if scheduler_config.prediction_type != "sample":
        raise NativeE5LoadError("scheduler prediction_type mismatch")

    normalizer = getattr(policy, "normalizer", None)
    if normalizer is None:
        raise NativeE5LoadError("policy has no normalizer")
    for name, tensor in normalizer.state_dict().items():
        if torch.is_tensor(tensor) and tensor.device != device:
            raise NativeE5LoadError(
                "normalizer device mismatch for {}: expected {}, got {}".format(
                    name, device, tensor.device
                )
            )


def load_native_e5(
    checkpoint_path: Path,
    device: str,
    checkpoint_relpath: str = DEFAULT_E5_CHECKPOINT_RELPATH,
    expected_identity: Optional[BCIdentity] = None,
) -> NativeBCRuntime:
    """Load the exact E5 behavior policy or fail before returning a runtime."""
    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise NativeE5LoadError("E5 checkpoint does not exist: {}".format(path))

    actual_checkpoint_hash = _sha256_file(path)
    if actual_checkpoint_hash != EXPECTED_E5_CHECKPOINT_SHA256:
        raise NativeE5LoadError(
            "checkpoint SHA256 mismatch: expected {}, got {}".format(
                EXPECTED_E5_CHECKPOINT_SHA256, actual_checkpoint_hash
            )
        )

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise NativeE5LoadError("trusted E5 checkpoint deserialization failed") from exc
    if not isinstance(payload, Mapping):
        raise NativeE5LoadError("checkpoint payload must be a mapping")
    if "cfg" not in payload:
        raise NativeE5LoadError("checkpoint is missing cfg")
    state_dicts = payload.get("state_dicts")
    if not isinstance(state_dicts, Mapping):
        raise NativeE5LoadError("checkpoint is missing state_dicts")
    state_dict = state_dicts.get(EXPECTED_WEIGHT_SOURCE)
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise NativeE5LoadError("checkpoint is missing state_dicts['model']")

    normalizer_sha256 = hash_normalizer_state_dict(state_dict)
    cfg = payload["cfg"]
    resolved_config = _plain_config(cfg)
    identity = BCIdentity.from_resolved_config(
        resolved_config=resolved_config,
        checkpoint_relpath=checkpoint_relpath,
        checkpoint_sha256=actual_checkpoint_hash,
        normalizer_sha256=normalizer_sha256,
        weight_source=EXPECTED_WEIGHT_SOURCE,
        evidence_caveats=(WEIGHT_SOURCE_CAVEAT,),
    )
    if expected_identity is not None and identity.content_hash != expected_identity.content_hash:
        raise NativeE5LoadError(
            "loaded BCIdentity does not match the caller-provided identity"
        )

    policy_config = cfg.policy if OmegaConf.is_config(cfg) else cfg["policy"]
    try:
        policy = hydra.utils.instantiate(policy_config)
    except Exception as exc:
        raise NativeE5LoadError("hydra native E5 policy construction failed") from exc

    try:
        incompatible = policy.load_state_dict(state_dict, strict=False)
    except Exception as exc:
        raise NativeE5LoadError("E5 state_dict load failed") from exc
    missing = tuple(incompatible.missing_keys)
    unexpected = tuple(incompatible.unexpected_keys)
    if missing or unexpected:
        raise NativeE5LoadError(
            "E5 state_dict is not exact: missing={} unexpected={}".format(
                missing, unexpected
            )
        )

    resolved_device = torch.device(device)
    try:
        policy.to(resolved_device)
        policy.normalizer.to(resolved_device)
    except Exception as exc:
        raise NativeE5LoadError(
            "failed to move E5 policy and normalizer to {}".format(resolved_device)
        ) from exc
    policy.eval()
    policy.requires_grad_(False)
    _audit_policy_contract(policy, resolved_device)

    return NativeBCRuntime(
        policy=policy,
        identity=identity,
        checkpoint_path=path,
        device=resolved_device,
    )
