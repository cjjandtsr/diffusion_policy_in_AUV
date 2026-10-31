"""Native Phase1aPolicyFM condition/uncondition encoding with provenance."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Mapping, Tuple

import torch

from .checkpoint_adapter import hash_module_state_dict


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ConditionEncodingError(RuntimeError):
    """Raised when observations cannot reproduce native FM conditioning."""


@dataclass(frozen=True)
class FMConditionBatch:
    cond: torch.Tensor
    uncond: torch.Tensor
    noise_cond: torch.Tensor
    raw_obs_sha256: Tuple[str, ...]
    encoder_sha256: str
    normalizer_sha256: str
    n_obs_steps: int


def _policy_device(policy) -> torch.device:
    value = getattr(policy, "device", None)
    if value is not None:
        return torch.device(value)
    for tensor in list(policy.parameters()) + list(policy.buffers()):
        return tensor.device
    raise ConditionEncodingError("cannot infer policy device")


def _fingerprint_sample(obs: Mapping[str, torch.Tensor], index: int) -> str:
    digest = hashlib.sha256()
    for key in sorted(obs):
        tensor = obs[key]
        if not torch.is_tensor(tensor):
            raise ConditionEncodingError(f"observation {key} is not a tensor")
        value = tensor[index].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _validate_observation(policy, obs: Mapping[str, torch.Tensor]) -> tuple[int, int]:
    if set(obs) != {"agent_pos", "point_cloud"}:
        raise ConditionEncodingError(
            "observation keys must be exactly agent_pos and point_cloud"
        )
    agent_pos = obs["agent_pos"]
    point_cloud = obs["point_cloud"]
    if agent_pos.ndim != 3 or point_cloud.ndim != 4:
        raise ConditionEncodingError(
            "agent_pos/point_cloud must have shapes (B,T,S)/(B,T,N,C)"
        )
    if agent_pos.shape[0] != point_cloud.shape[0]:
        raise ConditionEncodingError("observation batch sizes differ")
    if agent_pos.shape[1] != point_cloud.shape[1]:
        raise ConditionEncodingError("observation history lengths differ")
    n_obs_steps = getattr(policy, "n_obs_steps", None)
    if not isinstance(n_obs_steps, int) or n_obs_steps <= 0:
        raise ConditionEncodingError("policy n_obs_steps is invalid")
    if agent_pos.shape[1] < n_obs_steps:
        raise ConditionEncodingError(
            f"observation history is shorter than n_obs_steps={n_obs_steps}"
        )
    if agent_pos.shape[-1] < 7:
        raise ConditionEncodingError("agent_pos must contain goal fields 3:7")
    if point_cloud.shape[-1] < 3:
        raise ConditionEncodingError("point_cloud must contain XYZ")
    if getattr(policy, "use_pc_color", None) is not False:
        raise ConditionEncodingError("native FM condition requires use_pc_color=False")
    if float(getattr(policy, "cfg_lambda", 0.0)) <= 0:
        raise ConditionEncodingError("native FM condition requires CFG enabled")
    return int(agent_pos.shape[0]), n_obs_steps


def _flatten_history(obs: Mapping[str, torch.Tensor], n_obs_steps: int):
    return {
        key: value[:, :n_obs_steps, ...].reshape(-1, *value.shape[2:])
        for key, value in obs.items()
    }


def _require_finite_fp32(name: str, value: torch.Tensor) -> None:
    if value.dtype != torch.float32:
        raise ConditionEncodingError(f"{name} must be FP32, got {value.dtype}")
    if not torch.isfinite(value).all():
        raise ConditionEncodingError(f"{name} contains NaN or Inf")


def encode_conditions(
    policy,
    observation: Mapping[str, torch.Tensor],
    *,
    normalizer_sha256: str,
) -> FMConditionBatch:
    """Reproduce ``Phase1aPolicyFM.predict_action`` condition semantics."""

    if _SHA256_RE.fullmatch(normalizer_sha256 or "") is None:
        raise ConditionEncodingError("normalizer_sha256 must be a lowercase SHA256")
    batch_size, n_obs_steps = _validate_observation(policy, observation)
    fingerprints = tuple(
        _fingerprint_sample(observation, index) for index in range(batch_size)
    )
    device = _policy_device(policy)

    try:
        normalized = policy.normalizer.normalize(observation)
    except Exception as error:
        raise ConditionEncodingError("native normalizer failed") from error
    if not isinstance(normalized, Mapping):
        raise ConditionEncodingError("native normalizer must return a mapping")
    normalized = {
        key: value.to(device=device, dtype=torch.float32)
        for key, value in normalized.items()
    }
    normalized["point_cloud"] = normalized["point_cloud"][..., :3]
    conditional_flat = _flatten_history(normalized, n_obs_steps)

    unconditional = {key: value.clone() for key, value in normalized.items()}
    unconditional["agent_pos"][:, :, 3:7] = 0.0
    unconditional_flat = _flatten_history(unconditional, n_obs_steps)

    try:
        with torch.no_grad():
            conditional_frames = policy.obs_encoder(conditional_flat)
            unconditional_frames = policy.obs_encoder(unconditional_flat)
    except Exception as error:
        raise ConditionEncodingError("native obs_encoder failed") from error
    if conditional_frames.ndim != 2 or unconditional_frames.ndim != 2:
        raise ConditionEncodingError("obs_encoder output must have shape (B*T,F)")
    if conditional_frames.shape != unconditional_frames.shape:
        raise ConditionEncodingError("conditional/unconditional encoder shapes differ")
    if conditional_frames.shape[0] != batch_size * n_obs_steps:
        raise ConditionEncodingError("obs_encoder history/batch dimension mismatch")
    feature_dim = conditional_frames.shape[1]
    configured_feature_dim = getattr(policy, "obs_feature_dim", feature_dim)
    if configured_feature_dim != feature_dim:
        raise ConditionEncodingError(
            f"obs_encoder feature mismatch: expected {configured_feature_dim}, "
            f"got {feature_dim}"
        )

    cond = conditional_frames.reshape(batch_size, -1).detach()
    uncond = unconditional_frames.reshape(batch_size, -1).detach()
    noise_cond = conditional_frames.reshape(
        batch_size, n_obs_steps, feature_dim
    ).mean(dim=1).detach()
    for name, value in (
        ("cond", cond),
        ("uncond", uncond),
        ("noise_cond", noise_cond),
    ):
        _require_finite_fp32(name, value)

    try:
        encoder_sha256 = hash_module_state_dict(policy.obs_encoder)
    except Exception as error:
        raise ConditionEncodingError("failed to hash obs_encoder state") from error
    return FMConditionBatch(
        cond=cond,
        uncond=uncond,
        noise_cond=noise_cond,
        raw_obs_sha256=fingerprints,
        encoder_sha256=encoder_sha256,
        normalizer_sha256=normalizer_sha256,
        n_obs_steps=n_obs_steps,
    )


__all__ = ["ConditionEncodingError", "FMConditionBatch", "encode_conditions"]
