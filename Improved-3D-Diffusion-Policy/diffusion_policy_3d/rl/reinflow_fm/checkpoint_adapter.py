"""Native Hydra checkpoint construction for Phase1aPolicyFM."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import inspect
from pathlib import Path
import re
import textwrap
from typing import Any, Callable, Mapping, Optional

import torch

from .checkpoint_contract import FMCheckpointIdentity, FMIdentityError


EXPECTED_POLICY_TARGET = (
    "diffusion_policy_3d.policy.phase1a_policy_fm.Phase1aPolicyFM"
)
EXPECTED_WEIGHT_SOURCE = "model"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class NativeFMLoadError(RuntimeError):
    """Raised before returning an ambiguous or incompatible FM runtime."""


@dataclass(frozen=True)
class NativeFMRuntime:
    policy: Any
    identity: FMCheckpointIdentity
    checkpoint_path: Path
    device: torch.device


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_tensor_entries(entries: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    if not entries:
        raise NativeFMLoadError("tensor identity set is empty")
    for key in sorted(entries):
        tensor = entries[key]
        if not torch.is_tensor(tensor):
            raise NativeFMLoadError(f"state is not a tensor: {key}")
        value = tensor.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def hash_normalizer_state_dict(state_dict: Mapping[str, Any]) -> str:
    entries = {
        key: value for key, value in state_dict.items() if key.startswith("normalizer.")
    }
    if not entries:
        raise NativeFMLoadError("checkpoint model contains no normalizer state")
    return _hash_tensor_entries(entries)


def hash_module_state_dict(module: Any) -> str:
    state = module.state_dict()
    return _hash_tensor_entries(state)


def _plain_config(cfg: Any) -> Mapping[str, Any]:
    if isinstance(cfg, Mapping):
        return cfg
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            resolved = OmegaConf.to_container(cfg, resolve=True)
            if isinstance(resolved, Mapping):
                return resolved
    except ImportError:
        pass
    raise NativeFMLoadError("checkpoint cfg must resolve to a mapping")


def _native_hydra_instantiate(policy_cfg: Any) -> Any:
    try:
        import hydra
    except ImportError as error:
        raise NativeFMLoadError(
            "hydra-core is required to instantiate the native FM checkpoint"
        ) from error
    try:
        return hydra.utils.instantiate(policy_cfg)
    except Exception as error:
        raise NativeFMLoadError("native Phase1aPolicyFM construction failed") from error


def _qualified_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__name__}"


def _policy_source_sha256(policy: Any) -> str:
    source = inspect.getsourcefile(type(policy)) or inspect.getfile(type(policy))
    path = Path(source).resolve()
    if not path.is_file():
        raise NativeFMLoadError(f"policy source file is unavailable: {path}")
    return sha256_file(path)


def _anchor_is_disabled_in_predict_action(policy: Any) -> bool:
    try:
        source = textwrap.dedent(inspect.getsource(policy.predict_action))
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError) as error:
        raise NativeFMLoadError(
            "cannot audit predict_action anchor behavior from policy source"
        ) from error
    assignments = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "anchor_idx":
                    assignments.append(node.value)
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name) and target.id == "anchor_idx":
                assignments.append(node.value)
    return len(assignments) == 1 and isinstance(assignments[0], ast.Constant) and assignments[0].value is None


def _audit_fp32(policy: Any) -> None:
    for name, tensor in tuple(policy.named_parameters()) + tuple(policy.named_buffers()):
        if tensor.is_floating_point() and tensor.dtype != torch.float32:
            raise NativeFMLoadError(
                f"FM FP32 contract violated by {name}: {tensor.dtype}"
            )


def _runtime_values(policy: Any) -> dict[str, Any]:
    required = (
        "num_fm_steps",
        "sigma_min",
        "fm_t_scale",
        "cfg_lambda",
        "horizon",
        "action_dim",
        "n_obs_steps",
        "n_action_steps",
    )
    values = {name: getattr(policy, name, None) for name in required}
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise NativeFMLoadError(f"policy runtime attribute(s) missing: {missing}")
    if getattr(policy, "normalizer", None) is None:
        raise NativeFMLoadError("policy has no normalizer")
    if getattr(policy, "obs_encoder", None) is None:
        raise NativeFMLoadError("policy has no obs_encoder")
    if getattr(policy, "use_pc_color", None) is not False:
        raise NativeFMLoadError("FM contract requires point-cloud XYZ only")
    if not callable(getattr(policy, "_compute_cfg_velocity", None)):
        raise NativeFMLoadError("policy has no native _compute_cfg_velocity entrypoint")
    return values


def load_native_fm(
    checkpoint_path: Path | str,
    expected_sha256: str,
    device: str,
    *,
    expected_policy_target: str = EXPECTED_POLICY_TARGET,
    instantiate_policy: Optional[Callable[[Any], Any]] = None,
) -> NativeFMRuntime:
    """Instantiate ``cfg.policy`` and strictly load ``state_dicts['model']``."""

    if _SHA256_RE.fullmatch(expected_sha256 or "") is None:
        raise NativeFMLoadError("expected_sha256 must be a lowercase SHA256")
    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise NativeFMLoadError(f"FM checkpoint does not exist: {path}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise NativeFMLoadError(
            f"checkpoint SHA256 mismatch: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise NativeFMLoadError("trusted FM checkpoint deserialization failed") from error
    if not isinstance(payload, Mapping):
        raise NativeFMLoadError("checkpoint payload must be a mapping")
    cfg = payload.get("cfg")
    resolved_cfg = _plain_config(cfg)
    policy_cfg = resolved_cfg.get("policy")
    if not isinstance(policy_cfg, Mapping):
        raise NativeFMLoadError("checkpoint cfg.policy must be a mapping")
    configured_target = policy_cfg.get("_target_")
    if configured_target != expected_policy_target:
        raise NativeFMLoadError(
            "configured policy target mismatch: "
            f"expected {expected_policy_target}, got {configured_target}"
        )

    state_dicts = payload.get("state_dicts")
    if not isinstance(state_dicts, Mapping):
        raise NativeFMLoadError("checkpoint is missing state_dicts")
    state_dict = state_dicts.get(EXPECTED_WEIGHT_SOURCE)
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise NativeFMLoadError("checkpoint is missing state_dicts['model']")
    normalizer_sha256 = hash_normalizer_state_dict(state_dict)

    constructor = instantiate_policy or _native_hydra_instantiate
    try:
        policy = constructor(policy_cfg)
    except NativeFMLoadError:
        raise
    except Exception as error:
        raise NativeFMLoadError("native Phase1aPolicyFM construction failed") from error
    actual_target = _qualified_name(policy)
    if actual_target != expected_policy_target:
        raise NativeFMLoadError(
            f"policy class mismatch: expected {expected_policy_target}, got {actual_target}"
        )

    try:
        incompatible = policy.load_state_dict(state_dict, strict=True)
    except Exception as error:
        raise NativeFMLoadError("FM state_dict exact load failed") from error
    if tuple(incompatible.missing_keys) or tuple(incompatible.unexpected_keys):
        raise NativeFMLoadError(
            "FM state_dict is not exact: "
            f"missing={tuple(incompatible.missing_keys)} "
            f"unexpected={tuple(incompatible.unexpected_keys)}"
        )

    resolved_device = torch.device(device)
    try:
        policy.to(resolved_device)
        policy.normalizer.to(resolved_device)
    except Exception as error:
        raise NativeFMLoadError(
            f"failed to move FM policy to {resolved_device}"
        ) from error

    values = _runtime_values(policy)
    anchor_enabled = not _anchor_is_disabled_in_predict_action(policy)
    execution_slice = (
        int(values["n_obs_steps"]) - 1,
        int(values["n_obs_steps"]) - 1 + int(values["n_action_steps"]),
    )
    try:
        identity = FMCheckpointIdentity(
            checkpoint_sha256=actual_sha256,
            policy_target=actual_target,
            policy_source_sha256=_policy_source_sha256(policy),
            normalizer_sha256=normalizer_sha256,
            weight_source=EXPECTED_WEIGHT_SOURCE,
            num_fm_steps=int(values["num_fm_steps"]),
            sigma_min=float(values["sigma_min"]),
            fm_t_scale=float(values["fm_t_scale"]),
            cfg_lambda=float(values["cfg_lambda"]),
            anchor_enabled=anchor_enabled,
            horizon_steps=int(values["horizon"]),
            action_dim=int(values["action_dim"]),
            n_obs_steps=int(values["n_obs_steps"]),
            n_action_steps=int(values["n_action_steps"]),
            execution_slice=execution_slice,
        )
    except FMIdentityError as error:
        raise NativeFMLoadError(f"FM checkpoint contract rejected: {error}") from error

    _audit_fp32(policy)
    policy.eval()
    policy.requires_grad_(False)
    return NativeFMRuntime(
        policy=policy,
        identity=identity,
        checkpoint_path=path,
        device=resolved_device,
    )


__all__ = [
    "EXPECTED_POLICY_TARGET",
    "NativeFMLoadError",
    "NativeFMRuntime",
    "hash_module_state_dict",
    "hash_normalizer_state_dict",
    "load_native_fm",
    "sha256_file",
]
