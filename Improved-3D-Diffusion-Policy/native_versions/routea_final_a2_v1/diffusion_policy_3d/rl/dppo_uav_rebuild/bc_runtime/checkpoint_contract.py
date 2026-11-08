"""Strict native-E5 identity extracted from a resolved checkpoint config."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import PurePosixPath, PureWindowsPath
import re
from typing import Any, Mapping, Tuple

from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.contracts import (
    ActionSpec,
    ObservationSpec,
)


EXPECTED_E5_CHECKPOINT_SHA256 = (
    "c1fca4ac2735928b18fa7a9c069f90ad181eb84092acee379b073dc6a61bfc9a"
)
EXPECTED_E5_NORMALIZER_SHA256 = (
    "1edbc81f78c66a0f67270266cf333c06071307768e1273d2369aa945320ae1ca"
)
EXPECTED_POLICY_TARGET = (
    "diffusion_policy_3d.policy.diffusion_pointcloud_policy."
    "DiffusionPointcloudPolicy"
)
EXPECTED_WORKSPACE_TARGET = (
    "diffusion_policy_3d.workspace.idp3_workspace.iDP3Workspace"
)
EXPECTED_SCHEDULER_TARGET = (
    "diffusers.schedulers.scheduling_ddim.DDIMScheduler"
)
EXPECTED_WEIGHT_SOURCE = "model"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class BCIdentityError(ValueError):
    """Raised when a checkpoint/config would change native E5 behavior."""


def _canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BCIdentityError("resolved_config must be plain JSON-compatible data") from exc
    return hashlib.sha256(encoded).hexdigest()


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BCIdentityError("{} must be a mapping".format(name))
    return value


def _require_equal(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise BCIdentityError(
            "{} mismatch: expected {!r}, got {!r}".format(name, expected, actual)
        )


def _validate_sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise BCIdentityError("{} must be a lowercase SHA256".format(name))


def _validate_checkpoint_relpath(value: str) -> None:
    if not isinstance(value, str) or not value:
        raise BCIdentityError("checkpoint_relpath must be a non-empty string")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or ".." in posix.parts:
        raise BCIdentityError(
            "checkpoint_relpath must stay within the repository: {}".format(value)
        )


def _shape_tuple(meta: Mapping[str, Any], name: str) -> Tuple[int, ...]:
    shape = meta.get("shape")
    if not isinstance(shape, (list, tuple)):
        raise BCIdentityError("{} shape is missing".format(name))
    try:
        result = tuple(int(dim) for dim in shape)
    except (TypeError, ValueError) as exc:
        raise BCIdentityError("{} shape must contain integers".format(name)) from exc
    if any(dim <= 0 for dim in result):
        raise BCIdentityError("{} shape must be positive".format(name))
    return result


@dataclass(frozen=True)
class BCIdentity:
    checkpoint_relpath: str
    checkpoint_sha256: str
    normalizer_sha256: str
    weight_source: str
    policy_target: str
    workspace_target: str
    scheduler_target: str
    prediction_type: str
    num_train_timesteps: int
    num_inference_steps: int
    observation_spec: ObservationSpec
    action_spec: ActionSpec
    config_sha256: str
    evidence_caveats: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_checkpoint_relpath(self.checkpoint_relpath)
        _validate_sha256(self.checkpoint_sha256, "checkpoint_sha256")
        _validate_sha256(self.normalizer_sha256, "normalizer_sha256")
        _validate_sha256(self.config_sha256, "config_sha256")
        _require_equal(self.weight_source, EXPECTED_WEIGHT_SOURCE, "weight_source")
        _require_equal(self.policy_target, EXPECTED_POLICY_TARGET, "policy target")
        _require_equal(self.workspace_target, EXPECTED_WORKSPACE_TARGET, "workspace target")
        _require_equal(self.scheduler_target, EXPECTED_SCHEDULER_TARGET, "scheduler target")
        _require_equal(self.prediction_type, "sample", "prediction_type")
        _require_equal(self.num_train_timesteps, 50, "num_train_timesteps")
        _require_equal(self.num_inference_steps, 10, "num_inference_steps")
        if not isinstance(self.evidence_caveats, tuple):
            raise BCIdentityError("evidence_caveats must be an immutable tuple")
        if any(
            not isinstance(item, str) or not item.strip()
            for item in self.evidence_caveats
        ):
            raise BCIdentityError("evidence_caveats must contain non-empty strings")

    @classmethod
    def from_resolved_config(
        cls,
        resolved_config: Mapping[str, Any],
        checkpoint_relpath: str,
        checkpoint_sha256: str,
        normalizer_sha256: str,
        weight_source: str,
        evidence_caveats: Tuple[str, ...] = (),
    ) -> "BCIdentity":
        cfg = _require_mapping(resolved_config, "resolved_config")
        policy = _require_mapping(cfg.get("policy"), "policy")
        scheduler = _require_mapping(
            policy.get("noise_scheduler"), "policy.noise_scheduler"
        )

        _require_equal(cfg.get("_target_"), EXPECTED_WORKSPACE_TARGET, "workspace target")
        _require_equal(policy.get("_target_"), EXPECTED_POLICY_TARGET, "policy target")
        _require_equal(
            scheduler.get("_target_"), EXPECTED_SCHEDULER_TARGET, "scheduler target"
        )
        _require_equal(scheduler.get("prediction_type"), "sample", "prediction_type")
        _require_equal(
            scheduler.get("num_train_timesteps"), 50, "num_train_timesteps"
        )
        _require_equal(policy.get("num_inference_steps"), 10, "num_inference_steps")

        for owner_name, owner in (("root", cfg), ("policy", policy)):
            _require_equal(owner.get("n_obs_steps"), 3, "{}.n_obs_steps".format(owner_name))
            _require_equal(owner.get("horizon"), 16, "{}.horizon".format(owner_name))
            _require_equal(
                owner.get("n_action_steps"),
                8,
                "{}.n_action_steps".format(owner_name),
            )

        _require_equal(policy.get("obs_as_global_cond"), True, "obs_as_global_cond")
        _require_equal(policy.get("use_pc_color"), False, "use_pc_color")
        _require_equal(
            policy.get("pointnet_type"), "multi_stage_pointnet", "pointnet_type"
        )

        for owner_name, owner in (("root", cfg), ("policy", policy)):
            shape_meta = _require_mapping(
                owner.get("shape_meta"), "{}.shape_meta".format(owner_name)
            )
            obs_meta = _require_mapping(
                shape_meta.get("obs"), "{}.shape_meta.obs".format(owner_name)
            )
            point_meta = _require_mapping(
                obs_meta.get("point_cloud"),
                "{}.shape_meta.obs.point_cloud".format(owner_name),
            )
            state_meta = _require_mapping(
                obs_meta.get("agent_pos"),
                "{}.shape_meta.obs.agent_pos".format(owner_name),
            )
            action_meta = _require_mapping(
                shape_meta.get("action"), "{}.shape_meta.action".format(owner_name)
            )
            _require_equal(
                _shape_tuple(point_meta, "point_cloud"),
                (4096, 6),
                "{}.point_cloud shape".format(owner_name),
            )
            _require_equal(
                _shape_tuple(state_meta, "agent_pos"),
                (9,),
                "{}.agent_pos shape".format(owner_name),
            )
            _require_equal(
                _shape_tuple(action_meta, "action"),
                (11,),
                "{}.action shape".format(owner_name),
            )

        encoder_cfg = _require_mapping(
            policy.get("pointcloud_encoder_cfg"), "policy.pointcloud_encoder_cfg"
        )
        _require_equal(encoder_cfg.get("in_channels"), 3, "point encoder in_channels")
        _require_equal(encoder_cfg.get("out_channels"), 128, "point encoder out_channels")
        _require_equal(encoder_cfg.get("num_points"), 4096, "point encoder num_points")

        _validate_sha256(checkpoint_sha256, "checkpoint_sha256")
        _validate_sha256(normalizer_sha256, "normalizer_sha256")
        _require_equal(
            checkpoint_sha256,
            EXPECTED_E5_CHECKPOINT_SHA256,
            "checkpoint_sha256",
        )
        _require_equal(
            normalizer_sha256,
            EXPECTED_E5_NORMALIZER_SHA256,
            "normalizer_sha256",
        )
        _require_equal(weight_source, EXPECTED_WEIGHT_SOURCE, "weight_source")

        return cls(
            checkpoint_relpath=checkpoint_relpath,
            checkpoint_sha256=checkpoint_sha256,
            normalizer_sha256=normalizer_sha256,
            weight_source=weight_source,
            policy_target=policy["_target_"],
            workspace_target=cfg["_target_"],
            scheduler_target=scheduler["_target_"],
            prediction_type=scheduler["prediction_type"],
            num_train_timesteps=int(scheduler["num_train_timesteps"]),
            num_inference_steps=int(policy["num_inference_steps"]),
            observation_spec=ObservationSpec.e5_route_a(),
            action_spec=ActionSpec.e5(),
            config_sha256=_canonical_sha256(resolved_config),
            evidence_caveats=evidence_caveats,
        )

    @property
    def content_hash(self) -> str:
        return _canonical_sha256(asdict(self))
