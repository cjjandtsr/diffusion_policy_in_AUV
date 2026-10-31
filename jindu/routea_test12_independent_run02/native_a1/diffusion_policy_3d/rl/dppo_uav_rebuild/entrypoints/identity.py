"""Build the hash-linked runtime identity used by every artifact.

The production identity is a closure over the code that can execute and the
behavior-bearing inputs selected by the resolved configuration.  Paths stored
in the manifest are logical repository-relative paths so checkout location is
not part of the identity.
"""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping

from omegaconf import OmegaConf

from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.contracts import RuntimeIdentity


ENVIRONMENT_SOURCE_PATHS = (
    "diffusion_policy_3d/env/collision_geom.py",
    "diffusion_policy_3d/env/kinematic_uav_env.py",
    "diffusion_policy_3d/env/traj_utils.py",
    "diffusion_policy_3d/model/common/synth_pointcloud.py",
)
R1_EXECUTED_ENVIRONMENT_SOURCE_PATHS = (
    "diffusion_policy_3d/env/reward_v2_support/environment.py",
    "diffusion_policy_3d/env/reward_v2_support/first_event.py",
)

REWARD_SOURCE_PATHS = (
    "diffusion_policy_3d/env/reward_v17.py",
    "scripts/prepare_reward_v17.py",
    "scripts/generate_reward_v17_fixtures.py",
)

BEHAVIOR_ARTIFACT_PATHS = (
    "data/reward_v17/fixture_manifest_v2.json",
    "data/reward_v17/frozen_density_v1.json",
    "data/reward_v17/calibration_report_v2.json",
)

PRODUCTION_SOURCE_PATHS = (
    "diffusion_policy_3d/rl/dppo_uav/integration.py",
    "diffusion_policy_3d/rl/routea_rl_env_adapter_v1.py",
)

CLI_SOURCE_PATHS = (
    "scripts/train_dppo_uav_rebuild.py",
    "scripts/probe_dppo_credit.py",
    "scripts/smoke_routea_r1.py",
    "scripts/run_routea_v17_gates.py",
    "scripts/calibrate_routea_v17_pointcloud.py",
    "scripts/build_routea_v17_formal_plan.py",
    "scripts/evaluate_dppo_uav_formal_endpoint.py",
    "scripts/formal_endpoint_eval_core.py",
)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: {}".format(key))
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("non-finite JSON constant: {}".format(value))


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _input_path(repo_root: Path, value: Any, label: str) -> Path:
    path = Path(value)
    explicit_absolute = path.is_absolute()
    resolved = path.resolve(strict=True) if explicit_absolute else (repo_root / path).resolve(strict=True)
    if not explicit_absolute and not _is_inside(resolved, repo_root):
        raise ValueError("{} is outside repository root".format(label))
    return resolved


def _resolved_config(config: Any) -> Mapping[str, Any]:
    if OmegaConf.is_config(config):
        resolved = OmegaConf.to_container(config, resolve=True)
    else:
        resolved = config
    if not isinstance(resolved, Mapping):
        raise TypeError("resolved config must be a mapping")
    return resolved


def _relative_path(path: Path, repo_root: Path) -> str:
    return str(path.relative_to(repo_root)).replace("\\", "/")


def _iter_source_paths(repo_root: Path) -> Iterable[Path]:
    package = repo_root / "diffusion_policy_3d" / "rl" / "dppo_uav_rebuild"
    vendor_overlay = (
        repo_root / "third_party" / "dppo_official_cc7234a" / "uav_idp3"
    )
    for base, patterns in (
        (package, ("*.py", "*.yaml")),
        (vendor_overlay, ("*.py", "*.patch")),
    ):
        if not base.is_dir():
            continue
        for pattern in patterns:
            for path in base.rglob(pattern):
                if "__pycache__" not in path.parts:
                    yield path.resolve(strict=True)
    required = (
        ENVIRONMENT_SOURCE_PATHS
        + REWARD_SOURCE_PATHS
        + BEHAVIOR_ARTIFACT_PATHS
        + PRODUCTION_SOURCE_PATHS
        + CLI_SOURCE_PATHS[:2]
    )
    for relative in required:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError("required executed source missing: {}".format(relative))
        yield path.resolve(strict=True)
    for relative in CLI_SOURCE_PATHS[2:]:
        path = repo_root / relative
        if path.is_file():
            yield path.resolve(strict=True)
    reward_support = repo_root / "diffusion_policy_3d" / "env" / "reward_v17_support"
    reward_paths = sorted(reward_support.glob("*.py")) if reward_support.is_dir() else []
    if not reward_paths:
        raise FileNotFoundError(
            "required executed source missing: "
            "diffusion_policy_3d/env/reward_v17_support/*.py"
        )
    for path in reward_paths:
        yield path.resolve(strict=True)
    r1_support = repo_root / "diffusion_policy_3d" / "env" / "reward_v2_support"
    for path in sorted(r1_support.glob("*.py")) if r1_support.is_dir() else ():
        yield path.resolve(strict=True)
    credit_support = repo_root / "scripts" / "dppo_credit_support"
    credit_paths = sorted(credit_support.glob("*.py")) if credit_support.is_dir() else []
    if not credit_paths:
        raise FileNotFoundError("required executed source missing: scripts/dppo_credit_support/*.py")
    for path in credit_paths:
        yield path.resolve(strict=True)


def _source_rows(repo_root: Path) -> List[Dict[str, Any]]:
    by_relative: Dict[str, Path] = {}
    for path in _iter_source_paths(repo_root):
        if not _is_inside(path, repo_root):
            raise ValueError("executed source is outside repository root")
        relative = _relative_path(path, repo_root)
        previous = by_relative.get(relative)
        if previous is not None and previous != path:
            raise ValueError("duplicate executed-source relative path: {}".format(relative))
        by_relative[relative] = path
    return [
        {
            "relative_path": relative,
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for relative, path in sorted(by_relative.items())
    ]


def _world_manifest_path(manifest_path: Path, manifest: Mapping[str, Any]) -> Path:
    config = manifest.get("config")
    if not isinstance(config, Mapping):
        raise TypeError("task manifest config must be a mapping")
    value = config.get("world_manifest")
    if not isinstance(value, str) or not value:
        raise ValueError("task manifest must name config.world_manifest")
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve(strict=True)
    root = manifest_path.parent.resolve(strict=True)
    resolved = (root / candidate).resolve(strict=True)
    if not _is_inside(resolved, root):
        raise ValueError("world manifest path escapes task manifest directory")
    return resolved


def _obstacle_rows(repo_root: Path, config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    value = config.get("worlds_train")
    if not isinstance(value, str) or not value:
        raise ValueError("resolved config must name worlds_train")
    root = _input_path(repo_root, value, "worlds_train")
    if not root.is_dir():
        raise NotADirectoryError("worlds_train is not a directory")
    paths = sorted(root.rglob("*.world.obstacles.json"))
    if not paths:
        raise FileNotFoundError("worlds_train has no *.world.obstacles.json files")
    return [
        {
            "logical_path": "worlds_train/{}".format(
                str(path.relative_to(root)).replace("\\", "/")
            ),
            "sha256": _sha256(path),
            "size": path.stat().st_size,
        }
        for path in paths
    ]


def _has_symlink_component(path: Path, root: Path) -> bool:
    current = root
    for part in path.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _reward_artifact_rows(
    repo_root: Path, config: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    if config.get("reward_version") != "v17":
        return []
    rows: List[Dict[str, Any]] = []
    for name in ("reward_calibration", "reward_density"):
        entry = config.get(name)
        if not isinstance(entry, Mapping):
            raise TypeError("{} must be a mapping for Reward v1.7".format(name))
        logical_path = entry.get("path")
        declared_sha256 = entry.get("sha256")
        if not isinstance(logical_path, str) or not logical_path:
            raise ValueError("{}.path must be a non-empty string".format(name))
        if Path(logical_path).is_absolute():
            raise ValueError("{}.path must be repository-relative".format(name))
        if not isinstance(declared_sha256, str) or re.fullmatch(
            r"[0-9a-fA-F]{64}", declared_sha256
        ) is None:
            raise ValueError("{}.sha256 must be 64 hexadecimal characters".format(name))
        unresolved = repo_root / logical_path
        resolved = unresolved.resolve(strict=True)
        if not _is_inside(resolved, repo_root):
            raise ValueError("{}.path is outside repository root".format(name))
        if _has_symlink_component(unresolved, repo_root):
            raise ValueError("{}.path must not traverse a symlink".format(name))
        if not resolved.is_file():
            raise FileNotFoundError("{} is not a regular file".format(name))
        size = resolved.stat().st_size
        if size > 64 * 1024 * 1024:
            raise ValueError("{} exceeds 64 MiB".format(name))
        actual_sha256 = _sha256(resolved)
        if actual_sha256 != declared_sha256.lower():
            raise ValueError("{} SHA256 mismatch".format(name))
        rows.append(
            {
                "name": name,
                "logical_path": str(Path(logical_path)).replace("\\", "/"),
                "sha256": actual_sha256,
                "size": size,
            }
        )
    return rows


def build_executed_source_manifest(repo_root, config, manifest_path):
    """Return the canonical code/input closure for one production run."""

    root = Path(repo_root).resolve(strict=True)
    manifest_file = _input_path(root, manifest_path, "task manifest")
    if not manifest_file.is_file():
        raise FileNotFoundError("task manifest is not a file")
    if manifest_file.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("task manifest exceeds 64 MiB")
    try:
        task_manifest = json.loads(
            manifest_file.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("task manifest must be valid UTF-8 JSON") from exc
    if not isinstance(task_manifest, Mapping):
        raise TypeError("task manifest root must be a mapping")

    resolved_config = _resolved_config(config)
    checkpoint_value = resolved_config.get("checkpoint")
    if not isinstance(checkpoint_value, str) or not checkpoint_value:
        raise ValueError("resolved config must name checkpoint")
    checkpoint = _input_path(root, checkpoint_value, "BC checkpoint")
    if not checkpoint.is_file():
        raise FileNotFoundError("BC checkpoint is not a file")

    world_manifest = _world_manifest_path(manifest_file, task_manifest)
    if not world_manifest.is_file():
        raise FileNotFoundError("world manifest is not a file")

    result = {
        "schema_version": 1,
        "files": _source_rows(root),
        "inputs": {
            "resolved_config_sha256": hashlib.sha256(
                _canonical_json(resolved_config)
            ).hexdigest(),
            "task_manifest_sha256": _sha256(manifest_file),
            "world_manifest_sha256": _sha256(world_manifest),
            "bc_checkpoint_sha256": _sha256(checkpoint),
            "obstacle_files": _obstacle_rows(root, resolved_config),
            "reward_artifacts": _reward_artifact_rows(root, resolved_config),
        },
    }
    result["content_hash"] = _canonical_hash(result)
    return result


def build_environment_provenance(repo_root, *, reward_version="v17"):
    """Record executed-path and observation sources, without importing the env."""

    paths = ENVIRONMENT_SOURCE_PATHS
    if str(reward_version) == "v2":
        paths += R1_EXECUTED_ENVIRONMENT_SOURCE_PATHS
    sources = {
        relative: _sha256(Path(repo_root) / relative)
        for relative in paths
    }
    return {
        "schema_version": 1,
        "evidence_scope": "lightweight_kinematic",
        "source_sha256": sources,
        "content_hash": _canonical_hash(sources),
    }


def source_set_hash(repo_root, config=None, manifest_path=None):
    """Hash the full production closure, or the legacy code-only test fixture."""

    if config is not None or manifest_path is not None:
        if config is None or manifest_path is None:
            raise ValueError("config and manifest_path must be provided together")
        return build_executed_source_manifest(
            repo_root, config, manifest_path
        )["content_hash"]

    # Compatibility for the pre-F0 environment portability unit test only.
    root = Path(repo_root).resolve(strict=True)
    package = root / "diffusion_policy_3d" / "rl" / "dppo_uav_rebuild"
    vendor_overlay = root / "third_party" / "dppo_official_cc7234a" / "uav_idp3"
    paths = sorted(
        list(package.rglob("*.py"))
        + list(package.rglob("*.yaml"))
        + list(vendor_overlay.rglob("*.py"))
        + list(vendor_overlay.rglob("*.patch"))
        + [
            root / relative
            for relative in (
                ENVIRONMENT_SOURCE_PATHS
                + REWARD_SOURCE_PATHS
                + BEHAVIOR_ARTIFACT_PATHS
                + PRODUCTION_SOURCE_PATHS
                + CLI_SOURCE_PATHS
            )
        ]
        + list(
            (root / "diffusion_policy_3d" / "env" / "reward_v17_support").glob("*.py")
        )
        + list(
            (root / "diffusion_policy_3d" / "env" / "reward_v2_support").glob("*.py")
        )
        + list((root / "scripts" / "dppo_credit_support").glob("*.py"))
    )
    entries = [
        (_relative_path(path, root), _sha256(path))
        for path in sorted(set(paths))
        if "__pycache__" not in path.parts and path.is_file()
    ]
    return _canonical_hash(entries)


def build_runtime_identity(config, repo_root, system, manifest_path):
    resolved = OmegaConf.to_container(config, resolve=True)
    return RuntimeIdentity(
        bc_identity_hash=system.bc_runtime.identity.content_hash,
        vendor_commit=system.vendor_identity.commit,
        vendor_manifest_hash=system.vendor_identity.manifest_sha256,
        config_hash=_canonical_hash(resolved),
        source_set_hash=source_set_hash(repo_root, config, manifest_path),
        task_manifest_hash=_sha256(manifest_path),
        reward_identity_hash=system.reward.identity.source_sha256,
        observation_contract_hash=system.bc_runtime.identity.observation_spec.content_hash,
        action_contract_hash=system.bc_runtime.identity.action_spec.content_hash,
        caveats=system.bc_runtime.identity.evidence_caveats,
    )
