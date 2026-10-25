"""Behavioral contract for the Route A executed-source closure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf
import pytest

from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints import identity


REQUIRED_FILES = (
    "diffusion_policy_3d/rl/dppo_uav/integration.py",
    "diffusion_policy_3d/rl/routea_rl_env_adapter_v1.py",
    "diffusion_policy_3d/rl/dppo_uav_rebuild/entrypoints/identity.py",
    "third_party/dppo_official_cc7234a/uav_idp3/loss.py",
    "diffusion_policy_3d/env/collision_geom.py",
    "diffusion_policy_3d/env/kinematic_uav_env.py",
    "diffusion_policy_3d/env/traj_utils.py",
    "diffusion_policy_3d/env/reward_v17.py",
    "diffusion_policy_3d/env/reward_v17_support/config.py",
    "diffusion_policy_3d/model/common/synth_pointcloud.py",
    "scripts/train_dppo_uav_rebuild.py",
    "scripts/probe_dppo_credit.py",
    "scripts/prepare_reward_v17.py",
    "scripts/generate_reward_v17_fixtures.py",
    "scripts/dppo_credit_support/credit_bank.py",
    "scripts/dppo_credit_support/credit_probe.py",
    "data/reward_v17/fixture_manifest_v2.json",
    "data/reward_v17/frozen_density_v1.json",
    "data/reward_v17/calibration_report_v2.json",
)


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _fixture(root: Path):
    for index, relative in enumerate(REQUIRED_FILES):
        _write(root / relative, "file-{}\n".format(index).encode("ascii"))
    checkpoint = root / "fixtures/bc.ckpt"
    obstacle = root / "fixtures/worlds/route.world.obstacles.json"
    world_manifest = root / "fixtures/world_manifest.json"
    _write(checkpoint, b"frozen-bc\n")
    _write(obstacle, b'{"type":"sphere","radius":1.0}\n')
    _write(world_manifest, b'{"worlds":[]}\n')
    manifest = root / "fixtures/admission_manifest.json"
    manifest.write_text(
        json.dumps({"config": {"world_manifest": str(world_manifest)}}),
        encoding="utf-8",
    )
    config = OmegaConf.create(
        {
            "checkpoint": str(checkpoint),
            "admission_manifest": str(manifest),
            "worlds_train": str(root / "fixtures/worlds"),
            "run": {"seed": 19},
        }
    )
    return config, manifest


def test_production_sources_are_in_identity(tmp_path):
    """Catches omitting the two WSL production loaders from checkpoint identity."""

    config, manifest_path = _fixture(tmp_path)
    result = identity.build_executed_source_manifest(
        tmp_path, config, manifest_path
    )
    rows = {row["relative_path"]: row for row in result["files"]}
    assert "diffusion_policy_3d/rl/dppo_uav/integration.py" in rows
    assert "diffusion_policy_3d/rl/routea_rl_env_adapter_v1.py" in rows
    assert rows["diffusion_policy_3d/rl/dppo_uav/integration.py"]["size"] == 7
    assert rows["diffusion_policy_3d/rl/dppo_uav/integration.py"][
        "sha256"
    ] == hashlib.sha256(b"file-0\n").hexdigest()
    for required in (
        "diffusion_policy_3d/env/reward_v17.py",
        "diffusion_policy_3d/env/reward_v17_support/config.py",
        "scripts/prepare_reward_v17.py",
        "scripts/generate_reward_v17_fixtures.py",
        "data/reward_v17/fixture_manifest_v2.json",
        "data/reward_v17/frozen_density_v1.json",
        "data/reward_v17/calibration_report_v2.json",
    ):
        assert required in rows


def test_source_identity_changes_on_production_byte_change(tmp_path):
    """Catches a production edit that leaves the source-set hash unchanged."""

    config, manifest_path = _fixture(tmp_path)
    before = identity.source_set_hash(tmp_path, config, manifest_path)
    target = tmp_path / "diffusion_policy_3d/rl/dppo_uav/integration.py"
    target.write_bytes(b"different\n")
    assert identity.source_set_hash(tmp_path, config, manifest_path) != before


def test_input_identity_covers_config_manifest_obstacle_and_checkpoint(tmp_path):
    """Catches silently excluding any behavior-bearing non-source input."""

    config, manifest_path = _fixture(tmp_path)
    result = identity.build_executed_source_manifest(
        tmp_path, config, manifest_path
    )
    inputs = result["inputs"]
    assert inputs["resolved_config_sha256"] == hashlib.sha256(
        b'{"admission_manifest":"'
        + str(manifest_path).replace("\\", "\\\\").encode("utf-8")
        + b'","checkpoint":"'
        + str(tmp_path / "fixtures/bc.ckpt").replace("\\", "\\\\").encode("utf-8")
        + b'","run":{"seed":19},"worlds_train":"'
        + str(tmp_path / "fixtures/worlds").replace("\\", "\\\\").encode("utf-8")
        + b'"}'
    ).hexdigest()
    assert inputs["task_manifest_sha256"] == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    assert inputs["bc_checkpoint_sha256"] == hashlib.sha256(
        b"frozen-bc\n"
    ).hexdigest()
    assert inputs["world_manifest_sha256"] == hashlib.sha256(
        b'{"worlds":[]}\n'
    ).hexdigest()
    assert inputs["obstacle_files"] == [
        {
            "logical_path": "worlds_train/route.world.obstacles.json",
            "sha256": hashlib.sha256(
                b'{"type":"sphere","radius":1.0}\n'
            ).hexdigest(),
            "size": 31,
        }
    ]


def test_v17_artifact_bytes_are_bound_as_inputs(tmp_path):
    config, manifest_path = _fixture(tmp_path)
    config.reward_version = "v17"
    config.reward_density = {
        "path": "data/reward_v17/frozen_density_v1.json",
        "sha256": hashlib.sha256(
            (tmp_path / "data/reward_v17/frozen_density_v1.json").read_bytes()
        ).hexdigest(),
    }
    config.reward_calibration = {
        "path": "data/reward_v17/calibration_report_v2.json",
        "sha256": hashlib.sha256(
            (tmp_path / "data/reward_v17/calibration_report_v2.json").read_bytes()
        ).hexdigest(),
    }
    before = identity.build_executed_source_manifest(
        tmp_path, config, manifest_path
    )
    rows = before["inputs"]["reward_artifacts"]
    assert [row["name"] for row in rows] == [
        "reward_calibration", "reward_density"
    ]
    target = tmp_path / "data/reward_v17/calibration_report_v2.json"
    target.write_bytes(b"changed calibration\n")
    with pytest.raises(ValueError, match="reward_calibration SHA256 mismatch"):
        identity.build_executed_source_manifest(
            tmp_path, config, manifest_path
        )


def test_missing_production_source_fails_closed(tmp_path):
    """Catches identity generation proceeding with an incomplete source tree."""

    config, manifest_path = _fixture(tmp_path)
    (tmp_path / "diffusion_policy_3d/rl/dppo_uav/integration.py").unlink()
    with pytest.raises(FileNotFoundError, match="dppo_uav/integration.py"):
        identity.build_executed_source_manifest(tmp_path, config, manifest_path)


def test_explicit_absolute_manifest_outside_repo_is_hashed(tmp_path):
    """Catches rejecting the production layout's explicit external manifest."""

    repo = tmp_path / "repo"
    config, _ = _fixture(repo)
    external = tmp_path / "external"
    world_manifest = external / "world_manifest.json"
    _write(world_manifest, b'{"worlds":[]}\n')
    manifest = external / "admission_manifest.json"
    manifest.write_text(
        json.dumps({"config": {"world_manifest": str(world_manifest)}}),
        encoding="utf-8",
    )
    config.admission_manifest = str(manifest)
    result = identity.build_executed_source_manifest(repo, config, manifest)
    assert result["inputs"]["task_manifest_sha256"] == hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()


def test_relative_manifest_path_must_not_escape_repo_root(tmp_path):
    """Catches an unreviewed relative-path traversal."""

    repo = tmp_path / "repo"
    config, _ = _fixture(repo)
    escaped = tmp_path / "outside.json"
    escaped.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="outside repository root"):
        identity.build_executed_source_manifest(
            repo, config, "../outside.json"
        )


def test_task_manifest_duplicate_keys_are_rejected(tmp_path):
    config, manifest = _fixture(tmp_path)
    manifest.write_text(
        '{"config":{"world_manifest":"first"},"config":{"world_manifest":"second"}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        identity.build_executed_source_manifest(tmp_path, config, manifest)
