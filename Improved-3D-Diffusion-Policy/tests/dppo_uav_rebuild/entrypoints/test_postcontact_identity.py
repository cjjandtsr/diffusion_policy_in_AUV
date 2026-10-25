"""Environment changes must be visible to the existing checkpoint identity."""

import hashlib
from pathlib import Path

from omegaconf import OmegaConf
import pytest

from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.identity import (
    build_environment_provenance,
    source_set_hash,
)
from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.pipeline import (
    _actor_lr_for_update,
    _validate_run_config,
)


ENV_FILES = (
    "diffusion_policy_3d/env/collision_geom.py",
    "diffusion_policy_3d/env/kinematic_uav_env.py",
    "diffusion_policy_3d/env/traj_utils.py",
    "diffusion_policy_3d/model/common/synth_pointcloud.py",
)


def _fixture(root):
    for relative in ENV_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"original\n")


@pytest.mark.parametrize("relative", ENV_FILES)
def test_environment_edit_changes_identity_and_recorded_source_hash(tmp_path, relative):
    _fixture(tmp_path)
    before = source_set_hash(tmp_path)
    evidence = build_environment_provenance(tmp_path)
    assert evidence["source_sha256"][relative] == hashlib.sha256(b"original\n").hexdigest()
    assert evidence["evidence_scope"] == "lightweight_kinematic"
    (tmp_path / relative).write_bytes(b"modified\n")
    assert source_set_hash(tmp_path) != before
    after = build_environment_provenance(tmp_path)
    assert after["content_hash"] != evidence["content_hash"]
    assert after["source_sha256"][relative] == hashlib.sha256(b"modified\n").hexdigest()


def test_environment_identity_is_portable_across_checkout_locations(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    _fixture(left)
    _fixture(right)
    assert build_environment_provenance(left) == build_environment_provenance(right)
    assert source_set_hash(left) == source_set_hash(right)




def test_postcontact_run_preserves_formal_budget_and_changes_only_output():
    root = Path(__file__).resolve().parents[3]
    folder = root / "diffusion_policy_3d/rl/dppo_uav_rebuild/config"
    old = OmegaConf.load(folder / "train_e5_dppo_formal_seed1_v1.yaml")
    new = OmegaConf.load(folder / "train_e5_dppo_formal_seed1_contactfix_v1.yaml")
    _validate_run_config(new)
    old_fields = OmegaConf.to_container(old, resolve=True)
    new_fields = OmegaConf.to_container(new, resolve=True)
    assert new_fields.pop("output_dir") != old_fields.pop("output_dir")
    assert new_fields == old_fields
    assert new.run.updates * new.run.rollout_transitions == 262144
    assert _actor_lr_for_update(new, 2) == pytest.approx(2e-5)
    assert _actor_lr_for_update(new, 256) == pytest.approx(1e-5)
