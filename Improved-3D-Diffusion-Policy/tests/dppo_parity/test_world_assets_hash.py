"""Gate D reopening (2026-08-22): external world assets content hashing.

The worlds dir lives OUTSIDE this repo
(/home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds/train) — swapping a
world file mid-training was invisible to every identity axis. The
aggregate world_assets_hash (scene + obstacle files per admitted world)
must be deterministic, order-independent, and hard-fail on a missing
asset; a content change must change the hash (preflight → config hash →
resume identity all bind it).

Tested: ``entrypoint_guard.world_assets_hash_of`` (pure function over a
worlds dir + uid list) and the real preflight integration
(validate_official_environment returns a non-empty world_assets_hash).
"""

import json
from pathlib import Path

import pytest

from diffusion_policy_3d.rl.dppo_uav.entrypoint_guard import (
    world_assets_hash_of,
)

# The production worlds dir + admission manifest (real preflight check).
PX4_WORLDS_TRAIN = Path(
    "/home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds/train"
)
REPO_MANIFEST = Path(
    "/home/cjj/AUV/program/reports/routeA_rl_env_v1/calibration"
    "/admission_manifest.json"
)
REPO_ROOT = Path(
    "/home/cjj/AUV/Improved-3D-Diffusion-Policy-main/"
    "Improved-3D-Diffusion-Policy"
)


def _write_worlds(tmp_path: Path, uids, scene_bytes=b"scene",
                  obstacles_bytes=b"[]"):
    """Create a worlds dir with <uid>.world + <uid>.world.obstacles.json."""
    for uid in uids:
        (tmp_path / f"{uid}.world").write_bytes(scene_bytes)
        (tmp_path / f"{uid}.world.obstacles.json").write_bytes(
            obstacles_bytes)
    return tmp_path


def test_deterministic_same_input_same_hash(tmp_path):
    _write_worlds(tmp_path, ["w1", "w2"])
    assert world_assets_hash_of(tmp_path, ["w1", "w2"]) == \
        world_assets_hash_of(tmp_path, ["w1", "w2"])


def test_order_independent_aggregation(tmp_path):
    _write_worlds(tmp_path, ["w1", "w2"])
    assert world_assets_hash_of(tmp_path, ["w1", "w2"]) == \
        world_assets_hash_of(tmp_path, ["w2", "w1"])


def test_content_change_changes_hash(tmp_path):
    _write_worlds(tmp_path, ["w1"], obstacles_bytes=b'{"a": 1}')
    before = world_assets_hash_of(tmp_path, ["w1"])
    # obstacle content changed -> hash must change (swapped world file)
    _write_worlds(tmp_path, ["w1"], obstacles_bytes=b'{"a": 2}')
    after = world_assets_hash_of(tmp_path, ["w1"])
    assert before != after
    # scene content changed -> hash must change too
    _write_worlds(tmp_path, ["w1"], scene_bytes=b"scene-v2")
    assert after != world_assets_hash_of(tmp_path, ["w1"])


def test_missing_scene_hard_fails(tmp_path):
    (tmp_path / "w1.world.obstacles.json").write_text("[]")
    with pytest.raises(RuntimeError, match="scene missing"):
        world_assets_hash_of(tmp_path, ["w1"])


def test_missing_obstacles_hard_fails(tmp_path):
    (tmp_path / "w1.world").write_bytes(b"scene")
    with pytest.raises(RuntimeError, match="obstacles missing"):
        world_assets_hash_of(tmp_path, ["w1"])


def test_empty_world_set_hard_fails(tmp_path):
    with pytest.raises(RuntimeError, match="no admitted worlds"):
        world_assets_hash_of(tmp_path, [])


def test_per_world_binding_includes_both_files(tmp_path):
    """Each world binds BOTH files — a change to one file only changes
    the aggregate (two worlds with identical scenes but different
    obstacles hash differently)."""
    _write_worlds(tmp_path, ["w1"], obstacles_bytes=b"[]")
    _write_worlds(tmp_path, ["w2"], obstacles_bytes=b"[1]")
    assert world_assets_hash_of(tmp_path, ["w1"]) != \
        world_assets_hash_of(tmp_path, ["w2"])


def _repo_tree_clean() -> bool:
    """preflight requires a clean tree — during active fixes the tree is
    dirty by definition, so the integration test skips until the final
    commit (it must pass on the final clean HEAD)."""
    import subprocess

    out = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT,
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        return False
    return all(ln.startswith("?? temp/") for ln in out.stdout.splitlines())


@pytest.mark.skipif(
    not PX4_WORLDS_TRAIN.is_dir() or not REPO_MANIFEST.is_file()
    or not _repo_tree_clean(),
    reason="production worlds/manifest absent, or tree dirty (preflight "
           "requires clean; runs on the final commit)",
)
def test_real_preflight_records_world_assets_hash():
    """The real preflight binds the production world pool — a non-empty
    deterministic world_assets_hash enters the facts (and therefore the
    runtime identity / config layer / checkpoint / metrics / artifact)."""
    from diffusion_policy_3d.rl.dppo_uav.entrypoint_guard import (
        validate_official_environment,
    )

    facts = validate_official_environment(
        manifest_path=str(REPO_MANIFEST),
        bc_checkpoint_path=str(REPO_ROOT / "data" / "outputs" /
                               "2026.05.17" /
                               "13.36.20_v8_phase2_sweep_lambda0.5" /
                               "checkpoints" / "latest.ckpt"),
        worlds_train_dir=str(PX4_WORLDS_TRAIN),
    )
    wh = facts["world_assets_hash"]
    assert isinstance(wh, str) and len(wh) == 64
    # deterministic across calls on the same tree
    facts2 = validate_official_environment(
        manifest_path=str(REPO_MANIFEST),
        bc_checkpoint_path=str(REPO_ROOT / "data" / "outputs" /
                               "2026.05.17" /
                               "13.36.20_v8_phase2_sweep_lambda0.5" /
                               "checkpoints" / "latest.ckpt"),
        worlds_train_dir=str(PX4_WORLDS_TRAIN),
    )
    assert facts2["world_assets_hash"] == wh
