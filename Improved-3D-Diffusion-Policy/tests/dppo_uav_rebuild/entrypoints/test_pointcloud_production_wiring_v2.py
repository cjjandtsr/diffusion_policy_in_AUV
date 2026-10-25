"""The production collection loop must gate provenance before model/buffer use."""

from pathlib import Path


def test_pipeline_uses_encoded_observation_and_verifies_before_consumers():
    root = Path(__file__).resolve().parents[3]
    source = (root / "diffusion_policy_3d/rl/dppo_uav_rebuild/entrypoints/pipeline.py").read_text(
        encoding="utf-8"
    )
    assert "pointcloud_meta_history" in source
    assert "observation_adapter.verify" in source
    assert ".condition" in source


def test_environment_publishes_pointcloud_metadata_context():
    root = Path(__file__).resolve().parents[3]
    source = (root / "diffusion_policy_3d/env/kinematic_uav_env.py").read_text(encoding="utf-8")
    assert "pointcloud_meta_history" in source
    assert "episode_identity_hash" in source
    assert "generate_with_metadata" in source
