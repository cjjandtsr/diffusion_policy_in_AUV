import hashlib
from pathlib import Path

import numpy as np
import pytest

from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.contracts import (
    ActionBundle,
    ActionSpec,
    ObservationSpec,
    RuntimeIdentity,
)
from diffusion_policy_3d.rl.dppo_uav_rebuild.official_core import vendor_identity

EXPECTED_DPPO_COMMIT = vendor_identity.EXPECTED_DPPO_COMMIT
verify_vendor_snapshot = vendor_identity.verify_vendor_snapshot


def test_action_spec_splits_the_e5_full_trajectory_at_the_authoritative_offset():
    """Separates the policy, deployment, and lightweight-rollout protocols."""
    full = np.arange(16 * 11, dtype=np.float32).reshape(1, 16, 11)

    policy, deployment, rollout = ActionSpec.e5().split(full)

    np.testing.assert_array_equal(policy, full[:, 2:10, :])
    np.testing.assert_array_equal(deployment, full[:, 2:4, :])
    np.testing.assert_array_equal(rollout, full[:, 2:6, :])


def test_action_bundle_rejects_a_policy_slice_from_the_wrong_offset():
    """Catches callers constructing mutually inconsistent named action views."""
    full = np.arange(16 * 11, dtype=np.float32).reshape(1, 16, 11)

    with pytest.raises(ValueError, match="policy action does not match"):
        ActionBundle(
            spec=ActionSpec.e5(),
            full=full,
            policy=full[:, :8, :],
            deployment_executed=full[:, :2, :],
            rollout_executed=full[:, :4, :],
        )


def test_observation_spec_freezes_the_e5_route_a_bridge_contract():
    """Prevents the legacy 8x192=1536 feature from becoming E5 actor input."""
    spec = ObservationSpec.e5_route_a()

    assert (spec.environment_history, spec.policy_history) == (8, 3)
    assert (spec.state_dim, spec.point_count) == (9, 4096)
    assert (spec.checkpoint_point_dim, spec.policy_point_dim) == (6, 3)
    assert spec.history_order == "chronological_oldest_to_latest"
    assert spec.feature_dim_per_frame == 192
    assert spec.policy_condition_dim == 576
    assert spec.legacy_environment_condition_dim == 1536


def test_runtime_identity_hash_is_deterministic_and_caveat_sensitive():
    fields = dict(
        bc_identity_hash="1" * 64,
        vendor_commit=EXPECTED_DPPO_COMMIT,
        vendor_manifest_hash="2" * 64,
        config_hash="3" * 64,
        source_set_hash="4" * 64,
        task_manifest_hash="5" * 64,
        reward_identity_hash="6" * 64,
        observation_contract_hash="7" * 64,
        action_contract_hash="8" * 64,
    )

    first = RuntimeIdentity(**fields, caveats=("historical deployment log unavailable",))
    second = RuntimeIdentity(**fields, caveats=("historical deployment log unavailable",))
    different = RuntimeIdentity(**fields, caveats=())

    assert first.content_hash == second.content_hash
    assert first.content_hash != different.content_hash


def test_runtime_identity_rejects_a_non_sha256_identity_field():
    with pytest.raises(ValueError, match="bc_identity_hash"):
        RuntimeIdentity(
            bc_identity_hash="not-a-sha256",
            vendor_commit=EXPECTED_DPPO_COMMIT,
            vendor_manifest_hash="2" * 64,
            config_hash="3" * 64,
            source_set_hash="4" * 64,
            task_manifest_hash="5" * 64,
            reward_identity_hash="6" * 64,
            observation_contract_hash="7" * 64,
            action_contract_hash="8" * 64,
        )


def test_vendor_snapshot_matches_the_frozen_manifest(tmp_path: Path, monkeypatch):
    """Catches a modified, incomplete, or path-escaping official snapshot."""
    vendor = tmp_path / "dppo_official_cc7234a"
    vendor.mkdir()
    source = vendor / "model" / "diffusion" / "diffusion_ppo.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"official fixture\n")
    digest = "d6f3ce370b14147abf75559357b2842a0a14dfbd51fb0688792cf6ce8babfb6b"
    manifest = f"{digest}  model/diffusion/diffusion_ppo.py\n".encode()
    (vendor / "SHA256SUMS").write_bytes(manifest)
    (vendor / "UPSTREAM_COMMIT").write_text(EXPECTED_DPPO_COMMIT + "\n", encoding="utf-8")
    monkeypatch.setattr(
        vendor_identity,
        "EXPECTED_VENDOR_MANIFEST_SHA256",
        hashlib.sha256(manifest).hexdigest(),
    )
    monkeypatch.setattr(vendor_identity, "EXPECTED_VENDOR_FILE_COUNT", 1)

    identity = verify_vendor_snapshot(vendor)

    assert identity.commit == EXPECTED_DPPO_COMMIT
    assert identity.file_count == 1
    assert len(identity.manifest_sha256) == 64
    assert identity.overlay_files == ()


def test_vendor_snapshot_rejects_a_path_escape(tmp_path: Path):
    """Catches a malicious or malformed SHA256SUMS entry outside vendor root."""
    vendor = tmp_path / "dppo_official_cc7234a"
    vendor.mkdir()
    (vendor / "SHA256SUMS").write_text(f"{'0' * 64}  ../outside.py\n", encoding="utf-8")
    (vendor / "UPSTREAM_COMMIT").write_text(EXPECTED_DPPO_COMMIT + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="escapes vendor root"):
        verify_vendor_snapshot(vendor)


def test_vendor_snapshot_rejects_modified_content(tmp_path: Path):
    vendor = tmp_path / "dppo_official_cc7234a"
    vendor.mkdir()
    source = vendor / "official.py"
    source.write_bytes(b"modified\n")
    (vendor / "SHA256SUMS").write_text(
        f"{'0' * 64}  ./official.py\n", encoding="utf-8"
    )
    (vendor / "UPSTREAM_COMMIT").write_text(EXPECTED_DPPO_COMMIT + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="digest mismatch"):
        verify_vendor_snapshot(vendor)
