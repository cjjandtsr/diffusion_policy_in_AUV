"""R7-6A (guide R7-6 Step 1-2): checkpoint schema v6 required fields,
the v6 contract validation, vendor identity, and the v5 compatibility
boundary (v5 read-only audit vs v6-only formal resume).

The v6 field set extends v5 with:
  lora_name_order + actor_optimizer_param_names   (LoRA/optimizer mapping)
  checkpoint_phase = "quiescent_post_update" + pending_async_work = false
  observation_window_provenance + next_value_provenance
  official_vendor_commit + vendor_manifest_hash
  official_runtime_source_files (double axis of source_file_hashes)
  owned_generators (RNG states beyond the globals)
  metrics_writer_cursor + voided_episode_count   (writer + counters)

Validation order (Step 1): schema / runtime identity / LoRA names / shapes
/ checksum / optimizer mapping are validated BEFORE any tensor load — the
contract validators below are pure (no tensor loading), and
restore_training_state calls them before any state_dict load.
"""

import dataclasses
import json

import pytest
import torch

from diffusion_policy_3d.rl.dppo_uav.checkpoint import (
    SCHEMA_VERSION,
    CheckpointBuilder,
    CheckpointData,
    CheckpointError,
    validate_v6_contract,
    validate_vendor_identity,
)

V6_FIELDS = (
    "lora_name_order",
    "actor_optimizer_param_names",
    "checkpoint_phase",
    "pending_async_work",
    "observation_window_provenance",
    "next_value_provenance",
    "official_vendor_commit",
    "vendor_manifest_hash",
    "official_runtime_source_files",
    "owned_generators",
    "metrics_writer_cursor",
    "voided_episode_count",
)

OFFICIAL_COMMIT = "cc7234ad7ff39a8f32de3af903606723a16f0648"


def _base_data(**overrides) -> CheckpointData:
    layer = {"lr": 1e-3}
    kwargs = dict(
        schema_version=SCHEMA_VERSION,
        official_dppo_commit=OFFICIAL_COMMIT,
        project_git_commit="abc1234",
        project_tree_hash="def5678",
        source_file_hashes={"contracts.py": "abc"},
        contract_serialized=b"{}",
        contract_hash="cafe",
        manifest_hash="beef",
        bc_checkpoint_hash="1234",
        config_hash="hash",
        policy_state={"w": torch.tensor([1.0])},
        value_state={"w": torch.tensor([2.0])},
        lora_state={},
        lora_identity={
            "lora_rank": 0, "lora_alpha": 0.0, "n_adaptors": 0,
            "n_lora_param_tensors": 0,
            "lora_checksum": (
                "e3b0c44298fc1c149afbf4c8996fb924"
                "27ae41e4649b934ca495991b7852b855"
            ),
        },
        actor_optimizer_state={"step": 1},
        value_optimizer_state={"step": 2},
        lr_scheduler_state={
            "method": "none", "version": "v1",
            "actor": {"step_count": 7}, "critic": {"step_count": 7},
        },
        sampling_std_state={
            "adaptive": False,
            "min_sampling_denoising_std": 0.1,
            "min_logprob_denoising_std": 0.1,
        },
        reward_norm_state={
            "version": "v1.3", "source": "env", "method": "none",
            "params": {"state_version": "v1", "step_count": 7},
        },
        rng_states={"python": "py", "numpy": "np", "torch": "th",
                    "cuda": "cu", "env": ""},
        numerical_mode_state={
            "allow_tf32_matmul": False, "allow_tf32_cudnn": False,
            "deterministic_algorithms": True,
        },
        environment_rng_state={
            "seed_source": "sampler_state",
            "python_hash_seed": "7",
            "restorable": True,
        },
        sampler_state={
            "schema": "episode_task_sampler/v1", "base_seed": 7,
            "role_sampler_rng": "cGFja2xlLWJhc2U2NA==",
            "cursors": {"anticipatory": 1}, "global_attempt_index": 3,
            "global_episode_index": 3, "global_completed_index": 2,
            "task_spec_set_hash": "abc",
        },
        outer_iteration=42,
        ppo_update_count=7,
        optimizer_step_count=100,
        environment_transition_count=2500,
        completed_episode_count=10,
        wall_clock_timestamp="2026-08-24T00:00:00",
        exact_resolved_config={"lr": 1e-3, "config_layer": layer},
        # R7-6 v6 fields
        lora_name_order=["lora_0"],
        actor_optimizer_param_names=["lora_0.lora_A", "lora_0.lora_B"],
        checkpoint_phase="quiescent_post_update",
        pending_async_work=False,
        observation_window_provenance={
            "window_valid": True,
            "frame_hashes": ["f1", "f2", "f3"],
        },
        next_value_provenance={"source": "own_next_obs", "obs_hash": "n1"},
        official_vendor_commit=OFFICIAL_COMMIT,
        vendor_manifest_hash="vendor-mf",
        official_runtime_source_files={"contracts.py": "abc"},
        owned_generators={},
        metrics_writer_cursor=3,
        voided_episode_count=2,
    )
    kwargs.update(overrides)
    return CheckpointData(**kwargs)


def test_schema_version_is_v6():
    assert SCHEMA_VERSION == "6"


def test_v6_required_fields_declared():
    declared = {f.name for f in CheckpointData.__dataclass_fields__.values()}
    for name in V6_FIELDS:
        assert name in declared, f"missing v6 field {name}"


def test_build_serializes_all_v6_fields():
    data = _base_data()
    snapshot = CheckpointBuilder.build(data)
    assert snapshot["checkpoint_phase"] == "quiescent_post_update"
    assert snapshot["pending_async_work"] is False
    assert snapshot["lora_name_order"] == ["lora_0"]
    assert snapshot["actor_optimizer_param_names"] == [
        "lora_0.lora_A", "lora_0.lora_B"]
    assert snapshot["official_vendor_commit"] == OFFICIAL_COMMIT
    assert snapshot["vendor_manifest_hash"] == "vendor-mf"
    assert snapshot["metrics_writer_cursor"] == 3
    assert snapshot["voided_episode_count"] == 2
    assert snapshot["owned_generators"] == {}
    assert snapshot["observation_window_provenance"]["window_valid"] is True
    assert snapshot["next_value_provenance"]["source"] == "own_next_obs"
    assert (snapshot["official_runtime_source_files"]
            == snapshot["source_file_hashes"])


def test_save_load_roundtrip_preserves_v6_fields(tmp_path):
    path = tmp_path / "ckpt.pt"
    CheckpointBuilder.save(_base_data(), path)
    loaded = CheckpointBuilder.load(path)
    assert loaded.checkpoint_phase == "quiescent_post_update"
    assert loaded.pending_async_work is False
    assert loaded.lora_name_order == ["lora_0"]
    assert loaded.actor_optimizer_param_names == [
        "lora_0.lora_A", "lora_0.lora_B"]
    assert loaded.official_vendor_commit == OFFICIAL_COMMIT
    assert loaded.vendor_manifest_hash == "vendor-mf"
    assert loaded.owned_generators == {}
    assert loaded.metrics_writer_cursor == 3
    assert loaded.voided_episode_count == 2
    assert loaded.official_runtime_source_files == {"contracts.py": "abc"}


# ---- v6 contract validation (Step 1: before any tensor load) ----

def test_v6_contract_accepts_quiescent_checkpoint():
    validate_v6_contract(_base_data())  # must not raise


def test_v6_contract_rejects_non_quiescent_phase():
    with pytest.raises(CheckpointError, match="quiescent_post_update"):
        validate_v6_contract(_base_data(checkpoint_phase="mid_rollout"))


def test_v6_contract_rejects_pending_async_work():
    with pytest.raises(CheckpointError, match="pending_async_work"):
        validate_v6_contract(_base_data(pending_async_work=True))


def test_v6_contract_rejects_source_files_double_axis_mismatch():
    with pytest.raises(CheckpointError, match="official_runtime_source_files"):
        validate_v6_contract(_base_data(
            official_runtime_source_files={"other.py": "zzz"}))


# ---- vendor identity (Step 1) ----

def test_vendor_identity_matching_passes():
    validate_vendor_identity(
        _base_data(),
        {"official_vendor_commit": OFFICIAL_COMMIT,
         "vendor_manifest_hash": "vendor-mf"},
    )  # must not raise


def test_vendor_commit_mismatch_rejected():
    with pytest.raises(CheckpointError, match="official_vendor_commit"):
        validate_vendor_identity(
            _base_data(),
            {"official_vendor_commit": "wrong",
             "vendor_manifest_hash": "vendor-mf"},
        )


def test_vendor_manifest_hash_mismatch_rejected():
    with pytest.raises(CheckpointError, match="vendor_manifest_hash"):
        validate_vendor_identity(
            _base_data(),
            {"official_vendor_commit": OFFICIAL_COMMIT,
             "vendor_manifest_hash": "different"},
        )


def test_vendor_bound_checkpoint_without_runtime_basis_rejected():
    """Fail-closed: a checkpoint that CLAIMS a vendor binding cannot
    resume when the running side supplies no vendor identity basis."""
    with pytest.raises(CheckpointError, match="vendor"):
        validate_vendor_identity(_base_data(), None)


# ---- v5 compatibility boundary (Step 2) ----

def _v5_payload() -> dict:
    v5 = _base_data()
    return dataclasses.asdict(dataclasses.replace(
        v5,
        schema_version="5",
        lora_name_order=[],
        actor_optimizer_param_names=[],
        checkpoint_phase="",
        pending_async_work=False,
        observation_window_provenance={},
        next_value_provenance={},
        official_vendor_commit="",
        vendor_manifest_hash="",
        official_runtime_source_files={},
        owned_generators={},
        metrics_writer_cursor=0,
        voided_episode_count=0,
    ))


def test_v5_checkpoint_loadable_via_load_for_audit(tmp_path):
    """v5 checkpoints stay readable for read-only audit/eval."""
    path = tmp_path / "v5.pt"
    torch.save(_v5_payload(), path)
    loaded = CheckpointBuilder.load_for_audit(path)
    assert loaded.schema_version == "5"
    assert loaded.ppo_update_count == 7
    assert loaded.sampler_state["base_seed"] == 7


def test_v5_checkpoint_rejected_for_formal_resume(tmp_path):
    """Formal resume is v6-only — a v5 checkpoint hard-fails at load."""
    path = tmp_path / "v5.pt"
    torch.save(_v5_payload(), path)
    with pytest.raises(CheckpointError, match="unsupported resume schema"):
        CheckpointBuilder.load(path)
