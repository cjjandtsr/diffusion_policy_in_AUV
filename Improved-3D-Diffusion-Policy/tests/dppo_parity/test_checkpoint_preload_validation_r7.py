"""R7-6B (guide R7-6 Step 5): resume load order and preload validation.

The formal load order is:
  read header/schema only
  → validate runtime identity / vendor / config / contract / manifest
  → construct policy + registered LoRA + critic + optimizers
  → load policy/LoRA/critic/optimizer/scheduler/scaler
  → restore sampler/env/active slots/counters/writer cursor
  → restore all RNG states LAST
  → next operation must be the exact next sampling operation

These tests prove the preload-validation gates (lora_name_order drift,
actor_optimizer_param_names drift, non-empty owned_generators, v6
contract) hard-fail BEFORE any tensor load, that the RNG restore is
ordered LAST, and that the restore returns the v6 axes the
same-seed comparator consumes.
"""

import dataclasses
import os
from typing import Optional

import pytest
import torch

from diffusion_policy_3d.rl.dppo_uav.checkpoint import (
    SCHEMA_VERSION,
    CheckpointBuilder,
    CheckpointData,
    CheckpointError,
    RuntimeIdentity,
    config_hash_of,
)
from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.critic import (
    PolicyObsEncoder,
    ValueCritic,
)
from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
    DEFAULT_ROLE_TARGETS,
    EpisodeTaskSampler,
    EpisodeTaskSpec,
)
from diffusion_policy_3d.rl.dppo_uav.optimizer_contract import (
    resolved_optimizer_config,
)
from diffusion_policy_3d.rl.dppo_uav.train_loop import (
    _rng_states,
    restore_rng_states,
    restore_training_state,
)

from fake_policy import FakePolicy
from mock_uav_env import MockUAVEnv

OFFICIAL_COMMIT = "cc7234ad7ff39a8f32de3af903606723a16f0648"


def _checkpoint(sampler: Optional[EpisodeTaskSampler] = None,
                **overrides) -> CheckpointData:
    """A fully-specified v6 checkpoint that restores cleanly.

    With a live sampler, sampler_state is taken from sampler.state_dict()
    so the sampler-identity gate (base_seed + spec_set_hash) and the
    SEED-RESUME load both accept it; without one, a minimal placeholder
    keeps the schema serializable for the v6-contract-only tests.
    """
    layer = {"lr": 1e-3, "world_assets_hash": "world456"}
    sampler_state = (
        sampler.state_dict()
        if sampler is not None
        else {
            "schema": "episode_task_sampler/v1", "base_seed": 0,
            "role_sampler_rng": "cGFja2xlLWJhc2U2NA==",
            "cursors": {}, "global_attempt_index": 1,
            "global_episode_index": 1, "global_completed_index": 0,
            "task_spec_set_hash": "",
        }
    )
    kwargs = dict(
        schema_version=SCHEMA_VERSION,
        official_dppo_commit=OFFICIAL_COMMIT,
        project_git_commit="n/a",
        project_tree_hash="n/a",
        source_file_hashes={},
        contract_serialized=b"{}",
        contract_hash="cafe",
        manifest_hash="mf456",
        bc_checkpoint_hash="bc0123",
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
        actor_optimizer_state={"state": {}, "param_groups": []},
        value_optimizer_state={"state": {}, "param_groups": []},
        lr_scheduler_state={
            "method": "none", "version": "v1",
            "actor": {"step_count": 1}, "critic": {"step_count": 1},
        },
        sampling_std_state={
            "adaptive": False,
            "min_sampling_denoising_std": 0.1,
            "min_logprob_denoising_std": 0.1,
        },
        reward_norm_state={
            "version": "v1.3", "source": "env", "method": "none",
            "params": {"state_version": "v1", "step_count": 1},
        },
        rng_states={"python": "py", "numpy": "np", "torch": "th",
                    "cuda": "cu", "env": ""},
        numerical_mode_state={
            "allow_tf32_matmul": False, "allow_tf32_cudnn": False,
            "deterministic_algorithms": True,
            "cudnn_benchmark": False, "cudnn_deterministic": True,
            "cublas_workspace_config": ":4096:8",
        },
        environment_rng_state={
            "seed_source": "sampler_state",
            "python_hash_seed": os.environ.get("PYTHONHASHSEED", ""),
            "restorable": True,
        },
        sampler_state=sampler_state,
        outer_iteration=1,
        ppo_update_count=1,
        optimizer_step_count=1,
        environment_transition_count=12,
        completed_episode_count=1,
        wall_clock_timestamp="2026-08-24T00:00:00",
        exact_resolved_config={"lr": 1e-3, "config_layer": layer},
        lora_name_order=[],
        actor_optimizer_param_names=[],
        checkpoint_phase="quiescent_post_update",
        pending_async_work=False,
        observation_window_provenance={
            "window_mode": "latest3_gated", "passes": True,
        },
        next_value_provenance={
            "next_value_source": "critic(each_transition.next_obs)",
            "next_obs_count": 12, "next_value_count": 12,
            "chunk_size": 64, "next_obs_hash": "n1",
            "next_value_hash": "v1", "interleaved_episode_count": 1,
        },
        official_vendor_commit="",
        vendor_manifest_hash="",
        official_runtime_source_files={},
        owned_generators={},
        # Consistent with outer_iteration=1: the metrics writer cursor is
        # the records committed BEFORE this batch (batch-1) — the R7-6
        # Step 9 drift gate demands cursor == outer_iteration - 1.
        metrics_writer_cursor=0,
        voided_episode_count=0,
    )
    kwargs.update(overrides)
    return CheckpointData(**kwargs)


def _fixtures(tmp_path):
    contract = RLSystemContract()
    policy = FakePolicy(contract)
    critic = ValueCritic(contract=contract, encoder=PolicyObsEncoder(policy))
    actor_opt = torch.optim.AdamW(policy.model.parameters(), lr=1e-4)
    critic_opt = torch.optim.AdamW(critic.value_head_parameters(), lr=1e-4)
    specs = [
        EpisodeTaskSpec(
            world_uid=f"mock_{role}", obstacle_json="{}", role=role,
            split="rl_train", admission_id=f"mock__{role}",
            generator_version="mock", manifest_entry_hash=f"hash_{role}",
        )
        for role in ("anticipatory", "dense_recovery",
                     "support_expansion", "yaw_lateral_recovery")
    ]
    sampler = EpisodeTaskSampler(specs, DEFAULT_ROLE_TARGETS, base_seed=0)
    return contract, policy, critic, actor_opt, critic_opt, sampler


def _runtime_identity(data: CheckpointData) -> RuntimeIdentity:
    layer = (data.exact_resolved_config or {}).get("config_layer", {})
    return RuntimeIdentity(
        official_dppo_commit=data.official_dppo_commit,
        project_git_commit=data.project_git_commit,
        project_tree_hash=data.project_tree_hash,
        contract_hash=data.contract_hash,
        manifest_hash=data.manifest_hash,
        bc_checkpoint_hash=data.bc_checkpoint_hash,
        world_assets_hash=layer.get("world_assets_hash"),
        config_hash=data.config_hash,
        source_file_hashes=data.source_file_hashes,
    )


def _restore_ready(data: CheckpointData, policy, critic, actor_opt,
                   critic_opt) -> CheckpointData:
    """Replace the fixture's placeholder state payloads with REAL states
    from the live objects (mirrors test_r3_rework_negative._restore_ready).

    The positive restore paths strict-load every state dict — the fake
    payloads crash instead of restoring: {"w": [1.0]} fails strict key
    typing, the fake value state is not MLP-shaped, and the fake
    rng_states strings fail bytes.fromhex. R6-E §11.6.2: the frozen
    resolved optimizer contract (production resolver over the LIVE
    optimizers) rides the config layer, and config_hash is recomputed
    over the updated layer — the fixture mirrors what production saves.
    """
    layer = dict(data.exact_resolved_config["config_layer"])
    layer["optimizer_contract"] = {
        "actor": resolved_optimizer_config("actor", actor_opt).to_dict(),
        "value": resolved_optimizer_config("value", critic_opt).to_dict(),
    }
    return dataclasses.replace(
        data,
        policy_state=policy.state_dict(),
        value_state=critic.value_state_dict(),
        actor_optimizer_state=actor_opt.state_dict(),
        value_optimizer_state=critic_opt.state_dict(),
        rng_states=_rng_states(),
        exact_resolved_config={
            "config_layer": layer, "episode_metadata": {},
        },
        config_hash=config_hash_of(layer),
    )


def _restore(path, policy, critic, actor_opt, critic_opt, sampler,
             runtime_identity, **kwargs):
    return restore_training_state(
        str(path), policy, critic, {}, actor_opt, critic_opt,
        runtime_identity=runtime_identity, task_sampler=sampler,
        **kwargs,
    )


def test_restore_returns_v6_axes(tmp_path):
    """The v6 axes ride the restore return so the same-seed comparator
    can compare them across control/resume runs."""
    contract, policy, critic, a_opt, c_opt, sampler = _fixtures(tmp_path)
    data = _restore_ready(
        _checkpoint(sampler), policy, critic, a_opt, c_opt)
    path = tmp_path / "ckpt.pt"
    CheckpointBuilder.save(data, path)
    out = _restore(path, policy, critic, a_opt, c_opt, sampler,
                   _runtime_identity(data))
    assert out["metrics_writer_cursor"] == 0
    assert out["observation_window_provenance"]["passes"] is True
    assert out["next_value_provenance"]["next_obs_count"] == 12
    assert out["lora_name_order"] == []
    assert out["actor_optimizer_param_names"] == []
    assert out["owned_generators"] == {}
    assert out["voided_episode_count"] == 0


def test_rng_restored_after_sampler(tmp_path, monkeypatch):
    """Step 5: 'restore all RNG states LAST' — the sampler/env/counter
    state is restored BEFORE the RNG snapshot, so the next operation
    after restore is the exact next sampling operation with the exact
    RNG state of the checkpointing world."""
    contract, policy, critic, a_opt, c_opt, sampler = _fixtures(tmp_path)
    data = _restore_ready(
        _checkpoint(sampler), policy, critic, a_opt, c_opt)
    path = tmp_path / "ckpt.pt"
    CheckpointBuilder.save(data, path)

    order = []
    real_restore = restore_rng_states
    monkeypatch.setattr(
        "diffusion_policy_3d.rl.dppo_uav.train_loop.restore_rng_states",
        lambda states: (order.append("rng"), real_restore(states)),
    )
    real_load = sampler.load_state_dict
    monkeypatch.setattr(
        sampler, "load_state_dict",
        lambda state: (order.append("sampler"), real_load(state)),
    )
    _restore(path, policy, critic, a_opt, c_opt, sampler,
             _runtime_identity(data))
    assert order == ["sampler", "rng"], (
        f"restore order must restore the sampler BEFORE the RNG snapshot "
        f"(guide R7-6 Step 5), got {order}")


def test_lora_name_order_drift_rejected_before_tensor_load(tmp_path):
    """Step 9 negative: a checkpoint whose lora_name_order does not match
    its own lora_state keys is refused BEFORE any policy tensor load."""
    contract, policy, critic, a_opt, c_opt, sampler = _fixtures(tmp_path)
    # drift: checkpoint claims lora_name_order but its own lora_state={}
    data = _restore_ready(
        _checkpoint(sampler, lora_name_order=["lora_0"]),
        policy, critic, a_opt, c_opt)
    path = tmp_path / "ckpt.pt"
    CheckpointBuilder.save(data, path)

    calls = []

    def spy(state, strict=False):
        calls.append(1)
        return [], []

    policy.load_state_dict = spy
    with pytest.raises(CheckpointError, match="lora_name_order"):
        _restore(path, policy, critic, a_opt, c_opt, sampler,
                 _runtime_identity(data))
    assert calls == [], (
        "policy tensor load must stay untouched on lora_name_order drift")


def test_actor_optimizer_param_names_drift_rejected(tmp_path):
    """Step 9 negative: a checkpoint whose actor_optimizer_param_names
    does not match the live wiring is refused before any optimizer
    tensor load. (Fixture world has no LoRA bundle -> live mapping is
    []; a checkpoint claiming names is a drift.)"""
    contract, policy, critic, a_opt, c_opt, sampler = _fixtures(tmp_path)
    data = _restore_ready(
        _checkpoint(sampler, actor_optimizer_param_names=[
            "lora_0.lora_A", "lora_0.lora_B"]),
        policy, critic, a_opt, c_opt)
    path = tmp_path / "ckpt.pt"
    CheckpointBuilder.save(data, path)
    with pytest.raises(CheckpointError, match="actor_optimizer_param_names"):
        _restore(path, policy, critic, a_opt, c_opt, sampler,
                 _runtime_identity(data))


def test_owned_generators_nonempty_rejected(tmp_path):
    """Step 9 negative: a checkpoint claiming non-empty owned Generators
    cannot be restored by a world that owns none — hard-fail, never a
    silent partial RNG continuation."""
    contract, policy, critic, a_opt, c_opt, sampler = _fixtures(tmp_path)
    data = _restore_ready(
        _checkpoint(sampler, owned_generators={"synth": "deadbeef"}),
        policy, critic, a_opt, c_opt)
    path = tmp_path / "ckpt.pt"
    CheckpointBuilder.save(data, path)
    with pytest.raises(CheckpointError, match="owned_generators"):
        _restore(path, policy, critic, a_opt, c_opt, sampler,
                 _runtime_identity(data))


def test_v6_contract_violation_rejected_before_tensor_load(tmp_path):
    """The v6 quiescent-boundary gate fires before ANY policy load — the
    spy proof, mirroring the identity-gate spy."""
    contract, policy, critic, a_opt, c_opt, sampler = _fixtures(tmp_path)
    data = _restore_ready(
        _checkpoint(sampler, checkpoint_phase="mid_rollout"),
        policy, critic, a_opt, c_opt)
    path = tmp_path / "ckpt.pt"
    CheckpointBuilder.save(data, path)

    calls = []

    def spy(state, strict=False):
        calls.append(1)
        return [], []

    policy.load_state_dict = spy
    with pytest.raises(CheckpointError, match="quiescent_post_update"):
        _restore(path, policy, critic, a_opt, c_opt, sampler,
                 _runtime_identity(data))
    assert calls == [], (
        "policy tensor load must stay untouched on v6 contract violation")


# ── R7-6 Step 9: interruption negative tests (guide: every interruption
#    class hard-fails BEFORE any tensor load / rollout continuation) ──
#    Already covered above: LoRA name drift, actor optimizer param-name
#    drift, non-empty owned_generators, v6 contract violation. The seven
#    below close the rest of the list: interrupted tmp write, LoRA
#    checksum, missing active slot, metrics cursor drift, wrong config,
#    wrong manifest, wrong vendor.


def _spy_load(policy):
    calls = []

    def spy(state, strict=False):
        calls.append(1)
        return [], []

    policy.load_state_dict = spy
    return calls


def test_interrupted_tmp_sibling_rejected_before_tensor_load(tmp_path):
    """Step 9 negative: a leftover .tmp sibling (atomic save interrupted
    between tmp-write and os.replace) means the final file may be stale
    or partial — the formal load refuses the ambiguous state BEFORE any
    tensor load."""
    contract, policy, critic, a_opt, c_opt, sampler = _fixtures(tmp_path)
    data = _restore_ready(
        _checkpoint(sampler), policy, critic, a_opt, c_opt)
    path = tmp_path / "ckpt.pt"
    CheckpointBuilder.save(data, path)
    (tmp_path / "ckpt.pt.tmp").write_bytes(b"partial garbage")

    calls = _spy_load(policy)
    with pytest.raises(CheckpointError, match="tmp"):
        _restore(path, policy, critic, a_opt, c_opt, sampler,
                 _runtime_identity(data))
    assert calls == [], (
        "policy tensor load must stay untouched on an interrupted-save "
        "leftover (the final file's freshness cannot be verified)")


def test_lora_checksum_drift_rejected_before_tensor_load(tmp_path):
    """Step 9 negative: the checkpoint's declared LoRA checksum must
    equal the checksum recomputed over its own lora_state tensors — a
    saved string is never proof of the parameters it claims to bind."""
    contract, policy, critic, a_opt, c_opt, sampler = _fixtures(tmp_path)
    base = _restore_ready(
        _checkpoint(sampler), policy, critic, a_opt, c_opt)
    data = dataclasses.replace(
        base,
        lora_identity={**base.lora_identity, "lora_checksum": "deadbeef"},
    )
    path = tmp_path / "ckpt.pt"
    CheckpointBuilder.save(data, path)

    calls = _spy_load(policy)
    with pytest.raises(CheckpointError, match="checksum"):
        _restore(path, policy, critic, a_opt, c_opt, sampler,
                 _runtime_identity(data))
    assert calls == [], (
        "policy tensor load must stay untouched on LoRA checksum drift")


def test_missing_active_slot_field_rejected_before_tensor_load(tmp_path):
    """Step 9 negative: a v6 checkpoint whose payload LACKS the
    active_slot_states axis (every v6 save writes it) is a
    tampered/truncated payload — the formal load refuses it BEFORE any
    tensor load (a silent slot-less continuation would diverge)."""
    contract, policy, critic, a_opt, c_opt, sampler = _fixtures(tmp_path)
    data = _restore_ready(
        _checkpoint(sampler), policy, critic, a_opt, c_opt)
    path = tmp_path / "ckpt.pt"
    CheckpointBuilder.save(data, path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert "active_slot_states" in payload  # v6 save always writes it
    del payload["active_slot_states"]
    tampered = tmp_path / "tampered.pt"
    torch.save(payload, tampered)

    calls = _spy_load(policy)
    with pytest.raises(CheckpointError, match="active_slot_states"):
        _restore(tampered, policy, critic, a_opt, c_opt, sampler,
                 _runtime_identity(data))
    assert calls == [], (
        "policy tensor load must stay untouched when the active-slot axis "
        "is missing from the payload")


def test_metrics_cursor_drift_rejected_before_tensor_load(tmp_path):
    """Step 9 negative: the metrics writer cursor (records committed
    BEFORE this batch) must equal outer_iteration-1 — a drifted cursor
    would make the resumed writer renumber/desync metrics.jsonl."""
    contract, policy, critic, a_opt, c_opt, sampler = _fixtures(tmp_path)
    data = _restore_ready(
        _checkpoint(sampler, metrics_writer_cursor=2),  # outer=1 -> 0
        policy, critic, a_opt, c_opt)
    path = tmp_path / "ckpt.pt"
    CheckpointBuilder.save(data, path)

    calls = _spy_load(policy)
    with pytest.raises(CheckpointError, match="metrics_writer_cursor"):
        _restore(path, policy, critic, a_opt, c_opt, sampler,
                 _runtime_identity(data))
    assert calls == [], (
        "policy tensor load must stay untouched on metrics cursor drift")


def test_wrong_config_hash_rejected_before_tensor_load(tmp_path):
    """Step 9 negative: a checkpoint whose config_hash does not match the
    running side is refused BEFORE any tensor load (P0-2 hash gate)."""
    contract, policy, critic, a_opt, c_opt, sampler = _fixtures(tmp_path)
    base = _restore_ready(
        _checkpoint(sampler), policy, critic, a_opt, c_opt)
    data = dataclasses.replace(base, config_hash="tampered-hash")
    path = tmp_path / "ckpt.pt"
    CheckpointBuilder.save(data, path)

    calls = _spy_load(policy)
    with pytest.raises(CheckpointError, match="config_hash"):
        _restore(path, policy, critic, a_opt, c_opt, sampler,
                 _runtime_identity(base))  # runtime side = the ORIGINAL
    assert calls == [], (
        "policy tensor load must stay untouched on config hash drift")


def test_wrong_manifest_hash_rejected_before_tensor_load(tmp_path):
    """Step 9 negative: a checkpoint whose manifest_hash does not match
    the running side is refused BEFORE any tensor load."""
    contract, policy, critic, a_opt, c_opt, sampler = _fixtures(tmp_path)
    base = _restore_ready(
        _checkpoint(sampler), policy, critic, a_opt, c_opt)
    data = dataclasses.replace(base, manifest_hash="tampered-mf")
    path = tmp_path / "ckpt.pt"
    CheckpointBuilder.save(data, path)

    calls = _spy_load(policy)
    with pytest.raises(CheckpointError, match="manifest_hash"):
        _restore(path, policy, critic, a_opt, c_opt, sampler,
                 _runtime_identity(base))
    assert calls == [], (
        "policy tensor load must stay untouched on manifest hash drift")


def test_wrong_vendor_rejected_before_tensor_load(tmp_path):
    """Step 9 negative: a checkpoint claiming a vendor binding that does
    not match the running side's frozen vendor basis is refused BEFORE
    any tensor load (R7-1 fail-closed — never an unverified resume)."""
    contract, policy, critic, a_opt, c_opt, sampler = _fixtures(tmp_path)
    base = _restore_ready(
        _checkpoint(sampler), policy, critic, a_opt, c_opt)
    data = dataclasses.replace(
        base,
        official_vendor_commit="tampered-commit",
        vendor_manifest_hash="tampered-manifest",
    )
    path = tmp_path / "ckpt.pt"
    CheckpointBuilder.save(data, path)

    calls = _spy_load(policy)
    with pytest.raises(CheckpointError, match="official_vendor_commit"):
        _restore(
            path, policy, critic, a_opt, c_opt, sampler,
            _runtime_identity(base),
            vendor_identity={
                "official_vendor_commit": OFFICIAL_COMMIT,
                "vendor_manifest_hash": "mf456",
            })
    assert calls == [], (
        "policy tensor load must stay untouched on vendor identity drift")
