from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.contracts import ActionBundle, ActionSpec
from scripts.routea_causal_diagnostics.contracts import DiagnosticFailure
from scripts.routea_causal_diagnostics import runtime as causal_runtime
from scripts.routea_causal_diagnostics.runtime import (
    CheckpointContract,
    compare_checkpoint_contracts,
    install_read_only_tripwires,
    paired_forward,
    verify_forward_pairing,
)


def _contract(label, update, lora_value=0.0):
    lora_a = torch.tensor([0.25 + lora_value])
    lora_b = torch.tensor([lora_value])
    return CheckpointContract(
        label=label,
        update=update,
        source_set_hash="a" * 64,
        actor_state={
            "obs_encoder.weight": torch.tensor([1.0]),
            "denoiser.lora_a": lora_a.clone(),
            "denoiser.lora_b": lora_b.clone(),
        },
        lora_state={
            "denoiser.lora_a": lora_a.clone(),
            "denoiser.lora_b": lora_b.clone(),
        },
        runtime_contract={"action_spec": ActionSpec.e5().content_hash, "dtype": "float32"},
        checkpoint_identity={"label": label, "update": update},
    )


def _matrix():
    return [
        _contract("bc", 0),
        _contract("u0", 0),
        _contract("seed1-u8", 8, 0.1),
        _contract("seed2-u8", 8, 0.2),
        _contract("seed3-u32", 32, 0.3),
    ]


def test_non_lora_tensor_or_runtime_difference_blocks_forward():
    matrix = _matrix()
    matrix[-1].actor_state["obs_encoder.weight"][0] += 1
    report = compare_checkpoint_contracts(matrix)
    assert report["status"] == "CHECKPOINT_CONTRACT_MISMATCH"
    assert any(item.get("tensor") == "obs_encoder.weight" for item in report["mismatches"])


def test_zero_effect_lora_accepts_random_a_and_requires_exact_zero_b():
    state = {
        "linear.lora_a": torch.tensor([0.25, -0.5]),
        "linear.lora_b": torch.zeros(2),
        "conv.lora_a": torch.tensor([0.75]),
        "conv.lora_b": torch.zeros(1),
    }
    assert causal_runtime.validate_zero_effect_lora_state(state) == {
        "adapter_count": 2,
        "lora_a_tensor_count": 2,
        "lora_b_tensor_count": 2,
        "status": "ZERO_EFFECT_LORA",
    }

    state["conv.lora_b"][0] = 1e-12
    with pytest.raises(DiagnosticFailure, match="nonzero lora_b"):
        causal_runtime.validate_zero_effect_lora_state(state)

    state["conv.lora_b"].zero_()
    state["linear.lora_a"][0] = float("nan")
    with pytest.raises(DiagnosticFailure, match="non-finite LoRA tensor"):
        causal_runtime.validate_zero_effect_lora_state(state)


def test_u0_requires_exact_fresh_lora_identity_and_zero_effect():
    matrix = _matrix()
    matrix[1].lora_state["denoiser.lora_a"][0] += 1e-3
    assert compare_checkpoint_contracts(matrix)["status"] == "CHECKPOINT_CONTRACT_MISMATCH"

    matrix = _matrix()
    matrix[1].lora_state["denoiser.lora_b"][0] = 1e-12
    report = compare_checkpoint_contracts(matrix)
    assert report["status"] == "CHECKPOINT_CONTRACT_MISMATCH"
    assert any(item["field"] == "lora_zero_effect" for item in report["mismatches"])


def test_comparable_matrix_passes_and_only_lora_differs():
    report = compare_checkpoint_contracts(_matrix())
    assert report["status"] == "COMPARABLE"
    assert report["lora_key_count"] == 2
    assert report["non_lora_tensor_count"] == 1


def test_comparable_matrix_accepts_individually_verified_config_folded_source_hashes():
    matrix = [
        replace(contract, source_set_hash=f"{index + 1:064x}")
        for index, contract in enumerate(_matrix())
    ]
    report = compare_checkpoint_contracts(matrix)
    assert report["status"] == "COMPARABLE"
    assert [item["source_set_hash"] for item in report["contracts"]] == [
        contract.source_set_hash for contract in matrix
    ]


def test_reconstruct_executed_source_hash_uses_config_admission_manifest(
    tmp_path, monkeypatch
):
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints import identity

    config_path = tmp_path / "seed2.yaml"
    config_path.write_text(
        "admission_manifest: data/admission/manifest.json\nrun:\n  seed: 2\n",
        encoding="utf-8",
    )
    captured = {}

    def fake_build(root, config, manifest):
        captured.update(root=root, seed=int(config.run.seed), manifest=manifest)
        return {"content_hash": "c" * 64}

    monkeypatch.setattr(identity, "build_executed_source_manifest", fake_build)
    assert causal_runtime._reconstruct_executed_source_hash(tmp_path, config_path) == "c" * 64
    assert captured == {
        "root": tmp_path.resolve(),
        "seed": 2,
        "manifest": "data/admission/manifest.json",
    }


def test_reconstruct_executed_source_hash_rejects_missing_manifest(tmp_path):
    config_path = tmp_path / "missing.yaml"
    config_path.write_text("run:\n  seed: 1\n", encoding="utf-8")
    with pytest.raises(DiagnosticFailure, match="config has no admission_manifest"):
        causal_runtime._reconstruct_executed_source_hash(tmp_path, config_path)


class FakeOptimizer:
    def __init__(self):
        self.state = {"counter": torch.tensor([0])}

    def state_dict(self):
        return self.state

    def step(self):
        self.state["counter"] += 1


class FakeSampler:
    def __init__(self, mutate_initial=False):
        self.mutate_initial = mutate_initial

    def sample(self, condition, *, deterministic):
        chain = torch.randn(1, 3, 16, 11)
        if self.mutate_initial:
            # Goal-body x is column 3 of each flattened frame.  Injecting it
            # into the initial chain simulates an invalid sampler that does not
            # preserve common random numbers across goal variants.
            chain[:, 0] += float(condition["state"][0, 3])
        full = chain[:, -1]
        return SimpleNamespace(
            normalized_chain=chain,
            action=ActionBundle.from_full(full, ActionSpec.e5()),
        )


class FakeAdapter:
    def encode(self, observation, metadata, *, episode_identity_hash, decision_index):
        value = torch.as_tensor(observation["agent_pos"], dtype=torch.float32)
        return SimpleNamespace(
            condition={"state": value[-3:].reshape(1, -1)},
            provenance=SimpleNamespace(policy_input_hash="input-hash"),
        )

    def verify(self, encoded):
        assert encoded.condition["state"].shape[0] == 1


def fake_system(*, mutate_initial=False):
    model = torch.nn.Linear(2, 2)
    critic = torch.nn.Linear(2, 1)
    return SimpleNamespace(
        bc_runtime=SimpleNamespace(policy=SimpleNamespace(model=model)),
        lora_bundle=SimpleNamespace(adapter_state_dict=lambda: {"lora_A": torch.zeros(1)}),
        critic=critic,
        actor_optimizer=FakeOptimizer(),
        critic_optimizer=FakeOptimizer(),
        trainer=SimpleNamespace(counters=SimpleNamespace(updates=0), update=lambda: None),
        rollout_sampler=FakeSampler(mutate_initial=mutate_initial),
        observation_adapter=FakeAdapter(),
    )


def observation():
    return {
        "agent_pos": np.zeros((8, 9), dtype=np.float32),
        "point_cloud": np.zeros((8, 4, 3), dtype=np.float32),
        "pointcloud_meta_history": (),
        "episode_identity_hash": "episode",
        "decision_index": 0,
    }


def variants():
    result = []
    for index, name in enumerate(("original", "plus", "minus")):
        state = np.zeros((8, 9), dtype=np.float32)
        state[:, 3] = index
        result.append({"name": name, "agent_pos": state})
    return result


def test_variants_restore_same_rng_and_initial_chain():
    records = paired_forward(fake_system(), observation(), variants(), policy_seed=17)
    assert len({row.initial_chain_sha256 for row in records}) == 1
    assert len({row.post_call_rng_sha256 for row in records}) == 1
    assert verify_forward_pairing(records)["status"] == "VALID_NOISE_PAIRING"


def test_initial_chain_mismatch_is_rejected():
    records = paired_forward(
        fake_system(mutate_initial=True), observation(), variants(), policy_seed=17
    )
    with pytest.raises(DiagnosticFailure, match="INVALID_NOISE_PAIRING"):
        verify_forward_pairing(records)


def test_optimizer_and_trainer_calls_raise_and_state_verifies():
    system = fake_system()
    guard = install_read_only_tripwires(system)
    with pytest.raises(DiagnosticFailure, match="FORBIDDEN_TRAINING_CALL"):
        system.actor_optimizer.step()
    with pytest.raises(DiagnosticFailure, match="FORBIDDEN_TRAINING_CALL"):
        system.trainer.update()
    report = guard.verify()
    assert report["status"] == "READ_ONLY_VERIFIED"
    assert report["forbidden_call_count"] == 2


def test_state_mutation_is_detected():
    system = fake_system()
    guard = install_read_only_tripwires(system)
    with torch.no_grad():
        system.critic.weight[0, 0] += 1
    with pytest.raises(DiagnosticFailure, match="INVALID_STATE_MUTATION"):
        guard.verify()
