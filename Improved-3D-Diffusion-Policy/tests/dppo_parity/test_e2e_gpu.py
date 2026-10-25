"""T5 e2e (G1 gate): single-batch GPU forward+backward on the real BC ckpt.

Evidence run — closes the loop on GPU with the frozen phase1a policy + LoRA +
mock env in no-update and one-update modes: rollout sampling, buffer, GAE,
ppo_update forward+backward, checkpoint save, paired eval artifact validated
with the runtime contract hash. Skipped when CUDA or the BC checkpoint is
missing.
"""

import os

import numpy as np
import pytest
import torch

from diffusion_policy_3d.policy.lora_adapters import insert_lora
from diffusion_policy_3d.rl.dppo_uav.checkpoint import (
    CheckpointBuilder,
    contract_hash_of,
)
from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.capability_metrics import OfflineDensity
from diffusion_policy_3d.rl.dppo_uav.critic import PolicyObsEncoder, ValueCritic
from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
    DEFAULT_ROLE_TARGETS,
    EpisodeTaskSampler,
    EpisodeTaskSpec,
)
from diffusion_policy_3d.rl.dppo_uav.evaluator import (
    freeze_paired_bank,
    validate_eval_artifact,
)
from diffusion_policy_3d.rl.dppo_uav.policy_loader import (
    extract_lora_state,
    load_ddpm_policy,
    restore_lora_state,
)
from diffusion_policy_3d.rl.dppo_uav.train_loop import LoopConfig, run_e2e_batch
from diffusion_policy_3d.rl.dppo_uav.trainer import TrainerConfig

from mock_uav_env import MockUAVEnv


pytestmark = pytest.mark.gpu

_DEFAULT_CKPT = (
    "data/outputs/2026.05.17/13.36.20_v8_phase2_sweep_lambda0.5/"
    "checkpoints/latest.ckpt"
)


def _ckpt_path() -> str:
    return os.environ.get("DPPO_BC_CKPT", _DEFAULT_CKPT)


def _contract() -> RLSystemContract:
    return RLSystemContract()


def _task_specs() -> list:
    """One spec per role — the shared EpisodeTaskSampler (R2.1) hard-fails
    on an empty role bucket."""
    return [
        EpisodeTaskSpec(
            world_uid=f"mock_{role}",
            obstacle_json="{}",
            role=role,
            split="rl_train",
            admission_id=f"mock__{role}",
            generator_version="mock",
            manifest_entry_hash=f"hash_{role}",
        )
        for role in ("anticipatory", "dense_recovery",
                     "support_expansion", "yaw_lateral_recovery")
    ]


def _runtime(device: str) -> dict:
    return {
        "timestamp": "2026-08-06T00:00:00Z",
        "device": device,
        "reward_version": "v1.3",
        "bc_checkpoint_hash": "bc0123",
        "world_assets_hash": "wa456",
        "manifest_hash": "mf456",
        "source_file_hashes": {},
        "project_git_commit": "n/a",
        "project_tree_hash": "n/a",
    }


@pytest.fixture(scope="module")
def gpu_policy():
    """Real frozen BC policy + LoRA on CUDA (loaded once for the module).

    Returns (policy, lora_info) — lora_info is the insert_lora return dict:
    the optimizer trains the LIVE lora_params, the checkpoint state comes
    from extract_lora_state(lora_info) (adapter modules are not attached to
    the model, so named_parameters() never sees them).
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    path = _ckpt_path()
    if not os.path.exists(path):
        pytest.skip(f"BC checkpoint not found: {path}")
    policy = load_ddpm_policy(path, device="cuda")
    lora_info = insert_lora(policy.model, r=8, alpha=8)
    return policy, lora_info


def _run(
    policy_lora,
    mode: str,
    output_dir: str,
    eval_n: int = 0,
    max_ep_len: int = 4,
):
    policy, lora_info = policy_lora
    contract = _contract()
    critic = ValueCritic(contract=contract, encoder=PolicyObsEncoder(policy))
    critic = critic.to("cuda")
    cfg = LoopConfig(
        mode=mode,
        n_episodes=1,
        max_ep_len=max_ep_len,
        device="cuda",
        output_dir=output_dir,
        eval_n_episodes=eval_n,
    )
    specs = _task_specs()
    # Paired eval uses a DEDICATED n_envs=1 eval env — never the training
    # env (explicit interface boundary) — replaying the FROZEN bank
    # one entry per eval episode with its own frozen seed. Capability
    # metrics need the offline density field.
    eval_env_kw = {}
    if eval_n > 0:
        eval_env = MockUAVEnv(contract, seed=0)
        bank = freeze_paired_bank(
            eval_env, specs[:eval_n],
            tuple(1001 + i for i in range(eval_n)),
        )
        eval_env_kw = {
            "eval_env": eval_env, "paired_bank": bank,
            "bank_id": "bank-e2e_gpu",  # v3 provenance (validator requires)
            "offline_density": OfflineDensity(
                counts={(x, 0, 2): 5 for x in range(17)}
            ),
        }
    return run_e2e_batch(
        env=MockUAVEnv(contract, seed=0),
        policy=policy,
        critic=critic,
        contract=contract,
        loop_config=cfg,
        trainer_config=TrainerConfig(device="cuda"),
        task_specs=specs,
        actor_optimizer=torch.optim.AdamW(lora_info["lora_params"], lr=1e-4),
        critic_optimizer=torch.optim.AdamW(
            critic.value_head_parameters(), lr=1e-4
        ),  # value head only, never the shared encoder
        lora_info=lora_info,  # live state extracted post-update
        runtime=_runtime("cuda"),
        **eval_env_kw,
        task_sampler=EpisodeTaskSampler(
            specs, DEFAULT_ROLE_TARGETS, base_seed=0
        ),
    )


def test_no_update_gpu(gpu_policy, tmp_path):
    # max_memory_allocated is PROCESS-level peak; earlier GPU tests in the
    # same session pollute it (Gate D 2026-08-22: 11-test run failed at
    # 8.61GB while this test alone peaks <8GB). Reset so the guard asserts
    # THIS test's own peak — guard semantics unchanged.
    torch.cuda.reset_peak_memory_stats()
    result = _run(gpu_policy, "no_update", str(tmp_path))
    assert result["transitions"] > 0
    assert result["gate_summary"]["invalid_count"] == 0
    assert result["metrics"] is None
    assert result["checkpoint_path"] is None
    assert not list(tmp_path.glob("*.pt"))
    assert torch.cuda.max_memory_allocated() < 8 * 1024**3, (
        "must fit in the 8GB RTX 4060"
    )


def test_one_update_gpu(gpu_policy, tmp_path):
    # Same process-level peak pollution guard as test_no_update_gpu
    # (Gate D 2026-08-22).
    torch.cuda.reset_peak_memory_stats()
    # Snapshot all base params (policy.parameters() = everything except
    # LoRA adaptors and the value head) before the update
    policy, lora_info = gpu_policy
    base_before = [
        p.detach().cpu().clone() for p in policy.parameters()
    ]
    # Snapshot the live LoRA params before the update; after one update
    # they must have moved AND the checkpoint must hold the moved (live)
    # state, never the insertion-time snapshot
    lora_before = [
        p.detach().cpu().clone() for p in lora_info["lora_params"]
    ]
    # Three capability roles (anticipatory/dense_recovery/support_expansion)
    # — the capability metrics hard-fail on any missing role (D1).
    result = _run(gpu_policy, "one_update", str(tmp_path), eval_n=3)
    metrics = result["metrics"]
    assert metrics is not None and metrics["update_batches"] >= 1
    assert np.isfinite(metrics["ratio_mean"])
    assert np.isfinite(metrics["approx_kl"])
    assert np.isfinite(metrics["actor_grad_norm"])
    assert np.isfinite(metrics["critic_grad_norm"])
    # checkpoint with the runtime contract hash
    ckpt_files = list(tmp_path.glob("*.pt"))
    assert ckpt_files, "one-update must save a checkpoint"
    data = CheckpointBuilder.load(ckpt_files[0])
    assert data.contract_hash == contract_hash_of(_contract())
    assert data.ppo_update_count == 1
    # Paired eval artifact validates at the call site
    art = result["eval_artifact"]
    validate_eval_artifact(
        art,
        expected_exec_steps=4,
        expected_contract_hash=contract_hash_of(_contract()),
    )
    assert len(art.episodes) == 3
    assert art.lora_checksum
    # Every base param must have update norm == 0 (actor touches LoRA
    # only; critic touches the value head only)
    base_deltas = [
        (p.detach().cpu() - b).norm().item()
        for p, b in zip(policy.parameters(), base_before)
    ]
    assert base_deltas and max(base_deltas) == 0.0, (
        "base params must not move under one-update: "
        f"max update norm = {max(base_deltas)}"
    )
    # The actor update must have moved the LoRA params...
    lora_deltas = [
        (p.detach().cpu() - b).norm().item()
        for p, b in zip(lora_info["lora_params"], lora_before)
    ]
    assert lora_deltas and max(lora_deltas) > 0.0, (
        "one-update must move the LoRA params"
    )
    # ...and the checkpoint must hold the post-update (live) state
    live_state = result["lora_state"]
    assert live_state, "live LoRA state missing from the result"
    ckpt_lora = data.lora_state
    assert set(ckpt_lora) == set(live_state), "checkpoint/live LoRA key mismatch"
    for key in ckpt_lora:
        assert torch.equal(ckpt_lora[key], live_state[key]), (
            f"checkpoint LoRA {key} != live post-update state (N2: stale "
            "snapshot bug)"
        )
    assert torch.cuda.max_memory_allocated() < 8 * 1024**3, (
        "must fit in the 8GB RTX 4060"
    )


def test_lora_restore_roundtrip_gpu(gpu_policy):
    """N2 (ASSUMPTION-9): restore_lora_state must recover the trained
    adaptors exactly — the checkpoint tensors are the only restorable copy
    of the trained actor (adaptors are hook closures invisible to
    policy.state_dict())."""
    _, lora_info = gpu_policy
    trained = extract_lora_state(lora_info)
    assert trained, "no LoRA adaptors found"
    try:
        # zero the live adaptors -> extracted state must read as zeros
        with torch.no_grad():
            for p in lora_info["lora_params"]:
                p.zero_()
        zeroed = extract_lora_state(lora_info)
        assert all(
            torch.count_nonzero(v) == 0 for v in zeroed.values()
        ), "adaptors not actually zeroed"

        # restore the trained snapshot -> extracted state must match exactly
        restore_lora_state(lora_info, trained)
        restored = extract_lora_state(lora_info)
        for key in trained:
            assert torch.equal(restored[key], trained[key]), (
                f"restore round-trip mismatch on {key}"
            )
    finally:
        # never leave the shared module-scoped policy zeroed
        restore_lora_state(lora_info, trained)
