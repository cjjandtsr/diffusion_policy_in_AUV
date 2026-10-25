"""Ensure repo root and tests/dppo_parity are importable."""

import os
import sys
from pathlib import Path

import pytest

# Deterministic CuBLAS requires this env var before the first CUDA
# matmul (torch.use_deterministic_algorithms raises RuntimeError without
# it on CUDA >= 10.2). Set at the earliest test-process point; the formal
# entry also sets it (numerics.py).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# RULING-7 P0-2: the checkpoint env-RNG identity records the running
# PYTHONHASHSEED (restorable=True only for a fixed non-empty seed) and
# the formal restore refuses restorable=False. Pin a fixed seed for the
# test process so save-side build_environment_rng_state records
# restorable=True and resume tests round-trip. Note: setting it here does
# NOT change str-hash randomization (PYTHONHASHSEED is read only at
# interpreter start) — the cross-process tests in
# test_r3_rework3_positive.py explicitly remove it from their subprocess
# env to prove the env RNG no longer depends on hash().
os.environ.setdefault("PYTHONHASHSEED", "0")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# test-local fixtures (fake_policy, mock_uav_env) importable as top-level
TEST_DIR = Path(__file__).resolve().parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

# Production assets for the real-env gates (G3/G5/G6). Env var overrides let
# a CI/dev box point at mirrors without editing the defaults.
# R7-1 (2026-08-22 user ruling): base actor switched to the E5 retrain
# e5_lambda0p5 (V11-fixed collision loss, lambda_col=0.5); the loader
# allowlist (policy_loader.py, schema v2) is bound to this checkpoint's
# SHA256 c1fca4ac…
DEFAULT_BC_CKPT = str(
    REPO_ROOT
    / "data/outputs/2026.08.16/08.53.26_e5_lambda0p5"
    / "checkpoints/latest.ckpt"
)
DEFAULT_MANIFEST = (
    "/home/cjj/AUV/program/reports/routeA_rl_env_v1/calibration"
    "/admission_manifest.json"
)
DEFAULT_WORLDS_TRAIN = "/home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds/train"


def pytest_configure(config):
    """Register the gpu marker (T5 e2e evidence run; deselected on CPU)."""
    config.addinivalue_line(
        "markers", "gpu: requires CUDA + real BC checkpoint (evidence run)"
    )


@pytest.fixture(scope="session", autouse=True)
def _formal_numerical_mode():
    """Apply the formal numerical mode before every test session.

    run_e2e_batch asserts the mode is already configured (single
    configuration source, guide R7-3 Step 1) instead of applying it, so
    every caller — including module-scoped fixtures such as
    real_rollout_buffer that drive run_e2e_batch during setup — must
    configure first. Session scope guarantees the mode is set before any
    module-scoped fixture instantiates. Idempotent; the formal entry
    applies the same mode in preflight.
    """
    from diffusion_policy_3d.rl.dppo_uav.numerics import (
        configure_numerical_mode,
    )
    from diffusion_policy_3d.rl.dppo_uav.train_loop import (
        FORMAL_NUMERICAL_MODE,
    )

    configure_numerical_mode(FORMAL_NUMERICAL_MODE)
    yield


# ═══════════════════════════════════════════════════════════════════
# real-env GPU fixtures (module-scoped: one policy/adapter per gate test)
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def real_policy_lora():
    """BC phase1a policy (frozen loader) + zero-init LoRA on CUDA.

    Returns (policy, lora_info, device). Skipped when CUDA or the BC
    checkpoint is unavailable. R7-1 (2026-08-22 user ruling): the base
    actor is the E5 retrain e5_lambda0p5 (V11-fixed collision loss,
    lambda_col=0.5, sha256 c1fca4ac…) — see DEFAULT_BC_CKPT above.
    """
    import torch

    if not torch.cuda.is_available():
        pytest.skip("gpu marker: CUDA unavailable")
    ckpt = os.environ.get("DPPO_BC_CKPT", DEFAULT_BC_CKPT)
    if not os.path.isfile(ckpt):
        pytest.skip(f"BC checkpoint not found: {ckpt}")

    from diffusion_policy_3d.policy.lora_adapters import insert_lora
    from diffusion_policy_3d.rl.dppo_uav.policy_loader import load_ddpm_policy

    device = "cuda:0"
    policy = load_ddpm_policy(ckpt, device=device)
    lora_info = insert_lora(policy.model, r=8, alpha=8, verbose=False)
    return policy, lora_info, device


@pytest.fixture(scope="module")
def real_adapter():
    """Real UAVVectorEnvAdapter over the PRODUCTION admission pool.

    Returns (adapter, train_specs, contract) — 21 admitted rl_train worlds
    (G3-verified 8/8 gates), each spec carrying role + manifest_entry_hash.
    Skipped when the admission manifest or worlds dir are unavailable.
    """
    manifest = os.environ.get("DPPO_MANIFEST", DEFAULT_MANIFEST)
    worlds = os.environ.get("DPPO_WORLDS_TRAIN", DEFAULT_WORLDS_TRAIN)
    if not os.path.isfile(manifest):
        pytest.skip(f"admission manifest not found: {manifest}")
    if not os.path.isdir(worlds):
        pytest.skip(f"worlds dir not found: {worlds}")

    from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
    from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
        manifest_record_key,
    )
    from diffusion_policy_3d.rl.dppo_uav.integration import (
        build_adapter,
        build_task_specs,
        entry_metadata_hash,
        manifest_train_entries,
    )

    entries = manifest_train_entries(manifest)
    if not entries:
        pytest.skip("admission manifest has no admitted rl_train entries")
    specs, _ = build_task_specs(
        [dict(e, manifest_entry_hash=entry_metadata_hash(e)) for e in entries]
    )
    # Rule-6 records keyed by the composite (world_uid, role,
    # admission_id) — built FROM the specs so the keys always match the
    # registry's admission_id defaults.
    records = {
        manifest_record_key(s.world_uid, s.role, s.admission_id):
            s.manifest_entry_hash
        for s in specs
    }
    contract = RLSystemContract()
    adapter = build_adapter(
        contract,
        worlds_dir=worlds,
        manifest_path=manifest,
        manifest_records=records,
        device="cuda:0",
        max_episode_steps=60,
    )
    return adapter, specs, contract


@pytest.fixture(scope="module")
def real_rollout_buffer(real_policy_lora, real_adapter, tmp_path_factory):
    """Real no_update rollout buffer — old logprobs stored at batch 1.

    One 32-episode no_update batch over the production admission pool
    (the same path the G5 gate test rolls out, without its env.step spy):
    sample_dppo_rollout computes the stored logprobs at rollout time, B=1
    (the behavior batch) with the policy's CFG lambda, written once,
    never recomputed — via the SAME vendor runtime the trainer's update
    recompute runs (chain_logprobs -> OfficialPhase1aFacade). R7-3
    (user ruling 2026-08-23): the adjudicating gate is the effective-PPO
    identity diagnostic on one fixed B=128 minibatch (test
    test_effective_ppo_identity_no_optimizer_step) — the raw per-dim
    cross-shape diff (B=1-stored vs B=128-recomputed) is a recorded
    CROSS_SHAPE_NUMERICAL_DIAGNOSTIC, never the gate.
    The rollout's contract, device and min_logprob_denoising_std are
    attached to the buffer (plain class) so the recomputation uses the
    identical configuration the rollout used.
    """
    import torch

    if not torch.cuda.is_available():
        pytest.skip("gpu marker: CUDA unavailable")

    from diffusion_policy_3d.rl.dppo_uav.critic import (
        PolicyObsEncoder,
        ValueCritic,
    )
    from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
        DEFAULT_ROLE_TARGETS,
        EpisodeTaskSampler,
    )
    from diffusion_policy_3d.rl.dppo_uav.train_loop import (
        LoopConfig,
        run_e2e_batch,
    )
    from diffusion_policy_3d.rl.dppo_uav.trainer import TrainerConfig

    policy, lora_info, device = real_policy_lora
    adapter, specs, contract = real_adapter

    actor_optimizer = torch.optim.Adam(lora_info["lora_params"], lr=1e-3)
    critic = ValueCritic(contract, PolicyObsEncoder(policy)).to(device)
    critic_optimizer = torch.optim.Adam(
        critic.value_head_parameters(), lr=1e-3
    )
    runtime = {
        "device": device,
        "reward_version": "v16",
        "manifest_hash": "r1-identity-gate",
        "project_git_commit": "?",
        "project_tree_hash": "?",
        "source_file_hashes": {},
        "bc_checkpoint_hash": "?",
        "timestamp": "r1-identity-gate",
    }
    loop_config = LoopConfig(
        mode="no_update",
        n_episodes=32,
        max_ep_len=60,
        device=device,
        output_dir=str(tmp_path_factory.mktemp("r1_identity_rollout")),
    )
    # Production optimizer batch 128; likelihood is recomputed at
    # microbatch=1 inside ppo_update, so the stored B=1 rollout logprobs
    # stay reproducible.
    trainer_config = TrainerConfig(batch_size=128, device=device)

    result = run_e2e_batch(
        adapter, policy, critic, contract, loop_config, trainer_config,
        specs, actor_optimizer, critic_optimizer, lora_info, runtime,
        task_sampler=EpisodeTaskSampler(
            specs, DEFAULT_ROLE_TARGETS, base_seed=0
        ),
    )
    assert result["completed_episodes"] == 32
    buffer = result["buffer"]
    # test-local handles: recompute must use the rollout's exact config
    buffer._contract = contract
    buffer._device = device
    buffer._min_logprob_denoising_std = loop_config.min_logprob_denoising_std
    # the rollout-time critic — the diagnostic fills per-transition next
    # values with it (mirror of train_loop.py:2373-2378, one_update path)
    buffer._critic = critic
    return buffer
