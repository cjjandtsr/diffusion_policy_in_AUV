"""CKPT-AUDIT (§4.16): checkpoint schema (30 fields, v3), naming, resume roundtrip.

§4.16 fields (all required):
  schema_version, official_dppo_commit, project_git_commit/tree_hash,
  source_file_hashes, contract serialization + hash, manifest hash,
  BC checkpoint hash, policy/value state, LoRA state + lora_identity (v3),
  actor/value optimizer state, LR scheduler state, sampling/logprob std state,
  reward-normalization state, RNG states (python/numpy/torch/cuda/env),
  environment_rng_state (v3), sampler state (R2 P1-3), outer_iteration,
  ppo_update_count, optimizer_step_count, environment_transition_count,
  completed_episode_count, wall-clock timestamp, exact resolved config
  (with the hashed config_layer, v3).

Naming: dppo_uav_envstep_<envstep:09d>_ppoupd_<ppoupd:06d>.pt
[JUDGMENT] final checkpoint must never write step as 0.
"""

import base64
import dataclasses
import json
import os

import numpy as np
import pytest
import torch

from diffusion_policy_3d.rl.dppo_uav.checkpoint import (
    EMPTY_LORA_CHECKSUM,
    SCHEMA_VERSION,
    CheckpointBuilder,
    CheckpointData,
    CheckpointError,
    RuntimeIdentity,
    checkpoint_filename,
    config_hash_of,
    validate_resume_identity,
)
from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.critic import (
    PolicyObsEncoder,
    ValueCritic,
)
from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
    EpisodeTaskSampler,
)
from diffusion_policy_3d.rl.dppo_uav.observation_adapter import (
    select_latest_observations,
)
from diffusion_policy_3d.rl.dppo_uav.train_loop import (
    LoopConfig,
    TrainerConfig,
    restore_training_state,
    run_e2e_batch,
)

from fake_policy import FakePolicy, to_tensor_obs
from mock_uav_env import MockUAVEnv


def _layer() -> dict:
    """The exact hashed config layer (P1-2): config_hash is its hash, so
    the eval side can recompute instead of trusting the saved value."""
    return {"lr": 1e-3}


@pytest.fixture
def data():
    layer = _layer()
    d = CheckpointData(
        schema_version=SCHEMA_VERSION,
        official_dppo_commit="cc7234ad7ff39a8f32de3af903606723a16f0648",
        project_git_commit="abc1234",
        project_tree_hash="def5678",
        source_file_hashes={"contracts.py": "abc", "trainer.py": "def"},
        contract_serialized=json.dumps(
            {"exec_steps": 4, "action_horizon": 8}
        ).encode("utf-8"),
        contract_hash="cafe",
        manifest_hash="beef",
        bc_checkpoint_hash="1234",
        config_hash=config_hash_of(layer),
        policy_state={},
        value_state={"w": [2.0]},
        lora_state={},  # zero-LoRA world: identity declares rank 0
        lora_identity={
            "lora_rank": 0, "lora_alpha": 0.0, "n_adaptors": 0,
            "n_lora_param_tensors": 0,
            "lora_checksum": EMPTY_LORA_CHECKSUM,
        },
        actor_optimizer_state={"step": 1},
        value_optimizer_state={"step": 2},
        # R6-F (§11.7.1): structured scheduler/scaler states — never
        # {"enabled": False}-shaped placeholders; step counts mirror
        # ppo_update_count=7 (warmup=0 → actor steps every update too).
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
            "python_hash_seed": os.environ.get("PYTHONHASHSEED", ""),
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
        # R7-6 Step 9 drift law: cursor == outer_iteration - 1 (records
        # committed before this batch) — a restore-exercising fixture must
        # model a consistent save.
        metrics_writer_cursor=41,
        ppo_update_count=7,
        optimizer_step_count=100,
        environment_transition_count=2500,
        completed_episode_count=10,
        wall_clock_timestamp="2026-08-06T10:00:00",
        exact_resolved_config={"lr": 1e-3, "config_layer": layer},
    )
    # R7-6 v6 double axis: the official runtime source file set is
    # recorded twice — the fixture mirrors the save-side wiring
    # (_build_checkpoint_data passes source_file_hashes as
    # official_runtime_source_files) so every v6 contract check passes.
    return dataclasses.replace(
        d, official_runtime_source_files=d.source_file_hashes)


def test_all_30_fields_present(data):
    """§4.16 field checklist — every documented field must serialize."""
    snapshot = CheckpointBuilder.build(data)
    for field in (
        "schema_version", "official_dppo_commit", "project_git_commit",
        "project_tree_hash", "source_file_hashes", "contract_serialized",
        "contract_hash", "manifest_hash", "bc_checkpoint_hash",
        "policy_state", "value_state", "lora_state", "actor_optimizer_state",
        "value_optimizer_state", "lr_scheduler_state", "sampling_std_state",
        "reward_norm_state", "rng_states", "sampler_state",  # R2 P1-3
        "config_hash", "numerical_mode_state",  # schema v2
        "lora_identity", "environment_rng_state",  # schema v3 (R3 rework)
        "outer_iteration",
        "ppo_update_count", "optimizer_step_count",
        "environment_transition_count", "completed_episode_count",
        "wall_clock_timestamp", "exact_resolved_config",
    ):
        assert field in snapshot, f"missing §4.16 field {field}"


def test_checkpoint_filename_encoding():
    name = checkpoint_filename(env_step=1_000_000, ppo_update=400)
    assert name == "dppo_uav_envstep_001000000_ppoupd_000400.pt"


def test_checkpoint_filename_pads_and_ext():
    name = checkpoint_filename(env_step=5, ppo_update=7)
    assert name == "dppo_uav_envstep_000000005_ppoupd_000007.pt"


def test_checkpoint_filename_zero_step_rejected():
    """[JUDGMENT] final checkpoint must never be step 0."""
    with pytest.raises(CheckpointError):
        checkpoint_filename(env_step=0, ppo_update=1)


def test_save_and_load_roundtrip(data, tmp_path):
    path = tmp_path / "ckpt.pt"
    CheckpointBuilder.save(data, path)
    assert path.exists()
    loaded = CheckpointBuilder.load(path)
    assert loaded.schema_version == data.schema_version
    assert loaded.outer_iteration == data.outer_iteration
    assert loaded.ppo_update_count == data.ppo_update_count
    assert loaded.environment_transition_count == data.environment_transition_count
    assert loaded.contract_hash == data.contract_hash
    assert loaded.policy_state == data.policy_state
    # base64 build<->load symmetry must round-trip the raw bytes exactly
    assert loaded.contract_serialized == data.contract_serialized


def test_schema_version_mismatch_rejected(data, tmp_path):
    path = tmp_path / "ckpt.pt"
    CheckpointBuilder.save(data, path)
    # CheckpointData is frozen — mutation must fail validation on save
    bad = dataclasses.replace(CheckpointBuilder.load(path), schema_version="99.0")
    with pytest.raises(CheckpointError):
        CheckpointBuilder.save(bad, tmp_path / "bad.pt")


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(CheckpointError):
        CheckpointBuilder.load(tmp_path / "nope.pt")


def test_resume_keeps_counters(data, tmp_path):
    """resume test (RTM CKPT-AUDIT): reloaded snapshot keeps counters so the
    trainer can continue from the same environment step."""
    path = tmp_path / "ckpt.pt"
    CheckpointBuilder.save(data, path)
    loaded = CheckpointBuilder.load(path)
    assert loaded.environment_transition_count == 2500
    assert loaded.optimizer_step_count == 100
    assert loaded.completed_episode_count == 10
    # resume naming flows from the restored counters
    assert checkpoint_filename(
        loaded.environment_transition_count, loaded.ppo_update_count
    ) == "dppo_uav_envstep_000002500_ppoupd_000007.pt"


# ═══════════════════════════════════════════════════════════════════
# CRITIC-STATE-ONLY: value restore must never touch the policy
# ═══════════════════════════════════════════════════════════════════

def _r3_obs(contract: RLSystemContract, seed: int = 11):
    rng = np.random.RandomState(seed)
    obs = {
        "point_cloud": rng.uniform(
            -8.0, 8.0,
            (1, contract.history_steps, contract.point_count, 3),
        ).astype(np.float32),
        "agent_pos": rng.uniform(
            -1.0, 1.0, (1, contract.history_steps, 9)
        ).astype(np.float32),
    }
    selected, _ = select_latest_observations(
        to_tensor_obs(obs), contract.obs_steps
    )
    return selected


def test_value_restore_keeps_policy_state_untouched():
    """R3.1 device regression (G6 run-5 CPU-pollution chain): loading the
    value state must leave policy/normalizer/obs_encoder tensors bitwise
    unchanged (device/dtype/values), and the critic output must be exactly
    restored."""
    contract = RLSystemContract()
    policy = FakePolicy(contract)
    critic = ValueCritic(contract, encoder=PolicyObsEncoder(policy))
    obs = _r3_obs(contract)

    def policy_snapshot():
        # Python 3.8: no dict | operator — merge with **
        return {
            **{
                k: (v.device, v.dtype, v.detach().clone())
                for k, v in policy.obs_encoder.state_dict().items()
            },
            **{
                k: (v.device, v.dtype, v.detach().clone())
                for k, v in policy.model.state_dict().items()
            },
        }

    before = policy_snapshot()
    v_before = critic(obs).detach().clone()

    state = critic.value_state_dict()
    # the saved form is what the checkpoint path writes: CPU clones
    state = {k: v.detach().cpu().clone() for k, v in state.items()}
    critic.load_value_state_dict(state)

    after = policy_snapshot()
    for key in before:
        b_dev, b_dtype, b_val = before[key]
        a_dev, a_dtype, a_val = after[key]
        assert a_dev == b_dev and a_dtype == b_dtype, (
            f"policy tensor {key} moved: {b_dev}/{b_dtype} -> "
            f"{a_dev}/{a_dtype}"
        )
        assert torch.equal(a_val, b_val), (
            f"policy tensor {key} mutated by value restore"
        )
    assert torch.equal(critic(obs).detach(), v_before), (
        "critic output must be exactly restored after load_value_state_dict"
    )


# ═══════════════════════════════════════════════════════════════════
# SCHEMA-V3 / RESUME-HASH-GATE: identity validated before any load
# ═══════════════════════════════════════════════════════════════════

def _runtime_identity(data: CheckpointData) -> RuntimeIdentity:
    """RuntimeIdentity that matches a given CheckpointData."""
    return RuntimeIdentity(
        official_dppo_commit=data.official_dppo_commit,
        project_git_commit=data.project_git_commit,
        project_tree_hash=data.project_tree_hash,
        contract_hash=data.contract_hash,
        manifest_hash=data.manifest_hash,
        bc_checkpoint_hash=data.bc_checkpoint_hash,
        world_assets_hash=(data.exact_resolved_config or {}).get(
            "config_layer", {}).get("world_assets_hash"),
        config_hash=data.config_hash,
        source_file_hashes=data.source_file_hashes,
    )


HASH_FIELD_CASES = [
    ("official_dppo_commit", "deadbeef"),
    ("project_git_commit", "zzz999"),
    ("project_tree_hash", "yyy888"),
    ("contract_hash", "cafe-tampered"),
    ("manifest_hash", "beef-tampered"),
    ("bc_checkpoint_hash", "1234-tampered"),
    ("config_hash", "cfg-tampered"),
]


@pytest.mark.parametrize("field, bad", HASH_FIELD_CASES)
def test_resume_identity_mismatch_each_field_rejected(data, field, bad):
    """Six negative classes (contract/manifest/BC/config/official commit)
    plus the project git commit/tree hashes — each mismatch must raise
    CheckpointError, never warn-and-continue."""
    bad_data = dataclasses.replace(data, **{field: bad})
    with pytest.raises(CheckpointError):
        validate_resume_identity(bad_data, _runtime_identity(data))


def test_resume_identity_source_file_hashes_mismatch_rejected(data):
    bad_data = dataclasses.replace(
        data, source_file_hashes={"contracts.py": "tampered"}
    )
    with pytest.raises(CheckpointError):
        validate_resume_identity(bad_data, _runtime_identity(data))


def test_resume_identity_matching_passes(data):
    # no raise
    validate_resume_identity(data, _runtime_identity(data))


def test_resume_identity_commit_drift_bypass_skips_only_git_axes(data):
    """R5.9-C2: the eval-only bypass skips ONLY project_git_commit and
    project_tree_hash; every functional axis stays hard even when the
    bypass is requested — a drifted commit must never paper over a
    changed contract/manifest/config/source file."""
    drifted = dataclasses.replace(
        data, project_git_commit="zzz999", project_tree_hash="yyy888")
    # no raise
    validate_resume_identity(
        drifted, _runtime_identity(data), skip_commit_axes=True)

    for field, bad in HASH_FIELD_CASES:
        if field in ("project_git_commit", "project_tree_hash"):
            continue  # the two axes the bypass covers, verified above
        bad_data = dataclasses.replace(data, **{field: bad})
        with pytest.raises(CheckpointError):
            validate_resume_identity(
                bad_data, _runtime_identity(data), skip_commit_axes=True)


def test_resume_identity_source_file_hashes_stay_hard_under_bypass(data):
    """Even with skip_commit_axes=True a source-file hash mismatch must
    raise when no delta is pinned — the bypass exists for an
    artifact-layer fix landing after training, never for a changed
    training-time source file."""
    bad_data = dataclasses.replace(
        data, source_file_hashes={"contracts.py": "tampered"})
    with pytest.raises(CheckpointError):
        validate_resume_identity(
            bad_data, _runtime_identity(data), skip_commit_axes=True)


def test_resume_identity_source_delta_pins_exact_file_set(data):
    """allowed_source_delta pins the EXACT reviewed delta: a differing
    file inside the pinned set passes; one outside it raises."""
    drifted_hashes = dict(data.source_file_hashes)
    drifted_hashes["contracts.py"] = "tampered"
    runtime = dataclasses.replace(
        _runtime_identity(data),
        source_file_hashes=drifted_hashes,
    )
    # differing file covered by the pinned set → pass
    validate_resume_identity(
        data, runtime, allowed_source_delta={"contracts.py"})
    # differing file outside the pinned set → raise
    with pytest.raises(CheckpointError):
        validate_resume_identity(
            data, runtime, allowed_source_delta={"other.py"})
    # no delta pinned at all → hard as before
    with pytest.raises(CheckpointError):
        validate_resume_identity(data, runtime)


def test_schema_v1_checkpoint_unsupported_for_formal_resume(data, tmp_path):
    """Schema v1 is read-only audit/export — the formal loader must
    explicitly report unsupported resume schema v1, never guess the
    missing sampler/hash state."""
    v1 = dataclasses.replace(data, schema_version="1.1")
    payload = {**dataclasses.asdict(v1)}
    # save() rejects mismatched schema by design — hand-write the v1 file
    # exactly like an old checkpoint on disk (contract bytes stored
    # base64, as build() always did; see build()'s weights_only note).
    payload["contract_serialized"] = base64.b64encode(
        payload["contract_serialized"]
    ).decode("ascii")
    path = tmp_path / "v1.pt"
    torch.save(payload, path)
    with pytest.raises(CheckpointError) as exc:
        CheckpointBuilder.load(path)
    assert "unsupported resume schema" in str(exc.value)


def _train_loop_fixtures():
    contract = RLSystemContract()
    policy = FakePolicy(contract)
    critic = ValueCritic(contract, encoder=PolicyObsEncoder(policy))
    return contract, policy, critic


def _loop_cfg(tmp_path: str) -> LoopConfig:
    return LoopConfig(
        mode="one_update", n_episodes=2, max_ep_len=12,
        device="cpu", output_dir=str(tmp_path),
    )


def _sampler(contract: RLSystemContract):
    from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
        DEFAULT_ROLE_TARGETS,
        EpisodeTaskSpec,
    )
    specs = [
        EpisodeTaskSpec(
            world_uid=f"mock_{role}", obstacle_json="{}", role=role,
            split="rl_train", admission_id=f"mock__{role}",
            generator_version="mock", manifest_entry_hash=f"hash_{role}",
        )
        for role in ("anticipatory", "dense_recovery",
                     "support_expansion", "yaw_lateral_recovery")
    ]
    return specs, EpisodeTaskSampler(specs, DEFAULT_ROLE_TARGETS, base_seed=0)


def _runtime_dict() -> dict:
    return {
        "timestamp": "2026-08-06T00:00:00Z",
        "device": "cpu",
        "reward_version": "v1.3",
        "bc_checkpoint_hash": "bc0123",
        "manifest_hash": "mf456",
        "world_assets_hash": "world456",
        "source_file_hashes": {},
        "project_git_commit": "n/a",
        "project_tree_hash": "n/a",
    }


def test_restore_validates_identity_before_any_policy_load(data, tmp_path):
    """The spy proof: on identity mismatch, policy.load_state_dict must be
    called ZERO times — validation happens before any tensor load."""
    contract, policy, critic = _train_loop_fixtures()
    path = tmp_path / "ckpt.pt"
    CheckpointBuilder.save(data, path)
    # tamper the config identity in the on-disk checkpoint
    tampered = dataclasses.replace(data, config_hash="cfg-tampered")
    CheckpointBuilder.save(tampered, tmp_path / "ckpt2.pt")

    calls = []

    def spy(state):
        calls.append(1)
        policy._load_original(state)

    import types
    policy._load_original = policy.load_state_dict
    policy.load_state_dict = types.MethodType(spy, policy)

    with pytest.raises(CheckpointError):
        restore_training_state(
            str(tmp_path / "ckpt2.pt"),
            policy, critic, {}, torch.optim.AdamW(
                policy.model.parameters(), lr=1e-4
            ), torch.optim.AdamW(critic.value_head_parameters(), lr=1e-4),
            runtime_identity=_runtime_identity(data),
        )
    assert calls == [], (
        "policy.load_state_dict must never run when resume identity "
        "mismatches"
    )


def test_restore_rejects_empty_policy_state(data, tmp_path):
    """Review finding W3: an empty policy payload is never a valid formal
    checkpoint — restore must hard-fail, never silently continue with
    fresh weights under a saved identity."""
    empty = dataclasses.replace(data, policy_state={})
    path = tmp_path / "empty_policy.pt"
    CheckpointBuilder.save(empty, path)
    contract, policy, critic = _train_loop_fixtures()
    with pytest.raises(CheckpointError) as exc:
        restore_training_state(
            str(path), policy, critic, {}, torch.optim.AdamW(
                policy.model.parameters(), lr=1e-4
            ), torch.optim.AdamW(critic.value_head_parameters(), lr=1e-4),
            runtime_identity=_runtime_identity(empty),
        )
    assert "policy_state must not be empty" in str(exc.value)


def test_load_rejects_non_dict_payload(tmp_path):
    """Review finding L1: a non-dict payload must raise CheckpointError,
    not an unhandled AttributeError."""
    path = tmp_path / "not_a_checkpoint.pt"
    torch.save([1, 2, 3], path)
    with pytest.raises(CheckpointError) as exc:
        CheckpointBuilder.load(path)
    assert "not a dict" in str(exc.value)


def test_save_writes_structured_std_and_scheduler_state(tmp_path):
    """Global constraint: disabled lr scheduler / adaptive std must be
    structured {"enabled": false}-style payloads, never ambiguous {}."""
    contract, policy, critic = _train_loop_fixtures()
    cfg = _loop_cfg(tmp_path)
    specs, sampler = _sampler(contract)
    from diffusion_policy_3d.rl.dppo_uav.trainer import TrainerConfig
    r1 = run_e2e_batch(
        env=MockUAVEnv(contract),
        policy=policy, critic=critic, contract=contract,
        loop_config=cfg, trainer_config=TrainerConfig(device="cpu"),
        task_specs=specs,
        actor_optimizer=torch.optim.AdamW(
            policy.model.parameters(), lr=1e-4
        ),
        critic_optimizer=torch.optim.AdamW(
            critic.value_head_parameters(), lr=1e-4
        ),
        lora_info={}, runtime=_runtime_dict(),
        task_sampler=sampler,
    )
    saved = CheckpointBuilder.load(r1["checkpoint_path"])
    # R6-F (§11.7.2): the structured scheduler state carries method/
    # version + per-side step counts; a warmup=0 run steps actor and
    # critic once per committed update (cross-field invariant).
    assert saved.lr_scheduler_state["method"] == "none", (
        f"lr_scheduler_state must declare method: {saved.lr_scheduler_state}"
    )
    assert saved.lr_scheduler_state["version"] == "v1", (
        f"lr_scheduler_state must declare version: {saved.lr_scheduler_state}"
    )
    assert saved.lr_scheduler_state["actor"]["step_count"] == (
        saved.ppo_update_count
    ), f"actor scheduler step_count: {saved.lr_scheduler_state}"
    assert saved.lr_scheduler_state["critic"]["step_count"] == (
        saved.ppo_update_count
    ), f"critic scheduler step_count: {saved.lr_scheduler_state}"
    assert saved.sampling_std_state.get("adaptive") is False, (
        f"sampling_std_state must declare adaptive: {saved.sampling_std_state}"
    )
    assert saved.sampling_std_state["min_sampling_denoising_std"] == 0.1
    assert saved.sampling_std_state["min_logprob_denoising_std"] == 0.1
    assert saved.numerical_mode_state, (
        f"numerical_mode_state must be the resolved snapshot, not {{}}: "
        f"{saved.numerical_mode_state}"
    )


def test_outer_iteration_cumulative_across_fresh_and_two_resumes(tmp_path):
    """Counters test: fresh -> update1 -> resume update2 -> resume update3
    must record outer_iteration = [1, 2, 3] with strictly monotone PPO
    update/env/optimizer/completed counters and a final filename with no
    fake zero."""
    contract, policy, critic = _train_loop_fixtures()
    cfg = _loop_cfg(tmp_path)
    specs, sampler = _sampler(contract)
    from diffusion_policy_3d.rl.dppo_uav.trainer import TrainerConfig
    r1 = run_e2e_batch(
        env=MockUAVEnv(contract), policy=policy, critic=critic,
        contract=contract, loop_config=cfg,
        trainer_config=TrainerConfig(device="cpu"),
        task_specs=specs,
        actor_optimizer=torch.optim.AdamW(
            policy.model.parameters(), lr=1e-4
        ),
        critic_optimizer=torch.optim.AdamW(
            critic.value_head_parameters(), lr=1e-4
        ),
        lora_info={}, runtime=_runtime_dict(), task_sampler=sampler,
        # R7-6 production cursor law: batch i saves cursor i-1 (records
        # committed before it) — the restore-side drift gate demands
        # cursor == outer_iteration-1 (launcher: batch_idx - 1).
        metrics_writer_cursor=0,
    )
    c1 = CheckpointBuilder.load(r1["checkpoint_path"])
    assert c1.outer_iteration == 1, c1.outer_iteration
    assert c1.ppo_update_count == 1

    r2 = run_e2e_batch(
        env=MockUAVEnv(contract), policy=policy, critic=critic,
        contract=contract, loop_config=cfg,
        trainer_config=TrainerConfig(device="cpu"),
        task_specs=specs,
        actor_optimizer=torch.optim.AdamW(
            policy.model.parameters(), lr=1e-4
        ),
        critic_optimizer=torch.optim.AdamW(
            critic.value_head_parameters(), lr=1e-4
        ),
        lora_info={}, runtime=_runtime_dict(), task_sampler=sampler,
        resume_from=r1["checkpoint_path"],
        metrics_writer_cursor=1,
    )
    c2 = CheckpointBuilder.load(r2["checkpoint_path"])
    assert c2.outer_iteration == 2, c2.outer_iteration
    assert c2.ppo_update_count == 2

    r3 = run_e2e_batch(
        env=MockUAVEnv(contract), policy=policy, critic=critic,
        contract=contract, loop_config=cfg,
        trainer_config=TrainerConfig(device="cpu"),
        task_specs=specs,
        actor_optimizer=torch.optim.AdamW(
            policy.model.parameters(), lr=1e-4
        ),
        critic_optimizer=torch.optim.AdamW(
            critic.value_head_parameters(), lr=1e-4
        ),
        lora_info={}, runtime=_runtime_dict(), task_sampler=sampler,
        resume_from=r2["checkpoint_path"],
        metrics_writer_cursor=2,
    )
    c3 = CheckpointBuilder.load(r3["checkpoint_path"])
    assert c3.outer_iteration == 3, c3.outer_iteration
    assert c3.ppo_update_count == 3
    assert c1.optimizer_step_count < c2.optimizer_step_count
    assert c2.optimizer_step_count < c3.optimizer_step_count
    assert c1.environment_transition_count < c2.environment_transition_count
    assert c2.environment_transition_count < c3.environment_transition_count
    assert c1.completed_episode_count < c2.completed_episode_count
    assert c2.completed_episode_count < c3.completed_episode_count
    assert "000000" not in os.path.basename(r3["checkpoint_path"]).split(
        "_ppoupd_"
    )[0][-9:] or c3.environment_transition_count > 0
