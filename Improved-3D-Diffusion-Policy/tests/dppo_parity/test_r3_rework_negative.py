"""R3 裁决 rework — 负向路径 RED 测试（外部审查 2026-08-11）。

外部审查裁决 R3 = REWORK REQUIRED（4 FAIL + 3 PASS），本文件把审查的
每个负向要求钉成测试（测试先行：全部 RED，实现后全绿）：

  P0-1  LoRA 身份未闭合
    - test_restore_hard_fails_wired_lora_with_empty_state（train 侧）
    - test_restore_hard_fails_checkpoint_lora_state_without_live_wiring
    - test_restore_hard_fails_lora_rank_alpha_mismatch
    - test_eval_artifact_rejects_empty_lora_set_checksum（eval 侧：
      sha256(b"") 曾绕过非空检查，PPO==BC 被当作有效结果）
  P0-2  sampler 身份校验太晚
    - test_restore_sampler_mismatch_hard_fails_before_any_load
      （所有 state-load spy 均为 0）
    - test_config_hash_sensitive_to_each_identity_axis
      （base_seed/n_envs/lora_rank/lora_alpha/lr_actor/lr_value/optimizer）
  P1-1  eval-only world 被排除
    - test_eval_pool_accepts_eval_only_worlds（正向）
    - test_train_sampler_rejects_eval_only_specs（训练边界仍拒绝，GREEN guard）
    - test_build_adapter_accepts_explicit_pool
  P1-2  config_hash 自比恒等
    - test_config_layer_hash_matches_checkpoint_config_hash
    - test_verify_config_layer_identity_rejects_tampered_config
  P1-3  bank provenance 装饰
    - test_bank_entries_carry_bank_id_and_version
    - test_bank_id_binds_entry_bytes
    - test_bank_missing_bank_id_rejected
    - test_eval_artifact_records_bank_and_density_provenance
    - test_eval_artifact_missing_bank_or_density_provenance_hard_fails
  P1-4  NaN 被掩码成有效指标
    - test_capability_metrics_rejects_nan_trace_field / nan_pos / inf
    - test_eval_artifact_rejects_non_finite_trace_fields
  C1    无独立 environment_rng_state
    - test_save_load_roundtrip_keeps_new_identity_fields
    - test_restore_hard_fails_invalid_environment_rng_state
  C2    4 个 state 字段存而不验
    - test_restore_structured_state_mismatch_hard_fails_before_any_load
  Schema 升级
    - test_schema_v2_checkpoint_rejected_for_formal_resume
  SEED-RESUME 内部化
    - test_restore_loads_sampler_state_internally_when_matching
"""

import base64
import copy
import dataclasses
import hashlib
import json
import os

import pytest
import torch

from diffusion_policy_3d.rl.dppo_uav.capability_metrics import (
    OfflineDensity,
    OPPORTUNITY_INVALID_REASONS,
    compute_capability_metrics,
)
from diffusion_policy_3d.rl.dppo_uav.checkpoint import (
    SCHEMA_VERSION,
    CheckpointBuilder,
    CheckpointData,
    CheckpointError,
    EMPTY_LORA_CHECKSUM,
    config_hash_of,
    verify_config_layer_identity,
)
from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
    DEFAULT_ROLE_TARGETS,
    EpisodeTaskSampler,
)
from diffusion_policy_3d.rl.dppo_uav.evaluator import (
    BANK_VERSION,
    EvalArtifactError,
    PairedBankEntry,
    paired_bank_from_dict,
    paired_bank_to_dict,
    validate_eval_artifact,
)
from diffusion_policy_3d.rl.dppo_uav.integration import (
    build_adapter,
    build_routea_eval_pool,
)
from diffusion_policy_3d.rl.dppo_uav.optimizer_contract import (
    resolved_optimizer_config,
)
from diffusion_policy_3d.rl.dppo_uav.train_loop import (
    LoopConfig,
    TrainerConfig,
    StructuredStateIdentity,
    _config_dict,
    _rng_states,
    restore_training_state,
)
from diffusion_policy_3d.rl.dppo_uav.training_schedules import (
    SCHEDULER_STATE_VERSION,
)

from test_capability_metrics import _density as _cap_density
from test_capability_metrics import _record, _straight
from test_checkpoint import (
    _loop_cfg,
    _runtime_dict,
    _runtime_identity,
    _sampler,
    _train_loop_fixtures,
)
from test_eval_artifact import (
    _artifact,
    _density,
    _paired,
    _single_role_artifact,
    _three_role_records,
)
from test_integration_routea_pool import (
    _entry as _manifest_entry,
    _write_manifest,
    _write_obstacles,
)


def _config_layer(**training_identity) -> dict:
    """The reproducible config layer — every axis the config_hash covers."""
    return {
        "device": "cpu",
        "reward_version": "v1.3",
        "loop_mode": "one_update",
        "n_episodes": 2,
        "max_ep_len": 12,
        "eval_n_episodes": 0,
        "gate_phase": "rl",
        "min_sampling_denoising_std": 0.1,
        "min_logprob_denoising_std": 0.1,
        "numerical_mode": {
            "allow_tf32_matmul": False,
            "allow_tf32_cudnn": False,
            "deterministic_algorithms": True,
        },
        "policy_contract": {
            "policy_cfg_lambda": 1.5,
            "rollout_policy_batch_size": 1,
            "likelihood_microbatch_size": 1,
        },
        "trainer": {
            "update_epochs": 1,
            "batch_size": 4,
            "max_grad_norm": 0.5,
            "target_kl": 0.01,
            "next_value_batch_size_configured": 4,
        },
        "training_identity": {
            "base_seed": 0,
            "n_envs": 1,
            "lora_rank": 0,
            "lora_alpha": 0.0,
            "lr_actor": 1e-4,
            "lr_value": 1e-4,
            "actor_optimizer": "AdamW",
            "value_optimizer": "AdamW",
            **training_identity,
        },
    }


def _v3_data(**overrides) -> CheckpointData:
    """CheckpointData fixture base (30 fields; schema version follows the
    current SCHEMA_VERSION — v4 since rework #3). Name kept for historical
    continuity: v1/v2/v3-specific negatives override schema_version
    explicitly. The two identity fields (lora_identity,
    environment_rng_state) + config_layer inside exact_resolved_config,
    with config_hash computed over the layer."""
    layer = _config_layer()
    base = dict(
        schema_version=SCHEMA_VERSION,
        official_dppo_commit="cc7234ad7ff39a8f32de3af903606723a16f0648",
        project_git_commit="abc1234",
        project_tree_hash="def5678",
        source_file_hashes={"contracts.py": "abc", "trainer.py": "def"},
        # R7-6 v6 double axis: the save-side wiring (_build_checkpoint_data)
        # records the official runtime source file set alongside
        # source_file_hashes — a restore-exercising fixture must mirror it.
        official_runtime_source_files={
            "contracts.py": "abc", "trainer.py": "def"
        },
        contract_serialized=json.dumps(
            {"exec_steps": 4, "action_horizon": 8}
        ).encode("utf-8"),
        contract_hash="cafe",
        manifest_hash="beef",
        bc_checkpoint_hash="1234",
        config_hash=config_hash_of(layer),
        policy_state={"w": [1.0]},
        value_state={"w": [2.0]},
        lora_state={},
        lora_identity={
            "lora_rank": 0, "lora_alpha": 0.0, "n_adaptors": 0,
            "n_lora_param_tensors": 0,
            "lora_checksum": EMPTY_LORA_CHECKSUM,
        },
        actor_optimizer_state={"step": 1},
        value_optimizer_state={"step": 2},
        # R6-F (§11.7.1): structured scheduler/scaler states — never
        # {"enabled": False}-shaped placeholders; per-side step counts
        # mirror ppo_update_count=7 (warmup=0 → actor steps too).
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
        rng_states={
            "python": "py", "numpy": "np", "torch": "th",
            "cuda": "cu", "env": "",
        },
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
        ppo_update_count=7,
        optimizer_step_count=100,
        environment_transition_count=2500,
        completed_episode_count=10,
        # R7-6 Step 9 drift law: cursor == outer_iteration - 1 (records
        # committed before this batch) — a restore-exercising fixture must
        # model a consistent save.
        metrics_writer_cursor=41,
        wall_clock_timestamp="2026-08-06T10:00:00",
        exact_resolved_config={
            "config_layer": layer, "episode_metadata": {},
        },
    )
    return CheckpointData(**{**base, **overrides})


def _wired_lora_data(rank: int = 4, alpha: float = 1.0, n: int = 2,
                     **overrides) -> CheckpointData:
    """Checkpoint whose lora_state is non-empty and whose lora_identity
    declares rank/alpha/n_adaptors — the trained-with-LoRA case.

    注意：keys 为 lora_0_a/lora_0_b，刻意不遵守 lora_<i> 命名契约 —
    该 fixture 只服务 checksum/axis 失配测试（match="checksum"/"LoRA"），
    命名契约本身由 test_r3_rework2_negative.py 的
    test_lora_state_key_naming_mismatch_hard_fails 直接 pin。
    """
    return _v3_data(
        lora_state={
            "lora_0_a": torch.tensor([0.1]),
            "lora_0_b": torch.tensor([0.2]),
        },
        lora_identity={
            "lora_rank": rank, "lora_alpha": alpha, "n_adaptors": n,
            "n_lora_param_tensors": 2,  # 2 keys (lora_0_a/lora_0_b)
            "lora_checksum": "deadbeef",
        },
        **overrides,
    )


def _restore_ready(d: CheckpointData, policy, critic, opt1, opt2) -> CheckpointData:
    """Replace the fixture's placeholder state payloads with REAL states
    from the live objects.

    The positive restore paths strict-load every state dict — the fake
    payloads crash instead of restoring: {"w": [1.0]} fails strict key
    typing, {"w": [2.0]} is not an MLP-shaped value state, {"step": 1}
    has no param_groups, and the fake rng_states strings fail
    bytes.fromhex. The checkpoint a restore actually reads carries real
    states, so the fixture must too.

    R6-E §11.6.2: the frozen resolved optimizer contract (computed from
    the LIVE optimizers via the production resolver) rides the config
    layer, and config_hash is recomputed over the updated layer — the
    fixture mirrors what production now saves, so restore's per-field
    optimizer validation has a frozen reference to check against.
    """
    layer = dict(d.exact_resolved_config["config_layer"])
    layer["optimizer_contract"] = {
        "actor": resolved_optimizer_config("actor", opt1).to_dict(),
        "value": resolved_optimizer_config("value", opt2).to_dict(),
    }
    return dataclasses.replace(
        d,
        policy_state=policy.state_dict(),
        value_state=critic.value_state_dict(),
        actor_optimizer_state=opt1.state_dict(),
        value_optimizer_state=opt2.state_dict(),
        rng_states=_rng_states(),
        exact_resolved_config={
            "config_layer": layer, "episode_metadata": {},
        },
        config_hash=config_hash_of(layer),
    )


def _opt1(policy) -> torch.optim.Optimizer:
    return torch.optim.AdamW(policy.model.parameters(), lr=1e-4)


def _opt2(critic) -> torch.optim.Optimizer:
    return torch.optim.AdamW(critic.value_head_parameters(), lr=1e-4)


def _spy_state_loads(policy, critic, opt1, opt2) -> list:
    """Wrap every state_dict load with a recorder — the resume spy proof:
    on ANY pre-load validation failure, every entry must stay empty."""
    import types

    calls: list = []
    originals = {
        "policy": policy.load_state_dict,
        "critic": critic.load_value_state_dict,
        "opt1": opt1.load_state_dict,
        "opt2": opt2.load_state_dict,
    }

    def make_spy(name, target):
        def spy(self, state, *args, **kwargs):
            calls.append(name)
            return target(state, *args, **kwargs)

        return spy

    policy.load_state_dict = types.MethodType(
        make_spy("policy", originals["policy"]), policy
    )
    critic.load_value_state_dict = types.MethodType(
        make_spy("critic", originals["critic"]), critic
    )
    # Bound the same way as policy/critic — a raw function assigned to the
    # instance would receive `state` as `self` and miss the real argument.
    opt1.load_state_dict = types.MethodType(
        make_spy("opt1", originals["opt1"]), opt1
    )
    opt2.load_state_dict = types.MethodType(
        make_spy("opt2", originals["opt2"]), opt2
    )
    return calls


# ═══════════════════════════════════════════════════════════════════
# Schema v3: v2 显式 audit-only + 新身份字段 round-trip
# ═══════════════════════════════════════════════════════════════════

def test_schema_v2_checkpoint_rejected_for_formal_resume(tmp_path):
    """A schema-v2 file (no lora_identity / environment_rng_state) is
    audit-only: the formal loader must report unsupported resume schema
    v2, never load a checkpoint whose identity cannot be verified."""
    payload = dataclasses.asdict(_v3_data())
    payload["schema_version"] = "2"
    # build() stores contract bytes base64 — hand-write like an old file
    payload["contract_serialized"] = base64.b64encode(
        payload["contract_serialized"]
    ).decode("ascii")
    path = tmp_path / "v2.pt"
    torch.save(payload, path)
    with pytest.raises(CheckpointError, match="unsupported resume schema"):
        CheckpointBuilder.load(path)


def test_save_load_roundtrip_keeps_new_identity_fields(tmp_path):
    """C1: lora_identity + environment_rng_state + config_layer survive
    the save/load round-trip — the eval side must be able to verify them."""
    d = _v3_data()
    path = tmp_path / "ckpt.pt"
    CheckpointBuilder.save(d, path)
    loaded = CheckpointBuilder.load(path)
    assert loaded.lora_identity == d.lora_identity
    assert loaded.environment_rng_state == d.environment_rng_state
    assert loaded.exact_resolved_config["config_layer"] == (
        d.exact_resolved_config["config_layer"]
    )


# ═══════════════════════════════════════════════════════════════════
# P0-2: config_hash 必须覆盖训练输入身份（base_seed/n_envs/lora/lr/opt）
# ═══════════════════════════════════════════════════════════════════

def test_config_hash_sensitive_to_each_identity_axis(tmp_path):
    """Changing ANY training-input axis must change the config hash —
    a resume under different seeds/ranks/lrs/optimizers must never be
    invisible to the identity gate."""
    contract, policy, critic = _train_loop_fixtures()
    cfg = _loop_cfg(tmp_path)
    runtime = _runtime_dict()
    numerical = {
        "allow_tf32_matmul": False, "allow_tf32_cudnn": False,
        "deterministic_algorithms": True,
    }
    opt1, opt2 = _opt1(policy), _opt2(critic)
    base = dict(
        base_seed=0, n_envs=1, lora_info={},
        actor_optimizer=opt1, critic_optimizer=opt2,
    )
    base_hash = config_hash_of(_config_dict(
        contract, cfg, TrainerConfig(device="cpu"), runtime,
        numerical, policy, **base,
    ))
    mutations = {
        "base_seed": {"base_seed": 1},
        "n_envs": {"n_envs": 2},
        "lora_rank/alpha": {
            "lora_info": {"lora_rank": 4, "lora_alpha": 1.0,
                          "n_adaptors": 2},
        },
        "lr_actor": {"actor_optimizer": torch.optim.AdamW(
            policy.model.parameters(), lr=2e-4)},
        "lr_value": {"critic_optimizer": torch.optim.AdamW(
            critic.value_head_parameters(), lr=5e-5)},
        "optimizer identity": {"actor_optimizer": torch.optim.SGD(
            policy.model.parameters(), lr=1e-4)},
        # P2-3 (fresh review): value_optimizer 轴也必须敏感——critic
        # optimizer 类型变更（非仅 lr）不得被身份门禁忽略
        "value_optimizer identity": {"critic_optimizer": torch.optim.SGD(
            critic.value_head_parameters(), lr=1e-4)},
    }
    for name, mutate in mutations.items():
        changed = config_hash_of(_config_dict(
            contract, cfg, TrainerConfig(device="cpu"), runtime,
            numerical, policy, **{**base, **mutate},
        ))
        assert changed != base_hash, (
            f"config_hash must change when {name} changes"
        )


# ═══════════════════════════════════════════════════════════════════
# P1-2: config_layer 哈希可重算——eval 侧不再自比
# ═══════════════════════════════════════════════════════════════════

def test_config_layer_hash_matches_checkpoint_config_hash():
    """The checkpoint's config_hash must equal the hash recomputed over
    the persisted config_layer — the eval side recomputes instead of
    trusting the saved value."""
    d = _v3_data()
    erc = d.exact_resolved_config
    assert config_hash_of(erc["config_layer"]) == d.config_hash


def test_verify_config_layer_identity_rejects_tampered_config():
    """A config_layer that no longer hashes to the saved config_hash is
    a tampered/tautological checkpoint — hard fail, never trust."""
    d = _v3_data()
    erc = d.exact_resolved_config
    # positive: matching layer verifies
    verify_config_layer_identity(erc, d.config_hash)
    tampered = {
        **erc,
        "config_layer": {**erc["config_layer"], "base_seed": 999},
    }
    with pytest.raises(CheckpointError, match="config layer"):
        verify_config_layer_identity(tampered, d.config_hash)
    missing = {k: v for k, v in erc.items() if k != "config_layer"}
    with pytest.raises(CheckpointError, match="config_layer"):
        verify_config_layer_identity(missing, d.config_hash)


# ═══════════════════════════════════════════════════════════════════
# P0-1: LoRA 身份在 restore 前硬失败（空态/未接线/rank/alpha 失配）
# ═══════════════════════════════════════════════════════════════════

def test_restore_hard_fails_checkpoint_lora_state_without_live_wiring(
        tmp_path):
    """Checkpoint trained WITH LoRA resumed WITHOUT wiring would silently
    run the BC baseline (PPO==BC) — hard fail, zero loads."""
    path = tmp_path / "ckpt.pt"
    CheckpointBuilder.save(_wired_lora_data(), path)
    contract, policy, critic = _train_loop_fixtures()
    calls = _spy_state_loads(policy, critic, _opt1(policy), _opt2(critic))
    with pytest.raises(CheckpointError, match="LoRA"):
        restore_training_state(
            str(path), policy, critic, {}, _opt1(policy), _opt2(critic),
            runtime_identity=_runtime_identity(_wired_lora_data()),
        )
    assert calls == [], "no state load may run on a LoRA-wiring mismatch"


def test_restore_hard_fails_wired_lora_with_empty_state(tmp_path):
    """Live LoRA wired over a checkpoint with NO LoRA state leaves the
    adaptor tensors random — hard fail, never a silently different actor."""
    d = _v3_data()
    path = tmp_path / "ckpt.pt"
    CheckpointBuilder.save(d, path)  # lora_state={}, identity rank 0
    contract, policy, critic = _train_loop_fixtures()
    live = {"lora_rank": 4, "lora_alpha": 1.0, "n_adaptors": 2}
    calls = _spy_state_loads(policy, critic, _opt1(policy), _opt2(critic))
    with pytest.raises(CheckpointError, match="LoRA"):
        restore_training_state(
            str(path), policy, critic, live, _opt1(policy), _opt2(critic),
            runtime_identity=_runtime_identity(d),
        )
    assert calls == [], "no state load may run on an empty-LoRA checkpoint"


@pytest.mark.parametrize("live, match", [
    ({"lora_rank": 8, "lora_alpha": 1.0, "n_adaptors": 2}, "lora_rank"),
    ({"lora_rank": 4, "lora_alpha": 2.0, "n_adaptors": 2}, "lora_alpha"),
    ({"lora_rank": 4, "lora_alpha": 1.0, "n_adaptors": 1}, "n_adaptors"),
])
def test_restore_hard_fails_lora_rank_alpha_mismatch(live, match, tmp_path):
    """Rank/alpha/adaptor-count identity must match the checkpoint — a
    differently-scaled LoRA is a different policy (alpha scales output)."""
    path = tmp_path / "ckpt.pt"
    d = _wired_lora_data(rank=4, alpha=1.0, n=2)
    CheckpointBuilder.save(d, path)
    contract, policy, critic = _train_loop_fixtures()
    calls = _spy_state_loads(policy, critic, _opt1(policy), _opt2(critic))
    with pytest.raises(CheckpointError, match=match):
        restore_training_state(
            str(path), policy, critic, live, _opt1(policy), _opt2(critic),
            runtime_identity=_runtime_identity(d),
        )
    assert calls == [], "no state load may run on a LoRA identity mismatch"


# ═══════════════════════════════════════════════════════════════════
# P0-2: sampler 身份校验在 restore 内、任何加载之前
# ═══════════════════════════════════════════════════════════════════

def test_restore_sampler_mismatch_hard_fails_before_any_load(tmp_path):
    """base_seed/spec-set mismatch must fail INSIDE restore_training_state
    before ANY state load — the call-site-only check (R3.2) validated too
    late, after policy/critic/optimizer/RNG were already restored."""
    d = _v3_data()
    path = tmp_path / "ckpt.pt"
    CheckpointBuilder.save(d, path)  # sampler_state base_seed=7
    contract, policy, critic = _train_loop_fixtures()
    _, sampler = _sampler(contract)  # live base_seed=0 -> mismatch
    calls = _spy_state_loads(policy, critic, _opt1(policy), _opt2(critic))
    with pytest.raises(CheckpointError, match="sampler"):
        restore_training_state(
            str(path), policy, critic, {}, _opt1(policy), _opt2(critic),
            runtime_identity=_runtime_identity(d),
            task_sampler=sampler,
        )
    assert calls == [], (
        "no state load may run when the sampler identity mismatches"
    )


def test_restore_loads_sampler_state_internally_when_matching(tmp_path):
    """On a matching identity the sampler state is loaded INSIDE restore
    (SEED-RESUME continues, never restarts at base_seed) — the call site
    must not load it after the fact."""
    contract, policy, critic = _train_loop_fixtures()
    _, sampler = _sampler(contract)
    opt1, opt2 = _opt1(policy), _opt2(critic)
    d = _restore_ready(
        dataclasses.replace(_v3_data(), sampler_state=sampler.state_dict()),
        policy, critic, opt1, opt2,
    )
    path = tmp_path / "ckpt.pt"
    CheckpointBuilder.save(d, path)
    sampler.next_attempt()  # live state diverges from the saved one
    restore_training_state(
        str(path), policy, critic, {}, opt1, opt2,
        runtime_identity=_runtime_identity(d),
        task_sampler=sampler,
    )
    assert sampler.state_dict()["global_attempt_index"] == (
        d.sampler_state["global_attempt_index"]
    ), "sampler must continue from the checkpointed attempt counter"


# ═══════════════════════════════════════════════════════════════════
# C2: 4 个结构化 state 字段在 restore 前逐项验证（存而不验反模式）
# ═══════════════════════════════════════════════════════════════════

def _good_structured() -> StructuredStateIdentity:
    # R6-F (§11.7.1): STATIC identity axes only — scheduler {method,
    # version} and reward_norm {version, source, method}; the per-batch
    # mutable params are cross-field validated inside restore.
    return StructuredStateIdentity(
        lr_scheduler_state={
            "method": "none", "version": SCHEDULER_STATE_VERSION,
        },
        sampling_std_adaptive=False,
        min_sampling_denoising_std=0.1,
        min_logprob_denoising_std=0.1,
        reward_norm_state={
            "version": "v1.3", "source": "env", "method": "none",
        },
        numerical_mode={
            "allow_tf32_matmul": False, "allow_tf32_cudnn": False,
            "deterministic_algorithms": True,
        },
    )


@pytest.mark.parametrize("axis_kwargs", [
    {"lr_scheduler_state": {"method": "official_cosine_warmup",
                            "version": SCHEDULER_STATE_VERSION}},
    {"sampling_std_adaptive": True},
    {"min_sampling_denoising_std": 0.5},
    {"reward_norm_state": {"version": "v2.0", "source": "env",
                           "method": "none"}},
    {"numerical_mode": {"allow_tf32_matmul": True,
                        "allow_tf32_cudnn": False,
                        "deterministic_algorithms": True}},
])
def test_restore_structured_state_mismatch_hard_fails_before_any_load(
        axis_kwargs, tmp_path):
    """lr_scheduler/sampling_std/reward_norm/numerical_mode are SAVED but
    must also be VALIDATED — a checkpoint whose saved states disagree with
    the live config is refused before any tensor load."""
    contract, policy, critic = _train_loop_fixtures()
    opt1, opt2 = _opt1(policy), _opt2(critic)
    # the positive path strict-loads every state — real payloads only
    ready = _restore_ready(_v3_data(), policy, critic, opt1, opt2)
    path = tmp_path / "ckpt.pt"
    CheckpointBuilder.save(ready, path)
    bad = dataclasses.replace(_good_structured(), **axis_kwargs)
    calls = _spy_state_loads(policy, critic, opt1, opt2)
    with pytest.raises(CheckpointError, match="structured"):
        restore_training_state(
            str(path), policy, critic, {}, opt1, opt2,
            runtime_identity=_runtime_identity(ready),
            structured_identity=bad,
        )
    assert calls == [], (
        "no state load may run on a structured-state mismatch"
    )
    # matching identity restores cleanly
    restore_training_state(
        str(path), policy, critic, {}, opt1, opt2,
        runtime_identity=_runtime_identity(ready),
        structured_identity=_good_structured(),
    )


# ═══════════════════════════════════════════════════════════════════
# C1: environment_rng_state 存在且被验证
# ═══════════════════════════════════════════════════════════════════

def test_restore_hard_fails_invalid_environment_rng_state(tmp_path):
    """environment_rng_state is a required, VALIDATED field — an empty/
    malformed payload is refused before any state load."""
    d = dataclasses.replace(_v3_data(), environment_rng_state={})
    path = tmp_path / "ckpt.pt"
    CheckpointBuilder.save(d, path)
    contract, policy, critic = _train_loop_fixtures()
    calls = _spy_state_loads(policy, critic, _opt1(policy), _opt2(critic))
    with pytest.raises(CheckpointError, match="environment_rng_state"):
        restore_training_state(
            str(path), policy, critic, {}, _opt1(policy), _opt2(critic),
            runtime_identity=_runtime_identity(d),
        )
    assert calls == [], (
        "no state load may run on a malformed environment_rng_state"
    )


# ═══════════════════════════════════════════════════════════════════
# P0-1 eval 侧: 空 LoRA 集的 checksum 必须拒绝（sha256(b"") 曾绕过）
# ═══════════════════════════════════════════════════════════════════

def test_eval_artifact_rejects_empty_lora_set_checksum():
    """An empty LoRA parameter set evaluates to the BC baseline — the
    checksum of {} (sha256(b"")) is non-empty yet means NO LoRA, so it
    must be rejected explicitly (no warn-and-continue into PPO==BC)."""
    a = dataclasses.replace(_artifact(), lora_checksum=EMPTY_LORA_CHECKSUM)
    with pytest.raises(EvalArtifactError, match="empty LoRA"):
        validate_eval_artifact(a)


# ═══════════════════════════════════════════════════════════════════
# P1-4: 任一必需 trace 字段 NaN/Inf 均拒绝 artifact
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("mutate", [
    lambda ep: ep.trace["bc"][1].__setitem__("clearance", float("nan")),
    lambda ep: ep.trace["rl"][1].__setitem__("clearance", float("inf")),
    lambda ep: ep.trace["bc"][1].__setitem__("goal_dist", float("nan")),
    lambda ep: ep.trace["rl"][1].__setitem__("action_norm", float("inf")),
    lambda ep: ep.trace["bc"][1].__setitem__("elapsed", float("nan")),
    lambda ep: ep.trace["bc"][1]["pos"].__setitem__(0, float("nan")),
    lambda ep: ep.trace["rl"][1].__setitem__("z", float("inf")),
])
def test_eval_artifact_rejects_non_finite_trace_fields(mutate):
    """A NaN/Inf in ANY required trace field poisons every downstream
    metric — the artifact validator must refuse it (R3.5: NaN is never a
    valid metric)."""
    ep = _paired()
    mutate(ep)
    a = _single_role_artifact(ep)
    with pytest.raises(EvalArtifactError, match="finite"):
        validate_eval_artifact(a)


# ═══════════════════════════════════════════════════════════════════
# P1-3: bank 绑定（bank_id per-entry + 字节绑定 + artifact 记录）
# ═══════════════════════════════════════════════════════════════════

def _bank_entry(i: int = 0) -> PairedBankEntry:
    return PairedBankEntry(
        world_uid=f"w{i}",
        role="anticipatory",
        # R7-8 ruling (2026-08-24): the literal eval-only split is rl_val
        # (paired_bank_from_dict validates against SPLIT_NAMES).
        split="rl_val",
        admission_id=f"w{i}__anticipatory__rl_val",
        manifest_entry_hash=f"hash{i}",
        obstacle_json=f"w{i}.world.obstacles.json",
        generator_version="routeA_rl_v1.3",
        seed=42 + i,
        start=[-8.0, 0.0, 2.0],
        goal=[8.0, 0.0, 2.0],
        start_yaw=0.0,
        episode_config_hash=f"ep{i}",
        disturbance={"events": []},
        admission={
            "start_clearance": 1.0, "goal_clearance": 1.0,
            "reachability_pass": True, "path_length": 10.0,
            "path_stretch": 1.0,
        },
    )


def test_bank_entries_carry_bank_id_and_version():
    """Guide R3.4: every bank entry carries the bank id + version, and
    the top-level bank_id binds the entries' bytes."""
    bank = paired_bank_to_dict(
        [_bank_entry(0), _bank_entry(1)],
        source_hash="src123", creation_commit="abc",
    )
    assert bank["bank_version"] == BANK_VERSION
    assert bank["bank_id"]
    for entry in bank["entries"]:
        assert entry["bank_id"] == bank["bank_id"]
        assert entry["bank_version"] == BANK_VERSION


def test_bank_id_binds_entry_bytes():
    """Any CONTENT change to an entry (seed, start, goal, disturbance, ...)
    must break the binding — a stale bank_id is refused, never loaded
    (口径：canonical content hash — 语义变化绑定变化；格式变化不破坏，
    见 test_r3_rework2_negative.py::test_bank_id_survives_formatting_changes)."""
    bank = paired_bank_to_dict(
        [_bank_entry()], source_hash="src123", creation_commit="abc",
    )
    stale = copy.deepcopy(bank)
    stale["entries"][0]["seed"] += 1  # content changed, bank_id NOT
    with pytest.raises(ValueError, match="bank_id"):
        paired_bank_from_dict(stale)


def test_bank_missing_bank_id_rejected():
    bank = paired_bank_to_dict(
        [_bank_entry()], source_hash="src123", creation_commit="abc",
    )
    del bank["bank_id"]
    with pytest.raises(ValueError, match="bank_id"):
        paired_bank_from_dict(bank)


def _not_informative_outcomes():
    """R6-EVAL-2 (§11.5.3): 最小合法五件套 —— 零机会 NOT_INFORMATIVE
    形状。R6-EVAL-2 后 artifact 必须携带三组五件套（build_eval_artifact
    的默认 {} 在 validate 层 hard-fail），旧 fixture 统一补这一形状；
    gate 语义上零机会 → NOT_INFORMATIVE，绝无捏造 value。"""
    from diffusion_policy_3d.rl.dppo_uav.evaluator import CAPABILITY_METRICS

    return {
        group: {
            "opportunity_count": 0,
            "exposure_count": 0,
            "event_count": 0,
            "value": None,
            "validity_reason": OPPORTUNITY_INVALID_REASONS["no_opportunity"],
        }
        for group in CAPABILITY_METRICS
    }


def _artifact_with(**kwargs):
    """build_eval_artifact over the canonical three-role records with
    bank/density provenance bound."""
    from diffusion_policy_3d.rl.dppo_uav.evaluator import build_eval_artifact
    bank_id = kwargs.pop("bank_id", "bank123")
    records = _three_role_records(bank_id=bank_id)
    return build_eval_artifact(
        episodes=records,
        checkpoint_hash="ckpt123",
        config_hash="cfg456",
        contract_hash="contract789",
        manifest_hash="manifestabc",
        world_assets_hash="world123",
        code_hash="codedef",
        reward_version="v16",
        exec_steps=4,
        observation_invalid_count=0,
        missing_weight_keys=[],
        unexpected_weight_keys=[],
        lora_checksum="lora123",
        timestamp="2026-08-06T10:00:00",
        device="cuda:0",
        capability_metrics=compute_capability_metrics(records, _density()),
        capability_outcomes=kwargs.pop(
            "capability_outcomes", _not_informative_outcomes()),
        history_indices=[5, 6, 7],
        use_obstacle_projection=False,
        bank_id=bank_id,
        offline_density=kwargs.pop("offline_density", _density()),
        **kwargs,
    )


def test_eval_artifact_records_bank_and_density_provenance():
    """The artifact records which bank and which offline density it was
    built from — a density content change changes the recorded binding."""
    a = _artifact_with()
    d = a.to_dict()
    assert d["bank_id"] == "bank123"
    assert d["offline_density_sha256"]
    assert len(d["offline_density_sha256"]) == 64
    other = OfflineDensity(counts={(x, 0, 2): 7 for x in range(17)})
    a2 = _artifact_with(offline_density=other)
    assert a2.to_dict()["offline_density_sha256"] != d["offline_density_sha256"]


def test_eval_artifact_missing_bank_or_density_provenance_hard_fails():
    """An artifact that cannot say which bank/density produced it is not
    auditable — hard fail."""
    a = dataclasses.replace(
        _artifact(), bank_id="", offline_density_sha256=""
    )
    with pytest.raises(EvalArtifactError, match="bank"):
        validate_eval_artifact(a)


# ═══════════════════════════════════════════════════════════════════
# P1-4 指标层: NaN/Inf trace 输入 → 硬失败，不掩码
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("field", ["clearance", "goal_dist", "action_norm"])
def test_capability_metrics_rejects_nan_trace_field(field):
    """R3.5: NaN must never be masked into a valid-looking metric — the
    metric layer hard-fails on a non-finite required trace field."""
    bc = _straight(12)
    bc[5][field] = float("nan")
    records = [_record("anticipatory", bc, _straight(12),
                       bc_success=True, rl_success=True)]
    with pytest.raises(ValueError, match="finite"):
        compute_capability_metrics(records, _cap_density())


def test_capability_metrics_rejects_nan_pos():
    rl = _straight(12)
    rl[5]["pos"][1] = float("nan")
    records = [_record("anticipatory", _straight(12), rl,
                       bc_success=True, rl_success=True)]
    with pytest.raises(ValueError, match="finite"):
        compute_capability_metrics(records, _cap_density())


def test_capability_metrics_rejects_inf_value():
    bc = _straight(12)
    bc[3]["clearance"] = float("inf")
    records = [_record("anticipatory", bc, _straight(12),
                       bc_success=True, rl_success=True)]
    with pytest.raises(ValueError, match="finite"):
        compute_capability_metrics(records, _cap_density())


# ═══════════════════════════════════════════════════════════════════
# P1-1: eval 池接受 eval-only world；训练 sampler 边界仍拒绝
# ═══════════════════════════════════════════════════════════════════

def test_eval_pool_accepts_eval_only_worlds(tmp_path):
    """Design v1.3: '只评估不训练' worlds (w4/w8/w11/w12/w18 style) must
    be reachable by the canonical eval — the pool covers rl_eval buckets,
    not only rl_train."""
    uid = "routeA_rl_v1__anticipatory__rl_eval__f__000020"
    # R7-8 ruling (2026-08-24): the literal eval-only split is rl_val;
    # the uid string keeps the historical rl_eval marker. The eval pool
    # resolves each split in its OWN directory (worlds/val for rl_val),
    # so the obstacles file must be written to the val root and the pool
    # built with the val dir mapped (build_routea_eval_pool 3-arg).
    _write_obstacles(tmp_path, uid, root="val")
    entries = [_manifest_entry(uid, split="rl_val")]
    manifest = _write_manifest(tmp_path, entries)
    pool = build_routea_eval_pool(
        manifest, str(tmp_path / "train"), str(tmp_path / "val")
    )
    assert uid in {u for u, _ in pool}, (
        "eval-only worlds must be in the eval pool"
    )


def test_train_sampler_rejects_eval_only_specs():
    """The training sampler boundary stays closed: eval-only specs must
    never enter training (the '只评估不训练' boundary is one-way)."""
    contract = RLSystemContract()
    specs, _ = _sampler(contract)
    # R7-8 ruling (2026-08-24): the literal eval-only split is rl_val.
    bad = [dataclasses.replace(s, split="rl_val") for s in specs]
    with pytest.raises(ValueError, match="rl_val"):
        EpisodeTaskSampler(bad, DEFAULT_ROLE_TARGETS, base_seed=0)


def test_build_adapter_accepts_explicit_pool(tmp_path):
    """The eval entry can hand its own pool (eval buckets) to
    build_adapter — the pool is no longer train-pool-only."""
    world_file = tmp_path / "w_custom.world.obstacles.json"
    world_file.write_text(json.dumps(
        {"world_id": "w_custom", "obstacles": []}
    ))
    pool = [("w_custom", str(world_file))]
    adapter = build_adapter(RLSystemContract(), pool=pool)
    assert adapter._world_pool == pool
