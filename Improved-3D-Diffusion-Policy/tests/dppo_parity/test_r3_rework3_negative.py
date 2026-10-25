"""external-review #3 (RULING-7) 强制负向要求 — rework #3 negative matrix。

external-review #2 的负向矩阵覆盖"字段齐全但内容错误"，但其 fixture 把
"参数张量数"误当成"适配器数"（1 参数/适配器），与生产实现不一致——真实
insert_lora 的 n 个适配器产生 2n 个参数张量（A+B）。本文件按生产语义
重建 LoRA 计数轴断言，并新增 restorable 语义、PYTHONHASHSEED preflight、
schema v3 audit-only。全部 TDD RED 先行。
"""

import base64
import dataclasses
import json

import pytest
import torch

from diffusion_policy_3d.rl.dppo_uav.checkpoint import (
    CheckpointBuilder,
    CheckpointData,
    CheckpointError,
    EMPTY_LORA_CHECKSUM,
    SCHEMA_VERSION,
    build_environment_rng_state,
    build_lora_identity,
    lora_checksum_of,
    validate_checkpoint_lora_usable,
    validate_environment_rng_state,
    validate_lora_identity,
)

from test_r3_rework_negative import (
    _v3_data,
)

# ═══════════════════════════════════════════════════════════════════
# P0-1 (rework #3): LoRA 计数语义 — 参数张量轴独立于适配器模块轴
# ═══════════════════════════════════════════════════════════════════


def test_build_lora_identity_records_param_tensor_count():
    """build_lora_identity 必须记录 n_lora_param_tensors == len(lora_state)
    ——lora_state 按参数张量保存（n 适配器 = 2n keys，A+B），n_adaptors 是
    模块数，两者不可混用。"""
    state = {
        "lora_0": torch.zeros(8, 4),
        "lora_1": torch.zeros(4, 8),
        "lora_2": torch.zeros(8, 4),
        "lora_3": torch.zeros(4, 8),
    }
    lora_info = {
        "lora_rank": 4, "lora_alpha": 1.0, "n_adaptors": 2,
        "lora_params": [torch.zeros(8, 4), torch.zeros(4, 8),
                        torch.zeros(8, 4), torch.zeros(4, 8)],
    }
    ident = build_lora_identity(state, lora_info)
    assert ident.get("n_lora_param_tensors") == len(state), (
        "lora_identity must record the parameter-tensor count "
        f"(got {ident.get('n_lora_param_tensors')!r}, state has {len(state)})"
    )


def test_validate_lora_identity_requires_param_tensor_axis():
    """validate_lora_identity 的必填轴必须含 n_lora_param_tensors —— 缺
    该轴的 checkpoint 无法验证参数数一致性（pre-v4 audit-only）。"""
    data = _v3_data(lora_identity={
        "lora_rank": 4, "lora_alpha": 1.0, "n_adaptors": 2,
        "lora_checksum": "deadbeef",
    })
    with pytest.raises(CheckpointError, match="n_lora_param_tensors"):
        validate_lora_identity(data)


def test_lora_key_count_matches_param_tensors_not_adaptors():
    """真实生产关系：2 适配器 × (A,B) = 4 参数张量 = 4 keys —— 校验必须
    通过（不是拒绝）。旧实现用 n_adaptors=2 比较 4 keys 会误拒（P0-1）。"""
    state = {
        "lora_0": torch.zeros(8, 4),
        "lora_1": torch.zeros(4, 8),
        "lora_2": torch.zeros(8, 4),
        "lora_3": torch.zeros(4, 8),
    }
    data = _v3_data(
        lora_state=state,
        lora_identity={
            "lora_rank": 4, "lora_alpha": 1.0, "n_adaptors": 2,
            "n_lora_param_tensors": 4,
            "lora_checksum": lora_checksum_of(state),
        },
    )
    live = {
        "lora_rank": 4, "lora_alpha": 1.0, "n_adaptors": 2,
        "n_lora_param_tensors": 4,
        "lora_params": [torch.zeros(8, 4), torch.zeros(4, 8),
                        torch.zeros(8, 4), torch.zeros(4, 8)],
    }
    validate_checkpoint_lora_usable(data, live)  # must NOT raise


def test_lora_key_count_mismatch_param_tensors_hard_fails():
    """key 数 != saved n_lora_param_tensors 是内部不一致，validate 层
    硬失败（3 keys vs 4 param tensors）。"""
    state = {
        "lora_0": torch.zeros(8, 4),
        "lora_1": torch.zeros(4, 8),
        "lora_2": torch.zeros(8, 4),
    }
    data = _v3_data(
        lora_state=state,
        lora_identity={
            "lora_rank": 4, "lora_alpha": 1.0, "n_adaptors": 2,
            "n_lora_param_tensors": 4,
            "lora_checksum": lora_checksum_of(state),
        },
    )
    live = {
        "lora_rank": 4, "lora_alpha": 1.0, "n_adaptors": 2,
        "n_lora_param_tensors": 4,
        "lora_params": [torch.zeros(8, 4), torch.zeros(4, 8),
                        torch.zeros(8, 4), torch.zeros(4, 8)],
    }
    with pytest.raises(CheckpointError, match="lora_param_tensors"):
        validate_checkpoint_lora_usable(data, live)


# ═══════════════════════════════════════════════════════════════════
# P0-2 (rework #3): restorable 语义 — formal restore 拒绝不可恢复状态
# ═══════════════════════════════════════════════════════════════════


def test_validate_env_rng_rejects_restorable_false():
    """formal restore 必须拒绝 restorable=False —— 记录不可恢复的环境
    序列却照常恢复 = 静默继续一个不可复现的 run。"""
    data = _v3_data(environment_rng_state={
        "seed_source": "sampler_state",
        "python_hash_seed": "",
        "restorable": False,
    })
    with pytest.raises(CheckpointError, match="restorable"):
        validate_environment_rng_state(data)


def test_env_rng_builder_records_restorable_true(monkeypatch):
    """env RNG 修复（移除 hash(str(world_id)) + 固定 PYTHONHASHSEED）后，
    episode 序列跨进程确定，builder 必须记录 restorable=True（固定 seed
    下）。unset/random 时记录 False（validate 侧拒绝）。"""
    monkeypatch.setenv("PYTHONHASHSEED", "7")
    state = build_environment_rng_state()
    assert state["restorable"] is True
    assert state["python_hash_seed"] == "7"
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    assert build_environment_rng_state()["restorable"] is False


# ═══════════════════════════════════════════════════════════════════
# P0-2 (rework #3): PYTHONHASHSEED 入口 preflight
# ═══════════════════════════════════════════════════════════════════


def _require_fixed_python_hashseed():
    """Entrypoint preflight — imported lazily so the RED collection passes
    and the ImportError surfaces at runtime (feature missing today)."""
    from diffusion_policy_3d.rl.dppo_uav.entrypoint_guard import (
        require_fixed_python_hashseed,
    )

    return require_fixed_python_hashseed()


def test_python_hashseed_preflight_fails_when_unset(monkeypatch):
    """未设置 PYTHONHASHSEED → formal 入口 hard-fail（空值无法证明跨进程
    hash 一致，空==空比较是空验证）。"""
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    with pytest.raises(RuntimeError, match="PYTHONHASHSEED"):
        _require_fixed_python_hashseed()


def test_python_hashseed_preflight_fails_on_random(monkeypatch):
    """PYTHONHASHSEED=random → 每次解释器启动随机 → formal 入口 hard-fail。"""
    monkeypatch.setenv("PYTHONHASHSEED", "random")
    with pytest.raises(RuntimeError, match="PYTHONHASHSEED"):
        _require_fixed_python_hashseed()


def test_python_hashseed_preflight_passes_when_fixed(monkeypatch):
    """PYTHONHASHSEED 为固定十进制值 → preflight 通过。"""
    monkeypatch.setenv("PYTHONHASHSEED", "123")
    _require_fixed_python_hashseed()  # must NOT raise


# ═══════════════════════════════════════════════════════════════════
# Schema 封存：v3 降 audit-only
# ═══════════════════════════════════════════════════════════════════


def test_schema_v3_checkpoint_rejected_audit_only(tmp_path):
    """schema v4 升级后，v3 checkpoint 只能 audit/export，formal loader
    必须报 unsupported resume schema（与 v1/v2 同语义）。"""
    d = _v3_data()  # schema_version="3" fixture
    payload = {**dataclasses.asdict(d)}
    payload["schema_version"] = "3"
    payload["contract_serialized"] = base64.b64encode(
        payload["contract_serialized"]
    ).decode("ascii")
    path = tmp_path / "v3.pt"
    torch.save(payload, path)
    with pytest.raises(CheckpointError, match="unsupported resume schema"):
        CheckpointBuilder.load(path)
