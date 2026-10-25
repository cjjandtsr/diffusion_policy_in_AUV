"""external-review #2 (RULING-5) 强制负向要求 — "字段齐全但内容错误"
negative matrix。

rework #1 的 25 个 negative tests 覆盖"字段存在/缺失"与身份失配；
本文件覆盖"字段齐全但内容错误"：LoRA checksum 不绑定实际张量、
wired identity + 空 state、key 集/参数数/shape、env RNG 值级、reward_norm
完整 dict、episode outcome 类型/finite/范围、bank canonical-content-hash
口径。全部 TDD RED 先行（修复前因 feature missing 失败）。
"""

import dataclasses
import json

import pytest
import torch

from diffusion_policy_3d.rl.dppo_uav.checkpoint import (
    CheckpointBuilder,
    CheckpointError,
    EMPTY_LORA_CHECKSUM,
    lora_checksum_of,
    validate_checkpoint_lora_usable,
)
from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
    canonical_sha256,
)
from diffusion_policy_3d.rl.dppo_uav.evaluator import (
    EvalArtifactError,
    _bank_content,
    paired_bank_from_dict,
    paired_bank_to_dict,
    validate_eval_artifact,
)
from diffusion_policy_3d.rl.dppo_uav.train_loop import (
    restore_training_state,
)

from test_eval_artifact import _three_role_records
from test_r3_rework_negative import (
    _artifact_with,
    _bank_entry,
    _good_structured,
    _opt1,
    _opt2,
    _restore_ready,
    _runtime_identity,
    _spy_state_loads,
    _train_loop_fixtures,
    _v3_data,
    _wired_lora_data,
)

LIVE_WIRED = {
    "lora_rank": 4, "lora_alpha": 1.0, "n_adaptors": 2,
    "lora_params": [torch.zeros(1), torch.zeros(1)],
}


def _ready(d):
    """真实 payload 化：policy/critic/optimizer/RNG 用 live state 替换
    fixture 占位 — 使 RED 失败模式为 DID NOT RAISE（检查缺失时 restore
    完整走通）而非占位 payload 崩溃。"""
    _, policy, critic = _train_loop_fixtures()
    return _restore_ready(d, policy, critic, _opt1(policy), _opt2(critic))


# ═══════════════════════════════════════════════════════════════════
# P0-1 (rework #2): LoRA checksum 强一致 + wired-empty-state + key/shape
# ═══════════════════════════════════════════════════════════════════

def test_restore_hard_fails_lora_checksum_mismatch(tmp_path):
    """identity 声明的 lora_checksum 必须等于对实际 lora_state 张量重算的
    值 — saved=deadbeef vs actual 必须 hard-fail，零 state load。"""
    d = _ready(_wired_lora_data())  # identity checksum="deadbeef"
    path = tmp_path / "ckpt.pt"
    CheckpointBuilder.save(d, path)
    contract, policy, critic = _train_loop_fixtures()
    calls = _spy_state_loads(policy, critic, _opt1(policy), _opt2(critic))
    with pytest.raises(CheckpointError, match="checksum"):
        restore_training_state(
            str(path), policy, critic, LIVE_WIRED, _opt1(policy),
            _opt2(critic), runtime_identity=_runtime_identity(d),
        )
    assert calls == [], "no state load may run on a LoRA checksum mismatch"


def test_restore_hard_fails_wired_identity_with_empty_state(tmp_path):
    """checkpoint 的 identity 自称 wired (rank>0) 但 lora_state 为空 —
    checkpoint 内部不一致（identity 声明了它没携带的 LoRA），必须
    hard-fail（与"live wired over empty-state checkpoint"的轴失配不同）。"""
    d = _ready(_v3_data(lora_identity={
        "lora_rank": 4, "lora_alpha": 1.0, "n_adaptors": 2,
        "n_lora_param_tensors": 0,
        "lora_checksum": EMPTY_LORA_CHECKSUM,
    }))  # lora_state={} 保持默认
    path = tmp_path / "ckpt.pt"
    CheckpointBuilder.save(d, path)
    contract, policy, critic = _train_loop_fixtures()
    calls = _spy_state_loads(policy, critic, _opt1(policy), _opt2(critic))
    with pytest.raises(CheckpointError, match="LoRA"):
        restore_training_state(
            str(path), policy, critic, LIVE_WIRED, _opt1(policy),
            _opt2(critic), runtime_identity=_runtime_identity(d),
        )
    assert calls == [], (
        "no state load may run on a wired-identity + empty-state checkpoint"
    )


def test_lora_state_key_count_mismatch_hard_fails():
    """lora_state 的 key 数必须等于 identity 声明的 n_lora_param_tensors
    （参数张量数，非 n_adaptors 模块数 — rework #3 P0-1）— 3 个参数张量
    vs declared 2 是内部不一致，validate 层硬失败。"""
    state = {
        "lora_0": torch.tensor([0.1]),
        "lora_1": torch.tensor([0.2]),
        "lora_2": torch.tensor([0.3]),
    }
    data = _v3_data(
        lora_state=state,
        lora_identity={
            "lora_rank": 4, "lora_alpha": 1.0, "n_adaptors": 2,
            "n_lora_param_tensors": 2,
            "lora_checksum": lora_checksum_of(state),
        },
    )
    with pytest.raises(CheckpointError, match="key count"):
        validate_checkpoint_lora_usable(data, LIVE_WIRED)


def test_lora_state_tensor_shape_mismatch_hard_fails():
    """live 已插入 LoRA 参数（lora_params）时，每个 checkpoint 张量的
    shape 必须与 live 参数一致 — rank 决定 shape，shape 不同是不同 LoRA。"""
    state = {"lora_0": torch.zeros(8, 4), "lora_1": torch.zeros(4, 16)}
    data = _v3_data(
        lora_state=state,
        lora_identity={
            "lora_rank": 4, "lora_alpha": 1.0, "n_adaptors": 2,
            "n_lora_param_tensors": 2,
            "lora_checksum": lora_checksum_of(state),
        },
    )
    live = {
        "lora_rank": 4, "lora_alpha": 1.0, "n_adaptors": 2,
        "n_lora_param_tensors": 2,
        "lora_params": [torch.zeros(8, 8), torch.zeros(8, 16)],
    }
    with pytest.raises(CheckpointError, match="shape"):
        validate_checkpoint_lora_usable(data, live)


def test_lora_state_key_naming_mismatch_hard_fails():
    """直接单测命名契约（review P2-2）：键必须恰好是
    [lora_0, ..., lora_{n-1}] — lora_1/lora_2 违反契约，即使
    count/checksum/shape 全部正确也必须 hard-fail。"""
    state = {"lora_1": torch.tensor([0.1]), "lora_2": torch.tensor([0.2])}
    data = _v3_data(
        lora_state=state,
        lora_identity={
            "lora_rank": 4, "lora_alpha": 1.0, "n_adaptors": 2,
            "n_lora_param_tensors": 2,
            "lora_checksum": lora_checksum_of(state),
        },
    )
    with pytest.raises(CheckpointError, match="naming"):
        validate_checkpoint_lora_usable(data, LIVE_WIRED)


def test_restore_hard_fails_live_wired_without_lora_params(tmp_path):
    """live 声称 wired（axes 全部匹配）却拿不出 lora_params 时，形状
    校验无从进行 — 必须 hard-fail（review P2-1），不能静默跳过。"""
    state = {"lora_0": torch.tensor([0.1]), "lora_1": torch.tensor([0.2])}
    d = _ready(_v3_data(
        lora_state=state,
        lora_identity={
            "lora_rank": 4, "lora_alpha": 1.0, "n_adaptors": 2,
            "n_lora_param_tensors": 2,
            "lora_checksum": lora_checksum_of(state),
        },
    ))
    path = tmp_path / "ckpt.pt"
    CheckpointBuilder.save(d, path)
    contract, policy, critic = _train_loop_fixtures()
    live = {"lora_rank": 4, "lora_alpha": 1.0, "n_adaptors": 2}  # 无 lora_params
    calls = _spy_state_loads(policy, critic, _opt1(policy), _opt2(critic))
    with pytest.raises(CheckpointError, match="lora_params"):
        restore_training_state(
            str(path), policy, critic, live, _opt1(policy), _opt2(critic),
            runtime_identity=_runtime_identity(d),
        )
    assert calls == [], "no state load may run when shapes are unverifiable"


# ═══════════════════════════════════════════════════════════════════
# P0-2 (rework #2): environment_rng_state 值级验证
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("bad, match", [
    ({"seed_source": "garbage"}, "seed_source"),
    ({"restorable": "yes"}, "restorable"),
    ({"python_hash_seed": "different"}, "python_hash_seed"),
])
def test_restore_hard_fails_environment_rng_bad_values(bad, match,
                                                       tmp_path):
    """键齐全但值任意不再被接受：seed_source 枚举、restorable 类型、
    python_hash_seed 逐字符匹配（运行时未设置 PYTHONHASHSEED → 保存值
    必须为 ""）。"""
    d = _ready(_v3_data(environment_rng_state={
        "seed_source": "sampler_state",
        "python_hash_seed": "",
        "restorable": True,
        **bad,
    }))
    path = tmp_path / "ckpt.pt"
    CheckpointBuilder.save(d, path)
    contract, policy, critic = _train_loop_fixtures()
    calls = _spy_state_loads(policy, critic, _opt1(policy), _opt2(critic))
    with pytest.raises(CheckpointError, match=match):
        restore_training_state(
            str(path), policy, critic, {}, _opt1(policy), _opt2(critic),
            runtime_identity=_runtime_identity(d),
        )
    assert calls == [], (
        "no state load may run on a bad environment_rng_state value"
    )


# ═══════════════════════════════════════════════════════════════════
# P1-1 (rework #2): reward_norm 完整 canonical dict 对比
# ═══════════════════════════════════════════════════════════════════

def test_restore_hard_fails_reward_norm_semantics_mismatch(tmp_path):
    """reward_norm_state 只比 version 时 source=evil/method=ema/params 可
    任意通过 — 完整 canonical dict 对比必须拒绝任何语义偏差。"""
    d = _ready(_v3_data(reward_norm_state={
        "version": "v1.3", "source": "evil", "method": "ema",
        "params": {"mean": 99.0},
    }))
    path = tmp_path / "ckpt.pt"
    CheckpointBuilder.save(d, path)
    contract, policy, critic = _train_loop_fixtures()
    calls = _spy_state_loads(policy, critic, _opt1(policy), _opt2(critic))
    with pytest.raises(CheckpointError, match="reward_norm"):
        restore_training_state(
            str(path), policy, critic, {}, _opt1(policy), _opt2(critic),
            runtime_identity=_runtime_identity(d),
            structured_identity=_good_structured(),
        )
    assert calls == [], (
        "no state load may run on a reward_norm semantic mismatch"
    )


# ═══════════════════════════════════════════════════════════════════
# P1-2 (rework #2): episode outcome 类型/finite/范围
# ═══════════════════════════════════════════════════════════════════

def test_eval_artifact_rejects_non_finite_episode_return():
    """outcome 层 return=NaN 必须拒绝（trace 层已正确，outcome 层未闭合）"""
    records = [
        dataclasses.replace(ep, rl={**ep.rl, "return": float("nan")})
        for ep in _three_role_records()
    ]
    bad = dataclasses.replace(_artifact_with(), episodes=records)
    with pytest.raises(EvalArtifactError, match="return"):
        validate_eval_artifact(bad)


@pytest.mark.parametrize("mutate, match", [
    (lambda d: {**d, "success": "yes"}, "success"),
    (lambda d: {**d, "timeout": 1}, "timeout"),
    (lambda d: {**d, "steps": -1}, "steps"),
    (lambda d: {**d, "return": "x"}, "return"),
])
def test_eval_artifact_rejects_episode_outcome_bad_values(mutate, match):
    """success/collision/timeout 必须 bool；steps 必须非负 int；return
    必须有限数值 — 类型/范围错误拒绝，不做宽松比较。"""
    records = [
        dataclasses.replace(ep, bc=mutate(ep.bc), rl=mutate(ep.rl))
        for ep in _three_role_records()
    ]
    bad = dataclasses.replace(_artifact_with(), episodes=records)
    with pytest.raises(EvalArtifactError, match=match):
        validate_eval_artifact(bad)


def test_eval_artifact_accepts_huge_int_return():
    """任意精度 int 数学上恒有限（review P2-3）：float(ret) 对超大 int
    会 OverflowError — 有限性检查必须在不做 float 转换的前提下判定。"""
    records = [
        dataclasses.replace(
            ep, rl={**ep.rl, "return": 10 ** 400},
        )
        for ep in _three_role_records()
    ]
    good = dataclasses.replace(_artifact_with(), episodes=records)
    validate_eval_artifact(good)  # 不抛 OverflowError / EvalArtifactError


# ═══════════════════════════════════════════════════════════════════
# bank 口径 (rework #2): canonical content hash — 语义变化绑定变化，
# 格式变化（缩进/键序）不破坏绑定（裁决"任一字节变化"表述修正的实证）
# ═══════════════════════════════════════════════════════════════════

def test_bank_id_survives_formatting_changes():
    """bank_id 绑定 canonical content hash — 纯格式变化（缩进）不破坏
    绑定；paired_bank_from_dict 仍接受重排后的 bank。"""
    bank = paired_bank_to_dict(
        [_bank_entry()], source_hash="src123", creation_commit="abc",
    )
    reformatted = json.loads(json.dumps(bank, indent=4))
    assert canonical_sha256(_bank_content(reformatted)) == (
        canonical_sha256(_bank_content(bank))
    )
    assert paired_bank_from_dict(reformatted)  # 不抛 ValueError
