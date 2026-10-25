"""R3 rework #4 — 负向 RED 测试（external-review #4 裁决，2026-08-13）。

裁决 P1：RULING-7 要求三方一致

    len(lora_state) == saved n_lora_param_tensors == len(live lora_params)

但 validate_checkpoint_lora_usable 只实现了前两方（len(state_keys) ==
declared），未比较 live 侧参数张量数。saved=4/live=2 被 validator 接受，
完整 restore 在 restore_lora_state 才抛 ValueError —— 此时 policy 已加载
（部分 mutation），违反 validation-first 原子恢复合同。反向 saved=2/
live=4 则 KeyError（非 CheckpointError）。

本文件把两个方向都钉成测试（测试先行，全部 RED）：
  - direct validator：两方向都必须 CheckpointError；
  - 完整 restore_training_state + state-load spy：两方向都必须
    CheckpointError 且 spy == []（policy/critic/两个 optimizer 零 load）。
"""
import pytest
import torch

from test_checkpoint import _runtime_identity, _train_loop_fixtures
from test_r3_rework_negative import (
    _good_structured, _opt1, _opt2, _restore_ready, _spy_state_loads,
    _v3_data,
)
from diffusion_policy_3d.rl.dppo_uav.checkpoint import (
    CheckpointBuilder, CheckpointError, lora_checksum_of,
    validate_checkpoint_lora_usable,
)
from diffusion_policy_3d.rl.dppo_uav.train_loop import restore_training_state


def _live(n_params: int) -> dict:
    """Live wiring with n_params parameter tensors. Shapes all match the
    saved lora_<i> tensors, so ONLY the count can (wrongly) pass."""
    return {
        "lora_rank": 4,
        "lora_alpha": 1.0,
        "n_adaptors": 2,
        "lora_params": [torch.ones(2, 3) * v for v in range(1, n_params + 1)],
    }


def _data(n_saved: int):
    """CheckpointData whose lora_state holds n_saved contract-named
    tensors and whose identity declares exactly that count (real checksum
    over the saved tensors)."""
    state = {f"lora_{i}": torch.ones(2, 3) * (i + 1) for i in range(n_saved)}
    identity = {
        "lora_rank": 4, "lora_alpha": 1.0, "n_adaptors": 2,
        "n_lora_param_tensors": n_saved,
        "lora_checksum": lora_checksum_of(state),
    }
    return _v3_data(lora_state=state, lora_identity=identity)


@pytest.mark.parametrize("n_saved, n_live", [(4, 2), (2, 4)])
def test_validate_lora_usable_rejects_live_count_mismatch(n_saved, n_live):
    """Direct validator: live parameter-tensor count must equal the
    checkpoint-declared count — both directions hard-fail."""
    with pytest.raises(CheckpointError, match="n_lora_param_tensors"):
        validate_checkpoint_lora_usable(_data(n_saved), _live(n_live))


@pytest.mark.parametrize("n_saved, n_live", [(4, 2), (2, 4)])
def test_restore_rejects_live_count_mismatch_before_any_load(
        n_saved, n_live, tmp_path):
    """Full restore path: the count mismatch must be rejected inside
    validate_checkpoint_lora_usable — CheckpointError, and the state-load
    spy must stay EMPTY (no policy/critic/optimizer load may run)."""
    data = _data(n_saved)
    contract, policy, critic = _train_loop_fixtures()
    opt1, opt2 = _opt1(policy), _opt2(critic)
    data = _restore_ready(data, policy, critic, opt1, opt2)
    path = tmp_path / "ckpt.pt"
    CheckpointBuilder.save(data, path)
    calls = _spy_state_loads(policy, critic, opt1, opt2)
    with pytest.raises(CheckpointError, match="n_lora_param_tensors"):
        restore_training_state(
            str(path), policy, critic, _live(n_live), opt1, opt2,
            runtime_identity=_runtime_identity(data),
            structured_identity=_good_structured(),
        )
    assert calls == [], (
        f"no state load may run on a live-count mismatch (saved={n_saved}, "
        f"live={n_live}); got {calls}"
    )
