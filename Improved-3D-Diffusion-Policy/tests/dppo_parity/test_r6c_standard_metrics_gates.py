"""R6-C (§11.4): 标准指标、学习信号和门禁 — 统一可选统计结构
(OptionalStatistic 两态合同)、outcome 计数与不变量、严格 JSON 与
schema 迁移、per-denoise 五 index hard-fail。

RED phase (HEAD 52ce1eb + R6-A/B 工作树): OptionalStatistic 工厂、
outcome 计数/不变量、交叉检查、per-denoise 严格化、严格 JSON 与
schema_version 尚不存在 —— 相应测试以 ImportError/AttributeError/
断言失败呈现。已成立的既有语义（cond_adv 只统计 terminal、role
conditioned 有样本才有均值）作为回归约束同文件保留。
"""

import json
import math

import numpy as np
import pytest
import torch


# ═══════════════════════════════════════════════════════════════════
# §11.4.1 统一可选统计结构 — 两态合同
# ═══════════════════════════════════════════════════════════════════


def test_red1_optional_statistic_two_state_contract():
    """§11.4.1: 两态合同 — count>0 → finite value/valid=True/reason=None；
    count==0 → value=None/valid=False/冻结枚举。禁止第三种状态。"""
    from diffusion_policy_3d.rl.dppo_uav.metrics import optional_statistic

    # count==0 → 合法 null + 冻结枚举
    empty = optional_statistic([], "no_success_terminal")
    assert empty == {
        "sample_count": 0, "value": None, "valid": False,
        "invalid_reason": "no_success_terminal",
    }
    # count>0 → finite mean
    full = optional_statistic([1.0, 3.0, 5.0], "no_success_terminal")
    assert full["sample_count"] == 3
    assert full["value"] == 3.0
    assert full["valid"] is True
    assert full["invalid_reason"] is None
    # 禁止态：非 finite 值在 count>0 时绝不产出
    with pytest.raises(ValueError):
        optional_statistic([1.0, float("nan")], "no_success_terminal")
    with pytest.raises(ValueError):
        optional_statistic([float("inf")], "no_success_terminal")
    # 冻结枚举：非法 reason 拒绝
    with pytest.raises(ValueError):
        optional_statistic([], "free_form_reason")


def test_red2_cond_adv_empty_categories_are_null_not_nan():
    """§11.4.1/§11.4.5: 空 success/collision 集合产生合法
    null/count=0，绝不产生 NaN（现状是 math.nan —— RED 断言失败）。"""
    from diffusion_policy_3d.rl.dppo_uav.metrics import (
        success_collision_role_conditioned_advantages,
    )

    adv = torch.tensor([0.1, 0.2, 0.3, 0.4])
    term = np.array([False, False, False, True])
    out = success_collision_role_conditioned_advantages(
        advantages=adv, terminated=term,
        success=np.zeros(4, dtype=bool), collision=np.zeros(4, dtype=bool),
        roles=["reach", "reach", "avoid", "reach"],
    )
    for key in ("success_conditioned_advantage",
                "collision_conditioned_advantage"):
        stat = out[key]
        assert stat["sample_count"] == 0, stat
        assert stat["value"] is None, stat
        assert stat["valid"] is False, stat
    assert out["success_conditioned_advantage"][
        "invalid_reason"] == "no_success_terminal"
    assert out["collision_conditioned_advantage"][
        "invalid_reason"] == "no_collision_terminal"
    # 全字段递归检查：任何地方都不允许 NaN
    def _no_nan(v):
        if isinstance(v, dict):
            for x in v.values():
                _no_nan(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                _no_nan(x)
        elif isinstance(v, float):
            assert not math.isnan(v), f"NaN leaked: {v}"
    _no_nan(out)


def test_red3_cond_adv_two_success_oracle():
    """§11.4.5: 有 2 个 success terminals 时 count=2 且均值与手算
    oracle 一致（只统计 terminal transition，现状语义保留）。"""
    from diffusion_policy_3d.rl.dppo_uav.metrics import (
        success_collision_role_conditioned_advantages,
    )

    adv = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    term = np.array([False, True, False, True, True, False])
    success = np.array([False, True, False, False, True, False])
    collision = np.array([False, False, False, True, False, False])
    out = success_collision_role_conditioned_advantages(
        advantages=adv, terminated=term, success=success,
        collision=collision, roles=["r"] * 6,
    )
    succ = out["success_conditioned_advantage"]
    coll = out["collision_conditioned_advantage"]
    assert succ["sample_count"] == 2
    assert succ["value"] == (2.0 + 5.0) / 2  # hand-computed oracle
    assert succ["valid"] is True
    assert coll["sample_count"] == 1
    assert coll["value"] == 4.0
    # 非 terminal 的 success 标志（错误数据）绝不进统计
    adv2 = torch.tensor([1.0, 2.0, 3.0])
    out2 = success_collision_role_conditioned_advantages(
        advantages=adv2, terminated=np.array([False, False, False]),
        success=np.array([True, False, False]),
        collision=np.zeros(3, dtype=bool), roles=["r"] * 3,
    )
    assert out2["success_conditioned_advantage"]["sample_count"] == 0


def test_red4_explained_variance_constant_target_null():
    """§11.4.1/§11.4.5: constant value target → explained variance
    合法 null（constant_value_target），并保留 target_variance=0。"""
    from diffusion_policy_3d.rl.dppo_uav.metrics import explained_variance

    out = explained_variance(
        np.array([2.0, 2.0, 2.0]), np.array([1.0, 1.0, 1.0])
    )
    assert out["target_variance"] == 0.0
    stat = {k: out[k] for k in
            ("sample_count", "value", "valid", "invalid_reason")}
    assert stat["sample_count"] == 0
    assert stat["value"] is None
    assert stat["valid"] is False
    assert stat["invalid_reason"] == "constant_value_target"


def test_red5_explained_variance_value_oracle():
    """§11.4.1: var(y_true)>0 → value 与手算 oracle 一致（官方
    semantics: 1 - var(err)/var(y_true)），target_variance == var(y_true)。"""
    from diffusion_policy_3d.rl.dppo_uav.metrics import explained_variance

    y = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    p = np.array([0.1, 0.9, 2.1, 3.2], dtype=np.float64)
    out = explained_variance(y, p)
    expected = 1.0 - float(np.var(y - p)) / float(np.var(y))
    assert out["target_variance"] == float(np.var(y))
    assert out["sample_count"] == 4
    assert math.isclose(out["value"], expected, rel_tol=1e-12)
    assert out["valid"] is True


def test_red6_unexecuted_grad_ratio_statistic():
    """§11.4.1: executed/unexecuted gradient ratio → OptionalStatistic；
    denominator<=1e-8（F10 mask 完好）→ zero_executed_gradient_denominator
    null，绝不产出 ~1e8 伪信号或 NaN。"""
    from diffusion_policy_3d.rl.dppo_uav.metrics import (
        executed_unexecuted_ratio_statistic,
    )

    # F10 mask 完好：unexecuted 恒 0 → 合法 null
    null = executed_unexecuted_ratio_statistic(
        executed_mean=1.0, unexecuted_mean=0.0,
    )
    assert null["sample_count"] == 0
    assert null["value"] is None
    assert null["valid"] is False
    assert null["invalid_reason"] == "zero_executed_gradient_denominator"
    # denominator 有效 → finite ratio
    full = executed_unexecuted_ratio_statistic(
        executed_mean=4.0, unexecuted_mean=2.0,
    )
    assert full["sample_count"] == 1
    assert full["value"] == 2.0
    assert full["valid"] is True


def test_red7_per_denoise_mean_missing_index_hard_fails():
    """§11.4.1: per-denoise KL/clipfrac 不属于可选统计 —— 每个
    fine-tune denoise index 必须 sample_count>0 且 finite；缺 index
    hard-fail（现状 dict comprehension 产 NaN —— RED 断言失败）。"""
    from diffusion_policy_3d.rl.dppo_uav.metrics import per_denoise_mean_dict

    out = per_denoise_mean_dict(
        {0: [0.01, 0.02], 1: [0.03], 2: [0.04]}, "per_denoise_approx_kl"
    )
    assert out[0] == 0.015 and out[2] == 0.04
    with pytest.raises(RuntimeError) as e:
        per_denoise_mean_dict(
            {0: [0.01], 1: [], 2: [0.03]}, "per_denoise_approx_kl"
        )
    assert "index 1" in str(e.value) or "denoise" in str(e.value), e.value


# ═══════════════════════════════════════════════════════════════════
# §11.4.2 outcome 与计数一致性
# ═══════════════════════════════════════════════════════════════════


def test_red8_outcome_counts_full_oracle():
    """§11.4.2: 每 batch 必须形成全部可复算 outcome 计数，与手算
    oracle 一致。含 success/collision/timeout/OOB/altitude/loop-trunc
    /cut 混合场景。"""
    from diffusion_policy_3d.rl.dppo_uav.metrics import compute_outcome_counts

    n = 12
    terminated = np.zeros(n, dtype=bool)
    truncated = np.zeros(n, dtype=bool)
    done_reasons = ["none"] * n
    successes = np.zeros(n, dtype=bool)
    collisions = np.zeros(n, dtype=bool)
    roles = ["reach"] * n
    rollout_cut = np.zeros(n, dtype=bool)
    # t2: success (env terminal); t5: collision; t8: timeout;
    # t10: loop-trunc (truncated, done_reason none); t11: rollout cut
    terminated[2] = True
    successes[2] = True
    terminated[5] = True
    collisions[5] = True
    truncated[8] = True
    done_reasons[8] = "timeout"
    truncated[10] = True
    truncated[11] = True
    rollout_cut[11] = True
    out = compute_outcome_counts(
        terminated=terminated, truncated=truncated,
        done_reasons=done_reasons, successes=successes,
        collisions=collisions, roles=roles, rollout_cut=rollout_cut,
    )
    assert out["transition_count_total"] == 12
    # terminal = env terminal | (truncated & ~cut): t2, t5, t8, t10
    assert out["terminal_transition_count"] == 4
    assert out["success_event_count"] == 1
    assert out["collision_event_count"] == 1
    assert out["timeout_terminal_count"] == 1
    assert out["oob_terminal_count"] == 0
    assert out["altitude_terminal_count"] == 0
    assert out["rollout_cut_count"] == 1
    assert out["per_role_transition_count"] == {"reach": 12}
    assert out["per_role_completed_episode_count"] == {"reach": 4}


def test_red9_outcome_cut_never_counts_terminal():
    """§11.4.2 不变量 2: rollout cut 不增加 completed/terminal count
    —— cut transition 的 truncated=True 只是 GAE bootstrap，不算
    terminal。"""
    from diffusion_policy_3d.rl.dppo_uav.metrics import compute_outcome_counts

    n = 3
    out = compute_outcome_counts(
        terminated=np.zeros(n, dtype=bool),
        truncated=np.ones(n, dtype=bool),
        done_reasons=["none"] * n, successes=np.zeros(n, dtype=bool),
        collisions=np.zeros(n, dtype=bool), roles=["r"] * n,
        rollout_cut=np.array([True, True, True], dtype=bool),
    )
    assert out["terminal_transition_count"] == 0
    assert out["rollout_cut_count"] == 3


def test_red10_outcome_role_sums_equal_total():
    """§11.4.2 不变量: role transition counts 之和等于 buffer
    transitions。"""
    from diffusion_policy_3d.rl.dppo_uav.metrics import compute_outcome_counts

    n = 10
    roles = ["reach"] * 6 + ["avoid"] * 4
    out = compute_outcome_counts(
        terminated=np.zeros(n, dtype=bool),
        truncated=np.zeros(n, dtype=bool),
        done_reasons=["none"] * n, successes=np.zeros(n, dtype=bool),
        collisions=np.zeros(n, dtype=bool), roles=roles,
        rollout_cut=np.zeros(n, dtype=bool),
    )
    assert sum(out["per_role_transition_count"].values()) == n
    assert out["per_role_transition_count"] == {"reach": 6, "avoid": 4}
    assert sum(out["per_role_completed_episode_count"].values()) == 0


def test_red11_verify_invariants_catches_duplicate_terminal():
    """§11.4.2 不变量 1: 同一 episode 绝不可能有两个 terminal
    transitions（重复计数）→ verify hard-fail。"""
    from diffusion_policy_3d.rl.dppo_uav.metrics import (
        compute_outcome_counts,
        verify_outcome_invariants,
    )

    n = 4
    terminated = np.zeros(n, dtype=bool)
    truncated = np.zeros(n, dtype=bool)
    # episode "e1" 出现两次 terminal —— 数据损坏，必须被抓
    terminated[[1, 3]] = True
    counts = compute_outcome_counts(
        terminated=terminated, truncated=truncated,
        done_reasons=["none"] * n, successes=np.zeros(n, dtype=bool),
        collisions=np.zeros(n, dtype=bool), roles=["r"] * n,
        rollout_cut=np.zeros(n, dtype=bool),
    )
    with pytest.raises(RuntimeError) as e:
        verify_outcome_invariants(
            counts,
            episode_ids=["e1", "e1", "e1", "e1"],
            terminated=terminated, truncated=truncated,
            rollout_cut=np.zeros(n, dtype=bool),
        )
    assert "e1" in str(e.value), e.value


def test_red12_verify_invariants_cut_must_be_truncated():
    """§11.4.2 不变量 2: rollout_cut 标记的 transition 必须
    truncated=True 且 terminated=False（cut 语义：GAE bootstrap，
    不关闭 episode）。"""
    from diffusion_policy_3d.rl.dppo_uav.metrics import (
        compute_outcome_counts,
        verify_outcome_invariants,
    )

    n = 2
    terminated = np.zeros(n, dtype=bool)
    truncated = np.zeros(n, dtype=bool)
    # cut 标记了 0 号 transition 但 truncated=False —— 非法 cut
    counts = compute_outcome_counts(
        terminated=terminated, truncated=truncated,
        done_reasons=["none"] * n, successes=np.zeros(n, dtype=bool),
        collisions=np.zeros(n, dtype=bool), roles=["r"] * n,
        rollout_cut=np.array([True, False], dtype=bool),
    )
    with pytest.raises(RuntimeError):
        verify_outcome_invariants(
            counts,
            episode_ids=["e1", "e2"],
            terminated=terminated, truncated=truncated,
            rollout_cut=np.array([True, False], dtype=bool),
        )


def test_red13_terminal_reward_cross_check_success():
    """§11.4.2 不变量 4: success terminal 的 reward 必须携带 reach
    bonus（>= reach_reward/2）；缺失 → hard-fail 并报告 episode id。"""
    from diffusion_policy_3d.rl.dppo_uav.metrics import (
        cross_check_terminal_rewards,
    )

    rewards = np.array([0.1, 10.0, 95.0, 0.2], dtype=np.float64)
    terminated = np.array([False, True, True, False], dtype=bool)
    successes = np.array([False, True, True, False], dtype=bool)
    collisions = np.zeros(4, dtype=bool)
    episode_ids = ["e1", "e2", "e3", "e4"]
    with pytest.raises(RuntimeError) as e:
        cross_check_terminal_rewards(
            rewards=rewards, terminated=terminated, successes=successes,
            collisions=collisions, episode_ids=episode_ids,
            reach_reward=100.0,
        )
    assert "e2" in str(e.value), e.value
    assert "reward" in str(e.value), e.value


def test_red14_terminal_reward_cross_check_collision():
    """§11.4.2 不变量 4: collision terminal 的 reward 必须携带碰撞
    penalty（<= 0；r_progress 单步上界远小于 20）→ hard-fail。"""
    from diffusion_policy_3d.rl.dppo_uav.metrics import (
        cross_check_terminal_rewards,
    )

    rewards = np.array([0.1, 5.0, -25.0], dtype=np.float64)
    terminated = np.array([False, True, True], dtype=bool)
    successes = np.zeros(3, dtype=bool)
    collisions = np.array([False, True, True], dtype=bool)
    episode_ids = ["e1", "e2", "e3"]
    with pytest.raises(RuntimeError) as e:
        cross_check_terminal_rewards(
            rewards=rewards, terminated=terminated, successes=successes,
            collisions=collisions, episode_ids=episode_ids,
            reach_reward=100.0,
        )
    assert "e2" in str(e.value), e.value


def test_red15_terminal_reward_cross_check_clean_pass():
    """§11.4.2 不变量 4 合法路径：success 带 bonus、collision 带
    penalty → PASS；cut/普通 transition 不受检查。"""
    from diffusion_policy_3d.rl.dppo_uav.metrics import (
        cross_check_terminal_rewards,
    )

    rewards = np.array([0.1, 97.0, -18.0, 0.2], dtype=np.float64)
    terminated = np.array([False, True, True, False], dtype=bool)
    successes = np.array([False, True, False, False], dtype=bool)
    collisions = np.array([False, False, True, False], dtype=bool)
    # 无异常即 PASS
    cross_check_terminal_rewards(
        rewards=rewards, terminated=terminated, successes=successes,
        collisions=collisions, episode_ids=["e1", "e2", "e3", "e4"],
        reach_reward=100.0,
    )


# ═══════════════════════════════════════════════════════════════════
# §11.4.3 标准 JSON 与 schema 迁移
# ═══════════════════════════════════════════════════════════════════

NESTED_NAN_METRICS = {
    "approx_kl": 0.05,
    "per_denoise_approx_kl": {0: 0.01, 1: float("nan")},
    "success_conditioned_advantage": {"sample_count": 0, "value": None,
                                      "valid": False,
                                      "invalid_reason": "no_success_terminal"},
}


def test_red16_persist_nested_nan_hard_fails(tmp_path):
    """§11.4.3: 严格 JSON 禁止先把 NaN 写出再由 reader 特判 ——
    任意嵌套 NaN/Inf 必须 hard-fail（现状 NAN_SENTINEL 放行 →
    RED 断言失败）。"""
    from scripts.train_dppo_uav_official import persist_batch_metrics

    out = tmp_path / "metrics.jsonl"
    res = {
        "env_steps": 8, "checkpoint_path": None, "checkpoint_kind": "resume",
        "completed_episodes": 1, "voided_episodes": 0, "routing": {},
        "metrics": NESTED_NAN_METRICS,
    }
    with pytest.raises(ValueError):
        persist_batch_metrics(out, 1, res)


def test_red17_persist_standard_json_parseable(tmp_path):
    """§11.4.3: formal metrics.jsonl 每一行可由标准 JSON parser
    读取（NaN/Infinity 字面量一律拒绝）——现状写 NaN literal →
    parse_constant 检查失败。"""
    from scripts.train_dppo_uav_official import persist_batch_metrics

    out = tmp_path / "metrics.jsonl"
    res = {
        "env_steps": 8, "checkpoint_path": None, "checkpoint_kind": "resume",
        "completed_episodes": 1, "voided_episodes": 0, "routing": {},
        "metrics": {"approx_kl": 0.05, "policy_loss_mean": 0.3},
    }
    persist_batch_metrics(out, 1, res)
    line = out.read_text().splitlines()[0]

    def _reject(constant):
        raise ValueError(f"non-standard JSON constant: {constant}")

    record = json.loads(line, parse_constant=_reject)  # must not raise
    assert record["metrics"]["approx_kl"] == 0.05


def test_red18_persist_record_has_schema_version(tmp_path):
    """§11.4.3: metrics record 增加 schema_version（formal current
    run writer 只写新 schema）。"""
    from scripts.train_dppo_uav_official import persist_batch_metrics

    out = tmp_path / "metrics.jsonl"
    res = {
        "env_steps": 8, "checkpoint_path": None, "checkpoint_kind": "resume",
        "completed_episodes": 1, "voided_episodes": 0, "routing": {},
        "metrics": {"approx_kl": 0.05},
    }
    persist_batch_metrics(out, 1, res)
    record = json.loads(out.read_text().splitlines()[0])
    assert "schema_version" in record, record
    assert isinstance(record["schema_version"], str)
    assert record["schema_version"]


# ═══════════════════════════════════════════════════════════════════
# §11.4.4 gate 分层 — ENGINE_HARD_GATE 新增校验
# ═══════════════════════════════════════════════════════════════════


def test_red19_gate_optional_statistic_third_state_rejected():
    """§11.4.1/§11.4.4: 禁止第三种状态 —— count>0 但 value=None
    → gate FAIL（现状 reader 无法识别该结构 → ImportError RED）。"""
    from scripts.g7_gate_report import validate_optional_statistic

    problems: list = []
    validate_optional_statistic(
        {"sample_count": 2, "value": None, "valid": True,
         "invalid_reason": None},
        "success_conditioned_advantage", problems,
    )
    assert problems, "third state (count>0, value=None) must FAIL the gate"


def test_red20_gate_optional_statistic_contract_fields():
    """§11.4.1/§11.4.4: 字段集固定 {sample_count, value, valid,
    invalid_reason}；缺字段/多余字段/非 finite value → FAIL。"""
    from scripts.g7_gate_report import validate_optional_statistic

    problems: list = []
    validate_optional_statistic(
        {"sample_count": 2, "value": float("inf")},
        "collision_conditioned_advantage", problems,
    )
    assert problems, "missing fields / non-finite value must FAIL"
    problems2: list = []
    validate_optional_statistic(
        {"sample_count": 0, "value": 0.0, "valid": False,
         "invalid_reason": "no_success_terminal"},
        "collision_conditioned_advantage", problems2,
    )
    assert problems2, "count=0 with value=0 is the forbidden third state"
    problems3: list = []
    validate_optional_statistic(
        {"sample_count": 0, "value": None, "valid": False,
         "invalid_reason": "free_form"},
        "collision_conditioned_advantage", problems3,
    )
    assert problems3, "free-form invalid_reason must FAIL"


def test_red21_gate_outcome_conditioned_mismatch_fails():
    """§11.4.2 不变量 3/§11.4.5: outcome count>0 但 conditioned
    sample_count=0 → gate FAIL（学习信号与 outcome 脱节）。
    REWORK 2026-08-19 (P1-1/P1-2): check_outcome_invariants 签名改为
    接收完整 record（跨层不变量需要 record.completed_episodes），
    本测试构造 record 包装。"""
    from scripts.g7_gate_report import check_outcome_invariants

    m = {
        "outcome_counts": {
            "transition_count_total": 10,
            "terminal_transition_count": 0,
            "success_event_count": 1, "collision_event_count": 0,
            "rollout_cut_count": 0,
            "per_role_transition_count": {"reach": 10},
            "per_role_completed_episode_count": {},
        },
        "success_conditioned_advantage": {
            "sample_count": 0, "value": None, "valid": False,
            "invalid_reason": "no_success_terminal",
        },
        "collision_conditioned_advantage": {
            "sample_count": 0, "value": None, "valid": False,
            "invalid_reason": "no_collision_terminal",
        },
    }
    record = {"completed_episodes": 0, "voided_episodes": 0,
              "metrics": m}
    problems: list = []
    check_outcome_invariants(record, problems)
    assert problems, "success_event_count=1 but sample_count=0 must FAIL"


def test_red22_gate_per_denoise_missing_index_fails():
    """§11.4.1/§11.4.4: 五 denoise index 任一缺失（键集 != range(ft)）
    或值非 finite → ENGINE_HARD_GATE FAIL。"""
    from scripts.g7_gate_report import check_per_denoise

    problems: list = []
    check_per_denoise(
        {"per_denoise_approx_kl": {0: 0.01, 1: 0.02, 2: 0.01, 3: 0.02},
         "per_denoise_clipfrac": {0: 0.9, 1: 0.8, 2: 0.7, 3: 0.6}},
        ft_steps=5, problems=problems,
    )
    assert problems, "missing denoise index 4 must FAIL the engine gate"
    problems2: list = []
    check_per_denoise(
        {"per_denoise_approx_kl": {0: float("nan")},
         "per_denoise_clipfrac": {0: 0.9}},
        ft_steps=5, problems=problems2,
    )
    assert problems2, "non-finite per-denoise value must FAIL"


def test_red23_gate_legacy_record_marked(tmp_path):
    """§11.4.3/§11.4.5: 旧 G7 metrics（无 schema_version）可由历史
    reader 读取但被标记 legacy，不得作为 current gate 输入。"""
    from scripts.g7_gate_report import _load_metrics

    m = tmp_path / "metrics.jsonl"
    m.write_text(
        json.dumps({"metrics": {"approx_kl": 0.1}}) + "\n", encoding="utf-8",
    )
    records, problems = _load_metrics(m)
    assert len(records) == 1
    assert any("legacy" in p.lower() for p in problems), (
        "legacy (no schema_version) record must be marked, not gated"
    )


def test_red24_gpu_compare_metrics_two_state_contract():
    """GPU RED 2026-08-18 回归保护：g6 gate-9 resume 对比循环在
    OptionalStatistic 两态下收到 None（count==0 → value=None）时曾
    float(None) TypeError 崩溃（连续世界有样本、resume 世界无样本）。
    修复后两个 GPU 文件共享同一语义：双侧 None == 相等；单侧 None
    == 分歧（进 mismatch 列表，不崩溃）；有限值 isclose 对照不变。
    注意 r3 helper 对 parent 做 _jsonable 规范化（JSON 边界），child
    需 str-keyed；g6 helper 双侧同构（无 JSON 边界）。"""
    from test_g6_real_one_update_gpu import _compare_metrics as _g6
    from test_r3_rework3_gpu_gate import _compare_metrics as _r3

    # 双侧 None（同一 null 态）→ 相等
    g_parent = {"per_denoise_approx_kl": {0: None}, "counts": {"a": 0}}
    g_child = {"per_denoise_approx_kl": {0: None}, "counts": {"a": 0}}
    r_parent = {"per_denoise_approx_kl": {0: None}, "counts": {"a": 0}}
    r_child = {"per_denoise_approx_kl": {"0": None}, "counts": {"a": 0}}
    assert _g6(g_parent, g_child, "m") == []
    assert _r3(r_parent, r_child, "m") == []

    # RED 场景：parent 有限、child None（resume 世界 count==0）
    assert len(_g6({"per_denoise_approx_kl": {0: 0.5}},
                   {"per_denoise_approx_kl": {0: None}}, "m")) == 1
    mm = _r3({"per_denoise_approx_kl": {0: 0.5}},
             {"per_denoise_approx_kl": {"0": None}}, "m")
    assert len(mm) == 1 and "vs None" in mm[0]

    # 反向：parent None、child 有限
    assert len(_g6({"per_denoise_approx_kl": {0: None}},
                   {"per_denoise_approx_kl": {0: 0.5}}, "m")) == 1
    mm = _r3({"per_denoise_approx_kl": {0: None}},
             {"per_denoise_approx_kl": {"0": 0.5}}, "m")
    assert len(mm) == 1 and "None vs" in mm[0]

    # 顶层单侧 None：任一方向 = 分歧 → mismatch。递归比较器对 None 做
    # 存在性比较，消除了旧 isinstance 链"parent 顶层 None 意外跳过"的
    # 盲区；真实指标顶层无单侧 None（R7-4 后 next_value_batch_size 是
    # 双侧同值的 resolved chunk 数字，OptionalStatistic 均嵌套）
    assert len(_g6({"b": 2.0}, {"b": None}, "m")) == 1
    assert len(_r3({"b": 2.0}, {"b": None}, "m")) == 1
    assert len(_g6({"b": None}, {"b": 2.0}, "m")) == 1
    assert len(_r3({"b": None}, {"b": 2.0}, "m")) == 1

    # 有限值 isclose pass / fail（rtol=1e-4, atol=1e-6）；r3 child 经
    # JSON 边界后数值保持原生 float（JSON 数字类型），数字字符串是类型
    # 分歧而非可强转值（旧 float() 强转设计已废止）
    assert _g6({"a": {"x": 1.0}, "b": 2.0},
               {"a": {"x": 1.0000001}, "b": 2.0000001}, "m") == []
    assert _r3({"a": {"x": 1.0}, "b": 2.0},
               {"a": {"x": 1.0000001}, "b": 2.0000001}, "m") == []
    assert len(_g6({"a": {"x": 1.0}}, {"a": {"x": 2.0}}, "m")) == 1
    assert len(_r3({"a": {"x": 1.0}}, {"a": {"x": "2.0"}}, "m")) == 1


def test_red25_gpu_compare_metrics_structural_shapes():
    """GPU RED 2026-08-18 round 3 回归保护：g6 gate-9 对比循环在
    OptionalStatistic 两态之外还承载真实指标结构——dict 叶
    （executed_vs_unexecuted_grad_norm.ratio、outcome_counts 的
    per_role 映射）、str 叶（invalid_reason、hashes）、list 叶
    （action_variance_by_dim）。旧扁平两层循环对这些形状逐一
    float() TypeError（round 2 修掉 float(None) 后 round 3 暴露
    float(dict)；str/list 同类潜伏）。递归比较器把形状分歧与
    数值分歧统一为 mismatch，绝不崩溃。"""
    from test_g6_real_one_update_gpu import _compare_metrics as _g6
    from test_r3_rework3_gpu_gate import _compare_metrics as _r3

    # dict 叶（OptionalStatistic 记录）：双侧同构 → 递归比较内部字段
    g_parent = {"executed_vs_unexecuted_grad_norm": {
        "executed": 1.0, "unexecuted": 0.0,
        "ratio": {"sample_count": 0, "value": None, "valid": False,
                  "invalid_reason": "zero_executed_gradient_denominator"}}}
    g_child = {"executed_vs_unexecuted_grad_norm": {
        "executed": 1.0, "unexecuted": 0.0,
        "ratio": {"sample_count": 0, "value": None, "valid": False,
                  "invalid_reason": "zero_executed_gradient_denominator"}}}
    assert _g6(g_parent, g_child, "m") == []
    # 递归进 dict 叶：value 单侧 None → mismatch（两态契约在任意深度生效）
    g_child2 = {"executed_vs_unexecuted_grad_norm": {
        "executed": 1.0, "unexecuted": 0.0,
        "ratio": {"sample_count": 1, "value": 0.5, "valid": True,
                  "invalid_reason": None}}}
    mm = _g6(g_parent, g_child2, "m")
    # sample_count 0 vs 1 + value None vs 0.5 + valid False vs True
    # + invalid_reason str vs None
    assert len(mm) == 4, mm

    # 嵌套 dict 叶（outcome_counts per_role 映射）：等值 pass、差值 fail
    p3 = {"outcome_counts": {
        "transition_count_total": 100,
        "per_role_transition_count": {"hover": 40, "track": 60}}}
    c3 = {"outcome_counts": {
        "transition_count_total": 100,
        "per_role_transition_count": {"hover": 40, "track": 60}}}
    assert _g6(p3, c3, "m") == []
    c3b = {"outcome_counts": {
        "transition_count_total": 100,
        "per_role_transition_count": {"hover": 41, "track": 60}}}
    assert len(_g6(p3, c3b, "m")) == 1
    # 缺键（child 少一个 role）→ 结构分歧 mismatch，不 KeyError
    c3c = {"outcome_counts": {
        "transition_count_total": 100,
        "per_role_transition_count": {"hover": 40}}}
    assert len(_g6(p3, c3c, "m")) == 1

    # str 叶（invalid_reason 双侧同串）→ 相等；不同串 → mismatch
    assert _g6({"s": {"r": {"invalid_reason": "no_success_terminal"}}},
               {"s": {"r": {"invalid_reason": "no_success_terminal"}}},
               "m") == []
    assert len(_g6({"s": {"r": {"invalid_reason": "no_success_terminal"}}},
                   {"s": {"r": {"invalid_reason": "no_collision_terminal"}}},
                   "m")) == 1

    # list 叶（action_variance_by_dim）：逐元素 isclose、长度分歧
    assert _g6({"v": [0.1, 0.2, 0.3]}, {"v": [0.1, 0.20000001, 0.3]},
               "m") == []
    assert len(_g6({"v": [0.1, 0.2, 0.3]}, {"v": [0.1, 0.3, 0.3]},
                   "m")) == 1
    assert len(_g6({"v": [0.1, 0.2]}, {"v": [0.1, 0.2, 0.3]}, "m")) == 1

    # 类型分歧：scalar vs dict / dict vs list → mismatch 不崩溃
    assert len(_g6({"a": 1.0}, {"a": {"x": 1.0}}, "m")) == 1
    assert len(_g6({"a": {"x": 1.0}}, {"a": [1.0]}, "m")) == 1

    # 顶层缺键双向报告
    assert len(_g6({"a": 1.0}, {}, "m")) == 1
    assert len(_g6({}, {"a": 1.0}, "m")) == 1

    # r3 侧同构（child str-keyed）；REWORK 2026-08-19 (P1-4): 任一侧
    # 非有限（含双侧同为 NaN）→ mismatch hard-fail。R6-C 两态契约下
    # NaN 不是合法"无数据"表示（无数据 = OptionalStatistic count==0），
    # 旧"parent NaN 哨兵跳过"让 engine violation 被当作相等。
    r_p = {"per_denoise_approx_kl": {0: 1.0},
           "policy_loss_mean": float("nan")}
    r_c = {"per_denoise_approx_kl": {"0": 1.0},
           "policy_loss_mean": float("nan")}
    assert len(_r3(r_p, r_c, "m")) == 1
    r_p3 = {"per_denoise_approx_kl": {0: 1.0},
            "policy_loss_mean": float("nan")}
    r_c3 = {"per_denoise_approx_kl": {"0": 1.0},
            "policy_loss_mean": 1.0}
    assert len(_r3(r_p3, r_c3, "m")) == 1
    # dict 叶递归在 r3 侧同样生效（outcome_counts per_role）
    r_p2 = {"outcome_counts": {
        "per_role_completed_episode_count": {"hover": 3}}}
    r_c2 = {"outcome_counts": {
        "per_role_completed_episode_count": {"hover": 4}}}
    assert len(_r3(r_p2, r_c2, "m")) == 1


def test_red26_gate_conditioned_count_bidirectional_unconditional():
    """REWORK 2026-08-19 (P1-1): conditioned statistic 与 outcome 事件
    计数必须无条件双向相等。旧实现只在 events>0 时检查 → phantom
    （events=0 但 sample_count=1）会被接受；负数/bool/缺失字段同样
    拒收。"""
    from scripts.g7_gate_report import check_outcome_invariants

    def _run(record: dict) -> list:
        problems: list = []
        check_outcome_invariants(record, problems)
        return problems

    base_counts = {
        "transition_count_total": 10,
        "terminal_transition_count": 1,
        "success_event_count": 0, "collision_event_count": 1,
        "rollout_cut_count": 0,
        "per_role_transition_count": {"reach": 10},
        "per_role_completed_episode_count": {"reach": 1},
    }
    ok_stats = {
        "success_conditioned_advantage": {
            "sample_count": 0, "value": None, "valid": False,
            "invalid_reason": "no_success_terminal"},
        "collision_conditioned_advantage": {
            "sample_count": 1, "value": 0.3, "valid": True,
            "invalid_reason": None},
    }
    good = {"completed_episodes": 1, "voided_episodes": 0,
            "metrics": {"outcome_counts": base_counts, **ok_stats}}

    # 合法 null 态（events=0 ∧ sample_count=0）→ 干净
    assert _run(good) == []

    # PHANTOM：events=0 但 sample_count=1 → 必须 FAIL（旧实现接受）
    m = {"outcome_counts": {**base_counts,
                            "success_event_count": 0,
                            "collision_event_count": 0},
         "success_conditioned_advantage": {
             "sample_count": 1, "value": 0.5, "valid": True,
             "invalid_reason": None},
         "collision_conditioned_advantage": {
             "sample_count": 0, "value": None, "valid": False,
             "invalid_reason": "no_collision_terminal"}}
    assert _run({"completed_episodes": 0, "voided_episodes": 0,
                 "metrics": m}), "phantom sample_count=1/events=0 must FAIL"

    # 反向：events=1 但 sample_count=0 → FAIL（旧实现已覆盖，保留）
    m = {"outcome_counts": {**base_counts, "collision_event_count": 0},
         "success_conditioned_advantage": {
             "sample_count": 0, "value": None, "valid": False,
             "invalid_reason": "no_success_terminal"},
         "collision_conditioned_advantage": {
             "sample_count": 0, "value": None, "valid": False,
             "invalid_reason": "no_collision_terminal"}}
    assert _run({"completed_episodes": 0, "voided_episodes": 0,
                 "metrics": m}), "events=1/sample_count=0 must FAIL"

    # bool 事件计数（int 子类）→ 拒收
    m = {"outcome_counts": {**base_counts,
                            "success_event_count": True,
                            "collision_event_count": 0},
         **ok_stats}
    assert _run({"completed_episodes": 1, "voided_episodes": 0,
                 "metrics": m}), "bool event count must FAIL"

    # 负数事件计数 → 拒收
    m = {"outcome_counts": {**base_counts,
                            "success_event_count": -1,
                            "collision_event_count": 0},
         **ok_stats}
    assert _run({"completed_episodes": 1, "voided_episodes": 0,
                 "metrics": m}), "negative event count must FAIL"

    # 缺失事件字段（events=None）→ 拒收（旧实现 isinstance 检查静默跳过）
    m = {"outcome_counts": {"transition_count_total": 10,
                            "terminal_transition_count": 1,
                            "rollout_cut_count": 0,
                            "per_role_transition_count": {"reach": 10},
                            "per_role_completed_episode_count": {"reach": 1}},
         **ok_stats}
    assert _run({"completed_episodes": 1, "voided_episodes": 0,
                 "metrics": m}), "missing event field must FAIL"


def test_red27_gate_completed_episode_cross_layer_closure():
    """REWORK 2026-08-19 (P1-2): completed episode 跨层不变量闭合——
    outcome_counts.terminal_transition_count == record.completed_episodes
    ；per_role_completed_episode_count 汇总 == terminal；void/fault/
    rollout_cut 单列、绝不进入 completed 映射；字段类型/负数拒收。"""
    from scripts.g7_gate_report import check_outcome_invariants

    def _run(record: dict) -> list:
        problems: list = []
        check_outcome_invariants(record, problems)
        return problems

    counts = {
        "transition_count_total": 100,
        "terminal_transition_count": 3,
        "success_event_count": 2, "collision_event_count": 1,
        "rollout_cut_count": 1,
        "per_role_transition_count": {"reach": 50, "track": 50},
        "per_role_completed_episode_count": {"reach": 2, "track": 1},
    }
    stats = {
        "success_conditioned_advantage": {
            "sample_count": 2, "value": 0.5, "valid": True,
            "invalid_reason": None},
        "collision_conditioned_advantage": {
            "sample_count": 1, "value": 0.3, "valid": True,
            "invalid_reason": None},
    }
    good = {"completed_episodes": 3, "voided_episodes": 1,
            "metrics": {"outcome_counts": counts, **stats}}
    assert _run(good) == []

    # 篡改顶层 completed_episodes → FAIL（buffer/trainer 终局统计与
    # train-loop 完成计数偏离时不得通过）
    bad = {"completed_episodes": 4, "voided_episodes": 1,
           "metrics": {"outcome_counts": counts, **stats}}
    assert _run(bad), "completed_episodes != terminal_transition_count must FAIL"

    # per_role 汇总 != terminal → FAIL
    c2 = {**counts, "per_role_completed_episode_count": {"reach": 2}}
    assert _run({"completed_episodes": 3, "voided_episodes": 1,
                 "metrics": {"outcome_counts": c2, **stats}}), (
        "per-role completed sum != terminal_transition_count must FAIL")

    # rollout_cut/void/fault 泄露进 completed 映射 → FAIL
    c3 = {**counts, "per_role_completed_episode_count": {
        "reach": 2, "track": 1, "rollout_cut": 0}}
    assert _run({"completed_episodes": 3, "voided_episodes": 1,
                 "metrics": {"outcome_counts": c3, **stats}}), (
        "rollout_cut must never appear in completed map")

    # completed_episodes 为 bool / 缺字段 → FAIL
    assert _run({"completed_episodes": True, "voided_episodes": 1,
                 "metrics": {"outcome_counts": counts, **stats}})
    assert _run({"voided_episodes": 1,
                 "metrics": {"outcome_counts": counts, **stats}}), (
        "missing completed_episodes must FAIL")

    # per_role_completed_episode_count 缺失 / 负数 rollout_cut / bool
    # voided → FAIL
    c4 = {k: v for k, v in counts.items()
          if k != "per_role_completed_episode_count"}
    assert _run({"completed_episodes": 3, "voided_episodes": 1,
                 "metrics": {"outcome_counts": c4, **stats}})
    c5 = {**counts, "rollout_cut_count": -1}
    assert _run({"completed_episodes": 3, "voided_episodes": 1,
                 "metrics": {"outcome_counts": c5, **stats}})
    assert _run({"completed_episodes": 3, "voided_episodes": True,
                 "metrics": {"outcome_counts": counts, **stats}})


def test_red28_gate_schema_strict_allowlist(tmp_path):
    """REWORK 2026-08-19 (P1-3): reader 严格 schema allowlist——只接受
    routeA_rl_env_v1_r6c 为 current-gate 输入；其他非空版本（未知/
    伪造/旧版）hard-fail，绝不静默按当前规则读取；缺失/空串仍是
    legacy 标记（历史证据，RED-23 语义不变）。"""
    from scripts.g7_gate_report import _load_metrics

    def _load(schema) -> tuple:
        m = tmp_path / "m.jsonl"
        rec = {"metrics": {"approx_kl": 0.1}}
        if schema is not None:
            rec["schema_version"] = schema
        m.write_text(json.dumps(rec) + "\n", encoding="utf-8")
        return _load_metrics(m)

    # 精确当前 schema → 干净
    records, problems = _load("routeA_rl_env_v1_r6c")
    assert len(records) == 1 and problems == [], problems

    # 未知/旧版/伪造非空 schema → hard-fail（含 r6b 这类"看起来像版本"的）
    for bogus in ("routeA_rl_env_v1_r6b", "garbage", "v2", "0"):
        records, problems = _load(bogus)
        assert len(records) == 1, records
        assert any("schema" in p.lower() for p in problems), (
            f"unknown schema {bogus!r} must hard-fail: {problems}")

    # 缺失 / 空串 → legacy 标记（历史证据，非 current gate 输入）
    for missing in (None, ""):
        records, problems = _load(missing)
        assert len(records) == 1
        assert any("legacy" in p.lower() for p in problems), (
            f"missing/empty schema must be legacy-marked: {problems}")


def test_red29_gpu_compare_metrics_nonfinite_hard_fails():
    """REWORK 2026-08-19 (P1-4): GPU resume 比较器对 NaN/Inf 必须
    hard-fail——任一侧非有限（含双侧同为非有限）= mismatch，任意
    dict/list 深度递归生效。R6-C 两态契约下 NaN 不是合法"无数据"
    表示（无数据 = OptionalStatistic count==0/value=None）；旧
    "parent 非有限 → 跳过"把 engine violation 当相等。"""
    from test_g6_real_one_update_gpu import _compare_metrics as _g6
    from test_r3_rework3_gpu_gate import _compare_metrics as _r3

    for cmp_ in (_g6, _r3):
        # parent 非有限 vs child 有限（旧哨兵跳过会接受 → 必须 mismatch）
        assert len(cmp_({"x": float("nan")}, {"x": 1.0}, "m")) == 1
        # child 非有限 vs parent 有限
        assert len(cmp_({"x": 1.0}, {"x": float("nan")}, "m")) == 1
        # 双侧同为 NaN / 双侧同为 Inf — 不再视为"相等哨兵"
        assert len(cmp_({"x": float("nan")}, {"x": float("nan")}, "m")) == 1
        assert len(cmp_({"x": float("inf")}, {"x": float("inf")}, "m")) == 1
        # 任意深度递归：dict 叶内 / list 叶内
        assert len(cmp_({"a": {"b": float("nan")}},
                        {"a": {"b": 1.0}}, "m")) == 1
        assert len(cmp_({"v": [float("nan")]}, {"v": [1.0]}, "m")) == 1
        assert len(cmp_({"v": [float("nan")]},
                        {"v": [float("nan")]}, "m")) == 1


# ═══════════════════════════════════════════════════════════════════
# Gate B R6-METRIC-1 REWORK（2026-08-19）: F-03 反向交叉检查
# 原 cross_check_terminal_rewards 只查 outcome→reward 方向（success
# 必须带 bonus / collision 必须带 penalty）；reward→outcome 方向缺失：
# terminal 带 +100 reach bonus（或碰撞惩罚）却无对应 outcome flag 通过
# 全部四层 —— R5.9-C1 info-forwarding 断链签名（F-03 fix requirement 3
# + §11.4.2 invariant 4：reward 显示 bonus/penalty 与 outcome count 矛盾
# 必须 hard-fail 并报告 episode ids）。
# ═══════════════════════════════════════════════════════════════════

def test_red16_reverse_reach_bonus_without_outcome():
    """反向：terminal 带 reach bonus（>= reach_reward/2）但 success 与
    collision 都无记录 → hard-fail 并报告 episode id。"""
    from diffusion_policy_3d.rl.dppo_uav.metrics import (
        cross_check_terminal_rewards,
    )

    rewards = np.array([0.1, 95.0, -18.0], dtype=np.float64)
    terminated = np.array([False, True, True], dtype=bool)
    successes = np.zeros(3, dtype=bool)
    collisions = np.array([False, False, True], dtype=bool)
    with pytest.raises(RuntimeError) as e:
        cross_check_terminal_rewards(
            rewards=rewards, terminated=terminated, successes=successes,
            collisions=collisions, episode_ids=["e1", "e2", "e3"],
            reach_reward=100.0,
        )
    assert "e2" in str(e.value), e.value


def test_red17_reverse_collision_penalty_without_outcome():
    """反向：terminal 带碰撞惩罚（<= collision_penalty/2）但 collision
    无记录 → hard-fail。timeout-like 微负 reward（-0.3，r_smooth
    下界 ~-0.2）不触发 —— 阈值绑定 penalty 幅度，不误报 timeout。"""
    from diffusion_policy_3d.rl.dppo_uav.metrics import (
        cross_check_terminal_rewards,
    )

    rewards = np.array([-0.3, -12.0], dtype=np.float64)
    terminated = np.array([True, True], dtype=bool)
    successes = np.zeros(2, dtype=bool)
    collisions = np.zeros(2, dtype=bool)
    with pytest.raises(RuntimeError) as e:
        cross_check_terminal_rewards(
            rewards=rewards, terminated=terminated, successes=successes,
            collisions=collisions, episode_ids=["t1", "e2"],
            reach_reward=100.0,
        )
    assert "e2" in str(e.value), e.value


def test_red18_reverse_direction_clean_pass():
    """反向合法路径：timeout 微负 reward 不误报；success 带 bonus /
    collision 带 penalty 不重复报。"""
    from diffusion_policy_3d.rl.dppo_uav.metrics import (
        cross_check_terminal_rewards,
    )

    rewards = np.array([-0.3, 95.0, -12.0], dtype=np.float64)
    terminated = np.array([True, True, True], dtype=bool)
    successes = np.array([False, True, False], dtype=bool)
    collisions = np.array([False, False, True], dtype=bool)
    # 无异常即 PASS
    cross_check_terminal_rewards(
        rewards=rewards, terminated=terminated, successes=successes,
        collisions=collisions, episode_ids=["t1", "e2", "e3"],
        reach_reward=100.0,
    )


# ═══════════════════════════════════════════════════════════════════
# Gate B R6-EVAL-1: g7_gate_report 统计层接线 (§11.5.1)
# ═══════════════════════════════════════════════════════════════════

_R6E_BANK = "bank-r6eval1"
_R6E_BC_CKPT = "ckpt_bc" + "0" * 20
_R6E_RL_CKPT = "ckpt_rl" + "0" * 20


def _eval_rec(ep_hash, replica, *, bc_ok, rl_ok, ckpt=_R6E_RL_CKPT):
    """生产形状的 paired 记录：两侧 outcome 齐全（§11.6.4 keys）、
    真实推导的 policy_noise_seed、记录级 checkpoint hash = RL 的
    （生产 run_paired_eval 传被评估策略的 hash）。"""
    from diffusion_policy_3d.rl.dppo_uav.evaluator import (
        EpisodePairedRecord,
    )
    from diffusion_policy_3d.rl.dppo_uav.paired_rng import (
        policy_noise_seed,
    )

    def outcome(ok):
        return {
            "success": bool(ok), "collision": not bool(ok),
            "timeout": False, "done_reason": "reached" if ok else "collision",
            "out_of_bounds": False, "altitude_exceed": False,
            "terminated": True, "truncated": False,
            "terminal_category": "reached" if ok else "collision",
            "return": 10.0, "steps": 40,
        }

    return EpisodePairedRecord(
        episode_config={"episode_config_hash": ep_hash},
        seed=42, world="w1", role="anticipatory",
        start=[0.0, 0.0, 0.0], goal=[1.0, 2.0, 3.0],
        disturbance={}, bc=outcome(bc_ok), rl=outcome(rl_ok),
        policy_noise_seed=policy_noise_seed(_R6E_BANK, ep_hash, replica),
        replica_id=replica, rng_contract_version="r5.4",
        policy_checkpoint_hash=ckpt,
    )


def _artifact(records, *, bank_id=_R6E_BANK, ckpt=_R6E_RL_CKPT,
              bc_ckpt=_R6E_BC_CKPT):
    import dataclasses

    return {
        "episodes": [dataclasses.asdict(r) for r in records],
        "bank_id": bank_id,
        "checkpoint_hash": ckpt,
        "load_contract": {"bc_checkpoint_hash": bc_ckpt},
    }


def test_eval1_g7_side_summary_bc_identity_from_load_contract():
    """Gate B R6-EVAL-1 (§11.5.1): 生产配对记录带 RL checkpoint hash，
    但 BC 侧的统计身份必须来自 artifact 的 load_contract
    bc_checkpoint_hash —— 记录级 RL hash 不得混入 BC 侧 cluster
    （否则跨 artifact 的 BC 段永不折叠，18/24 伪重复在生产路径复活）。
    Gate B 审查发现的 raw len(eps) pooling 已被统计层替换：
    n_episodes = 去重后的 effective n。"""
    from scripts.g7_gate_report import _eval_side_summary

    records = [
        _eval_rec("ep%03d" % i, 0, bc_ok=(i < 9), rl_ok=(i < 9))
        for i in range(12)
    ]
    summary = _eval_side_summary(_artifact(records), "bc")
    assert summary["n_episodes"] == 12
    assert summary["raw_n"] == 12
    assert summary["sr"] == 9 / 12
    assert summary["policy_clusters"] == 1, (
        "BC side must cluster by the load_contract bc_checkpoint_hash, "
        "never by the RL hash mirrored onto the record"
    )


def test_eval1_g7_side_summary_rl_clusters_by_record_hash():
    """RL 侧按记录级 checkpoint hash 成 cluster：12 entries × 5
    replicas = 60 unique，policy_clusters=1（同一策略）。"""
    from scripts.g7_gate_report import _eval_side_summary

    records = [
        _eval_rec("ep%03d" % i, r, bc_ok=(i < 9), rl_ok=(i < 9))
        for i in range(12) for r in range(5)
    ]
    summary = _eval_side_summary(_artifact(records), "rl")
    assert summary["n_episodes"] == 60
    assert summary["unique_n"] == 60
    assert summary["cluster_n"] == 12
    assert summary["policy_clusters"] == 1
    assert summary["cr"] == 15 / 60  # 3 collision entries × 5 replicas


def test_eval1_g7_side_summary_tampered_rng_hard_fails():
    """R5.4 RNG-provenance 校验在 gate 路径上生效：tampered
    policy_noise_seed 不得通过统计层。"""
    import dataclasses

    from scripts.g7_gate_report import _eval_side_summary

    records = [_eval_rec("ep000", 0, bc_ok=True, rl_ok=True)]
    tampered = dataclasses.replace(
        records[0], policy_noise_seed=records[0].policy_noise_seed + 1)
    with pytest.raises(ValueError, match="RNG provenance"):
        _eval_side_summary(_artifact([tampered]), "rl")


def test_eval1_g7_side_summary_missing_bank_id_hard_fails():
    """无 bank_id 的 artifact 进不了统计层 —— 不是静默 NaN。"""
    from scripts.g7_gate_report import _eval_side_summary

    records = [_eval_rec("ep000", 0, bc_ok=True, rl_ok=True)]
    with pytest.raises(ValueError, match="bank_id"):
        _eval_side_summary(_artifact(records, bank_id=""), "bc")


def test_eval1_g7_side_summary_reports_three_n():
    """§11.5.5: raw/unique/cluster 三 n 全量报告 —— 同单元完全一致的
    重复记录折叠（raw 2 → unique 1），绝不只给一个 n。"""
    from scripts.g7_gate_report import _eval_side_summary

    rec = _eval_rec("ep000", 0, bc_ok=True, rl_ok=True)
    summary = _eval_side_summary(_artifact([rec, rec]), "rl")
    assert summary["raw_n"] == 2
    assert summary["unique_n"] == 1
    assert summary["cluster_n"] == 1


# ═══════════════════════════════════════════════════════════════════
# Gate B R6-EVAL-2: g7 gate 由 capability_informativeness 驱动
# （§11.5.3 — presence 检查是 Gate B 发现的死闸；0-opportunity 的
# 真 frozen bank 必须 NOT_INFORMATIVE/BLOCKED，不得 PASS）
# ═══════════════════════════════════════════════════════════════════

_NOT_INFO = {
    "opportunity_count": 0, "exposure_count": 0, "event_count": 0,
    "value": None,
    "validity_reason": "no opportunity — absent or zero (opportunity_count=0)",
}
_INFO = {
    "opportunity_count": 1, "exposure_count": 1, "event_count": 1,
    "value": 0.5, "validity_reason": None,
}


def test_eval2_gate_zero_opportunity_bank_blocked_not_pass():
    """Gate B R6-EVAL-2: 真 frozen bank（0 opportunity）不得 PASS ——
    NOT_INFORMATIVE 组显式进入 capability_blocked（BLOCKED 通道，
    §11.4.4），与 engine verdict 分离。"""
    from scripts.g7_gate_report import _capability_outcome_verdict

    outcomes = {
        "anticipatory_avoidance": dict(_NOT_INFO),
        "fullspace_support_expansion": dict(_NOT_INFO),
        "dense_obstacle_ceiling": dict(_NOT_INFO),
    }
    verdicts, blocked = _capability_outcome_verdict(outcomes)
    assert set(verdicts) == set(outcomes)
    assert all(v["status"] == "NOT_INFORMATIVE" for v in verdicts.values())
    assert len(blocked) == 3
    assert all("NOT_INFORMATIVE" in b for b in blocked)


def test_eval2_gate_informative_outcomes_pass():
    """三组都有 opportunity 证据 → INFORMATIVE，blocked 为空 ——
    gate 的 PASS 必须以机会存在为前提，而非 artifact 字段存在。"""
    from scripts.g7_gate_report import _capability_outcome_verdict

    outcomes = {
        "anticipatory_avoidance": dict(_INFO),
        "fullspace_support_expansion": dict(_INFO),
        "dense_obstacle_ceiling": dict(_INFO),
    }
    verdicts, blocked = _capability_outcome_verdict(outcomes)
    assert all(v["status"] == "INFORMATIVE" for v in verdicts.values())
    assert blocked == []


def test_eval2_gate_partial_opportunity_marks_absent_groups():
    """部分组有机会、部分没有 → 缺机会的组显式 NOT_INFORMATIVE 并
    blocked（Guide: 不具 opportunity 的组明确 BLOCKED/NOT_INFORMATIVE）。"""
    from scripts.g7_gate_report import _capability_outcome_verdict

    outcomes = {
        "anticipatory_avoidance": dict(_INFO),
        "fullspace_support_expansion": dict(_NOT_INFO),
        "dense_obstacle_ceiling": dict(_NOT_INFO),
    }
    verdicts, blocked = _capability_outcome_verdict(outcomes)
    assert verdicts["anticipatory_avoidance"]["status"] == "INFORMATIVE"
    assert len(blocked) == 2
    assert any("fullspace_support_expansion" in b for b in blocked)
    assert any("dense_obstacle_ceiling" in b for b in blocked)


def test_eval2_gate_missing_outcomes_block_hard_fails():
    """capability_outcomes 缺失/缺组是 schema 违反 —— 结构层 ValueError
    （main 收进 problems），不得被当作 NOT_INFORMATIVE 静默吞掉
    （§11.5.3: 结构审计与能力审计不得共用一个 PASS）。"""
    from scripts.g7_gate_report import _capability_outcome_verdict

    with pytest.raises(ValueError, match="capability_outcomes"):
        _capability_outcome_verdict(None)
    with pytest.raises(ValueError, match="capability_outcomes"):
        _capability_outcome_verdict({})
    with pytest.raises(ValueError, match="capability_outcomes"):
        _capability_outcome_verdict({"anticipatory_avoidance": dict(_INFO)})
