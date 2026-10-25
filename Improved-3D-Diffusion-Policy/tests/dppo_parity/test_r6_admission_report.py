"""R6-G (F-05 / R6-EVID-1): final-tree admission evidence orchestration.

Guide §11.8.1–11.8.4 + §11.9 G 包负向测试（commit drift、interrupted
canary、incomplete matrix）+ §11.8.4 机械核对（resolved run matrix
程序验证，禁止人工目测）+ §12.4 Gate B 审查前置。

被测模块：
- ``scripts.r6_admission_report`` — verify_run_matrix / verify_canary_
  evidence / freeze_g8_prereg（canonical 验证与报告，evidence glue，
  不改变任何训练/评估语义）；
- ``scripts.eval_dppo_uav_official`` — ``--replicas N`` canonical 60/side
  生产者（R6-D 只建了 12×5 校验器 validate_paired_matrix，生产者
  缺失：run_paired_eval 硬编码 replica_id=0，无法产出 R6.4/G8 要求
  的 12 entries × 5 policy-noise replicas）。

RED 语义（§11.9 G 行）：三类反例 ——
1. commit drift：canary 证据声称的 HEAD/tree 与实际不一致 → 拒绝；
2. interrupted canary：resume>2、缺 interrupted 标记、continuation 与
   control 结构不完整（索引集不等/双方截断/resume_points 伪 int）→
   拒绝；逐项容差超限 → REPORT 附件不 gate（2026-08-21 裁决 A，
   Guide §11.8.2 "只判 engine composition"）；
3. incomplete matrix：3×160k / 100 updates / 1 release / ≤2 resume /
   60 paired/side / strict hashes 任一不满足 → 机械验证拒绝。
"""

import json

import pytest

from scripts.r6_admission_report import (
    VERIFY_MATRIX_BUDGETS,
    freeze_g8_prereg,
    rebuild_canary_series,
    tolerance_diff_report,
    verify_canary_artifacts,
    verify_canary_evidence,
    verify_matrix_artifacts,
    verify_prereg_runtime_binding,
    verify_run_matrix,
)


# ---------------------------------------------------------------- matrix

def _valid_matrix():
    """§11.8.4 的完整 resolved run matrix（默认合法样本）。"""
    return {
        "g8": {
            "seeds": [101, 202, 303],
            "transitions_per_seed": 160000,
            "transitions_per_update": 1600,
            "updates_per_seed": 100,
            "n_envs": 1,
            "releases_per_seed": 1,
            "max_resumes_per_seed": 2,
            "paired_bc_records_per_side": 60,
            "paired_rl_records_per_seed": 60,
            "common_bc_checkpoint_once": True,
            "hashes": {
                "final_commit": "a" * 40,
                "config_hash": "b" * 64,
                "checkpoint_hash": "c" * 64,
                "bank_hash": "d" * 64,
            },
        },
        "config": {"actor_lr": 1e-5, "critic_lr": 1e-3},
    }


def test_matrix_valid_passes():
    problems = verify_run_matrix(_valid_matrix())
    assert problems == []


@pytest.mark.parametrize("seeds", [[101, 202], [101], []])
def test_matrix_missing_seed_rejected(seeds):
    m = _valid_matrix()
    m["g8"]["seeds"] = seeds
    assert verify_run_matrix(m) != []


def test_matrix_wrong_update_count_rejected():
    m = _valid_matrix()
    m["g8"]["updates_per_seed"] = 99
    assert verify_run_matrix(m) != []


def test_matrix_transition_total_mismatch_rejected():
    m = _valid_matrix()
    m["g8"]["transitions_per_seed"] = 159999
    assert verify_run_matrix(m) != []


def test_matrix_multi_release_rejected():
    m = _valid_matrix()
    m["g8"]["releases_per_seed"] = 2
    assert verify_run_matrix(m) != []


def test_matrix_resume_overflow_rejected():
    m = _valid_matrix()
    m["g8"]["max_resumes_per_seed"] = 3
    assert verify_run_matrix(m) != []


def test_matrix_paired_count_shortfall_rejected():
    m = _valid_matrix()
    m["g8"]["paired_bc_records_per_side"] = 12
    assert verify_run_matrix(m) != []


def test_matrix_multi_env_rejected():
    m = _valid_matrix()
    m["g8"]["n_envs"] = 4
    assert verify_run_matrix(m) != []


def test_matrix_hash_drift_rejected():
    m = _valid_matrix()
    # 非 hex / 错误长度 → strict-hash 结构漂移拒绝（§11.8.4）
    m["g8"]["hashes"]["config_hash"] = "z" * 64
    assert verify_run_matrix(m) != []


def test_matrix_hash_missing_rejected():
    m = _valid_matrix()
    del m["g8"]["hashes"]["bank_hash"]
    assert verify_run_matrix(m) != []


# ------------------------------------------------------ canary evidence

def _per_update(i: int) -> dict:
    """一次 update 的五项 per-update metric（continuation/control 共用）。"""
    return {"approx_kl": 0.01, "clipfrac": 0.3,
            "actor_grad_norm": 1.0, "critic_grad_norm": 10.0,
            "value_loss": 2.0}


def _valid_evidence():
    """R6.4 canary 证据（默认合法样本：双 seed、seed202 interrupted）。

    P2 glue（2026-08-19 Gate B）：updates_per_seed=5，continuation 与
    control 必须覆盖全部 5 次 update —— 两段各自截断的对比证明不了
    engine parity（§11.8.2），fixture 因此从 2 次扩展到 5 次。
    """
    # 每侧独立的内层 dict：continuation 与 control 不得共享同一对象
    # （mismatch 测试要能只改一侧）
    full = {str(i): dict(_per_update(i)) for i in range(1, 6)}
    cont_side = {k: dict(v) for k, v in full.items()}
    ctrl_side = {k: dict(v) for k, v in full.items()}
    return {
        "claims": {
            "head": "a" * 40,
            "tree": "clean",
            "seeds": {"101": "uninterrupted", "202": "interrupted"},
            "resume_count": {"202": 1},
            "transitions_per_seed": 8000,
            "updates_per_seed": 5,
            "releases_per_seed": 1,
            "paired_records_per_side": 60,
        },
        "tolerances": {
            "per_update": {
                "approx_kl": 0.05,
                "clipfrac": 0.05,
                "actor_grad_norm": 0.5,
                "critic_grad_norm": 0.5,
                "value_loss": 0.5,
            }
        },
        "interrupted": {
            "202": {
                "resume_points": [2],
                "resume_continuation": cont_side,
            }
        },
        "control": {
            "202": ctrl_side,
        },
    }


def test_commit_drift_rejected():
    ev = _valid_evidence()
    # 证据声称的 HEAD 与当前实际 HEAD 不一致 → commit drift 拒绝
    assert verify_canary_evidence(ev, current_head="b" * 40) != []


def test_interrupted_resume_overflow_rejected():
    ev = _valid_evidence()
    ev["claims"]["resume_count"]["202"] = 3
    assert verify_canary_evidence(ev, current_head="a" * 40) != []


def test_missing_interrupt_marker_rejected():
    ev = _valid_evidence()
    # 声称 interrupted 但无 interrupted 段记录 → 证据不完整
    del ev["interrupted"]["202"]
    assert verify_canary_evidence(ev, current_head="a" * 40) != []


def test_continuation_control_mismatch_reported_not_gated():
    ev = _valid_evidence()
    # 2026-08-21 裁决 A：逐项差超容差 → REPORT 附件，不 gate —— Guide
    # §11.8.2 通过条件 = "只判 engine composition" 清单（结构完整性
    # 仍在 verify 中 gate；tolerance 数值不动）
    ev["interrupted"]["202"]["resume_continuation"]["2"]["approx_kl"] = 0.9
    assert verify_canary_evidence(ev, current_head="a" * 40) == []
    report = tolerance_diff_report(ev)
    assert any("update 2 approx_kl" in line and "REPORT" in line
               for line in report)


def test_within_frozen_tolerance_no_report():
    ev = _valid_evidence()
    # 逐项差在冻结容差内 → verify PASS 且无报告行
    assert verify_canary_evidence(ev, current_head="a" * 40) == []
    assert tolerance_diff_report(ev) == []


def test_incomplete_seed_matrix_rejected():
    ev = _valid_evidence()
    # 只覆盖 seed101（无 seed202 interrupted 证据）→ incomplete matrix
    ev["claims"]["seeds"] = {"101": "uninterrupted"}
    assert verify_canary_evidence(ev, current_head="a" * 40) != []


# ─────────────────────────── P2 glue (Gate B 2026-08-19) ─────────────────────
# 覆盖计数 / resume_points 内容 / resume_count 完整性：三段独立断言，
# 每段都是证据不可达即拒绝（负向测试，§11.9 G 包 evidence glue 仅）。

def test_interrupted_seed_missing_resume_count_entry_rejected():
    ev = _valid_evidence()
    # 声称 interrupted 的 seed 在 resume_count 块中无条目 → 完整性违反
    del ev["claims"]["resume_count"]["202"]
    assert verify_canary_evidence(ev, current_head="a" * 40) != []


def test_resume_points_entries_must_be_int():
    ev = _valid_evidence()
    # 浮点 / bool 都是 int 伪装 —— update index 必须是真 int
    ev["interrupted"]["202"]["resume_points"] = [2.0]
    assert verify_canary_evidence(ev, current_head="a" * 40) != []
    ev = _valid_evidence()
    ev["interrupted"]["202"]["resume_points"] = [True]
    assert verify_canary_evidence(ev, current_head="a" * 40) != []


def test_resume_points_count_must_match_resume_count():
    ev = _valid_evidence()
    # resume_count=2 但只记录 1 个 resume point → 自相矛盾
    ev["claims"]["resume_count"]["202"] = 2
    assert verify_canary_evidence(ev, current_head="a" * 40) != []


def test_continuation_control_index_set_mismatch_rejected():
    ev = _valid_evidence()
    # control 缺第 5 次 update：continuation 与 control 必须逐同一批
    # update 对比 —— 缺一方的 index 不能静默跳过
    del ev["control"]["202"]["5"]
    assert verify_canary_evidence(ev, current_head="a" * 40) != []


def test_both_truncated_continuation_control_rejected():
    ev = _valid_evidence()
    # 双方都只覆盖前 2 次 update（updates_per_seed=5）：两段截断的
    # 对比证明不了 engine parity —— 截断尾不得用"逐项相等"糊弄
    short = {str(i): _per_update(i) for i in range(1, 3)}
    ev["interrupted"]["202"]["resume_continuation"] = dict(short)
    ev["control"]["202"] = dict(short)
    assert verify_canary_evidence(ev, current_head="a" * 40) != []


# ---------------------------------------------------------- prereg freeze

def test_prereg_schema_required_fields():
    prereg = {"judgements": {}, "thresholds": {}}
    with pytest.raises(ValueError):
        freeze_g8_prereg(prereg, output_path=None)


def test_prereg_frozen_immutable_hash():
    prereg = {
        "schema_version": "g8-prereg-v1",
        "budget": {"seeds": [101, 202, 303], "transitions_per_seed": 160000},
        "judgements": {"engine_pass": "5 gates", "learning_pass": "KL ok"},
        "thresholds": {"kl_upper": 0.5, "clipfrac_upper": 0.9},
        "stop_rules": ["hard_fault", "budget_exhausted"],
        "resolved_config": {"actor_lr": 1e-5, "critic_lr": 1e-3},
        "tolerances": {"approx_kl": 0.05},
    }
    blob = freeze_g8_prereg(prereg, output_path=None)
    frozen = json.loads(blob)
    assert frozen["content_hash"] == frozen["content_hash"]
    # 同一输入 → 同一冻结 hash（确定性，冻结后可校验篡改）
    blob2 = freeze_g8_prereg(prereg, output_path=None)
    assert json.loads(blob2)["content_hash"] == frozen["content_hash"]


# ──────────────── runtime binding (Gate D reopening 2026-08-22) ────────────────

def _frozen_prereg_v2(tmp_path, **overrides):
    """Write a frozen g8-prereg-v2 file; return its path."""
    prereg = {
        "schema_version": "g8-prereg-v2",
        "budget": {
            "seeds": [101, 202, 303],
            "transitions_per_seed": 160000,
            "transitions_per_update": 1600,
            "updates_per_seed": 100,
            "n_envs": 1,
            "releases_per_seed": 1,
            "max_resumes_per_seed": 2,
            "paired_bc_records_per_side": 60,
            "paired_rl_records_per_seed": 60,
        },
        "judgements": {"engine_pass": "p", "learning_pass": "l"},
        "thresholds": {"kl_upper": 0.0392, "clipfrac_upper": 0.626},
        "stop_rules": ["hard_fault", "budget_exhausted"],
        "resolved_config": {
            "stage": "formal",
            "actor_lr": 1e-5,
            "critic_lr": 1e-3,
            "update_epochs": 1,
            "lr_scheduler": "none",
            "reward_scale": "none",
            "lora_rank": 8,
            "lora_alpha": 8,
            "batch_size": 128,
            "max_ep_len": 100,
            "n_episodes": 32,
            "manifest_path": "/m",
            "bc_checkpoint_path": "/b",
            "worlds_train_path": "/w",
            "paired_bank_path": None,
            "offline_density_path": None,
        },
        "tolerances": {"per_update": {"approx_kl": 0.05}},
    }
    prereg.update(overrides)
    path = tmp_path / "prereg.json"
    freeze_g8_prereg(prereg, output_path=str(path))
    return path


def _valid_cli():
    """CLI fields matching _frozen_prereg_v2 (G8 shape)."""
    return {
        "stage": "formal",
        "base_seed": 101,
        "lr_actor": 1e-5,
        "lr_value": 1e-3,
        "update_epochs": 1,
        "lr_scheduler": "none",
        "reward_scale": "none",
        "lora_rank": 8,
        "lora_alpha": 8,
        "batch_size": 128,
        "max_ep_len": 100,
        "n_episodes": 32,
        "n_envs": 1,
        "target_transitions": 160000,
        "rollout_transitions_per_update": 1600,
        "admission_manifest": "/m",
        "bc_ckpt": "/b",
        "worlds_train": "/w",
        "paired_bank": None,
        "offline_density": None,
    }


def test_prereg_binding_valid_passes(tmp_path):
    assert verify_prereg_runtime_binding(
        str(_frozen_prereg_v2(tmp_path)), _valid_cli()) == []


def test_prereg_binding_missing_file(tmp_path):
    problems = verify_prereg_runtime_binding(
        str(tmp_path / "absent.json"), _valid_cli())
    assert any("cannot read" in p for p in problems)


def test_prereg_binding_tampered_after_freeze(tmp_path):
    path = _frozen_prereg_v2(tmp_path)
    with open(path) as f:
        data = json.load(f)
    data["thresholds"]["kl_upper"] = 0.9  # judgement drift after freeze
    with open(path, "w") as f:
        json.dump(data, f)
    problems = verify_prereg_runtime_binding(str(path), _valid_cli())
    assert any("tampered" in p for p in problems)


def test_prereg_binding_wrong_seed_rejected(tmp_path):
    cli = _valid_cli()
    cli["base_seed"] = 999  # not in frozen seeds
    problems = verify_prereg_runtime_binding(
        str(_frozen_prereg_v2(tmp_path)), cli)
    assert any("base_seed" in p for p in problems)


def test_prereg_binding_wrong_lr_rejected(tmp_path):
    cli = _valid_cli()
    cli["lr_actor"] = 1e-4  # R6.5 ruled out
    problems = verify_prereg_runtime_binding(
        str(_frozen_prereg_v2(tmp_path)), cli)
    assert any("lr_actor" in p for p in problems)


def test_prereg_binding_wrong_bank_rejected(tmp_path):
    cli = _valid_cli()
    cli["paired_bank"] = "/other-bank.json"  # frozen null
    problems = verify_prereg_runtime_binding(
        str(_frozen_prereg_v2(tmp_path)), cli)
    assert any("paired_bank" in p for p in problems)


def test_prereg_binding_wrong_manifest_rejected(tmp_path):
    cli = _valid_cli()
    cli["admission_manifest"] = "/other-manifest.json"
    problems = verify_prereg_runtime_binding(
        str(_frozen_prereg_v2(tmp_path)), cli)
    assert any("admission_manifest" in p for p in problems)


# ────────── artifacts: 真实数据重建层（Gate D reopening 2026-08-22）──────────
# 恶意矩阵/证据负面测试：hash 必须与真实文件比对（非格式检查）、
# claims/control 必须等于从原始 metrics 的重建值、seeds 精确且唯一。

def _fake_release_ckpt(tmp_path):
    """最小 release checkpoint：config_hash + config_layer（可重算）。"""
    import torch

    from diffusion_policy_3d.rl.dppo_uav.checkpoint import config_hash_of

    layer = {"trainer": "dppo", "actor_lr": 1e-5, "reward_version": "v1.3"}
    path = tmp_path / "release.pt"
    torch.save({"config_hash": config_hash_of(layer),
                "exact_resolved_config": {"config_layer": layer}}, path)
    return str(path), config_hash_of(layer)


def _valid_matrix_artifacts(tmp_path, head="a" * 40):
    """合法 matrix + 真实 artifacts 块 + 与真实文件一致的 hashes。"""
    import hashlib

    m = _valid_matrix()
    ckpt, cfg_hash = _fake_release_ckpt(tmp_path)
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps({"bank_id": "v3", "entries": []}))
    m["g8"]["artifacts"] = {"checkpoint_path": ckpt,
                            "paired_bank_path": str(bank)}
    m["g8"]["hashes"]["final_commit"] = head
    m["g8"]["hashes"]["config_hash"] = cfg_hash
    m["g8"]["hashes"]["checkpoint_hash"] = hashlib.sha256(
        open(ckpt, "rb").read()).hexdigest()
    m["g8"]["hashes"]["bank_hash"] = hashlib.sha256(
        open(str(bank), "rb").read()).hexdigest()
    return m, ckpt, str(bank)


def test_matrix_artifacts_valid_passes(tmp_path):
    m, _, _ = _valid_matrix_artifacts(tmp_path)
    assert verify_matrix_artifacts(m, git_head="a" * 40, tree_clean=True) == []


def test_matrix_artifacts_duplicate_seed_rejected(tmp_path):
    m, _, _ = _valid_matrix_artifacts(tmp_path)
    m["g8"]["seeds"] = [101, 101, 303]  # 非唯一
    assert verify_matrix_artifacts(m, git_head="a" * 40, tree_clean=True) != []


def test_matrix_artifacts_wrong_seed_set_rejected(tmp_path):
    m, _, _ = _valid_matrix_artifacts(tmp_path)
    m["g8"]["seeds"] = [101, 202, 404]  # 非冻结精确集合
    assert verify_matrix_artifacts(m, git_head="a" * 40, tree_clean=True) != []


def test_matrix_artifacts_commit_drift_rejected(tmp_path):
    m, _, _ = _valid_matrix_artifacts(tmp_path)
    # 声称 HEAD 与真实 git HEAD 不一致 → 拒绝（git 由调用层真实探测）
    assert verify_matrix_artifacts(m, git_head="b" * 40, tree_clean=True) != []


def test_matrix_artifacts_checkpoint_hash_drift_rejected(tmp_path):
    m, _, _ = _valid_matrix_artifacts(tmp_path)
    h = m["g8"]["hashes"]["checkpoint_hash"]
    m["g8"]["hashes"]["checkpoint_hash"] = ("0" if h[0] != "0" else "1") + h[1:]
    assert verify_matrix_artifacts(m, git_head="a" * 40, tree_clean=True) != []


def test_matrix_artifacts_config_hash_drift_rejected(tmp_path):
    m, _, _ = _valid_matrix_artifacts(tmp_path)
    m["g8"]["hashes"]["config_hash"] = "c" * 64  # 与真实 ckpt 重算不一致
    assert verify_matrix_artifacts(m, git_head="a" * 40, tree_clean=True) != []


def test_matrix_artifacts_bank_hash_drift_rejected(tmp_path):
    m, _, _ = _valid_matrix_artifacts(tmp_path)
    m["g8"]["hashes"]["bank_hash"] = "d" * 64
    assert verify_matrix_artifacts(m, git_head="a" * 40, tree_clean=True) != []


def test_matrix_artifacts_missing_checkpoint_rejected(tmp_path):
    m, _, _ = _valid_matrix_artifacts(tmp_path)
    m["g8"]["artifacts"]["checkpoint_path"] = str(tmp_path / "absent.pt")
    assert verify_matrix_artifacts(m, git_head="a" * 40, tree_clean=True) != []


def test_matrix_artifacts_missing_bank_rejected(tmp_path):
    m, _, _ = _valid_matrix_artifacts(tmp_path)
    m["g8"]["artifacts"]["paired_bank_path"] = str(tmp_path / "absent.json")
    assert verify_matrix_artifacts(m, git_head="a" * 40, tree_clean=True) != []


def test_matrix_artifacts_dirty_tree_rejected(tmp_path):
    m, _, _ = _valid_matrix_artifacts(tmp_path)
    assert verify_matrix_artifacts(m, git_head="a" * 40, tree_clean=False) != []


def _fake_metrics(tmp_path, name, batch_seq):
    """写 metrics.jsonl：行序 = 全局序，cumulative_env_steps = 1600×i，
    batch_seq 控制 resume 重计数（值随全局 idx 变化，可检测篡改）。"""
    rows = []
    for i, b in enumerate(batch_seq, 1):
        m = {"approx_kl": 0.01 * i, "clipfrac": 0.3 + 0.01 * i,
             "actor_grad_norm": 1.0 * i, "critic_grad_norm": 10.0 * i,
             "value_loss_mean": 2.0 * i}
        rows.append({
            "batch": b,
            "cumulative_env_steps": 1600 * i,
            "env_steps": 1600,
            "checkpoint_kind": "release" if i == len(batch_seq) else "resume",
            "metrics": m,
        })
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(path)


def _valid_evidence_artifacts(tmp_path, head="a" * 40):
    """合法 canary 证据：claims/continuation/control 与真实 metrics 重建
    完全一致（真实键名 value_loss_mean）。"""
    m101 = _fake_metrics(tmp_path, "m101.jsonl", [1, 2, 3, 4, 5])
    m202 = _fake_metrics(tmp_path, "m202.jsonl", [1, 2, 1, 2, 3])
    r101 = rebuild_canary_series(m101)
    r202 = rebuild_canary_series(m202)
    ev = {
        "claims": {
            "head": head,
            "tree": "clean",
            "seeds": {"101": "uninterrupted", "202": "interrupted"},
            "resume_count": {"202": 1},
            "transitions_per_seed": 8000,
            "updates_per_seed": 5,
            "releases_per_seed": 1,
            "paired_records_per_side": 60,
        },
        "tolerances": {"per_update": {"approx_kl": 0.05}},
        "interrupted": {"202": {"resume_points": [2],
                                "resume_continuation": r202["per_update"]}},
        "control": {"101": r101["per_update"], "202": r202["per_update"]},
        "artifacts": {"metrics": {"101": m101, "202": m202}},
    }
    return ev, m101, m202


def test_canary_artifacts_valid_passes(tmp_path):
    ev, _, _ = _valid_evidence_artifacts(tmp_path)
    assert verify_canary_artifacts(ev, git_head="a" * 40, tree_clean=True) == []


def test_canary_artifacts_control_tampered_rejected(tmp_path):
    ev, _, _ = _valid_evidence_artifacts(tmp_path)
    # 手工复制 control 时改了一位数值 → 与原始 metrics 重建不一致
    ev["control"]["202"]["3"]["approx_kl"] += 0.001
    assert verify_canary_artifacts(ev, git_head="a" * 40, tree_clean=True) != []


def test_canary_artifacts_continuation_tampered_rejected(tmp_path):
    ev, _, _ = _valid_evidence_artifacts(tmp_path)
    ev["interrupted"]["202"]["resume_continuation"]["4"]["clipfrac"] += 0.01
    assert verify_canary_artifacts(ev, git_head="a" * 40, tree_clean=True) != []


def test_canary_artifacts_resume_count_mismatch_rejected(tmp_path):
    ev, _, _ = _valid_evidence_artifacts(tmp_path)
    ev["claims"]["resume_count"]["202"] = 2  # 原始 metrics 只有 1 次 batch reset
    assert verify_canary_artifacts(ev, git_head="a" * 40, tree_clean=True) != []


def test_canary_artifacts_resume_points_mismatch_rejected(tmp_path):
    ev, _, _ = _valid_evidence_artifacts(tmp_path)
    ev["interrupted"]["202"]["resume_points"] = [3]  # 真实 reset 在 update 2
    assert verify_canary_artifacts(ev, git_head="a" * 40, tree_clean=True) != []


def test_canary_artifacts_update_count_mismatch_rejected(tmp_path):
    ev, _, _ = _valid_evidence_artifacts(tmp_path)
    ev["claims"]["updates_per_seed"] = 4  # 真实 metrics 5 行
    assert verify_canary_artifacts(ev, git_head="a" * 40, tree_clean=True) != []


def test_canary_artifacts_missing_release_rejected(tmp_path):
    ev, m101, _ = _valid_evidence_artifacts(tmp_path)
    # 末行 checkpoint_kind 改为 resume → 1 release/seed 不可证明
    rows = [json.loads(line) for line in open(m101)]
    rows[-1]["checkpoint_kind"] = "resume"
    with open(m101, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    assert verify_canary_artifacts(ev, git_head="a" * 40, tree_clean=True) != []


def test_canary_artifacts_head_drift_rejected(tmp_path):
    ev, _, _ = _valid_evidence_artifacts(tmp_path)
    assert verify_canary_artifacts(ev, git_head="b" * 40, tree_clean=True) != []


def test_canary_artifacts_dirty_tree_rejected(tmp_path):
    ev, _, _ = _valid_evidence_artifacts(tmp_path)
    assert verify_canary_artifacts(ev, git_head="a" * 40, tree_clean=False) != []


def test_canary_artifacts_uninterrupted_but_reset_rejected(tmp_path):
    # claims 声称 101 uninterrupted，但真实 metrics 有 batch reset
    m101 = _fake_metrics(tmp_path, "m101b.jsonl", [1, 2, 1, 2, 3])
    ev, _, _ = _valid_evidence_artifacts(tmp_path)
    ev["artifacts"]["metrics"]["101"] = m101
    assert verify_canary_artifacts(ev, git_head="a" * 40, tree_clean=True) != []


def test_prereg_binding_budget_update_count_derived(tmp_path):
    # 160000 / 1600 = 100 == updates_per_seed — consistent
    cli = _valid_cli()
    assert verify_prereg_runtime_binding(
        str(_frozen_prereg_v2(tmp_path)), cli) == []
    # break the derived invariant: frozen updates_per_seed=99
    problems = verify_prereg_runtime_binding(
        str(_frozen_prereg_v2(
            tmp_path, budget={
                "seeds": [101, 202, 303],
                "transitions_per_seed": 160000,
                "transitions_per_update": 1600,
                "updates_per_seed": 99,
                "n_envs": 1,
                "releases_per_seed": 1,
                "max_resumes_per_seed": 2,
                "paired_bc_records_per_side": 60,
                "paired_rl_records_per_seed": 60,
            })), cli)
    assert any("updates_per_seed" in p for p in problems)


# ────────── R7-7 Step 6: 局部 verifier 不得再签全局结论 ──────────

def _global_prereg():
    """A complete prereg object freeze-g8 accepts (fields per
    PREREG_REQUIRED_FIELDS)."""
    return {
        "schema_version": "g8-prereg-v2",
        "budget": {"seeds": [101, 202, 303], "transitions_per_seed": 160000},
        "judgements": {"engine_pass": "p", "learning_pass": "l"},
        "thresholds": {"kl_upper": 0.5, "clipfrac_upper": 0.9},
        "stop_rules": ["hard_fault", "budget_exhausted"],
        "resolved_config": {"actor_lr": 1e-5, "critic_lr": 1e-3},
        "tolerances": {"approx_kl": 0.05},
    }


def test_global_stage_hard_fails(capsys):
    """R7-7 Step 6: a request for a GLOBAL engine/admission conclusion
    from r6_admission_report.py must hard-fail and point at the only
    canonical signer (scripts/r7_final_admission.py)."""
    from scripts.r6_admission_report import _main

    rc = _main(["r6_admission_report.py", "global"])
    assert rc != 0
    err = capsys.readouterr().err
    assert "r7_final_admission.py" in err
    assert "cannot sign" in err


def test_freeze_g8_stage_reports_local_scope(tmp_path, capsys):
    """R7-7 Step 6: the freeze-g8 PASS output must carry
    scope=LOCAL_ONLY_NOT_TRAINING_ADMISSION — freezing a prereg is a
    LOCAL slice, never a global training admission."""
    from scripts.r6_admission_report import _main

    prereg = tmp_path / "prereg.json"
    prereg.write_text(json.dumps(_global_prereg()))
    out = tmp_path / "frozen.json"
    rc = _main(["r6_admission_report.py", "freeze-g8",
                       str(prereg), "--out", str(out)])
    assert rc == 0
    out_text = capsys.readouterr().out
    assert "LOCAL_ONLY_NOT_TRAINING_ADMISSION" in out_text
    assert "freeze OK" in out_text
