"""R6-B (§11.3): checkpoint/release/retention 与磁盘事务 — artifact
audit (metrics↔file 一致性 + 单 release + counter==total) 与显式磁盘
预检（oracle / 历史实测 / 明细输出）。

RED phase (HEAD 52ce1eb + R6-A 工作树): artifact_audit / 新版
_disk_preflight / _validate_batch_publish 尚不存在 —— 相应测试以
AttributeError/TypeError/断言失败呈现。已成立的既有语义（孤儿 release
不保护、torn line 容错）作为回归测试同文件保留，防止 B 包重写时退化。
"""

import json
import time
from pathlib import Path

import pytest

from scripts.train_dppo_uav_official import _retain_checkpoints


def _canonical(i: int) -> str:
    return f"dppo_uav_envstep_{i:09d}_ppoupd_{i:03d}.pt"


def _make_run(out: Path, n_files: int, release_idx: int = -1,
              release_total: int = None):
    """Seed a canonical output dir: n canonical .pt files, metrics.jsonl
    with one record per batch; ``release_idx`` >= 0 marks the batch whose
    record is a release (counter = release_total when given)."""
    out.mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(1, n_files + 1):
        p = out / _canonical(i)
        p.write_bytes(b"x" * (10 + i))
        kind = "release" if i == release_idx else "resume"
        rec = {"batch": i, "checkpoint_kind": kind,
               "checkpoint_path": str(p),
               "cumulative_env_steps": release_total if (
                   i == release_idx and release_total is not None)
               else i}
        lines.append(rec)
        p.touch()
        time.sleep(0.01)
    (out / "metrics.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")
    return out


# ═══════════════════════════════════════════════════════════════════
# §11.3.2 artifact audit — 三项 hard-fail
# ═══════════════════════════════════════════════════════════════════


def test_red1_metrics_points_to_missing_file_hard_fails(tmp_path):
    """§11.3.2: metrics 指向不存在的 checkpoint 文件 → artifact audit
    hard-fail（file 已被外部删除/清理 = 证据断裂，绝不允许静默继续）。"""
    from scripts.train_dppo_uav_official import artifact_audit

    out = _make_run(tmp_path, 2)
    broken = out / _canonical(2)
    broken.unlink()
    with pytest.raises(RuntimeError) as e:
        artifact_audit(out)
    assert "missing" in str(e.value) or "不存在" in str(e.value), e.value


def test_red2_multiple_final_releases_hard_fails(tmp_path):
    """§11.3.2: 同一 run 出现多个 final release（batch 1 就 release +
    后续又 release）→ hard-fail —— 每 run 只能有 exactly one release。"""
    from scripts.train_dppo_uav_official import artifact_audit

    out = tmp_path / "run"
    out.mkdir()
    for i in (1, 2):
        p = out / _canonical(i)
        p.write_bytes(b"r")
        (out / "metrics.jsonl").open("a", encoding="utf-8").write(
            json.dumps({"batch": i, "checkpoint_kind": "release",
                        "checkpoint_path": str(p),
                        "cumulative_env_steps": i}) + "\n")
    with pytest.raises(RuntimeError) as e:
        artifact_audit(out)
    assert "release" in str(e.value), e.value


def test_red3_release_counter_mismatch_hard_fails(tmp_path):
    """§11.3.2: transition 模式 release record 的累计 transition 计数 !=
    run-level total → hard-fail（release counter 必须等于 total）。"""
    from scripts.train_dppo_uav_official import artifact_audit

    # release record 声称 counter=24，但 run total 是 25 —— 终局 audit
    # 必须 hard-fail（release counter 必须等于 total）
    out = _make_run(tmp_path, 2, release_idx=2, release_total=24)
    with pytest.raises(RuntimeError) as e:
        artifact_audit(out, total_transitions=25)
    assert "25" in str(e.value), e.value


def test_red4_transition_run_without_release_hard_fails(tmp_path):
    """§11.3.2/§11.3.4: transition 模式跑完必须恰有一个 final release
    —— 只有 resume 记录的 run 是未终局状态，audit hard-fail。"""
    from scripts.train_dppo_uav_official import artifact_audit

    out = _make_run(tmp_path, 4)  # all resume
    with pytest.raises(RuntimeError) as e:
        artifact_audit(out, total_transitions=25)
    assert "release" in str(e.value), e.value


def test_green1_clean_run_audit_passes(tmp_path):
    """§11.3.2 合法终局：单 release + counter==total + 所有 file 存在 →
    audit PASS，返回 release 路径集。"""
    from scripts.train_dppo_uav_official import artifact_audit

    out = _make_run(tmp_path, 100, release_idx=100, release_total=1600)
    report = artifact_audit(out, total_transitions=1600)
    assert len(report["release_paths"]) == 1
    assert report["release_paths"][0].name == _canonical(100)


# ═══════════════════════════════════════════════════════════════════
# §11.3.4 crash window / torn line — 既有语义回归保护
# ═══════════════════════════════════════════════════════════════════


def test_green2_orphan_release_not_protected(tmp_path):
    """§11.3.4: checkpoint 已写、metrics 未 append 时 crash → 重启后该
    孤儿 release 文件无对应 record → retention 必须轮转它，绝不把它
    当 release 保护（release 的 truth 只有 metrics sidecar）。"""
    out = tmp_path / "run"
    out.mkdir()
    files = []
    for i in range(1, 5):
        p = out / _canonical(i)
        p.write_bytes(b"x")
        files.append(p)
        time.sleep(0.01)
    # 孤儿 release 文件（batch 4）mtime 最早；只有 batch 1-3 有 record
    files[3].touch()
    time.sleep(0.01)
    for p in files[:3]:
        p.touch()
        time.sleep(0.01)
    (out / "metrics.jsonl").write_text(
        "".join(json.dumps(
            {"batch": i, "checkpoint_kind": "resume",
             "checkpoint_path": str(files[i - 1]),
             "cumulative_env_steps": i}) + "\n"
            for i in (1, 2, 3)), encoding="utf-8")
    removed = _retain_checkpoints(out, keep_latest=2)
    survivors = sorted(p.name for p in out.glob("dppo_uav_*.pt"))
    assert removed == 2, removed
    assert _canonical(4) not in survivors, (
        "orphan release must NOT be protected by retention — the metrics "
        "sidecar is the only release truth (crash window: file written, "
        "record not appended)")


def test_green3_torn_line_does_not_protect_anything(tmp_path):
    """§11.3.2/§11.3.4: torn trailing metrics line 被报告并忽略 ——
    它既不能保护一个未提交 release，也不破坏此前合法 records。"""
    from scripts.train_dppo_uav_official import artifact_audit

    out = _make_run(tmp_path, 2)
    # 追加一条 torn 行，其中包含 "release" 字样碎片
    with (out / "metrics.jsonl").open("a", encoding="utf-8") as f:
        f.write('{"checkpoint_kind": "rele')
    # 不传 total_transitions（episode 语义）——本测试只验证 torn line
    # 被报告且不物化 release，不涉及 transition 无 release 分支。
    report = artifact_audit(out)
    assert len(report["problems"]) == 1, report["problems"]
    assert len(report["release_paths"]) == 0, (
        "a torn line must never materialize a release")


# ═══════════════════════════════════════════════════════════════════
# §11.3.3 磁盘预检 — 显式估计 + 明细输出
# ═══════════════════════════════════════════════════════════════════


def test_red5_empty_dir_uses_g7_oracle_detail(tmp_path):
    """§11.3.3/§11.3.4: 空 output dir 必须用实测 G7 量级 oracle
    （resume≈7.5MB、release≈293MB）估计，而不是 96MiB×N；明细必须
    输出各项（resume_keep=2 / release=1 / temp / reserves）与来源。"""
    from scripts.train_dppo_uav_official import (
        DISK_ORACLE_RELEASE_BYTES,
        DISK_ORACLE_RESUME_BYTES,
        _disk_preflight,
    )

    detail = _disk_preflight(tmp_path, max_updates=100)
    assert detail["source"] == "oracle", detail
    assert detail["resume_bytes"] == DISK_ORACLE_RESUME_BYTES
    assert detail["release_bytes"] == DISK_ORACLE_RELEASE_BYTES
    assert detail["resume_keep_count"] == 2
    assert detail["release_count"] == 1
    expected = (2 * DISK_ORACLE_RESUME_BYTES + DISK_ORACLE_RELEASE_BYTES
                + DISK_ORACLE_RESUME_BYTES  # temporary_publish ≈ 1 resume
                + detail["metrics_and_eval_reserve_bytes"]
                + detail["safety_reserve_bytes"])
    assert detail["projected_bytes"] == expected, detail


def test_red6_history_max_sizes_when_present(tmp_path):
    """§11.3.3: dir 含同 schema 历史 checkpoint 时，resume/release 估计
    来自历史实测最大值（source="history"），不用 oracle。"""
    from scripts.train_dppo_uav_official import _disk_preflight

    out = _make_run(tmp_path, 3, release_idx=3)
    # 加大文件体量以区分 resume/release
    for i in range(1, 4):
        (out / _canonical(i)).write_bytes(b"y" * (50_000 + i * 10_000))
    detail = _disk_preflight(out, max_updates=10)
    assert detail["source"] == "history", detail
    assert detail["resume_bytes"] == 70_000, detail   # max resume file
    assert detail["release_bytes"] == 80_000, detail  # release file
    assert "free_bytes" in detail and detail["free_bytes"] > 0


# ═══════════════════════════════════════════════════════════════════
# §11.3.1.2 入口发布验证 — 每批契约
# ═══════════════════════════════════════════════════════════════════


def test_red7_entry_publish_validation_rejects_missing_checkpoint(tmp_path):
    """§11.3.1.2: 批次结果没有物化 checkpoint → 入口验证 hard-fail
    （发布契约：每批必须有可验证的 checkpoint）。"""
    from scripts.train_dppo_uav_official import _validate_batch_publish

    with pytest.raises(RuntimeError) as e:
        _validate_batch_publish(
            {"checkpoint_path": None, "checkpoint_kind": "resume",
             "env_steps": 8},
            cumulative_transitions=8, total_transitions=None,
        )
    assert "checkpoint" in str(e.value), e.value


def test_red8_entry_publish_validation_release_without_finality(tmp_path):
    """§11.3.1.2: 声称 release 但累计计数未达 total → 入口验证
    hard-fail（transition 模式 release ⟺ counter == total）。"""
    from scripts.train_dppo_uav_official import _validate_batch_publish

    ckpt = tmp_path / "r.pt"
    ckpt.write_bytes(b"x")
    with pytest.raises(RuntimeError) as e:
        _validate_batch_publish(
            {"checkpoint_path": str(ckpt), "checkpoint_kind": "release",
             "env_steps": 8},
            cumulative_transitions=24, total_transitions=25,
        )
    assert "25" in str(e.value), e.value


def test_green4_entry_publish_validation_passes_resume(tmp_path):
    """§11.3.1.2 合法路径：resume 批次 + 文件存在 + 计数未超 → PASS。"""
    from scripts.train_dppo_uav_official import _validate_batch_publish

    ckpt = tmp_path / "r.pt"
    ckpt.write_bytes(b"x")
    _validate_batch_publish(
        {"checkpoint_path": str(ckpt), "checkpoint_kind": "resume",
         "env_steps": 8},
        cumulative_transitions=8, total_transitions=25,
    )
    _validate_batch_publish(
        {"checkpoint_path": str(ckpt), "checkpoint_kind": "release",
         "env_steps": 17},
        cumulative_transitions=25, total_transitions=25,
    )


# ═══════════════════════════════════════════════════════════════════
# §11.3.2 retention 轮换豁免 — Gate B R6-DISK-1 REWORK 回归
# ═══════════════════════════════════════════════════════════════════


def test_green5_audit_passes_rotated_resumes(tmp_path):
    """§11.3.2 S3: 被 retention 合法轮换的旧 resume 缺失不是证据断裂
    —— release + 最新 keep_latest 个 resume 存在即 audit PASS（Gate B
    R6-DISK-1 REWORK：旧版把 retention 的预期结果当断裂，任何 >=4
    batch 的合法运行终局必 audit FAIL）。"""
    from scripts.train_dppo_uav_official import artifact_audit

    out = _make_run(tmp_path, 6, release_idx=6, release_total=1600)
    for i in (1, 2, 3, 4):  # retention 轮换掉的旧 resume
        (out / _canonical(i)).unlink()
    report = artifact_audit(out, total_transitions=1600)
    assert len(report["release_paths"]) == 1
    assert report["release_paths"][0].name == _canonical(6)
    assert not report["problems"]


def test_red9_audit_fails_missing_newest_resume(tmp_path):
    """§11.3.2: 最新 keep_latest 个 resume 缺失 = 证据断裂 → audit
    hard-fail（豁免只覆盖被轮换的旧 resume，不覆盖尚未轮换的新 resume
    —— 否则审计可以被任意删文件静默绕过）。"""
    from scripts.train_dppo_uav_official import artifact_audit

    out = _make_run(tmp_path, 6, release_idx=6, release_total=1600)
    (out / _canonical(5)).unlink()  # 最新 resume（keep_latest=2 内）
    with pytest.raises(RuntimeError) as e:
        artifact_audit(out, total_transitions=1600)
    assert "missing" in str(e.value) or "不存在" in str(e.value), e.value


def test_red10_audit_release_missing_still_fails(tmp_path):
    """§11.3.2: release 永远必须存在 —— retention 豁免不适用于 release，
    resume 全部完好也救不回缺失的 release（终局证据必须完整）。"""
    from scripts.train_dppo_uav_official import artifact_audit

    out = _make_run(tmp_path, 6, release_idx=6, release_total=1600)
    (out / _canonical(6)).unlink()  # final release
    with pytest.raises(RuntimeError) as e:
        artifact_audit(out, total_transitions=1600)
    assert "missing" in str(e.value) or "不存在" in str(e.value), e.value


# ═══════════════════════════════════════════════════════════════════
# §11.3.1 step 2 入口 batch target 验证 + §11.2.4 INCOMPLETE_BUDGET
# ═══════════════════════════════════════════════════════════════════


def test_green6_incomplete_budget_marker_content(tmp_path):
    """§11.2.4: INCOMPLETE_BUDGET marker 是 batch 完整性证据链 ——
    JSON 精确记录 planned/actual/cumulative/total 四个数字字段与
    说明消息（Gate B R6-DISK-1 REWORK CE-2：partial batch 曾静默
    通过）。"""
    from scripts.train_dppo_uav_official import _write_incomplete_budget_marker

    _write_incomplete_budget_marker(
        tmp_path, planned=5000, actual=3000, cumulative=3000, total=16000)
    marker = tmp_path / "INCOMPLETE_BUDGET"
    assert marker.is_file()
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["planned_batch_transitions"] == 5000
    assert payload["actual_batch_transitions"] == 3000
    assert payload["cumulative_transitions"] == 3000
    assert payload["total_transitions"] == 16000
    assert "partial batch" in payload["message"]


def test_verify_batch_target_partial_writes_marker(tmp_path):
    """§11.3.1 step 2: transition 模式 actual != 本批派生目标 且 budget
    未满 = admission/sampler 用尽的 partial batch → 落 marker + 警告，
    但不抛异常（§11.2.4 保险丝语义：partial 允许继续，后续 batch 重算
    目标补齐）。"""
    from scripts.train_dppo_uav_official import _verify_batch_target

    _verify_batch_target(
        tmp_path, batch_idx=2, max_updates=10,
        actual=3000, planned=5000, cumulative=3000, total=16000,
    )
    assert (tmp_path / "INCOMPLETE_BUDGET").is_file()


def test_verify_batch_target_complete_batch_no_marker(tmp_path):
    """§11.3.1 step 2 合法路径:actual == planned → no-op（无 marker）。"""
    from scripts.train_dppo_uav_official import _verify_batch_target

    _verify_batch_target(
        tmp_path, batch_idx=1, max_updates=10,
        actual=5000, planned=5000, cumulative=5000, total=16000,
    )
    assert not (tmp_path / "INCOMPLETE_BUDGET").exists()


def test_verify_batch_target_episode_mode_noop(tmp_path):
    """§11.3.1 step 2: episode 模式（total=None）无 transition budget
    校验 → no-op，即使 actual != planned。"""
    from scripts.train_dppo_uav_official import _verify_batch_target

    _verify_batch_target(
        tmp_path, batch_idx=1, max_updates=10,
        actual=3000, planned=5000, cumulative=3000, total=None,
    )
    assert not (tmp_path / "INCOMPLETE_BUDGET").exists()


def test_verify_batch_target_budget_complete_partial_ok(tmp_path):
    """§11.3.1 step 2 边界：末批 actual != planned 但 budget 已满
    （派生目标超出剩余额度）→ 不算 partial，无 marker。"""
    from scripts.train_dppo_uav_official import _verify_batch_target

    _verify_batch_target(
        tmp_path, batch_idx=9, max_updates=10,
        actual=1000, planned=5000, cumulative=16000, total=16000,
    )
    assert not (tmp_path / "INCOMPLETE_BUDGET").exists()


def test_green7_entry_loop_dry_run_transition_audit_passes(tmp_path):
    """§11.3.1/§11.3.2 entry-loop e2e dry-run（R6-DISK-1）：复刻入口
    循环 transition 模式每批调用序列 —— _verify_batch_target →
    _validate_batch_publish → persist_batch_metrics → _retain_checkpoints，
    终局恰好一个 release（counter==total）、retention 轮换旧 resume 后
    artifact audit PASS。直接验证 Gate B R6-DISK-1 修复：>=4 batch 的
    合法 transition 运行终局 audit 必须通过（release 落地后只留下
    最新 1 个 resume + release）。"""
    import time

    from diffusion_policy_3d.rl.dppo_uav.train_loop import compute_batch_target
    from scripts.train_dppo_uav_official import (
        _retain_checkpoints,
        _validate_batch_publish,
        _verify_batch_target,
        artifact_audit,
        persist_batch_metrics,
    )

    out = tmp_path / "run"
    out.mkdir()
    horizon, total = 1000, 4000  # 4 批 [1000,1000,1000,1000]
    cumulative = 0
    metrics_path = out / "metrics.jsonl"
    for batch_idx in range(1, 5):
        planned = compute_batch_target(total, cumulative, horizon)
        assert planned == 1000
        kind = "release" if cumulative + planned >= total else "resume"
        ckpt = out / _canonical(batch_idx)
        ckpt.write_bytes(b"x" * (10 + batch_idx))
        ckpt.touch()
        time.sleep(0.01)  # 文件 mtime 序 = 批次序（retention 依赖）
        res = {
            "env_steps": planned,
            "completed_episodes": 1,
            "voided_episodes": 0,
            "routing": {},
            "metrics": {"update_batches": 1, "ratio_mean": 1.0},
            "checkpoint_path": str(ckpt),
            "checkpoint_kind": kind,
        }
        _verify_batch_target(
            out, batch_idx, max_updates=4,
            actual=planned, planned=planned,
            cumulative=cumulative + planned, total=total,
        )
        _validate_batch_publish(
            res, cumulative_transitions=cumulative + planned,
            total_transitions=total,
        )
        persist_batch_metrics(
            metrics_path, batch_idx, res,
            cumulative_env_steps=cumulative + planned,
        )
        cumulative += planned
        _retain_checkpoints(out)
    audit = artifact_audit(out, total_transitions=total)
    assert len(audit["release_paths"]) == 1
    assert audit["release_paths"][0].name == _canonical(4)
    assert not audit["problems"]
    # retention 按预期轮换：release + 最新 resume 存活，旧 resume 已删
    survivors = sorted(p.name for p in out.glob("dppo_uav_*.pt"))
    assert survivors == [_canonical(3), _canonical(4)], survivors
