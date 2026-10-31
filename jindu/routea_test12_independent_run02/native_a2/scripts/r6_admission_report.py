"""R6-G (F-05 / R6-EVID-1): final-tree admission evidence orchestration.

Guide §11.8.1–11.8.4 + §11.9 G 包 —— canonical 验证与报告（evidence
glue），不改变任何训练/评估语义：

- ``verify_run_matrix`` — §11.8.4 resolved run matrix 的机械验证
  （程序验证，禁止人工目测）：3 seeds × 160,000 transitions、
  1,600/update、100 updates/seed、1 env、exactly 1 release/seed、
  ≤2 resume/seed、60 paired BC once + 60 paired RL/seed、strict
  final commit/tree/config/checkpoint/bank hashes；
- ``verify_canary_evidence`` — R6.4 canary 证据链验证（§11.9 G 包
  负向测试）：commit drift（claims HEAD/tree vs 实际）、interrupted
  canary（resume≤2、interrupted marker、continuation vs control 逐项
  容差）、incomplete matrix（种子/预算/配对缺项）；
- ``freeze_g8_prereg`` — G8 预注册冻结（§8.2/§11.8.4）：必需字段
  schema + 确定性 content hash；冻结后判据不可修改（训练后改判据 =
  伪注册）。

CLI stages（canary/calibrate 的 GPU 编排驱动器在 Phase 8/9 使用）：
  --stage matrix     ：读取 resolved matrix JSON → 机械验证 → 报告
  --stage evidence   ：读取 canary 证据 JSON → 验证 → 报告
  --stage freeze-g8  ：生成/校验冻结 prereg JSON

数字来自 Guide 固定条款，不是本模块发明：
  G8: 3 seeds x 160k / 100 updates / 1 release / <=2 resume / 60 paired
  R6.4 canary: seeds {101,202} / 8k / 5 updates / 1 release / 60 paired
"""

import hashlib
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

# §11.8.4 G8 固定数字（pre-registration 判据，训练前冻结）
VERIFY_MATRIX_BUDGETS: Dict[str, Any] = {
    "seeds": 3,
    "transitions_per_seed": 160000,
    "transitions_per_update": 1600,
    "updates_per_seed": 100,
    "n_envs": 1,
    "releases_per_seed": 1,
    "max_resumes_per_seed": 2,
    "paired_bc_records_per_side": 60,
    "paired_rl_records_per_seed": 60,
}

# §11.8.2 R6.4 engine canary 固定配置
CANARY_BUDGET: Dict[str, Any] = {
    "seeds": ["101", "202"],
    "transitions_per_seed": 8000,
    "transitions_per_update": 1600,
    "updates_per_seed": 5,
    "n_envs": 1,
    "releases_per_seed": 1,
    "max_resumes_per_seed": 2,
    "paired_records_per_side": 60,
}

# §11.8.2 固定保守超参（避免 canary 变成超参试验）
CANARY_FIXED_CONFIG: Dict[str, Any] = {
    "actor_lr": 1e-5,
    "critic_lr": 1e-3,
    "update_epochs": 1,
    "lr_scheduler": "none",
    "reward_scale": "none",
}

# §11.8.3 R6.5 阶段 A 候选（单轴）
R65_SCREENING_ACTOR_LRS: List[float] = [1e-5, 3e-5, 1e-4]

HEX_RE = re.compile(r"^[0-9a-f]+$")
SHA256_HEX_LEN = 64
GIT_COMMIT_HEX_LEN = 40


def _problems_add(problems: List[str], msg: str) -> None:
    problems.append(msg)


def _check_hex(problems: List[str], value: Any, name: str,
               length: int) -> None:
    if not isinstance(value, str) or len(value) != length \
            or not HEX_RE.match(value):
        _problems_add(
            problems,
            f"strict hash drift: {name} must be {length}-char lowercase "
            f"hex, got {value!r}",
        )


def verify_run_matrix(matrix: Dict[str, Any],
                      ) -> List[str]:
    """§11.8.4 resolved run matrix 机械验证 → problems（[] = PASS）。

    校验 G8 固定数字（VERIFY_MATRIX_BUDGETS）与 strict hashes 结构。
    程序验证，禁止人工目测；任何不满足项都是 matrix 拒绝，不是警告。
    """
    problems: List[str] = []
    if not isinstance(matrix, dict) or "g8" not in matrix:
        return ["matrix missing required 'g8' block"]
    g8 = matrix["g8"]
    if not isinstance(g8, dict):
        return ["matrix['g8'] must be an object"]

    seeds = g8.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != \
            VERIFY_MATRIX_BUDGETS["seeds"]:
        _problems_add(
            problems,
            f"incomplete matrix: exactly 3 seeds required, got {seeds!r}",
        )
    else:
        for s in seeds:
            if not isinstance(s, int):
                _problems_add(
                    problems, f"seed must be int, got {s!r}")

    checks = [
        ("transitions_per_seed", VERIFY_MATRIX_BUDGETS["transitions_per_seed"]),
        ("transitions_per_update", VERIFY_MATRIX_BUDGETS["transitions_per_update"]),
        ("updates_per_seed", VERIFY_MATRIX_BUDGETS["updates_per_seed"]),
        ("n_envs", VERIFY_MATRIX_BUDGETS["n_envs"]),
        ("releases_per_seed", VERIFY_MATRIX_BUDGETS["releases_per_seed"]),
        ("paired_bc_records_per_side", VERIFY_MATRIX_BUDGETS["paired_bc_records_per_side"]),
        ("paired_rl_records_per_seed", VERIFY_MATRIX_BUDGETS["paired_rl_records_per_seed"]),
    ]
    for key, expected in checks:
        if g8.get(key) != expected:
            _problems_add(
                problems,
                f"incomplete matrix: {key} must equal {expected}, "
                f"got {g8.get(key)!r}",
            )
    max_resume = g8.get("max_resumes_per_seed")
    if max_resume != 2 and max_resume != 1:
        _problems_add(
            problems,
            f"incomplete matrix: max_resumes_per_seed must be <= 2, "
            f"got {max_resume!r}",
        )

    hashes = g8.get("hashes")
    if not isinstance(hashes, dict):
        _problems_add(problems, "matrix missing required hashes block")
    else:
        _check_hex(problems, hashes.get("final_commit"),
                   "hashes.final_commit", GIT_COMMIT_HEX_LEN)
        for hkey in ("config_hash", "checkpoint_hash", "bank_hash"):
            _check_hex(problems, hashes.get(hkey), f"hashes.{hkey}",
                       SHA256_HEX_LEN)

    if not isinstance(matrix.get("config"), dict) or \
            not matrix["config"]:
        _problems_add(problems, "matrix missing resolved config block")
    return problems


def _require(problems: List[str], evidence: Dict[str, Any], key: str,
             expected: Any) -> None:
    if evidence.get(key) != expected:
        _problems_add(
            problems,
            f"incomplete canary evidence: {key} must equal {expected}, "
            f"got {evidence.get(key)!r}",
        )


def verify_canary_evidence(evidence: Dict[str, Any],
                           current_head: str,
                           ) -> List[str]:
    """R6.4 canary 证据链验证 → problems（[] = PASS）。

    §11.9 G 包负向测试：
    - commit drift：claims.head != current_head → 拒绝（fresh final-tree
      规则：canary 只对启动它的 final HEAD 有效）；
    - interrupted canary：claims 声称 interrupted 的 seed 必须有
      interrupted 段（resume_points/continuation）+ 同 seed control 段，
      resume 次数 ≤2，continuation/control 索引集相等且覆盖全部 update；
    - incomplete matrix：种子覆盖（101 uninterrupted + 202 interrupted）、
      预算（8k/5 updates/1 release/60 paired）缺一即拒绝。

    per-update 容差对比（2026-08-21 裁决 A）降级为报告性附件
    （tolerance_diff_report）不 gate —— Guide §11.8.2 通过条件 =
    "R6.4 只判 engine composition：exact [1600]×5、one release、
    resume≤2、standard JSON、strict identity、五 index、LoRA
    update/base zero、GAE/oracle、paired provenance"。
    """
    problems: List[str] = []
    if not isinstance(evidence, dict) or "claims" not in evidence:
        return ["canary evidence missing required claims block"]
    claims = evidence["claims"]
    if not isinstance(claims, dict):
        return ["claims must be an object"]

    if claims.get("head") != current_head:
        _problems_add(
            problems,
            "commit drift: canary claims HEAD "
            f"{claims.get('head')!r} != current HEAD {current_head!r} — "
            "the canary is only valid for the final HEAD it started on "
            "(§11.8.1)",
        )
    if claims.get("tree") != "clean":
        _problems_add(
            problems,
            "commit drift: canary claims tree "
            f"{claims.get('tree')!r} != 'clean' — canary must start on "
            "a clean final-commit tree (§11.8.1)",
        )

    seeds = claims.get("seeds")
    if not isinstance(seeds, dict):
        _problems_add(problems, "claims missing seeds block")
        return problems
    if "101" not in seeds or seeds.get("101") != "uninterrupted":
        _problems_add(
            problems,
            "incomplete matrix: seed101 must be present as "
            "uninterrupted control",
        )
    if "202" not in seeds or seeds.get("202") != "interrupted":
        _problems_add(
            problems,
            "incomplete matrix: seed202 must be present as interrupted "
            "(cold-resume required, §11.8.2)",
        )

    _require(problems, claims, "transitions_per_seed", 8000)
    _require(problems, claims, "updates_per_seed", 5)
    _require(problems, claims, "releases_per_seed", 1)
    paired = claims.get("paired_records_per_side")
    if not isinstance(paired, int) or paired < 60:
        _problems_add(
            problems,
            "incomplete matrix: paired_records_per_side must be >= 60 "
            "(12 entries x 5 replicas, §11.8.2), got "
            f"{paired!r}",
        )

    resume_counts = claims.get("resume_count")
    if not isinstance(resume_counts, dict):
        _problems_add(problems, "claims missing resume_count block")
        return problems
    for seed, count in resume_counts.items():
        # P2 glue (2026-08-19 Gate B): bool 是 int 的伪装 —— 计数必须是
        # 真 int，且 <=2（§11.8.2）
        if isinstance(count, bool) or not isinstance(count, int) or \
                count < 0 or count > 2:
            _problems_add(
                problems,
                f"interrupted canary: seed {seed} resume_count={count!r} "
                "exceeds the <=2 limit (§11.8.2)",
            )

    interrupted = evidence.get("interrupted")
    control = evidence.get("control")
    # P2 glue: 完整性 —— 声称 interrupted 的 seed 必须有 resume_count 条目
    for seed, kind in seeds.items():
        if kind == "interrupted" and seed not in resume_counts:
            _problems_add(
                problems,
                f"interrupted canary: seed {seed} claimed interrupted "
                "but has no resume_count entry (§11.8.2)",
            )
    for seed, kind in seeds.items():
        if kind == "interrupted":
            if not isinstance(interrupted, dict) or seed not in interrupted:
                _problems_add(
                    problems,
                    f"missing interrupt marker: seed {seed} claimed "
                    "interrupted but has no interrupted segment record",
                )
                continue
            seg = interrupted[seed]
            resume_points = seg.get("resume_points")
            if not isinstance(resume_points, list) or not resume_points:
                _problems_add(
                    problems,
                    f"interrupted canary: seed {seed} missing non-empty "
                    "resume_points",
                )
            else:
                # P2 glue: 内容 —— 每个 resume point 必须是真 int update
                # index（bool/float 拒绝，不得混入"第 2.0 次"这种伪条目）
                for rp in resume_points:
                    if isinstance(rp, bool) or not isinstance(rp, int):
                        _problems_add(
                            problems,
                            f"interrupted canary: seed {seed} resume_points "
                            f"entry {rp!r} must be an int update index",
                        )
                        break
                # P2 glue: 条目数必须与 resume_count 一致 —— 每次 resume
                # 都必须被列出（resume 没记录 = 证据缺失）
                rcount = resume_counts.get(seed)
                if isinstance(rcount, int) and \
                        not isinstance(rcount, bool) and \
                        len(resume_points) != rcount:
                    _problems_add(
                        problems,
                        f"interrupted canary: seed {seed} resume_count "
                        f"{rcount} != len(resume_points) "
                        f"{len(resume_points)} — every resume must be "
                        "listed",
                    )
            cont = seg.get("resume_continuation")
            if not isinstance(cont, dict) or not cont:
                _problems_add(
                    problems,
                    f"interrupted canary: seed {seed} missing "
                    "resume_continuation (cold-restart continuation "
                    "metrics)",
                )
            if not isinstance(control, dict) or seed not in control or \
                    not isinstance(control[seed], dict) or \
                    not control[seed]:
                _problems_add(
                    problems,
                    f"interrupted canary: seed {seed} missing "
                    "uninterrupted control block for comparison",
                )
            else:
                ctrl = control[seed]
                # P2 glue: 覆盖计数 —— continuation 与 control 必须逐同一
                # 批 update 对比（缺一方 index 不得静默跳过）
                cont_idxs = set(cont) if isinstance(cont, dict) else set()
                ctrl_idxs = set(ctrl)
                if cont_idxs != ctrl_idxs:
                    _problems_add(
                        problems,
                        f"interrupted canary: seed {seed} continuation "
                        f"indices {sorted(cont_idxs)} != control indices "
                        f"{sorted(ctrl_idxs)} — the same updates must be "
                        "compared (§11.8.2)",
                    )
                else:
                    # P2 glue: 双方都截断（都不覆盖全部 updates_per_seed）
                    # —— 两段截断的对比证明不了 engine parity
                    updates = claims.get("updates_per_seed")
                    if isinstance(updates, int) and \
                            not isinstance(updates, bool) and updates >= 1:
                        expected = {str(i) for i in range(1, updates + 1)}
                        if cont_idxs != expected:
                            _problems_add(
                                problems,
                                f"interrupted canary: seed {seed} "
                                "continuation and control are both "
                                "truncated (covered "
                                f"{sorted(cont_idxs)} of "
                                f"{sorted(expected)}) — two truncated "
                                "segments prove nothing about the "
                                "missing tail",
                            )
                # 2026-08-21 裁决 A：per-update 容差超限降级为报告性
                # （tolerance_diff_report），不再 gate —— Guide §11.8.2
                # 通过条件 = "只判 engine composition" 清单（docstring 详述）。
    return problems


# ────────────────── Gate D reopening (2026-08-22): artifacts ──────────────────
# admission verifier 重写 —— 从原始数据（metrics.jsonl / checkpoint 链 /
# paired bank / git 文件系统）直接重建结论，不再信任手工复制的 claims 和
# control 字典。结构层（verify_run_matrix / verify_canary_evidence）保留
# 冻结判据校验；*_artifacts 层与真实文件逐项比对，任何不一致即拒绝。
# hash 必须与真实文件比对（SHA256 重算），而非仅检查 hex 格式。

# 真实 metrics.jsonl 的 per-update 指标键名（value_loss_mean，非旧
# evidence 手工字典里的 value_loss）
CANARY_REBUILT_METRICS: Tuple[str, ...] = (
    "approx_kl", "clipfrac", "actor_grad_norm", "critic_grad_norm",
    "value_loss_mean",
)

# §11.8.4 G8 冻结种子集 —— 精确集合（seeds 必须精确且唯一，非仅数量 3）
FROZEN_G8_SEEDS: List[int] = [101, 202, 303]


def _file_sha256(path: str) -> str:
    """真实文件内容的 SHA256（准入 hash 与真实文件比对，非格式检查）。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def git_rev_parse(repo_root: str) -> str:
    """真实 git HEAD（subprocess；失败 → 空串，由调用层拒绝）。"""
    import subprocess

    out = subprocess.run(
        ["git", "-C", repo_root, "rev-parse", "HEAD"],
        capture_output=True, text=True,
    )
    return out.stdout.strip() if out.returncode == 0 else ""


def git_tree_clean(repo_root: str) -> bool:
    """真实 git tree 状态（porcelain 空 = clean）。"""
    import subprocess

    out = subprocess.run(
        ["git", "-C", repo_root, "status", "--porcelain"],
        capture_output=True, text=True,
    )
    return out.returncode == 0 and not out.stdout.strip()


def verify_matrix_artifacts(matrix: Dict[str, Any],
                            git_head: str, tree_clean: bool) -> List[str]:
    """G8 resolved matrix 真实数据重建层 → problems（[] = PASS）。

    结构层（verify_run_matrix）只查 hex 格式；本层把 hashes 与真实
    文件/git 比对：
    - seeds 必须精确 == {101, 202, 303} 且无重复（精确集合，非仅数量）；
    - tree 必须 clean（准入只对 final-commit 树有效，§11.8.1）；
    - hashes.final_commit 必须 == 真实 git HEAD；
    - hashes.checkpoint_hash == 真实 release checkpoint 文件 SHA256；
    - hashes.config_hash == 从 checkpoint 的 config_layer 重算
      （与 checkpoint.verify_config_layer_identity 同款，防 tautology）；
    - hashes.bank_hash == 真实 paired bank 文件 SHA256。
    路径来自 matrix['g8']['artifacts']（checkpoint_path / paired_bank_path）。
    """
    problems: List[str] = []
    g8 = matrix.get("g8")
    if not isinstance(g8, dict):
        return problems  # 结构层已拒绝；本层避免二次崩溃
    seeds = g8.get("seeds")
    if not isinstance(seeds, list) or sorted(seeds) != FROZEN_G8_SEEDS \
            or len(set(seeds)) != len(FROZEN_G8_SEEDS):
        _problems_add(
            problems,
            f"matrix artifacts: seeds must be exactly {FROZEN_G8_SEEDS} "
            f"(precise set, no dupes), got {seeds!r}",
        )
    if not tree_clean:
        _problems_add(
            problems,
            "matrix artifacts: working tree is not clean — the resolved "
            "matrix is only valid on the final-commit tree (§11.8.1)",
        )
    hashes = g8.get("hashes")
    if not isinstance(hashes, dict):
        return problems
    if hashes.get("final_commit") != git_head:
        _problems_add(
            problems,
            "commit drift vs real git: matrix final_commit "
            f"{hashes.get('final_commit')!r} != git HEAD {git_head!r} — "
            "rebuilt from git, never from the hand-copied matrix",
        )
    artifacts = g8.get("artifacts")
    if not isinstance(artifacts, dict):
        _problems_add(
            problems,
            "matrix artifacts: missing g8.artifacts block "
            "(checkpoint_path / paired_bank_path)",
        )
        return problems
    for key in ("checkpoint_path", "paired_bank_path"):
        path = artifacts.get(key)
        if not isinstance(path, str) or not os.path.isfile(path):
            _problems_add(
                problems,
                f"matrix artifacts: {key} {path!r} does not exist",
            )
    ckpt_path = artifacts.get("checkpoint_path")
    if isinstance(ckpt_path, str) and os.path.isfile(ckpt_path):
        if hashes.get("checkpoint_hash") != _file_sha256(ckpt_path):
            _problems_add(
                problems,
                "hash drift vs real file: matrix checkpoint_hash "
                f"{hashes.get('checkpoint_hash')!r} != SHA256 of "
                f"{ckpt_path}",
            )
        try:
            import torch

            ckpt = torch.load(ckpt_path, map_location="cpu",
                              weights_only=False)
            from diffusion_policy_3d.rl.dppo_uav.checkpoint import (
                config_hash_of,
            )

            layer = (ckpt.get("exact_resolved_config") or {}).get(
                "config_layer")
            if not isinstance(layer, dict):
                _problems_add(
                    problems,
                    f"matrix artifacts: cannot recompute config_hash from "
                    f"{ckpt_path} — exact_resolved_config.config_layer "
                    "missing",
                )
            else:
                recomputed = config_hash_of(layer)
                if hashes.get("config_hash") != recomputed:
                    _problems_add(
                        problems,
                        "hash drift vs real file: matrix config_hash "
                        f"{hashes.get('config_hash')!r} != recomputed "
                        f"{recomputed} from the real checkpoint config "
                        "layer",
                    )
        except Exception as e:  # noqa: BLE001 — 任何读取/重算失败都拒绝
            _problems_add(
                problems,
                f"matrix artifacts: cannot recompute config_hash from "
                f"{ckpt_path}: {e}",
            )
    bank_path = artifacts.get("paired_bank_path")
    if isinstance(bank_path, str) and os.path.isfile(bank_path):
        if hashes.get("bank_hash") != _file_sha256(bank_path):
            _problems_add(
                problems,
                "hash drift vs real file: matrix bank_hash "
                f"{hashes.get('bank_hash')!r} != SHA256 of {bank_path}",
            )
    return problems


def rebuild_canary_series(metrics_path: str) -> Dict[str, Any]:
    """从原始 metrics.jsonl 重建 canary 序列（Gate D reopening）。

    返回 {"per_update": {str(全局 update): 五项}, "resumes": int,
    "resume_points": [int], "last_kind": str}。
    - 全局索引 = 文件行序（写入即全局序）；cumulative_env_steps 必须
      严格递增，乱序/跳变文件 = 证据异常 → ValueError；
    - resume 检测 = batch 字段下降（resume 后 batch 从 1 重新计数，
      train_dppo_uav_official.py persist 注释）；resume point = 下降
      处的前一个全局 update；
    - per-update 五项用真实 metrics 键名（value_loss_mean）。
    """
    with open(metrics_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    if not rows:
        raise ValueError(f"empty metrics file: {metrics_path}")
    prev_steps: Optional[int] = None
    prev_batch: Optional[int] = None
    resumes = 0
    points: List[int] = []
    per_update: Dict[str, Dict[str, Any]] = {}
    for pos, r in enumerate(rows, 1):
        steps = r.get("cumulative_env_steps")
        if not isinstance(steps, int) or (prev_steps is not None
                                          and steps <= prev_steps):
            raise ValueError(
                f"metrics {metrics_path}: cumulative_env_steps must be "
                f"strictly increasing (global order), got {steps!r} "
                f"after {prev_steps!r}"
            )
        prev_steps = steps
        b = r.get("batch")
        if prev_batch is not None and isinstance(b, int) and b < prev_batch:
            resumes += 1
            points.append(pos - 1)
        prev_batch = b if isinstance(b, int) else prev_batch
        m = r.get("metrics") or {}
        per_update[str(pos)] = {k: m[k] for k in CANARY_REBUILT_METRICS
                                if k in m}
    return {
        "per_update": per_update,
        "resumes": resumes,
        "resume_points": points,
        "last_kind": rows[-1].get("checkpoint_kind"),
    }


def verify_canary_artifacts(evidence: Dict[str, Any],
                            git_head: str, tree_clean: bool) -> List[str]:
    """R6.4 canary 真实数据重建层 → problems（[] = PASS）。

    结构层信任手工复制的 claims/control；本层从原始数据重建并逐项
    比对，不一致即拒绝：
    - claims.head 必须 == 真实 git HEAD；tree 必须真实 clean；
    - 每个 claimed seed 的 metrics.jsonl
      （evidence['artifacts']['metrics'][seed]）重建完整五项序列 /
      resume 次数 / resume points / release 标记；
    - claims.resume_count / updates_per_seed / resume_points 与重建一致；
    - interrupted 的 resume_continuation 与 control 声称值必须逐项
      等于重建序列（手工复制错一位即拒绝）；
    - 最后一 update 必须 checkpoint_kind == release（1 release/seed）。
    """
    problems: List[str] = []
    claims = evidence.get("claims")
    if not isinstance(claims, dict):
        return problems
    if claims.get("head") != git_head:
        _problems_add(
            problems,
            "commit drift vs real git: claims.head "
            f"{claims.get('head')!r} != git HEAD {git_head!r}",
        )
    if claims.get("tree") != "clean" or not tree_clean:
        _problems_add(
            problems,
            "tree drift vs real git: claims.tree "
            f"{claims.get('tree')!r} but working tree is "
            f"{'clean' if tree_clean else 'dirty'} — canary only valid "
            "on the final-commit tree (§11.8.1)",
        )
    artifacts = evidence.get("artifacts")
    if not isinstance(artifacts, dict) or not isinstance(
            artifacts.get("metrics"), dict):
        _problems_add(
            problems,
            "canary artifacts: missing artifacts.metrics block "
            "{seed: metrics.jsonl path}",
        )
        return problems
    metrics_map = artifacts["metrics"]
    seeds = claims.get("seeds")
    if not isinstance(seeds, dict):
        return problems
    for seed, kind in seeds.items():
        path = metrics_map.get(seed)
        if not isinstance(path, str) or not os.path.isfile(path):
            _problems_add(
                problems,
                f"canary artifacts: seed {seed} metrics path {path!r} "
                "does not exist",
            )
            continue
        try:
            rebuilt = rebuild_canary_series(path)
        except ValueError as e:
            _problems_add(problems, f"canary artifacts: {e}")
            continue
        per_update = rebuilt["per_update"]
        expected_updates = claims.get("updates_per_seed")
        if len(per_update) != expected_updates:
            _problems_add(
                problems,
                f"canary artifacts: seed {seed} real update count "
                f"{len(per_update)} != claims {expected_updates}",
            )
        if rebuilt["last_kind"] != "release":
            _problems_add(
                problems,
                f"canary artifacts: seed {seed} last checkpoint_kind "
                f"{rebuilt['last_kind']!r} != 'release' — 1 release/seed "
                "must be provable from the raw metrics",
            )
        seg = (evidence.get("interrupted") or {}).get(seed) or {}
        if kind == "interrupted":
            resume_counts = claims.get("resume_count") or {}
            claimed_resumes = resume_counts.get(seed)
            if claimed_resumes != rebuilt["resumes"]:
                _problems_add(
                    problems,
                    f"canary artifacts: seed {seed} claims resume_count "
                    f"{claimed_resumes!r} but raw metrics show "
                    f"{rebuilt['resumes']} batch reset(s) — resume must "
                    "be rebuilt from the metrics, not asserted by hand",
                )
            claimed_points = seg.get("resume_points")
            if claimed_points != rebuilt["resume_points"]:
                _problems_add(
                    problems,
                    f"canary artifacts: seed {seed} claims resume_points "
                    f"{claimed_points!r} but raw metrics place resets at "
                    f"{rebuilt['resume_points']}",
                )
            cont = seg.get("resume_continuation")
            if cont != per_update:
                _problems_add(
                    problems,
                    f"canary artifacts: seed {seed} resume_continuation "
                    "does not match the raw metrics rebuild — hand-copied "
                    "control dicts are not trusted",
                )
        elif rebuilt["resumes"] != 0:
            _problems_add(
                problems,
                f"canary artifacts: seed {seed} claims uninterrupted but "
                f"raw metrics show {rebuilt['resumes']} batch reset(s)",
            )
        ctrl = (evidence.get("control") or {}).get(seed)
        if ctrl != per_update:
            _problems_add(
                problems,
                f"canary artifacts: seed {seed} control does not match "
                "the raw metrics rebuild — hand-copied control dicts are "
                "not trusted",
            )
    return problems


def _continuation_tolerance_diffs(cont: Dict[str, Any],
                                  ctrl: Dict[str, Any],
                                  tolerances: Any,
                                  seed: str) -> List[str]:
    """continuation vs control 逐项容差超限行（报告性，不 gate）。

    2026-08-21 裁决 A（语义收敛）：Guide §11.8.2 通过条件为"R6.4 只判
    engine composition：exact [1600]×5、one release、resume≤2、
    standard JSON、strict identity、五 index、LoRA update/base zero、
    GAE/oracle、paired provenance"；容差段（"若严格 bitwise 不能跨
    GPU kernel 保证，必须在运行前列出 tensor/metric 的逐项容差，
    不能运行后放宽"）定位为 bitwise 不可得时的 fallback 证据，
    不属于"只判"清单 → 超限项作为 REPORT 附件呈现，不进入 gate。
    tolerance 数值本身不改（预冻结值保持）。结构完整性检查
    （resume_points/索引集/truncated）仍在 verify 中 gate。
    """
    if not isinstance(tolerances, dict):
        return [f"interrupted canary: seed {seed} per-update tolerances "
                "missing from evidence (frozen values required by "
                "§11.8.2)"]
    per_update = tolerances.get("per_update")
    if not isinstance(per_update, dict) or not per_update:
        return [f"interrupted canary: seed {seed} tolerances missing "
                "per_update block"]
    diffs: List[str] = []
    for key, tol in per_update.items():
        if not isinstance(tol, (int, float)) or tol < 0:
            diffs.append(f"tolerance {key} must be a non-negative number")
            continue
        for idx, val in cont.items():
            cval = ctrl.get(idx)
            if not isinstance(cval, dict) or key not in cval:
                continue
            c_metric = cval[key]
            if not isinstance(c_metric, (int, float)):
                continue
            cont_val = val.get(key)
            if not isinstance(cont_val, (int, float)):
                continue
            if abs(cont_val - c_metric) > tol:
                diffs.append(
                    f"interrupted canary: seed {seed} update {idx} "
                    f"{key} continuation {cont_val} vs control "
                    f"{c_metric} exceeds pre-frozen tolerance {tol} "
                    "(REPORT, not gate — §11.8.2 '只判 engine composition')")
    return diffs


def tolerance_diff_report(evidence: Dict[str, Any]) -> List[str]:
    """收集全部容差超限报告行（裁决 A：报告性附件，不 gate）。"""
    lines: List[str] = []
    if not isinstance(evidence, dict):
        return lines
    seeds = (evidence.get("claims") or {}).get("seeds")
    if not isinstance(seeds, dict):
        return lines
    interrupted = evidence.get("interrupted")
    control = evidence.get("control")
    tolerances = evidence.get("tolerances")
    for seed, kind in seeds.items():
        if kind != "interrupted":
            continue
        if not isinstance(interrupted, dict) or seed not in interrupted:
            continue
        seg = interrupted[seed]
        cont = seg.get("resume_continuation")
        ctrl = (control or {}).get(seed)
        if isinstance(cont, dict) and isinstance(ctrl, dict):
            lines.extend(
                _continuation_tolerance_diffs(cont, ctrl, tolerances, seed))
    return lines


# ------------------------------------------------------------- prereg

PREREG_REQUIRED_FIELDS = (
    "schema_version", "budget", "judgements", "thresholds",
    "stop_rules", "resolved_config", "tolerances",
)


def freeze_g8_prereg(prereg: Dict[str, Any],
                     output_path: Optional[str]) -> str:
    """G8 预注册冻结（§8.2/§11.8.4）。

    必需字段：schema_version / budget / judgements / thresholds /
    stop_rules / resolved_config / tolerances。缺字段 → ValueError
    （pre-registration 不完整即拒绝，不是警告）。

    冻结 = 必需字段 + 确定性 content_hash（SHA256 over canonical JSON，
    不含 hash 自身）→ 返回 JSON 字符串；output_path 给出时写文件。
    训练后对冻结文件的任何修改都会使 content_hash 校验失败——
    判据在训练后不可修改（防"训练后解释"）。
    """
    if not isinstance(prereg, dict):
        raise ValueError("prereg must be an object")
    missing = [f for f in PREREG_REQUIRED_FIELDS if f not in prereg]
    if missing:
        raise ValueError(
            "G8 prereg incomplete: missing required fields "
            f"{missing} — pre-registration must be complete before "
            "training (§8.2/§11.8.4)"
        )
    if "content_hash" in prereg:
        raise ValueError(
            "content_hash is derived by freeze_g8_prereg, not supplied"
        )

    canonical = json.dumps(
        prereg, sort_keys=True, ensure_ascii=False, indent=2,
    )
    content_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    frozen = dict(prereg)
    frozen["content_hash"] = content_hash
    frozen["frozen"] = True
    blob = json.dumps(frozen, sort_keys=True, ensure_ascii=False, indent=2)
    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(blob + "\n")
    return blob


def verify_frozen_prereg(path: str) -> List[str]:
    """已冻结 prereg 文件的完整性校验（content_hash 重算对照）。"""
    problems: List[str] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        return [f"cannot read frozen prereg {path}: {e}"]
    if not data.get("frozen"):
        problems.append("prereg file not marked frozen")
    recorded = data.get("content_hash")
    canonical = json.dumps(
        {k: v for k, v in data.items()
         if k not in ("content_hash", "frozen")},
        sort_keys=True, ensure_ascii=False, indent=2,
    )
    recomputed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if recorded != recomputed:
        problems.append(
            "frozen prereg tampered: recorded content_hash "
            f"{recorded} != recomputed {recomputed} — judgements/"
            "thresholds/budget changed after freeze"
        )
    return problems


# Gate D reopening (2026-08-22): G8 prereg 运行时强制合同 — 训练入口
# 在加载 policy 前逐字段核验 CLI 与冻结 prereg。每条映射：
# (prereg_section, prereg_field) = 冻结侧字段；CLI 侧字段名见映射键。
# "seeds" 是成员语义（一次运行一个 base_seed，必须 ∈ 冻结种子集）。
# 路径字段冻结绝对路径（入口 _root_abs 解析后比较），比较字符串相等。
PREREG_BINDING_FIELDS = {
    "stage": ("resolved_config", "stage"),
    "base_seed": ("budget", "seeds"),          # membership
    "lr_actor": ("resolved_config", "actor_lr"),
    "lr_value": ("resolved_config", "critic_lr"),
    "update_epochs": ("resolved_config", "update_epochs"),
    "lr_scheduler": ("resolved_config", "lr_scheduler"),
    "reward_scale": ("resolved_config", "reward_scale"),
    "lora_rank": ("resolved_config", "lora_rank"),
    "lora_alpha": ("resolved_config", "lora_alpha"),
    "batch_size": ("resolved_config", "batch_size"),
    "max_ep_len": ("resolved_config", "max_ep_len"),
    "n_episodes": ("resolved_config", "n_episodes"),
    "n_envs": ("budget", "n_envs"),
    "target_transitions": ("budget", "transitions_per_seed"),
    "rollout_transitions_per_update": ("budget", "transitions_per_update"),
    "admission_manifest": ("resolved_config", "manifest_path"),
    "bc_ckpt": ("resolved_config", "bc_checkpoint_path"),
    "worlds_train": ("resolved_config", "worlds_train_path"),
    "paired_bank": ("resolved_config", "paired_bank_path"),
    "offline_density": ("resolved_config", "offline_density_path"),
}


def verify_prereg_runtime_binding(
    prereg_path: str, cli: Dict[str, Any]
) -> List[str]:
    """运行时强制合同：冻结 prereg ↔ CLI 逐字段核验（[] = 一致）。

    Gate D reopening (2026-08-22): the frozen prereg is not advisory —
    a formal/G8 launch must present it and every CLI field it pins must
    match, or the entry exits non-zero BEFORE loading the policy. The
    prereg must itself pass verify_frozen_prereg (tampered freeze =
    rejected). Derived contract: transitions_per_seed must be an exact
    multiple of transitions_per_update, and their quotient must equal
    the frozen updates_per_seed (the budget, not the stage fuse, is the
    update count).
    """
    problems = verify_frozen_prereg(prereg_path)
    if problems:
        return problems
    with open(prereg_path, "r", encoding="utf-8") as f:
        prereg = json.load(f)
    budget = prereg.get("budget") or {}
    resolved = prereg.get("resolved_config") or {}

    for cli_key, (section, field) in PREREG_BINDING_FIELDS.items():
        frozen = (budget if section == "budget" else resolved).get(field)
        actual = cli.get(cli_key)
        if cli_key == "base_seed":
            # membership: one run = one seed from the frozen seed set
            seeds = budget.get("seeds")
            if not isinstance(seeds, list) or actual not in seeds:
                _problems_add(
                    problems,
                    f"prereg binding: base_seed {actual!r} not in frozen "
                    f"seeds {seeds!r} — a G8 run must use a registered "
                    "seed",
                )
            continue
        if frozen != actual:
            _problems_add(
                problems,
                f"prereg binding: {cli_key} {actual!r} != frozen "
                f"{section}.{field} {frozen!r} — CLI and the frozen "
                "pre-registration disagree; refusing to launch",
            )

    # derived: the transition budget implies the update count
    tps = budget.get("transitions_per_seed")
    tpu = budget.get("transitions_per_update")
    ups = budget.get("updates_per_seed")
    if isinstance(tps, int) and isinstance(tpu, int) and tpu > 0:
        if tps % tpu != 0:
            _problems_add(
                problems,
                f"prereg binding: transitions_per_seed {tps} is not a "
                f"multiple of transitions_per_update {tpu} — the update "
                "count is not an integer",
            )
        elif ups != tps // tpu:
            _problems_add(
                problems,
                f"prereg binding: updates_per_seed {ups!r} != "
                f"{tps}//{tpu} = {tps // tpu} — the budget's update "
                "count and the transition budget disagree",
            )
    return problems


# ------------------------------------------------------------------ CLI

def _main(argv: List[str]) -> int:
    if len(argv) < 2:
        print(
            "usage: r6_admission_report.py {matrix|evidence|freeze-g8} "
            "<json path> [--current-head <hex>] [--out <path>] "
            "(global admission is signed ONLY by scripts/"
            "r7_final_admission.py — R7-7 Step 6)",
            file=sys.stderr,
        )
        return 2
    stage = argv[1]
    # R7-7 Step 6: 局部 verifier 永远不能签全局 engine/admission 结论。
    # 请求 global → hard-fail 并指向唯一 canonical signer。
    if stage in ("global", "admission"):
        print(
            "r6_admission_report.py cannot sign a global engine/admission "
            "conclusion — use scripts/r7_final_admission.py (the only "
            "canonical fail-closed aggregator, R7-7 Step 6)",
            file=sys.stderr,
        )
        return 1
    # R7-7 Step 6: 本模块所有验证都是 LOCAL slice，输出 schema 必须
    # 声明 scope —— 局部验证通过 ≠ 训练准入。
    LOCAL_SCOPE = "LOCAL_ONLY_NOT_TRAINING_ADMISSION"
    repo_root = os.getcwd()
    if stage == "matrix":
        with open(argv[2], "r", encoding="utf-8") as f:
            matrix = json.load(f)
        problems = verify_run_matrix(matrix)
        # Gate D reopening: hash 与真实 git/文件比对（CLI 层查真实
        # git，不信任手工复制的 claims）
        problems += verify_matrix_artifacts(
            matrix, git_rev_parse(repo_root), git_tree_clean(repo_root))
    elif stage == "evidence":
        with open(argv[2], "r", encoding="utf-8") as f:
            evidence = json.load(f)
        head = argv[argv.index("--current-head") + 1] \
            if "--current-head" in argv else ""
        problems = verify_canary_evidence(evidence, head)
        # 2026-08-21 裁决 A：容差超限为报告性附件（不 gate）
        for line in tolerance_diff_report(evidence):
            print(f"REPORT: {line}")
        # Gate D reopening: claims/continuation/control 与真实
        # metrics.jsonl 重建比对（git 状态由 CLI 层真实探测）
        problems += verify_canary_artifacts(
            evidence, git_rev_parse(repo_root), git_tree_clean(repo_root))
    elif stage == "freeze-g8":
        with open(argv[2], "r", encoding="utf-8") as f:
            prereg = json.load(f)
        out = argv[argv.index("--out") + 1] if "--out" in argv else None
        try:
            freeze_g8_prereg(prereg, output_path=out)
        except ValueError as e:
            print(f"freeze FAIL: {e}", file=sys.stderr)
            return 1
        print(
            f"freeze OK: {out} — scope={LOCAL_SCOPE}",
        )
        return 0
    else:
        print(f"unknown stage {stage!r}", file=sys.stderr)
        return 2
    if problems:
        for p in problems:
            print(f"VERIFY FAIL: {p}", file=sys.stderr)
        return 1
    print(f"VERIFY PASS ({stage}) — scope={LOCAL_SCOPE}")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
