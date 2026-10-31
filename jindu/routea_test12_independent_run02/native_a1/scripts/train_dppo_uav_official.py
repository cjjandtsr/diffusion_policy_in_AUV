#!/usr/bin/env python
"""Canonical formal training entry (A-Official-Core, R3.3).

The ONLY entry that may run formal RL training — every legacy
``scripts/train_ddpm_ppo*.py`` rejects ``--mode formal`` at its own gate
(HISTORICAL_INVALID_FOR_FORMAL_TRAINING), and this script is the unique
canonical path. The preflight (git clean/commit, vendor SHA256SUMS,
BC/manifest/contract hashes, projection=False, train/eval split, formal
numerical mode) runs BEFORE any env/optimizer/policy construction — a
single missing check exits non-zero.

Stages (stage -> (mode, max_updates)):
  no_update   (no_update, 1)    one rollout batch, no PPO update
  one_update  (one_update, 1)   one rollout batch + one PPO update
  g7_micro    (one_update, 30)  G7 micro-train (design §5.9: 30-60
                                updates; 30 pre-registered)
  micro_train (one_update, 10)  short curriculum
  pilot       (one_update, 100) pilot training (G8-scale)
  formal      (one_update, 1000) full formal training

Consecutive batches chain via ``resume_from`` (each batch resumes the
previous batch's checkpoint), so checkpoint counters (outer_iteration /
ppo_update / env_steps / episodes) are cumulative and the episode seed
sequence continues (SEED-RESUME) instead of restarting per batch.

Usage:
  python scripts/train_dppo_uav_official.py --stage formal \
      --output_dir data/outputs/official --device cuda:0
"""

import argparse
import datetime
import json
import os
import random
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from diffusion_policy_3d.policy.lora_adapters import insert_lora
from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.critic import PolicyObsEncoder, ValueCritic
from diffusion_policy_3d.rl.dppo_uav.entrypoint_guard import (
    validate_official_environment,
)
from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
    DEFAULT_ROLE_TARGETS,
    EpisodeTaskSampler,
    manifest_record_key,
)
from diffusion_policy_3d.rl.dppo_uav.evaluator import read_paired_bank_file
from diffusion_policy_3d.rl.dppo_uav.capability_metrics import OfflineDensity
from diffusion_policy_3d.rl.dppo_uav.integration import (
    build_adapter,
    build_routea_eval_pool,
    build_task_specs,
    entry_metadata_hash,
    manifest_train_entries,
)
from diffusion_policy_3d.rl.dppo_uav.optimizer_contract import (
    assert_optimizer_parameter_partition,
)
from diffusion_policy_3d.rl.dppo_uav.policy_loader import load_ddpm_policy
from diffusion_policy_3d.rl.dppo_uav.reward_norm import RewardScaleConfig
from diffusion_policy_3d.rl.dppo_uav.train_loop import (
    EpisodeCollectionIncomplete,
    LoopConfig,
    compute_batch_target,
    run_e2e_batch,
)
from diffusion_policy_3d.rl.dppo_uav.training_schedules import (
    LRSchedulerConfig,
)
from diffusion_policy_3d.rl.dppo_uav.trainer import TrainerConfig

DEFAULT_BC_CKPT = (
    "data/outputs/2026.05.17/13.36.20_v8_phase2_sweep_lambda0.5"
    "/checkpoints/latest.ckpt"
)
DEFAULT_MANIFEST = (
    "/home/cjj/AUV/program/reports/routeA_rl_env_v1/calibration"
    "/admission_manifest.json"
)
DEFAULT_WORLDS_TRAIN = (
    "/home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds/train"
)

STAGE_CONFIG: Dict[str, Tuple[str, int]] = {
    "no_update": ("no_update", 1),
    "one_update": ("one_update", 1),
    "g7_micro": ("one_update", 30),
    "micro_train": ("one_update", 10),
    "pilot": ("one_update", 100),
    "formal": ("one_update", 1000),
}


def _root_abs(p: str) -> str:
    """Resolve a possibly-relative path against the project root.

    The preflight and the policy loader resolve against the repo root, so
    the entry does too — a relative path must never silently depend on CWD.
    """
    path = Path(p)
    return str(path if path.is_absolute() else REPO_ROOT / path)


# R6-C (§11.4.1/§11.4.3): the NaN-as-no-data sentinel convention is
# GONE. Every no-data statistic migrates to the OptionalStatistic
# two-state contract (sample_count==0 -> value=None/valid=False) at the
# producer; any NaN/Inf that still reaches the persist gate is a
# programming error. Strict JSON is enforced with allow_nan=False — the
# reader must never special-case NaN literals (F-03).
SCHEMA_VERSION = "routeA_rl_env_v1_r6c"


def _json_safe_metrics(metrics: Dict[str, Any]) -> None:
    """§11.4.3 strict-JSON guard: any non-finite number anywhere in the
    metrics dict hard-fails BEFORE the record is written (allow_nan=False
    equivalence; the writer never emits NaN/Inf literals)."""
    def _check(value: Any, where: str) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                _check(v, f"{where}.{k}" if where else str(k))
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                _check(v, f"{where}[{i}]")
        elif isinstance(value, float) and not np.isfinite(value):
            raise ValueError(f"non-finite metric {where}={value}")

    for key, value in metrics.items():
        _check(value, str(key))


def persist_batch_metrics(
    path: Path, batch_idx: int, res: Dict[str, Any], *,
    cumulative_env_steps: Optional[int] = None,
    world_assets_hash: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Append one batch's §4.15 learning-signal record to the output
    dir's metrics.jsonl (the G7 gate consumes this stream: per-denoise
    KL/clipfrac structured output + action variance per update).
    R6-C (§11.4.3): STRICT JSON — ``allow_nan=False`` + ``sort_keys``;
    non-finite metrics hard-fail BEFORE the record is written and every
    record carries ``schema_version`` (legacy no-schema records are
    historical evidence only, never current-gate input). Returns None
    (no record) when the batch has no metrics (no_update mode)."""
    metrics = res["metrics"]
    # no_update mode returns metrics=None (no PPO update ran) — nothing to
    # record, not an error (review_R4_part1 P1-2: the unconditional call
    # crashed the no_update stage).
    if metrics is None:
        return None
    _json_safe_metrics(metrics)
    record = {
        "schema_version": SCHEMA_VERSION,
        "batch": int(batch_idx),
        "env_steps": res["env_steps"],
        "checkpoint_path": res.get("checkpoint_path"),
        "checkpoint_kind": res.get("checkpoint_kind"),
        "completed_episodes": res["completed_episodes"],
        "voided_episodes": res["voided_episodes"],
        "admission_blocked_episodes": res.get(
            "admission_blocked_episodes", 0
        ),
        "routing": res["routing"],
        "metrics": metrics,
    }
    if cumulative_env_steps is not None:
        # R5.8-A A8/H-6: the CUMULATIVE counter lets every jsonl record
        # be joined to its checkpoint without recomputation — the
        # per-run ``batch`` index restarts at 1 on resume and is only a
        # positional suffix.
        record["cumulative_env_steps"] = int(cumulative_env_steps)
    if world_assets_hash is not None:
        # Gate D reopening (2026-08-22): every record binds the world
        # assets it trained under — the verifier rebuilds this axis from
        # the raw metrics, never from a hand-copied claims dict.
        record["world_assets_hash"] = world_assets_hash
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")
        # R5.8-D (H-4): fsync the record — a crash right after a
        # "completed" batch must not lose the tail silently.
        f.flush()
        os.fsync(f.fileno())
    return record


# R6-B (§11.3.4): 空目录磁盘估计的 G7 实测量级 oracle —— resume
# ≈ 7.5 MB（lightweight，不含 frozen base）、release ≈ 293 MB（全量
# policy）。任何估计来源未知时取保守上界并标注来源。
DISK_ORACLE_RESUME_BYTES = 7_500_000
DISK_ORACLE_RELEASE_BYTES = 293_000_000
# R6-B (§11.3.3): metrics.jsonl + paired eval JSON + gate 报告的
# 保守预留。
DISK_RESERVE_METRICS_EVAL_BYTES = 1 * 2**30


def artifact_audit(output_dir: Path,
                   total_transitions: Optional[int] = None,
                   keep_latest: int = 2,
                   ) -> Dict[str, Any]:
    """§11.3.2: metrics.jsonl ↔ checkpoint 文件的一致性审计。

    hard-fail（RuntimeError）:
      - metrics 指向不存在的 checkpoint 文件（证据断裂）——但只针对
        最新 ``keep_latest`` 条记录（时间序，含 release 自身）对应的
        文件：被 retention 合法轮换的旧记录（§11.3.2 S3，release
        落地后只留下最新 1 个 resume + release）文件缺失不是断裂
        （Gate B R6-DISK-1 REWORK 2026-08-19：旧版把 retention 的
        预期结果当证据断裂，任何 >=4 batch 的合法运行终局必 audit
        FAIL）；
      - 同一 run 多于一个 final release；
      - transition 模式（total_transitions 给定）下没有 final release，
        或 release record 的累计 transition 计数 != run total。
    torn trailing metrics line 记为 problem（crash residue，可报告可
    忽略），但绝不物化为 release。返回 {"release_paths", "problems"}。
    """
    metrics_path = Path(output_dir) / "metrics.jsonl"
    records: List[Dict[str, Any]] = []
    problems: List[str] = []
    if metrics_path.is_file():
        for lineno, line in enumerate(
                metrics_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                problems.append(
                    f"torn/trailing metrics line {lineno} ignored"
                )
    release_paths: List[Path] = []
    # records 按 metrics.jsonl 行序 = 时间序 = 文件 mtime 序（生产路径
    # 每批按序写文件）。retention（S3, keep_latest=2）轮换最旧的
    # 文件 —— 终局时存活 = 最新 keep_latest 条记录对应的文件（含
    # release 自身：release 是最新记录，且被 retention 保护）。因此
    # must_exist = 最新 keep_latest 条记录，而不是最新 keep_latest 个
    # resume —— release 落地后 retention 只留下最新 1 个 resume +
    # release（2026-08-19 修复：先前的 keep_latest 个 resume 窗口在
    # >=4 batch 运行上必 audit FAIL，正是 Gate B R6-DISK-1 核心指控；
    # 用 list 而非 set —— dict 不可哈希）。
    must_exist = records[-keep_latest:]
    for rec in records:
        path = Path(rec["checkpoint_path"]) if rec.get(
            "checkpoint_path") else None
        if rec.get("checkpoint_kind") == "release":
            if path is None or not path.is_file():
                raise RuntimeError(
                    "artifact audit FAIL: metrics points to a missing "
                    f"checkpoint file: {rec.get('checkpoint_path')!r} "
                    "(§11.3.2)"
                )
            if total_transitions is not None:
                counter = int(rec.get("cumulative_env_steps", -1))
                if counter != total_transitions:
                    raise RuntimeError(
                        "artifact audit FAIL: release counter "
                        f"{counter} != run total {total_transitions} (§11.3.2)"
                    )
            release_paths.append(path)
        elif rec in must_exist:
            # 最新 keep_latest 条记录对应的文件必须存在；被轮换的旧
            # 记录缺失是 retention 的预期结果，不算证据断裂。
            if path is None or not path.is_file():
                raise RuntimeError(
                    "artifact audit FAIL: metrics points to a missing "
                    f"checkpoint file: {rec.get('checkpoint_path')!r} "
                    "(§11.3.2 — newest record's file must exist)"
                )
    if len(release_paths) > 1:
        raise RuntimeError(
            f"artifact audit FAIL: {len(release_paths)} final release "
            "checkpoints in one run — exactly one allowed (§11.3.2)"
        )
    if total_transitions is not None and not release_paths:
        raise RuntimeError(
            "artifact audit FAIL: transition run finished without a "
            "final release checkpoint (§11.3.2/§11.3.4)"
        )
    return {"release_paths": release_paths, "problems": problems}


def _disk_preflight(output_dir: Path, max_updates: int,
                    reserve_bytes: int = 20 * 2**30) -> Dict[str, Any]:
    """S4 + R6-B (§11.3.3): hard-fail BEFORE rollout when the projected
    footprint + reserve exceeds the free space. 取代旧的
    per_ckpt×(max_updates+2) 单文件模型 —— 基于运行计划的显式估计：
    resume_keep=2 + release=1 + tmp 副本 + metrics/eval 预留 + safety
    reserve。估计来源优先级：同 schema 历史实测最大值（metrics.jsonl
    分类 resume/release）> 空目录 G7 实测量级 oracle（resume≈7.5MB、
    release≈293MB）。返回明细 dict（来源未知时保守上界 + 来源标注）。"""
    import shutil

    # The output dir does not exist on a first run — disk_usage on a
    # missing path raises FileNotFoundError. Walk up to the nearest
    # existing ancestor (the volume holding the future output dir).
    base = Path(output_dir)
    while not base.exists():
        base = base.parent
    free = shutil.disk_usage(base).free

    resume_bytes = release_bytes = 0
    source = "oracle"
    out = Path(output_dir)
    if out.is_dir():
        protected = (
            _release_paths_from_metrics(out)
            if (out / "metrics.jsonl").is_file() else set()
        )
        for p in out.glob("dppo_uav_*.pt"):
            size = p.stat().st_size
            if p in protected:
                release_bytes = max(release_bytes, size)
            else:
                resume_bytes = max(resume_bytes, size)
        if resume_bytes or release_bytes:
            source = ("history" if (out / "metrics.jsonl").is_file()
                      else "history-unclassified")
    if resume_bytes == 0:
        resume_bytes = DISK_ORACLE_RESUME_BYTES
    if release_bytes == 0:
        release_bytes = DISK_ORACLE_RELEASE_BYTES

    resume_keep_count = 2   # §11.3.2: committed resume 最多 2 个
    release_count = 1       # §11.3.2: final release 恰好 1 个
    temporary_publish_bytes = resume_bytes  # 原子发布的一份 tmp 副本
    metrics_and_eval_reserve_bytes = DISK_RESERVE_METRICS_EVAL_BYTES
    safety_reserve_bytes = reserve_bytes    # 默认 20 GiB
    projected = (resume_keep_count * resume_bytes
                 + release_count * release_bytes
                 + temporary_publish_bytes
                 + metrics_and_eval_reserve_bytes
                 + safety_reserve_bytes)
    detail = {
        "max_updates": max_updates,
        "resume_keep_count": resume_keep_count,
        "release_count": release_count,
        "resume_bytes": resume_bytes,
        "release_bytes": release_bytes,
        "temporary_publish_bytes": temporary_publish_bytes,
        "metrics_and_eval_reserve_bytes": metrics_and_eval_reserve_bytes,
        "safety_reserve_bytes": safety_reserve_bytes,
        "projected_bytes": projected,
        "free_bytes": free,
        "source": source,
    }
    if projected > free:
        raise RuntimeError(
            "disk preflight FAIL (R6-B §11.3.3): projected "
            f"{projected / 2**30:.1f} GiB > free {free / 2**30:.1f} GiB "
            f"— refusing to start. Detail: {detail}"
        )
    return detail


def _validate_batch_publish(res: Dict[str, Any], *,
                            cumulative_transitions: int,
                            total_transitions: Optional[int]) -> None:
    """§11.3.1.2: 每批发布契约（入口验证）。

    - 批次必须物化可验证的 checkpoint 文件；
    - transition 模式（total 给定）下 release ⟺ 累计计数 == total ——
      提前 release 是 F-01 复发，必须 loud；
    - episode 模式的 release 由调用方 update-limit fuse 裁定（此处
      不校验 counter，但 kind 必须是合法值）。
    """
    ckpt = res.get("checkpoint_path")
    if not ckpt or not Path(ckpt).is_file():
        raise RuntimeError(
            f"batch publish FAIL: no materialized checkpoint "
            f"({ckpt!r}) — every batch must publish (§11.3.1.2)"
        )
    kind = res.get("checkpoint_kind")
    if kind == "release":
        if (total_transitions is not None
                and cumulative_transitions != total_transitions):
            raise RuntimeError(
                f"batch publish FAIL: checkpoint_kind='release' but "
                f"cumulative {cumulative_transitions} != total "
                f"{total_transitions} — release is post-hoc finality "
                "(§11.3.1.2/R6-A)"
            )
    elif kind != "resume":
        raise RuntimeError(
            f"batch publish FAIL: unknown checkpoint_kind {kind!r} "
            "(§11.3.1.2)"
        )


def _write_incomplete_budget_marker(output_dir: Path, *, planned: int,
                                    actual: int, cumulative: int,
                                    total: int) -> None:
    """§11.2.4 保险丝：partial batch（actual != planned 且 budget 未满）
    或 max_updates 耗尽而 budget 未完成 → 写 INCOMPLETE_BUDGET marker
    文件（不得发布 release、不得打印 TRAINING COMPLETE —— 退出非零由
    调用方保证）。marker 是 batch 完整性证据链的一部分（Gate B
    R6-DISK-1 REWORK CE-2：partial batch 曾静默通过）。"""
    marker = Path(output_dir) / "INCOMPLETE_BUDGET"
    marker.write_text(
        json.dumps({
            "planned_batch_transitions": planned,
            "actual_batch_transitions": actual,
            "cumulative_transitions": cumulative,
            "total_transitions": total,
            "message": "partial batch: actual != planned target and/or "
                       "budget not completed — run must exit non-zero "
                       "without a release (§11.2.4)",
        }, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _verify_batch_target(output_dir: Path, batch_idx: int, max_updates: int,
                         *, actual: int, planned: Optional[int],
                         cumulative: int, total: Optional[int]) -> None:
    """§11.3.1 step 2 入口验证：transition 模式（total 给定）下实际收集
    != 本批派生目标 且 budget 未满 = admission/sampler 用尽的 partial
    batch —— loud 警告并落 INCOMPLETE_BUDGET marker（§11.2.4 保险丝
    语义：partial 允许继续，由后续 batch 重算目标补齐；若最终仍未完成，
    预算校验分支退出非零、不发 release）。完整 batch（actual ==
    planned）、budget 已满的末批（派生目标超出剩余额度）与 episode 模式
    均为 no-op（Gate B R6-DISK-1 REWORK CE-2：partial batch 曾静默
    通过）。"""
    if total is None:
        return
    if actual != planned and cumulative < total:
        _write_incomplete_budget_marker(
            output_dir, planned=planned, actual=actual,
            cumulative=cumulative, total=total,
        )
        print(
            f"WARNING: batch {batch_idx}/{max_updates} collected "
            f"{actual} != planned {planned} transitions and the budget "
            f"({cumulative}/{total}) is not complete — partial batch "
            "(§11.2.4): continuing; the next batch recomputes the "
            "target, a final shortfall exits non-zero without a release "
            f"(INCOMPLETE_BUDGET: {output_dir / 'INCOMPLETE_BUDGET'})",
            file=sys.stderr,
        )


def _retain_checkpoints(output_dir: Path, keep_latest: int = 2) -> int:
    """S3: rotating retention — keep the newest ``keep_latest`` resume
    checkpoints; every checkpoint whose metrics.jsonl record says
    checkpoint_kind=release is PROTECTED (never rotated). The sidecar is
    the source of truth across process restarts. Returns the number
    removed."""
    protected = _release_paths_from_metrics(output_dir)
    ckpts = sorted(output_dir.glob("dppo_uav_*.pt"),
                   key=lambda x: x.stat().st_mtime)
    removed = 0
    for old in ckpts[:-keep_latest]:
        if old in protected:
            continue
        old.unlink()
        removed += 1
    return removed


def _release_paths_from_metrics(output_dir: Path) -> set:
    """The release-checkpoint set from metrics.jsonl records — a
    resumed process protects previous releases without re-loading the
    (potentially 300MB) checkpoint files."""
    metrics_path = output_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        return set()
    protected = set()
    for line in metrics_path.read_text(
            encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            # R5.8-D (H-4): a torn trailing line from a crash — the
            # writer is fsync'd, but tolerate any legacy torn file.
            continue
        if rec.get("checkpoint_kind") == "release":
            ckpt = rec.get("checkpoint_path")
            if ckpt:
                protected.add(Path(ckpt))
    return protected


def build_runtime(facts: Dict[str, str], device: str) -> Dict[str, str]:
    """The runtime dict run_e2e_batch records into checkpoint/artifact."""
    return {
        "device": device,
        # Must match UAVVectorEnvAdapter's reward_version default — the
        # checkpoint binds the reward the env actually computed.
        "reward_version": "v16",
        "manifest_hash": facts["manifest_hash"],
        "project_git_commit": facts["project_git_commit"],
        "project_tree_hash": facts["project_tree_hash"],
        "source_file_hashes": facts["source_file_hashes"],
        "bc_checkpoint_hash": facts["bc_checkpoint_hash"],
        # Gate D reopening (2026-08-22): the external world assets are a
        # training-input identity axis (scene + obstacle content hash).
        "world_assets_hash": facts["world_assets_hash"],
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description="Canonical formal RL training (A-Official-Core)."
    )
    p.add_argument("--stage", required=True, choices=sorted(STAGE_CONFIG))
    p.add_argument("--bc_ckpt", default=DEFAULT_BC_CKPT)
    p.add_argument("--admission_manifest", default=DEFAULT_MANIFEST)
    p.add_argument("--worlds_train", default=DEFAULT_WORLDS_TRAIN)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max_ep_len", type=int, default=100)
    p.add_argument("--n_episodes", type=int, default=32)
    # 1 = the audited production path (G3/G5/G6 evidence ran single-slot).
    p.add_argument("--n_envs", type=int, default=1)
    p.add_argument("--eval_n_episodes", type=int, default=0)
    p.add_argument("--paired_bank", type=str, default=None,
        help="frozen paired bank JSON — REQUIRED when "
        "--eval_n_episodes>0; the eval replays the bank's entries exactly",
    )
    p.add_argument("--offline_density", type=str, default=None,
        help="frozen offline visit-count JSON (schema_version 1) — "
        "REQUIRED when --eval_n_episodes>0; capability metrics cannot "
        "be computed without it",
    )
    p.add_argument("--prereg", type=str, default=None,
        help="frozen G8 pre-registration JSON (schema_version g8-prereg-"
        "v2) — REQUIRED for --stage formal; every pinned CLI field is "
        "verified field-by-field BEFORE the policy loads and any "
        "mismatch exits non-zero (Gate D reopening 2026-08-22)",
    )
    p.add_argument("--final_admission", type=str, default=None,
        help="final admission decision JSON written by scripts/"
        "r7_final_admission.py (schema_version r7-final-admission-v1) — "
        "REQUIRED for --stage formal; the ONLY global engine_correct/"
        "training_admitted_for_g8 authority (R7-7 Step 7); the preflight "
        "verifies engine_correct/training_admitted_for_g8/content hash/"
        "final identity/artifact hashes BEFORE the policy loads",
    )
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr_actor", type=float, default=3e-4)
    p.add_argument("--lr_value", type=float, default=1e-3)
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=8)
    p.add_argument("--base_seed", type=int, default=1)
    p.add_argument("--target_transitions", type=int, default=None,
        help="R5.3 (review F3): fixed environment-transition budget — "
        "when set, the run stops once the CUMULATIVE transition count "
        "reaches it (updates are observation counters); active mid-"
        "episode slots continue across updates via the checkpoint",
    )
    p.add_argument("--update_epochs", type=int, default=1,
        help="PPO update epochs per batch (official default 1)")
    p.add_argument("--target_kl", type=float, default=None,
        help="official target-KL early stop (None = off)")
    p.add_argument("--max_grad_norm", type=float, default=None,
        help="actor gradient clip (official default None)")
    p.add_argument("--vf_coef", type=float, default=1.0,
        help="value loss coefficient (official default 1.0)")
    p.add_argument("--value_clip", type=float, default=None,
        help="value loss clipping (official default None)")
    p.add_argument("--critic_warmup_updates", type=int, default=0,
        help="R5.8-G: critic-only warmup window (official "
        "n_critic_warmup_itr semantics; 0 = off)")
    p.add_argument("--critic_grad_norm_limit", type=float, default=None,
        help="R5.8-G: hard-fail the update when the critic grad norm "
        "exceeds this (None = off; divergence alarm)")
    p.add_argument("--rollout_transitions_per_update", type=int, default=None,
        help="R5.8-A (P0-1): fixed per-update collection horizon in "
        "transitions — each batch collects min(horizon, total - "
        "committed); the final batch collects the exact remaining count",
    )
    p.add_argument("--resume_from", default=None,
                   help="checkpoint path to resume the first batch from")
    p.add_argument("--min_sampling_denoising_std", type=float, default=0.1)
    p.add_argument("--min_logprob_denoising_std", type=float, default=0.1)
    # R6-F (F-07 → R6-TUNE-1, §11.7): scheduler + reward-scaling
    # branches. Defaults are the none branches (LR constant at the
    # optimizer's configured lr, raw training reward) — the pre-R6-F
    # behavior; every field enters the resolved config hash.
    p.add_argument("--lr_scheduler", default="none",
                   choices=["none", "official_cosine_warmup"],
                   help="official CosineAnnealingWarmupRestarts port "
                        "(vendor cc7234a) or none")
    p.add_argument("--scheduler_first_cycle_steps", type=int, default=10)
    p.add_argument("--scheduler_warmup_steps", type=int, default=0)
    p.add_argument("--scheduler_min_lr", type=float, default=1e-4)
    p.add_argument("--scheduler_max_lr", type=float, default=3e-4)
    p.add_argument("--scheduler_cycle_mult", type=float, default=1.0)
    p.add_argument("--scheduler_gamma", type=float, default=1.0)
    p.add_argument("--reward_scale", default="none",
                   choices=["none", "official_running_return_std"],
                   help="official PPG running-return-std scaler "
                        "(vendor cc7234a) or none")
    p.add_argument("--reward_scale_cliprew", type=float, default=10.0)
    args = p.parse_args()
    args.bc_ckpt = _root_abs(args.bc_ckpt)
    args.admission_manifest = _root_abs(args.admission_manifest)
    args.worlds_train = _root_abs(args.worlds_train)
    args.output_dir = _root_abs(args.output_dir)
    if args.paired_bank:
        args.paired_bank = _root_abs(args.paired_bank)
    if args.offline_density:
        args.offline_density = _root_abs(args.offline_density)
    if args.prereg:
        args.prereg = _root_abs(args.prereg)
    if args.final_admission:
        args.final_admission = _root_abs(args.final_admission)

    mode, max_updates = STAGE_CONFIG[args.stage]
    # Gate D reopening (2026-08-22): formal (G8) launches MUST present a
    # frozen pre-registration — the entry verifies every pinned CLI field
    # against it BEFORE loading the policy; a missing prereg is a CLI
    # usage error (exit 2), never a silently unregistered run.
    # formal (G8) launches MUST present BOTH frozen contracts before any
    # model construction: the G8 pre-registration (Gate D reopening
    # 2026-08-22) and the R7-7 final admission decision (the ONLY
    # engine_correct / training_admitted_for_g8 authority).  A missing
    # one is a CLI usage error (exit 2), never a silently unadmitted
    # run; a config can never hand-write engine_pass and an env var can
    # never bypass the checks (both JSONs are re-verified against the
    # real git tree and artifact files below).
    if args.stage == "formal":
        missing_contracts = []
        if not args.prereg:
            missing_contracts.append(
                "--prereg <frozen JSON> (G8 pre-registration, Gate D "
                "reopening)")
        if not args.final_admission:
            missing_contracts.append(
                "--final_admission <decision.json> (r7_final_admission.py "
                "decision, R7-7 Step 7)")
        if missing_contracts:
            print(
                "CLI error: --stage formal requires "
                + " and ".join(missing_contracts)
                + " — a formal run without these frozen contracts "
                "cannot be audited",
                file=sys.stderr,
            )
            return 2
    # R6-A (F-01, §11.2.2): transition collection requires BOTH the
    # run-level total and the per-update horizon — a target-only budget
    # (the F-01 self-proving state) is a CLI-level contract error, never
    # a single-batch target. Episode mode needs neither.
    if (args.target_transitions is None) != (
            args.rollout_transitions_per_update is None):
        missing = (
            "--rollout_transitions_per_update"
            if args.target_transitions is not None
            else "--target_transitions"
        )
        print(
            "CLI error: transition collection requires BOTH "
            "--target_transitions (run-level total) and "
            "--rollout_transitions_per_update (per-update horizon); "
            f"missing {missing} — target-only budgets are forbidden (R6-A)",
            file=sys.stderr,
        )
        return 2
    # R6-A (G8 hard pin, §11.2.3): transition collection runs n_envs=1 —
    # multi-env cannot guarantee the exact remaining capacity.
    if args.target_transitions is not None and args.n_envs != 1:
        print(
            "CLI error: transition collection is pinned to --n_envs 1 "
            "(G8 hard pin, R6-A) — multi-env cannot guarantee the exact "
            "remaining transition capacity",
            file=sys.stderr,
        )
        return 2
    if args.eval_n_episodes > 0 and mode != "one_update":
        print(
            "CLI error: --eval_n_episodes>0 requires a one_update stage "
            f"({args.stage} runs mode={mode}; paired eval is only "
            "supported in one_update mode)",
            file=sys.stderr,
        )
        return 2
    if args.eval_n_episodes > 0 and not args.paired_bank:
        print(
            "CLI error: --eval_n_episodes>0 requires --paired_bank —"
            " the eval replays the frozen bank's entries exactly and"
            " never re-selects tasks at runtime",
            file=sys.stderr,
        )
        return 2
    if args.eval_n_episodes > 0 and not args.offline_density:
        print(
            "CLI error: --eval_n_episodes>0 requires --offline_density —"
            " capability metrics (support expansion, dense recovery)"
            " cannot be computed without the frozen offline density field",
            file=sys.stderr,
        )
        return 2
    # Bank schema is validated at CLI level (before preflight) — a
    # malformed bank is a usage error, exit 2, not a preflight failure.
    paired_bank = None
    bank_id = ""
    if args.paired_bank:
        try:
            paired_bank = read_paired_bank_file(args.paired_bank)
        except ValueError as e:
            print(f"CLI error: {e}", file=sys.stderr)
            return 2
        # The bank's own bank_id binds the eval artifact to the exact
        # frozen bank (bank v3 provenance) — an old bank without one is
        # a clean CLI failure.
        try:
            with open(args.paired_bank, "r", encoding="utf-8") as f:
                bank_id = json.load(f).get("bank_id", "")
        except (OSError, ValueError) as e:
            print(
                f"CLI error: cannot read bank_id from {args.paired_bank}: "
                f"{e}",
                file=sys.stderr,
            )
            return 2
        if not bank_id:
            print(
                "CLI error: paired bank has no bank_id (bank v3 "
                "provenance required by the eval artifact)",
                file=sys.stderr,
            )
            return 2
    offline_density = None
    # Loaded only when an eval stage will actually use it — a stale or
    # malformed field must not fail a no-eval run (L7).
    if args.eval_n_episodes > 0:
        try:
            offline_density = OfflineDensity.read_offline_density_file(
                args.offline_density
            )
        except ValueError as e:
            print(f"CLI error: {e}", file=sys.stderr)
            return 2

    # Gate D reopening (2026-08-22): the frozen prereg is a RUNTIME
    # contract — field-by-field CLI ↔ prereg verification happens here,
    # BEFORE the preflight and the policy load (a wrong seed/lr/bank/
    # manifest exits non-zero before any model construction).
    if args.prereg:
        from scripts.r6_admission_report import (
            verify_prereg_runtime_binding,
        )

        binding = verify_prereg_runtime_binding(
            args.prereg,
            {
                "stage": args.stage,
                "base_seed": args.base_seed,
                "lr_actor": args.lr_actor,
                "lr_value": args.lr_value,
                "update_epochs": args.update_epochs,
                "lr_scheduler": args.lr_scheduler,
                "reward_scale": args.reward_scale,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "batch_size": args.batch_size,
                "max_ep_len": args.max_ep_len,
                "n_episodes": args.n_episodes,
                "n_envs": args.n_envs,
                "target_transitions": args.target_transitions,
                "rollout_transitions_per_update": (
                    args.rollout_transitions_per_update
                ),
                "admission_manifest": args.admission_manifest,
                "bc_ckpt": args.bc_ckpt,
                "worlds_train": args.worlds_train,
                "paired_bank": args.paired_bank,
                "offline_density": args.offline_density,
            },
        )
        if binding:
            for p in binding:
                print(f"prereg FAIL: {p}", file=sys.stderr)
            return 1

    # R7-7 Step 7: canonical final admission — the ONLY global
    # engine_correct / training_admitted_for_g8 check.  Verified
    # BEFORE any env/optimizer/policy construction: decision content
    # hash, final identity == current clean worktree, every bound
    # artifact still exists and hash-matches.  Any problem exits 1.
    if args.final_admission:
        from diffusion_policy_3d.rl.dppo_uav.admission_aggregate import (
            validate_final_admission_for_training,
        )

        admission_problems = validate_final_admission_for_training(
            args.final_admission, os.getcwd())
        if admission_problems:
            for p in admission_problems:
                print(f"final admission FAIL: {p}", file=sys.stderr)
            return 1

    # Preflight BEFORE any env/optimizer/policy construction — a single
    # missing check exits non-zero (R3.3 Step 3). ValueError covers a
    # malformed admission manifest surfacing from inside the preflight
    # (still exit 1, same clean message — never a raw traceback).
    try:
        facts = validate_official_environment(
            manifest_path=args.admission_manifest,
            bc_checkpoint_path=args.bc_ckpt,
            worlds_train_dir=args.worlds_train,
        )
    except (RuntimeError, ValueError) as e:
        print(f"preflight FAIL: {e}", file=sys.stderr)
        return 1

    torch.manual_seed(args.base_seed)
    np.random.seed(args.base_seed)
    random.seed(args.base_seed)
    torch.cuda.manual_seed_all(args.base_seed)

    contract = RLSystemContract()
    if contract.use_obstacle_projection:
        print("preflight FAIL: contract projection must be False", file=sys.stderr)
        return 1

    policy = load_ddpm_policy(args.bc_ckpt, device=args.device)
    lora_info = insert_lora(
        policy.model, r=args.lora_rank, alpha=args.lora_alpha, verbose=False
    )
    critic = ValueCritic(contract, PolicyObsEncoder(policy)).to(args.device)
    # R7-3 Step 2: the actor optimizer owns exactly the registered LoRA
    # bundle trainable parameters; AdamW is the pre-registered R7 change
    # (Adam weight_decay=0.0 -> AdamW weight_decay=1e-2), recorded
    # automatically in the checkpoint resolved optimizer config
    # (resolved_optimizer_config captures type=AdamW; resume validation
    # then rejects any runtime switch — the change is never applied
    # mid-run).
    lora_bundle = policy.model.dppo_lora_adapters
    actor_params = tuple(lora_bundle.trainable_parameters())
    assert actor_params, "registered LoRA bundle has no trainable parameters"
    actor_optimizer = torch.optim.AdamW(actor_params, lr=args.lr_actor)
    critic_optimizer = torch.optim.AdamW(
        critic.value_head_parameters(), lr=args.lr_value
    )
    # R7-3 Step 2 partition contract — actor holds exactly the LoRA
    # bundle, critic holds only value-head parameters, actor ∩ critic
    # = ∅, base UNet/encoder parameters in neither. This also enforces
    # R6-E §11.6.2 (F-08 → R6-OPT-1): both optimizers hold exactly one
    # param group, verified at creation (and again on resume and inside
    # every ppo_update).
    assert_optimizer_parameter_partition(
        actor_optimizer,
        critic_optimizer,
        lora_params=lora_info["lora_params"],
        base_model=policy.model,
    )

    entries = manifest_train_entries(args.admission_manifest)
    specs, _ = build_task_specs(
        [dict(e, manifest_entry_hash=entry_metadata_hash(e)) for e in entries]
    )
    records = {
        manifest_record_key(s.world_uid, s.role, s.admission_id):
            s.manifest_entry_hash
        for s in specs
    }
    adapter = build_adapter(
        contract,
        worlds_dir=args.worlds_train,
        manifest_path=args.admission_manifest,
        manifest_records=records,
        device=args.device,
        max_episode_steps=args.max_ep_len,
        n_envs=args.n_envs,
    )
    eval_env = None
    if args.eval_n_episodes > 0:
        # Dedicated single-slot eval env — never the training env
        # (run_e2e_batch hard-fails on reuse).
        eval_env = build_adapter(
            contract,
            worlds_dir=args.worlds_train,
            manifest_path=args.admission_manifest,
            manifest_records=records,
            device=args.device,
            max_episode_steps=args.max_ep_len,
            n_envs=1,
            # R5.8-E (G-3): the in-train paired eval must reach its bank
            # worlds. The train pool stays rl_train-only (2-arg call,
            # R7-8 ruling 2026-08-24): the R7-8 eval bank lives on the
            # canonical eval entry (rl_val ∪ rl_test worlds), never on
            # the training run — an eval-only bank world passed here
            # still fails loudly instead of silently evaluating against
            # a phantom world.
            pool=build_routea_eval_pool(
                args.admission_manifest, args.worlds_train),
            # R5.8-E (P1-3): paired eval replays the frozen bank exactly
            # — disturbance OFF on both eval sides (training keeps ON).
            disturb_enable=False,
        )

    _disk_preflight(Path(args.output_dir), max_updates)

    runtime = build_runtime(facts, args.device)
    # review partB P1-1: the BATCH-level target is the REMAINING budget
    # P0-1 fix (2026-08-17): the resumed cumulative transition count
    # MUST be loaded BEFORE the LoopConfig construction — the per-batch
    # remaining target derives from it. (Previously the read happened
    # before the assignment → UnboundLocalError at startup; the load
    # happened after the construction → resumed budgeted runs re-
    # collected the whole budget on top of the old count.)
    prev_ckpt = args.resume_from
    cumulative_transitions = 0
    batch_start = 1
    if args.resume_from:
        # review partB P1-3: a resumed budgeted run must start its
        # cumulative count from the checkpoint's recorded transitions —
        # otherwise it re-collects the whole budget on top of the old one.
        from diffusion_policy_3d.rl.dppo_uav.checkpoint import (
            CheckpointBuilder,
        )

        saved = CheckpointBuilder.load(Path(args.resume_from))
        if args.target_transitions is not None:
            cumulative_transitions = saved.environment_transition_count
        # R7-6 Step 9 (metrics cursor wiring): the resumed process
        # continues GLOBAL batch numbering after the last saved
        # checkpoint (outer_iteration = its batch number) — the metrics
        # stream must not renumber/desync (records 1..K exist already).
        # The cursor passed to run_e2e_batch is batch_idx-1 (records
        # committed before this batch — the save happens before the
        # launcher persists this batch's record).
        batch_start = saved.outer_iteration + 1

    loop_config = LoopConfig(
        mode=mode, n_episodes=args.n_episodes, max_ep_len=args.max_ep_len,
        device=args.device, output_dir=args.output_dir,
        eval_n_episodes=args.eval_n_episodes,
        min_sampling_denoising_std=args.min_sampling_denoising_std,
        min_logprob_denoising_std=args.min_logprob_denoising_std,
        target_transitions=compute_batch_target(
            args.target_transitions, cumulative_transitions,
            args.rollout_transitions_per_update,
        ),
        total_transition_budget=args.target_transitions,
        rollout_transitions_per_update=args.rollout_transitions_per_update,
        # R6-F (§11.7.1): every scheduler/scaler config field enters the
        # resolved config hash (a resume under a different schedule or
        # scaling is refused at the identity gate).
        lr_scheduler=LRSchedulerConfig(
            method=args.lr_scheduler,
            first_cycle_steps=args.scheduler_first_cycle_steps,
            warmup_steps=args.scheduler_warmup_steps,
            min_lr=args.scheduler_min_lr,
            max_lr=args.scheduler_max_lr,
            cycle_mult=args.scheduler_cycle_mult,
            gamma=args.scheduler_gamma,
        ),
        reward_scale=RewardScaleConfig(
            method=args.reward_scale,
            cliprew=args.reward_scale_cliprew,
            # gamma stays the contract single-source discount (0.99) —
            # the scaler hard-fails on any mismatch.
        ),
    )
    trainer_config = TrainerConfig(
        batch_size=args.batch_size, device=args.device,
        update_epochs=args.update_epochs,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        vf_coef=args.vf_coef,
        clip_vloss_coef=args.value_clip,
        n_critic_warmup_updates=args.critic_warmup_updates,
        critic_grad_norm_limit=args.critic_grad_norm_limit,
    )
    sampler = EpisodeTaskSampler(
        specs, DEFAULT_ROLE_TARGETS, base_seed=args.base_seed
    )

    print(
        f"official train: stage={args.stage} mode={mode} "
        f"max_updates={max_updates} device={args.device}\n"
        f"  commit={facts['project_git_commit'][:12]} "
        f"tree={facts['project_tree_hash'][:12]}\n"
        f"  manifest={facts['manifest_hash'][:12]} "
        f"bc={facts['bc_checkpoint_hash'][:12]} "
        f"contract={facts['contract_hash'][:12]}\n"
        f"  worlds={facts['world_assets_hash'][:12]}"
    )
    try:
        for batch_idx in range(batch_start, max_updates + 1):
            if (args.target_transitions is not None
                    and cumulative_transitions
                    >= args.target_transitions):
                # Gate B R6-DISK-1 REWORK (2026-08-19): 原实现在此直接
                # return 0 —— §11.3.2 终局 artifact audit（895 行）在
                # transition 模式从未执行。break fall-through 到统一
                # 收尾（budget 校验 + audit + TRAINING COMPLETE）。
                break
            # P0-1 fix: rebuild the per-batch remaining target BEFORE
            # every batch — the constant run-level fields
            # (total_transition_budget / rollout_transitions_per_update)
            # stay put (and stay hashed), only the derived accounting
            # knob varies. replace() keeps the config hash stable across
            # checkpoints (the hashed layer reads the constant fields).
            loop_config = replace(
                loop_config,
                target_transitions=compute_batch_target(
                    args.target_transitions, cumulative_transitions,
                    args.rollout_transitions_per_update,
                ),
            )
            # R5.8-D (N6)/R6-A (F-01, §11.2.4): the release decision must
            # NEVER be pre-proven — the old predicate
            # (remaining <= derived target) is ALWAYS true and made every
            # batch a release checkpoint. Episode mode keeps the absolute
            # update-limit fuse as its release signal; transition mode
            # passes NO kind — the batch driver decides post-hoc via
            # is_budget_complete on the ACTUAL env steps.
            if args.target_transitions is None:
                is_final_batch = batch_idx == max_updates
                checkpoint_kind = (
                    "release" if is_final_batch else "resume"
                )
            else:
                checkpoint_kind = None
            # Fresh wall-clock per batch — every checkpoint records its own
            # save time (the config-hash layer excludes the timestamp).
            runtime["timestamp"] = (
                datetime.datetime.now().isoformat(timespec="seconds")
            )
            res = run_e2e_batch(
                adapter, policy, critic, contract, loop_config,
                trainer_config, specs, actor_optimizer, critic_optimizer,
                lora_info, runtime, resume_from=prev_ckpt, eval_env=eval_env,
                task_sampler=sampler, paired_bank=paired_bank,
                offline_density=offline_density, bank_id=bank_id,
                # R5.8-D (N6)/R6-A (F-01): episode mode carries its
                # update-limit release signal; transition mode passes
                # None — the driver's post-hoc is_budget_complete verdict
                # on the ACTUAL env steps materializes exactly one
                # release per run (never a pre-proven predicate).
                checkpoint_kind=checkpoint_kind,
                # R7-6 Step 9: the metrics writer cursor — records
                # committed BEFORE this batch (the save happens inside
                # run_e2e_batch, before this batch's record is persisted
                # below); the restore-side drift gate demands
                # cursor == outer_iteration - 1.
                metrics_writer_cursor=batch_idx - 1,
            )
            prev_ckpt = res["checkpoint_path"]
            cumulative_transitions += int(res["env_steps"])
            if (args.target_transitions is not None
                    and cumulative_transitions
                    > args.target_transitions):
                print(
                    "preflight FAIL: transition budget overshoot — "
                    f"committed {cumulative_transitions} > total "
                    f"{args.target_transitions}",
                    file=sys.stderr,
                )
                return 1
            # §11.3.1 step 2: 入口验证 actual batch target。实际收集
            # != 本批派生目标（且 budget 未满）= admission/sampler 用尽
            # 的 partial batch —— loud 警告并落 INCOMPLETE_BUDGET
            # marker（§11.2.4 保险丝语义：partial 后由后续 batch 重算
            # 目标补齐；若最终仍未完成，budget 校验分支退出非零）。
            # 不得把 partial 当完整 batch 静默吞掉（Gate B R6-DISK-1
            # REWORK CE-2）。
            _verify_batch_target(
                Path(args.output_dir), batch_idx, max_updates,
                actual=int(res["env_steps"]),
                # episode mode (no run-level budget): None — the
                # verifier's first line (total is None) is the no-op
                # early return; int(None) must not be evaluated here
                # (one_update/no_update/g7_micro regressed to
                # TypeError since R6 batch-target verification).
                planned=(
                    int(loop_config.target_transitions)
                    if loop_config.target_transitions is not None
                    else None
                ),
                cumulative=cumulative_transitions,
                total=args.target_transitions,
            )
            # R6-B (§11.3.1.2): 入口发布验证 —— checkpoint 已物化、
            # kind 合法、transition 模式 release ⟺ counter==total。
            # 放 metrics append 之前：验证不过就不留记录（fail-before-
            # publish persistence 事务边界）。
            _validate_batch_publish(
                res, cumulative_transitions=cumulative_transitions,
                total_transitions=args.target_transitions,
            )
            persist_batch_metrics(
                Path(args.output_dir) / "metrics.jsonl", batch_idx, res,
                cumulative_env_steps=cumulative_transitions,
                world_assets_hash=runtime["world_assets_hash"],
            )
            # S3 retention: rotating latest (milestone/best/final materialize
            # separately — this training run keeps rotating latest only).
            _retain_checkpoints(Path(args.output_dir))
            print(
                f"batch {batch_idx}/{max_updates}: "
                f"env_steps={res['env_steps']} "
                f"completed={res['completed_episodes']} "
                f"voided={res['voided_episodes']} "
                f"ckpt={prev_ckpt}"
            )
            if res["eval_json_path"]:
                print(f"  paired eval: {res['eval_json_path']}")
    except EpisodeCollectionIncomplete as e:
        print(f"batch FAILED: {e}", file=sys.stderr)
        return 1
    except FloatingPointError as e:
        # R6-E §11.6.3 (F-09 → R6-DISK-1 E side): a non-finite parameter
        # or Adam state inside ppo_update raised BEFORE any publish — the
        # failed batch's checkpoint set / metrics.jsonl did not grow (the
        # transactional boundary is fail-before-publish, not a memory
        # rollback).
        print(f"batch FAILED (non-finite, fail-before-publish): {e}",
              file=sys.stderr)
        return 1
    except (RuntimeError, ValueError) as e:
        print(f"batch FAILED: {e}", file=sys.stderr)
        return 1

    if (args.target_transitions is not None
            and cumulative_transitions < args.target_transitions):
        # §11.2.4 保险丝：max_updates 先耗尽而 budget 未完成 → 退出非零
        # + INCOMPLETE_BUDGET marker（若 partial batch 已写过则保留）。
        marker = Path(args.output_dir) / "INCOMPLETE_BUDGET"
        if not marker.is_file():
            _write_incomplete_budget_marker(
                Path(args.output_dir),
                planned=int(loop_config.target_transitions),
                actual=int(res.get("env_steps", 0)),
                cumulative=cumulative_transitions,
                total=args.target_transitions,
            )
        print(
            f"budget NOT met: collected {cumulative_transitions} < "
            f"target {args.target_transitions} — refusing to report "
            "success on an incomplete transition budget (§11.2.4 "
            f"INCOMPLETE_BUDGET: {marker})",
            file=sys.stderr,
        )
        return 1
    # R6-B (§11.3.2): 终局 artifact audit —— metrics↔file 一致性、
    # 单 release、release counter == total。transition 模式强制恰好
    # 一个 final release；两种模式都检查证据断裂与多 release。
    audit = artifact_audit(
        Path(args.output_dir), total_transitions=args.target_transitions)
    print(
        "artifact audit PASS: "
        f"release={[p.name for p in audit['release_paths']] or 'none'}"
        + (f" problems={audit['problems']}" if audit["problems"] else "")
    )
    print(f"TRAINING COMPLETE stage={args.stage} last_ckpt={prev_ckpt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
