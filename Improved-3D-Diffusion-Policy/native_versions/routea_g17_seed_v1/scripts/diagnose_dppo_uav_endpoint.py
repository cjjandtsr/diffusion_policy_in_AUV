#!/usr/bin/env python3
"""Read a completed fixed32 evidence pack; never run or retune a policy."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.formal_endpoint_eval_core import PairedEpisodeRecord, PairedEvaluationBank


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def invalid(value):
        raise ValueError("nonfinite JSON token: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def load_evidence(source):
    source = Path(source)
    rows = [json.loads(line) for line in (source / "paired_episodes.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    bank = PairedEvaluationBank(PairedEpisodeRecord(**row) for row in rows)
    report = read_json(source / "endpoint_report.json")
    provenance = read_json(source / "evaluation_provenance.json")
    worlds = Counter((bc.split, bc.world_id) for bc, _ in bank.pairs)
    if (len(bank.pairs) != 128 or len(worlds) != 32 or set(worlds.values()) != {4}
            or Counter(split for split, _ in worlds) != {"rl_test": 12, "layerA": 20}):
        raise ValueError("source must be the full 128-pair fixed32 bank")
    if len({bc.task_id for bc, _ in bank.pairs}) != 128:
        raise ValueError("task_id is not unique across source pairs")
    if not bank.assert_not_training_source() or any(r.reward_schema_version != "v17" for r in bank.records):
        raise ValueError("requires eval-only v17 evidence")
    if report.get("bank_content_hash") != bank.content_hash or provenance.get("bank_content_hash") != bank.content_hash:
        raise ValueError("source report/provenance/episodes bank hash mismatch")
    if (report.get("pair_count") != 128 or report.get("record_count") != 256
            or report.get("world_count") != 32 or provenance.get("benchmark") != "fixed32"
            or provenance.get("episodes_per_policy") != 128):
        raise ValueError("source counts disagree")
    if report.get("evidence_scope") != "lightweight_kinematic" or provenance.get("gazebo_px4_evidence") is not False:
        raise ValueError("incorrect evidence scope")
    for policy in ("bc", "rl"):
        identities = {r.policy_identity_hash for r in bank.records if r.policy == policy}
        if len(identities) != 1 or sorted(identities) != report["policy_identity_hashes"][policy]:
            raise ValueError("mixed or mismatched policy identity")
    return bank, report, provenance


def world_ci(grouped):
    """Equal-weight worlds, intact paired replicas; one training seed only."""
    arrays = [np.asarray(grouped[k], dtype=np.float64) for k in sorted(grouped) if len(grouped[k])]
    if any(a.ndim != 1 or not np.isfinite(a).all() for a in arrays):
        raise ValueError("world cluster must contain finite scalar pair deltas")
    means = np.array([a.mean() for a in arrays])
    result = dict(mean=float(means.mean()) if len(means) else None, lower=None, upper=None,
                  worlds=len(means), pairs=sum(len(a) for a in arrays),
                  resampling_unit="world", samples=10000, seed=42,
                  scope="descriptive_single_training_seed")
    if len(means) >= 2:
        draws = np.random.default_rng(42).integers(0, len(means), size=(10000, len(means)))
        lo, hi = np.quantile(means[draws].mean(axis=1), [.025, .975])
        result.update(lower=float(lo), upper=float(hi))
    return result


def select_replay_pairs(bank):
    pairs = sorted(bank.pairs, key=lambda pair: pair[0].task_id)
    selected = {bc.task_id: "outcome_changed" for bc, rl in pairs if bc.done_reason != rl.done_reason}
    for split in sorted({bc.split for bc, _ in pairs}):
        successes = [(bc, rl) for bc, rl in pairs if bc.split == split and bc.success and rl.success]
        if successes:
            selected[successes[0][0].task_id] = "both_success_lexical_first_in_split"
        timeouts = [(bc, rl) for bc, rl in pairs if bc.split == split and bc.time_limit and rl.time_limit]
        if timeouts:
            def rank(pair):
                bc, rl = pair
                delta = (rl.v17_metrics or {}).get("above_demo_stall_seconds", 0.) - (bc.v17_metrics or {}).get("above_demo_stall_seconds", 0.)
                return (-delta, bc.task_id)
            selected[min(timeouts, key=rank)[0].task_id] = "both_timeout_largest_stall_delta_in_split"
    return [dict(task_id=bc.task_id, world_id=bc.world_id, split=bc.split,
                 bc_reason=bc.done_reason, rl_reason=rl.done_reason, selection_reason=selected[bc.task_id])
            for bc, rl in pairs if bc.task_id in selected]


def _efficiency(pairs):
    fields = (("length_relative_delta", "executed_length_m", True),
              ("duration_relative_delta", "observed_duration_seconds", True),
              ("climb_delta_m", "positive_climb_m", False))
    result = {"pairs": len(pairs), "scope": "both_success_same_task_only"}
    for label, field, relative in fields:
        grouped = defaultdict(list)
        undefined = 0
        for bc, rl in pairs:
            left, right = (bc.v17_metrics or {}).get(field), (rl.v17_metrics or {}).get(field)
            if left is None or right is None or (relative and left <= 0):
                undefined += 1
                continue
            grouped[(bc.split, bc.world_id)].append((right - left) / left if relative else right - left)
        result[label] = dict(world_ci(grouped), undefined_pairs=undefined)
    return result


def analyze_bank(bank):
    matrix = defaultdict(Counter)
    per_world = {}
    deltas = {name: defaultdict(list) for name in ("safe_arrival", "collision", "time_limit")}
    paired_rows = []
    for bc, rl in bank.pairs:
        matrix[bc.done_reason][rl.done_reason] += 1
        key = (bc.split, bc.world_id)
        row = per_world.setdefault(key, dict(split=bc.split, world_id=bc.world_id,
                                            pairs=0, bc_success=0, rl_success=0))
        row["pairs"] += 1
        row["bc_success"] += int(bc.success)
        row["rl_success"] += int(rl.success)
        for name, value in (("safe_arrival", int(rl.success) - int(bc.success)),
                            ("collision", int(rl.collision) - int(bc.collision)),
                            ("time_limit", int(rl.time_limit) - int(bc.time_limit))):
            deltas[name][key].append(value)
        paired_rows.append(dict(task_id=bc.task_id, world_id=bc.world_id, split=bc.split,
            role=bc.role, bc_reason=bc.done_reason, rl_reason=rl.done_reason,
            bc_steps=bc.steps, rl_steps=rl.steps, bc_return=bc.episode_return, rl_return=rl.episode_return,
            bc_max_z=bc.max_xyz[2], rl_max_z=rl.max_xyz[2],
            bc_above_stall=(bc.v17_metrics or {}).get("above_demo_stall_seconds"),
            rl_above_stall=(rl.v17_metrics or {}).get("above_demo_stall_seconds")))
    both = [(bc, rl) for bc, rl in bank.pairs if bc.success and rl.success]
    high_support = [(bc, rl) for bc, rl in both
                    if (rl.v17_metrics or {}).get("support_eligible_above_demo", 0) > 0
                    and (bc.v17_metrics or {}).get("support_eligible_above_demo", 0) == 0]
    totals = {p: dict(Counter(r.done_reason for r in bank.records if r.policy == p)) for p in ("bc", "rl")}
    lost = sum(bc.success and not rl.success for bc, rl in bank.pairs)
    gained = sum(rl.success and not bc.success for bc, rl in bank.pairs)
    high = {}
    for policy in ("bc", "rl"):
        chosen = [r for r in bank.records if r.policy == policy]
        above = [r for r in chosen if (r.v17_metrics or {}).get("above_demo_height")]
        sums = {k: sum((r.v17_metrics or {}).get(k, 0.) for r in chosen) for k in
                ("above_demo_dwell_seconds", "above_demo_unknown_seconds", "above_demo_stall_seconds")}
        known = sums["above_demo_dwell_seconds"] - sums["above_demo_unknown_seconds"]
        high[policy] = dict(sums, episodes_above=len(above), reached_above=sum(r.success for r in above),
                            stall_rate=sums["above_demo_stall_seconds"] / known if known > 0 else None)
    return dict(pairs=len(bank.pairs), totals=totals, success_lost_pairs=lost, success_gained_pairs=gained,
        safe_arrival_delta=(gained - lost) / len(bank.pairs) if bank.pairs else None,
        transition_counts={k: dict(v) for k, v in sorted(matrix.items())},
        world_cluster_ci={k: world_ci(v) for k, v in deltas.items()},
        per_world=[dict(v, safe_arrival_delta=(v["rl_success"] - v["bc_success"]) / v["pairs"])
                   for _, v in sorted(per_world.items())], paired_rows=sorted(paired_rows, key=lambda x: x["task_id"]),
        both_success_efficiency=_efficiency(both), high_support_matched_efficiency=_efficiency(high_support),
        high_altitude=high, no_ray_hit="unavailable_from_episode_summaries",
        interpretation="Single-seed diagnostic; fixed32 must not be used to select reward parameters.")


def write_analysis(source, output):
    bank, report, provenance = load_evidence(source)
    result = analyze_bank(bank)
    selection = select_replay_pairs(bank)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    hashes = {name: sha256(Path(source) / name) for name in
              ("paired_episodes.jsonl", "endpoint_report.json", "evaluation_provenance.json")}
    result["source_sha256"] = hashes
    result["bank_content_hash"] = bank.content_hash
    write_json(output / "analysis.json", result)
    write_json(output / "replay_selection.json", dict(schema_version=1, source_sha256=hashes,
        bank_content_hash=bank.content_hash, pairs=selection, selection_count=len(selection),
        selection_rule="all changed outcomes + one lexical both-success and one maximum RL-minus-BC above-stall both-timeout per split; ties by task_id",
        scope="diagnostic_subset_not_new_endpoint"))
    with (output / "paired_migrations.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["paired_rows"][0]))
        writer.writeheader()
        writer.writerows(result["paired_rows"])
    reasons = ["reached", "collision", "time_limit", "crash_ground", "out_of_bounds", "altitude_exceed", "invalid_state"]
    lines = ["# Reward v1.7 fixed32 配对迁移分析", "", "[FACT] 来源：完整原始JSON；未重跑策略。",
             "", "| BC → RL | " + " | ".join(reasons) + " |", "|---|" + "---:|" * len(reasons)]
    for left in reasons:
        lines.append("| " + left + " | " + " | ".join(str(result["transition_counts"].get(left, {}).get(right, 0)) for right in reasons) + " |")
    lines += ["", "[FACT] 成功损失{}组，成功新增{}组；配对差{:+.4f}。".format(
        result["success_lost_pairs"], result["success_gained_pairs"], result["safe_arrival_delta"]),
        "", "## World整组bootstrap（单训练seed描述性诊断）", ""]
    for metric, values in result["world_cluster_ci"].items():
        lines.append("- {}：mean={}，95% CI=[{}, {}]，{} worlds。".format(metric, values["mean"], values["lower"], values["upper"], values["worlds"]))
    lines += ["", "## 高空统计", "", "```json", json.dumps(result["high_altitude"], indent=2), "```",
              "", "[JUDGMENT] unknown是历史窗口不足，不能替代no-ray-hit。高空时长差异不证明其导致失败。",
              "", "## 轨迹诊断范围", "", "选定{}组配对（{}回合），见replay_selection.json；原128组结论保留。".format(len(selection), 2 * len(selection)),
              "", "[FACT] 同任务双方成功共{}组；效率结果见analysis.json。".format(result["both_success_efficiency"]["pairs"]),
              "", "[JUDGMENT] 现有JSON只有回合汇总，不能恢复完整逐步轨迹；本报告不作停滞/绕圈/no-ray-hit因果结论。",
              "", "[JUDGMENT] fixed32只诊断本次结果，后续机制验证与调参回到训练/开发世界。"]
    with (output / "analysis.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    print(json.dumps(dict(pairs=result["pairs"], lost=result["success_lost_pairs"],
        gained=result["success_gained_pairs"], replay_pairs=len(selection), output=str(output)), ensure_ascii=False))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    return write_analysis(args.source_dir, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
