"""Read-only Route A analysis linking goal progress to DPPO actor credit.

The script consumes immutable training metrics, held-out credit probes, and the
derived endpoint-trajectory table.  It never loads a checkpoint or steps an
optimizer.  Its purpose is to distinguish reward/credit misalignment from an
endpoint accounting defect and from the weaker claim of insufficient budget.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


def _number(record: Mapping[str, Any], *names: str) -> float:
    for name in names:
        value: Any = record
        for part in name.split("."):
            if not isinstance(value, Mapping) or part not in value:
                value = None
                break
            value = value[part]
        if value is not None:
            return float(value)
    raise KeyError("none of the numeric fields exists: {}".format(names))


def _rate(values: Iterable[bool]) -> float:
    rows = list(values)
    return float(sum(rows) / len(rows)) if rows else 0.0


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    return float(sum(rows) / len(rows)) if rows else 0.0


def _done_reason(record: Mapping[str, Any]) -> str:
    terminal = record.get("terminal_facts")
    if isinstance(terminal, Mapping):
        return str(terminal.get("done_reason", "none"))
    return str(record.get("done_reason_at_transition", "none"))


def recompute_finite_horizon_gae(
    records: Iterable[Mapping[str, Any]],
    *,
    timeout_penalty: float,
    cut_timeout_bootstrap: bool,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> list[float]:
    """Recompute GAE while optionally treating time-limit as task terminal.

    Collection boundaries remain truncations with bootstrap.  Only the real
    environment ``time_limit`` outcome is changed by the counterfactual.
    """

    rows = list(records)
    advantages = [0.0] * len(rows)
    carry = 0.0
    for index in range(len(rows) - 1, -1, -1):
        row = rows[index]
        timeout = _done_reason(row) == "time_limit"
        terminated = bool(row.get("terminated", False))
        truncated = bool(row.get("truncated", False) or row.get("collection_boundary", False))
        reward = float(row["reward"]) - (float(timeout_penalty) if timeout else 0.0)
        bootstrap = not terminated and not (timeout and cut_timeout_bootstrap)
        continuation = not (terminated or truncated)
        delta = reward + gamma * float(row["next_value"]) * float(bootstrap) - float(row["value"])
        carry = delta + gamma * lam * float(continuation) * carry
        advantages[index] = carry
    return advantages


def summarize_counterfactual_grid(
    records: Iterable[Mapping[str, Any]],
    *,
    penalties: Iterable[float] = (0.0, 5.0, 10.0, 20.0, 50.0),
) -> list[dict[str, Any]]:
    """Measure credit after imposing finite-horizon timeout semantics offline."""

    rows = list(records)
    current = recompute_finite_horizon_gae(
        rows, timeout_penalty=0.0, cut_timeout_bootstrap=False
    )
    recorded = [_number(row, "advantage_raw", "raw_advantage") for row in rows]
    verification_error = max(
        (abs(left - right) for left, right in zip(current, recorded)), default=0.0
    )
    away_indices = [
        index for index, row in enumerate(rows)
        if _number(row, "net_goal_approach_3d", "reward_metrics.raw_progress") < 0.0
    ]
    timeout_indices = [
        index for index, row in enumerate(rows) if _done_reason(row) == "time_limit"
    ]
    result = []
    for penalty in penalties:
        advantages = recompute_finite_horizon_gae(
            rows,
            timeout_penalty=float(penalty),
            cut_timeout_bootstrap=True,
        )
        mean = _mean(advantages)
        if len(advantages) > 1:
            variance = sum((value - mean) ** 2 for value in advantages) / (len(advantages) - 1)
            std = math.sqrt(variance)
        else:
            std = 0.0
        normalized = [
            (value - mean) / (std + 1e-8) for value in advantages
        ]
        result.append({
            "timeout_penalty": float(penalty),
            "current_gae_max_abs_error": verification_error,
            "away_positive_normalized_rate": _rate(normalized[index] > 0.0 for index in away_indices),
            "timeout_terminal_positive_normalized_rate": _rate(
                normalized[index] > 0.0 for index in timeout_indices
            ),
            "timeout_terminal_raw_advantage_mean": _mean(
                advantages[index] for index in timeout_indices
            ),
            "raw_advantage_mean": mean,
            "raw_advantage_std": std,
        })
    return result


def summarize_transition_credit(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize whether away-from-goal transitions receive positive credit."""

    rows = list(records)
    enriched = []
    for row in rows:
        raw = _number(row, "advantage_raw", "raw_advantage")
        normalized = _number(
            row, "advantage_normalized_outer_update", "normalized_advantage"
        )
        progress = _number(row, "net_goal_approach_3d", "reward_metrics.raw_progress")
        metrics = row.get("reward_metrics", {})
        enriched.append((row, raw, normalized, progress, bool(metrics.get("guard_context"))))

    away = [item for item in enriched if item[3] < 0.0]
    guard_away = [item for item in away if item[4]]
    timeout_terminal = [item for item in enriched if _done_reason(item[0]) == "time_limit"]
    reached_terminal = [item for item in enriched if _done_reason(item[0]) == "reached"]
    collision_terminal = [item for item in enriched if _done_reason(item[0]) == "collision"]

    def positive_normalized(items):
        return _rate(item[2] > 0.0 for item in items)

    def sign_flip(items):
        return _rate(item[1] <= 0.0 and item[2] > 0.0 for item in items)

    def mean_at(items, index):
        return _mean(item[index] for item in items)

    return {
        "transition_count": len(enriched),
        "raw_advantage_mean": mean_at(enriched, 1),
        "normalized_advantage_mean": mean_at(enriched, 2),
        "positive_normalized_rate": positive_normalized(enriched),
        "away_transition_count": len(away),
        "away_raw_advantage_mean": mean_at(away, 1),
        "away_normalized_advantage_mean": mean_at(away, 2),
        "away_positive_normalized_rate": positive_normalized(away),
        "away_raw_nonpositive_to_normalized_positive_rate": sign_flip(away),
        "guard_away_transition_count": len(guard_away),
        "guard_away_positive_normalized_count": sum(item[2] > 0.0 for item in guard_away),
        "guard_away_positive_normalized_rate": positive_normalized(guard_away),
        "time_limit_terminal_count": len(timeout_terminal),
        "time_limit_terminal_raw_advantage_mean": mean_at(timeout_terminal, 1),
        "time_limit_terminal_normalized_advantage_mean": mean_at(timeout_terminal, 2),
        "time_limit_terminal_positive_normalized_rate": positive_normalized(timeout_terminal),
        "time_limit_terminal_raw_nonpositive_to_normalized_positive_rate": sign_flip(timeout_terminal),
        "reached_terminal_count": len(reached_terminal),
        "reached_terminal_normalized_advantage_mean": mean_at(reached_terminal, 2),
        "collision_terminal_count": len(collision_terminal),
        "collision_terminal_normalized_advantage_mean": mean_at(collision_terminal, 2),
        "progress_reward_sum": sum(
            float(item[0].get("reward_terms", {}).get("progress", 0.0))
            for item in enriched
        ),
        "future_clearance_reward_sum": sum(
            float(item[0].get("reward_terms", {}).get("future_clearance", 0.0))
            for item in enriched
        ),
    }


def summarize_episode_rewards(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, float | int]]:
    """Aggregate complete episode return and reward terms by final outcome."""

    episodes: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for index, row in enumerate(records):
        episode = row.get("episode_id", row.get("episode_serial"))
        if episode is None:
            raise KeyError("probe record has no episode_id or episode_serial")
        episodes[str(episode)].append(row)

    per_outcome: dict[str, list[dict[str, float]]] = defaultdict(list)
    for rows in episodes.values():
        outcomes = {
            str(row.get("outcome", _done_reason(row)))
            for row in rows
            if str(row.get("outcome", _done_reason(row))) != "none"
        }
        if len(outcomes) != 1:
            raise ValueError("episode must have exactly one non-none outcome")
        outcome = next(iter(outcomes))
        totals = defaultdict(float)
        for row in rows:
            totals["episode_return"] += float(row["reward"])
            terms = row.get("reward_terms", {})
            for name, value in terms.items():
                totals[str(name)] += float(value)
        named = {"terminal", "progress", "future_clearance", "ground"}
        per_outcome[outcome].append({
            "episode_return": totals["episode_return"],
            "terminal": totals["terminal"],
            "progress": totals["progress"],
            "future_clearance": totals["future_clearance"],
            "ground": totals["ground"],
            "other": sum(
                value for name, value in totals.items()
                if name not in named and name != "episode_return"
            ),
        })

    result: dict[str, dict[str, float | int]] = {}
    for outcome, rows in sorted(per_outcome.items()):
        result[outcome] = {
            "episode_count": len(rows),
            "episode_return_mean": _mean(row["episode_return"] for row in rows),
            "terminal_reward_mean": _mean(row["terminal"] for row in rows),
            "progress_reward_mean": _mean(row["progress"] for row in rows),
            "future_clearance_reward_mean": _mean(
                row["future_clearance"] for row in rows
            ),
            "ground_reward_mean": _mean(row["ground"] for row in rows),
            "other_reward_mean": _mean(row["other"] for row in rows),
        }
    return result


def summarize_endpoint_fact_checks(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Count endpoint labels versus actual entry into the 0.5 m goal sphere."""

    grouped: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {
            "episodes": 0,
            "reported_reached": 0,
            "nonreached_entered_goal_0p5": 0,
        })
    )
    for row in rows:
        seed, policy = str(row["seed"]), str(row["policy"])
        target = grouped[seed][policy]
        target["episodes"] += 1
        reached = str(row["done"]) == "reached"
        entered = str(row["entered_goal_0p5"]).lower() == "true"
        target["reported_reached"] += int(reached)
        target["nonreached_entered_goal_0p5"] += int(entered and not reached)
    return {seed: dict(policies) for seed, policies in grouped.items()}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty analysis table")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _training_rows(training_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results, counterfactuals = [], []
    for path in sorted(training_root.glob("seed*/metrics_update_*.json")):
        match = re.search(r"seed(?P<seed>\d+).*metrics_update_(?P<update>\d+)", str(path))
        if match is None:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload["rollout_audit"]["transition_records"]
        row = {
            "seed": int(match.group("seed")),
            "update": int(match.group("update")),
            **summarize_transition_credit(records),
        }
        results.append(row)
        for counterfactual in summarize_counterfactual_grid(records):
            counterfactuals.append({
                "seed": row["seed"],
                "update": row["update"],
                **counterfactual,
            })
    return results, counterfactuals


def _probe_rows(
    credit_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results, episode_results = [], []
    for path in sorted(credit_root.glob("seed*_update_*/transitions.jsonl")):
        match = re.search(r"seed(?P<seed>\d+)_update_(?P<update>\d+)", str(path))
        if match is None:
            continue
        with path.open("r", encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        results.append({
            "seed": int(match.group("seed")),
            "update": int(match.group("update")),
            **summarize_transition_credit(records),
        })
        for outcome, summary in summarize_episode_rewards(records).items():
            episode_results.append({
                "seed": int(match.group("seed")),
                "update": int(match.group("update")),
                "outcome": outcome,
                **summary,
            })
    return results, episode_results


def _round_nested(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6) if math.isfinite(value) else value
    if isinstance(value, dict):
        return {key: _round_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_nested(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--credit-root", type=Path, required=True)
    parser.add_argument("--endpoint-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    training, counterfactuals = _training_rows(args.training_root.resolve())
    probes, probe_episode_rewards = _probe_rows(args.credit_root.resolve())
    with args.endpoint_metrics.resolve().open("r", encoding="utf-8", newline="") as stream:
        endpoint = summarize_endpoint_fact_checks(csv.DictReader(stream))

    _write_csv(output / "training_update_credit.csv", training)
    _write_csv(output / "finite_horizon_counterfactual.csv", counterfactuals)
    _write_csv(output / "checkpoint_credit_probe.csv", probes)
    _write_csv(output / "checkpoint_episode_rewards.csv", probe_episode_rewards)
    summary = {
        "schema": "routea_credit_goal_alignment_v1",
        "training_files": len(training),
        "probe_files": len(probes),
        "endpoint_fact_checks": endpoint,
        "training": training,
        "finite_horizon_counterfactual": counterfactuals,
        "probes": probes,
        "probe_episode_rewards": probe_episode_rewards,
    }
    (output / "aggregate.json").write_text(
        json.dumps(_round_nested(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
