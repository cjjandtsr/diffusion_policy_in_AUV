#!/usr/bin/env python3
"""Score the candidate running cost on accepted traces without rerunning policies."""

import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diffusion_policy_3d.env.reward_v2_support.reward import space_running_cost


def summary(values):
    a = np.asarray(values, dtype=np.float64)
    return {"count": len(a), "min": float(a.min()) if len(a) else None,
            "mean": float(a.mean()) if len(a) else None,
            "max": float(a.max()) if len(a) else None}


def analyze(full_dir, budget=1.):
    episodes, by_condition = [], defaultdict(list)
    local = defaultdict(list)
    local_on_success = defaultdict(list)
    transitions = 0
    for job in sorted(Path(full_dir).iterdir()):
        if not job.is_dir() or not (job / "commits").is_dir():
            continue
        for commit in sorted((job / "commits").glob("*.json")):
            ref = json.loads(commit.read_text(encoding="utf-8"))
            rows = []
            with (job / ref["trace"]).open(encoding="utf-8") as stream:
                header = json.loads(next(stream))
                start = np.asarray(header["task"]["start_pos"], dtype=np.float64)
                goal = np.asarray(header["task"]["goal_pos"], dtype=np.float64)
                for line in stream:
                    row = json.loads(line)
                    if row["type"] == "transition":
                        path = np.asarray(row["executed_path"], dtype=np.float64)
                        durations = np.full(len(path) - 1, .3)
                        if row["event_fraction"] is not None:
                            durations[-1] *= row["event_fraction"]
                        if abs(durations.sum() - row["executed_time_s"]) > 1e-6:
                            raise ValueError("trace time/path disagreement: " + str(commit))
                        cost = space_running_cost(path, start, goal, durations, budget, 72.)
                        approach = float(np.linalg.norm(path[0] - goal) - np.linalg.norm(path[-1] - goal))
                        terms = row["reward_terms"]
                        category = ("retreat" if approach < -.05 else "recovery_without_newbest"
                                    if approach > .05 and all(abs(terms[k]) < 1e-9 for k in ("terminal", "newbest", "risk"))
                                    else "other")
                        rows.append((float(row["reward"]), cost, category))
                    elif row["type"] == "episode_end":
                        end = row
            if len(rows) != ref["summary"]["decisions"]:
                raise ValueError("incomplete accepted episode")
            costs = np.array([r[1] for r in rows])
            rewards = np.array([r[0] for r in rows])
            total_cost = float(costs.sum())
            if not -budget - 1e-6 <= total_cost <= 1e-8:
                raise ValueError("episode space budget violated")
            discount = .99 ** np.arange(len(rows))
            record = {"policy": job.name, "task_id": header["task_id"], "repeat": header["repeat"],
                      "safe_arrival": bool(end["safe_arrival"]), "space_cost": total_cost,
                      "before": float(rewards.sum()), "after": float((rewards + costs).sum()),
                      "discounted_before": float(rewards @ discount),
                      "discounted_after": float((rewards + costs) @ discount)}
            episodes.append(record)
            by_condition[(record["task_id"], record["repeat"])].append(record)
            transitions += len(rows)
            for reward, cost, category in rows:
                local[category].append(cost)
                if record["safe_arrival"]:
                    local_on_success[category].append(cost)
        print("SCORED", job.name, len(episodes), flush=True)
    comparisons = {}
    for prefix in ("", "discounted_"):
        flips = []
        success_lost = []
        pairs = 0
        for condition, group in by_condition.items():
            for left, right in combinations(group, 2):
                pairs += 1
                before = left[prefix + "before"] - right[prefix + "before"]
                after = left[prefix + "after"] - right[prefix + "after"]
                if before * after < -1e-12:
                    item = {"task_id": condition[0], "repeat": condition[1],
                            "left": left["policy"], "right": right["policy"],
                            "margin_before": before, "margin_after": after}
                    flips.append(item)
                    if left["safe_arrival"] != right["safe_arrival"]:
                        direction = 1 if left["safe_arrival"] else -1
                        if direction * before > 0 and direction * after < 0:
                            success_lost.append(item)
        comparisons[prefix + "ranking"] = {"paired_comparisons": pairs, "strict_reversals": len(flips),
                                            "success_loses_to_failure": success_lost, "reversal_examples": flips[:10]}
    return {"scope": "accepted v3 fixed traces; counterfactual scoring, not retraining or causal evidence",
            "full_dir": str(Path(full_dir).resolve()), "budget": budget, "horizon_s": 72.,
            "episodes": len(episodes), "transitions": transitions,
            "episode_space_cost": summary([e["space_cost"] for e in episodes]),
            "local_cost": {k: summary(v) for k, v in local.items()},
            "local_cost_in_eventually_successful_episodes": {k: summary(v) for k, v in local_on_success.items()},
            **comparisons, "records": episodes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    report = analyze(args.full)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
