"""Descriptive analysis, versionable without changing the collection contract."""
import collections
import numpy as np


def paired_summary(records, treatment, reference):
    groups = {}
    for policy in (treatment, reference):
        entries = [r for r in records if r["policy"] == policy]
        indexed = {(r["task_id"], r["repeat"]): r for r in entries}
        if len(indexed) != len(entries) or not entries:
            raise ValueError("empty/duplicate policy episodes: " + policy)
        groups[policy] = indexed
    if set(groups[treatment]) != set(groups[reference]):
        raise ValueError("missing paired episode")
    rescue = broken = 0
    tasks, worlds = collections.defaultdict(list), collections.defaultdict(list)
    for key in sorted(groups[treatment]):
        a, b = groups[treatment][key], groups[reference][key]
        if a.get("status") != "valid" or b.get("status") != "valid":
            raise ValueError("invalid episode must be reported; cannot drop pair")
        for field in ("world_id", "role", "initial_hash"):
            if a[field] != b[field]:
                raise ValueError("paired field mismatch: " + field)
        delta = int(a["safe_arrival"]) - int(b["safe_arrival"])
        rescue += delta == 1
        broken += delta == -1
        tasks[key[0]].append(delta)
        worlds[a["world_id"]].append(delta)
    values = np.array([np.mean(x) for x in worlds.values()])
    ci = None
    if len(values) >= 2:
        rng = np.random.default_rng(20261009)
        draws = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
        ci = [float(x) * 100 for x in np.quantile(draws, [.025, .975])]
    return {"treatment": treatment, "reference": reference,
            "pairs": len(groups[treatment]), "tasks": len(tasks), "worlds": len(worlds),
            "rescue": int(rescue), "break": int(broken), "net": int(rescue - broken),
            "delta_pp": 100. * (rescue - broken) / len(groups[treatment]),
            "improved_tasks": sum(bool(np.mean(x) > 0) for x in tasks.values()),
            "world_mean_delta_pp": float(values.mean()) * 100,
            "world_cluster_descriptive_ci95_pp": ci,
            "inference_scope": "training-domain dev, fixed BC checkpoint; not holdout"}
