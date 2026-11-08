"""Validate the accepted frozen task bank."""
from pathlib import Path

import numpy as np

from .core import ROLES, canonical_hash, file_hash, observation_hash, read_json


EXPECTED_WORLD_ROLES = dict(zip(ROLES, (2, 15, 2, 2)))


def bank_contract(kind):
    if kind == 'routea_paper_test_v1':
        return 12, 24, 120, 2
    if kind == 'legacy':
        return 21, 84, 420, 4
    raise ValueError('unknown bank contract')


def episode_key(task_id, repeat):
    return canonical_hash([task_id, int(repeat)])[:24]


def check_bank(path):
    path = Path(path)
    bank = read_json(path)
    paper = bank.get('schema') == 'routea_paper_test_v1'
    nw, nt, ne, per_world = bank_contract('routea_paper_test_v1' if paper else 'legacy')
    if canonical_hash({k: v for k, v in bank.items() if k != "content_hash"}) != bank["content_hash"]:
        raise ValueError("frozen bank content changed")
    if len(bank["tasks"]) != nt or len(bank["episodes"]) != ne:
        raise ValueError("bank count differs from frozen contract")
    keys = [e["key"] for e in bank["episodes"]]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate bank episode key")
    if len(bank["worlds"]) != nw:
        raise ValueError("bank world count differs")
    actual_counts = {r: sum(w["role"] == r for w in bank["worlds"].values()) for r in ROLES}
    expected_roles = dict(zip(ROLES,(1,9,1,1))) if paper else EXPECTED_WORLD_ROLES
    if actual_counts != expected_roles or bank["role_world_counts"] != actual_counts:
        raise ValueError("bank role/world structure differs")
    tasks = {t["task_id"]: t for t in bank["tasks"]}
    if len(tasks) != nt:
        raise ValueError("duplicate task id")
    for uid, world in bank["worlds"].items():
        selected = [t for t in tasks.values() if t["world_id"] == uid]
        if len(selected) != per_world or any(t["role"] != world["role"] for t in selected):
            raise ValueError("world/task role mismatch")
    for task_id in tasks:
        repeats = [e["repeat"] for e in bank["episodes"] if e["task_id"] == task_id]
        if sorted(repeats) != [1,2,3,4,5]:
            raise ValueError("task repeat set differs")
    for world in bank["worlds"].values():
        if file_hash(path.parent / world["obstacle_copy"]) != world["obstacle_sha256"]:
            raise ValueError("frozen obstacle bytes changed")
    for entry in bank["episodes"]:
        if entry["task_id"] not in tasks or entry["key"] != episode_key(entry["task_id"],entry["repeat"]):
            raise ValueError("episode key/task mismatch")
        p = path.parent / entry["initial_npz"]
        if file_hash(p) != entry["initial_npz_sha256"]:
            raise ValueError("frozen initial observation file changed")
        with np.load(p, allow_pickle=False) as arrays:
            if (set(arrays.files) != {"point_cloud","agent_pos"}
                    or arrays["agent_pos"].shape != (8,9)
                    or arrays["point_cloud"].shape[:2] != (8,4096)
                    or arrays["point_cloud"].shape[-1] not in (3,6)
                    or any(arrays[k].dtype != np.float32 for k in arrays.files)):
                raise ValueError("initial observation shape/dtype mismatch")
            if observation_hash({k: arrays[k] for k in arrays.files}) != entry["observation_hash"]:
                raise ValueError("frozen initial observation arrays changed")
    return bank
