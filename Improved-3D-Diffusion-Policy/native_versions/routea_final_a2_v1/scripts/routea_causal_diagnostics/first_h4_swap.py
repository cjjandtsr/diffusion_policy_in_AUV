"""Pure, preregistered selection and outcome logic for first-H4 swaps."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np


ARM_SPECS = {
    "bc_bc": ("u0", "u0"),
    "rl_rl": ("seed3-u32", "seed3-u32"),
    "bc_rl": ("u0", "seed3-u32"),
    "rl_bc": ("seed3-u32", "u0"),
}


def _distance(entry: Mapping[str, Any]) -> float:
    config = entry["episode_config"]
    start = np.asarray(config["start_pos"], dtype=np.float64)
    goal = np.asarray(config["goal_pos"], dtype=np.float64)
    if start.shape != (3,) or goal.shape != (3,) or not np.isfinite(start).all() or not np.isfinite(goal).all():
        raise ValueError("invalid fixed start/goal")
    return float(np.linalg.norm(goal - start))


def select_matched_tasks(targets: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """One repeat per rescue task; closest distinct break task in the same role.

    The full rescue task set is retained; this prevents outcome or distance
    cherry-picking. Within each task, the lowest repeat is chosen before any
    new rollout. Break matching uses only initial distance and role.
    """

    by_task = defaultdict(list)
    for entry in targets:
        if entry["outcome_class"] in ("break", "rescue"):
            if int(entry["k_div"]) != 0:
                continue
            by_task[str(entry["task_id"])].append(entry)
    representatives = []
    for task_id, entries in sorted(by_task.items()):
        kinds = {(e["outcome_class"], e["role"]) for e in entries}
        if len(kinds) != 1:
            raise ValueError("task has inconsistent outcome or role: {}".format(task_id))
        representatives.append(min(entries, key=lambda e: int(e["repeat"])))
    rescues = sorted(
        (e for e in representatives if e["outcome_class"] == "rescue"),
        key=lambda e: (str(e["role"]), _distance(e), str(e["task_id"])),
    )
    breaks = [e for e in representatives if e["outcome_class"] == "break"]
    if not rescues:
        raise ValueError("no rescue tasks with first-decision divergence")
    used = set()
    pairs = []
    for rescue in rescues:
        candidates = [
            e for e in breaks
            if e["role"] == rescue["role"] and e["task_id"] not in used
        ]
        if not candidates:
            raise ValueError("no role-matched break for {}".format(rescue["task_id"]))
        chosen = min(
            candidates,
            key=lambda e: (
                abs(_distance(e) - _distance(rescue)),
                str(e["task_id"]),
            ),
        )
        used.add(chosen["task_id"])
        pairs.append(
            {
                "role": str(rescue["role"]),
                "initial_distance_gap_m": abs(_distance(chosen) - _distance(rescue)),
                "rescue": dict(rescue),
                "break": dict(chosen),
            }
        )
    return pairs


def first_actions(
    rows: Sequence[Mapping[str, Any]], task_id: str, repeat: int
) -> Dict[str, Dict[str, Any]]:
    selected = {}
    for row in rows:
        if str(row["task_id"]) != str(task_id) or int(row["repeat"]) != int(repeat):
            continue
        if int(row["decision_index"]) != 0:
            continue
        lane = str(row["lane"])
        if lane not in ("u0", "seed3-u32") or lane in selected:
            raise ValueError("duplicate or unknown first-action lane")
        action = np.asarray(row["policy_action_8x11"], dtype=np.float32)
        executed = np.asarray(row["executed_action_4x11"], dtype=np.float32)
        path = np.asarray(row["executed_h4_path_5x3"], dtype=np.float32)
        if action.shape != (8, 11) or executed.shape != (4, 11) or path.shape != (5, 3):
            raise ValueError("invalid first-action H8/H4 shape")
        if not np.isfinite(action).all() or not np.isfinite(path).all():
            raise ValueError("non-finite first action")
        if not np.array_equal(action[:4].view(np.uint32), executed.view(np.uint32)):
            raise ValueError("executed H4 differs from first policy H8")
        selected[lane] = {"action": action, "path": path, "row": row}
    if set(selected) != {"u0", "seed3-u32"}:
        raise ValueError("missing paired first actions for {} r{}".format(task_id, repeat))
    a = selected["u0"]["row"]
    b = selected["seed3-u32"]["row"]
    for field in (
        "action_seed", "policy_input_hash", "raw_frame_hash",
        "pointcloud_sha256", "position_before_xyz", "goal_xyz",
    ):
        if a.get(field) != b.get(field):
            raise ValueError("paired first-action {} differs".format(field))
    return selected


def classify_swap(bc_bc: bool, rl_rl: bool, bc_rl: bool, rl_bc: bool) -> str:
    """Classify only the two crossed endpoints; controls must be opposite."""

    if bool(bc_bc) == bool(rl_rl):
        raise ValueError("original controls must have opposite safe-arrival outcomes")
    if bool(bc_rl) == bool(bc_bc) and bool(rl_bc) == bool(rl_rl):
        return "FIRST_H4_DOMINANT"
    if bool(bc_rl) == bool(rl_rl) and bool(rl_bc) == bool(bc_bc):
        return "DOWNSTREAM_POLICY_DOMINANT"
    return "MIXED"
