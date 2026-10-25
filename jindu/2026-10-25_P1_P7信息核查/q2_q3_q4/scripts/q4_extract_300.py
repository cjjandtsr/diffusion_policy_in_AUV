#!/usr/bin/env python3
"""Q4: extract per-decision (position, yaw, target) for the 300 episodes in the
"non-dense, initially clear" group (100 conditions x {BC, u128, u256}) from the
1260 accepted A1 evaluation traces.

Classification is derived, not assumed:
  dense        = trace role == 'dense_recovery'   (per P4: dense defined by role)
  initial LOS  = 2D segment start->goal vs cylinder obstacles (P4: line criterion)
Validation: group counts must equal P4's {125, 175, 20, 100} and per-group
collision counts must equal P4's table, otherwise the script aborts.
"""
import csv
import json
import math
import sys
from pathlib import Path

FULL = Path("/home/cjj/AUV/program/results/routea_paired_evaluation/"
            "routea_common_a1_v1_run01/full")
POLICIES = ("bc", "rl_391a6656a0f7_u128", "rl_391a6656a0f7_u256")
OUT = Path("/home/cjj/AUV/plan/q2q3q4_20261025/q4_output")

P4_GROUP_COUNTS = {"dense_blocked": 125, "dense_clear": 175,
                   "rest_blocked": 20, "rest_clear": 100}
P4_COLLISIONS = {
    "bc": {"dense_blocked": 90, "dense_clear": 31, "rest_blocked": 10, "rest_clear": 0},
    "rl_391a6656a0f7_u128": {"dense_blocked": 109, "dense_clear": 23, "rest_blocked": 10, "rest_clear": 8},
    "rl_391a6656a0f7_u256": {"dense_blocked": 105, "dense_clear": 27, "rest_blocked": 10, "rest_clear": 17},
}


def classify(record):
    import sys as _sys
    _sys.path.insert(0, "/home/cjj/AUV/Improved-3D-Diffusion-Policy-main/"
                        "Improved-3D-Diffusion-Policy")
    from scripts.analyze_routea_trajectory_behavior import Scene
    role = record.get("role", "")
    dense = role == "dense_recovery"
    goal = record["task"]["goal_pos"]
    start = record["task"].get("start_pos")
    if start is None:
        start = record["start_xy"]
    # exact P4 criterion: Scene.clear with 0.10 m proxy and vertical exemption
    blocked = not Scene(record.get("geometry", [])).clear(start, goal)
    if dense and blocked:
        group = "dense_blocked"
    elif dense:
        group = "dense_clear"
    elif blocked:
        group = "rest_blocked"
    else:
        group = "rest_clear"
    return group, role


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # read one policy's episode_start records to build the condition table
    bc_dir = FULL / "bc" / "traces"
    conditions = {}
    for path in sorted(bc_dir.glob("*.jsonl")):
        lines = [json.loads(line) for line in path.read_text().splitlines()]
        start = lines[0]
        assert start["type"] == "episode_start"
        first_trans = next(r for r in lines if r.get("type") == "transition")
        key = (start["task_id"], start.get("repeat"))
        start_xy = first_trans["pre_state"]["position"][:2]
        conditions[key] = {
            "task_id": start["task_id"], "repeat": start.get("repeat"),
            "world_id": start["world_id"],
            "start_xy": start_xy, "start": first_trans["pre_state"],
            "record": start, "bc_file": path.name,
        }
    print("conditions:", len(conditions))

    classification = {}
    for key, info in conditions.items():
        group, role = classify(
            {**info["record"], "start_xy": info["start_xy"]})
        classification[key] = {"group": group, "role": role}
    count_by_group = {}
    for key, val in classification.items():
        count_by_group[val["group"]] = count_by_group.get(val["group"], 0) + 1
    print("group counts:", count_by_group)
    assert count_by_group == P4_GROUP_COUNTS, (
        "group counts mismatch P4: %s vs %s" % (count_by_group, P4_GROUP_COUNTS))

    # collision counts per policy per group (validate classification)
    collisions = {p: {g: 0 for g in P4_GROUP_COUNTS} for p in POLICIES}
    per_decision_rows = []
    rest_clear_keys = {k for k, v in classification.items()
                       if v["group"] == "rest_clear"}
    print("rest_clear conditions:", len(rest_clear_keys))
    for policy in POLICIES:
        pdir = FULL / policy / "traces"
        for key, info in conditions.items():
            path = pdir / info["bc_file"]
            lines = [json.loads(line) for line in path.read_text().splitlines()]
            group = classification[key]["group"]
            collided = any(r.get("done_reason") in ("obstacle_collision", "ground_crash")
                           or (r.get("done_reason") or "").startswith("collision")
                           for r in lines if r.get("type") == "transition")
            if collided:
                collisions[policy][group] += 1
            if key in rest_clear_keys:
                goal = lines[0]["task"]["goal_pos"]
                for r in lines:
                    if r.get("type") != "transition":
                        continue
                    per_decision_rows.append({
                        "policy": policy,
                        "task_id": info["task_id"],
                        "repeat": info["repeat"],
                        "decision": r["decision"],
                        "pos_x": r["pre_state"]["position"][0],
                        "pos_y": r["pre_state"]["position"][1],
                        "pos_z": r["pre_state"]["position"][2],
                        "yaw": r["pre_state"]["yaw"],
                        "goal_x": goal[0], "goal_y": goal[1], "goal_z": goal[2],
                        "done_reason": r.get("done_reason"),
                    })
    print("collisions:", collisions)
    assert collisions == P4_COLLISIONS, (
        "collision counts mismatch P4: %s vs %s" % (collisions, P4_COLLISIONS))
    print("per-decision rows:", len(per_decision_rows))

    with (OUT / "q4_group_classification.json").open("w") as stream:
        json.dump({"group_counts": count_by_group, "collisions": collisions,
                   "classification": {
                       "{0}__rep{1}".format(k[0], k[1]): v
                       for k, v in classification.items()}}, stream, indent=2)
    with (OUT / "q4_per_decision_300.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "policy", "task_id", "repeat", "decision", "pos_x", "pos_y",
            "pos_z", "yaw", "goal_x", "goal_y", "goal_z", "done_reason"])
        writer.writeheader()
        writer.writerows(per_decision_rows)
    print("saved under", OUT)


if __name__ == "__main__":
    main()
