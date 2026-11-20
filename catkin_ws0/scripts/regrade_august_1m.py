#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对 gazebo_3method_288_run01 的 288 条件按 1.0 m 到达判据离线重判汇总。

判据（与运动学 terminal_contract.reached_goal 同结构, 仅半径 0.5 → 1.0）:
  - 到达 = 轨迹任意帧 3D 欧氏距离(pos → goal_odom) ≤ 1.0 m (closed boundary)
  - first-event 语义: collision > lost > reached > timeout
      collision 优先 (monitor JSON collision=true → collision, 即使中途到过 1.0 m)
      lost 优先 (越界逃逸不算到达)
  - setup_failed / no_result: 基础设施失败, 不参与成败率分母
坐标系:
  - result_state.csv = /mavros/local_position/odom 的 (px,py,pz), 原点 = spawn = 任务起点
  - goal_odom = (goal_world.x - start.x, goal_world.y - start.y, goal_world.z)
    (SPAWN_Z=0, odom z 原点在地面 ⇒ odom z == world z)
输出 (OUT_DIR 内):
  - regrade_1m.csv        每条件明细 (双口径: monitor 原口径 + 1.0m 新口径)
  - summary_1m.json       每方法新口径计数与到达率
  - summary_monitor.json  每方法 monitor 原口径计数 (对照)
用法: python3 regrade_288_1m.py [--radius 1.0] [--out <out_dir>]
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
from pathlib import Path

ROOT = Path("/root/autodl-tmp/AUV")
OUT_DIR = ROOT / "program/results/gazebo_august_condition_3method_run01"
TASKS_JSON = ROOT / "program/results/routea_gazebo_august_layerA_tasks.json"


def load_condition(d: Path) -> dict:
    cj = d / "condition.json"
    cond = json.load(open(cj)) if cj.exists() else {}
    # condition.json 不存 lost 布尔 (run_288_schedule 只取了 status 字符串),
    # 布尔判据以 monitor 原始 result.json 为准
    rj = d / "result.json"
    if rj.exists():
        try:
            r = json.load(open(rj))
            cond.setdefault("_result_collision", bool(r.get("collision")))
            cond.setdefault("_result_lost", bool(r.get("lost")))
            cond.setdefault("_result_timeout", bool(r.get("timeout")))
            cond.setdefault("_result_success_deploy", bool(r.get("success_deploy")))
        except Exception:
            pass
    return cond


def load_traj_csv(csv_path: Path):
    """Return list of (t, x, y, z) from result_state.csv."""
    rows = []
    if not csv_path.exists():
        return rows
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append((float(row["timestamp"]),
                         float(row["px"]), float(row["py"]), float(row["pz"])))
    return rows


def min3d_and_first_hit(traj, goal_odom, radius: float):
    """Return (min_3d_dist, first_hit_time, hit)."""
    gx, gy, gz = goal_odom
    best = math.inf
    first_hit = None
    t0 = traj[0][0] if traj else 0.0
    for t, x, y, z in traj:
        d = math.sqrt((x - gx) ** 2 + (y - gy) ** 2 + (z - gz) ** 2)
        if d < best:
            best = d
        if d <= radius and first_hit is None:
            first_hit = t - t0
    return best, first_hit


def monitor_status(cond: dict) -> str:
    if cond.get("setup_failed"):
        return "setup_failed"
    return cond.get("status") or "no_result"


def regrade(cond: dict, traj, goal_odom, radius: float) -> dict:
    """New-outcome row for one condition."""
    res = {"monitor_status": monitor_status(cond)}
    if res["monitor_status"] in ("setup_failed", "no_result"):
        res["new_status"] = res["monitor_status"]
        res["min3d"] = None
        res["first_hit_s"] = None
        return res
    min3d, first_hit = min3d_and_first_hit(traj, goal_odom, radius)
    res["min3d"] = round(min3d, 4)
    res["first_hit_s"] = None if first_hit is None else round(first_hit, 2)
    if cond.get("collision") or cond.get("_result_collision"):
        res["new_status"] = "collision"
    elif cond.get("lost") or cond.get("_result_lost") \
            or cond.get("status") == "lost":
        res["new_status"] = "lost"
    elif first_hit is not None:
        res["new_status"] = "reached"
    else:
        res["new_status"] = "timeout"
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=1.0)
    ap.add_argument("--out", type=str, default=str(OUT_DIR))
    args = ap.parse_args()
    out = Path(args.out)

    bank = json.load(open(TASKS_JSON))
    by_id = {t["task"]["task_id"]: t for t in bank["tasks"]}

    rows = []
    for method_dir in sorted(out.iterdir()):
        if method_dir.name in ("run_config.json", "run_log.jsonl", "STOP") \
                or not method_dir.is_dir():
            continue
        method = method_dir.name
        for cond_dir in sorted(method_dir.iterdir()):
            if not cond_dir.is_dir():
                continue
            cond = load_condition(cond_dir)
            if not cond:
                continue
            # august condition.json 自带 start_pos/goal_pos (world 系), 无需查 bank
            p = cond
            sx, sy, _sz = (cond.get("start_pos") or [0, 0, 0])[:3]
            gx, gy, gz = (cond.get("goal_pos") or [0, 0, 0])[:3]
            goal_odom = (gx - sx, gy - sy, gz)
            traj = load_traj_csv(cond_dir / "result_state.csv")
            r = regrade(cond, traj, goal_odom, args.radius)
            rows.append({
                "method": method,
                "task_id": cond.get("task_id", "")[:12],
                "repeat": cond.get("repeat"),
                "world": cond.get("world_name", ""),
                "category": cond.get("task_category", ""),
                "role": "",
                "monitor_status": r["monitor_status"],
                "monitor_final_dxy": cond.get("final_dist_xy"),
                "monitor_final_dz": cond.get("final_dist_z"),
                "monitor_time_to_goal": cond.get("time_to_goal"),
                "min3d_m": r["min3d"],
                "new_status_1m": r["new_status"],
                "first_hit_s": r["first_hit_s"],
            })

    # 明细 CSV
    flds = ["method", "task_id", "repeat", "world", "category", "role",
            "monitor_status", "monitor_final_dxy", "monitor_final_dz",
            "monitor_time_to_goal", "min3d_m", "new_status_1m", "first_hit_s"]
    with open(out / "regrade_1m.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flds)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # 汇总
    def summarize(status_key: str):
        c = collections.Counter()
        for r in rows:
            c[(r["method"], r[status_key])] += 1
        summary = {}
        for m in ("EGO", "P6", "P0"):
            d = {k: c[(m, k)] for k in
                 ("reached", "collision", "timeout", "lost",
                  "setup_failed", "no_result")}
            denom = 100 - d["setup_failed"] - d["no_result"]
            d["total"] = 100
            d["valid"] = denom
            d["arrival_rate"] = round(d["reached"] / denom, 4) if denom else None
            summary[m] = d
        return summary

    with open(out / "summary_1m.json", "w") as f:
        json.dump({"radius_m": args.radius,
                   "criterion": "3D euclidean <= radius, first-event, collision/lost priority",
                   "methods": summarize("new_status_1m")},
                  f, indent=2, ensure_ascii=False)
    with open(out / "summary_monitor.json", "w") as f:
        json.dump({"criterion": "monitor 原口径 (success_deploy dxy<0.5&dz<0.3, collision/lost/timeout)",
                   "methods": summarize("monitor_status")},
                  f, indent=2, ensure_ascii=False)

    print(json.dumps(summarize("new_status_1m"), indent=2, ensure_ascii=False))
    print(f"明细: {out / 'regrade_1m.csv'}  ({len(rows)} 条)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
