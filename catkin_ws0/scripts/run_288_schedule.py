#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按 tasks.json 的 schedule 跑完 288 条件（P0 / P6@u256 / EGO），逐条件落盘、断点续跑。

设计约束（2026-11-07 指令）：
  - 三种方法共用同一条 Gazebo 闭环链路：run_task_flight.sh 的启动 / 起飞 / 控制交接 /
    感知 / 监控 / 清理流程完全一致，唯一差别是 ENGINE（bc|ego）与 LoRA 门控。
  - 同一 task 的 world / start / goal / start_yaw 对三种方法逐字相同。
  - 逐条件保存 monitor JSON（原件）、monitor 轨迹 CSV、planner 日志、实际命令与环境。
  - 断点续跑：result.json 存在即视为该条件已完成，不再重跑。
  - 基础设施失败（setup_failed）保留原件，在队尾最多重排队 1 次。

用法：
  python3 run_288_schedule.py                 # 跑全部 288
  python3 run_288_schedule.py --limit 3       # 只跑前 3 条（冒烟）
  python3 run_288_schedule.py --methods P0    # 只跑某方法
  python3 run_288_schedule.py --requeue       # 只做基础设施失败的重排队
  touch <out>/STOP                            # 当前 trial 结束后优雅停止
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path("/root/autodl-tmp/AUV")
DP3 = ROOT / "Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy"
TASKS_JSON = ROOT / "program/results/routea_gazebo_a1_tasks_run01/tasks.json"
WORLDS = ROOT / "PX4-Autopilot/Tools/sitl_gazebo/worlds/test"
RUNNER = ROOT / "catkin_ws0/scripts/run_task_flight.sh"
E5_CKPT = DP3 / "data/outputs/2026.08.16/08.53.26_e5_lambda0p5/checkpoints/latest.ckpt"
P6_DIR = DP3 / "data/outputs/dppo_uav_rebuild/routea_a2_selective_keep_run01/T/checkpoint_update_000256"
OUT = ROOT / "program/results/gazebo_3method_288_run01"
PLANNER_LOG_DIR = Path(os.path.expanduser("~/ml_planner_log"))

# 三方法统一的单次任务时限（秒）。三臂完全相同，不因结果好坏调整。
MISSION_TIMEOUT = 240
# 子进程硬上限 = 任务时限 + 固定开销（SITL 起、起飞、交接、清理）
HARD_KILL_AFTER = MISSION_TIMEOUT + 420

# 判为"基础设施失败"的退出码（run_task_flight.sh 的约定）
SETUP_RC = {3: "mavros_timeout", 5: "no_pos_cmd", 6: "no_pointcloud",
            7: "com_rcl_except_failed", 2: "asset_missing"}

METHOD_ENV = {
    "P0": {"ENGINE": "bc", "POLICY_TYPE": "p6_lora", "LORA_ENABLED": "false"},
    "P6": {"ENGINE": "bc", "POLICY_TYPE": "p6_lora", "LORA_ENABLED": "true"},
    "EGO": {"ENGINE": "ego"},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_bank():
    d = json.load(open(TASKS_JSON))
    by_id = {t["task"]["task_id"]: t for t in d["tasks"]}
    return d, by_id


def cond_dir(method: str, task_id: str, repeat: int) -> Path:
    return OUT / method / f"{task_id[:12]}_r{repeat}"


def build_cmd(task: dict, method: str, repeat: int, out_json: Path) -> list:
    p = task["task"]["payload"]
    wpath = WORLDS / f"{p['world_name']}.world"
    sx, sy, sz = p["start_pos"]
    gx, gy, gz = p["goal_pos"]
    return ["bash", str(RUNNER), str(wpath),
            f"{sx:.10f}", f"{sy:.10f}", f"{p['start_yaw']:.10f}", f"{sz:.10f}",
            f"{gx:.10f}", f"{gy:.10f}", f"{gz:.10f}",
            str(out_json), method, str(task["task"]["world_id"]), str(repeat)]


def build_env(method: str) -> dict:
    env = dict(os.environ)
    env.pop("PROF_LAUNCH", None)
    env.pop("AUV_PC_MODE", None)
    env.update({
        "CKPT": str(E5_CKPT),
        "TIMEOUT": str(MISSION_TIMEOUT),
        "SPAWN_Z": "0",
        "TAKEOFF_S": "30",
        "WITH_RVIZ": "0",
        "GOAL_FRAME": "world",
        "PYTHONUNBUFFERED": "1",
    })
    if method != "EGO":
        env["LORA_CKPT"] = str(P6_DIR)
    env.update(METHOD_ENV[method])
    return env


def collect_planner_logs(since_ts: float, dest: Path) -> list:
    """把 ~/ml_planner_log 下本次 trial 新写出的文件复制过去."""
    dest.mkdir(parents=True, exist_ok=True)
    got = []
    if not PLANNER_LOG_DIR.is_dir():
        return got
    for f in sorted(PLANNER_LOG_DIR.iterdir()):
        if not f.is_file():
            continue
        try:
            if f.stat().st_mtime < since_ts - 2:
                continue
        except OSError:
            continue
        shutil.copy2(f, dest / f.name)
        got.append(f.name)
    return got


def summarize(result: dict, rc: int) -> str:
    if "setup_failed" in result:
        return f"setup_failed:{result['setup_failed']}"
    if result.get("collision"):
        return "collision"
    if result.get("success_deploy"):
        return "reached"
    if result.get("timeout"):
        return "timeout"
    if result.get("lost"):
        return "lost"
    return "other"


def run_one(entry: dict, task: dict, attempt: int) -> dict:
    method, repeat = entry["method"], int(entry["repeat"])
    tid = entry["task_id"]
    d = cond_dir(method, tid, repeat)
    d.mkdir(parents=True, exist_ok=True)
    out_json = d / "result.json"

    # 保留失败原件：已有 result.json 且本次是重排队 → 改名留档
    if out_json.exists() and attempt > 1:
        prev = json.load(open(out_json))
        tag = summarize(prev, 0).replace(":", "_")
        shutil.move(str(out_json), str(d / f"result.attempt{attempt - 1}_{tag}.json"))
        for suf in ("_state.csv", "_pos_cmd.csv"):
            src = d / f"result.attempt{attempt - 1}{suf}"
            old = d / f"result{suf}"
            if old.exists():
                shutil.move(str(old), str(src))

    cmd = build_cmd(task, method, repeat, out_json)
    env = build_env(method)

    meta = {
        "method": method, "task_id": tid, "repeat": repeat, "attempt": attempt,
        "task_category": task.get("category"),
        "role": task["task"]["payload"].get("role"),
        "world_id": task["task"]["world_id"],
        "world_name": task["task"]["payload"]["world_name"],
        "world_file": str(WORLDS / f"{task['task']['payload']['world_name']}.world"),
        "start_pos": task["task"]["payload"]["start_pos"],
        "start_yaw": task["task"]["payload"]["start_yaw"],
        "goal_pos": task["task"]["payload"]["goal_pos"],
        "spawn_offset": task["world"].get("spawn_offset"),
        "obstacle_path": task["world"].get("obstacle_path"),
        "obstacle_sha256": task["world"].get("obstacle_sha256"),
        "mission_timeout_s": MISSION_TIMEOUT,
        "checkpoint": str(E5_CKPT),
        "lora_ckpt": None if method == "EGO" else str(P6_DIR),
        "lora_enabled": {"P0": False, "P6": True, "EGO": None}[method],
        "command": cmd,
        "env": {k: v for k, v in env.items() if k in
                ("CKPT", "TIMEOUT", "SPAWN_Z", "TAKEOFF_S", "WITH_RVIZ",
                 "GOAL_FRAME", "ENGINE", "POLICY_TYPE", "LORA_ENABLED", "LORA_CKPT")},
    }

    t0 = time.time()
    meta["started_at"] = datetime.fromtimestamp(t0).isoformat(timespec="seconds")
    log(f"▶ {method} {tid[:12]} r{repeat} a{attempt}  world={meta['world_name']}")
    try:
        proc = subprocess.run(cmd, env=env, cwd=str(ROOT),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              timeout=HARD_KILL_AFTER)
        rc = proc.returncode
        out_txt = proc.stdout.decode("utf-8", "replace")
    except subprocess.TimeoutExpired as e:
        rc = -9
        out_txt = (e.stdout or b"").decode("utf-8", "replace")
        meta["exception"] = f"subprocess_timeout>{HARD_KILL_AFTER}s"
        subprocess.run("pkill -9 -f gzserver; pkill -9 -f ml_planner; "
                       "pkill -9 -f px4ctrl; pkill -9 -f ego_planner; "
                       "pkill -9 -f fast_test.launch; pkill -9 -f px4_sitl_default",
                       shell=True)
        time.sleep(3)
    elapsed = time.time() - t0

    (d / "stdout.log").write_text(out_txt)

    if not out_json.exists():
        meta.update({"status": "no_result", "rc": rc, "elapsed_s": round(elapsed, 1)})
        if rc != 0:
            meta.setdefault("exception", SETUP_RC.get(rc, f"rc={rc}"))
    else:
        result = json.load(open(out_json))
        meta.update({"status": summarize(result, rc), "rc": rc,
                     "elapsed_s": round(elapsed, 1),
                     "success_deploy": result.get("success_deploy"),
                     "success_flightbench": result.get("success_flightbench"),
                     "collision": result.get("collision"),
                     "timeout": result.get("timeout"),
                     "final_dist_xy": result.get("final_dist_xy"),
                     "final_dist_z": result.get("final_dist_z"),
                     "min_clearance": result.get("min_clearance"),
                     "time_to_goal": result.get("time_to_goal")})
        if "setup_failed" in result:
            meta["setup_failed"] = result["setup_failed"]
        # 判为基础设施失败但脚本没写 setup_failed 的，用退出码兜底
        if rc in SETUP_RC and "setup_failed" not in result:
            meta["setup_failed"] = SETUP_RC[rc]

    meta["planner_logs"] = collect_planner_logs(t0, d / "planner_logs")
    (d / "condition.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    log(f"  ← {meta['status']}  rc={rc}  {elapsed:.0f}s  dxy={meta.get('final_dist_xy')}")
    return meta


def is_done(d: Path) -> bool:
    cj = d / "condition.json"
    if not cj.exists():
        return False
    try:
        meta = json.load(open(cj))
    except Exception:
        return False
    # 基础设施失败视为"未完成"，留给重排队阶段
    return not meta.get("setup_failed") and meta.get("status") not in (None, "no_result")


def write_run_config(bank: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "tasks_json": str(TASKS_JSON),
        "tasks_json_sha256": sha256(TASKS_JSON),
        "schema": bank.get("schema"),
        "order_seed": bank.get("order_seed"),
        "selection_seed": bank.get("selection_seed"),
        "schedule_entries": len(bank["schedule"]),
        "mission_timeout_s": MISSION_TIMEOUT,
        "hard_kill_after_s": HARD_KILL_AFTER,
        "runner": str(RUNNER),
        "runner_sha256": sha256(RUNNER),
        "checkpoints": {
            "P0_backbone_and_P6_backbone": {
                "path": str(E5_CKPT), "sha256": sha256(E5_CKPT)},
            "P6_lora_carrier": {
                "path": str(P6_DIR),
                "header": json.load(open(P6_DIR / "header.json")),
                "state_sha256": sha256(P6_DIR / "state.pt")},
        },
        "method_env": METHOD_ENV,
        "note": ("P0 与 P6 共用同一 backbone（state_dicts['model']，"
                 "由 load_native_e5 构造，identity=040a3378…）；"
                 "P0 = LoRA gate off（base 分支），P6 = gate on。"
                 "EGO 不加载任何扩散权重，goal 经 single_run_in_gazebo_task.launch 的 "
                 "point0_x/y/z 传入（原 launch 写死 16/0/2，命令行无法覆盖）。"),
    }
    (OUT / "run_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))


def expand_schedule(bank: dict, by_id: dict, methods=None, limit=None):
    rows = []
    for s in bank["schedule"]:
        if methods and s["method"] not in methods:
            continue
        rows.append(s)
        if limit and len(rows) >= limit:
            break
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--methods", default="")
    ap.add_argument("--requeue", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    methods = [m for m in args.methods.split(",") if m] or None
    bank, by_id = load_bank()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.requeue:
        write_run_config(bank)

    rows = expand_schedule(bank, by_id, methods, args.limit or None)
    log(f"schedule 展开 {len(rows)} 条  methods={methods or 'all'}  out={OUT}")

    if args.dry_run:
        for s in rows[:5]:
            t = by_id[s["task_id"]]
            print(json.dumps({"schedule": s, "cmd": build_cmd(
                t, s["method"], s["repeat"], Path("/tmp/x.json"))},
                ensure_ascii=False))
        return 0

    stop_file = OUT / "STOP"
    n_done = n_skip = n_fail = 0
    logf = open(OUT / "run_log.jsonl", "a")

    for i, s in enumerate(rows, 1):
        if stop_file.exists():
            log(f"检测到 {stop_file}，停止（已完成 {i - 1}/{len(rows)}）")
            break
        d = cond_dir(s["method"], s["task_id"], int(s["repeat"]))
        cj = d / "condition.json"
        attempt = 1
        if args.requeue:
            if not cj.exists():
                continue
            prev = json.load(open(cj))
            if not prev.get("setup_failed"):
                continue
            attempt = int(prev.get("attempt", 1)) + 1
            if attempt > 2:          # 每条最多重排队 1 次
                continue
        elif is_done(d):
            n_skip += 1
            continue
        task = by_id[s["task_id"]]
        log(f"进度 {i}/{len(rows)}")
        meta = run_one(s, task, attempt=attempt)
        logf.write(json.dumps(meta, ensure_ascii=False) + "\n")
        logf.flush()
        n_done += 1
        if meta.get("setup_failed") or meta["status"] == "no_result":
            n_fail += 1

    logf.close()
    log(f"本轮结束: 新跑 {n_done}（其中失败 {n_fail}），跳过已完成 {n_skip}")

    # 队尾：基础设施失败最多重排队 1 次
    retry = []
    for s in rows:
        d = cond_dir(s["method"], s["task_id"], int(s["repeat"]))
        cj = d / "condition.json"
        if not cj.exists():
            continue
        m = json.load(open(cj))
        if m.get("setup_failed") and int(m.get("attempt", 1)) < 2:
            retry.append(s)
    if retry:
        log(f"基础设施失败重排队 {len(retry)} 条（每条最多 1 次）")
        logf = open(OUT / "run_log.jsonl", "a")
        for s in retry:
            if stop_file.exists():
                break
            meta = run_one(s, by_id[s["task_id"]], attempt=2)
            logf.write(json.dumps(meta, ensure_ascii=False) + "\n")
            logf.flush()
        logf.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
