#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 8 月条件（固定 spawn/goal, layerA 20 world）三方法任务表。

条件（2026-11-08 核自 backup/ckpt_e5_20260819 与 run_batch.sh/fast_test.launch）:
  - spawn 固定: world 系 (-8, 0, 0), yaw=0 (fast_test.launch:6-11)
  - 起飞高度: 2.0 m
  - goal 固定: odom 系 (12, 0, 2) → world 系 (4, 0, 2)
  - world: ood_experiment_worlds/layerA/env_000_obs2 .. env_019_obs25 (20 个)
  - trials: 5/个 (同 8 月 trials_layerA.txt)
  - 超时: 1200 s (8 月口径; 8 月成功 trial median 452 s 全部 >240 s, 短超时不可比)
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

ROOT = Path("/root/autodl-tmp/AUV")
WORLD_DIR = ROOT / "PX4-Autopilot/Tools/sitl_gazebo/worlds/ood_experiment_worlds/layerA"
OUT = ROOT / "program/results/routea_gazebo_august_layerA_tasks.json"

# 8 月 trials_layerA.txt 的 world 顺序 (world_id 0..19, 各 5 trials)
LAYER_A_WORLDS = [
    "env_000_obs2", "env_001_obs10", "env_002_obs9", "env_003_obs3",
    "env_004_obs19", "env_005_obs9", "env_006_obs19", "env_007_obs27",
    "env_008_obs20", "env_009_obs17", "env_010_obs2", "env_011_obs23",
    "env_012_obs22", "env_013_obs9", "env_014_obs20", "env_015_obs21",
    "env_016_obs18", "env_017_obs21", "env_018_obs25", "env_019_obs25",
]
TRIALS_PER_WORLD = 5
SPAWN = (-8.0, 0.0, 0.0)   # world 系 (z=地面)
START_YAW = 0.0
START_Z = 2.0               # 起飞高度 (8 月首帧 z≈2.06)
GOAL_WORLD = (4.0, 0.0, 2.0)  # = odom (12,0,2) + spawn (-8,0)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    tasks = []
    for wid, wname in enumerate(LAYER_A_WORLDS):
        wpath = WORLD_DIR / f"{wname}.world"
        obs = WORLD_DIR / f"{wname}.obstacles.json"
        tasks.append({
            "world_id": wname,
            "world_idx": wid,
            "category": "layerA",
            "task": {
                "task_id": hashlib.sha256(wname.encode()).hexdigest(),
                "payload": {
                    "world_name": wname,
                    "start_pos": [SPAWN[0], SPAWN[1], START_Z],
                    "start_yaw": START_YAW,
                    "goal_pos": list(GOAL_WORLD),
                },
            },
            "world": {
                "world_path": str(wpath),
                "obstacle_path": str(obs),
                "obstacle_sha256": sha256(obs) if obs.exists() else None,
            },
        })

    # schedule: 3 方法 × 20 world × 5 trials = 300; 交错顺序 (同 288 的 method 维度在前)
    rng = random.Random(2026110801)
    schedule = []
    for wname in LAYER_A_WORLDS:
        tid = next(t["task"]["task_id"] for t in tasks
                   if t["task"]["payload"]["world_name"] == wname)
        for trial in range(TRIALS_PER_WORLD):
            for method in ("P0", "P6", "EGO"):
                schedule.append({"method": method, "repeat": trial + 1,
                                 "task_id": tid})
    rng.shuffle(schedule)

    bank = {
        "schema": "routea_gazebo_august_layerA_v1",
        "selection_seed": 2026110801,
        "order_seed": 2026110801,
        "repeats": TRIALS_PER_WORLD,
        "flights": len(schedule),
        "spawn": list(SPAWN), "start_yaw": START_YAW, "start_z": START_Z,
        "goal_world": list(GOAL_WORLD),
        "timeout_s": 1200,
        "note": ("8 月条件复刻: 固定 spawn(-8,0,0)/起飞 2.0m/goal odom(12,0,2), "
                 "layerA 20 world × 5 trials × 3 方法 = 300 flights"),
        "tasks": tasks,
        "schedule": schedule,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(bank, f, indent=2, ensure_ascii=False)
    print(f"生成 {OUT}: {len(tasks)} world, {len(schedule)} flights")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
