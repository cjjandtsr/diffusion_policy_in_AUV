#!/usr/bin/env python3
"""
phase_alpha_closed_loop_batch.py — Phase α 闭环考试批量运行脚本
=================================================================
按 Guide §7.2 验收标准，在 Layer A 缺口 world 上批量跑闭环 trial，
每个 trial 用 collision_timeout_monitor 记录完整指标。

输出: phase_alpha_closed_loop_results/ 目录下每个 trial 一个 JSON，
      + 汇总 summary.json

用法 (在服务器上):
  python phase_alpha_closed_loop_batch.py \
    --ckpt <Phase α ckpt 路径> \
    --worlds 4 11 12 18 \
    --trials_per_world 5 \
    --timeout 60

协同:
  - 每轮: 启动 Gazebo world → px4ctrl → ML planner → monitor → 记录
  - 需要 ml_gazebo_fm.sh 作为启动参考
  - collision_timeout_monitor.py 作为每 trial 的监控
"""

import argparse
import json
import os
import subprocess
import sys
import time
import signal
import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WS_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # catkin_ws0
WORLDS_DIR = "/root/autodl-tmp/AUV/outputs/ood_experiment_worlds/layerA"
MONITOR_SCRIPT = os.path.join(SCRIPT_DIR, "collision_timeout_monitor.py")

# Layer A gap worlds (Guide §7.2 指定的 B 类型缺口)
DEFAULT_WORLDS = [4, 11, 12, 18]

# Waypoint: per-world target (LOCAL frame, NED, origin = takeoff)
# 每个 world 的默认 goal: 正前方 12m, 高度 2m
DEFAULT_WAYPOINTS = {
    4:  (12.0, 0.0, 2.0),
    11: (12.0, 0.0, 2.0),
    12: (12.0, 0.0, 2.0),
    18: (12.0, 0.0, 2.0),
}


def get_world_path(world_id: int) -> str:
    """返回 .world 文件路径"""
    path = os.path.join(WORLDS_DIR, f"env_{world_id:03d}_obs*.world")
    matches = glob.glob(path)
    if not matches:
        raise FileNotFoundError(f"World file not found: {path}")
    return matches[0]


def get_obstacles_path(world_id: int) -> str:
    """返回 .obstacles.json 路径"""
    path = os.path.join(WORLDS_DIR, f"env_{world_id:03d}_obs*.obstacles.json")
    matches = glob.glob(path)
    if not matches:
        raise FileNotFoundError(f"Obstacles file not found: {path}")
    return matches[0]


def run_trial(
    ckpt_path: str,
    world_id: int,
    trial_idx: int,
    goal: tuple,
    timeout_s: float,
    output_dir: str,
    dry_run: bool = False,
) -> dict:
    """
    运行单次闭环 trial。

    流程:
      1. 启动 Gazebo (带指定 world)
      2. 启动 px4ctrl
      3. 启动 ML Planner (FM, 带 ckpt + waypoint)
      4. 运行 collision_timeout_monitor
      5. 停止所有进程
      6. 返回结果

    注意: 这是同步单 trial 模式。完整自动化需要更复杂的进程管理。
          当前版本提供每个 trial 的手动命令，由用户按序执行。
    """
    obstacles_json = get_obstacles_path(world_id)
    gx, gy, gz = goal
    output_file = os.path.join(output_dir, f"trial_w{world_id:02d}_t{trial_idx:02d}.json")

    cmd_monitor = [
        "rosrun", "ego_planner", "collision_timeout_monitor.py",
        "--obstacles_json", obstacles_json,
        "--goal_x", str(gx),
        "--goal_y", str(gy),
        "--goal_z", str(gz),
        "--timeout", str(timeout_s),
        "--world_id", str(world_id),
        "--trial", str(trial_idx),
        "--tier", "layerA_gap",
        "--filename", f"env_{world_id:03d}",
        "--output", output_file,
    ]

    return {
        "world_id": world_id,
        "trial": trial_idx,
        "goal": list(goal),
        "obstacles_json": obstacles_json,
        "output_file": output_file,
        "monitor_cmd": " ".join(cmd_monitor),
    }


def generate_protocol(
    ckpt_path: str,
    worlds: list,
    trials_per_world: int,
    timeout_s: float,
    output_dir: str,
) -> list:
    """生成完整考试 protocol (每 trial 的操作序列)"""
    os.makedirs(output_dir, exist_ok=True)
    protocol = []

    for world_id in worlds:
        goal = DEFAULT_WAYPOINTS.get(
            world_id, (12.0, 0.0, 2.0))
        world_file = get_world_path(world_id)

        for t in range(trials_per_world):
            trial_info = run_trial(
                ckpt_path, world_id, t, goal, timeout_s, output_dir)
            trial_info["world_file"] = world_file
            protocol.append(trial_info)

    return protocol


def print_manual_protocol(protocol: list, ckpt_path: str):
    """打印手动操作指南"""
    worlds = sorted(set(t["world_id"] for t in protocol))
    trials_per_world = len([t for t in protocol if t["world_id"] == worlds[0]])

    print("=" * 70)
    print("  Phase α 闭环考试 — 手动操作指南")
    print("=" * 70)
    print(f"  ckpt: {ckpt_path}")
    print(f"  worlds: {worlds}")
    print(f"  trials/world: {trials_per_world}")
    print(f"  total trials: {len(protocol)}")
    print("=" * 70)
    print()
    print("前置准备:")
    print("  1. 确保 Gazebo + PX4 环境正常")
    print("  2. catkin_make 并 source devel/setup.bash")
    print("  3. 将 Phase α ckpt 复制到服务器对应路径")
    print()
    print("每个 trial 的操作流程:")
    print()

    for i, t in enumerate(protocol):
        wid = t["world_id"]
        tid = t["trial"]
        gx, gy, gz = t["goal"]
        world_file = t["world_file"]

        print(f"{'─'*60}")
        print(f"  Trial {i+1}/{len(protocol)}: world={wid} trial={tid}")
        print(f"  Goal: ({gx}, {gy}, {gz})  World: {os.path.basename(world_file)}")
        print(f"{'─'*60}")
        print()
        print(f"  # 1. 启动 Gazebo (新 terminal)")
        print(f"  roslaunch px4 fast_test.launch world:={world_file}")
        print()
        print(f"  # 2. 启动 px4ctrl + ML Planner (FM)")
        print(f"  bash ~/AUV/catkin_ws0/scripts/ml_gazebo_fm.sh \\")
        print(f"    {ckpt_path} full 4 1.5 -1 -1")
        print()
        print(f"  # 3. 运行 monitor (新 terminal, 等 drone 起飞后)")
        print(f"  source ~/AUV/catkin_ws0/devel/setup.bash")
        print(f"  {t['monitor_cmd']}")
        print()
        print(f"  # 4. trial 结束后, 停止所有进程:")
        print(f"  pkill -f ml_planner; pkill -f px4ctrl; pkill -f roslaunch; pkill -f gzserver")
        print(f"  sleep 5")
        print()
        print(f"  # 5. 检查结果:")
        print(f"  cat {t['output_file']}")
        print()

    print("=" * 70)
    print("  考试完成后汇总:")
    print(f"  ls {os.path.dirname(protocol[0]['output_file'])}/*.json")
    print()
    print(f"  python {__file__} --summarize {os.path.dirname(protocol[0]['output_file'])}")
    print("=" * 70)


def summarize_results(results_dir: str):
    """汇总所有 trial 结果"""
    import glob as g
    files = sorted(g.glob(os.path.join(results_dir, "trial_*.json")))
    if not files:
        print(f"No trial JSONs found in {results_dir}")
        return

    summary = {
        "n_trials": len(files),
        "success_deploy": 0,
        "success_flightbench": 0,
        "collision": 0,
        "timeout": 0,
        "lost": 0,
        "min_clearances": [],
        "time_to_goal": [],
        "final_dist_xy": [],
        "per_world": {},
    }

    for f in files:
        with open(f) as fp:
            d = json.load(fp)
        wid = d.get("world_id", -1)
        if wid not in summary["per_world"]:
            summary["per_world"][wid] = {
                "n": 0, "success": 0, "collision": 0,
                "timeout": 0, "lost": 0, "clearances": [],
            }
        pw = summary["per_world"][wid]
        pw["n"] += 1
        if d.get("success_deploy"): summary["success_deploy"] += 1; pw["success"] += 1
        if d.get("collision"): summary["collision"] += 1; pw["collision"] += 1
        if d.get("timeout"): summary["timeout"] += 1; pw["timeout"] += 1
        if d.get("lost"): summary["lost"] += 1; pw["lost"] += 1
        if d.get("min_clearance") is not None:
            summary["min_clearances"].append(d["min_clearance"])
            pw["clearances"].append(d["min_clearance"])
        if d.get("time_to_goal"):
            summary["time_to_goal"].append(d["time_to_goal"])
        if d.get("final_dist_xy"):
            summary["final_dist_xy"].append(d["final_dist_xy"])

    # Print summary
    n = summary["n_trials"]
    print("=" * 60)
    print("  Phase α 闭环考试 — 汇总")
    print("=" * 60)
    print(f"  Trials: {n}")
    print(f"  Success (deploy): {summary['success_deploy']}/{n} = {summary['success_deploy']/n*100:.1f}%")
    print(f"  Success (FlightBench): {summary['success_flightbench']}/{n}")
    print(f"  Collision: {summary['collision']}/{n} = {summary['collision']/n*100:.1f}%")
    print(f"  Timeout: {summary['timeout']}/{n}")
    print(f"  Lost: {summary['lost']}/{n}")
    if summary["min_clearances"]:
        import numpy as np
        c = np.array(summary["min_clearances"])
        print(f"  min_clearance: p05={np.percentile(c,5):.3f}m, p50={np.median(c):.3f}m, min={c.min():.3f}m")
    if summary["time_to_goal"]:
        import numpy as np
        t = np.array(summary["time_to_goal"])
        print(f"  time_to_goal: mean={t.mean():.1f}s, p50={np.median(t):.1f}s")
    print()
    print("  Per world:")
    for wid in sorted(summary["per_world"].keys()):
        pw = summary["per_world"][wid]
        import numpy as np
        c_arr = np.array(pw["clearances"]) if pw["clearances"] else np.array([0])
        print(f"    w{wid:02d}: success={pw['success']}/{pw['n']}, "
              f"collision={pw['collision']}/{pw['n']}, "
              f"clearance p50={np.median(c_arr):.3f}m")
    print("=" * 60)

    # Save summary
    summary_path = os.path.join(results_dir, "summary.json")
    # Convert numpy arrays
    summary_serializable = {
        "n_trials": summary["n_trials"],
        "success_deploy": summary["success_deploy"],
        "success_flightbench": summary["success_flightbench"],
        "collision": summary["collision"],
        "timeout": summary["timeout"],
        "lost": summary["lost"],
        "min_clearances": [float(x) for x in summary["min_clearances"]],
        "time_to_goal": [float(x) for x in summary["time_to_goal"]],
        "final_dist_xy": [float(x) for x in summary["final_dist_xy"]],
        "per_world": {
            str(wid): {
                "n": pw["n"], "success": pw["success"],
                "collision": pw["collision"],
                "timeout": pw["timeout"], "lost": pw["lost"],
                "clearance_p50": float(
                    __import__("numpy").median(pw["clearances"])
                ) if pw["clearances"] else 0.0,
            }
            for wid, pw in summary["per_world"].items()
        }
    }
    with open(summary_path, "w") as f:
        json.dump(summary_serializable, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase α 闭环考试批量运行脚本"
    )
    parser.add_argument("--ckpt", required=True,
                        help="Phase α checkpoint 路径")
    parser.add_argument("--worlds", type=int, nargs="+",
                        default=DEFAULT_WORLDS,
                        help=f"测试 world ID 列表 (默认: {DEFAULT_WORLDS})")
    parser.add_argument("--trials_per_world", type=int, default=5,
                        help="每个 world 的 trial 数 (默认 5)")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="每 trial 超时秒数 (默认 60)")
    parser.add_argument("--output_dir", default=None,
                        help="输出目录 (默认: phase_alpha_closed_loop_results)")
    parser.add_argument("--summarize", default=None,
                        help="仅汇总已有结果目录")
    args = parser.parse_args()

    if args.summarize:
        summarize_results(args.summarize)
        return

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "phase_alpha_closed_loop_results")

    protocol = generate_protocol(
        args.ckpt, args.worlds, args.trials_per_world,
        args.timeout, output_dir)

    # Save protocol
    protocol_path = os.path.join(output_dir, "protocol.json")
    with open(protocol_path, "w") as f:
        json.dump(protocol, f, indent=2)
    print(f"Protocol saved to {protocol_path}\n")

    # Print manual guide
    print_manual_protocol(protocol, args.ckpt)


if __name__ == "__main__":
    main()
