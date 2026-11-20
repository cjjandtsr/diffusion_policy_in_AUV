#!/usr/bin/env python3
"""
closed_loop_runner.py — Batch closed-loop test runner
=====================================================
iDP3-UAV OOD Experiment — Task 2 (P1)

Orchestrates batch closed-loop testing across multiple worlds.
For each world, runs N trials with fixed seed ordering, resets drone
via set_model_state between trials, and records results via the
collision_timeout_monitor.

Architecture:
  - One Gazebo+PX4 instance per world batch
  - set_model_state to reset drone between trials (no Gazebo restart)
  - ML planner restarted per world (to reset waypoints/trajectory state)
  - Collision monitor spawned per trial
  - Dual-threshold SR reporting (deploy + FlightBench)
  - Results aggregated to CSV + JSON

Prerequisites:
  - ROS master running (roscore)
  - Gazebo + PX4 SITL started (e.g., roslaunch px4 fast_test.launch)
  - px4ctrl running
  - This script manages the ML planner and trial loop

Usage:
  # Single-world test (3 trials)
  python closed_loop_runner.py \
    --worlds_dir outputs/ood_experiment_worlds/layerA \
    --n_trials 20 --output results_layerA.csv

  # Multi-world with specific worlds
  python closed_loop_runner.py \
    --world_list env_000_obs9.world env_001_obs11.world \
    --worlds_dir outputs/ood_experiment_worlds/layerA \
    --n_trials 20

  # With custom goal (corner-case worlds have start/goal in metadata)
  python closed_loop_runner.py \
    --worlds_dir outputs/ood_experiment_worlds/corner_narrow_gap \
    --n_trials 10 --use_world_metadata_goal
"""

import argparse
import csv
import json
import math
import os
import signal
import subprocess
import sys
import time
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import rospy
from gazebo_msgs.srv import SetModelState, SetModelStateRequest
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose, Twist


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DRONE_MODEL_NAME = "iris"
SPAWN_WORLD_XY = (-12.0, 0.0)  # world-frame spawn coordinates
SPAWN_WORLD_Z = 0.1            # spawn height (ground level)
TAKEOFF_HEIGHT = 2.0           # hover height above ground
DEFAULT_GOAL = (12.0, 0.0, 2.0)
DEFAULT_TIMEOUT = 60.0         # seconds per trial
DRONE_COLLISION_RADIUS = 0.35  # F450 half-diagonal + safety margin


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrialRecord:
    world_id: int
    trial: int
    tier: str
    world_filename: str

    outcome: str  # "success_deploy", "success_fb", "collision", "timeout", "lost"
    time_to_goal: float
    min_clearance: float
    final_dist_xy: float
    final_dist_z: float
    final_vel_xy: float
    n_collisions: int

    success_deploy: bool = False
    success_flightbench: bool = False

    timestamp: str = ""


@dataclass
class WorldSummary:
    world_id: int
    tier: str
    world_filename: str
    n_trials: int

    n_success_deploy: int = 0
    n_success_fb: int = 0
    n_collision: int = 0
    n_timeout: int = 0
    n_lost: int = 0

    mean_time: float = 0.0
    mean_clearance: float = 0.0
    mean_final_dist: float = 0.0

    sr_deploy: float = 0.0
    sr_fb: float = 0.0

    trials: List[TrialRecord] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Gazebo helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_set_model_state_proxy() -> rospy.ServiceProxy:
    """Get a proxy to /gazebo/set_model_state."""
    rospy.wait_for_service("/gazebo/set_model_state", timeout=10.0)
    return rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)


def set_drone_pose(
    proxy: rospy.ServiceProxy,
    x: float, y: float, z: float,
    yaw: float = 0.0,
    zero_velocity: bool = True,
) -> bool:
    """
    Set drone pose via /gazebo/set_model_state.
    Returns True on success.
    """
    msg = ModelState()
    msg.model_name = DRONE_MODEL_NAME
    msg.pose = Pose()
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.position.z = z
    msg.pose.orientation.w = 1.0
    msg.pose.orientation.z = math.sin(yaw / 2.0)
    if zero_velocity:
        msg.twist = Twist()
    msg.reference_frame = "world"

    req = SetModelStateRequest(model_state=msg)
    try:
        resp = proxy(req)
        return resp.success
    except rospy.ServiceException as e:
        rospy.logerr(f"set_model_state failed: {e}")
        return False


def wait_for_drone(timeout: float = 15.0) -> bool:
    """Wait for drone to appear in /gazebo/model_states."""
    from gazebo_msgs.msg import ModelStates
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            msg = rospy.wait_for_message(
                "/gazebo/model_states", ModelStates, timeout=2.0
            )
            names = msg.name
            if DRONE_MODEL_NAME in names:
                return True
            for alt in ["iris_iris", "uav"]:
                if alt in names:
                    return True
        except rospy.ROSException:
            pass
    return False


# ─────────────────────────────────────────────────────────────────────────────
# World metadata
# ─────────────────────────────────────────────────────────────────────────────

def load_world_metadata(obstacles_json_path: str) -> dict:
    """Load .obstacles.json and extract start/goal metadata."""
    with open(obstacles_json_path) as f:
        data = json.load(f)

    goal = DEFAULT_GOAL
    start = (SPAWN_WORLD_XY[0], SPAWN_WORLD_XY[1], TAKEOFF_HEIGHT)

    if "goal_world" in data:
        g = data["goal_world"]
        goal = (float(g[0]), float(g[1]), float(g[2]))
    if "start_world" in data:
        s = data["start_world"]
        start = (float(s[0]), float(s[1]), float(s[2]))

    tier = data.get("tier", "unknown")
    difficulty = data.get("difficulty", {})

    return {
        "goal": goal,
        "start": start,
        "tier": tier,
        "difficulty": difficulty,
        "n_obstacles": data.get("n_obstacles", 0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Trial execution
# ─────────────────────────────────────────────────────────────────────────────

def run_single_trial(
    world_dir: str,
    world_filename: str,
    world_id: int,
    trial: int,
    tier: str,
    goal: Tuple[float, float, float],
    start: Tuple[float, float, float],
    timeout_s: float,
    drone_radius: float,
    set_pose_proxy: rospy.ServiceProxy,
    monitor_script: str,
) -> TrialRecord:
    """
    Run a single closed-loop trial.

    1. Reset drone to start via set_model_state
    2. Spawn collision_timeout_monitor as subprocess
    3. Wait for termination
    4. Parse result
    """
    rospy.loginfo(f"[Runner] Trial {world_id}/{trial}: resetting drone to {start}")

    # Step 1: Reset drone position
    sx, sy, sz = start
    ok = set_drone_pose(set_pose_proxy, sx, sy, sz)
    if not ok:
        rospy.logerr(f"[Runner] Failed to reset drone position for trial {trial}")
        return TrialRecord(
            world_id=world_id, trial=trial, tier=tier,
            world_filename=world_filename,
            outcome="error", time_to_goal=0, min_clearance=0,
            final_dist_xy=0, final_dist_z=0, final_vel_xy=0,
            n_collisions=0,
        )

    # Brief pause to let drone stabilize
    time.sleep(0.5)

    # Step 2: Build obstacles.json path
    base = world_filename.replace(".world", "")
    obstacles_json = os.path.join(world_dir, f"{base}.obstacles.json")
    if not os.path.exists(obstacles_json):
        rospy.logerr(f"[Runner] Obstacles JSON not found: {obstacles_json}")
        return TrialRecord(
            world_id=world_id, trial=trial, tier=tier,
            world_filename=world_filename,
            outcome="error", time_to_goal=0, min_clearance=0,
            final_dist_xy=0, final_dist_z=0, final_vel_xy=0,
            n_collisions=0,
        )

    # Step 3: Run collision monitor as subprocess
    output_json = f"/tmp/trial_{world_id:03d}_{trial:03d}.json"
    cmd = [
        "rosrun", "ego_planner", "collision_timeout_monitor.py",
        "--obstacles_json", obstacles_json,
        "--goal_x", str(goal[0]),
        "--goal_y", str(goal[1]),
        "--goal_z", str(goal[2]),
        "--drone_radius", str(drone_radius),
        "--timeout", str(timeout_s),
        "--output", output_json,
        "--world_id", str(world_id),
        "--trial", str(trial),
        "--tier", tier,
        "--filename", world_filename,
    ]

    rospy.loginfo(f"[Runner] Launching monitor: {' '.join(cmd)}")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s + 30)
    except subprocess.TimeoutExpired:
        rospy.logerr(f"[Runner] Monitor process timed out for trial {trial}")
        return TrialRecord(
            world_id=world_id, trial=trial, tier=tier,
            world_filename=world_filename,
            outcome="error", time_to_goal=timeout_s, min_clearance=0,
            final_dist_xy=0, final_dist_z=0, final_vel_xy=0,
            n_collisions=0,
        )

    # Step 4: Parse result
    if not os.path.exists(output_json):
        rospy.logerr(f"[Runner] Monitor output not found: {output_json}")
        rospy.logerr(f"[Runner] stdout: {proc.stdout[-500:]}")
        rospy.logerr(f"[Runner] stderr: {proc.stderr[-500:]}")
        return TrialRecord(
            world_id=world_id, trial=trial, tier=tier,
            world_filename=world_filename,
            outcome="error", time_to_goal=0, min_clearance=0,
            final_dist_xy=0, final_dist_z=0, final_vel_xy=0,
            n_collisions=0,
        )

    with open(output_json) as f:
        result_data = json.load(f)

    # Determine outcome
    if result_data.get("collision"):
        outcome = "collision"
    elif result_data.get("timeout"):
        outcome = "timeout"
    elif result_data.get("lost"):
        outcome = "lost"
    elif result_data.get("success_deploy"):
        outcome = "success_deploy"
    elif result_data.get("success_flightbench"):
        outcome = "success_fb"
    else:
        outcome = "unknown"

    record = TrialRecord(
        world_id=world_id,
        trial=trial,
        tier=tier,
        world_filename=world_filename,
        outcome=outcome,
        time_to_goal=result_data.get("time_to_goal", 0),
        min_clearance=result_data.get("min_clearance", float("inf")),
        final_dist_xy=result_data.get("final_dist_xy", 0),
        final_dist_z=result_data.get("final_dist_z", 0),
        final_vel_xy=result_data.get("final_vel_xy", 0),
        n_collisions=result_data.get("n_collisions", 0),
        success_deploy=result_data.get("success_deploy", False),
        success_flightbench=result_data.get("success_flightbench", False),
        timestamp=datetime.now().isoformat(),
    )

    # Cleanup temp file
    try:
        os.remove(output_json)
    except OSError:
        pass

    return record


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch closed-loop test runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test Layer A worlds
  python closed_loop_runner.py \\
    --worlds_dir outputs/ood_experiment_worlds/layerA \\
    --n_trials 20 --output results_layerA

  # Test corner-case (use per-world start/goal from metadata)
  python closed_loop_runner.py \\
    --worlds_dir outputs/ood_experiment_worlds/corner_narrow_gap \\
    --n_trials 10 --use_world_metadata_goal --output results_narrow_gap
        """,
    )

    parser.add_argument("--worlds_dir", required=True,
                        help="Directory containing .world files + .obstacles.json")
    parser.add_argument("--world_list", nargs="*", default=None,
                        help="Specific .world files (default: all .world in dir)")
    parser.add_argument("--n_trials", type=int, default=20,
                        help="Number of trials per world")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="Timeout per trial (seconds)")
    parser.add_argument("--drone_radius", type=float, default=DRONE_COLLISION_RADIUS,
                        help="Drone collision radius (meters)")
    parser.add_argument("--use_world_metadata_goal", action="store_true",
                        help="Use start/goal from .obstacles.json metadata")
    parser.add_argument("--goal_x", type=float, default=DEFAULT_GOAL[0])
    parser.add_argument("--goal_y", type=float, default=DEFAULT_GOAL[1])
    parser.add_argument("--goal_z", type=float, default=DEFAULT_GOAL[2])
    parser.add_argument("--start_x", type=float, default=SPAWN_WORLD_XY[0])
    parser.add_argument("--start_y", type=float, default=SPAWN_WORLD_XY[1])
    parser.add_argument("--start_z", type=float, default=TAKEOFF_HEIGHT)
    parser.add_argument("--output", default="results",
                        help="Output basename (generates .csv + .json)")
    parser.add_argument("--monitor_script", default=None,
                        help="Path to collision_timeout_monitor.py "
                             "(default: auto-find)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Validate worlds but don't run trials")
    args = parser.parse_args()

    # Find monitor script
    if args.monitor_script:
        monitor_script = args.monitor_script
    else:
        # Auto-find next to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        monitor_script = os.path.join(script_dir, "collision_timeout_monitor.py")

    if not os.path.exists(monitor_script):
        print(f"ERROR: Monitor script not found: {monitor_script}")
        print("Specify with --monitor_script or place in same directory")
        return 1

    # Discover worlds
    world_dir = args.worlds_dir
    if args.world_list:
        world_files = args.world_list
    else:
        world_files = sorted([
            f for f in os.listdir(world_dir)
            if f.endswith(".world")
        ])

    if not world_files:
        print(f"No .world files found in {world_dir}")
        return 1

    print(f"Closed-Loop Runner")
    print(f"{'='*60}")
    print(f"Worlds dir: {world_dir}")
    print(f"Worlds: {len(world_files)}")
    print(f"Trials per world: {args.n_trials}")
    print(f"Total trials: {len(world_files) * args.n_trials}")
    print(f"Timeout: {args.timeout}s")
    print(f"Monitor: {monitor_script}")
    print()

    if args.dry_run:
        print("DRY RUN — validating world files only")
        for wf in world_files:
            base = wf.replace(".world", "")
            json_path = os.path.join(world_dir, f"{base}.obstacles.json")
            if os.path.exists(json_path):
                meta = load_world_metadata(json_path)
                print(f"  ✓ {wf}: tier={meta['tier']}, "
                      f"obs={meta['n_obstacles']}, goal={meta['goal']}")
            else:
                print(f"  ✗ {wf}: missing .obstacles.json")
        return 0

    # Initialize ROS
    rospy.init_node("closed_loop_runner", anonymous=True)

    # Get set_model_state proxy
    try:
        set_pose_proxy = get_set_model_state_proxy()
        rospy.loginfo("[Runner] Connected to /gazebo/set_model_state")
    except Exception as e:
        rospy.logerr(f"[Runner] Cannot connect to /gazebo/set_model_state: {e}")
        rospy.logerr("Is Gazebo running?")
        return 1

    # Wait for drone
    if not wait_for_drone():
        rospy.logerr("[Runner] Drone not found in Gazebo. Is PX4 SITL running?")
        return 1
    rospy.loginfo("[Runner] Drone confirmed in Gazebo")

    # ── Run trials ──
    all_summaries = []
    default_goal = (args.goal_x, args.goal_y, args.goal_z)
    default_start = (args.start_x, args.start_y, args.start_z)

    for wi, wf in enumerate(world_files):
        base = wf.replace(".world", "")
        json_path = os.path.join(world_dir, f"{base}.obstacles.json")

        if not os.path.exists(json_path):
            print(f"  [{wi}] SKIP {wf}: missing .obstacles.json")
            continue

        meta = load_world_metadata(json_path)
        goal = meta["goal"] if args.use_world_metadata_goal else default_goal
        start = meta["start"] if args.use_world_metadata_goal else default_start
        tier = meta["tier"]

        print(f"\n{'─'*60}")
        print(f"World {wi+1}/{len(world_files)}: {wf}")
        print(f"  Tier: {tier}, Obstacles: {meta['n_obstacles']}")
        print(f"  Goal: {goal}, Start: {start}")
        print(f"{'─'*60}")

        summary = WorldSummary(
            world_id=wi,
            tier=tier,
            world_filename=wf,
            n_trials=args.n_trials,
        )

        for trial in range(args.n_trials):
            record = run_single_trial(
                world_dir=world_dir,
                world_filename=wf,
                world_id=wi,
                trial=trial,
                tier=tier,
                goal=goal,
                start=start,
                timeout_s=args.timeout,
                drone_radius=args.drone_radius,
                set_pose_proxy=set_pose_proxy,
                monitor_script=monitor_script,
            )

            summary.trials.append(record)

            # Update counts
            if record.outcome == "success_deploy":
                summary.n_success_deploy += 1
                summary.n_success_fb += 1
            elif record.outcome == "success_fb":
                summary.n_success_fb += 1
            elif record.outcome == "collision":
                summary.n_collision += 1
            elif record.outcome == "timeout":
                summary.n_timeout += 1
            elif record.outcome == "lost":
                summary.n_lost += 1

            # Running stats
            status = "✓" if record.success_deploy else (
                "○" if record.success_flightbench else "✗"
            )
            print(f"  [{trial+1:3d}/{args.n_trials}] {status} "
                  f"outcome={record.outcome:15s} "
                  f"T={record.time_to_goal:5.1f}s "
                  f"clr={record.min_clearance:5.2f}m "
                  f"dxy={record.final_dist_xy:4.2f}m")

        # Compute summary stats
        n = summary.n_trials
        if n > 0:
            summary.sr_deploy = summary.n_success_deploy / n
            summary.sr_fb = summary.n_success_fb / n
            t_times = [t.time_to_goal for t in summary.trials]
            t_clear = [t.min_clearance for t in summary.trials
                       if t.min_clearance < float("inf")]
            t_dist = [t.final_dist_xy for t in summary.trials]
            summary.mean_time = sum(t_times) / len(t_times)
            summary.mean_clearance = sum(t_clear) / len(t_clear) if t_clear else 0
            summary.mean_final_dist = sum(t_dist) / len(t_dist)

        all_summaries.append(summary)

        print(f"  ── World Summary ──")
        print(f"  SR_deploy={summary.sr_deploy:.1%} "
              f"SR_fb={summary.sr_fb:.1%} "
              f"collision={summary.n_collision/n:.1%} "
              f"timeout={summary.n_timeout/n:.1%} "
              f"mean_T={summary.mean_time:.1f}s "
              f"mean_clr={summary.mean_clearance:.2f}m")

    # ── Write outputs ──
    csv_path = f"{args.output}.csv"
    json_path = f"{args.output}.json"

    # CSV: one row per trial
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "world_id", "trial", "tier", "world_filename",
            "outcome", "success_deploy", "success_flightbench",
            "time_to_goal", "min_clearance", "final_dist_xy",
            "final_dist_z", "final_vel_xy", "n_collisions",
        ])
        writer.writeheader()
        for summary in all_summaries:
            for t in summary.trials:
                writer.writerow({
                    "world_id": t.world_id,
                    "trial": t.trial,
                    "tier": t.tier,
                    "world_filename": t.world_filename,
                    "outcome": t.outcome,
                    "success_deploy": t.success_deploy,
                    "success_flightbench": t.success_flightbench,
                    "time_to_goal": round(t.time_to_goal, 2),
                    "min_clearance": round(t.min_clearance, 3),
                    "final_dist_xy": round(t.final_dist_xy, 3),
                    "final_dist_z": round(t.final_dist_z, 3),
                    "final_vel_xy": round(t.final_vel_xy, 3),
                    "n_collisions": t.n_collisions,
                })

    # JSON: full summary with per-world stats
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "worlds_dir": world_dir,
            "n_worlds": len(world_files),
            "n_trials_per_world": args.n_trials,
            "timeout_s": args.timeout,
            "drone_radius": args.drone_radius,
        },
        "summaries": [],
        "aggregate": {},
    }

    for s in all_summaries:
        output_data["summaries"].append({
            "world_id": s.world_id,
            "tier": s.tier,
            "world_filename": s.world_filename,
            "n_trials": s.n_trials,
            "n_success_deploy": s.n_success_deploy,
            "n_success_fb": s.n_success_fb,
            "n_collision": s.n_collision,
            "n_timeout": s.n_timeout,
            "n_lost": s.n_lost,
            "sr_deploy": round(s.sr_deploy, 4),
            "sr_fb": round(s.sr_fb, 4),
            "mean_time_to_goal": round(s.mean_time, 2),
            "mean_min_clearance": round(s.mean_clearance, 3),
            "mean_final_dist_xy": round(s.mean_final_dist, 3),
        })

    # Aggregate by tier
    from collections import defaultdict
    tier_agg = defaultdict(lambda: {
        "n_trials": 0, "n_success_deploy": 0, "n_success_fb": 0,
        "n_collision": 0, "n_timeout": 0, "n_lost": 0,
    })
    for s in all_summaries:
        a = tier_agg[s.tier]
        a["n_trials"] += s.n_trials
        a["n_success_deploy"] += s.n_success_deploy
        a["n_success_fb"] += s.n_success_fb
        a["n_collision"] += s.n_collision
        a["n_timeout"] += s.n_timeout
        a["n_lost"] += s.n_lost

    output_data["aggregate"] = {
        tier: {
            **counts,
            "sr_deploy": round(counts["n_success_deploy"] / counts["n_trials"], 4)
            if counts["n_trials"] > 0 else 0,
            "sr_fb": round(counts["n_success_fb"] / counts["n_trials"], 4)
            if counts["n_trials"] > 0 else 0,
        }
        for tier, counts in tier_agg.items()
    }

    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)

    # ── Final report ──
    print(f"\n{'='*60}")
    print(f"FINAL REPORT")
    print(f"{'='*60}")
    print(f"Worlds tested: {len(all_summaries)}")
    print(f"Total trials: {sum(s.n_trials for s in all_summaries)}")
    print()
    print(f"{'Tier':30s} {'Worlds':>6s} {'SR_deploy':>10s} {'SR_fb':>10s} "
          f"{'Coll%':>7s} {'TO%':>7s}")
    print(f"{'─'*30} {'─'*6} {'─'*10} {'─'*10} {'─'*7} {'─'*7}")
    for tier, a in sorted(tier_agg.items()):
        n = a["n_trials"]
        if n == 0:
            continue
        print(f"{tier:30s} {a['n_trials']:6d} "
              f"{a['n_success_deploy']/n:10.1%} "
              f"{a['n_success_fb']/n:10.1%} "
              f"{a['n_collision']/n:7.1%} "
              f"{a['n_timeout']/n:7.1%}")

    print(f"\nResults: {csv_path} + {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
