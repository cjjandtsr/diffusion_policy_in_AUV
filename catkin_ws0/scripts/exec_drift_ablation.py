#!/usr/bin/env python3
"""
exec_drift_ablation.py — Execution drift ablation via coarse-grained set_model_state
=====================================================================================
iDP3-UAV OOD Experiment — Task 3 (P1)

Implements D3 option (a): coarse-grained set_model_state at trajectory chunk
boundaries to separate "PX4 tracking error" from "environment generalization."

Principle:
  - Policy outputs 8-step trajectory chunk at ~3.3Hz
  - Deployment executes first N steps (deployment_action_steps, typically 2)
  - With ideal tracking: at each new chunk boundary, set_model_state places
    drone at planned position+velocity — eliminating inter-chunk PX4 drift
  - Without ideal tracking: PX4 tracking error accumulates across chunks

  SR(ideal_tracking) - SR(real_PX4)     = execution drift contribution
  SR(ideal_tracking, LayerA vs LayerB)   = true environment generalization

Coarse-grained design (Zread D3):
  "轨迹步间无 PX4 延迟，步内保持动力学约束"
  - Chunk boundary: set_model_state corrects accumulated error
  - Within chunk: PX4 controls normally, preserving inertial dynamics

Architecture:
  - Standalone ROS node, runs alongside ml_planner_gazebo.py
  - Subscribes to /planning/pos_cmd to detect new trajectory chunks
  - At each new chunk, applies set_model_state to place drone at
    the first planned state (position + velocity)
  - Records planned vs executed trajectory for analysis
  - Also works with collision_timeout_monitor for closed-loop testing

Usage:
  # Run as companion node during closed-loop testing
  rosrun ego_planner exec_drift_ablation.py --enable_correction

  # Record-only mode (measure PX4 drift without correction)
  rosrun ego_planner exec_drift_ablation.py --record_only

  # With specific drone model
  rosrun ego_planner exec_drift_ablation.py --drone_name iris --enable_correction
"""

import argparse
import csv
import json
import math
import os
import sys
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np
import rospy
from gazebo_msgs.srv import SetModelState, SetModelStateRequest
from gazebo_msgs.msg import ModelStates, ModelState
from geometry_msgs.msg import Pose, Twist, Point, Vector3
from nav_msgs.msg import Odometry
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ChunkRecord:
    """Record for one trajectory chunk (between two planner calls)."""
    chunk_id: int
    t_received: float
    planned_positions: List[Tuple[float, float, float]] = field(default_factory=list)
    planned_velocities: List[Tuple[float, float, float]] = field(default_factory=list)
    planned_yaws: List[float] = field(default_factory=list)
    correction_applied: bool = False
    correction_pos: Tuple[float, float, float] = (0, 0, 0)
    correction_vel: Tuple[float, float, float] = (0, 0, 0)
    # Executed (from odometry / model states)
    executed_positions: List[Tuple[float, float, float]] = field(default_factory=list)
    executed_velocities: List[Tuple[float, float, float]] = field(default_factory=list)
    # Drift metrics
    max_pos_error: float = 0.0
    mean_pos_error: float = 0.0
    max_vel_error: float = 0.0


@dataclass
class AblationResult:
    """Complete ablation run result."""
    n_chunks: int = 0
    n_corrections: int = 0
    total_pos_error: float = 0.0
    total_vel_error: float = 0.0
    mean_pos_error: float = 0.0
    mean_vel_error: float = 0.0
    max_pos_error: float = 0.0
    chunks: List[ChunkRecord] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# set_model_state helper
# ─────────────────────────────────────────────────────────────────────────────

def get_set_model_state_proxy(timeout: float = 10.0) -> rospy.ServiceProxy:
    rospy.wait_for_service("/gazebo/set_model_state", timeout=timeout)
    return rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)


def set_drone_state(
    proxy: rospy.ServiceProxy,
    drone_name: str,
    x: float, y: float, z: float,
    vx: float = 0.0, vy: float = 0.0, vz: float = 0.0,
    yaw: float = 0.0,
) -> bool:
    """
    Set drone pose AND velocity via /gazebo/set_model_state.
    """
    msg = ModelState()
    msg.model_name = drone_name
    msg.pose = Pose()
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.position.z = z
    msg.pose.orientation.w = math.cos(yaw / 2.0)
    msg.pose.orientation.z = math.sin(yaw / 2.0)
    msg.twist = Twist()
    msg.twist.linear.x = vx
    msg.twist.linear.y = vy
    msg.twist.linear.z = vz
    msg.reference_frame = "world"

    req = SetModelStateRequest(model_state=msg)
    try:
        resp = proxy(req)
        return resp.success
    except rospy.ServiceException as e:
        rospy.logerr(f"[DriftAblation] set_model_state failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Ablation node
# ─────────────────────────────────────────────────────────────────────────────

class ExecDriftAblation:
    """
    ROS node that applies coarse-grained set_model_state at trajectory
    chunk boundaries, and records planned vs executed trajectory.

    Two modes:
      - enable_correction=True: applies set_model_state at chunk start
      - enable_correction=False (record_only): only records drift, no correction
    """

    def __init__(
        self,
        drone_name: str = "iris",
        enable_correction: bool = True,
        correction_after_n_steps: int = 2,  # deployment_action_steps
        record_buffer_size: int = 1000,
    ):
        self.drone_name = drone_name
        self.enable_correction = enable_correction
        self.correction_after_n_steps = correction_after_n_steps

        # State
        self._lock = threading.Lock()
        self._drone_x = 0.0
        self._drone_y = 0.0
        self._drone_z = 0.0
        self._drone_vx = 0.0
        self._drone_vy = 0.0
        self._drone_vz = 0.0
        self._drone_yaw = 0.0

        # Chunk tracking
        self._chunk_id = 0
        self._last_chunk_time = 0.0
        self._pos_cmd_buffer: deque = deque(maxlen=record_buffer_size)
        self._executed_buffer: deque = deque(maxlen=record_buffer_size)

        # Results
        self.result = AblationResult()

        # set_model_state proxy (lazy init)
        self._set_pose_proxy = None

        # Subscribers
        # pos_cmd from ml_planner
        self._pos_cmd_sub = rospy.Subscriber(
            "/planning/pos_cmd", MultiDOFJointTrajectory,
            self._pos_cmd_cb, queue_size=3,
        )
        # Drone state
        self._model_sub = rospy.Subscriber(
            "/gazebo/model_states", ModelStates,
            self._model_states_cb, queue_size=1,
        )
        self._odom_sub = rospy.Subscriber(
            "/mavros/local_position/odom", Odometry,
            self._odom_cb, queue_size=1,
        )

    def _get_set_pose_proxy(self) -> rospy.ServiceProxy:
        if self._set_pose_proxy is None:
            self._set_pose_proxy = get_set_model_state_proxy()
        return self._set_pose_proxy

    def _model_states_cb(self, msg: ModelStates):
        try:
            idx = msg.name.index(self.drone_name)
        except ValueError:
            for alt in ["iris_iris", "uav"]:
                try:
                    idx = msg.name.index(alt)
                    break
                except ValueError:
                    continue
            else:
                return

        pose = msg.pose[idx]
        twist = msg.twist[idx]

        # Extract yaw from quaternion
        q = pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny, cosy)

        with self._lock:
            self._drone_x = pose.position.x
            self._drone_y = pose.position.y
            self._drone_z = pose.position.z
            self._drone_vx = twist.linear.x
            self._drone_vy = twist.linear.y
            self._drone_vz = twist.linear.z
            self._drone_yaw = yaw

        # Record executed position
        self._executed_buffer.append({
            "t": time.time(),
            "x": pose.position.x,
            "y": pose.position.y,
            "z": pose.position.z,
            "vx": twist.linear.x,
            "vy": twist.linear.y,
            "vz": twist.linear.z,
            "yaw": yaw,
        })

    def _odom_cb(self, msg: Odometry):
        with self._lock:
            self._drone_vx = msg.twist.twist.linear.x
            self._drone_vy = msg.twist.twist.linear.y
            self._drone_vz = msg.twist.twist.linear.z

    def _pos_cmd_cb(self, msg: MultiDOFJointTrajectory):
        """
        Receive new trajectory chunk from ML planner.

        On new chunk arrival:
        1. Record the planned trajectory
        2. If enable_correction, apply set_model_state to place drone
           at the first action's position + velocity
        """
        now = time.time()

        if not msg.points:
            return

        # Extract planned positions and velocities
        planned_pos = []
        planned_vel = []
        planned_yaw = []

        for pt in msg.points:
            if pt.transforms:
                t = pt.transforms[0]
                planned_pos.append((t.translation.x, t.translation.y, t.translation.z))
                if pt.velocities:
                    v = pt.velocities[0]
                    planned_vel.append((v.linear.x, v.linear.y, v.linear.z))
                else:
                    planned_vel.append((0.0, 0.0, 0.0))
                # Yaw from rotation
                q = t.rotation
                siny = 2.0 * (q.w * q.z + q.x * q.y)
                cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
                planned_yaw.append(math.atan2(siny, cosy))

        if not planned_pos:
            return

        self._chunk_id += 1
        chunk_id = self._chunk_id

        # Record planned trajectory
        self._pos_cmd_buffer.append({
            "t": now,
            "chunk_id": chunk_id,
            "positions": planned_pos,
            "velocities": planned_vel,
            "yaws": planned_yaw,
        })

        # Apply correction at chunk boundary (coarse-grained)
        if self.enable_correction and planned_pos:
            # Place drone at the first action's state
            target = planned_pos[0]
            tv = planned_vel[0] if planned_vel else (0.0, 0.0, 0.0)
            ty = planned_yaw[0] if planned_yaw else 0.0

            try:
                proxy = self._get_set_pose_proxy()
                ok = set_drone_state(
                    proxy, self.drone_name,
                    target[0], target[1], target[2],
                    tv[0], tv[1], tv[2],
                    ty,
                )
                if ok:
                    self.result.n_corrections += 1
                    rospy.logdebug(
                        f"[DriftAblation] Chunk {chunk_id}: correction applied "
                        f"to ({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f})"
                    )
                else:
                    rospy.logwarn(
                        f"[DriftAblation] Chunk {chunk_id}: correction FAILED"
                    )
            except Exception as e:
                rospy.logerr(f"[DriftAblation] Correction error: {e}")

        # Compute drift from previous chunk
        if chunk_id > 1 and self._pos_cmd_buffer:
            # Compare executed position at time of new chunk vs last planned
            # end position of previous chunk
            with self._lock:
                exec_pos = (self._drone_x, self._drone_y, self._drone_z)
                exec_vel = (self._drone_vx, self._drone_vy, self._drone_vz)

            # Last position of previous chunk's plan (before correction)
            if len(self._pos_cmd_buffer) >= 2:
                prev_plan = self._pos_cmd_buffer[-2]
                if prev_plan["positions"]:
                    last_planned = prev_plan["positions"][
                        min(self.correction_after_n_steps - 1,
                            len(prev_plan["positions"]) - 1)
                    ]
                    pos_err = math.hypot(
                        exec_pos[0] - last_planned[0],
                        exec_pos[1] - last_planned[1],
                        exec_pos[2] - last_planned[2],
                    )
                    self.result.total_pos_error += pos_err
                    self.result.n_chunks = max(self.result.n_chunks, chunk_id)
                    if pos_err > self.result.max_pos_error:
                        self.result.max_pos_error = pos_err
                    self.result.mean_pos_error = (
                        self.result.total_pos_error / self.result.n_chunks
                    )

    def get_drift_summary(self) -> dict:
        """Return current drift statistics."""
        with self._lock:
            return {
                "n_chunks": self.result.n_chunks,
                "n_corrections": self.result.n_corrections,
                "mean_pos_error": round(self.result.mean_pos_error, 4),
                "max_pos_error": round(self.result.max_pos_error, 4),
                "total_pos_error": round(self.result.total_pos_error, 3),
                "enable_correction": self.enable_correction,
            }

    def save_results(self, output_path: str):
        """Save drift records to JSON."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "enable_correction": self.enable_correction,
                "correction_after_n_steps": self.correction_after_n_steps,
                "drone_name": self.drone_name,
            },
            "summary": {
                "n_chunks": self.result.n_chunks,
                "n_corrections": self.result.n_corrections,
                "mean_pos_error": round(self.result.mean_pos_error, 4),
                "max_pos_error": round(self.result.max_pos_error, 4),
                "total_pos_error": round(self.result.total_pos_error, 3),
            },
            "pos_cmd_buffer": list(self._pos_cmd_buffer)[-100:],
            "executed_buffer": list(self._executed_buffer)[-1000:],
        }
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        rospy.loginfo(f"[DriftAblation] Results saved to {output_path}")

    def shutdown(self):
        self._pos_cmd_sub.unregister()
        self._model_sub.unregister()
        self._odom_sub.unregister()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Execution drift ablation via coarse-grained set_model_state"
    )
    parser.add_argument("--enable_correction", action="store_true", default=True,
                        help="Apply set_model_state at chunk boundaries")
    parser.add_argument("--record_only", action="store_true",
                        help="Only record drift, no correction "
                             "(overrides --enable_correction)")
    parser.add_argument("--drone_name", default="iris",
                        help="Drone model name in Gazebo")
    parser.add_argument("--correction_after_n_steps", type=int, default=2,
                        help="Number of deployment steps per chunk")
    parser.add_argument("--output_dir", default="/tmp/drift_ablation",
                        help="Output directory for results")
    parser.add_argument("--duration", type=float, default=0,
                        help="Run duration in seconds (0=indefinite)")
    args = parser.parse_args()

    if args.record_only:
        args.enable_correction = False

    rospy.init_node("exec_drift_ablation", anonymous=True)

    ablation = ExecDriftAblation(
        drone_name=args.drone_name,
        enable_correction=args.enable_correction,
        correction_after_n_steps=args.correction_after_n_steps,
    )

    mode = "CORRECTION" if args.enable_correction else "RECORD_ONLY"
    rospy.loginfo(
        f"[DriftAblation] Started in {mode} mode, "
        f"drone={args.drone_name}, "
        f"correction_after={args.correction_after_n_steps} steps"
    )

    # Run
    rate = rospy.Rate(1.0)  # print stats at 1Hz
    start_time = time.time()

    try:
        while not rospy.is_shutdown():
            summary = ablation.get_drift_summary()
            rospy.loginfo(
                f"[DriftAblation] chunks={summary['n_chunks']}, "
                f"corrections={summary['n_corrections']}, "
                f"mean_err={summary['mean_pos_error']:.3f}m, "
                f"max_err={summary['max_pos_error']:.3f}m"
            )

            if args.duration > 0 and (time.time() - start_time) > args.duration:
                break

            rate.sleep()
    except KeyboardInterrupt:
        pass
    finally:
        # Save results
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_suffix = "corrected" if args.enable_correction else "raw"
        output_path = os.path.join(args.output_dir, f"drift_{mode_suffix}_{ts}.json")
        ablation.save_results(output_path)
        ablation.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
