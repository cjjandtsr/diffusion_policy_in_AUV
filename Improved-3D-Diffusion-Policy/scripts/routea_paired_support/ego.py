"""EGO ROS service adapter for the unchanged common H8/H4 environment."""
from dataclasses import dataclass
from pathlib import Path
import math
import subprocess
import time

import numpy as np

from .core import canonical_hash, file_hash


def world_cloud(points, position, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    return (np.asarray(points, np.float64) @ rotation.T + position).astype(np.float32)


def reference_actions(reference, position, yaw):
    """Absolute world xyz/velocity/acceleration/yaw/rate -> existing delta API."""
    a = np.asarray(reference, np.float32).copy()
    if a.shape != (8, 11) or not np.isfinite(a).all():
        raise ValueError('EGO reference must be finite 8x11')
    a[:, :3] = np.diff(np.vstack((np.asarray(position, np.float32), a[:, :3])), axis=0)
    angles = np.diff(np.r_[yaw, a[:, 9].astype(np.float64)])
    a[:, 9] = np.arctan2(np.sin(angles), np.cos(angles))
    full = np.zeros((1, 16, 11), np.float32)
    full[0, 2:10] = a
    from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.contracts import ActionBundle
    return ActionBundle.from_full(full)


def native_yaw_step(target, previous, previous_rate, dt):
    """traj_server.calculate_yaw rate limit and branch-sensitive LPF."""
    pi = 3.1415926
    limit = pi*dt
    difference = target-previous
    if difference > pi:
        if difference-2*pi < -limit:
            yaw, rate = previous-limit, -pi
            if yaw < -pi: yaw += 2*pi
        else:
            yaw = target
            rate = -pi if yaw-previous > pi else (yaw-previous)/dt
    elif difference < -pi:
        if difference+2*pi > limit:
            yaw, rate = previous+limit, pi
            if yaw > pi: yaw -= 2*pi
        else:
            yaw = target
            rate = pi if yaw-previous < -pi else (yaw-previous)/dt
    elif difference < -limit:
        yaw, rate = previous-limit, -pi
        if yaw < -pi: yaw += 2*pi
    elif difference > limit:
        yaw, rate = previous+limit, pi
        if yaw > pi: yaw -= 2*pi
    else:
        yaw, rate = target, difference/dt
    if abs(yaw-previous) <= limit:
        yaw = .5*previous+.5*yaw
    return yaw, .5*previous_rate+.5*rate


def sample_reference(message, now, position, yaw, yaw_rate=0.):
    """Native cubic B-spline position and 100 Hz look-ahead yaw, copied locally.

    Sampling never advances ROS time or commits the unexecuted H8 yaw state.
    """
    from scipy.interpolate import BSpline
    points = np.array([[p.x, p.y, p.z] for p in message.pos_pts], np.float64)
    knots, degree = np.asarray(message.knots), int(message.order)
    spline = BSpline(knots, points, degree, extrapolate=False)
    velocity, acceleration = spline.derivative(), spline.derivative(2)
    begin, end = float(knots[degree]), float(knots[-degree-1])
    start = message.start_time.to_sec()
    out = []
    last_yaw, last_rate = float(yaw), float(yaw_rate)
    for tick in range(1, 241):
        t = now - start + tick * .01
        local = min(end, max(begin, begin + t))
        p = spline(local)
        if begin + t >= end:
            v, acc, rate = np.zeros(3), np.zeros(3), 0.
        else:
            v, acc = velocity(local), acceleration(local)
            direction = spline(min(end, local + 1.0)) - p
            target = math.atan2(direction[1], direction[0]) if np.linalg.norm(direction) > .1 else last_yaw
            last_yaw, rate = native_yaw_step(target,last_yaw,last_rate,.01)
        last_rate = rate
        if tick % 30 == 0:
            out.append(np.r_[p, v, acc, last_yaw, rate])
    return np.asarray(out, np.float32)


@dataclass
class EgoProvenance:
    raw_frame_hash: str
    episode_identity_hash: str
    decision_index: int
    policy_kind: str = 'ego'
    input_frame: str = 'body_to_world'


class EgoPolicy:
    use_base_policy = None
    policy_kind = 'ego'

    def __init__(self, launch, map_size, log_dir):
        self.launch = Path(launch).resolve()
        if not self.launch.is_file():
            raise FileNotFoundError(self.launch)
        self.map_size = tuple(float(x) for x in map_size)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.identity_hash = canonical_hash(dict(launch_sha256=file_hash(self.launch),
            map_size=self.map_size, execution='common_H4', sensor='4096_body_points'))
        self.process = self.stream = self.original_generate = None
        self.last_provenance = None

    def reset_episode(self, task, entry, env, observation, context):
        import rospy
        from ego_planner.srv import EvaluationStep
        self.close()
        if not rospy.core.is_initialized():
            rospy.init_node('paper_ego_client', anonymous=True, disable_signals=True)
        self.env, self.task, self.entry = env, task, entry
        self.elapsed, self.yaw_rate = 0., 0.
        self.frames, self.calls = [], []
        self.stream = (self.log_dir/(entry['key']+'.log')).open('a', encoding='utf-8')
        self.process = subprocess.Popen(['roslaunch', str(self.launch)] +
            ['map_size_'+axis+':='+str(value) for axis, value in zip('xyz', self.map_size)],
            stdout=self.stream, stderr=subprocess.STDOUT, start_new_session=True)
        service = '/drone_0_ego_planner_node/evaluation_step'
        rospy.wait_for_service(service, timeout=45.)
        self.service = rospy.ServiceProxy(service, EvaluationStep)
        self.runtime_parameters = rospy.get_param('/drone_0_ego_planner_node')
        self.response = self._call(observation['point_cloud'][-1], env._drone_pos,
                                   env._drone_yaw, env._drone_vel, 0., True)
        self.original_generate = env.synth_pc.generate_with_metadata

        def capture(*args, **kwargs):
            result = self.original_generate(*args, **kwargs)
            self.frames.append((np.asarray(args[0]).copy(), float(args[1]), result[0].copy()))
            return result
        env.synth_pc.generate_with_metadata = capture

    def _call(self, points, position, yaw, velocity, seconds, initialize=False,
              decision_boundary=False, acceleration=None):
        import rospy
        from std_msgs.msg import Header
        from sensor_msgs.point_cloud2 import create_cloud_xyz32
        from ego_planner.srv import EvaluationStepRequest
        request = EvaluationStepRequest()
        request.time_s = 1. + seconds
        request.initialize = initialize
        request.decision_boundary = decision_boundary
        if acceleration is not None:
            request.acceleration.x, request.acceleration.y, request.acceleration.z = map(float,acceleration)
        request.seed = int(self.entry['reset_seed']) & 0xffffffff
        request.goal.x, request.goal.y, request.goal.z = self.env.goal_w.tolist()
        odom = request.odometry
        odom.header = Header(stamp=rospy.Time.from_sec(request.time_s), frame_id='world')
        odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = map(float, position)
        odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = math.sin(yaw/2), math.cos(yaw/2)
        odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z = map(float, velocity)
        request.cloud = create_cloud_xyz32(odom.header, world_cloud(points, position, yaw))
        started = time.monotonic()
        # rospy services have no call timeout. Bound a stalled planner without
        # counting a transport failure as a navigation failure.
        import queue
        import threading
        result = queue.Queue(maxsize=1)
        def invoke():
            try:
                result.put((True, self.service(request)))
            except Exception as exc:
                result.put((False, exc))
        threading.Thread(target=invoke, daemon=True).start()
        try:
            ok, response = result.get(timeout=120.)
        except queue.Empty:
            raise TimeoutError('EGO service exceeded 120 wall seconds; preserve invalid attempt')
        if not ok:
            raise response
        self.calls.append(dict(time_s=seconds, trajectory_count=response.trajectory_count,
            failed_plans=response.failed_plans, planning_seconds=response.planning_seconds,
            service_seconds=time.monotonic()-started, state=response.state,
            has_trajectory=response.has_trajectory))
        return response

    def predict(self, observation, context, *, action_seed):
        from dataclasses import asdict
        meta = asdict(context['pointcloud_meta_history'][-1])
        self.last_provenance = EgoProvenance(meta['raw_array_hash'],
            context['episode_identity_hash'], int(context['decision_index']))
        if self.response.has_trajectory:
            reference = sample_reference(self.response.trajectory, 1.+self.elapsed,
                self.env._drone_pos, self.env._drone_yaw, self.yaw_rate)
        else:
            # No valid first trajectory: stay at the actual state while the
            # native FSM retries; the environment still consumes physical time.
            reference = np.zeros((8, 11), np.float32)
            reference[:, :3], reference[:, 9] = self.env._drone_pos, self.env._drone_yaw
        self.reference = reference
        self.frames.clear()
        return reference_actions(reference, self.env._drone_pos, self.env._drone_yaw)

    def after_step(self, context):
        duration = float(context['executed_time_s'])
        if not 1 <= len(self.frames) <= 4:
            raise ValueError('EGO did not capture the actual execution frames')
        for i, (position, yaw, points) in enumerate(self.frames):
            final = i == len(self.frames)-1
            seconds = self.elapsed + (duration if final else (i+1)*.3)
            velocity = self.env._drone_vel if final else self.reference[i, 3:6]
            # No new planning after a physical terminal event.
            if context['done_reason'] != 'none' and final:
                continue
            self.response = self._call(points, position, yaw, velocity, seconds,
                                       decision_boundary=final, acceleration=self.reference[i,6:9])
        self.elapsed += duration
        self.yaw_rate = float(self.reference[len(self.frames)-1, 10])
        result = dict(action_origin='ego_bspline', full_container_padding='outside [2:10] unused',
                      updates=list(self.calls), effective_execution='H4_committed')
        self.calls.clear()
        return result

    def close(self):
        if self.original_generate is not None:
            self.env.synth_pc.generate_with_metadata = self.original_generate
            self.original_generate = None
        if self.process is not None:
            import os
            import signal
            os.killpg(self.process.pid, signal.SIGINT) if self.process.poll() is None else None
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
            self.process = None
        if self.stream is not None:
            self.stream.close()
            self.stream = None
