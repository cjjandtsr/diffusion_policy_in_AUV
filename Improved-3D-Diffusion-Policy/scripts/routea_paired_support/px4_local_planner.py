"""Official PX4 waypoint generation under ideal H4 kinematic tracking."""
from dataclasses import asdict
import math
import time

import numpy as np

from .ego import EgoPolicy, EgoProvenance, reference_actions, world_cloud

# 2026-11-12 (user decision after the no-FC batch): the official planner's
# 1.2 s reference was executed by ideal kinematics at up to 3.0 m/s descent,
# while the official system's descent limit lives in the PX4 flight controller.
# local_planner.cpp never uses param_mpc_z_vel_max_dn for waypoint computation;
# avoidance_node.cpp:118,143 only reads it from the FC (full-tree grep), so the
# limiter is part of the platform, not the planner. These constants are the
# upstream defaults from setDefaultPx4Parameters() (local_planner.cpp:190-202).
PX4_Z_VEL_MAX_DN = 1.0   # MPC_Z_VEL_MAX_DN, m/s
PX4_Z_VEL_MAX_UP = 3.0   # MPC_Z_VEL_MAX_UP
PX4_ACC_HOR = 5.0        # MPC_ACC_HOR, m/s^2
PX4_ACC_UP = 10.0        # MPC_ACC_UP_MAX
PX4_ACC_DOWN = 10.0      # MPC_ACC_DOWN_MAX
PX4_XY_VEL_MAX = 3.0     # MPC_XY_CRUISE, the planner's commanded speed cap


def _px4_fc_limit_velocity(target, previous, dt):
    """One reference step of the PX4 position controller's documented limits.

    Emulates the missing flight-controller layer: vertical velocity is clamped
    to [-MPC_Z_VEL_MAX_DN, +MPC_Z_VEL_MAX_UP], the horizontal vector to the
    planner's cruise command, and the per-axis change to MPC_ACC_* * dt. The
    controller's internal P gains are not modelled; the limit is the documented
    one, applied to the commanded velocity.
    """
    v = np.asarray(target, dtype=np.float64).copy()
    v[2] = min(max(v[2], -PX4_Z_VEL_MAX_DN), PX4_Z_VEL_MAX_UP)
    horizontal = float(np.linalg.norm(v[:2]))
    if horizontal > PX4_XY_VEL_MAX:
        v[:2] *= PX4_XY_VEL_MAX / horizontal
    previous = np.asarray(previous, dtype=np.float64)
    dv_xy = v[:2] - previous[:2]
    limit_xy = PX4_ACC_HOR * dt
    norm_xy = float(np.linalg.norm(dv_xy))
    if norm_xy > limit_xy:
        dv_xy *= limit_xy / norm_xy
    dv_z = min(max(v[2] - previous[2], -PX4_ACC_DOWN * dt), PX4_ACC_UP * dt)
    return np.array([previous[0] + dv_xy[0], previous[1] + dv_xy[1], previous[2] + dv_z])


def waypoint_reference(response, position, yaw, speed, velocity=None):
    """Advance at the native cruise speed along the commanded direction for the
    whole decision window; retain velocity-mask meaning; execute the result
    through the PX4 flight-controller limits.

    2026-11-12 adaptation (user decision "参考按原生巡航速度推进整个决策窗"):
    the official local_planner flies a 10-30 Hz loop (local_planner_nodelet.cpp
    spin_dt_ = 0.1 s) in which the waypoint moves forward with the vehicle; the
    vehicle never has to stop at a queried waypoint. Evaluating one bridge call
    per 1.2 s decision with the earlier "interpolate to the waypoint and stop"
    rule instead restarted the reference from rest every decision, so the H4
    executor's 0.25 m/s^2 cap limited travel to ~0.18 m per decision and the
    measured executed speed fell to 0.055-0.31 m/s against the 0.5 m/s native
    limit (median displacement 0.173 m per 1.2 s decision). Advancing the
    reference at the native cruise_velocity along the current commanded
    direction for the full window restores the native operating speed. The
    cost, recorded as an adaptation deviation, is that the reference may pass
    beyond the queried waypoint within one window.

    2026-11-12 flight-controller emulation (user decision after the no-FC batch):
    the same full-window advance executed by ideal kinematics produced descents
    up to 3.0 m/s and 686/930 ground crashes, because PX4's descent limit
    (MPC_Z_VEL_MAX_DN = 1.0) is applied by the flight controller, which this
    evaluation replaces with ideal tracking. The window is now integrated
    through _px4_fc_limit_velocity, so the reference never descends faster than
    1.0 m/s and respects the documented acceleration limits. The planner's
    output (direction, cruise speed, setpoint) is unchanged.
    """
    position = np.asarray(position, dtype=np.float64)
    target = np.array([response.position.x, response.position.y, response.position.z])
    native_velocity = np.array([response.velocity.x, response.velocity.y, response.velocity.z])
    offset = np.where(np.isfinite(target), target-position, 0.)
    distance = float(np.linalg.norm(offset))
    times = np.arange(1, 9)*.3
    # Nominal per-step velocity before the flight-controller limits.
    nominal = np.zeros((times.size, 3), dtype=np.float64)
    if distance > 1e-9 and speed > 0:
        nominal[:] = speed*offset/distance
    # Native mixed control: velocity-controlled axes have no position target.
    for axis in range(3):
        if not np.isfinite(target[axis]):
            nominal[:, axis] = native_velocity[axis]
    previous_velocity = np.zeros(3) if velocity is None else np.asarray(velocity, dtype=np.float64)
    if not np.isfinite(previous_velocity).all():
        previous_velocity = np.zeros(3)
    limited = np.zeros_like(nominal)
    for k in range(times.size):
        previous_velocity = _px4_fc_limit_velocity(nominal[k], previous_velocity, .3)
        limited[k] = previous_velocity
    points = position + np.cumsum(limited*.3, axis=0)
    reference = np.zeros((8, 11), np.float32)
    reference[:, :3] = points
    reference[:, 3:6] = limited
    delta = math.atan2(math.sin(response.yaw-yaw), math.cos(response.yaw-yaw))
    if math.isfinite(response.yaw_rate) and response.yaw_rate != 0:
        yaw_travel = np.minimum(abs(response.yaw_rate)*times, abs(delta))
        reference[:, 9] = yaw+math.copysign(1., delta)*yaw_travel
    else:
        reference[:, 9] = yaw+delta
    reference[:, 10] = np.diff(np.r_[yaw, reference[:, 9]])/.3
    return reference


class Px4LocalPlannerPolicy(EgoPolicy):
    policy_kind = 'px4'
    service_module = 'paper_px4_bridge.srv'
    service_name = '/paper_px4_bridge/step'
    parameter_namespace = '/paper_px4_bridge'

    def _call(self, points, position, yaw, velocity, seconds, initialize=False,
              decision_boundary=False, acceleration=None):
        import rospy
        from paper_px4_bridge.srv import EvaluationStepRequest
        from std_msgs.msg import Header
        from sensor_msgs.point_cloud2 import create_cloud_xyz32
        request = EvaluationStepRequest(time_s=1.+seconds, initialize=initialize)
        request.goal.x, request.goal.y, request.goal.z = self.env.goal_w.tolist()
        request.odometry.header = Header(stamp=rospy.Time.from_sec(request.time_s), frame_id='world')
        p = request.odometry.pose.pose
        p.position.x, p.position.y, p.position.z = map(float, position)
        p.orientation.z, p.orientation.w = math.sin(yaw/2), math.cos(yaw/2)
        v = request.odometry.twist.twist.linear
        v.x, v.y, v.z = map(float, velocity)
        request.cloud = create_cloud_xyz32(request.odometry.header, world_cloud(points, position, yaw))
        started = time.monotonic()
        response = self.service(request)
        native_velocity = [response.velocity.x,response.velocity.y,response.velocity.z]
        native_position = [response.position.x,response.position.y,response.position.z]
        self.calls.append(dict(time_s=seconds, planning_seconds=response.planning_seconds,
            service_seconds=time.monotonic()-started,
            native_position=[v if math.isfinite(v) else None for v in native_position],
            native_velocity=[v if math.isfinite(v) else None for v in native_velocity],
            velocity_control_mask=[not math.isfinite(v) for v in native_position],
            native_yaw=response.yaw if math.isfinite(response.yaw) else None,
            native_yaw_rate=response.yaw_rate if math.isfinite(response.yaw_rate) else None,
            native_speed=response.native_speed,
            conversion='native_cruise_velocity_full_window_ideal_tracking'))
        return response

    def reset_episode(self, task, entry, env, observation, context):
        # Use the shared lifecycle but select the native PX4 service.
        super().reset_episode(task, entry, env, observation, context)
        # The FC limits are Python constants in this module, not ROS params, so
        # without this the trace header would not show the execution limits that
        # waypoint_reference actually applied.
        self.runtime_parameters = dict(self.runtime_parameters)
        self.runtime_parameters['px4_fc_limits'] = dict(
            mpc_z_vel_max_dn=PX4_Z_VEL_MAX_DN, mpc_z_vel_max_up=PX4_Z_VEL_MAX_UP,
            mpc_acc_hor=PX4_ACC_HOR, mpc_acc_up=PX4_ACC_UP, mpc_acc_down=PX4_ACC_DOWN,
            mpc_xy_vel_max=PX4_XY_VEL_MAX,
            source='scripts/routea_paired_support/px4_local_planner.py')

    def predict(self, observation, context, *, action_seed):
        meta = asdict(context['pointcloud_meta_history'][-1])
        self.last_provenance = EgoProvenance(meta['raw_array_hash'],
            context['episode_identity_hash'], int(context['decision_index']), policy_kind='px4')
        self.reference = waypoint_reference(self.response, self.env._drone_pos,
            self.env._drone_yaw, self.response.native_speed, self.env._drone_vel)
        self.frames.clear()
        return reference_actions(self.reference, self.env._drone_pos, self.env._drone_yaw)

    def after_step(self, context):
        result = super().after_step(context)
        result['action_origin'] = 'px4_native_waypoint'
        return result
