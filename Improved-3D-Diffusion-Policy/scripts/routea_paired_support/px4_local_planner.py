"""Official PX4 waypoint generation under ideal H4 kinematic tracking."""
from dataclasses import asdict
import math
import time

import numpy as np

from .ego import EgoPolicy, EgoProvenance, reference_actions, world_cloud


def waypoint_reference(response, position, yaw, speed):
    """Time-resample the native position target; retain velocity-mask meaning."""
    position = np.asarray(position, dtype=np.float64)
    target = np.array([response.position.x, response.position.y, response.position.z])
    velocity = np.array([response.velocity.x, response.velocity.y, response.velocity.z])
    offset = np.where(np.isfinite(target), target-position, 0.)
    distance = np.linalg.norm(offset)
    travel = distance/speed if distance and speed > 0 else float('inf')
    times = np.arange(1, 9)*.3
    fraction = np.minimum(times/max(travel, .000001), 1.)
    points = position + fraction[:, None]*offset
    # Native mixed control: velocity-controlled axes have no position target.
    for axis in range(3):
        if not np.isfinite(target[axis]):
            points[:, axis] = position[axis]+times*velocity[axis]
    reference = np.zeros((8, 11), np.float32)
    reference[:, :3] = points
    reference[:, 3:6] = np.diff(np.vstack([position, points]), axis=0)/.3
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
            conversion='native_waypoint_linear_time_resampling_ideal_tracking'))
        return response

    def reset_episode(self, task, entry, env, observation, context):
        # Use the shared lifecycle but select the native PX4 service.
        super().reset_episode(task, entry, env, observation, context)

    def predict(self, observation, context, *, action_seed):
        meta = asdict(context['pointcloud_meta_history'][-1])
        self.last_provenance = EgoProvenance(meta['raw_array_hash'],
            context['episode_identity_hash'], int(context['decision_index']), policy_kind='px4')
        self.reference = waypoint_reference(self.response, self.env._drone_pos,
            self.env._drone_yaw, self.response.native_speed)
        self.frames.clear()
        return reference_actions(self.reference, self.env._drone_pos, self.env._drone_yaw)

    def after_step(self, context):
        result = super().after_step(context)
        result['action_origin'] = 'px4_native_waypoint'
        return result
