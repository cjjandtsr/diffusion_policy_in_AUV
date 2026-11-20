"""SUPER native polynomial reference under the common H8/H4 protocol.

The native planner runs in an isolated catkin workspace (paper_super_ws) and
is driven synchronously by the paper_super_sync service: one call per executed
0.3 s frame, one replan per 1.2 s decision boundary, logical time carried by
the request. The committed 7th-order polynomial trajectory is sampled here
with the delivered scripts/routea_paired_support/super_reference.py; the path
is never reshaped.
"""
from dataclasses import asdict
from pathlib import Path
import math
import subprocess
import time

import numpy as np

from . import super_reference as _native
from .core import canonical_hash, file_hash
from .ego import EgoPolicy, EgoProvenance, reference_actions, world_cloud


def sample_reference(message, now, position=None, yaw=None, yaw_rate=0.):
    """Committed native trajectory -> 8x11 rows at 0.3 s steps.

    Position (and yaw, when the official system generated a yaw trajectory)
    uses the delivered super_reference.py polynomial parsing verbatim. When no
    yaw trajectory exists (goal yaw disabled), the upstream command path holds
    the last yaw (SuperPlanner::getOneCommandFromTraj), so the yaw column is
    the native current yaw with zero rate, matching that upstream semantics.
    """
    if getattr(message, 'coef_yaw', None):
        return _native.sample_reference(message, now)
    rows = np.zeros((8, 11), np.float32)
    elapsed = now - message.start_WT_pos.to_sec()
    for step in range(1, 9):
        t = elapsed + step * .3
        xyz = np.array([_native.polynomial_state(c, message.time_pos,
                                                 message.order_pos, t)
                        for c in (message.coef_pos_x, message.coef_pos_y,
                                  message.coef_pos_z)])
        rows[step - 1, :3] = xyz[:, 0]
        rows[step - 1, 3:6] = xyz[:, 1]
        rows[step - 1, 6:9] = xyz[:, 2]
    rows[:, 9] = float(message.native_yaw)
    return rows


class SuperPolicy(EgoPolicy):
    policy_kind = 'super'
    service_module = 'paper_super_sync.srv'
    service_name = '/paper_super_sync/step'
    parameter_namespace = '/paper_super_sync'
    reference_sampler = staticmethod(sample_reference)

    def __init__(self, launch, map_size, log_dir, config_path=None):
        self.config_path = None if config_path is None else Path(config_path).resolve()
        if self.config_path is None or not self.config_path.is_file():
            raise FileNotFoundError('SUPER requires a generated sync config (--super-config)')
        super().__init__(launch, map_size, log_dir)
        base_yaml = self.config_path.with_suffix(self.config_path.suffix + '.base.yaml')
        if not base_yaml.is_file():
            raise FileNotFoundError('missing base config beside the sync config: ' + str(base_yaml))
        self.identity_hash = canonical_hash(dict(base=self.identity_hash,
            super_config_sha256=file_hash(self.config_path),
            super_config_base_sha256=file_hash(base_yaml)))

    def _call(self, points, position, yaw, velocity, seconds, initialize=False,
              decision_boundary=False, acceleration=None):
        import rospy
        import importlib
        from std_msgs.msg import Header
        from sensor_msgs.point_cloud2 import create_cloud_xyz32
        EvaluationStepRequest = importlib.import_module(self.service_module).EvaluationStepRequest
        request = EvaluationStepRequest()
        request.time_s = 1. + seconds
        request.initialize = initialize
        request.decision_boundary = decision_boundary
        request.seed = int(self.entry['reset_seed']) & 0xffffffff
        request.goal.x, request.goal.y, request.goal.z = self.env.goal_w.tolist()
        odom = request.odometry
        odom.header = Header(stamp=rospy.Time.from_sec(request.time_s), frame_id='world')
        odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = map(float, position)
        odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z = map(float, velocity)
        request.cloud = create_cloud_xyz32(odom.header, world_cloud(points, position, yaw))
        started = time.monotonic()
        # Same bounded synchronous call as EgoPolicy._call: a stalled planner is
        # a transport failure (120 s wall timeout), not a navigation failure.
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
            raise TimeoutError('SUPER service exceeded 120 wall seconds; preserve invalid attempt')
        if not ok:
            raise response
        self.calls.append(dict(time_s=seconds, state=response.state,
            has_trajectory=response.has_trajectory,
            trajectory_flag=response.trajectory_flag,
            planning_seconds=response.planning_seconds,
            service_seconds=time.monotonic() - started,
            ret_code=response.ret_code, ret_code_str=response.ret_code_str,
            replan_count=response.replan_count, failed_plans=response.failed_plans,
            state_iterations=response.state_iterations, traj_finish=response.traj_finish,
            requested_goal=[response.requested_goal.x, response.requested_goal.y,
                            response.requested_goal.z],
            native_goal=[response.native_goal.x, response.native_goal.y,
                         response.native_goal.z],
            goal_accepted=response.goal_accepted, goal_adjusted=response.goal_adjusted,
            native_yaw=response.native_yaw if math.isfinite(response.native_yaw) else None,
            native_yaw_dot=response.native_yaw_dot if math.isfinite(response.native_yaw_dot) else None,
            module_seconds=list(response.module_seconds),
            conversion='native_polynomial_7th_order_common_H4'))
        return response

    def reset_episode(self, task, entry, env, observation, context):
        import rospy
        import importlib
        EvaluationStep = importlib.import_module(self.service_module).EvaluationStep
        self.close()
        if not rospy.core.is_initialized():
            rospy.init_node('paper_super_client', anonymous=True, disable_signals=True)
        self.env, self.task, self.entry = env, task, entry
        self.elapsed, self.yaw_rate = 0., 0.
        self.frames, self.calls = [], []
        self.stream = (self.log_dir / (entry['key'] + '.log')).open('a', encoding='utf-8')
        self.process = subprocess.Popen(['roslaunch', str(self.launch),
            'config_path:=' + str(self.config_path)],
            stdout=self.stream, stderr=subprocess.STDOUT, start_new_session=True)
        service = self.service_name
        rospy.wait_for_service(service, timeout=45.)
        self.service = rospy.ServiceProxy(service, EvaluationStep, persistent=True)
        self.runtime_parameters = rospy.get_param(self.parameter_namespace)
        self.response = self._call(observation['point_cloud'][-1], env._drone_pos,
                                   env._drone_yaw, env._drone_vel, 0., True)
        self.original_generate = env.synth_pc.generate_with_metadata

        def capture(*args, **kwargs):
            result = self.original_generate(*args, **kwargs)
            self.frames.append((np.asarray(args[0]).copy(), float(args[1]), result[0].copy()))
            return result
        env.synth_pc.generate_with_metadata = capture
        self.runtime_parameters = dict(self.runtime_parameters)
        self.runtime_parameters['super_sync'] = dict(
            service='paper_super_sync/step', logical_time_driven=True,
            timers='none', replan_cadence='one_replan_per_decision_common_H4',
            config=str(self.config_path),
            source='scripts/routea_paired_support/super_planner_policy.py')

    def predict(self, observation, context, *, action_seed):
        meta = asdict(context['pointcloud_meta_history'][-1])
        self.last_provenance = EgoProvenance(meta['raw_array_hash'],
            context['episode_identity_hash'], int(context['decision_index']), policy_kind='super')

        def hold():
            # No usable trajectory for this window: stay at the actual state
            # while the native FSM retries; the environment still consumes
            # physical time (shared failure handling of the three arms).
            r = np.zeros((8, 11), np.float32)
            r[:, :3], r[:, 9] = self.env._drone_pos, self.env._drone_yaw
            return r

        self.reference_invalid = False
        if self.response.has_trajectory:
            reference = self.reference_sampler(self.response, 1. + self.elapsed,
                self.env._drone_pos, self.env._drone_yaw, self.yaw_rate)
            if not np.isfinite(reference).all():
                self.reference_invalid = True
                self.invalid_references += 1
                print('NONFINITE_REFERENCE decision=%d elapsed=%.3f'
                      % (int(context['decision_index']), self.elapsed), flush=True)
                reference = hold()
        else:
            reference = hold()
        self.reference = reference
        self.frames.clear()
        return reference_actions(reference, self.env._drone_pos, self.env._drone_yaw)

    def after_step(self, context):
        result = super().after_step(context)
        result['action_origin'] = 'super_native_polynomial'
        return result
