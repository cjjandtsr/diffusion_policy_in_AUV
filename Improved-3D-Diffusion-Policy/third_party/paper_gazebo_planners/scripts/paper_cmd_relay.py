#!/usr/bin/env python3
"""Planner-side half of the Gazebo command bridge (generic).

Each planner workspace carries its OWN `quadrotor_msgs` definition:

    catkin_ws0 (px4ctrl / ego)   4c4fec5ab2ba29a45210c6ec5d10157a
    paper_fast_ws (Fast official) 4712f0609ca29a79af79a35ca3e3967a
    paper_super_ws (SUPER official) d008e86de36e11deb1e4033ac2c394a9

They are same-named but not interchangeable, so no two of them may share a topic.
This node runs inside the planner's own environment, reads that planner's command
topic, and republishes the numbers as std_msgs/Float64MultiArray. The matching
paper_cmd_import.py (running in catkin_ws0) turns them back into the ego
definition on /planning/pos_cmd, which is what px4ctrl subscribes.

Wire layout (18 doubles, fixed):
  0..2 position, 3..5 velocity, 6..8 acceleration, 9..11 jerk,
  12 yaw, 13 yaw_dot, 14 trajectory_id, 15 trajectory_flag,
  16 vel_norm, 17 acc_norm
Fields absent in a planner's definition are sent as 0.
"""
import rospy
from std_msgs.msg import Float64MultiArray


def pack(command):
    message = Float64MultiArray()
    # The three quadrotor_msgs definitions in play differ in exactly these ways:
    # Fast-Planner's PositionCommand has no `jerk`, SUPER's has no `trajectory_flag`
    # in some builds, and only SUPER's carries vel_norm/acc_norm. Read every
    # optional field defensively instead of assuming one layout.
    jerk = getattr(command, 'jerk', None)

    def _vec(vec, axis):
        return float(getattr(vec, axis, 0.0)) if vec is not None else 0.0

    message.data = [
        command.position.x, command.position.y, command.position.z,
        command.velocity.x, command.velocity.y, command.velocity.z,
        command.acceleration.x, command.acceleration.y, command.acceleration.z,
        _vec(jerk, 'x'), _vec(jerk, 'y'), _vec(jerk, 'z'),
        float(getattr(command, 'yaw', 0.0)), float(getattr(command, 'yaw_dot', 0.0)),
        float(getattr(command, 'trajectory_id', 0)),
        float(getattr(command, 'trajectory_flag', 0)),
        float(getattr(command, 'vel_norm', 0.0)),
        float(getattr(command, 'acc_norm', 0.0)),
    ]
    return message


def run(default_source, node_name='paper_cmd_relay'):
    rospy.init_node(node_name)
    source = rospy.get_param('~source_topic', default_source)
    target = rospy.get_param('~neutral_topic', '/paper/pos_cmd_neutral')
    frame = rospy.get_param('~frame_id', 'world')
    from quadrotor_msgs.msg import PositionCommand
    publisher = rospy.Publisher(target, Float64MultiArray, queue_size=10)
    counter = {'n': 0}

    def callback(command):
        publisher.publish(pack(command))
        counter['n'] += 1
        if counter['n'] % 500 == 1:
            rospy.loginfo('[paper_cmd_relay] forwarded %d commands (%s -> %s)',
                          counter['n'], source, target)

    rospy.Subscriber(source, PositionCommand, callback, queue_size=10)
    rospy.loginfo('[paper_cmd_relay] %s md5=%s -> %s (frame %s)',
                  source, PositionCommand._md5sum, target, frame)
    rospy.spin()
