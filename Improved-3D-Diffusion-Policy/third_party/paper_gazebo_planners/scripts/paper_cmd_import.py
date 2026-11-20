#!/usr/bin/env python3
"""px4ctrl side of the Gazebo command bridge.

Runs in the catkin_ws0 environment, so `quadrotor_msgs` here is the ego-planner
definition that px4ctrl was compiled against. It rebuilds the command from the
neutral Float64MultiArray published by paper_super_command_relay.py and publishes
it on /planning/pos_cmd, which is where px4ctrl subscribes (`cmd`).

Wire layout (18 doubles, fixed):
  0..2 position, 3..5 velocity, 6..8 acceleration, 9..11 jerk,
  12 yaw, 13 yaw_dot, 14 trajectory_id, 15 trajectory_flag,
  16 vel_norm (unused), 17 acc_norm (unused)
"""
import rospy
from std_msgs.msg import Float64MultiArray


def main():
    rospy.init_node('paper_cmd_import')
    neutral_topic = rospy.get_param('~neutral_topic', '/paper/pos_cmd_neutral')
    command_topic = rospy.get_param('~command_topic', '/planning/pos_cmd')
    frame_id = rospy.get_param('~frame_id', 'world')
    from quadrotor_msgs.msg import PositionCommand
    publisher = rospy.Publisher(command_topic, PositionCommand, queue_size=10)

    def callback(message):
        data = list(message.data)
        if len(data) != 18:
            rospy.logerr_throttle(5.0, '[paper_cmd_import] expected 18 values, got %d', len(data))
            return
        command = PositionCommand()
        command.header.stamp = rospy.Time.now()
        command.header.frame_id = frame_id
        command.position.x, command.position.y, command.position.z = data[0:3]
        command.velocity.x, command.velocity.y, command.velocity.z = data[3:6]
        command.acceleration.x, command.acceleration.y, command.acceleration.z = data[6:9]
        command.jerk.x, command.jerk.y, command.jerk.z = data[9:12]
        command.yaw = data[12]
        command.yaw_dot = data[13]
        command.trajectory_id = int(data[14])
        if hasattr(command, 'trajectory_flag'):
            command.trajectory_flag = int(data[15])
        publisher.publish(command)

    rospy.Subscriber(neutral_topic, Float64MultiArray, callback, queue_size=10)
    rospy.loginfo('[paper_cmd_import] %s (neutral) -> %s (px4ctrl PositionCommand)',
                  neutral_topic, command_topic)


    rospy.spin()


if __name__ == '__main__':
    main()
