#!/usr/bin/env python3
"""SUPER-side half of the paper command bridge: native PositionCommand -> legacy wire.

Wire is fixed by scripts/paper_super_legacy_command.py:copy_reference:

    [stamp, p(3), v(3), a(3), yaw, yaw_dot, native_flag]   (13 doubles, stamp first)
    native_flag: SUPER 1 = normal / 2 = backup trajectory -> legacy 1; else 0
                 (legacy 2 would mean emergency, which is a different thing)

This node runs inside the SUPER overlay (roslaunch inherits NATIVE_SETUP), so the
`quadrotor_msgs` imported here is SUPER's own definition, md5 d008e86d...; the legacy
end runs in the controller workspace with md5 4c4fec5a..., which is why the two ends
never share a topic.
"""
import rospy
from std_msgs.msg import Float64MultiArray


def main():
    rospy.init_node('paper_super_command_relay')
    source = rospy.get_param('~source_topic', '/paper_super/position_cmd')
    target = rospy.get_param('~reference_topic', '/paper_super/reference')
    from quadrotor_msgs.msg import PositionCommand
    publisher = rospy.Publisher(target, Float64MultiArray, queue_size=10)
    counter = {'n': 0}

    def callback(command):
        message = Float64MultiArray()
        number = getattr(command, 'position', None)
        message.data = [
            command.header.stamp.to_sec(),
            number.x, number.y, number.z,
            command.velocity.x, command.velocity.y, command.velocity.z,
            command.acceleration.x, command.acceleration.y, command.acceleration.z,
            float(getattr(command, 'yaw', 0.0)), float(getattr(command, 'yaw_dot', 0.0)),
            float(getattr(command, 'trajectory_flag', 0)),
        ]
        publisher.publish(message)
        counter['n'] += 1
        if counter['n'] % 500 == 1:
            rospy.loginfo('[paper_super_command_relay] forwarded %d (%s -> %s)',
                          counter['n'], source, target)

    rospy.Subscriber(source, PositionCommand, callback, queue_size=10)
    rospy.loginfo('[paper_super_command_relay] %s md5=%s -> %s (13 doubles, stamp first)',
                  source, PositionCommand._md5sum, target)
    rospy.spin()


if __name__ == '__main__':
    main()
