#!/usr/bin/env python3
"""Run under px4ctrl's legacy messages; consume SUPER's neutral numeric wire."""


def copy_reference(values, command):
    """Wire: stamp,p[3],v[3],a[3],yaw,yaw_rate,native_flag."""
    for field, start in [('position', 1), ('velocity', 4), ('acceleration', 7)]:
        target = getattr(command, field)
        target.x, target.y, target.z = values[start:start+3]
    command.yaw, command.yaw_dot = values[10:12]
    # SUPER 2 means backup trajectory; legacy 2 means emergency, not backup.
    command.trajectory_flag = 1 if int(values[12]) in (1, 2) else 0
    return command


def main():
    import rospy
    from quadrotor_msgs.msg import PositionCommand
    from std_msgs.msg import Float64MultiArray
    rospy.init_node('paper_super_legacy_command')
    publisher = rospy.Publisher('/planning/pos_cmd', PositionCommand, queue_size=10)

    def relay(message):
        command = copy_reference(message.data, PositionCommand())
        command.header.stamp = rospy.Time.from_sec(message.data[0])
        command.header.frame_id = 'world'
        publisher.publish(command)

    rospy.Subscriber('/paper_super/reference', Float64MultiArray, relay, queue_size=10)


    rospy.spin()


if __name__ == '__main__':
    main()
