#!/usr/bin/env python3
"""Run under SUPER's message environment, avoiding PositionCommand MD5 collision."""
import rospy
from quadrotor_msgs.msg import PositionCommand
from std_msgs.msg import Float64MultiArray


def main():
    rospy.init_node('paper_super_command_relay')
    publisher = rospy.Publisher('/paper_super/reference', Float64MultiArray, queue_size=10)

    def relay(command):
        p, v, a = command.position, command.velocity, command.acceleration
        publisher.publish(Float64MultiArray(data=[
            command.header.stamp.to_sec(), p.x, p.y, p.z,
            v.x, v.y, v.z, a.x, a.y, a.z,
            command.yaw, command.yaw_dot, float(command.trajectory_flag)]))

    rospy.Subscriber('/paper_super/position_cmd', PositionCommand, relay, queue_size=10)
    rospy.spin()


if __name__ == '__main__':
    main()
