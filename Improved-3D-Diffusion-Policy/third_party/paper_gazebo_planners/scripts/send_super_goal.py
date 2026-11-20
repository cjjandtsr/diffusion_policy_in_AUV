#!/usr/bin/env python3
"""Publish the goal to SUPER's click-goal entry, released by the runner's gate.

Contract (delivered super.launch + prepare_paper_super.py):
    config           fsm.click_goal_topic = /paper_super/goal   (geometry_msgs/PoseStamped)
    launch params    goal_topic, goal_x, goal_y, goal_z         (local/MAVROS frame)
    gate             /paper_goal_enabled, opened by the runner once the monitor subscribes,
                     so the monitor measures the flight instead of starting after it.
click_height is -10 in the generated config (i.e. cancelled), so the goal height is used
exactly as given - the paper's own task height, not an override.
"""
import rospy
from geometry_msgs.msg import PoseStamped


def main():
    rospy.init_node('paper_super_goal')
    topic = rospy.get_param('~goal_topic', '/paper_super/goal')
    x = float(rospy.get_param('~goal_x'))
    y = float(rospy.get_param('~goal_y'))
    z = float(rospy.get_param('~goal_z'))
    gate = rospy.get_param('~gate', '/paper_goal_enabled')
    timeout = float(rospy.get_param('~timeout', 300.0))

    publisher = rospy.Publisher(topic, PoseStamped, queue_size=1, latch=True)
    rospy.loginfo('[paper_super_goal] waiting for %s, then PoseStamped -> %s at (%.2f, %.2f, %.2f)',
                  gate, topic, x, y, z)
    deadline = rospy.Time.now() + rospy.Duration(timeout)
    while not rospy.is_shutdown():
        if rospy.get_param(gate, False):
            break
        if rospy.Time.now() > deadline:
            rospy.logerr('[paper_super_goal] gate never opened; publishing anyway')
            break
        rospy.sleep(0.2)

    goal = PoseStamped()
    goal.header.stamp = rospy.Time.now()
    goal.header.frame_id = 'world'
    goal.pose.position.x = x
    goal.pose.position.y = y
    goal.pose.position.z = z
    goal.pose.orientation.w = 1.0
    for _ in range(5):
        publisher.publish(goal)
        rospy.sleep(0.1)
    rospy.loginfo('[paper_super_goal] goal published after the monitor was ready')


if __name__ == '__main__':
    main()
