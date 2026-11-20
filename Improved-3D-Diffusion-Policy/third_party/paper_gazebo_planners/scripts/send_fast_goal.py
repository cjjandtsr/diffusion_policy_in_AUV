#!/usr/bin/env python3
"""Publish the goal to Fast-Planner's own entry point.

Code fact (kino_replan_fsm.cpp:62): in MANUAL_TARGET the kino FSM subscribes
    /waypoint_generator/waypoints   (nav_msgs/Path)
and reads the first pose. That is Fast-Planner's upstream interactive entry
(the RViz waypoint_generator forwards /move_base_simple/goal into it). EGO uses
/move_base_simple/goal directly, which is why the EGO branch of the runner cannot
drive this planner.

The goal is released only once the runner opens /paper_goal_enabled (right before
the monitor subscribes), so the monitor measures the whole flight instead of
starting after it.
"""
import argparse

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--x', type=float, required=True)
    parser.add_argument('--y', type=float, required=True)
    parser.add_argument('--z', type=float, required=True)
    parser.add_argument('--topic', default='/waypoint_generator/waypoints')
    parser.add_argument('--gate', default='/paper_goal_enabled')
    parser.add_argument('--timeout', type=float, default=300.0)
    args, _unknown = parser.parse_known_args()   # roslaunch appends __name/__log

    rospy.init_node('paper_fast_goal')
    publisher = rospy.Publisher(args.topic, Path, queue_size=1, latch=True)
    rospy.loginfo('[paper_fast_goal] waiting for %s, then Path(1 pose) -> %s at '
                  '(%.2f, %.2f, %.2f)', args.gate, args.topic, args.x, args.y, args.z)
    stamp_deadline = rospy.Time.now() + rospy.Duration(args.timeout)
    while not rospy.is_shutdown():
        if rospy.get_param(args.gate, False):
            break
        if rospy.Time.now() > stamp_deadline:
            rospy.logerr('[paper_fast_goal] gate never opened; publishing anyway')
            break
        rospy.sleep(0.2)

    path = Path()
    path.header.stamp = rospy.Time.now()
    path.header.frame_id = 'world'
    pose = PoseStamped()
    pose.header = path.header
    pose.pose.position.x = args.x
    pose.pose.position.y = args.y
    pose.pose.position.z = args.z
    pose.pose.orientation.w = 1.0
    path.poses.append(pose)
    for _ in range(5):
        publisher.publish(path)
        rospy.sleep(0.1)
    rospy.loginfo('[paper_fast_goal] goal Path published after the monitor was ready')


if __name__ == '__main__':
    main()
