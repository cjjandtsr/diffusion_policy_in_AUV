#!/usr/bin/env python3
"""World-frame observation bridge for the Gazebo arms — no tf2 dependency.

Why this exists: this machine's ROS install has `tf2_py` but neither `tf2_ros`
nor `tf2_sensor_msgs` usable from the evaluation interpreter, so the original
tf2-based bridge died on import and the planner never received a cloud.

What upstream requires (SUPER README issue #10): when a planner is driven by an
external simulator, the point cloud must already be in the **world** frame —
ROG-Map (SUPER) and Fast-Planner's `plan_env` ignore frame_id and do no TF lookup
of their own. So this node does exactly that transform itself:

    p_world = R_drone(attitude) * ( R_mount * p_cam + t_mount ) + p_drone

- drone pose comes from the flight controller odometry (`~odom_topic`, world frame)
- mount extrinsic defaults to the iris_D435i SDF: the D435i is included at
  (0.1, 0, 0) with no rotation relative to the body frame
- everything else (time sync, publishing) uses plain rospy + message_filters

Both outputs are stamped with the same header so a planner sees a consistent pair.
"""
import numpy as np
import rospy
import message_filters
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header


def quaternion_to_matrix(w, x, y, z):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def main():
    rospy.init_node('paper_local_cloud_bridge')
    cloud_topic = rospy.get_param('~cloud_topic',
                                  '/iris_D435i/realsense/depth_camera/depth/points')
    odom_topic = rospy.get_param('~odom_topic', '/mavros/local_position/odom')
    mount = [float(v) for v in rospy.get_param('~mount_xyz', [0.1, 0.0, 0.0])]
    mount_rpy = [float(v) for v in rospy.get_param('~mount_rpy', [0.0, 0.0, 0.0])]
    world_frame = rospy.get_param('~world_frame', 'world')

    roll, pitch, yaw = mount_rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    r_mount = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])
    t_mount = np.asarray(mount, dtype=float)

    cloud_pub = rospy.Publisher('/paper/cloud_local', PointCloud2, queue_size=1)
    odom_pub = rospy.Publisher('/paper/odom_local', Odometry, queue_size=1)

    def callback(cloud_msg, odom_msg):
        points = np.asarray(list(pc2.read_points(cloud_msg, field_names=('x', 'y', 'z'),
                                                 skip_nans=True)), dtype=np.float32)
        pose = odom_msg.pose.pose
        r_drone = quaternion_to_matrix(pose.orientation.w, pose.orientation.x,
                                       pose.orientation.y, pose.orientation.z)
        p_drone = np.array([pose.position.x, pose.position.y, pose.position.z])
        if points.size:
            local = points @ r_mount.T + t_mount
            world = local @ r_drone.T + p_drone
        else:
            world = np.zeros((0, 3), np.float32)

        header = Header(stamp=cloud_msg.header.stamp, frame_id=world_frame)
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1)]
        cloud_pub.publish(pc2.create_cloud(header, fields,
                                           world.astype(np.float32).tolist()))

        state = Odometry()
        state.header = header
        state.child_frame_id = pose_child_frame = 'base_link'
        state.pose.pose = pose
        state.pose.pose.position.x = float(p_drone[0])
        state.pose.pose.position.y = float(p_drone[1])
        state.pose.pose.position.z = float(p_drone[2])
        odom_pub.publish(state)

    cloud_sub = message_filters.Subscriber(cloud_topic, PointCloud2)
    odom_sub = message_filters.Subscriber(odom_topic, Odometry)
    sync = message_filters.ApproximateTimeSynchronizer([cloud_sub, odom_sub], 20, 0.08)
    sync.registerCallback(callback)
    rospy.loginfo('[paper_local_cloud_bridge] %s + %s -> /paper/cloud_local (world frame, '
                  'mount=%s rpy=%s)', cloud_topic, odom_topic, mount, mount_rpy)
    rospy.spin()


if __name__ == '__main__':
    main()
