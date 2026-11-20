#include <ros/ros.h>
#include <mutex>
#include <message_filters/subscriber.h>
#include <message_filters/cache.h>
#include <nav_msgs/Odometry.h>
#include <quadrotor_msgs/PositionCommand.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/Imu.h>
#include "ego_planner/SyncData.h"

using namespace message_filters;

class SyncRecorder {
public:
    SyncRecorder() : nh_("~") {
        // 订阅五个话题，每个使用 Cache 缓存最新一条消息
        odom_sub_.subscribe(nh_, "/mavros/local_position/odom", 1);
        cmd_sub_.subscribe(nh_, "/planning/pos_cmd", 1);
        pc_sub_.subscribe(nh_, "/iris_D435i/realsense/depth_camera/depth/points", 1);
        goal_sub_.subscribe(nh_, "/move_base_simple/goal", 1);
        imu_sub_.subscribe(nh_, "/mavros/imu/data", 1);

        odom_cache_.reset(new Cache<nav_msgs::Odometry>(odom_sub_, 1));
        cmd_cache_.reset(new Cache<quadrotor_msgs::PositionCommand>(cmd_sub_, 1));
        pc_cache_.reset(new Cache<sensor_msgs::PointCloud2>(pc_sub_, 1));
        goal_cache_.reset(new Cache<geometry_msgs::PoseStamped>(goal_sub_, 1));
        imu_cache_.reset(new Cache<sensor_msgs::Imu>(imu_sub_, 1));

        // 定时器 3.3 Hz
        timer_ = nh_.createTimer(ros::Duration(1.0 / 3.3), &SyncRecorder::timerCallback, this);

        sync_pub_ = nh_.advertise<ego_planner::SyncData>("/sync_data", 10);

        ROS_INFO("SyncRecorder started at 3.3 Hz. Waiting for data...");
    }

private:
    ros::NodeHandle nh_;
    ros::Timer timer_;
    ros::Publisher sync_pub_;

    message_filters::Subscriber<nav_msgs::Odometry> odom_sub_;
    message_filters::Subscriber<quadrotor_msgs::PositionCommand> cmd_sub_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> pc_sub_;
    message_filters::Subscriber<geometry_msgs::PoseStamped> goal_sub_;
    message_filters::Subscriber<sensor_msgs::Imu> imu_sub_;

    std::unique_ptr<Cache<nav_msgs::Odometry>> odom_cache_;
    std::unique_ptr<Cache<quadrotor_msgs::PositionCommand>> cmd_cache_;
    std::unique_ptr<Cache<sensor_msgs::PointCloud2>> pc_cache_;
    std::unique_ptr<Cache<geometry_msgs::PoseStamped>> goal_cache_;
    std::unique_ptr<Cache<sensor_msgs::Imu>> imu_cache_;

    void timerCallback(const ros::TimerEvent& e) {
        // 获取每个话题的最新消息（getElemAfterTime(ros::Time(0)) 返回最新的一条）
        auto odom_msg = odom_cache_->getElemAfterTime(ros::Time(0));
        auto cmd_msg  = cmd_cache_->getElemAfterTime(ros::Time(0));
        auto pc_msg   = pc_cache_->getElemAfterTime(ros::Time(0));
        auto goal_msg = goal_cache_->getElemAfterTime(ros::Time(0));
        auto imu_msg  = imu_cache_->getElemAfterTime(ros::Time(0));

        // 必须所有话题都有数据才能发布
        if (!odom_msg || !cmd_msg || !pc_msg || !goal_msg || !imu_msg) {
            ROS_WARN_THROTTLE(5.0, "Waiting for all topics to receive data...");
            return;
        }

        ego_planner::SyncData sync_msg;
        sync_msg.header.stamp = e.current_real;   // 使用定时器触发时间作为主时间戳
        sync_msg.odom = *odom_msg;
        sync_msg.pos_cmd = *cmd_msg;
        sync_msg.pointcloud = *pc_msg;
        sync_msg.goal = *goal_msg;
        sync_msg.imu = *imu_msg;

        sync_pub_.publish(sync_msg);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "sync_recorder_node");
    SyncRecorder sr;
    ros::spin();
    return 0;
}