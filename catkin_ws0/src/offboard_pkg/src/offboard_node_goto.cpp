// offboard_node_goto.cpp — offboard_node.cpp 的副本, 唯一改动: 起飞目标点由命令行给
//   原文件 offboard_node.cpp 未改, 原 target offboard_node 行为不变。
//
//   用法: rosrun offboard_pkg offboard_node_goto <flight_duration_s> [x] [y] [z]
//     默认 (0, 0, 2) = 原文件硬编码值
//   坐标系: MAVROS local/odom 系, 原点 = 无人机出生点
#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>
#include <signal.h>
#include <cstdlib>

mavros_msgs::State current_state;

void state_cb(const mavros_msgs::State::ConstPtr& msg){
    current_state = *msg;
}

// 全局变量，用于在信号处理函数中访问
ros::ServiceClient set_mode_client;

// 信号处理函数，处理Ctrl+C等中断信号
void signalHandler(int signum) {
    ROS_INFO("Interrupt signal received, switching to AUTO.LOITER mode");

    // 切换到AUTO.LOITER模式
    mavros_msgs::SetMode auto_loiter_set_mode;
    auto_loiter_set_mode.request.custom_mode = "AUTO.LOITER";

    if (set_mode_client.call(auto_loiter_set_mode) && auto_loiter_set_mode.response.mode_sent) {
        ROS_INFO("AUTO.LOITER mode enabled - drone will maintain position and altitude");
    } else {
        ROS_ERROR("Failed to set AUTO.LOITER mode");
    }

    // 等待一段时间确保模式切换完成
    ros::Duration(1.0).sleep();

    ros::shutdown();
    exit(signum);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "offboard_node_goto");
    ros::NodeHandle nh;

    // 注册信号处理函数
    signal(SIGINT, signalHandler);

    // 设置飞行时间参数（单位：秒）
    double flight_duration = 30.0; // 默认飞行30秒后切换模式
    if (argc > 1) {
        flight_duration = atof(argv[1]);
    }

    // 起飞目标点 (odom 系), 缺省与原 offboard_node.cpp 一致
    double target_x = 0.0;
    double target_y = 0.0;
    double target_z = 2.0;
    if (argc > 4) {
        target_x = atof(argv[2]);
        target_y = atof(argv[3]);
        target_z = atof(argv[4]);
    }

    ROS_INFO("[offboard_node_goto] duration=%.1fs target=(%.3f, %.3f, %.3f) odom",
             flight_duration, target_x, target_y, target_z);

    ros::Subscriber state_sub = nh.subscribe<mavros_msgs::State>
            ("mavros/state", 10, state_cb);
    ros::Publisher local_pos_pub = nh.advertise<geometry_msgs::PoseStamped>
            ("mavros/setpoint_position/local", 10);
    ros::ServiceClient arming_client = nh.serviceClient<mavros_msgs::CommandBool>
            ("mavros/cmd/arming");
    set_mode_client = nh.serviceClient<mavros_msgs::SetMode>
            ("mavros/set_mode");

    //the setpoint publishing rate MUST be faster than 2Hz
    ros::Rate rate(20.0);

    // wait for FCU connection
    while(ros::ok() && !current_state.connected){
        ros::spinOnce();
        rate.sleep();
    }

    geometry_msgs::PoseStamped pose;
    pose.pose.position.x = target_x;
    pose.pose.position.y = target_y;
    pose.pose.position.z = target_z;

    //send a few setpoints before starting
    for(int i = 100; ros::ok() && i > 0; --i){
        local_pos_pub.publish(pose);
        ros::spinOnce();
        rate.sleep();
    }

    mavros_msgs::SetMode offb_set_mode;
    offb_set_mode.request.custom_mode = "OFFBOARD";

    mavros_msgs::CommandBool arm_cmd;
    arm_cmd.request.value = true;

    ros::Time last_request = ros::Time::now();
    ros::Time start_time = ros::Time::now();
    bool mode_switched = false;

    while(ros::ok()){
        // 检查是否到达切换模式的时间
        if (!mode_switched && (ros::Time::now() - start_time > ros::Duration(flight_duration))) {
            ROS_INFO("Flight duration reached, switching to AUTO.LOITER mode");

            mavros_msgs::SetMode auto_loiter_set_mode;
            auto_loiter_set_mode.request.custom_mode = "AUTO.LOITER";

            if (set_mode_client.call(auto_loiter_set_mode) && auto_loiter_set_mode.response.mode_sent) {
                ROS_INFO("AUTO.LOITER mode enabled - drone will maintain position and altitude");
                mode_switched = true;
            } else {
                ROS_ERROR("Failed to set AUTO.LOITER mode");
            }
            last_request = ros::Time::now();
        }

        if (!mode_switched) {
            // 只有在未切换模式时才执行OFFBOARD相关逻辑
            if( current_state.mode != "OFFBOARD" &&
                (ros::Time::now() - last_request > ros::Duration(5.0))){
                if( set_mode_client.call(offb_set_mode) &&
                    offb_set_mode.response.mode_sent){
                    ROS_INFO("Offboard enabled");
                }
                last_request = ros::Time::now();
            } else {
                if( !current_state.armed &&
                    (ros::Time::now() - last_request > ros::Duration(5.0))){
                    if( arming_client.call(arm_cmd) &&
                        arm_cmd.response.success){
                        ROS_INFO("Vehicle armed");
                    }
                    last_request = ros::Time::now();
                }
            }

            local_pos_pub.publish(pose);
        }

        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}
