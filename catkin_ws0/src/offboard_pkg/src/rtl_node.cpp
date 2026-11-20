#include <ros/ros.h>
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

// 信号处理函数
void signalHandler(int signum) {
    ROS_INFO("Interrupt signal received, shutting down");
    ros::shutdown();
    exit(signum);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "rtl_simple_node");
    ros::NodeHandle nh;

    // 注册信号处理函数
    signal(SIGINT, signalHandler);
    
    // 设置等待时间参数（单位：秒）
    double wait_time = 3.0; // 默认等待3秒后执行RTL
    if (argc > 1) {
        wait_time = atof(argv[1]);
    }

    ros::Subscriber state_sub = nh.subscribe<mavros_msgs::State>
            ("mavros/state", 10, state_cb);
    set_mode_client = nh.serviceClient<mavros_msgs::SetMode>
            ("mavros/set_mode");

    ros::Rate rate(10.0);

    // 等待 FCU 连接
    ROS_INFO("等待飞控连接...");
    while(ros::ok() && !current_state.connected){
        ros::spinOnce();
        rate.sleep();
    }

    ROS_INFO("飞控已连接，等待 %.1f 秒后执行自动返航...", wait_time);
    ros::Duration(wait_time).sleep();

    // 切换到 RTL 模式
    mavros_msgs::SetMode rtl_set_mode;
    rtl_set_mode.request.custom_mode = "AUTO.RTL";

    if (set_mode_client.call(rtl_set_mode) && rtl_set_mode.response.mode_sent) {
        ROS_INFO("自动返航模式已启动 - 无人机将返回起飞点并降落");
        
        // 监控返航状态
        ros::Time start_time = ros::Time::now();
        while(ros::ok() && (ros::Time::now() - start_time < ros::Duration(120.0))) {
            if (current_state.mode == "AUTO.RTL") {
                ROS_INFO_THROTTLE(5, "自动返航执行中...");
            } else {
                ROS_WARN_THROTTLE(5, "等待进入自动返航模式...");
            }
            ros::spinOnce();
            rate.sleep();
        }
        
        ROS_INFO("自动返航命令执行完成");
    } else {
        ROS_ERROR("无法启动自动返航模式");
        return 1;
    }

    return 0;
}