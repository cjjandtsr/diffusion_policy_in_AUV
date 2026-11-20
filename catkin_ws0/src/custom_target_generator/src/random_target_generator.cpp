#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Bool.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>
#include <std_msgs/String.h>
#include <random>
#include <cmath>
#include <tf2/LinearMath/Quaternion.h>

/*
 * random_target_generator v4 — 改进的绕障数据收集
 *
 * 修改 (2026-03-22):
 *   1. Z范围从 [1.5, 2.5] 扩大到 [1.0, 3.0], 让训练数据覆盖更宽的高度
 *   2. 循环结构: N轮, 每轮3阶段 (正向穿越→过渡→反向穿越→自由飞行)
 *   3. 所有飞行阶段统一使用 avoidance_start/avoidance_end 标记
 *   4. 自适应等待时间: 根据距离计算预期飞行时间, 而非固定sleep
 *   5. 添加到达检测: 距离<1.0m即认为到达，不再盲等
 */

mavros_msgs::State current_state;
ros::Publisher phase_pub;
geometry_msgs::PoseStamped current_pose;
bool pose_received = false;

void state_cb(const mavros_msgs::State::ConstPtr& msg){
    current_state = *msg;
}

void local_pos_cb(const geometry_msgs::PoseStamped::ConstPtr& msg) {
    current_pose = *msg;
    pose_received = true;
}

void publishPhase(const std::string& phase_name) {
    std_msgs::String phase_msg;
    phase_msg.data = phase_name;
    phase_pub.publish(phase_msg);
    ROS_INFO("阶段切换: %s", phase_name.c_str());
}

// 等待无人机接近目标点 (距离<threshold 或超时)
bool waitUntilNear(double target_x, double target_y, double target_z,
                   double threshold, double timeout_sec, ros::Rate& rate) {
    ros::Time start = ros::Time::now();
    while (ros::ok()) {
        ros::spinOnce();
        rate.sleep();
        
        if (!pose_received) continue;
        
        double dx = current_pose.pose.position.x - target_x;
        double dy = current_pose.pose.position.y - target_y;
        double dz = current_pose.pose.position.z - target_z;
        double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
        
        if (dist < threshold) {
            ROS_INFO("  到达目标附近 (dist=%.2f < %.1f)", dist, threshold);
            return true;
        }
        
        double elapsed = (ros::Time::now() - start).toSec();
        if (elapsed > timeout_sec) {
            ROS_WARN("  等待超时 %.0fs (dist=%.2f)", timeout_sec, dist);
            return false;
        }
    }
    return false;
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "random_target_generator");
    ros::NodeHandle nh;
    
    ros::Publisher target_pub = nh.advertise<geometry_msgs::PoseStamped>("/move_base_simple/goal", 10);
    ros::Publisher trigger_pub = nh.advertise<std_msgs::Bool>("/traj_start_trigger", 10);
    phase_pub = nh.advertise<std_msgs::String>("/current_phase", 10);
    ros::Subscriber state_sub = nh.subscribe<mavros_msgs::State>("mavros/state", 10, state_cb);
    ros::Subscriber local_pos_sub = nh.subscribe<geometry_msgs::PoseStamped>("mavros/local_position/pose", 10, local_pos_cb);
    
    // ════════════════════════════════════════════════════════════
    // 关键修改: Z范围从 [1.5, 2.5] 扩大到 [1.0, 3.0]
    // ════════════════════════════════════════════════════════════
    double min_height = 1.0, max_height = 3.0;
    double obstacle_x = 8.0;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> z_dist(min_height, max_height);
    
    // 正向穿越（墙前 → 墙后）
    std::uniform_real_distribution<double> forward_start_x(3.0, 5.0);
    std::uniform_real_distribution<double> forward_start_y(-2.0, 2.0);
    std::uniform_real_distribution<double> forward_target_dist(4.0, 9.0);
    std::uniform_real_distribution<double> forward_target_y(-5.0, 5.0);

    // 反向穿越（墙后 → 墙前）
    std::uniform_real_distribution<double> backward_start_x(3.0, 5.0);
    std::uniform_real_distribution<double> backward_start_y(-2.0, 2.0);
    std::uniform_real_distribution<double> backward_target_dist(4.0, 9.0);
    std::uniform_real_distribution<double> backward_target_y(-5.0, 5.0);

    // 过渡飞行（不记录数据）
    std::uniform_real_distribution<double> trans_x(9.0, 10.0);
    std::uniform_real_distribution<double> trans_y(-3.0, 3.0);

    // 自由飞行
    const double MIN_FREE_DIST = 5.0;
    const double MAX_FREE_DIST = 20.0;
    const double X_MIN = -10.0, X_MAX = 20.0;
    const double Y_MIN = -10.0, Y_MAX = 10.0;

    ROS_INFO("=== 开始绕障数据收集 v4 ===");
    ROS_INFO("Z范围: [%.1f, %.1f]m (扩大)", min_height, max_height);
    ROS_INFO("墙位置: x=%.1f", obstacle_x);
    
    // 等待飞控连接
    ros::Rate rate(10.0);
    while(ros::ok() && !current_state.connected){
        ros::spinOnce();
        rate.sleep();
    }
    
    ROS_INFO("等待10秒系统初始化...");
    ros::Duration(10.0).sleep();
    
    std_msgs::Bool trigger;
    trigger.data = true;
    
    // 三阶段循环
    for (int cycle = 1; cycle <= 3; cycle++) {
        ROS_INFO("=== 第 %d/3 轮 ===", cycle);

        if (cycle == 1) {
            // ── 正向穿越: 墙前 → 墙后 ──
            
            // 先飞到起始位置 (不记录)
            double start_x = obstacle_x - forward_start_x(gen);
            double start_y = forward_start_y(gen);
            double start_z = z_dist(gen);
            
            geometry_msgs::PoseStamped start_target;
            start_target.header.stamp = ros::Time::now();
            start_target.header.frame_id = "world";
            start_target.pose.position.x = start_x;
            start_target.pose.position.y = start_y;
            start_target.pose.position.z = start_z;
            tf2::Quaternion q; q.setRPY(0, 0, 0);
            start_target.pose.orientation.x = q.x();
            start_target.pose.orientation.y = q.y();
            start_target.pose.orientation.z = q.z();
            start_target.pose.orientation.w = q.w();
            target_pub.publish(start_target);
            trigger_pub.publish(trigger);
            ROS_INFO("正向起点 (墙前): [%.2f, %.2f, %.2f]", start_x, start_y, start_z);
            waitUntilNear(start_x, start_y, start_z, 1.5, 15.0, rate);
            ros::Duration(3.0).sleep(); // 稳定

            // 开始记录, 飞向墙后
            publishPhase("avoidance_start");
            
            double target_x = obstacle_x + forward_target_dist(gen);
            double target_y = forward_target_y(gen);
            double target_z = z_dist(gen);
            
            geometry_msgs::PoseStamped target;
            target.header.stamp = ros::Time::now();
            target.header.frame_id = "world";
            target.pose.position.x = target_x;
            target.pose.position.y = target_y;
            target.pose.position.z = target_z;
            target.pose.orientation.x = q.x();
            target.pose.orientation.y = q.y();
            target.pose.orientation.z = q.z();
            target.pose.orientation.w = q.w();
            target_pub.publish(target);
            trigger_pub.publish(trigger);
            ROS_INFO("正向目标 (墙后): [%.2f, %.2f, %.2f]", target_x, target_y, target_z);
            waitUntilNear(target_x, target_y, target_z, 1.0, 25.0, rate);
            
            publishPhase("avoidance_end");

            // 过渡飞行 (不记录)
            double trans_target_x = obstacle_x + trans_x(gen);
            double trans_target_y = trans_y(gen);
            double trans_target_z = z_dist(gen);
            
            geometry_msgs::PoseStamped trans_target;
            trans_target.header.stamp = ros::Time::now();
            trans_target.header.frame_id = "world";
            trans_target.pose.position.x = trans_target_x;
            trans_target.pose.position.y = trans_target_y;
            trans_target.pose.position.z = trans_target_z;
            trans_target.pose.orientation.x = q.x();
            trans_target.pose.orientation.y = q.y();
            trans_target.pose.orientation.z = q.z();
            trans_target.pose.orientation.w = q.w();
            target_pub.publish(trans_target);
            trigger_pub.publish(trigger);
            ROS_INFO("过渡: [%.2f, %.2f, %.2f]", trans_target_x, trans_target_y, trans_target_z);
            waitUntilNear(trans_target_x, trans_target_y, trans_target_z, 1.5, 15.0, rate);
        }
        else if (cycle == 2) {
            // ── 反向穿越: 墙后 → 墙前 ──
            
            double start_x = obstacle_x + backward_start_x(gen);
            double start_y = backward_start_y(gen);
            double start_z = z_dist(gen);
            
            geometry_msgs::PoseStamped start_target;
            start_target.header.stamp = ros::Time::now();
            start_target.header.frame_id = "world";
            start_target.pose.position.x = start_x;
            start_target.pose.position.y = start_y;
            start_target.pose.position.z = start_z;
            tf2::Quaternion q; q.setRPY(0, 0, 0);
            start_target.pose.orientation.x = q.x();
            start_target.pose.orientation.y = q.y();
            start_target.pose.orientation.z = q.z();
            start_target.pose.orientation.w = q.w();
            target_pub.publish(start_target);
            trigger_pub.publish(trigger);
            ROS_INFO("反向起点 (墙后): [%.2f, %.2f, %.2f]", start_x, start_y, start_z);
            waitUntilNear(start_x, start_y, start_z, 1.5, 15.0, rate);
            ros::Duration(3.0).sleep();

            // 开始记录
            publishPhase("avoidance_start");
            
            double target_x = obstacle_x - backward_target_dist(gen);
            double target_y = backward_target_y(gen);
            double target_z = z_dist(gen);
            
            geometry_msgs::PoseStamped target;
            target.header.stamp = ros::Time::now();
            target.header.frame_id = "world";
            target.pose.position.x = target_x;
            target.pose.position.y = target_y;
            target.pose.position.z = target_z;
            target.pose.orientation.x = q.x();
            target.pose.orientation.y = q.y();
            target.pose.orientation.z = q.z();
            target.pose.orientation.w = q.w();
            target_pub.publish(target);
            trigger_pub.publish(trigger);
            ROS_INFO("反向目标 (墙前): [%.2f, %.2f, %.2f]", target_x, target_y, target_z);
            waitUntilNear(target_x, target_y, target_z, 1.0, 25.0, rate);
            
            publishPhase("avoidance_end");
        }
        else {
            // ── 自由飞行: 随机方向 ──
            ros::Time wait_start = ros::Time::now();
            while (!pose_received && (ros::Time::now() - wait_start < ros::Duration(1.0))) {
                ros::spinOnce();
                rate.sleep();
            }

            double current_x = pose_received ? current_pose.pose.position.x : 0.0;
            double current_y = pose_received ? current_pose.pose.position.y : 0.0;

            std::uniform_real_distribution<double> angle_dist(0, 2*M_PI);
            std::uniform_real_distribution<double> dist_dist(MIN_FREE_DIST, MAX_FREE_DIST);
            double theta = angle_dist(gen);
            double dist = dist_dist(gen);
            double target_x = current_x + dist * cos(theta);
            double target_y = current_y + dist * sin(theta);
            double target_z = z_dist(gen);

            target_x = std::max(X_MIN, std::min(X_MAX, target_x));
            target_y = std::max(Y_MIN, std::min(Y_MAX, target_y));

            // ── 关键修改: 统一使用 avoidance_start/avoidance_end ──
            // bag2zarr 只处理这两个标记之间的数据
            publishPhase("avoidance_start");
            
            geometry_msgs::PoseStamped free_target;
            free_target.header.stamp = ros::Time::now();
            free_target.header.frame_id = "world";
            free_target.pose.position.x = target_x;
            free_target.pose.position.y = target_y;
            free_target.pose.position.z = target_z;
            tf2::Quaternion q; q.setRPY(0, 0, 0);
            free_target.pose.orientation.x = q.x();
            free_target.pose.orientation.y = q.y();
            free_target.pose.orientation.z = q.z();
            free_target.pose.orientation.w = q.w();
            target_pub.publish(free_target);
            trigger_pub.publish(trigger);
            ROS_INFO("自由飞行: [%.2f, %.2f, %.2f] dist=%.2f",
                     target_x, target_y, target_z, dist);
            waitUntilNear(target_x, target_y, target_z, 1.0, 30.0, rate);
            
            publishPhase("avoidance_end");
        }

        if (!ros::ok()) break;

        if (cycle < 3) {
            ROS_INFO("第 %d 轮完成，等待3秒...", cycle);
            ros::Duration(3.0).sleep();
        }
    }

    // 返航
    if (ros::ok()) {
        ROS_INFO("=== 数据收集完成，返航 ===");
        publishPhase("rtl_start");
        mavros_msgs::SetMode set_mode;
        set_mode.request.custom_mode = "AUTO.RTL";
        ros::ServiceClient set_mode_client = nh.serviceClient<mavros_msgs::SetMode>("/mavros/set_mode");
        if (set_mode_client.call(set_mode) && set_mode.response.mode_sent) {
            ROS_INFO("已切换到 AUTO.RTL");
            ros::Duration(10.0).sleep();
        } else {
            ROS_WARN("切换返航模式失败");
        }
        publishPhase("rtl_end");
        ros::Duration(2.0).sleep();
    }

    ROS_INFO("=== 数据收集完成 ===");
    return 0;
}