#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Bool.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>
#include <std_msgs/String.h>
#include <random>
#include <cmath>
#include <tf2/LinearMath/Quaternion.h>
#include <fstream>

mavros_msgs::State current_state;
ros::Publisher phase_pub;  // 发布当前阶段的发布器

void state_cb(const mavros_msgs::State::ConstPtr& msg){
    current_state = *msg;
}

// 发布阶段信息
void publishPhase(const std::string& phase_name) {
    std_msgs::String phase_msg;
    phase_msg.data = phase_name;
    phase_pub.publish(phase_msg);
    ROS_INFO("阶段切换: %s", phase_name.c_str());
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "random_target_generator");
    ros::NodeHandle nh;
    
    ros::Publisher target_pub = nh.advertise<geometry_msgs::PoseStamped>("/move_base_simple/goal", 10);
    ros::Publisher trigger_pub = nh.advertise<std_msgs::Bool>("/traj_start_trigger", 10);
    phase_pub = nh.advertise<std_msgs::String>("/current_phase", 10);  // 新增阶段发布器
    ros::Subscriber state_sub = nh.subscribe<mavros_msgs::State>("mavros/state", 10, state_cb);
    
    double min_height = 1.5, max_height = 2.5;
    double obstacle_x = 8.0;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> z_dist(min_height, max_height);

    // ═══ 改动1: 大幅增加随机范围 ═══
    //
    // 原来:  x = obstacle_x ± 10.0 + 10*U(-0.1,0.1)  → x ∈ [15, 17] 或 [-5, -3]
    //        y = 10*U(-0.2, 0.2)                       → y ∈ [-2, +2]
    //
    // 现在:  x = obstacle_x ± U(5,12) + U(-2,2)       → x ∈ [9, 20] 或 [-8, 3]
    //        y = U(-8, 8)                               → y ∈ [-8, +8]
    //
    // 效果: 从无人机视角看，目标方向从 ±7° 扩大到 ±40°+
    //        EGO 会规划出完全不同的绕墙路径
    //
    std::uniform_real_distribution<double> dist_past_wall(5.0, 10.0);  // 过墙多远
    std::uniform_real_distribution<double> x_jitter(-1.0, 1.0);       // x方向抖动
    std::uniform_real_distribution<double> y_target(-4.0, 4.0);       // y范围大幅增加
    
    // 重置位置的y也随机化，让下一轮起始角度不同
    std::uniform_real_distribution<double> y_reset(-5.0, 5.0);
    
    ROS_INFO("=== 开始绕障运动数据收集 (多样化版) ===");
    ROS_INFO("  墙位置: x=%.1f", obstacle_x);
    ROS_INFO("  目标x: 墙后 5~12m + 抖动±2m");
    ROS_INFO("  目标y: ±8m");
    ROS_INFO("  目标z: %.1f ~ %.1f", min_height, max_height);
    
    // 等待飞控连接
    ROS_INFO("等待飞控连接...");
    ros::Rate rate(10.0);
    while(ros::ok() && !current_state.connected){
        ros::spinOnce();
        rate.sleep();
    }
    
    // 初始延迟
    ROS_INFO("等待10秒系统初始化...");
    ros::Duration(10.0).sleep();
    
    int cycle_count = 0;
    bool front_side = true;

    // ═══ 改动2: 每个bag采集更多轨迹 ═══
    const int MAX_CYCLES = 4;
    
    while (ros::ok() && cycle_count < MAX_CYCLES) {
        cycle_count++;
        ROS_INFO("=== 第 %d/%d 轮绕障运动 ===", cycle_count, MAX_CYCLES);
        
        // 阶段1: 绕障运动 (这是我们主要收集的数据)
        publishPhase("avoidance_start");  // 标记绕障开始
        
        geometry_msgs::PoseStamped avoid_target;
        avoid_target.header.stamp = ros::Time::now();
        avoid_target.header.frame_id = "world";
        
        tf2::Quaternion q_avoid;
        double x, y;
        
        // ═══ 改动3: 目标点随机化 ═══
        double dist = dist_past_wall(gen);
        double jit  = x_jitter(gen);

        if (front_side) {
            x = obstacle_x + dist + jit;
            y = y_target(gen);
            q_avoid.setRPY(0, 0, -M_PI/2);
            ROS_INFO("发布障碍物前方绕障目标 (过墙%.1fm)", dist);
        } else {
            x = obstacle_x - dist + jit;
            y = y_target(gen);
            q_avoid.setRPY(0, 0, M_PI/2);
            ROS_INFO("发布障碍物后方绕障目标 (过墙%.1fm)", dist);
        }

        avoid_target.pose.position.x = x;
        avoid_target.pose.position.y = y;
        avoid_target.pose.position.z = z_dist(gen);
        avoid_target.pose.orientation.x = q_avoid.x();
        avoid_target.pose.orientation.y = q_avoid.y();
        avoid_target.pose.orientation.z = q_avoid.z();
        avoid_target.pose.orientation.w = q_avoid.w();
        
        target_pub.publish(avoid_target);
        
        std_msgs::Bool trigger;
        trigger.data = true;
        trigger_pub.publish(trigger);
        
        ROS_INFO("绕障目标: [%.2f, %.2f, %.2f]", 
                 avoid_target.pose.position.x,
                 avoid_target.pose.position.y,
                 avoid_target.pose.position.z);
        
        // 等待绕障完成 (这个阶段是我们主要收集的数据)
        ROS_INFO("绕障运动中...等待25秒收集数据");
        ros::Duration(25.0).sleep();
        
        publishPhase("avoidance_end");  // 标记绕障结束
        
        if (!ros::ok()) break;
        
        // 阶段2: 重置位置，为下一轮准备不同的起始点
        publishPhase("reset_start");
        
        geometry_msgs::PoseStamped reverse_target;
        reverse_target.header.stamp = ros::Time::now();
        reverse_target.header.frame_id = "world";
        
        tf2::Quaternion q_reverse;

        // ═══ 改动4: 重置位置y也随机化 ═══
        //   这样下一轮飞行的起始位置不同，
        //   即使目标方向相似，绕墙角度也会不同
        if (front_side) {
            reverse_target.pose.position.x = x + 4.0;
            reverse_target.pose.position.y = y_reset(gen);  // 随机y，不再 = y
            q_reverse.setRPY(0, 0, M_PI/2);
        } else {
            reverse_target.pose.position.x = x - 4.0;
            reverse_target.pose.position.y = y_reset(gen);  // 随机y
            q_reverse.setRPY(0, 0, -M_PI/2);
        }
        
        reverse_target.pose.position.z = z_dist(gen);
        reverse_target.pose.orientation.x = q_reverse.x();
        reverse_target.pose.orientation.y = q_reverse.y();
        reverse_target.pose.orientation.z = q_reverse.z();
        reverse_target.pose.orientation.w = q_reverse.w();
        
        target_pub.publish(reverse_target);
        trigger_pub.publish(trigger);
        
        ROS_INFO("重置目标: [%.2f, %.2f, %.2f]", 
                 reverse_target.pose.position.x,
                 reverse_target.pose.position.y,
                 reverse_target.pose.position.z);
        
        // 短暂等待重置完成
        ROS_INFO("重置位置中...等待8秒");
        ros::Duration(8.0).sleep();
        
        publishPhase("reset_end");
        
        if (!ros::ok()) break;
        
        front_side = !front_side;
        ROS_INFO("第 %d 轮绕障数据收集完成", cycle_count);
        
        // 短暂暂停
        if (cycle_count < MAX_CYCLES) {
            ROS_INFO("等待10秒准备下一轮绕障...");
            ros::Duration(10.0).sleep();
        }
    }
    
    // 切换到返航模式（RTL）
    if (ros::ok()) {
        ROS_INFO("=== 完成 %d 轮绕障数据收集，切换到自动返航模式 ===", MAX_CYCLES);
        
        publishPhase("rtl_start");
        
        mavros_msgs::SetMode set_mode;
        set_mode.request.custom_mode = "AUTO.RTL";
        
        ros::ServiceClient set_mode_client = nh.serviceClient<mavros_msgs::SetMode>("/mavros/set_mode");
        
        if (set_mode_client.call(set_mode) && set_mode.response.mode_sent) {
            ROS_INFO("成功切换到 AUTO.RTL 模式 - 无人机将返回起飞点并降落");
            
            // 监控返航状态一段时间
            ros::Time start_time = ros::Time::now();
            while(ros::ok() && (ros::Time::now() - start_time < ros::Duration(10.0))) {
                if (current_state.mode == "AUTO.RTL") {
                    ROS_INFO_THROTTLE(2, "自动返航执行中...");
                }
                ros::spinOnce();
                rate.sleep();
            }
        } else {
            ROS_WARN("切换返航模式失败");
        }
        
        publishPhase("rtl_end");
        ros::Duration(2.0).sleep();
    }
    
    ROS_INFO("=== 绕障数据收集完成 ===");
    return 0;
}