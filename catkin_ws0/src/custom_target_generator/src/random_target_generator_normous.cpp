#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Bool.h>
#include <std_msgs/String.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>
#include <random>
#include <cmath>
#include <tf2/LinearMath/Quaternion.h>

/*
 * random_target_generator_v12 – 无硬编码障碍物版本，去掉第一个cycle的单独start飞行
 * 
 * 修改点:
 *   - 移除第一个 cycle 中采样并飞向 start 的过程
 *   - 第一个有效避障轨迹起点固定为 warmup 点 (-8,0,2)
 *   - 后续 cycle 仍以上一个 goal 作为起点
 *   - 所有避障段均正确发布 avoidance_start/end 消息
 */

// ========== 全局参数 ==========
mavros_msgs::State current_state;
ros::Publisher phase_pub;
geometry_msgs::PoseStamped current_pose;
bool pose_received = false;

// 采样范围与约束
const double MAP_MIN_XY = -12.0;
const double MAP_MAX_XY =  12.0;
const double MIN_Z = 1.0;
const double MAX_Z = 3.0;                 // 飞行高度范围
const double MIN_DIST = 12.0;             // start ↔ goal 最小水平距离
const double SPAWN_WORLD_X = -12.0;       // 起飞点世界 X
const double SPAWN_WORLD_Y =  0.0;        // 起飞点世界 Y
const double MIN_SPAWN_DIST = 3.0;        // 仅用于历史，此处不再使用（第一个起点固定）

const double WARMUP_WORLD_X = -8.0;
const double WARMUP_WORLD_Y =  0.0;
const double WARMUP_WORLD_Z =  2.0;

const int MAX_CYCLES = 5;
const int MAX_SAMPLE_ATTEMPTS = 20;

const double REACH_THRESHOLD = 1.5;       // 到达判定阈值 (米)
const double TIMEOUT_SEC = 50.0;          // 单次飞行超时 (秒)
const double STABLE_WAIT_SEC = 2.0;       // 到达后额外等待时间，确保停稳

// 坐标系偏移（世界 -> 局部）：局部坐标 = 世界坐标 - 起飞点坐标
const double OFFSET_X = SPAWN_WORLD_X;
const double OFFSET_Y = SPAWN_WORLD_Y;
const double OFFSET_Z = 0.0;

// ---------- 辅助函数 ----------
inline void worldToLocal(double wx, double wy, double wz,
                         double &lx, double &ly, double &lz) {
    lx = wx - OFFSET_X;
    ly = wy - OFFSET_Y;
    lz = wz - OFFSET_Z;
}

void state_cb(const mavros_msgs::State::ConstPtr& msg) {
    current_state = *msg;
}

void local_pos_cb(const geometry_msgs::PoseStamped::ConstPtr& msg) {
    current_pose = *msg;
    pose_received = true;
}

void publishPhase(ros::Publisher& pub, const std::string& name) {
    std_msgs::String msg;
    msg.data = name;
    pub.publish(msg);
    ROS_INFO("[Phase] %s", name.c_str());
}

double dist2d(double x1, double y1, double x2, double y2) {
    double dx = x1 - x2, dy = y1 - y2;
    return std::sqrt(dx*dx + dy*dy);
}

// 等待到达局部坐标点，到达后额外等待 STABLE_WAIT_SEC
bool waitUntilNearAndStable(double lx, double ly, double lz,
                            double threshold, double timeout, ros::Rate& rate) {
    ros::Time start = ros::Time::now();
    bool reached = false;
    while (ros::ok()) {
        ros::spinOnce();
        rate.sleep();
        if (!pose_received) continue;
        double d = std::sqrt(
            std::pow(current_pose.pose.position.x - lx, 2) +
            std::pow(current_pose.pose.position.y - ly, 2) +
            std::pow(current_pose.pose.position.z - lz, 2));
        if (!reached && d < threshold) {
            ROS_INFO("  reached (error=%.2f m), waiting %.1f sec to stabilize",
                     d, STABLE_WAIT_SEC);
            ros::Duration(STABLE_WAIT_SEC).sleep();
            return true;
        }
        if ((ros::Time::now() - start).toSec() > timeout) {
            ROS_WARN("  timeout (error=%.2f m)", d);
            return false;
        }
    }
    return false;
}

// 发布局部目标点（frame_id = "map"，但数值为局部坐标偏移）
void sendLocalTarget(ros::Publisher& target_pub, ros::Publisher& trigger_pub,
                     double lx, double ly, double lz, double yaw = 0.0) {
    geometry_msgs::PoseStamped t;
    t.header.stamp = ros::Time::now();
    t.header.frame_id = "map";
    t.pose.position.x = lx;
    t.pose.position.y = ly;
    t.pose.position.z = lz;
    tf2::Quaternion q;
    q.setRPY(0, 0, yaw);
    t.pose.orientation.x = q.x();
    t.pose.orientation.y = q.y();
    t.pose.orientation.z = q.z();
    t.pose.orientation.w = q.w();
    target_pub.publish(t);
    std_msgs::Bool trig;
    trig.data = true;
    trigger_pub.publish(trig);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "random_target_generator_v12");
    ros::NodeHandle nh;

    ros::Publisher target_pub = nh.advertise<geometry_msgs::PoseStamped>("/move_base_simple/goal", 10);
    ros::Publisher trigger_pub = nh.advertise<std_msgs::Bool>("/traj_start_trigger", 10);
    phase_pub = nh.advertise<std_msgs::String>("/current_phase", 10);

    ros::Subscriber state_sub = nh.subscribe<mavros_msgs::State>("mavros/state", 10, state_cb);
    ros::Subscriber pose_sub = nh.subscribe<geometry_msgs::PoseStamped>("mavros/local_position/pose", 10, local_pos_cb);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> xy_dist(MAP_MIN_XY, MAP_MAX_XY);
    std::uniform_real_distribution<double> z_dist(MIN_Z, MAX_Z);

    ROS_INFO("=== v12 动态世界数据采集（无硬编码障碍物，第一个避障段从warmup点开始）===");
    ROS_INFO("World sampling box: [%.1f, %.1f] x [%.1f, %.1f], Z=[%.1f, %.1f]",
             MAP_MIN_XY, MAP_MAX_XY, MAP_MIN_XY, MAP_MAX_XY, MIN_Z, MAX_Z);
    ROS_INFO("Local coordinate origin: world (%.1f, %.1f, %.1f)",
             SPAWN_WORLD_X, SPAWN_WORLD_Y, OFFSET_Z);
    ROS_INFO("Stable wait: %.1f sec", STABLE_WAIT_SEC);
    ROS_INFO("Cycles per bag: %d", MAX_CYCLES);
    ROS_INFO("First cycle start = warmup point (%.1f, %.1f, %.1f)", 
             WARMUP_WORLD_X, WARMUP_WORLD_Y, WARMUP_WORLD_Z);
    ROS_INFO("Subsequent cycles start from previous goal.");

    // 等待飞控连接
    ros::Rate rate(10.0);
    while (ros::ok() && !current_state.connected) {
        ros::spinOnce();
        rate.sleep();
    }
    ROS_INFO("FCU connected");
    ros::Duration(2.0).sleep();

    // ---------- Warmup (不被记录) ----------
    double warmup_lx, warmup_ly, warmup_lz;
    worldToLocal(WARMUP_WORLD_X, WARMUP_WORLD_Y, WARMUP_WORLD_Z,
                 warmup_lx, warmup_ly, warmup_lz);
    ROS_INFO("Warmup -> local (%.1f, %.1f, %.1f)", warmup_lx, warmup_ly, warmup_lz);
    sendLocalTarget(target_pub, trigger_pub, warmup_lx, warmup_ly, warmup_lz);
    waitUntilNearAndStable(warmup_lx, warmup_ly, warmup_lz,
                           REACH_THRESHOLD, TIMEOUT_SEC, rate);
    ros::Duration(1.0).sleep();

    // ---------- 主循环 ----------
    double prev_goal_x_w = 0.0, prev_goal_y_w = 0.0, prev_goal_z_w = 0.0; // 上一个goal的世界坐标

    for (int cycle = 1; cycle <= MAX_CYCLES; cycle++) {
        ROS_INFO("========== Cycle %d / %d ==========", cycle, MAX_CYCLES);

        double sx_w, sy_w, sz_w;   // 当前起点（世界坐标）

        if (cycle == 1) {
            // 第一个cycle：起点固定为 warmup 点
            sx_w = WARMUP_WORLD_X;
            sy_w = WARMUP_WORLD_Y;
            sz_w = WARMUP_WORLD_Z;
            ROS_INFO("Cycle 1 start = warmup point: world(%.2f, %.2f, %.2f)", sx_w, sy_w, sz_w);
        } else {
            // 后续 cycle：以上一个 goal 作为起点（无需额外飞行）
            sx_w = prev_goal_x_w;
            sy_w = prev_goal_y_w;
            sz_w = prev_goal_z_w;
            ROS_INFO("Cycle %d start = previous goal: world(%.2f, %.2f, %.2f)", cycle, sx_w, sy_w, sz_w);
        }

        // 采样 goal（只检查与起点距离，不检查障碍物）
        double gx_w, gy_w, gz_w;
        bool goal_ok = false;
        for (int att = 0; att < MAX_SAMPLE_ATTEMPTS; att++) {
            gx_w = xy_dist(gen);
            gy_w = xy_dist(gen);
            gz_w = z_dist(gen);
            if (dist2d(sx_w, sy_w, gx_w, gy_w) >= MIN_DIST) {
                goal_ok = true;
                break;
            }
        }
        if (!goal_ok) {
            ROS_ERROR("Cycle %d: cannot sample valid goal (too close to start)", cycle);
            continue;
        }

        // 录制飞向 goal
        double gx_l, gy_l, gz_l;
        worldToLocal(gx_w, gy_w, gz_w, gx_l, gy_l, gz_l);
        ROS_INFO("Fly to goal: world(%.2f, %.2f, %.2f) -> local(%.2f, %.2f, %.2f), dist=%.1f m",
                 gx_w, gy_w, gz_w, gx_l, gy_l, gz_l, dist2d(sx_w, sy_w, gx_w, gy_w));

        publishPhase(phase_pub, "avoidance_start");
        sendLocalTarget(target_pub, trigger_pub, gx_l, gy_l, gz_l);
        waitUntilNearAndStable(gx_l, gy_l, gz_l,
                               REACH_THRESHOLD, TIMEOUT_SEC, rate);
        publishPhase(phase_pub, "avoidance_end");

        // 更新上一个 goal 为当前 goal
        prev_goal_x_w = gx_w;
        prev_goal_y_w = gy_w;
        prev_goal_z_w = gz_w;

        ros::Duration(2.0).sleep();
    }

    // ---------- RTL ----------
    if (ros::ok()) {
        publishPhase(phase_pub, "rtl_start");
        mavros_msgs::SetMode set_mode;
        set_mode.request.custom_mode = "AUTO.RTL";
        ros::ServiceClient client = nh.serviceClient<mavros_msgs::SetMode>("/mavros/set_mode");
        if (client.call(set_mode) && set_mode.response.mode_sent) {
            ROS_INFO("RTL command sent");
            ros::Duration(10.0).sleep();
        } else {
            ROS_WARN("RTL failed");
        }
        publishPhase(phase_pub, "rtl_end");
    }

    ROS_INFO("=== Data collection finished ===");
    return 0;
}