#include <paper_px4_bridge/EvaluationStep.h>
#include <local_planner/local_planner.h>
#include <local_planner/waypoint_generator.h>
#include <pcl_conversions/pcl_conversions.h>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>

std::unique_ptr<avoidance::LocalPlanner> planner;
std::unique_ptr<avoidance::WaypointGenerator> generator;
Eigen::Vector3f previous_goal;
float hfov, vfov;
double max_velocity = 0.5, max_acceleration = 0.25;
// 2026-11-12 (user decision "三种策略都按照官方的设置"): by default the planner
// runs on setDefaultPx4Parameters() alone, i.e. the upstream px4_avoidance
// defaults (local_planner.cpp:190-202: param_mpc_xy_cruise 3.0, acc_hor 5.0,
// z_vel_max_up 3 / dn 1, acc_up/down_max 10). Setting override_px4_params=true
// restores the 2026-11-12 0.5 m/s run's six-parameter override.
bool override_px4_params = false;
bool log_waypoint = false;

// 2026-11-12 fix: the official WaypointGenerator smooths its setpoint with an
// explicit-Euler integrator (waypoint_generator.cpp smoothWaypoint:
//   v += dt*(P*(desired-e) + D*(v_des-v));  e += dt*v
// with P = smoothing_speed_xy = 10, D = 2*sqrt(P) = 6.32). The official nodelet
// drives it at spin_dt_ = 0.1 s (local_planner_nodelet.cpp:18). Calling it once
// per 1.2 s decision (dt = 0.3 s per call) is outside that integrator's stability
// envelope: the xy recursion diverges (measured growth ~1.6x per call, targets
// reached 1e23-1e35 m, native yaw rate 1e32), while z (P = 3) stays bounded --
// exactly the observed asymmetry. Sub-stepping at the official cadence restores
// the module's operating point. The planner call count, goal, point cloud and
// simulated time are unchanged.
static constexpr double kGeneratorSpinDt = 0.1;
static double last_generator_time_s = 0.0;
// 2026-11-12, second fix -- only bites at the official cruise speed:
// adaptSpeed() places its adapted setpoint at `position_ + unit(goto-direction)
// * speed_` (waypoint_generator.cpp, adaptSpeed: pose_to_wp *= speed_), i.e. the
// look-ahead is `speed_` metres, and smoothWaypoint converges to it. The
// adapter then commands the vehicle `speed_ * window` metres per decision, so
// the vehicle overshoots the settled setpoint by speed_*(window-1) and the next
// call measures `position_wp - position` pointing BACKWARDS. At the 0.5 m/s
// override the overshoot is 0.1 m and the smoothing/FOV scaling keeps the sign;
// at the official 3.0 m/s it is 0.6 m and the commanded direction locks
// exactly opposite to the goal. Measured on the frozen-bank smoke:
//   cruise 3.0: median cos(commanded, goal) = -0.82, episode 5141fd5c = -1.00,
//               direction change 0.1 deg/decision (frozen) for 59 decisions
//   cruise 0.5: median cos = +0.74, direction change 3.9-22.7 deg/decision
// With this flag the sub-step loop moves a synthetic vehicle position along
// the last commanded direction at the native speed, which is the invariant the
// official 10 Hz nodelet has (position and clock advance together), so the
// setpoint keeps leading the vehicle instead of falling behind it. Default
// false so the 2026-11-12 0.5 m/s run stays byte-for-byte reproducible.
bool window_vehicle_advance = false;

bool step(paper_px4_bridge::EvaluationStep::Request& req, paper_px4_bridge::EvaluationStep::Response& res) {
  ros::Time::setNow(ros::Time(req.time_s));
  const auto& pose = req.odometry.pose.pose;
  const auto& twist = req.odometry.twist.twist.linear;
  Eigen::Vector3f pos(pose.position.x, pose.position.y, pose.position.z);
  Eigen::Vector3f vel(twist.x, twist.y, twist.z);
  Eigen::Quaternionf q(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
  Eigen::Vector3f goal(req.goal.x, req.goal.y, req.goal.z);
  if (req.initialize) {
    planner.reset(new avoidance::LocalPlanner());
    generator.reset(new avoidance::WaypointGenerator());
    avoidance::LocalPlannerNodeConfig config = avoidance::LocalPlannerNodeConfig::__getDefault__();
    planner->dynamicReconfigureSetParams(config, 0);
    planner->setDefaultPx4Parameters();
    if (override_px4_params) {
      planner->px4_.param_mpc_xy_cruise = max_velocity;
      planner->px4_.param_mpc_z_vel_max_up = max_velocity;
      planner->px4_.param_mpc_z_vel_max_dn = max_velocity;
      planner->px4_.param_mpc_acc_hor = max_acceleration;
      planner->px4_.param_mpc_acc_up_max = max_acceleration;
      planner->px4_.param_mpc_acc_down_max = max_acceleration;
    }
    ROS_INFO("PX4_PARAMS override=%d cruise=%.3f acc_hor=%.3f z_up=%.3f z_dn=%.3f acc_up=%.3f acc_dn=%.3f",
             override_px4_params ? 1 : 0,
             planner->px4_.param_mpc_xy_cruise, planner->px4_.param_mpc_acc_hor,
             planner->px4_.param_mpc_z_vel_max_up, planner->px4_.param_mpc_z_vel_max_dn,
             planner->px4_.param_mpc_acc_up_max, planner->px4_.param_mpc_acc_down_max);
    planner->currently_armed_ = true;
    previous_goal = pos;
    planner->setPreviousGoal(pos);
    planner->setGoal(goal);
    avoidance::FOV fov(0.f, 0.f, hfov, vfov);
    planner->setFOV(0, fov);
    generator->setFOV(0, fov);
  }
  auto start = std::chrono::steady_clock::now();
  planner->setState(pos, vel, q);
  planner->original_cloud_vector_.resize(1);
  pcl::fromROSMsg(req.cloud, planner->original_cloud_vector_[0]);
  planner->runPlanner();
  auto output = planner->getAvoidanceOutput();
  generator->setPlannerInfo(output);
  res.native_speed = output.cruise_velocity;

  // Drive the generator at the official spin_dt_; see kGeneratorSpinDt above.
  int substeps;
  if (req.initialize) {
    last_generator_time_s = req.time_s;
    substeps = 1;
  } else {
    const double elapsed = req.time_s - last_generator_time_s;
    substeps = (elapsed > 0.0) ? static_cast<int>(std::ceil(elapsed / kGeneratorSpinDt)) : 1;
  }
  avoidance::waypointResult wp;
  Eigen::Vector3f moving = pos;
  Eigen::Vector3f heading = Eigen::Vector3f::Zero();
  for (int i = 1; i <= substeps; ++i) {
    const double t = (substeps == 1 || req.initialize)
                         ? req.time_s
                         : last_generator_time_s + (req.time_s - last_generator_time_s) * i / substeps;
    if (window_vehicle_advance && i > 1) {
      const double dt = (req.time_s - last_generator_time_s) / substeps;
      if (heading.squaredNorm() > 1e-12 && std::isfinite(output.cruise_velocity) && output.cruise_velocity > 0)
        moving += heading * static_cast<float>(output.cruise_velocity * dt);
    }
    ros::Time::setNow(ros::Time(t));
    generator->updateState(moving, q, goal, previous_goal, vel, false, true,
        avoidance::NavigationState::offboard, false, false, Eigen::Vector3f::Zero());
    wp = generator->getWaypoints();
    if (window_vehicle_advance) {
      Eigen::Vector3f d = wp.goto_position - moving;
      if (d.allFinite() && d.squaredNorm() > 1e-12) heading = d.normalized();
    }
  }
  last_generator_time_s = req.time_s;
  if (log_waypoint) {
    // Diagnostic only (log_waypoint:=true): the generator's raw answer, before
    // the adapter's constant-direction extrapolation. state 0=TRY_PATH
    // 1=ALTITUDE_CHANGE 2=LOITER 3=DIRECT.
    ROS_INFO("WGPROBE t=%.2f pos=[%.2f %.2f %.2f] goal=[%.2f %.2f %.2f] goto=[%.2f %.2f %.2f] wp=[%.2f %.2f %.2f] state=%d nsub=%d",
             req.time_s, pos.x(), pos.y(), pos.z(), goal.x(), goal.y(), goal.z(),
             wp.goto_position.x(), wp.goto_position.y(), wp.goto_position.z(),
             wp.position_wp.x(), wp.position_wp.y(), wp.position_wp.z(),
             static_cast<int>(wp.waypoint_type), substeps);
  }
  res.position.x = wp.position_wp.x(); res.position.y = wp.position_wp.y(); res.position.z = wp.position_wp.z();
  res.velocity.x = wp.linear_velocity_wp.x(); res.velocity.y = wp.linear_velocity_wp.y(); res.velocity.z = wp.linear_velocity_wp.z();
  const auto& o = wp.orientation_wp;
  res.yaw = std::atan2(2*(o.w()*o.z()+o.x()*o.y()), 1-2*(o.y()*o.y()+o.z()*o.z()));
  // 2026-11-12: waypointResult::angular_velocity_wp is only written by
  // transformPositionToVelocityWaypoint(), which the position-control states used
  // here never call. Reading it returned uninitialised memory (a constant 8.13e32
  // in all 55k recorded updates). Report the undefined axis explicitly instead,
  // matching the NaN velocity-mask convention already used for linear velocity.
  res.yaw_rate = std::numeric_limits<double>::quiet_NaN();
  res.planning_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  return true;
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "paper_px4_bridge");
  ros::NodeHandle nh("~");
  double horizontal, vertical;
  nh.getParam("horizontal_fov_deg", horizontal); nh.getParam("vertical_fov_deg", vertical);
  nh.param<double>("max_velocity", max_velocity, max_velocity);
  nh.param<double>("max_acceleration", max_acceleration, max_acceleration);
  nh.param<bool>("override_px4_params", override_px4_params, false);
  nh.param<bool>("log_waypoint", log_waypoint, false);
  nh.param<bool>("window_vehicle_advance", window_vehicle_advance, false);
  hfov = horizontal; vfov = vertical;
  ROS_INFO("paper_px4_bridge: override_px4_params=%d window_vehicle_advance=%d",
           override_px4_params ? 1 : 0, window_vehicle_advance ? 1 : 0);
  auto server = nh.advertiseService("step", step);
  ros::spin();
}
