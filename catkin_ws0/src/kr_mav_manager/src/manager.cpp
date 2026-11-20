// kr_mav_manager
#include <kr_mav_manager/manager.h>

// Standard C++
#include <math.h>

#include <string>

// ROS Related
#include <actionlib/client/simple_action_client.h>
#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Empty.h>
#include <std_msgs/UInt8.h>
#include <tf/transform_datatypes.h>

// kr_mav_control
#include <kr_tracker_msgs/Transition.h>
#include <kr_tracker_msgs/VelocityGoal.h>

namespace kr_mav_manager
{
// Strings
static const std::string line_tracker_distance("kr_trackers/LineTrackerDistance");
static const std::string line_tracker_min_jerk("kr_trackers/LineTrackerMinJerk");
static const std::string velocity_tracker_str("kr_trackers/VelocityTracker");
static const std::string null_tracker_str("kr_trackers/NullTracker");
static const std::string circle_tracker_str("kr_trackers/CircleTracker");
static const std::string lissajous_tracker_str("kr_trackers/LissajousTracker");
static const std::string lissajous_adder_str("kr_trackers/LissajousAdder");

MAVManager::MAVManager(std::string ns)
    : nh_(ns),
      priv_nh_("~"),
      active_tracker_(""),
      status_(FLYING),
      last_odom_t_(0.0),
      odom_q_(1.0, 0.0, 0.0, 0.0),
      need_odom_(true),
      line_tracker_distance_client_(nh_, "trackers_manager/line_tracker_distance/LineTracker", true),
      line_tracker_min_jerk_client_(nh_, "trackers_manager/line_tracker_min_jerk/LineTracker", true),
      circle_tracker_client_(nh_, "trackers_manager/circle_tracker/CircleTracker", true),
      lissajous_tracker_client_(nh_, "trackers_manager/lissajous_tracker/LissajousTracker", true),
      lissajous_adder_client_(nh_, "trackers_manager/lissajous_adder/LissajousAdder", true)
{
  // Action servers.
  float server_wait_timeout;
  priv_nh_.param("server_wait_timeout", server_wait_timeout, 0.5f);

  if(!line_tracker_distance_client_.waitForServer(ros::Duration(server_wait_timeout)))
  {
    ROS_ERROR("LineTrackerDistance server not found.");
  }

  if(!line_tracker_min_jerk_client_.waitForServer(ros::Duration(server_wait_timeout)))
  {
    ROS_ERROR("LineTrackerMinJerk server not found.");
  }

  // Optional trackers.
  if(!circle_tracker_client_.waitForServer(ros::Duration(server_wait_timeout)))
  {
    ROS_WARN("CircleTracker server not found.");
  }

  if(!lissajous_tracker_client_.waitForServer(ros::Duration(server_wait_timeout)))
  {
    ROS_WARN("LissajousTracker server not found.");
  }

  if(!lissajous_adder_client_.waitForServer(ros::Duration(server_wait_timeout)))
  {
    ROS_WARN("LissajousAdder server not found.");
  }

  pub_status_ = priv_nh_.advertise<std_msgs::UInt8>("status", 10);


  // Subscribers
  odom_sub_ = nh_.subscribe("odom", 10, &MAVManager::odometry_cb, this, ros::TransportHints().tcpNoDelay());
  heartbeat_sub_ = nh_.subscribe("heartbeat", 10, &MAVManager::heartbeat_cb, this, ros::TransportHints().tcpNoDelay());
  tracker_status_sub_ = nh_.subscribe("trackers_manager/status", 10, &MAVManager::tracker_status_cb, this,
                                      ros::TransportHints().tcpNoDelay());

  // Services
  srv_transition_ = nh_.serviceClient<kr_tracker_msgs::Transition>("trackers_manager/transition");

  srv_transition_.waitForExistence();
  if(!this->transition(null_tracker_str))
    ROS_FATAL("Activation of NullTracker failed.");
}

void MAVManager::tracker_done_callback(const actionlib::SimpleClientGoalState &state,
                                       const kr_tracker_msgs::LineTrackerResultConstPtr &result)
{
  ROS_INFO("Goal (%2.2f, %2.2f, %2.2f, %2.2f) finished with state %s after %2.2f s. and %2.2f m.", result->x, result->y,
           result->z, result->yaw, state.toString().c_str(), result->duration, result->length);
}

void MAVManager::circle_tracker_done_callback(const actionlib::SimpleClientGoalState &state,
                                              const kr_tracker_msgs::CircleTrackerResultConstPtr &result)
{
  ROS_INFO("Circle tracking completed after %2.2f seconds.", result->duration);
}

void MAVManager::lissajous_tracker_done_callback(const actionlib::SimpleClientGoalState &state,
                                                 const kr_tracker_msgs::LissajousTrackerResultConstPtr &result)
{
  ROS_INFO("Lissajous tracking completed. Duration: %2.2f seconds, distance: %2.2f m, now located at (%2.2f, %2.2f, "
           "%2.2f, %2.2f).",
           result->duration, result->length, result->x, result->y, result->z, result->yaw);
}

void MAVManager::lissajous_adder_done_callback(const actionlib::SimpleClientGoalState &state,
                                               const kr_tracker_msgs::LissajousAdderResultConstPtr &result)
{
  ROS_INFO("Lissajous tracking completed. Duration: %2.2f seconds, distance: %2.2f m, now located at (%2.2f, %2.2f, "
           "%2.2f, %2.2f).",
           result->duration, result->length, result->x, result->y, result->z, result->yaw);
}

void MAVManager::odometry_cb(const nav_msgs::Odometry::ConstPtr &msg)
{
  pos_(0) = msg->pose.pose.position.x;
  pos_(1) = msg->pose.pose.position.y;
  pos_(2) = msg->pose.pose.position.z;

  vel_(0) = msg->twist.twist.linear.x;
  vel_(1) = msg->twist.twist.linear.y;
  vel_(2) = msg->twist.twist.linear.z;

  odom_q_ = Quat(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x, msg->pose.pose.orientation.y,
                 msg->pose.pose.orientation.z);

  yaw_ = tf::getYaw(msg->pose.pose.orientation);
  yaw_dot_ = msg->twist.twist.angular.z;

  last_odom_t_ = ros::Time::now();

  this->heartbeat();
}

bool MAVManager::goHome()
{
  if(home_set_)
    return this->goTo(home_ + Vec3(0, 0, 0.15), home_yaw_);
  else
  {
    ROS_WARN("Home not set. Cannot go home.");
    return false;
  }
}

bool MAVManager::land()
{
  // if(!this->motors() || status_ != FLYING)
  // {
  //   ROS_WARN("Not landing since the robot is not already flying.");
  //   return false;
  // }

  kr_tracker_msgs::LineTrackerGoal goal;
  goal.x = pos_(0);
  goal.y = pos_(1);
  goal.z = home_(2);
  std::cout << " landing at " << goal.x << " " << goal.y << " " << goal.z << "\n";
  line_tracker_distance_client_.sendGoal(goal, boost::bind(&MAVManager::tracker_done_callback, this, _1, _2),
                                         ClientType::SimpleActiveCallback(), ClientType::SimpleFeedbackCallback());

  return this->transition(line_tracker_distance);
}

bool MAVManager::goTo(float x, float y, float z, float yaw, float v_des, float a_des, bool relative)
{
  // if(!this->motors() || status_ != FLYING)
  // {
  //   ROS_WARN("The robot must be flying before using the goTo method.");
  //   return false;
  // }

  kr_tracker_msgs::LineTrackerGoal goal;
  goal.x = x;
  goal.y = y;

  goal.relative = relative;
  // Convert relative translation in body frame to global frame
  if(relative)
  {
    goal.x = x * std::cos(yaw_) - y * std::sin(yaw_);
    goal.y = x * std::sin(yaw_) + y * std::cos(yaw_);
  }

  goal.z = z;
  goal.yaw = yaw;
  goal.v_des = v_des;
  goal.a_des = a_des;

  line_tracker_min_jerk_client_.sendGoal(goal, boost::bind(&MAVManager::tracker_done_callback, this, _1, _2),
                                         ClientType::SimpleActiveCallback(), ClientType::SimpleFeedbackCallback());

  return this->transition(line_tracker_min_jerk);
}

bool MAVManager::goToTimed(float x, float y, float z, float yaw, float v_des, float a_des, bool relative,
                           ros::Duration duration, ros::Time t_start)
{
  kr_tracker_msgs::LineTrackerGoal goal;
  goal.x = x;
  goal.y = y;

  goal.relative = relative;
  // Convert relative translation in body frame to global frame
  if(relative)
  {
    goal.x = x * std::cos(yaw_) - y * std::sin(yaw_);
    goal.y = x * std::sin(yaw_) + y * std::cos(yaw_);
  }

  goal.z = z;
  goal.yaw = yaw;
  goal.duration = duration;
  goal.t_start = t_start;
  goal.v_des = v_des;
  goal.a_des = a_des;

  line_tracker_min_jerk_client_.sendGoal(goal, boost::bind(&MAVManager::tracker_done_callback, this, _1, _2),
                                         ClientType::SimpleActiveCallback(), ClientType::SimpleFeedbackCallback());
  ROS_INFO("Going to {%2.2f, %2.2f, %2.2f, %2.2f}%s with duration %2.2f", x, y, z, yaw,
           (relative ? " relative to the current position." : ""), duration.toSec());

  return this->transition(line_tracker_min_jerk);
}

bool MAVManager::goTo(Vec4 xyz_yaw, Vec2 v_and_a_des)
{
  return this->goTo(xyz_yaw(0), xyz_yaw(1), xyz_yaw(2), xyz_yaw(3), v_and_a_des(0), v_and_a_des(1));
}
bool MAVManager::goTo(Vec3 xyz, float yaw, Vec2 v_and_a_des)
{
  return this->goTo(xyz(0), xyz(1), xyz(2), yaw, v_and_a_des(0), v_and_a_des(1));
}
bool MAVManager::goTo(Vec3 xyz, Vec2 v_and_a_des)
{
  return this->goTo(xyz(0), xyz(1), xyz(2), yaw_, v_and_a_des(0), v_and_a_des(1));
}
bool MAVManager::goToYaw(float yaw)
{
  return this->goTo(pos_(0), pos_(1), pos_(2), yaw);
}

bool MAVManager::circle(float Ax, float Ay, float T, float duration)
{
  // if(!this->motors() || status_ != FLYING)
  // {
  //   ROS_WARN("The robot must be flying before using the circle method.");
  //   return false;
  // }

  kr_tracker_msgs::CircleTrackerGoal goal;
  goal.Ax = Ax;
  goal.Ay = Ay;
  goal.T = T;
  goal.duration = duration;

  circle_tracker_client_.sendGoal(goal, boost::bind(&MAVManager::circle_tracker_done_callback, this, _1, _2),
                                  CircleClientType::SimpleActiveCallback(), CircleClientType::SimpleFeedbackCallback());

  return this->transition(circle_tracker_str);
}

bool MAVManager::lissajous(float x_amp, float y_amp, float z_amp, float yaw_amp, float x_num_periods,
                           float y_num_periods, float z_num_periods, float yaw_num_periods, float period,
                           float num_cycles, float ramp_time)
{
  // if(!this->motors() || status_ != FLYING)
  // {
  //   ROS_WARN("The robot must be flying to execute a Lissajous.");
  //   return false;
  // }

  kr_tracker_msgs::LissajousTrackerGoal goal;
  goal.x_amp = x_amp;
  goal.y_amp = y_amp;
  goal.z_amp = z_amp;
  goal.yaw_amp = yaw_amp;
  goal.x_num_periods = x_num_periods;
  goal.y_num_periods = y_num_periods;
  goal.z_num_periods = z_num_periods;
  goal.yaw_num_periods = yaw_num_periods;
  goal.period = period;
  goal.num_cycles = num_cycles;
  goal.ramp_time = ramp_time;

  lissajous_tracker_client_.sendGoal(goal, boost::bind(&MAVManager::lissajous_tracker_done_callback, this, _1, _2),
                                     LissajousClientType::SimpleActiveCallback(),
                                     LissajousClientType::SimpleFeedbackCallback());

  return this->transition(lissajous_tracker_str);
}

bool MAVManager::compound_lissajous(float x_amp[2], float y_amp[2], float z_amp[2], float yaw_amp[2],
                                    float x_num_periods[2], float y_num_periods[2], float z_num_periods[2],
                                    float yaw_num_periods[2], float period[2], float num_cycles[2], float ramp_time[2])
{
  // if(!this->motors() || status_ != FLYING)
  // {
  //   ROS_WARN("The robot must be flying to execute a Lissajous.");
  //   return false;
  // }

  kr_tracker_msgs::LissajousAdderGoal goal;
  goal.x_amp[0] = x_amp[0];
  goal.x_amp[1] = x_amp[1];
  goal.y_amp[0] = y_amp[0];
  goal.y_amp[1] = y_amp[1];
  goal.z_amp[0] = z_amp[0];
  goal.z_amp[1] = z_amp[1];
  goal.yaw_amp[0] = yaw_amp[0];
  goal.yaw_amp[1] = yaw_amp[1];
  goal.x_num_periods[0] = x_num_periods[0];
  goal.x_num_periods[1] = x_num_periods[1];
  goal.y_num_periods[0] = y_num_periods[0];
  goal.y_num_periods[1] = y_num_periods[1];
  goal.z_num_periods[0] = z_num_periods[0];
  goal.z_num_periods[1] = z_num_periods[1];
  goal.yaw_num_periods[0] = yaw_num_periods[0];
  goal.yaw_num_periods[1] = yaw_num_periods[1];
  goal.period[0] = period[0];
  goal.period[1] = period[1];
  goal.num_cycles[0] = num_cycles[0];
  goal.num_cycles[1] = num_cycles[1];
  goal.ramp_time[0] = ramp_time[0];
  goal.ramp_time[1] = ramp_time[1];

  lissajous_adder_client_.sendGoal(goal, boost::bind(&MAVManager::lissajous_adder_done_callback, this, _1, _2),
                                   CompoundLissajousClientType::SimpleActiveCallback(),
                                   CompoundLissajousClientType::SimpleFeedbackCallback());

  return this->transition(lissajous_adder_str);
}

// World Velocity commands
bool MAVManager::setDesVelInWorldFrame(float x, float y, float z, float yaw, bool use_position_feedback)
{
  // if(!this->motors() || status_ != FLYING)
  // {
  //   ROS_WARN("The robot must be flying with motors on before setting a desired velocity.");
  //   return false;
  // }

  kr_tracker_msgs::VelocityGoal goal;
  goal.header.stamp = ros::Time::now();
  goal.vx = x;
  goal.vy = y;
  goal.vz = z;
  goal.vyaw = yaw;
  goal.use_position_gains = use_position_feedback;

  pub_goal_velocity_.publish(goal);

  ROS_INFO("Desired World velocity: (%1.4f, %1.4f, %1.4f, %1.4f)", goal.vx, goal.vy, goal.vz, goal.vyaw);

  // Since this could be called quite often,
  // only try to transition if it is not the active tracker.
  if(active_tracker_.compare(velocity_tracker_str) != 0)
  {
    return this->transition(velocity_tracker_str);
  }

  return true;
}

// Body Velocity commands
bool MAVManager::setDesVelInBodyFrame(float x, float y, float z, float yaw, bool use_position_feedback)
{
  Vec3 vel(x, y, z);
  vel = odom_q_ * vel;
  return this->setDesVelInWorldFrame(vel(0), vel(1), vel(2), yaw, use_position_feedback);
}


void MAVManager::tracker_status_cb(const kr_tracker_msgs::TrackerStatus::ConstPtr &msg)
{
  active_tracker_ = msg->tracker;
}

void MAVManager::heartbeat_cb(const std_msgs::Empty::ConstPtr &msg)
{
  this->heartbeat();
}

// TODO: This should be done in a separate thread
void MAVManager::heartbeat()
{
  const float freq = 10;  // Hz

  // Only need to do monitoring at the specified frequency
  ros::Time t = ros::Time::now();
  float dt = (t - last_heartbeat_t_).toSec();
  if(dt < 1 / freq)
    return;
  else
    last_heartbeat_t_ = t;

  // Publish the status
  std_msgs::UInt8 status_msg;
  status_msg.data = status_;
  pub_status_.publish(status_msg);

}


bool MAVManager::hover()
{
  const float a_des(0.8);  //, yaw_a_des(0.1);

  const float v_norm = vel_.norm();

  if(v_norm > 1e-2)
  {
    Vec3 dir = vel_ / v_norm;

    // Acceleration should be opposite the velocity component
    const Vec3 acc = -dir * a_des;

    // acc(3) = - copysign(yaw_a_des, yaw_dot_);

    // vf = vo + a t   ->    t = (vf - vo) / a
    const float t = v_norm / a_des;
    // float t_yaw = - yaw_dot_ / yaw_a_des;

    // xf = xo + vo * t + 1/2 * a * t^2
    Vec4 goal(pos_(0) + vel_(0) * t + 0.5f * acc(0) * t * t, pos_(1) + vel_(1) * t + 0.5f * acc(1) * t * t,
              pos_(2) + vel_(2) * t + 0.5f * acc(2) * t * t,
              yaw_);  //    + yaw_dot_ * t_yaw + 0.5 * yaw_a_des * t_yaw * t_yaw);

    Vec2 v_and_a_des(std::sqrt(vel_.dot(vel_)), a_des);

    ROS_DEBUG("Coasting to hover...");
    return this->goTo(goal, v_and_a_des);
  }

  ROS_DEBUG("Hovering in place...");
  return this->goTo(pos_(0), pos_(1), pos_(2), yaw_);
}

bool MAVManager::ehover()
{
  // if(!this->motors() || status_ != FLYING)
  // {
  //   ROS_WARN("Will not call emergency hover unless the robot is already flying.");
  //   return false;
  // }

  kr_tracker_msgs::LineTrackerGoal goal;
  goal.x = pos_(0);
  goal.y = pos_(1);
  goal.z = pos_(2);
  line_tracker_distance_client_.sendGoal(goal, boost::bind(&MAVManager::tracker_done_callback, this, _1, _2),
                                         ClientType::SimpleActiveCallback(), ClientType::SimpleFeedbackCallback());

  return this->transition(line_tracker_distance);
}

bool MAVManager::transition(const std::string &tracker_str)
{
  // usleep(100000);
  kr_tracker_msgs::Transition transition_cmd;
  transition_cmd.request.tracker = tracker_str;

  if(srv_transition_.call(transition_cmd) && transition_cmd.response.success)
  {
    active_tracker_ = tracker_str;
    ROS_INFO("Current tracker: %s", tracker_str.c_str());
    return true;
  }

  return false;
}

}  // namespace kr_mav_manager
