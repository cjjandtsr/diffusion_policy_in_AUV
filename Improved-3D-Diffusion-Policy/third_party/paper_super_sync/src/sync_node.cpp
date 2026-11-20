// Paper adapter: synchronous SUPER evaluation service.
//
// Drives the OFFICIAL SUPER FSM/map/planner stack directly from service
// requests. The node creates no ROS timers and no subscribers, so wall-clock
// time can never advance planner state between requests: every official time
// query (Fsm::ros_ptr_->getSimTime, ROGMap::getSystemWalltimeNow) resolves to
// the logical time carried by the request currently being handled. The
// handler is single-threaded by design (one synchronous request at a time).
//
// Official upstream files are unmodified; this file is a new adapter.
#include <ros/ros.h>

#include <chrono>
#include <cstring>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <pcl_conversions/pcl_conversions.h>

#include <super_core/super_planner.h>
#include <fsm/fsm.h>
#include <rog_map_ros/rog_map_ros1.hpp>

#include <paper_super_sync/EvaluationStep.h>

namespace {

// Logical time of the request currently being handled. Set at the top of the
// service handler; read by the two overrides below. Single-threaded use.
double g_sync_time_s = 0.0;

class SyncRosInterface : public ros_interface::Ros1Interface {
public:
    explicit SyncRosInterface(const ros::NodeHandle &nh) : Ros1Interface(nh) {}
    void setSimTime(const double &sim_time) override { g_sync_time_s = sim_time; }
    double getSimTime() override { return g_sync_time_s; }
    void getSimTime(int32_t &sec, uint32_t &nsec) override {
        sec = static_cast<int32_t>(std::floor(g_sync_time_s));
        nsec = static_cast<uint32_t>((g_sync_time_s - sec) * 1e9);
    }
};

class SyncROGMapROS : public rog_map::ROGMapROS {
public:
    SyncROGMapROS(const ros::NodeHandle &nh, const std::string &cfg_path)
        : ROGMapROS(nh, cfg_path) {}

private:
    const double getSystemWalltimeNow() override { return g_sync_time_s; }

public:
    // Episode isolation without re-constructing the map: ProbMap::initProbMap()
    // carries a process-wide one-shot guard, so the official map object is built
    // exactly once per process and cleared in place here. The clear follows the
    // official mapSliding() path used when the robot teleports out of the map
    // (resetLocalMap + reset the inflation map), so no official file is touched.
    void resetEpisodeMap(const super_utils::Vec3f &start_pos) {
        resetLocalMap();
        inf_map_->resetLocalMap();
        // Re-centre the local map on this episode's start pose, exactly as
        // ROGMap::init() does with (0,0,0). Without this the rolling window (and
        // therefore the raycast update box clipped to the map bounds) would still
        // be anchored on the previous episode's last pose.
        mapSliding(start_pos);
        inf_map_->mapSliding(start_pos);
        // ProbMap::updateMap() and InfMap treat the very first update specially
        // (map_empty_ gates the virtual ceiling/ground stamping and the first
        // inflation). Restoring the pristine flag is what makes every episode
        // behave exactly like a fresh process.
        map_empty_ = true;
    }

    // The official updateProbMap() clears the unknown cells inside
    // raycast_range_min around the robot on the first frame only, guarded by a
    // function-local static. Replaying that ball after the first map update of
    // every episode keeps later episodes on exactly the same footing as the
    // first one (the update saturates at l_min, so applying it twice is a no-op).
    void clearUnknownBallAt(const super_utils::Vec3f &pos) {
        for (double dx = -cfg_.raycast_range_min; dx <= cfg_.raycast_range_min;
             dx += cfg_.resolution) {
            for (double dy = -cfg_.raycast_range_min; dy <= cfg_.raycast_range_min;
                 dy += cfg_.resolution) {
                for (double dz = -cfg_.raycast_range_min; dz <= cfg_.raycast_range_min;
                     dz += cfg_.resolution) {
                    const super_utils::Vec3f offset(dx, dy, dz);
                    if (offset.norm() <= cfg_.raycast_range_min) {
                        const super_utils::Vec3f pp = pos + offset;
                        const int hash_id = getHashIndexFromPos(pp);
                        missPointUpdate(pp, hash_id, 999);
                    }
                }
            }
        }
    }

    // Objective episode-isolation fingerprint over the official occupancy
    // buffer, expressed in the official grid classes (isOccupied/isKnownFree/
    // isUnknown). Cell *counts* are used instead of a raw byte hash because the
    // raycast accumulates log-odds floats in an internally reordered queue, so
    // the low bits of individual cells are not bit-stable; the classification of
    // 30M cells is.
    uint64_t occupancyFingerprint(int64_t &occupied, int64_t &known_free,
                                  int64_t &unknown) const {
        occupied = 0;
        known_free = 0;
        unknown = 0;
        const float *data = occupancy_buffer_.data();
        const size_t n = occupancy_buffer_.size();
        for (size_t i = 0; i < n; ++i) {
            const double prob = static_cast<double>(data[i]);
            if (isOccupied(prob)) {
                ++occupied;
            } else if (isKnownFree(prob)) {
                ++known_free;
            } else {
                ++unknown;
            }
        }
        uint64_t h = 1469598103934665603ULL;
        for (int64_t v : {occupied, known_free, unknown}) {
            h = (h ^ static_cast<uint64_t>(v)) * 1099511628211ULL;
        }
        return h;
    }
};

class FsmSync : public fsm::Fsm {
public:
    ros::NodeHandle nh_;
    std::string cfg_path_;
    // The base Fsm holds no map member; the official FsmRos1 declares its own
    // (fsm_ros1.hpp). Same here: the map is rebuilt on every episode reset.
    std::shared_ptr<SyncROGMapROS> map_ptr_;
    int episode_counter_{0};
    bool traj_ready_{false};
    geometry_utils::Trajectory committed_pos_, committed_yaw_;
    super_utils::Vec3f last_goal_{super_utils::Vec3f::Zero()};

    void init(const ros::NodeHandle &nh, const std::string &cfg_path) {
        nh_ = nh;
        cfg_path_ = cfg_path;
        cfg_ = fsm::Config(cfg_path);
        buildStack();
    }

    void buildStack() {
        // The official map may be constructed only once per process (see
        // ProbMap::initProbMap), so it is created on first use and then reused.
        if (!map_ptr_) {
            map_ptr_ = std::make_shared<SyncROGMapROS>(nh_, cfg_path_);
        }
        planner_ptr_.reset();
        ros_ptr_.reset();
        ros_ptr_ = std::make_shared<SyncRosInterface>(nh_);
        planner_ptr_ = std::make_shared<super_planner::SuperPlanner>(
            cfg_path_, ros_ptr_, map_ptr_);
    }

    void resetEpisode(const super_utils::Vec3f &start_pos) {
        if (map_ptr_) {
            map_ptr_->resetEpisodeMap(start_pos);
        }
        buildStack();
        machine_state_ = INIT;
        started_ = false;
        plan_from_rest_ = false;
        finish_plan = false;
        traj_finish_ = false;
        gi_.new_goal = false;
        gi_.goal_p = super_utils::Vec3f::Zero();
        gi_.goal_yaw = NAN;
        last_goal_ = super_utils::Vec3f::Zero();
        replan_logs_.clear();
        log_module_time.assign(9, 0.0);
        traj_ready_ = false;
        committed_pos_ = geometry_utils::Trajectory();
        committed_yaw_ = geometry_utils::Trajectory();
        ++episode_counter_;
    }

    // Official Fsm abstract members: capture instead of publishing.
    void publishPolyTraj() override {
        planner_ptr_->lockCommittedTraj();
        committed_pos_ = planner_ptr_->getCommittedPositionTrajectory();
        committed_yaw_ = planner_ptr_->getCommittedYawTrajectory();
        planner_ptr_->unlockCommittedTraj();
        traj_ready_ = !committed_pos_.empty();
    }
    void publishCurPoseToPath() override {}
    void resetVisualizedPath() override {}

    void updateMap(const rog_map::PointCloud &cloud,
                   const super_utils::Pose &pose) {
        planner_ptr_->updateROGMap(cloud, pose);
    }

    bool stepService(paper_super_sync::EvaluationStep::Request &req,
                     paper_super_sync::EvaluationStep::Response &res) {
        const auto wall_start = std::chrono::steady_clock::now();
        g_sync_time_s = req.time_s;
        res.requested_goal = req.goal;

        const auto &odom = req.odometry.pose.pose;
        super_utils::Pose pose;
        pose.first = super_utils::Vec3f(odom.position.x, odom.position.y,
                                        odom.position.z);
        pose.second = super_utils::Quatf(odom.orientation.w, odom.orientation.x,
                                         odom.orientation.y, odom.orientation.z);

        if (req.initialize) {
            resetEpisode(pose.first);
        }

        // 1. Feed the real local observation into the official map; this also
        //    updates the official robot state (position/quaternion/rcv_time)
        //    exactly like the upstream odom callback (velocity stays 0 there).
        rog_map::PointCloud cloud;
        pcl::fromROSMsg(req.cloud, cloud);
        updateMap(cloud, pose);
        if (req.initialize) {
            map_ptr_->clearUnknownBallAt(pose.first);
        }

        // 2. Official goal handling: set once per episode, retried while the
        //    goal has not been accepted (upstream skips a deeply-occupied
        //    goal; the map grows each decision so it can become feasible).
        super_utils::Vec3f goal(req.goal.x, req.goal.y, req.goal.z);
        const bool goal_changed = !goal.isApprox(last_goal_);
        if (req.initialize || goal_changed || (!started_ && !gi_.new_goal)) {
            setGoalPosiAndYaw(goal, super_utils::Quatf(NAN, NAN, NAN, NAN));
            last_goal_ = goal;
        }
        res.native_goal.x = gi_.goal_p.x();
        res.native_goal.y = gi_.goal_p.y();
        res.native_goal.z = gi_.goal_p.z();
        res.goal_accepted = started_ && gi_.new_goal;
        res.goal_adjusted = res.goal_accepted && !gi_.goal_p.isApprox(goal);

        // 3. Drive the official state machine until it settles. GENERATE_TRAJ
        //    runs PlanFromRest synchronously inside callMainFsmOnce, so this
        //    loop replicates the upstream 100 Hz main-FSM settling at one
        //    logical instant; it is bounded to bound per-decision wall time.
        // INIT -> WAIT_GOAL -> GENERATE_TRAJ -> FOLLOW_TRAJ are separate official
        // ticks, so the loop stops only when the state stops changing (settled or
        // a failing transition) or when the machine reaches a terminal state.
        int iterations = 0;
        bool planned_from_rest = false;
        fsm::Fsm::MACHINE_STATE prev_state = machine_state_;
        for (; iterations < 8; ++iterations) {
            prev_state = machine_state_;
            if (prev_state == GENERATE_TRAJ) {
                planned_from_rest = true;
            }
            callMainFsmOnce();
            if (machine_state_ == FOLLOW_TRAJ || machine_state_ == EMER_STOP) {
                break;
            }
            if (machine_state_ == prev_state) {
                break;
            }
        }
        // 4. One replan per decision boundary (the common H4 1.2 s cadence;
        //    the upstream 15 Hz replan timer is documented as a protocol
        //    deviation of the shared evaluation).
        // A trajectory produced by PlanFromRest in this same instant already is
        // the fresh plan; replanning again immediately would double the planning
        // calls per decision, so it is skipped.
        if (machine_state_ == FOLLOW_TRAJ && !planned_from_rest
            && req.decision_boundary) {
            callReplanOnce();
            double swt = 0.0;
            bool traj_finish = false;
            planner_ptr_->getOneHeartbeatTime(swt, traj_finish);
            traj_finish_ = traj_finish;
            if (traj_finish) {
                // Upstream cmd-publish semantics (fsm_ros1.hpp): finished
                // trajectory -> wait at goal or plan again from rest.
                if (closeToGoal(0.1)) {
                    ChangeState("sync_step", WAIT_GOAL);
                } else {
                    ChangeState("sync_step", GENERATE_TRAJ);
                }
                for (; iterations < 8; ++iterations) {
                    prev_state = machine_state_;
                    callMainFsmOnce();
                    if (machine_state_ == FOLLOW_TRAJ ||
                        machine_state_ == EMER_STOP ||
                        machine_state_ == prev_state) {
                        break;
                    }
                }
            }
        }

        // 5. Respond with the committed native trajectory and full status.
        res.state = MACHINE_STATE_STR[static_cast<int>(machine_state_)];
        res.state_iterations = iterations;
        res.traj_finish = traj_finish_;
        res.has_trajectory = traj_ready_ &&
                             machine_state_ == FOLLOW_TRAJ && !traj_finish_;
        if (traj_ready_) {
            const int piece_num = committed_pos_.getPieceNum();
            res.order_pos = 7;
            res.time_pos.resize(piece_num);
            res.coef_pos_x.resize(piece_num * 8);
            res.coef_pos_y.resize(piece_num * 8);
            res.coef_pos_z.resize(piece_num * 8);
            for (int i = 0; i < piece_num; ++i) {
                const Eigen::Matrix<double, 3, 8> coef =
                    committed_pos_[i].getCoeffMat();
                Eigen::Map<Eigen::VectorXd>(&res.coef_pos_x[8 * i], 8) =
                    coef.row(0);
                Eigen::Map<Eigen::VectorXd>(&res.coef_pos_y[8 * i], 8) =
                    coef.row(1);
                Eigen::Map<Eigen::VectorXd>(&res.coef_pos_z[8 * i], 8) =
                    coef.row(2);
                res.time_pos[i] = committed_pos_[i].getDuration();
            }
            res.start_WT_pos = ros::Time(committed_pos_.start_WT);
            if (!committed_yaw_.empty()) {
                const int yaw_num = committed_yaw_.getPieceNum();
                res.order_yaw = 7;
                res.time_yaw.resize(yaw_num);
                res.coef_yaw.resize(yaw_num * 8);
                for (int i = 0; i < yaw_num; ++i) {
                    Eigen::VectorXd ycoef = committed_yaw_[i].getCoeffMat().row(0);
                    Eigen::Map<Eigen::VectorXd>(&res.coef_yaw[8 * i], 8) = ycoef;
                    res.time_yaw[i] = committed_yaw_[i].getDuration();
                }
                res.start_WT_yaw = ros::Time(committed_yaw_.start_WT);
            }
            super_utils::StatePVAJ pvaj;
            double yaw = 0.0, yaw_dot = 0.0;
            bool on_backup = false, tf = false;
            planner_ptr_->getOneCommandFromTraj(pvaj, yaw, yaw_dot, on_backup, tf);
            res.native_yaw = yaw;
            res.native_yaw_dot = yaw_dot;
            res.trajectory_flag = on_backup ? 2 : 1;
        } else {
            res.order_pos = 0;
            res.order_yaw = 0;
            res.trajectory_flag = 0;
            res.native_yaw = NAN;
            res.native_yaw_dot = NAN;
        }

        res.replan_count = static_cast<int32_t>(replan_logs_.size());
        int32_t failed = 0;
        int32_t ret = -1;
        for (const auto &entry : replan_logs_) {
            const int32_t rc = static_cast<int32_t>(entry.getRetCode());
            // FAILED=0, EMER=5, OPT_FAILED=6, INIT_ERROR=7 in the official enum.
            if (rc == 0 || rc == 5 || rc == 6 || rc == 7) {
                ++failed;
            }
        }
        if (!replan_logs_.empty()) {
            ret = static_cast<int32_t>(replan_logs_.back().getRetCode());
        }
        res.failed_plans = failed;
        res.ret_code = ret;
        res.ret_code_str = (ret >= 0 &&
                            ret < static_cast<int32_t>(
                                      super_utils::RET_CODE_STR.size()))
                               ? super_utils::RET_CODE_STR[ret]
                               : "UNKNOWN";
        if (req.diagnostics) {
            int64_t occ = 0, known_free = 0, unknown = 0;
            res.map_checksum = map_ptr_->occupancyFingerprint(occ, known_free, unknown);
            res.map_occupied = occ;
            res.map_known_free = known_free;
            res.map_unknown = unknown;
        } else {
            res.map_checksum = 0ULL;
            res.map_occupied = -1;
            res.map_known_free = -1;
            res.map_unknown = -1;
        }
        res.module_seconds = log_module_time;
        res.planning_seconds = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - wall_start)
                                   .count();
        return true;
    }
};

}  // namespace

int main(int argc, char **argv) {
    ros::init(argc, argv, "paper_super_sync_node");
    ros::NodeHandle pnh("~");
    std::string cfg_path;
    pnh.param("config_path", cfg_path, std::string(""));
    if (cfg_path.empty()) {
        ROS_ERROR("paper_super_sync_node requires the config_path parameter");
        return 1;
    }
    pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);

    auto fsm = std::make_shared<FsmSync>();
    fsm->init(pnh, cfg_path);
    ROS_INFO("paper_super_sync ready (config: %s, episode 0)",
             cfg_path.c_str());

    ros::NodeHandle nh;
    ros::ServiceServer service = nh.advertiseService(
        "/paper_super_sync/step", &FsmSync::stepService, fsm.get());
    ROS_INFO("paper_super_sync service: /paper_super_sync/step");
    ros::spin();
    return 0;
}
