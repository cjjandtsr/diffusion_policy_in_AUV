# catkin_ws0 — ROS simulation workspace

ROS Noetic catkin workspace for the closed-loop simulation of the diffusion-policy UAV project.
Source only: `build/`, `devel/`, bags, spreadsheets and caches are not included.

## Packages (`src/`)

| Package | Role |
|---|---|
| `px4ctrl` | PX4/MAVROS position controller, converts `PositionCommand` trajectories to attitude/thrust |
| `kr_mav_manager` | Trajectory manager and goal interface |
| `trackers` (`kr_trackers*`) | Reference trackers (line, circle, lissajous, smooth-vel) |
| `ego-planner-swarm` | EGO-Planner plus `uav_simulator` (point-cloud sensing, gazebo/quadrotor sim) |
| `custom_target_generator` | Random target / phase publisher used for dataset collection |
| `kr_mav_launch` | Launch files and helper scripts for the MAV stack |
| `offboard_pkg` | Offboard-mode entry nodes (goto, rtl) |

Policy inference nodes live in
`src/ego-planner-swarm/src/planner/plan_manage/scripts/ml_planner_gazebo*.py`
(`ml_planner_node.py`). They subscribe to the point cloud / odometry, run the diffusion policy
and publish the 11-dim trajectory chunk (position + velocity + acceleration) at 50 Hz.

## Scripts (`scripts/`)

- `start_sitl.sh`, `start_sitl_at.sh`, `start_sitl_rl.sh`, `stop_sitl.sh` — PX4 SITL + Gazebo + MAVROS lifecycle
- `ml_gazebo_fm.sh`, `ml_gazebo_fm_auto.sh`, `ml_gazebo.sh` — EGO-Planner + policy closed loop
- `generate_worlds_routeA_rl_v1.py`, `generate_worlds_ood.py`, `generate_worlds_v8_3.py`,
  `validate_routeA_rl_worlds_v1.py`, `calibrate_routeA_rl_worlds_v1.py`,
  `generate_obstacles_json_routeA_rl_v1.py` — evaluation-world generation and validation
- `closed_loop_runner.py`, `rollout_gazebo_eval.py`, `phase_alpha_closed_loop_batch.py` — batch rollout/scoring
- `run_trial.sh`, `run_trial_ego.sh`, `run_one_trial.sh`, `run_batch*.sh` — single/batch trial drivers
- `bag2zarr_body_v5.py`, `merge_zarr_episodes.py` — rosbag → training dataset conversion
- `collision_timeout_monitor.py`, `routea_rl_env_validation.py`, `exec_drift_ablation.py` — run monitors and checks

## Build

```bash
source /opt/ros/noetic/setup.bash
catkin_make            # or: catkin build
source devel/setup.bash
```

PX4-Autopilot and MAVROS are external dependencies (see the top-level README).
On WSL the ROS loopback is unreliable: set `ROS_IP` and `ROS_MASTER_URI` to the primary
NIC address instead of `127.0.0.1`.
