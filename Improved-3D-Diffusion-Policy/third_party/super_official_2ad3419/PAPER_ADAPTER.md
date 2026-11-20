# SUPER paper adapter

Date: 2026-11-12. Official commit: 2ad3419c127a617c6d7df6925e81a14175a9c096.
All fetched upstream files are unmodified. `PAPER_DOWNLOAD_MANIFEST.json` records the archive hash, each retained file hash, and omitted demo assets. The local sensor adapter does not load an official demonstration map.

## Build

Use an isolated ROS Noetic catkin workspace. Link this source tree into its `src`, run `bash scripts/select_ros_version.sh ROS1` from the linked source, then build with `catkin_make -DCMAKE_BUILD_TYPE=Release`. The upstream selector copies ROS1 build templates; preserve those build-time adaptations in the run identity. Do not overlay SUPER's `quadrotor_msgs` over the existing flight controller process: the two `PositionCommand` definitions differ.

Generate a configuration with `scripts/prepare_paper_super.py config --help`. All vehicle limits, altitude bounds, range limits and input topics are explicit arguments; select them from the frozen comparison protocol, not test performance. Use `--cloud-topic /paper/cloud_local --odom-topic /paper/odom_local --goal-topic /paper_super/goal`. Supply local-world coordinates consistently. The goal publisher and cloud bridge must use this same coordinate system.

The generator disables demo fixed goal altitude (`click_height=-10`) and PCD loading, enables native raycasting with the supplied camera range, and keeps the official backup trajectory enabled. Input remains the existing forward camera's observed point cloud; this adapter must not synthesize full-LiDAR visibility. Demo thrust/body-rate constraints and trajectory optimizer weights remain upstream defaults, requiring execution-side validation against the intended vehicle; no claim of complete dynamic equivalence is made.

## Native Gazebo path

Under SUPER-only ROS workspace:

```
roslaunch /path/to/repo/launches/paper_super_native.launch config:=/absolute/generated_super.yaml
python3 /path/to/repo/scripts/paper_super_command_relay.py
```

Under the existing flight controller workspace (without SUPER overlay):

```
python3 /path/to/repo/scripts/paper_super_legacy_command.py
```

The relay's `/paper_super/reference` wire is `Float64MultiArray` containing timestamp, position xyz, velocity xyz, acceleration xyz, yaw, yaw rate, and native trajectory flag (13 values). It preserves native reference geometry and timing. Native flags 1 and 2 both map to legacy READY=1; native 2 means backup, whereas legacy 2 means emergency. The intermediate topic retains the original flag for logging. No yaw/position smoothing or path replacement is performed.

The official FSM may choose a nearby nonoccupied goal within 3 m (`src/super_core/fsm.cpp`, `setGoalPosiAndYaw`). This upstream behavior is unchanged; paper success must always be adjudicated against the ORIGINAL task goal, not the planner's internal selected goal. Record native goal messages/logs.

## Evidence boundary

Windows syntax and configuration probes passed. A direct field-copy probe verified the 11 reference values and backup-to-READY status mapping. Official coefficients were checked in `src/utils/piece.cpp` to be descending powers in physical seconds.

`scripts/routea_paired_support/super_reference.py` is only a native polynomial sampling helper, not a completed synchronous 186-task adapter. It must not be reported as an available `--policy super` evaluation arm. SUPER motion evaluation still needs a synchronized FSM/map driver and runtime identity integration; ROS compile, message communication, all ROS coordinate checks and flight smoke remain unexecuted.
