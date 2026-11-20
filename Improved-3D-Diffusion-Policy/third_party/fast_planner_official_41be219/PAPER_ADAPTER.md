# Fast-Planner paper adapter

Date: 2026-11-10. Upstream: HKUST-Aerial-Robotics/Fast-Planner,
commit `41be219fe4ecc43bf0e0c2b42a523f8755ccc0bd`.
Source was downloaded as the official fixed-commit archive. Upstream algorithms
and configuration remain in place; additions expose a synchronous observation
and clock barrier to the existing common environment.

The service invokes native SDFMap odometry/cloud/ESDF callbacks and native kino
FSM planning. Only decision boundaries run the FSM; intermediate observations
update the map. Each episode starts a new process, resetting map and history.
The 3D goal is assigned directly: upstream's interactive waypoint callback
forces z=1, which cannot represent the fixed 3D evaluation goals. This bypass
changes the goal input adapter, not kinodynamic search or spline optimization.
Position and yaw are sampled from the upstream position/yaw splines, including
upstream terminal hover behavior. No EGO yaw rule is used.

Build this source in a separate catkin workspace from EGO: package names overlap.
Run from a fresh ROS Noetic shell, after installing the upstream README's
dependencies (including NLopt and Armadillo). Example, with `REPO` set to the
existing project root and `FAST_WS` set to a new workspace:

```bash
source /opt/ros/noetic/setup.bash
mkdir -p "$FAST_WS/src"
ln -s "$REPO/third_party/fast_planner_official_41be219/fast_planner" "$FAST_WS/src/fast_planner"
ln -s "$REPO/third_party/fast_planner_official_41be219/uav_simulator" "$FAST_WS/src/uav_simulator"
cd "$FAST_WS"
catkin_make -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash
```

The Python factory is `FastPolicy(launch, map_size, log_dir)`. Launch file:
`fast_planner/plan_manage/launch/paper_fast_eval.launch`. It includes the official
kino configuration without a simulator, global obstacle map or waypoint node.
Revision v1_1 (2026-11-10): its paper launch uses 0.5 m/s and 0.25 m/s²,
matching the saved EGO collection/evaluation launch parameters. This identifies
the current saved configuration, not every historical demonstration run.

Validation completed: Python syntax, launch XML, native spline numerical probe.
Not validated here: ROS compilation, transport, ESDF update, complete episode.
These implementation additions must not be reported as successful planner runs.
Modified upstream originals are retained with SHA256 in
`program/code_backups/20261110_fast_adapter_before/` at the project root.
