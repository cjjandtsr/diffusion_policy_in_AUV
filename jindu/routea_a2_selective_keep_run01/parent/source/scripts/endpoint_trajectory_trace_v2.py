"""Evaluation-only observers. Call the original sampler exactly once per frame.

v2 of `endpoint_trajectory_trace`. The only change is the frame hook:

  v1 profiled `SynthPointCloud.synthesize` via `synth_pc_factory` and read
  `hit_points` / `N_rays` out of the traced frame's locals. On the current
  `synth_pointcloud.py` neither `synthesize` nor `generate_with_metadata`
  binds those names — v17 observations go through `generate_with_metadata`
  (kinematic_uav_env.py:852) — so the capture never fires and reset raises
  "expected one reset frame repeated across observation history".

  v2 hooks `KinematicUAVEnv._push_obs` instead. That is the single point at
  which a frame enters the observation history, it is called for both the
  v13 and the v17 synthesis branch, and it receives the real point cloud.
  Frame counts follow the env: 8 on reset (one frame replicated across
  `n_obs_steps`) and `n_exec_steps` (4) per decision step.

Per-decision identity. Each transition also carries the identity of the input
the policy actually consumed, taken from `ObservationProvenance`
(`observation_provenance.py:109`): raw frame/history hashes, the normalizer
identity, the normalized-history hash, the condition hash the policy was
called with, the episode identity, the decision index, and the action seed.

This is what separates "the pipeline fed BC and RL different inputs" from "the
policy produced different actions". Compare the two policies' transition N in
the same task: while their physics has not yet diverged the observation and
`policy_input_hash` must be identical. A difference there indicts the
evaluation pipeline; identical inputs with a differing `full_action` indicts
the policy. `PolicyRecorder` is the tap for that; it is a transparent proxy, so
`_evaluate_one` still sees an object with the same `predict` / `last_provenance`
contract.

Coordinate frames. Every emitted line carries a `frames` block declaring which
frame each position-bearing field is in, plus the world's `spawn_offset`.
`agent_pos` is odom; `position` / `executed_path` / `goal` are world, so the two
legitimately differ by that offset (e.g. +8.0 m on x for the layerA worlds).
Recorded per line so a sliced transition stays self-locating; see
`frame_contract` for the citations.

Trajectory tables. `write_trajectory_csv` turns one episode into a single CSV
holding two tables: the policy's planned trajectory (8 world-frame waypoints per
decision step) and the executed trajectory (positions only, one row per time
step). The env keeps only the executed path -- `wp` is built at
`reward_v17_support/environment.py:59` and dropped -- so the plan is recomputed
here from the same three inputs the env used, and the executed prefix is checked
bit for bit against it before anything is written. A drift raises instead of
being recorded.

No production file is modified; the hook is installed on the adapter's own
environment instance and removed by `close()`.
"""

from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from diffusion_policy_3d.env.traj_utils import ACC, POS, VEL, YAW, YAW_RT, action_to_world


def array(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def digest_array(value):
    a = np.ascontiguousarray(array(value))
    h = hashlib.sha256((str(a.dtype) + repr(a.shape)).encode("ascii"))
    h.update(a.tobytes())
    return h.hexdigest()


def json_safe(value, *, missing, path="$"):
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, np.ndarray) or hasattr(value, "detach"):
        value = array(value).tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        return {str(k): json_safe(v, missing=missing, path=path + "." + str(k)) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v, missing=missing, path="{}[{}]".format(path, i)) for i, v in enumerate(value)]
    if isinstance(value, float) and not math.isfinite(value):
        missing.append(path)
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError("unsupported trace value at {}: {}".format(path, type(value).__name__))


def rng_state_hashes():
    """Hash the process-global RNG states the policy is about to inherit.

    `_E5Policy.predict:376` calls `torch.manual_seed(action_seed)` on entry, so
    the state a policy installs is deterministic and uninteresting. The state it
    *inherits* is not: it carries whatever the previous episode consumed. Both
    policies seed on entry, so this should not reach the output -- but replaying
    BC and RL out of order changed the numbers, so something consumes global RNG
    and this records it. Reading state does not consume it.
    """
    import torch

    digest = hashlib.sha256()
    digest.update(torch.get_rng_state().numpy().tobytes())
    digest.update(np.random.get_state()[1].tobytes())
    if torch.cuda.is_available():
        try:
            digest.update(torch.cuda.get_rng_state_all()[0].numpy().tobytes())
        except (RuntimeError, IndexError):
            pass
    return digest.hexdigest()


def compare_record(expected, actual):
    """Compare every original summary field; do not just match success labels."""
    problems = []

    def compare(left, right, path):
        if isinstance(left, dict):
            if not isinstance(right, dict) or set(left) != set(right):
                problems.append(path + ": keys differ")
                return
            for key in left:
                compare(left[key], right[key], path + "." + key)
        elif isinstance(left, (tuple, list)):
            if not isinstance(right, (tuple, list)) or len(left) != len(right):
                problems.append(path + ": length differs")
                return
            for i, (x, y) in enumerate(zip(left, right)):
                compare(x, y, "{}[{}]".format(path, i))
        elif isinstance(left, float):
            if not isinstance(right, (int, float)) or not math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-6):
                problems.append(path + ": numeric drift")
        elif left != right:
            problems.append(path + ": differs")
    compare(asdict(expected), asdict(actual), "record")
    return problems


def frame_contract(spawn_offset):
    """Declare the coordinate frame of every position-bearing trace field.

    Two frames are in play. Subtracting across them shifts x by the world's spawn
    offset, which reads as "the pipeline fed BC and RL different inputs" when it
    is only a frame difference:

      world -- obstacle-geometry frame (obstacles.json, synthesized point cloud)
      odom  -- world - spawn_offset; the frame the 9D state is built in

    `KinematicUAVEnv` synthesizes the point cloud in world and builds the state
    from odom (`kinematic_uav_env.py:773-776`, `:869-871`); the v17 endpoint path
    builds observations the same way (`reward_v17_support/environment.py:144`).
    The odom conversion is the 2026-06-27 frame fix, recorded in
    `program/backup/handoff_routeB_20260627.md:17` as "state 计算 frame 混用
    (goal=world, pos=odom) -- 确认并修复".

    Recorded on every line so a single sliced transition stays self-locating.
    `spawn_offset` is read from the live env, so it follows the world loaded.
    """
    offset = array(spawn_offset).tolist()
    return dict(
        spawn_offset=offset,
        pointcloud="world",
        agent_pos="odom",
        position="world",
        velocity_reference="world",
        yaw="world",
        goal="world",
        executed_path="world",
        rule="agent_pos = world - spawn_offset",
    )


PLANNED_COLUMNS = ("decision_step", "k", "time_s", "x", "y", "z",
                   "vx", "vy", "vz", "ax", "ay", "az", "yaw", "yaw_rate")
EXECUTED_COLUMNS = ("decision_step", "k", "time_s", "x", "y", "z")


def plan_waypoints(policy_action, before):
    """Recompute the env's world-frame plan for one decision step.

    `reward_v17_support/environment.py:59` builds
    `wp = action_to_world(policy_slice, pre, env._drone_vel, pre_yaw)` and keeps
    only `path = vstack((pre, wp[:n_exec_steps, POS]))` (`:68`). `wp` is published
    nowhere, so it is recomputed here from the same three inputs. `before` is
    `TraceAdapter._state()` captured before the env stepped, so
    `before["position"] / ["velocity_reference"] / ["yaw"]` are exactly the
    `pre / _drone_vel / _drone_yaw` the env used. `action_to_world` is pure, so
    the result is bit-identical -- `_record_trajectory` asserts that.
    """
    raw = array(policy_action)
    if raw.ndim == 3:
        if raw.shape[0] != 1:
            raise RuntimeError("expected one policy chunk, got shape {}".format(raw.shape))
        raw = raw[0]
    return action_to_world(
        np.ascontiguousarray(raw, dtype=np.float32),
        np.asarray(before["position"], dtype=np.float32),
        np.asarray(before["velocity_reference"], dtype=np.float32),
        float(before["yaw"]),
    )


def _cell(value):
    """`repr` round-trips a float exactly; `%.6f` would silently lose digits."""
    return repr(value) if isinstance(value, float) else str(value)


def write_trajectory_csv(path, header_lines, planned_rows, executed_rows):
    """Write both trajectory tables into one CSV.

    CSV has no multi-table syntax, so the layout is: `#`-prefixed header lines,
    then a `# table: planned` marker with its column row and data, then a
    `# table: executed` marker with its column row and data.

    The markers are themselves comment lines, so a reader cannot simply drop
    every `#` line -- that would discard the markers too and leave the 14-column
    and 6-column rows interleaved. Read in two passes: locate the two literal
    marker lines first, split there, and only then drop the remaining `#` lines
    (and the column row) within each table.
    """
    path = Path(path)
    tables = (("# table: planned", PLANNED_COLUMNS, planned_rows),
              ("# table: executed", EXECUTED_COLUMNS, executed_rows))
    with path.open("x", encoding="utf-8", newline="") as stream:
        for line in header_lines:
            stream.write("# " + line + "\n")
        for marker, columns, rows in tables:
            stream.write(marker + "\n")
            stream.write(",".join(columns) + "\n")
            for row in rows:
                if len(row) != len(columns):
                    raise RuntimeError("row width {} does not match {}".format(len(row), columns))
                stream.write(",".join(_cell(value) for value in row) + "\n")


class FrameObserver:
    """Emit one record per observation frame as it enters the env history."""

    def __init__(self, environment, emit):
        self.environment = environment
        self.emit = emit
        self.original = environment._push_obs
        environment._push_obs = self._push_obs

    def _push_obs(self, obs, pointcloud_meta=None):
        self.original(obs, pointcloud_meta=pointcloud_meta)
        pc = array(obs["point_cloud"])
        self.emit(dict(
            pointcloud_sha256=digest_array(pc),
            pointcloud_shape=list(pc.shape),
            pointcloud_finite=bool(np.isfinite(pc).all()),
            agent_pos=array(obs["agent_pos"]).tolist(),
        ))

    def close(self):
        self.environment._push_obs = self.original


class PolicyRecorder:
    """Transparent proxy around the endpoint policy that records each input.

    `_evaluate_one` calls `policy.predict(observation, context, ...)` and then
    `adapter.step(action)`. The proxy sits on the first call and the
    `TraceAdapter` on the second, so pairing them by call order binds every
    transition to the exact input the policy consumed.

    Every attribute other than `predict` is forwarded, including writes:
    `_evaluate_one` sets `policy.last_provenance = None` before the loop and
    reads it after, and `compare_record` checks the value it ends up with.
    """

    def __init__(self, policy, sink, action_seed_offset=0):
        object.__setattr__(self, "_policy", policy)
        object.__setattr__(self, "_sink", sink)
        object.__setattr__(self, "_offset", int(action_seed_offset))

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_policy"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_policy"), name, value)

    def predict(self, observation, observation_context, *, action_seed):
        policy = object.__getattribute__(self, "_policy")
        # A repeat shifts the seed `_evaluate_one` derived from the config hash,
        # so the policy's own sampling draw is fresh rather than a replay of
        # the first run. The effective seed is what gets recorded, so a trace
        # never claims a seed the policy did not actually use.
        offset = object.__getattribute__(self, "_offset")
        effective = (int(action_seed) + offset) & 0xFFFFFFFF
        inherited_rng = rng_state_hashes()
        bundle = policy.predict(observation, observation_context, action_seed=effective)
        provenance = policy.last_provenance
        object.__getattribute__(self, "_sink").append(dict(
            action_seed=int(effective),
            inherited_rng_state_hash=inherited_rng,
            provenance=None if provenance is None else asdict(provenance),
        ))
        return bundle


class TraceAdapter:
    def __init__(self, delegate, stream, metadata):
        """`stream=None` records trajectory rows without writing event lines.

        The trajectory tables are the only output the collector always needs;
        the per-decision event stream is optional. Both are produced here, so
        suppressing the stream must not suppress `planned_rows`/`executed_rows`
        -- `_record_trajectory` runs either way.
        """
        self.delegate, self.stream, self.metadata = delegate, stream, metadata
        self.decision = 0
        self.frames = []
        self.pending = []
        self.history_ids = []
        self.observation = None
        self.policy_inputs = []
        # Trajectory tables. Times follow the env: one decision step is
        # `n_exec_steps * traj_dt` seconds, one waypoint is `traj_dt` seconds.
        self.planned_rows = []
        self.executed_rows = []
        self.exec_steps = int(self.environment.n_exec_steps)
        self.policy_steps = int(self.environment.rl_action_steps)
        self.waypoint_seconds = float(self.environment.traj_dt)
        self.decision_seconds = self.exec_steps * self.waypoint_seconds
        # Read the offset from the live env, after _set_world has loaded the world.
        self.metadata = dict(metadata, frames=frame_contract(self.environment._spawn_offset))
        self.observer = FrameObserver(self.environment, self._frame)

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def emit(self, event, **fields):
        if self.stream is None:
            return
        missing = []
        payload = json_safe(dict(self.metadata, event=event, decision_step=self.decision, **fields), missing=missing)
        payload["nonfinite_paths"] = missing
        self.stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        self.stream.flush()

    def _frame(self, data):
        frame_id = len(self.frames)
        data = dict(data, frame_id=frame_id, frame_in_step=len(self.pending), phase="reset" if self.decision == 0 else "transition")
        self.frames.append(data)
        self.pending.append(frame_id)
        self.emit("pointcloud", **data)

    def _state(self):
        env = self.environment
        return dict(position=env._drone_pos.copy(), velocity_reference=env._drone_vel.copy(),
                    yaw=float(env._drone_yaw), goal=env.goal_w.copy(),
                    distance_to_goal=float(np.linalg.norm(env.goal_w - env._drone_pos)))

    def _record_trajectory(self, before, action, info):
        """Append this decision step's planned and executed rows.

        The plan is recomputed, never trusted blind: `wp[:exec_steps, POS]` must
        equal the env's own `executed_path[1:]` bit for bit, otherwise this
        raises. The first four waypoints are the only ones the env executes, so
        matching them proves the whole array came from the same call, and a
        future change to the env's action slice fails loudly here rather than
        writing a plan the policy never produced.
        """
        executed = info.get("executed_path")
        if executed is None:
            # `_invalid` returns no executed path; there is no trajectory to record.
            return
        executed = np.asarray(executed, dtype=np.float32)
        planned = plan_waypoints(action.policy, before)
        if len(executed) != self.exec_steps + 1:
            raise RuntimeError("executed path has {} points, expected {}".format(
                len(executed), self.exec_steps + 1))
        if not np.array_equal(planned[: self.exec_steps, POS], executed[1:]):
            raise RuntimeError(
                "recomputed plan disagrees with the env's executed path at decision {}; "
                "refusing to write a trajectory the policy did not produce".format(self.decision))
        base = (self.decision - 1) * self.decision_seconds
        for k in range(len(planned)):
            row = [self.decision, k, base + (k + 1) * self.waypoint_seconds]
            row += [float(v) for v in planned[k, POS]]
            row += [float(v) for v in planned[k, VEL]]
            row += [float(v) for v in planned[k, ACC]]
            row += [float(planned[k, YAW]), float(planned[k, YAW_RT])]
            self.planned_rows.append(row)
        # `executed[0]` is the previous decision's last waypoint, already written.
        first = 0 if self.decision == 1 else 1
        for k in range(first, len(executed)):
            row = [self.decision, k, base + k * self.waypoint_seconds]
            row += [float(v) for v in executed[k]]
            self.executed_rows.append(row)

    def _observation(self, observation):
        if observation is None:
            return None
        return dict(agent_pos=observation["agent_pos"].copy(),
                    pointcloud_sha256=digest_array(observation["point_cloud"]),
                    actor_latest3_frame_ids=self.history_ids[-3:],
                    history_frame_ids=list(self.history_ids))

    def reset(self, *args, **kwargs):
        self.pending = []
        observation, info = self.delegate.reset(*args, **kwargs)
        if len(self.pending) != self.environment.n_obs_steps:
            raise RuntimeError("expected one reset frame replicated across observation history")
        self.history_ids = list(self.pending[-len(observation["agent_pos"]):])
        self.observation = observation
        self.emit("reset", state=self._state(), observation=self._observation(observation),
                  goal_tol_distance=self.environment.goal_tol_xy, goal_tol_z=self.environment.goal_tol_z,
                  traj_dt=self.waypoint_seconds, n_exec_steps=self.exec_steps)
        return observation, info

    def step(self, action):
        # `predict` ran immediately before this call, so exactly one record must
        # be pending. Anything else means the pairing is wrong and every
        # identity field below would be attributed to the wrong decision.
        if len(self.policy_inputs) != 1:
            raise RuntimeError(
                "expected exactly one recorded policy input per decision, got {}; "
                "wrap the policy in PolicyRecorder(..., trace.policy_inputs)".format(
                    len(self.policy_inputs)))
        policy_input = self.policy_inputs.pop()
        before = self._state()
        prior_observation = self._observation(self.observation)
        full_action = array(action.full).copy()
        self.pending = []
        self.decision += 1
        result = self.delegate.step(action)
        observation, reward, terminated, truncated, info = result
        if observation is not None:
            if len(self.pending) != self.environment.n_exec_steps:
                raise RuntimeError("expected four observed frames for valid H4 transition")
            self.history_ids = (self.history_ids + self.pending)[-len(observation["agent_pos"]):]
        self._record_trajectory(before, action, info)
        history = self.reward_state.history
        self.emit("transition", before=before, after=self._state(),
            time_start_seconds=(self.decision - 1) * 1.2, time_end_seconds=self.decision * 1.2,
            full_action=full_action, executed_path=info.get("executed_path"),
            policy_input=policy_input,
            observation_before=prior_observation, observation_after=self._observation(observation),
            reward=reward, reward_terms=info.get("reward_terms"), reward_metrics=info.get("reward_metrics"),
            transition_facts=info.get("transition_facts"), terminated=terminated, truncated=truncated,
            trap_history={k: history.get(k) for k in ("trap_state", "enter_count", "exit_count", "segment_id", "entry_clearance")},
            eligible_cells=history.get("eligible_cells"), support_paid=history.get("support_paid"))
        self.observation = observation
        return result

    def close(self):
        self.observer.close()
        self.environment.close()
