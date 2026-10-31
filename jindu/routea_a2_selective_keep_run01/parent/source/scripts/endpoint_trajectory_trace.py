"""Evaluation-only observers. Call the original sampler exactly once per frame."""

from dataclasses import asdict, is_dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys

import numpy as np


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


class PointCloudObserver:
    """Use a temporary return profiler to read true pre-resampling hit count.

    No source patches, extra rays, RNG draws, warning filters or tensor writes.
    The original synthesize code is pinned by the replay provenance checks.
    Single-thread evaluation only; do not combine with an existing profiler.
    """

    def __init__(self, delegate, emit):
        self.delegate, self.emit = delegate, emit

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def synthesize(self, *args, **kwargs):
        if sys.getprofile() is not None:
            raise RuntimeError("trace observer cannot replace an existing profiler")
        method = self.delegate.synthesize
        body_code = inspect.unwrap(method).__code__
        captured = {}

        def capture(frame, event, result):
            if event == "return" and frame.f_code is body_code:
                local = frame.f_locals
                if "hit_points" in local and "N_rays" in local:
                    captured.update(hit_count=int(local["hit_points"].shape[0]),
                                    ray_count=int(local["N_rays"]),
                                    drone_pos_world=np.asarray(local["drone_pos_world"]).copy(),
                                    drone_yaw=float(local["drone_yaw"]),
                                    drone_pitch=float(local["drone_pitch"]),
                                    ground_z=float(local["ground_z"]), rng_seed=local["rng_seed"])
        sys.setprofile(capture)
        try:
            result = method(*args, **kwargs)
        finally:
            sys.setprofile(None)
        if not captured or captured["hit_count"] > captured["ray_count"]:
            raise RuntimeError("true ray-hit locals unavailable; never substitute unknown-time or warning counts")
        captured.update(fallback=captured["hit_count"] == 0, output_sha256=digest_array(result),
                        output_shape=list(result.shape), near=float(self.cam.near), far=float(self.cam.far))
        self.emit(captured)
        return result


class TraceAdapter:
    def __init__(self, delegate, stream, metadata):
        self.delegate, self.stream, self.metadata = delegate, stream, metadata
        self.decision = 0
        self.frames = []
        self.pending = []
        self.history_ids = []
        self.observation = None
        self.original_factory = self.environment.synth_pc_factory
        self.environment.synth_pc_factory = lambda path: PointCloudObserver(self.original_factory(path), self._frame)

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def emit(self, event, **fields):
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
        if len(self.pending) != 1:
            raise RuntimeError("expected one reset frame repeated across observation history")
        self.history_ids = self.pending * len(observation["agent_pos"])
        self.observation = observation
        self.emit("reset", state=self._state(), observation=self._observation(observation),
                  goal_tol_distance=self.environment.goal_tol_xy, goal_tol_z=self.environment.goal_tol_z,
                  traj_dt=.3, n_exec_steps=4)
        return observation, info

    def step(self, action):
        before = self._state()
        prior_observation = self._observation(self.observation)
        full_action = array(action.full).copy()
        self.pending = []
        self.decision += 1
        result = self.delegate.step(action)
        observation, reward, terminated, truncated, info = result
        if observation is not None:
            if len(self.pending) != 4:
                raise RuntimeError("expected four observed frames for valid H4 transition")
            self.history_ids = (self.history_ids + self.pending)[-len(observation["agent_pos"]):]
        history = self.reward_state.history
        self.emit("transition", before=before, after=self._state(),
            time_start_seconds=(self.decision - 1) * 1.2, time_end_seconds=self.decision * 1.2,
            full_action=full_action, executed_path=info.get("executed_path"),
            observation_before=prior_observation, observation_after=self._observation(observation),
            reward=reward, reward_terms=info.get("reward_terms"), reward_metrics=info.get("reward_metrics"),
            transition_facts=info.get("transition_facts"), terminated=terminated, truncated=truncated,
            trap_history={k: history.get(k) for k in ("trap_state", "enter_count", "exit_count", "segment_id", "entry_clearance")},
            eligible_cells=history.get("eligible_cells"), support_paid=history.get("support_paid"))
        self.observation = observation
        return result

    def close(self):
        self.environment.synth_pc_factory = self.original_factory
        self.environment.close()
