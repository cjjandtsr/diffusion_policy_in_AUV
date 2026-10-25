"""Deterministic, auditable export of the frozen v6.5 demonstration state."""

import hashlib
import math

import numpy as np


SCHEMA_VERSION = "reward_v17_frozen_demo_export_v1"
STATE_LAYOUT = "pos_world(3) + goal_body(3) + dist(1) + sin_yaw + cos_yaw"
GOAL_JUMP_M = .01
CONTRACT_TOL = 1e-4
REQUIRED_SOURCE_BASENAMES = {
    "ego_gazebo.sh", "sync_recorder_node.cpp", "random_target_generator.cpp",
    "offboard_node.cpp", "bag2zarr_body_v5.py", "convert_v5_to_v6lite.py",
    "merge_zarr_episodes.py", "generate_worlds_v8_3.py",
}


def array_sha256(values):
    """Hash decoded array bytes with dtype/shape kept separately in provenance."""
    return hashlib.sha256(np.ascontiguousarray(values).tobytes(order="C")).hexdigest()


def _valid_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdefABCDEF" for c in value)


def reconstruct_goals(state):
    state = np.asarray(state, dtype=np.float64)
    if state.ndim != 2 or state.shape[1] != 9 or not np.isfinite(state).all():
        raise ValueError("state must be finite Nx9 v6.5 data")
    position, body = state[:, :3], state[:, 3:6]
    sine, cosine = state[:, 7], state[:, 8]
    if np.max(np.abs(sine * sine + cosine * cosine - 1.)) > CONTRACT_TOL:
        raise ValueError("state violates yaw unit circle contract")
    if np.max(np.abs(np.linalg.norm(body, axis=1) - state[:, 6])) > CONTRACT_TOL:
        raise ValueError("state violates goal-body distance contract")
    world_delta = np.column_stack((cosine * body[:, 0] - sine * body[:, 1],
                                   sine * body[:, 0] + cosine * body[:, 1], body[:, 2]))
    return position + world_delta


def _boundaries(episode_ends, frame_count):
    ends = np.asarray(episode_ends)
    if ends.ndim != 1 or not len(ends) or not np.issubdtype(ends.dtype, np.integer):
        raise ValueError("episode_ends must be a nonempty integer vector")
    ends = ends.astype(np.int64)
    if np.any(np.diff(ends) <= 0) or ends[-1] != frame_count:
        raise ValueError("episode_ends do not partition the state array")
    return np.r_[0, ends[:-1]], ends


def _episode_goal(goals, episode_index):
    jumps = np.flatnonzero(np.linalg.norm(np.diff(goals, axis=0), axis=1) > GOAL_JUMP_M) + 1
    drop = 1 if np.array_equal(jumps, np.array([1])) else 0
    if len(jumps) and not drop:
        raise ValueError("episode {} has unexplained goal change offsets {}".format(
            episode_index, jumps.tolist()))
    kept = goals[drop:]
    if not len(kept):
        raise ValueError("episode {} is empty after stale-goal removal".format(episode_index))
    goal = np.median(kept, axis=0)
    spread = np.max(np.linalg.norm(kept - goal, axis=1))
    if spread > CONTRACT_TOL:
        raise ValueError("episode {} goal is not constant after explained prefix".format(episode_index))
    return drop, goal, float(spread)


def export_frozen_demo(state, episode_ends, *, ground_z, state_array_sha256,
                       episode_ends_array_sha256, source_files):
    """Create the JSON-compatible source consumed by ``prepare_reward_v17 density``."""
    raw_state = np.asarray(state)
    if raw_state.dtype != np.float32:
        raise ValueError("frozen state must retain source float32 dtype")
    if not math.isfinite(ground_z) or float(ground_z) != 0.:
        raise ValueError("this evidence contract binds nominal ground_z=0.0")
    if not _valid_sha256(state_array_sha256) or not _valid_sha256(episode_ends_array_sha256):
        raise ValueError("decoded source arrays require SHA256 identities")
    if array_sha256(raw_state) != state_array_sha256.lower():
        raise ValueError("state array SHA256 does not match decoded source")
    raw_ends = np.asarray(episode_ends)
    if array_sha256(raw_ends.astype(np.int64)) != episode_ends_array_sha256.lower():
        raise ValueError("episode_ends SHA256 does not match decoded source")
    if not source_files or any(not e.get("path") or not _valid_sha256(e.get("sha256")) for e in source_files):
        raise ValueError("complete recording source evidence is required")
    supplied = {str(e["path"]).replace("\\", "/").rsplit("/", 1)[-1] for e in source_files}
    if supplied != REQUIRED_SOURCE_BASENAMES or len(source_files) != len(REQUIRED_SOURCE_BASENAMES):
        raise ValueError("source evidence must contain exactly the eight verified recording-chain files")

    goals = reconstruct_goals(raw_state)
    starts, ends = _boundaries(raw_ends, len(raw_state))
    episodes, dropped, negative = [], 0, 0
    for index, (first, last) in enumerate(zip(starts, ends)):
        drop, goal, spread = _episode_goal(goals[first:last], index)
        kept_first = int(first + drop)
        positions = raw_state[kept_first:last, :3]
        dropped += drop
        negative += int(np.count_nonzero(positions[:, 2] < 0.))
        episodes.append(dict(
            episode_id="merged_body_12:{:06d}".format(index), source="frozen_ego", split="bc_demo",
            start=positions[0].astype(float).tolist(), goal=goal.astype(float).tolist(),
            ground_z=float(ground_z),
            initial_yaw=float(math.atan2(raw_state[kept_first, 7], raw_state[kept_first, 8])),
            positions=positions.astype(float).tolist(), source_episode_index=index,
            source_frame_range=[kept_first, int(last)], dropped_prefix_frames=drop,
            goal_constancy_max_error_m=spread,
        ))
    return dict(
        schema_version=SCHEMA_VERSION,
        source_evidence=dict(
            files=[dict(path=e["path"], sha256=e["sha256"].lower()) for e in source_files],
            coordinate_reference=dict(
                state_layout=STATE_LAYOUT, goal_reconstruction="inverse_Rz_yaw_goal_body",
                start_reference="first retained avoidance sample",
                ground_z=float(ground_z), ground_reference_kind="recording_coordinate_contract",
                ground_precision_claim="nominal coordinate reference, not independent centimetre survey",
                stale_goal_rule="drop frame 0 only when sole >1cm goal jump is at offset 1",
                original_data_mutated=False,
            ),
            decoded_arrays=dict(
                state=dict(dtype=str(raw_state.dtype), shape=list(raw_state.shape), sha256=state_array_sha256.lower()),
                episode_ends=dict(dtype="int64", shape=list(raw_ends.shape), sha256=episode_ends_array_sha256.lower()),
            ),
        ),
        summary=dict(episodes=len(episodes), source_frames=len(raw_state),
                     exported_frames=len(raw_state) - dropped,
                     stale_goal_prefix_episodes=dropped, dropped_stale_goal_frames=dropped,
                     negative_z_frames_retained=negative),
        episodes=episodes,
    )
