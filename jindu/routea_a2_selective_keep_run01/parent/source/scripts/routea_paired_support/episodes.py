"""Incremental raw episode traces and independently checked commits."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .core import (action_views, canonical_hash, file_hash, observation_hash, parse_json,
                   read_json, trace_metrics, write_json)
from .runtime import metadata


def action_seed(task_id, repeat, decision):
    return int(canonical_hash(["policy-sample-v1", task_id, repeat, decision])[:8], 16)


def validate_rows(header, rows, summary):
    from diffusion_policy_3d.env.collision_geom import ObstacleSpec
    from diffusion_policy_3d.env.reward_v2_support.first_event import locate_first_event
    from diffusion_policy_3d.env.traj_utils import action_to_world, POS
    if not rows or len(rows) > 60 or summary["status"] != "valid":
        raise ValueError("incomplete/invalid episode")
    obstacles = [ObstacleSpec(**o) for o in header["geometry"]]
    goal = np.asarray(header["task"]["goal_pos"], np.float64)
    path = [np.asarray(header["task"]["start_pos"], np.float32)]
    seconds = 0.
    last_reason = None
    for index, row in enumerate(rows):
        if row["decision"] != index:
            raise ValueError("decision order changed")
        full, policy, executed = action_views(row["full_action_16x11"],
                                              row["policy_action_8x11"], row["executed_action_4x11"])
        pre = row["pre_state"]
        before = np.asarray(pre["position"], np.float32)
        if not np.allclose(before, path[-1], atol=1e-6, rtol=0.):
            raise ValueError("executed path discontinuity")
        wp = action_to_world(policy, before, np.asarray(pre["velocity"], np.float32), pre["yaw"])
        candidate = np.vstack((before, wp[:4, POS]))
        if not np.allclose(candidate, row["candidate_h4_path"], atol=1e-6, rtol=0.):
            raise ValueError("saved H4 candidate disagrees with actual action")
        event = locate_first_event(
            candidate, goal, obstacles, .1, header["task"]["z_range"][1],
            terminate_at_space_limits=header.get("terminate_at_space_limits", True))
        if event is None:
            expected = candidate
            expected_time = 1.2
            if row["done_reason"] not in ("none", "time_limit"):
                raise ValueError("terminal label without physical event")
            if row["done_reason"] == "time_limit" and index != 59:
                raise ValueError("early timeout")
            if row["event_segment"] is not None or row["event_fraction"] is not None:
                raise ValueError("event fields on non-event transition")
        else:
            kind = event.kind
            if row["done_reason"] != kind or index != len(rows) - 1:
                raise ValueError("first physical event/terminal order mismatch")
            if (row["event_segment"] != event.segment_index
                    or abs(row["event_fraction"] - event.fraction) > 1e-6):
                raise ValueError("first-event location mismatch")
            expected = np.vstack((candidate[:event.segment_index + 1],
                                  event.position.astype(np.float32)))
            expected_time = (event.segment_index + event.fraction) * .3
        actual = np.asarray(row["executed_path"], np.float32)
        if actual.shape != expected.shape or not np.allclose(actual, expected, atol=1e-5, rtol=0.):
            raise ValueError("actual path disagrees with physical first event")
        if not np.allclose(actual[-1], row["post_state"]["position"], atol=1e-6, rtol=0.):
            raise ValueError("post state not on actual executed path")
        if abs(expected_time - row["executed_time_s"]) > 1e-6:
            raise ValueError("physical time mismatch")
        path.extend(actual[1:])
        seconds += expected_time
        last_reason = row["done_reason"]
    if last_reason == "none" or summary["done_reason"] != last_reason:
        raise ValueError("episode not terminated")
    if bool(summary["safe_arrival"]) != (last_reason == "reached"):
        raise ValueError("safe arrival inconsistent with earliest event")
    expected_stats = trace_metrics(path, goal)
    for name, value in expected_stats.items():
        if abs(summary["trajectory_metrics"][name] - value) > 1e-6:
            raise ValueError("trajectory statistic mismatch: " + name)
    if summary["decisions"] != len(rows) or abs(summary["physical_time_s"] - seconds) > 1e-6:
        raise ValueError("episode time/count mismatch")
    return np.asarray(path)


def load_commit(job, commit_path):
    job = Path(job).resolve()
    commit = read_json(commit_path)
    path = (job / commit["trace"]).resolve()
    path.relative_to(job)
    if file_hash(path) != commit["sha256"]:
        raise ValueError("committed raw trace bytes changed")
    rows = [parse_json(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows or rows[0].get("type") != "episode_start" or rows[-1].get("type") != "episode_end":
        raise ValueError("partial trace cannot be a completed episode")
    header, summary = rows[0], rows[-1]
    if commit["summary"] != summary or any(r.get("type") != "transition" for r in rows[1:-1]):
        raise ValueError("commit/trace summary mismatch")
    validate_rows(header, rows[1:-1], summary)
    return header, rows[1:-1], summary


def ambiguous_events(header, rows):
    from diffusion_policy_3d.env.reward_v2_support.first_event import _ball_entry
    goal = np.asarray(header["task"]["goal_pos"], np.float64)
    result = []
    for row in rows:
        index = row["event_segment"]
        if index is None or row["done_reason"] == "reached":
            continue
        path = np.asarray(row["candidate_h4_path"], np.float64)
        goal_fraction = _ball_entry(path[index],path[index+1],goal,.5)
        if goal_fraction is not None and abs(goal_fraction-row["event_fraction"]) <= 1e-6:
            result.append({"decision":row["decision"],"failure":row["done_reason"],
                           "failure_fraction":row["event_fraction"],"goal_fraction":goal_fraction,
                           "resolution":"failure-first within approved 1e-6 segment tolerance"})
    return result


def record_episode(job, policy_name, policy, adapter, task, entry, world, bank_hash):
    from scripts.evaluate_dppo_uav_formal_endpoint import reset_fixed_episode
    from diffusion_policy_3d.env.traj_utils import action_to_world, POS
    env = adapter.environment
    observation, context = reset_fixed_episode(env, task["payload"], seed=entry["reset_seed"],
                                              reset_callable=adapter.reset)
    if observation_hash(observation) != entry["observation_hash"]:
        raise ValueError("initial observation pairing failed: " + entry["key"])
    job = Path(job)
    trace_dir = job / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    commit_path = job / "commits" / (entry["key"] + ".json")
    if commit_path.exists():
        raise FileExistsError("episode already committed")
    attempt = 1
    while (trace_dir / (entry["key"] + ".attempt%03d.jsonl" % attempt)).exists():
        attempt += 1
    trace = trace_dir / (entry["key"] + ".attempt%03d.jsonl" % attempt)
    header = {"type": "episode_start", "task_id": entry["task_id"], "repeat": entry["repeat"],
              "policy": policy_name, "use_base_policy": policy.use_base_policy,
              "world_id": world["world_id"], "role": task["role"], "task": task["payload"],
              "bank_hash": bank_hash, "initial_hash": entry["initial_hash"],
              "geometry": world["geometry"], "obstacle_sha256": world["obstacle_sha256"],
              "spawn_offset": world["spawn_offset"], "position_frame": "world",
              "policy_position_frame": "odom = world - spawn_offset",
              "action_dtype": "float32", "reset_seed": entry["reset_seed"],
              "initial_pointcloud_metadata": metadata(context),
              "policy_identity_hash": policy.identity_hash, "reward_version": "v2"}
    header["terminate_at_space_limits"] = bool(getattr(env, "terminate_at_space_limits", True))
    rows, path = [], [env._drone_pos.copy()]
    seconds = 0.
    with trace.open("x", encoding="utf-8", newline="\n") as stream:
        def emit(value):
            stream.write(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
        emit(header)
        try:
            for decision in range(60):
                pre = {"position": env._drone_pos.tolist(), "velocity": env._drone_vel.tolist(),
                       "yaw": float(env._drone_yaw)}
                seed = action_seed(entry["task_id"], entry["repeat"], decision)
                meta_before = metadata(context)
                bundle = policy.predict(observation, context, action_seed=seed)
                arrays = action_views(bundle.full[0].detach().cpu().numpy(),
                                      bundle.policy[0].detach().cpu().numpy(),
                                      bundle.rollout_executed[0].detach().cpu().numpy())
                wp = action_to_world(arrays[1], np.asarray(pre["position"], np.float32),
                                     np.asarray(pre["velocity"], np.float32), pre["yaw"])
                candidate = np.vstack((np.asarray(pre["position"], np.float32), wp[:4, POS]))
                provenance = asdict(policy.last_provenance)
                observation, reward, terminated, truncated, context = adapter.step(bundle)
                actual = context["executed_path"]
                row = {"type": "transition", "decision": decision, "action_seed": seed,
                       "pre_state": pre, "post_state": {"position": env._drone_pos.tolist(),
                           "velocity": env._drone_vel.tolist(), "yaw": float(env._drone_yaw)},
                       "full_action_16x11": arrays[0].tolist(), "policy_action_8x11": arrays[1].tolist(),
                       "executed_action_4x11": arrays[2].tolist(), "candidate_h4_path": candidate.tolist(),
                       "planned_policy_path": np.vstack((np.asarray(pre["position"], np.float32), wp[:, POS])).tolist(),
                       "executed_path": actual.tolist(), "done_reason": context["done_reason"],
                       "event_segment": context["event_segment"], "event_fraction": context["event_fraction"],
                       "executed_time_s": context["executed_time_s"],
                       "observation_provenance": provenance, "pointcloud_metadata_before": meta_before,
                       "pointcloud_metadata_after": metadata(context),
                       "reward": float(reward), "reward_terms": context["reward_terms"]}
                emit(row)
                rows.append(row)
                path.extend(np.asarray(actual)[1:])
                seconds += row["executed_time_s"]
                if terminated or truncated:
                    break
            summary = {"type": "episode_end", "status": "valid", "task_id": entry["task_id"],
                       "repeat": entry["repeat"], "world_id": world["world_id"], "role": task["role"],
                       "policy": policy_name, "initial_hash": entry["initial_hash"], "bank_hash": bank_hash,
                       "safe_arrival": context["done_reason"] == "reached", "done_reason": context["done_reason"],
                       "terminate_at_space_limits": header["terminate_at_space_limits"],
                       "decisions": len(rows), "physical_time_s": seconds,
                       "trajectory_metrics": trace_metrics(path, env.goal_w)}
            validate_rows(header, rows, summary)
            emit(summary)
        except Exception as exc:
            emit({"type": "episode_error", "status": "invalid", "error_type": type(exc).__name__,
                  "detail": str(exc), "completed_decisions": len(rows)})
            raise
    write_json(commit_path, {"trace": str(trace.relative_to(job)).replace("\\", "/"),
                             "sha256": file_hash(trace), "summary": summary})
    return summary
