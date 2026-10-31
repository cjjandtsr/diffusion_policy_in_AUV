"""R1 kinematic transaction with fractional physical first-event commit."""

import math

import numpy as np

from ..traj_utils import action_to_world, normalize_angle, POS, VEL, YAW
from .first_event import locate_first_event
from .reward import compute_reward_v2, space_running_cost, TERM_NAMES
from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.terminal_contract import resolve_terminal


def reset_reward_v2(env):
    start = env._drone_pos.copy()
    initial = float(np.linalg.norm(start.astype(np.float64) - env.goal_w))
    if not math.isfinite(initial) or initial <= .5:
        raise ValueError("R1 reset must start outside the goal sphere")
    env._episode_start_world = start
    env._reward_history = dict(initial_distance=initial, best_distance=initial)
    env._done_reason = "none"
    env._last_path = None


def _invalid(env, detail):
    env.episode_step += 1
    env._done_reason = "invalid_state"
    facts = resolve_terminal(reached=False, collision=False, ground_crash=False,
                             out_of_bounds=False, altitude_exceed=False,
                             invalid_state=True, time_limit=False,
                             raw_done_reason="invalid_state")
    terms = dict.fromkeys(TERM_NAMES, 0.)
    if getattr(env, "space_cost_budget", None) is not None:
        terms["space"] = 0.
    terms["terminal"] = -50.
    metrics = dict(training_usable=False, invalid_detail=detail)
    env._last_reward_terms, env._last_reward_metrics = terms, metrics
    return None, -50., True, False, dict(
        step=env.episode_step, done_reason=facts.done_reason,
        transition_facts=facts, reward_terms=terms, reward_metrics=metrics,
        reward_version="v2")


def _interpolate_yaw(before, after, fraction):
    delta = math.atan2(math.sin(after - before), math.cos(after - before))
    return normalize_angle(before + fraction * delta)


def step_v2(env, action):
    a = np.asarray(action)
    if a.shape != (8, 11):
        raise ValueError("R1 expects the unchanged 8x11 policy chunk")
    if env._reward_history is None or env._done_reason not in ("none", None):
        raise RuntimeError("reset R1 environment before step")
    pre = env._drone_pos.copy()
    pre_yaw = float(env._drone_yaw)
    pre_vel = env._drone_vel.copy()
    if not all(np.isfinite(x).all() for x in (a, pre, pre_yaw, pre_vel, env.goal_w)):
        return _invalid(env, "nonfinite_action_or_state")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            wp = action_to_world(a, pre, pre_vel, pre_yaw)
            planned = np.vstack((pre, wp[:, POS]))
            # Search H1-H3 before drawing noise used only when H4 is reached.
            path = np.vstack((pre, wp[:3, POS]))
            event = locate_first_event(path, env.goal_w, env.obstacles,
                                       env.collision_thresh, env.z_max,
                                       terminate_at_space_limits=getattr(
                                           env, "terminate_at_space_limits", True))
            if event is None:
                end = wp[3, POS].copy()
                if env.exec_noise_scale > 0:
                    end += env._rng.normal(0, env.exec_noise_scale, 3).astype(np.float32)
                if env.disturb_enable and env._rng.random() < env.disturb_prob:
                    end += env._rng.normal(0, env.disturb_pos_scale, 3).astype(np.float32)
                path = np.vstack((path, end))
                event = locate_first_event(path, env.goal_w, env.obstacles,
                                           env.collision_thresh, env.z_max,
                                           terminate_at_space_limits=getattr(
                                               env, "terminate_at_space_limits", True))
            if event is None:
                actual = path
                pos, yaw, vel = path[-1].copy(), float(wp[3, YAW]), wp[3, VEL].copy()
                frame_count = 4
            else:
                index, fraction = event.segment_index, event.fraction
                actual = np.vstack((path[:index + 1], event.position.astype(np.float32)))
                pos = actual[-1].copy()
                yaw_before = pre_yaw if index == 0 else float(wp[index - 1, YAW])
                vel_before = pre_vel if index == 0 else wp[index - 1, VEL]
                yaw = _interpolate_yaw(yaw_before, float(wp[index, YAW]), fraction)
                vel = (vel_before + fraction * (wp[index, VEL] - vel_before)).astype(np.float32)
                frame_count = index + 1
    except (FloatingPointError, OverflowError):
        return _invalid(env, "nonfinite_action_conversion_or_execution")
    if not all(np.isfinite(x).all() for x in (wp, planned, actual, pos, yaw, vel)):
        return _invalid(env, "nonfinite_executed_path")

    next_step = env.episode_step + 1
    kind = event.kind if event is not None else "none"
    facts = resolve_terminal(
        reached=kind == "reached", collision=kind == "collision",
        ground_crash=kind == "ground_crash", out_of_bounds=kind == "out_of_bounds",
        altitude_exceed=kind == "altitude_exceed", invalid_state=False,
        time_limit=next_step >= env.max_episode_steps,
        raw_done_reason=("collide({})".format(event.obstacle_name)
                         if kind == "collision" else kind))
    reason = facts.done_reason
    total, terms, metrics, best = compute_reward_v2(
        actual, planned, env.goal_w, env.obstacles,
        env._reward_history["best_distance"],
        env._reward_history["initial_distance"], reason,
        env.collision_thresh, 0.)
    if getattr(env, "space_cost_budget", None) is not None:
        durations = np.full(len(actual) - 1, env.traj_dt, dtype=np.float64)
        if event is not None:
            durations[-1] *= event.fraction
        terms["space"] = space_running_cost(
            actual, env._episode_start_world, env.goal_w, durations,
            env.space_cost_budget, env.max_episode_steps * env.n_exec_steps * env.traj_dt)
        total = float(sum(terms.values()))
        metrics.update(space_elapsed_s=float(durations.sum()),
                       space_cost_budget=float(env.space_cost_budget))
    env._drone_pos, env._drone_yaw, env._drone_vel = pos, yaw, vel
    env._last_path = actual
    env._reward_history["best_distance"] = best
    env._done_reason = reason
    env.episode_step = next_step
    env._last_reward_terms, env._last_reward_metrics = terms, metrics

    for index in range(frame_count):
        wp_pos = pos if index == frame_count - 1 else wp[index, POS]
        wp_yaw = yaw if index == frame_count - 1 else float(wp[index, YAW])
        frame_index = env._observation_frame_index + 1
        seed = (env._episode_rng_seed + frame_index * 1000003
                if hasattr(env, "_episode_rng_seed") else None)
        pc, pointcloud_meta = env.synth_pc.generate_with_metadata(
            wp_pos, wp_yaw, num_points=4096, add_noise=True, rng_seed=seed,
            episode_identity_hash=env._episode_identity_hash,
            decision_index=frame_index)
        s9 = env._build_state(env._world_to_odom(wp_pos), wp_yaw,
                              env._world_to_odom(env.goal_w))
        if not np.isfinite(pc).all() or not np.isfinite(s9).all():
            raise FloatingPointError("nonfinite R1 observation")
        env._push_obs(dict(point_cloud=pc.astype(np.float32),
                           agent_pos=s9.astype(np.float32)),
                      pointcloud_meta=pointcloud_meta)
        env._observation_frame_index = frame_index
    obs = env._get_obs_dict()
    info = dict(step=next_step, pos=pos.copy(), z_world=float(pos[2]),
                dist=float(np.linalg.norm(pos - env.goal_w)),
                transition_facts=facts, done_reason=reason,
                reward_version="v2", reward_terms=terms, reward_metrics=metrics,
                executed_path=actual.copy(),
                event_fraction=None if event is None else event.fraction,
                event_segment=None if event is None else event.segment_index,
                executed_time_s=((3. + 1.) if event is None else
                                 (event.segment_index + event.fraction)) * env.traj_dt,
                pointcloud_meta_history=tuple(env._pc_meta_history),
                episode_identity_hash=env._episode_identity_hash,
                decision_index=env._observation_frame_index)
    return obs, total, facts.terminated, facts.truncated, info
