"""v17 environment transaction; legacy step and collision code stay untouched."""

import numpy as np

from ..traj_utils import action_to_world, normalize_angle, POS, VEL, YAW
from ..reward_v17 import compute_reward_v17, create_reward_history, invalid_result
from .geometry import GeometryContext
from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.terminal_contract import (
    reached_goal,
    resolve_terminal,
)


def reset_reward(env):
    env._episode_start_world = env._drone_pos.copy()
    env._reward_history = create_reward_history(initial_yaw=env._drone_yaw,
                                                episode_id=getattr(env, "_episode_rng_seed", 0))
    env._done_reason = "none"
    env._last_path = None


def _invalid(env, detail):
    env._done_reason = "invalid_state"
    env.episode_step += 1
    total, terms, metrics, history = invalid_result(env._reward_history, env.reward_v17_cfg)
    env._reward_history = history
    metrics["invalid_detail"] = detail
    facts = resolve_terminal(
        reached=False,
        collision=False,
        ground_crash=False,
        out_of_bounds=False,
        altitude_exceed=False,
        invalid_state=True,
        time_limit=False,
        raw_done_reason="invalid_state",
    )
    env._last_reward_terms, env._last_reward_metrics = terms, metrics
    # No fabricated observation/next state. Training must stop before encoding.
    return None, total, True, False, dict(step=env.episode_step, done_reason=facts.done_reason,
                                        transition_facts=facts, reward_terms=terms,
                                        reward_metrics=metrics, reward_version="v1.7")


def step_v17(env, action):
    a = np.asarray(action)
    if a.shape != (env.rl_action_steps, 11):
        raise ValueError("v17 expects the unchanged 8x11 policy chunk")
    if env._reward_history is None:
        raise RuntimeError("reset v17 environment before step")
    if env._done_reason not in ("none", None):
        raise RuntimeError("episode already ended; reset before step")
    pre = env._drone_pos.copy()
    pre_yaw = env._drone_yaw
    if not all(np.isfinite(x).all() for x in (a, pre, pre_yaw, env._drone_vel, env.goal_w)):
        return _invalid(env, "nonfinite_action_or_state")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            wp = action_to_world(a, pre, env._drone_vel, pre_yaw)
            last = env.n_exec_steps - 1
            pos, yaw, vel = wp[last, POS].copy(), float(wp[last, YAW]), wp[last, VEL].copy()
            if env.exec_noise_scale > 0:
                pos += env._rng.normal(0, env.exec_noise_scale, 3).astype(np.float32)
            if env.disturb_enable and env._rng.random() < env.disturb_prob:
                pos += env._rng.normal(0, env.disturb_pos_scale, 3).astype(np.float32)
            pos, vel = pos.astype(np.float32), vel.astype(np.float32)
            yaw = normalize_angle(yaw)
            path = np.vstack((pre, wp[:env.n_exec_steps, POS]))
            path[-1] = pos
    except (FloatingPointError, OverflowError):
        return _invalid(env, "nonfinite_action_conversion_or_execution")
    if not all(np.isfinite(x).all() for x in (wp, path, pos, vel, yaw)):
        return _invalid(env, "nonfinite_executed_path")
    env._drone_pos, env._drone_yaw, env._drone_vel = pos, yaw, vel
    env._last_path = path
    env._done_reason = "none"
    contact = bool(env._check_collision(path_wp=path))
    contact_reason = str(getattr(env, "_done_reason", "none"))
    ground_crash = bool(np.any(path[:, 2] < 0.2))
    collision = contact and not ground_crash
    out = bool(np.any(np.abs(path[:, :2]) > 30.))
    distance = float(np.linalg.norm(pos - env.goal_w))
    altitude = bool(np.any(path[:, 2] > env.z_max + 1.0))
    reached = reached_goal(pos, env.goal_w, 0.5)
    raw = (
        contact_reason
        if contact
        else (
            "out_of_bounds"
            if out
            else ("altitude_exceed" if altitude else ("reached" if reached else "none"))
        )
    )
    next_step = env.episode_step + 1
    facts = resolve_terminal(
        reached=reached,
        collision=collision,
        ground_crash=ground_crash,
        out_of_bounds=out,
        altitude_exceed=altitude,
        invalid_state=False,
        time_limit=next_step >= env.max_episode_steps,
        raw_done_reason=raw,
    )
    result = compute_reward_v17(
        dict(pos=pre, yaw=pre_yaw), path, dict(pos=pos, yaw=yaw), facts, env.obstacles,
        env.goal_w, env._episode_start_world, GeometryContext(), env._reward_history,
        env.offline_density, env.reward_v17_cfg)
    total, terms, metrics, history = result
    if not metrics["training_usable"]:
        # Derived overflow is a hard implementation/numerical failure; retained
        # invalid record does not assert a usable next observation exists.
        env._done_reason = "invalid_state"
        facts = resolve_terminal(
            reached=False,
            collision=False,
            ground_crash=False,
            out_of_bounds=False,
            altitude_exceed=False,
            invalid_state=True,
            time_limit=False,
            raw_done_reason="invalid_state",
        )
        obs = None
    else:
        # Exactly the existing per-waypoint observation path and RNG seeds.
        for k in range(env.n_exec_steps):
            wp_pos = pos if k == last else wp[k, POS]
            wp_yaw = yaw if k == last else float(wp[k, YAW])
            frame_index = env._observation_frame_index + 1
            seed = (
                env._episode_rng_seed + frame_index * 1000003
                if hasattr(env, "_episode_rng_seed") else None
            )
            pc, pointcloud_meta = env.synth_pc.generate_with_metadata(
                wp_pos,
                wp_yaw,
                num_points=4096,
                add_noise=True,
                rng_seed=seed,
                episode_identity_hash=env._episode_identity_hash,
                decision_index=frame_index,
            )
            s9 = env._build_state(env._world_to_odom(wp_pos), wp_yaw, env._world_to_odom(env.goal_w))
            if not np.isfinite(pc).all() or not np.isfinite(s9).all():
                raise FloatingPointError("nonfinite observation from finite v17 state")
            env._push_obs(
                dict(point_cloud=pc.astype(np.float32), agent_pos=s9.astype(np.float32)),
                pointcloud_meta=pointcloud_meta,
            )
            env._observation_frame_index = frame_index
        obs = env._get_obs_dict()
        env._done_reason = raw if facts.terminated else ("time_limit" if facts.truncated else "none")
    env.episode_step = next_step
    env._reward_history = history
    env._last_reward_terms, env._last_reward_metrics = terms, metrics
    info = dict(step=next_step, pos=pos.copy(), z_world=float(pos[2]), dist=distance,
                clr=metrics.get("d_min"), transition_facts=facts, done_reason=facts.done_reason,
                reward_version="v1.7", reward_terms=terms, reward_metrics=metrics,
                executed_path=path.copy(),
                pointcloud_meta_history=tuple(env._pc_meta_history),
                episode_identity_hash=env._episode_identity_hash,
                decision_index=env._observation_frame_index)
    return obs, total, facts.terminated, facts.truncated, info
