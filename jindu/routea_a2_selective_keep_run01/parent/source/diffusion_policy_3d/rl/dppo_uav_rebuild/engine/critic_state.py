"""Add episode history to critic only; preserve the actor condition dictionary."""

import math

import numpy as np
import torch


def with_history_scalar(actor_condition, history):
    initial = float(history["initial_distance"])
    best = float(history["best_distance"])
    if not (math.isfinite(initial) and math.isfinite(best) and initial > .5 and
            0. <= best <= initial):
        raise ValueError("invalid R1 critic history")
    reference = actor_condition["global_cond"]
    if reference.ndim != 2 or reference.shape[1] != 576:
        raise ValueError("actor global_cond must be (B,576)")
    beta = torch.full((reference.shape[0], 1), best / initial,
                      device=reference.device, dtype=reference.dtype)
    return dict(actor_condition, beta=beta)


def with_episode_context(actor_condition, environment):
    """Finite-task critic context at the current state, before any reset.

    Ten metres is a scale, not a flight limit. Actor inputs are untouched.
    """
    env = environment
    beta = with_history_scalar(actor_condition, env._reward_history)["beta"]
    offset = (np.asarray(env._episode_start_world, dtype=np.float64) -
              np.asarray(env._drone_pos, dtype=np.float64)) / 10.
    c, s = math.cos(env._drone_yaw), math.sin(env._drone_yaw)
    remaining = (env.max_episode_steps - env.episode_step) / env.max_episode_steps
    context = [float(beta[0, 0]), env._reward_history["initial_distance"] / 10.,
               c * offset[0] + s * offset[1], -s * offset[0] + c * offset[1],
               offset[2], remaining]
    if not np.isfinite(context).all() or not 0. <= remaining <= 1.:
        raise ValueError("invalid finite-task critic context")
    reference = actor_condition["global_cond"]
    tensor = torch.tensor(context, dtype=reference.dtype, device=reference.device)
    return dict(actor_condition, episode_context=tensor.expand(reference.shape[0], -1))
