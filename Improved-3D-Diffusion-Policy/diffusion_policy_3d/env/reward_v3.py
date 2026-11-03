"""F5 adds a bounded actual-path closing cost to the unchanged v2 reward."""

import numpy as np

from .reward_v2_support.reward import compute_reward_v2, _obstacle_clearances


def closing_cost(executed_path, obstacles):
    """The caller supplies the already committed first-event path; do not clip it again."""
    path = np.asarray(executed_path, dtype=np.float64)
    if (path.ndim != 2 or path.shape[1] != 3 or len(path) < 2 or
            not np.isfinite(path).all()):
        raise ValueError("v3 closing cost requires a finite executed Nx3 path")
    clearances = np.array([_obstacle_clearances(np.stack((p, p)), obstacles)[0]
                           for p in path], dtype=np.float64)
    closing = np.maximum(0., clearances[:-1] - clearances[1:])
    barrier = np.clip((1.2 - clearances[1:]) / 1.1, 0., 1.) ** 2
    cost = -4. * min(float(np.sum(closing * barrier)), .5)
    return cost, dict(risk_close=cost, closing_m=float(closing.sum()),
                      min_clearance_m=float(clearances.min()))


def compute_reward_v3(executed_path, planned_path, goal, obstacles,
                      best_distance, initial_distance, done_reason,
                      collision_thresh=.1, ground_z=0.):
    total, terms, metrics, best = compute_reward_v2(
        executed_path, planned_path, goal, obstacles, best_distance,
        initial_distance, done_reason, collision_thresh, ground_z)
    cost, closing_metrics = closing_cost(executed_path, obstacles)
    terms['risk_close'] = cost
    metrics.update(closing_metrics)
    return total + cost, terms, metrics, best


def step_v3(env, action):
    from .reward_v2_support.environment import step_v2
    obs, total, terminated, truncated, info = step_v2(env, action)
    if info['done_reason'] == 'invalid_state':
        cost = 0.
        metrics = dict(risk_close=0., closing_m=0., min_clearance_m=None)
    else:
        cost, metrics = closing_cost(info['executed_path'], env.obstacles)
    info['reward_terms']['risk_close'] = cost
    info['reward_metrics'].update(metrics)
    info['reward_version'] = 'v3'
    return obs, total + cost, terminated, truncated, info
