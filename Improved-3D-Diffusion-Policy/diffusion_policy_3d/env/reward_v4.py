"""F6: v2 plus a finite-horizon, nonzero-terminal goal potential."""
import numpy as np


def goal_potential(p, goal, K=10., R=2.):
    distance = float(np.linalg.norm(np.asarray(p, dtype=np.float64) -
                                    np.asarray(goal, dtype=np.float64)))
    return float(K * (1. - min(distance, R) / R))


def potential_shaping(executed_path, goal, gamma=.99, K=10., R=2.):
    path = np.asarray(executed_path, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 3 or len(path) < 2 or not np.isfinite(path).all():
        raise ValueError('v4 requires a finite actual Nx3 path')
    before, after = (goal_potential(p, goal, K, R) for p in (path[0], path[-1]))
    return gamma * after - before, dict(phi_before=before, phi_after=after,
        goal_dist_before_m=float(np.linalg.norm(path[0] - np.asarray(goal, dtype=np.float64))),
        goal_dist_after_m=float(np.linalg.norm(path[-1] - np.asarray(goal, dtype=np.float64))))


def step_v4(env, action):
    from .reward_v2_support.environment import step_v2
    obs, total, terminated, truncated, info = step_v2(env, action)
    if info['done_reason'] == 'invalid_state':
        shaping = 0.
        metrics = dict.fromkeys(('phi_before', 'phi_after', 'goal_dist_before_m', 'goal_dist_after_m'))
    else:
        shaping, metrics = potential_shaping(info['executed_path'], env.goal_w)
    info['reward_terms']['goal_potential'] = shaping
    info['reward_metrics'].update(metrics)
    info['reward_version'] = 'v4'
    return obs, total + shaping, terminated, truncated, info
