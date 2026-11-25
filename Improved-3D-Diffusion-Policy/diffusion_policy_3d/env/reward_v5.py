"""F7 v1.2: memoryless CBF-type cost on the committed physical path."""
import numpy as np
from .reward_v2_support.reward import _obstacle_clearances


def approach_from_clearances(clearances, durations, weight=2.):
    b = np.asarray(clearances, dtype=np.float64) - .30
    dt = np.asarray(durations, dtype=np.float64)
    active = dt > 0.
    violation = np.zeros_like(dt)
    violation[active] = -((b[1:][active]-b[:-1][active])/dt[active]
                         + (b[:-1][active]+b[1:][active])/6.)
    return -float(weight) * float(np.sum(dt*np.clip(violation, 0., 2.)))


def approach_cost(path, obstacles, durations, weight=2.):
    c = np.array([_obstacle_clearances(np.stack((p,p)), obstacles)[0]
                  for p in np.asarray(path, dtype=np.float64)])
    return approach_from_clearances(c, durations, weight), c


def segment_durations(info, traj_dt):
    dt = np.full(len(info['executed_path'])-1, traj_dt, dtype=np.float64)
    if info['event_fraction'] is not None:
        dt[-1] *= info['event_fraction']
    return dt


def step_v5(env, action):
    from .reward_v2_support.environment import step_v2
    obs, total, terminated, truncated, info = step_v2(env, action)
    if info['done_reason'] == 'invalid_state':
        cost, clearance = 0., None
    else:
        dt = segment_durations(info, env.traj_dt)
        cost, c = approach_cost(info['executed_path'], env.obstacles, dt, env.approach_weight)
        clearance = float(c.min())
        info['segment_durations'] = dt.tolist()
    info['reward_terms']['approach_cbf'] = cost
    info['reward_metrics'].update(approach_cbf=cost, approach_min_clearance=clearance)
    info['reward_version'] = 'v5'
    return obs, total+cost, terminated, truncated, info
