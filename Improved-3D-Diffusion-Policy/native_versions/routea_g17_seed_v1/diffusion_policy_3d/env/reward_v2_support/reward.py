"""Bounded R1 reward on actual executed prefix and short plan intention."""

import math

import numpy as np

from ..reward_v17_support.geometry import segment_clearances


TERM_NAMES = ("terminal", "newbest", "risk", "time")


def space_running_cost(executed_path, start, goal, segment_durations, budget, horizon_s):
    """Task-relative bounded running cost, trapezoidal in actual elapsed time."""
    path = np.asarray(executed_path, dtype=np.float64)
    start, goal = np.asarray(start, dtype=np.float64), np.asarray(goal, dtype=np.float64)
    durations = np.asarray(segment_durations, dtype=np.float64)
    if (path.ndim != 2 or path.shape[1] != 3 or len(path) < 2 or
            start.shape != (3,) or goal.shape != (3,) or
            durations.shape != (len(path) - 1,) or
            not all(np.isfinite(x).all() for x in (path, start, goal, durations)) or
            np.any(durations < 0.) or not math.isfinite(budget) or budget < 0. or
            not math.isfinite(horizon_s) or horizon_s <= 0.):
        raise ValueError("invalid actual path/time for space running cost")
    initial = np.linalg.norm(start - goal)
    if initial <= .5:
        raise ValueError("space cost requires start outside goal sphere")
    distance_sum = np.linalg.norm(path - start, axis=1) + np.linalg.norm(path - goal, axis=1)
    q = np.maximum(0., 1. - initial / distance_sum)
    return -float(budget / horizon_s * np.sum(durations * (q[:-1] + q[1:]) / 2.))


def _barrier(clearance, warning, contact):
    return float(np.clip((warning - clearance) / (warning - contact), 0., 1.) ** 2)


def _segment_goal_distance(a, b, goal):
    delta = b - a
    scale = float(delta @ delta)
    t = 0. if scale == 0. else float(np.clip(((goal - a) @ delta) / scale, 0., 1.))
    return float(np.linalg.norm(a + t * delta - goal))


def _obstacle_clearances(path, obstacles):
    starts, ends = path[:-1], path[1:]
    result = np.full(len(starts), 100., dtype=np.float64)
    if not obstacles:
        return result
    solids = [o for o in obstacles if o.shape in ("box", "cylinder")]
    if solids:
        result = np.minimum(result, segment_clearances(starts, ends, solids).min(axis=1))
    for obstacle in obstacles:
        if obstacle.shape == "sphere":
            centre = np.array([obstacle.x, obstacle.y, obstacle.z], dtype=np.float64)
            delta = ends - starts
            length_sq = np.sum(delta * delta, axis=1)
            fraction = np.divide(np.sum((centre - starts) * delta, axis=1),
                                 length_sq, out=np.zeros_like(length_sq),
                                 where=length_sq > 0.)
            fraction = np.clip(fraction, 0., 1.)
            nearest = starts + fraction[:, None] * delta
            result = np.minimum(result, np.linalg.norm(nearest - centre, axis=1)
                                - obstacle.radius)
        elif obstacle.shape not in ("box", "cylinder"):
            raise ValueError("unknown obstacle shape")
    return result


def _risk_for_segments(path, obstacles, ground_z, indices):
    best = 0.
    clearances = _obstacle_clearances(np.asarray(path, dtype=np.float64), obstacles)
    provenance = dict(argmax_segment=None, risk_type="none", clearance=None,
                      weight=0., risk_source="none")
    for (a, b, c), j in zip(zip(path[:-1], path[1:], clearances), indices):
        z = float(min(a[2], b[2]) - ground_z)
        obstacle_part = _barrier(c, .45, .10)
        ground_part = _barrier(z, 1., .20)
        weight = 1. if j <= 4 else 2. ** (-(j - 4) / 4.)
        value = .5 * weight * (obstacle_part + ground_part)
        if value > best:
            best = value
            provenance = dict(argmax_segment=j,
                              risk_type="obstacle" if obstacle_part >= ground_part else "ground",
                              clearance=c if obstacle_part >= ground_part else z,
                              weight=weight,
                              risk_source="exec" if j <= 4 else "tail")
    return best, provenance


def compute_reward_v2(executed_path, planned_path, goal, obstacles,
                      best_distance, initial_distance, done_reason,
                      collision_thresh=.1, ground_z=0.):
    """Return total, signed terms, audit metrics, and next episode best distance."""
    executed = np.asarray(executed_path, dtype=np.float64)
    planned = np.asarray(planned_path, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    if (executed.ndim != 2 or executed.shape[1] != 3 or len(executed) < 2 or
        planned.shape != (9, 3) or goal.shape != (3,) or
        not np.isfinite(executed).all() or not np.isfinite(planned).all() or
        not np.isfinite(goal).all()):
        raise ValueError("R1 reward requires finite executed Nx3 and planned 9x3")
    if not (math.isfinite(best_distance) and math.isfinite(initial_distance) and
            0. <= best_distance <= initial_distance and initial_distance > .5):
        raise ValueError("invalid R1 episode best distance")
    if not (math.isfinite(collision_thresh) and collision_thresh >= 0 and
            math.isfinite(ground_z)):
        raise ValueError("invalid R1 geometry context")
    if done_reason not in ("none", "reached", "collision", "ground_crash",
                           "out_of_bounds", "altitude_exceed", "time_limit",
                           "invalid_state"):
        raise ValueError("unknown R1 done reason")

    nearest = min(_segment_goal_distance(a, b, goal)
                  for a, b in zip(executed[:-1], executed[1:]))
    next_best = min(float(best_distance), nearest)
    newbest = 10. * (best_distance - next_best) / initial_distance
    if done_reason == "reached":
        terminal = 100.
    elif done_reason in ("collision", "ground_crash", "out_of_bounds",
                         "altitude_exceed", "invalid_state"):
        terminal = -50.
    else:
        terminal = 0.

    exec_risk, exec_provenance = _risk_for_segments(
        executed, obstacles, ground_z, range(1, len(executed)))
    tail_risk = 0.
    tail_provenance = dict(argmax_segment=None, risk_type="none", clearance=None,
                           weight=0., risk_source="none")
    if done_reason == "none":
        tail_risk, tail_provenance = _risk_for_segments(
            planned[4:9], obstacles, ground_z, range(5, 9))
    risk = max(exec_risk, tail_risk)
    provenance = exec_provenance if exec_risk >= tail_risk else tail_provenance
    terms = dict(terminal=terminal, newbest=float(newbest), risk=-float(risk), time=-.05)
    metrics = dict(risk=float(risk), risk_exec=float(exec_risk),
                   risk_tail=float(tail_risk), best_before=float(best_distance),
                   best_after=float(next_best), training_usable=done_reason != "invalid_state",
                   **provenance)
    if not (0. <= newbest <= 10. + 1e-10 and 0. <= risk <= 1. + 1e-10):
        raise ValueError("R1 reward bound violation")
    return float(sum(terms.values())), terms, metrics, float(next_best)
