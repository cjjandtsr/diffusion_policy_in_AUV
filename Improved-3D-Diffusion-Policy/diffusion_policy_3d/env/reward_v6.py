"""F12 R12': reward v2 unchanged except three registered edits.

1. newbest' = newbest * g(D_f),  g = clip(D_f / D0, g_min, 1).
   D_f = free distance from the decision start along the 3D direction of the
   executed displacement (executed_path[-1] - executed_path[0]); sampled every
   0.1 m up to 10 m; the first sample with obstacle clearance < 0.3 m or
   z < 0.2 m ends the free segment.  Same definition as the offline O1b check
   (研究报告/analysis_20261126_f12_phase1/o1b_v2_preserving.py, ray_free).
   Exemption (F12 v1_4 纠错): when the goal is no farther from the decision start
   than D_f (the obstacle lies beyond the goal), g = 1.  Without it an obstacle
   behind the goal would discount the final approach, contradicting the design
   claim "g = 1 near the goal" and risking the old circling/timeout failure.
2. terminal for collision / ground_crash: -50 -> -100.  reached +100,
   time_limit 0, out_of_bounds / altitude_exceed / invalid_state -50 unchanged.
3. stall: when the last 5 decisions (this one included) moved < 0.3 m in total
   (sum of per-decision displacement norms), add -0.2.  Not applied on a
   terminating event other than time_limit.

Everything else (risk, time, space, termination, observation) is v2 verbatim.
"""
import math

import numpy as np

DEFAULTS = dict(D0=6.0, g_min=0.2, ray_step=0.1, ray_cap=10.0, ray_clearance=0.3,
                ray_ground=0.2, collision_terminal=-100.0, stall_window=5,
                stall_distance=0.3, stall_penalty=-0.2)
DRONE_RADIUS_Z = 0.1
_COLLISION_REASONS = ("collision", "ground_crash")


def _obstacle_array(obstacles):
    rows = []
    for o in obstacles:
        if o.shape == "cylinder":
            rows.append((0., o.x, o.y, o.z, 0., o.radius, o.height, 0., 0., 0.))
        elif o.shape == "box":
            rows.append((1., o.x, o.y, o.z, float(getattr(o, "yaw", 0.)), 0., 0.,
                         o.size_x, o.size_y, o.size_z))
        elif o.shape == "sphere":
            rows.append((2., o.x, o.y, o.z, 0., o.radius, 0., 0., 0., 0.))
        else:
            raise ValueError("unknown obstacle shape")
    return np.asarray(rows, dtype=np.float64).reshape(-1, 10)


def point_clearances(points, obstacles):
    """Vectorised collision_geom.compute_obstacle_clearance(check_3d=True)."""
    P = np.atleast_2d(np.asarray(points, dtype=np.float64))
    A = _obstacle_array(obstacles)
    if not len(A):
        return np.full(len(P), np.inf)
    dx = P[:, None, 0] - A[None, :, 1]
    dy = P[:, None, 1] - A[None, :, 2]
    cyl, box, sph = A[:, 0] == 0, A[:, 0] == 1, A[:, 0] == 2
    dxy_c = np.hypot(dx, dy) - A[None, :, 5]
    c, s = np.cos(-A[:, 4]), np.sin(-A[:, 4])
    lx, ly = dx * c - dy * s, dx * s + dy * c
    dxy_b = np.hypot(np.maximum(np.abs(lx) - A[:, 7] / 2, 0.),
                     np.maximum(np.abs(ly) - A[:, 8] / 2, 0.))
    dxy = np.where(cyl, dxy_c, dxy_b)
    half = np.where(cyl, A[:, 6], A[:, 9]) / 2.
    top, bot = A[:, 3] + half, A[:, 3] - half
    db, dt = P[:, None, 2] - DRONE_RADIUS_Z, P[:, None, 2] + DRONE_RADIUS_Z
    fly = (db > top) | (dt < bot)
    dz = np.minimum(np.abs(db - top), np.abs(dt - bot))
    c3 = np.where(fly, np.hypot(np.maximum(dxy, 0.), dz), dxy)
    dz_s = P[:, None, 2] - A[None, :, 3]
    c3 = np.where(sph, np.sqrt(dx * dx + dy * dy + dz_s * dz_s) - A[None, :, 5], c3)
    return c3.min(axis=1)


def forward_free_distance(start, direction, obstacles, *, step=.1, cap=10., clearance=.3,
                          ground=.2):
    d = np.asarray(direction, dtype=np.float64)
    n = float(np.linalg.norm(d))
    if not math.isfinite(n) or n < 1e-6:
        return float(cap)
    t = np.arange(step, cap + 1e-9, step)
    pts = np.asarray(start, dtype=np.float64)[None] + t[:, None] * (d / n)[None]
    blocked = np.nonzero((point_clearances(pts, obstacles) < clearance) | (pts[:, 2] < ground))[0]
    return float(t[blocked[0]]) if len(blocked) else float(cap)


def direction_gate(free_distance, D0=6., g_min=.2):
    return float(np.clip(float(free_distance) / float(D0), float(g_min), 1.))


def stall_term(history, displacement, *, window=5, distance=.3, penalty=-.2):
    """Append this decision's displacement; return (term, window_sum)."""
    history.append(float(displacement))
    del history[:-int(window)]
    if len(history) < int(window):
        return 0., None
    total = float(sum(history))
    return (float(penalty) if total < float(distance) else 0.), total


def params(env):
    return {k: getattr(env, "f12_" + k, v) for k, v in DEFAULTS.items()}


def step_v6(env, action):
    from .reward_v2_support.environment import step_v2
    p = params(env)
    if env.episode_step == 0:
        env._f12_stall_history = []
    obs, total, terminated, truncated, info = step_v2(env, action)
    terms, metrics = info["reward_terms"], info["reward_metrics"]
    reason = info["done_reason"]
    if reason == "invalid_state":
        terms["stall"] = 0.
        metrics.update(newbest_raw=None, f12_gate=None, f12_free_distance=None,
                       f12_stall_sum=None)
        info["reward_version"] = "v6"
        return obs, float(sum(terms.values())), terminated, truncated, info
    path = np.asarray(info["executed_path"], dtype=np.float64)
    move = path[-1] - path[0]
    free = forward_free_distance(path[0], move, env.obstacles, step=p["ray_step"],
                                 cap=p["ray_cap"], clearance=p["ray_clearance"],
                                 ground=p["ray_ground"])
    goal_distance = float(np.linalg.norm(np.asarray(env.goal_w, dtype=np.float64) - path[0]))
    goal_before_obstacle = goal_distance <= free
    gate = 1. if goal_before_obstacle else direction_gate(free, p["D0"], p["g_min"])
    raw = float(terms["newbest"])
    terms["newbest"] = raw * gate
    if reason in _COLLISION_REASONS:
        terms["terminal"] = float(p["collision_terminal"])
    stall, window_sum = stall_term(env._f12_stall_history, float(np.linalg.norm(move)),
                                   window=p["stall_window"], distance=p["stall_distance"],
                                   penalty=p["stall_penalty"])
    if reason not in ("none", "time_limit"):
        stall = 0.
    terms["stall"] = stall
    metrics.update(newbest_raw=raw, f12_gate=gate, f12_free_distance=free,
                   f12_goal_distance=goal_distance, f12_goal_before_obstacle=bool(goal_before_obstacle),
                   f12_stall_sum=window_sum, f12_displacement=float(np.linalg.norm(move)))
    info["reward_version"] = "v6"
    total = float(sum(terms.values()))
    env._last_reward_terms, env._last_reward_metrics = terms, metrics
    return obs, total, terminated, truncated, info
