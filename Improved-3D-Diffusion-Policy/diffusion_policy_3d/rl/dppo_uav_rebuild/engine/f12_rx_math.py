"""F12 RX pure functions (no package-relative imports; loadable by file in probes)."""
import math

import numpy as np


def rotate_goal(agent_pos, theta):
    """Rotate the goal vector (state dims 3,4) of every history frame by theta (rad)."""
    a = np.asarray(agent_pos, np.float32).copy()
    c, s = math.cos(theta), math.sin(theta)
    gx, gy = a[..., 3].copy(), a[..., 4].copy()
    a[..., 3], a[..., 4] = c * gx - s * gy, s * gx + c * gy
    return a


def mirror_full(full, yaw):
    """Lateral mirror in the body frame. full: (...,16,11) world-frame [dpos, dvel, acc, dyaw, yaw_rate].

    Patch 2 (2026-11-27): position, velocity and acceleration xy are all mirrored (O2's
    mirror_action mirrored position and yaw only, leaving velocity/acceleration unmirrored,
    which makes the mirrored chunk internally inconsistent as an RX training sample)."""
    a = np.asarray(full, np.float32).copy()
    c, s = math.cos(yaw), math.sin(yaw)
    for i, j in ((0, 1), (3, 4), (6, 7)):
        x, y = a[..., i].copy(), a[..., j].copy()
        bx, by = c * x + s * y, -(-s * x + c * y)
        a[..., i], a[..., j] = c * bx - s * by, s * bx + c * by
    a[..., 9], a[..., 10] = -a[..., 9], -a[..., 10]
    return a


def group_advantages(scores, eps=1e-6):
    """Population-std normalisation inside a group; all-equal -> zeros (group skipped)."""
    v = np.asarray(scores, np.float64)
    sd = float(v.std())
    if not np.isfinite(v).all():
        raise FloatingPointError('non-finite RX score')
    if sd < eps:
        return np.zeros_like(v), False
    return (v - v.mean()) / sd, True


def td_score(reward, value_before, value_after, bootstrap, gamma=.99):
    return float(reward) + (float(gamma) * float(value_after) if bootstrap else 0.) - float(value_before)


def horizontal_free(position, goal, obstacles, *, step=.05, cap=10., clearance=.3, ground=.1):
    """Phase-1 f12_common.free_distance along the horizontal goal bearing."""
    from diffusion_policy_3d.env.reward_v6 import point_clearances
    d = np.asarray(goal, np.float64) - np.asarray(position, np.float64)
    d[2] = 0.
    n = float(np.linalg.norm(d[:2]))
    if n < 1e-9:
        return 0.
    r = np.arange(0., cap + 1e-9, step)
    pts = np.asarray(position, np.float64)[None] + r[:, None] * (d / n)[None]
    hit = np.nonzero((pts[:, 2] < ground) | (point_clearances(pts, obstacles) <= clearance + 1e-9))[0]
    return float(r[hit[0]]) if len(hit) else float(cap)
